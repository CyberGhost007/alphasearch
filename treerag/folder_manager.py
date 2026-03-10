"""
Folder Manager — handles all file operations with comprehensive error handling.

Edge cases handled:
- Corrupt/invalid/empty PDFs → Validated before indexing
- Non-PDF/CSV/Excel files → Rejected with clear error
- File hash caching → Skip re-indexing unchanged files
- Batch add with partial failures → Report successes AND failures
- Index files manually deleted → Detected, offers re-index
- PDF/file deleted but meta-tree entry remains → Detected on health check
- Duplicate filenames from different paths → Warning to user
- Disk full during write → Atomic writes prevent corrupt state
- Orphaned index files → Cleanup command
- Folder must exist before add → Explicit error
- Thread-safe meta-tree writes → File-level locking
"""

import os
import shutil
import threading
from pathlib import Path
from typing import Optional
from rich.console import Console

from .config import TreeRAGConfig
from .models import (
    FolderIndex, FolderDocEntry, DocumentIndex,
    compute_file_hash,
)
from .indexer import Indexer
from .pdf_processor import PDFProcessor
from .tabular_processor import TabularProcessor, SUPPORTED_TABULAR_EXTENSIONS
from .tabular_indexer import TabularIndexer
from .exceptions import (
    FolderNotFoundError, FolderAlreadyExistsError,
    DocumentNotFoundError, DuplicateFilenameWarning,
    IndexNotFoundError, IndexCorruptError, PDFMissingError,
    IndexingFailedError, BatchIndexingError,
    InvalidFileError, CorruptPDFError, EmptyPDFError, FileTooLargeError,
    InvalidTabularError, EmptyTabularError,
)

SUPPORTED_EXTENSIONS = {".pdf"} | SUPPORTED_TABULAR_EXTENSIONS

console = Console()


class FolderManager:
    """
    Manages folders and documents with full error handling.
    Thread-safe for concurrent access via per-folder locks.
    """

    def __init__(self, config: TreeRAGConfig, pipeline=None):
        self.config = config
        self._pipeline = pipeline  # Lazy reference to pipeline for LLM access
        self._indexer = None
        self._tabular_indexer = None
        self.pdf_processor = PDFProcessor(
            dpi=config.indexer.image_dpi,
            max_pages=config.indexer.max_pages,
        )
        self.tabular_processor = TabularProcessor()
        self.base_dir = Path(config.folder.base_dir) / "folders"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    @property
    def indexer(self):
        """Lazy-init PDF indexer — only created when actually indexing (needs API key)."""
        if self._indexer is None:
            if self._pipeline:
                self._indexer = Indexer(self.config, self._pipeline.llm)
            else:
                from .llm_client import LLMClient
                llm = LLMClient(self.config.model)
                self._indexer = Indexer(self.config, llm)
        return self._indexer

    @property
    def tabular_indexer(self):
        """Lazy-init tabular indexer — only created when indexing CSV/Excel."""
        if self._tabular_indexer is None:
            if self._pipeline:
                self._tabular_indexer = TabularIndexer(self.config, self._pipeline.llm)
            else:
                from .llm_client import LLMClient
                llm = LLMClient(self.config.model)
                self._tabular_indexer = TabularIndexer(self.config, llm)
        return self._tabular_indexer

    def _is_tabular(self, file_path: Path) -> bool:
        """Check if file is a tabular type (CSV/Excel)."""
        return file_path.suffix.lower() in SUPPORTED_TABULAR_EXTENSIONS

    def _get_indexer(self, file_path: Path):
        """Return the right indexer for the file type."""
        if self._is_tabular(file_path):
            return self.tabular_indexer
        return self.indexer

    def _get_folder_lock(self, folder_name: str) -> threading.Lock:
        with self._locks_lock:
            if folder_name not in self._locks:
                self._locks[folder_name] = threading.Lock()
            return self._locks[folder_name]

    # =========================================================================
    # Folder CRUD
    # =========================================================================

    def create_folder(self, folder_name: str) -> FolderIndex:
        """Create a new empty folder. Raises if already exists."""
        self._validate_folder_name(folder_name)
        folder_dir = self.base_dir / folder_name

        if folder_dir.exists() and (folder_dir / "folder_index.json").exists():
            raise FolderAlreadyExistsError(
                f"Folder '{folder_name}' already exists. "
                f"Use 'folder info {folder_name}' to see its contents."
            )

        folder_dir.mkdir(parents=True, exist_ok=True)
        (folder_dir / "indices").mkdir(exist_ok=True)
        (folder_dir / "pdfs").mkdir(exist_ok=True)
        (folder_dir / "files").mkdir(exist_ok=True)

        folder_index = FolderIndex(folder_name=folder_name, folder_path=str(folder_dir))
        folder_index.save(folder_dir / "folder_index.json")
        console.print(f"[green]Created folder:[/green] {folder_name}")
        return folder_index

    def load_folder(self, folder_name: str) -> FolderIndex:
        """Load folder. Raises FolderNotFoundError if missing."""
        index_path = self.base_dir / folder_name / "folder_index.json"
        if not index_path.exists():
            available = self.list_folders()
            msg = f"Folder '{folder_name}' not found."
            if available:
                msg += f" Available folders: {', '.join(available)}"
            msg += " Create it first with: folder create <name>"
            raise FolderNotFoundError(msg)
        return FolderIndex.load(index_path)

    def list_folders(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted([
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "folder_index.json").exists()
        ])

    def delete_folder(self, folder_name: str) -> bool:
        folder_dir = self.base_dir / folder_name
        if not folder_dir.exists():
            console.print(f"[yellow]Folder '{folder_name}' does not exist.[/yellow]")
            return False
        shutil.rmtree(folder_dir)
        console.print(f"[red]Deleted folder:[/red] {folder_name}")
        return True

    # =========================================================================
    # Document Management
    # =========================================================================

    def add_document(
        self,
        folder_name: str,
        pdf_path: str | Path,
        copy_pdf: bool = True,
        skip_validation: bool = False,
    ) -> FolderIndex:
        """
        Add a document (PDF, CSV, or Excel) to a folder with full validation.

        Validates: file exists, supported type, not corrupt, not empty, not too large.
        Caches: skips indexing if file hash matches.
        Warns: duplicate filenames from different source paths.
        Cleans up: removes partial files on failure.
        Thread-safe: uses per-folder locks.
        """
        pdf_path = Path(pdf_path).resolve()
        lock = self._get_folder_lock(folder_name)

        # 1. Validate folder exists (explicit create required)
        folder_index = self.load_folder(folder_name)  # Raises FolderNotFoundError
        folder_dir = self.base_dir / folder_name

        # 2. Validate file
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        ext = pdf_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise InvalidFileError(
                f"Unsupported file type: '{pdf_path.name}' (extension: {ext}). "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Dispatch validation by file type
        is_tabular = self._is_tabular(pdf_path)
        if is_tabular:
            file_info = self.tabular_processor.validate(pdf_path)
        else:
            file_info = self.pdf_processor.validate(pdf_path)

        # 3. Check cache
        file_hash = compute_file_hash(pdf_path)
        existing = folder_index.get_document(pdf_path.name)

        if existing and existing.file_hash == file_hash:
            # Same file, same content — skip
            console.print(f"[dim]{pdf_path.name}: Unchanged — using cached index[/dim]")
            return folder_index

        # 4. Warn about duplicate filenames from different paths
        if existing and existing.source_path and str(pdf_path) != existing.source_path:
            console.print(
                f"[yellow]Warning: '{pdf_path.name}' was previously added from "
                f"'{existing.source_path}'. Replacing with version from '{pdf_path}'.[/yellow]"
            )

        # 5. Copy file to folder (atomic: copy to temp then rename)
        # Store in files/ for tabular, pdfs/ for PDFs (backward compat)
        stored_pdf_path = pdf_path
        if copy_pdf:
            storage_subdir = "files" if is_tabular else "pdfs"
            dest = folder_dir / storage_subdir / pdf_path.name
            if str(pdf_path) != str(dest):
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    # Copy to temp first, then rename (atomic)
                    tmp_dest = dest.with_suffix(".tmp")
                    shutil.copy2(pdf_path, tmp_dest)
                    tmp_dest.rename(dest)
                    stored_pdf_path = dest
                except OSError as e:
                    # Cleanup temp
                    tmp_dest = dest.with_suffix(".tmp")
                    if tmp_dest.exists():
                        tmp_dest.unlink()
                    raise IndexingFailedError(
                        f"Failed to copy '{pdf_path.name}' to folder: {e}. Check disk space."
                    ) from e

        # 6. Build document tree index (dispatch by file type)
        index_path = folder_dir / "indices" / f"{pdf_path.stem}_tree.json"
        indexer = self._get_indexer(pdf_path)
        try:
            doc_index = indexer.index_document(
                stored_pdf_path, save_path=index_path,
                skip_validation=skip_validation,
            )
        except IndexingFailedError:
            # Cleanup copied file if indexing failed
            if copy_pdf and stored_pdf_path != pdf_path and stored_pdf_path.exists():
                stored_pdf_path.unlink()
            raise

        # 7. Generate document-level summary
        console.print("[bold]Generating document summary for meta-tree...[/bold]")
        summary, keywords = indexer.generate_document_summary(stored_pdf_path)

        # 8. Create entry and update folder index (thread-safe)
        entry = FolderDocEntry(
            document_id=doc_index.document_id,
            filename=pdf_path.name,
            total_pages=doc_index.total_pages,
            summary=summary or doc_index.description,
            keywords=keywords,
            index_path=str(index_path),
            pdf_path=str(stored_pdf_path),
            file_hash=file_hash,
            source_path=str(pdf_path),
        )

        with lock:
            # Reload to avoid overwriting concurrent changes
            folder_index = self.load_folder(folder_name)
            folder_index.add_document(entry)
            folder_index.save(folder_dir / "folder_index.json")

        console.print(f"[green]Added to '{folder_name}':[/green] {pdf_path.name}")
        return folder_index

    def add_documents_batch(
        self,
        folder_name: str,
        pdf_paths: list[str | Path],
        copy_pdf: bool = True,
        skip_validation: bool = False,
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """
        Add multiple documents. Continues on failure.
        Returns: (successes: [filename], failures: [(filename, error_message)])
        """
        successes = []
        failures = []

        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)
            try:
                self.add_document(folder_name, pdf_path, copy_pdf, skip_validation)
                successes.append(pdf_path.name)
            except Exception as e:
                failures.append((pdf_path.name, str(e)))
                console.print(f"[red]Failed: {pdf_path.name} — {e}[/red]")

        # Summary
        if successes:
            console.print(f"\n[green]{len(successes)} documents added successfully[/green]")
        if failures:
            console.print(f"[red]{len(failures)} documents failed:[/red]")
            for fname, err in failures:
                console.print(f"  [red]{fname}: {err}[/red]")

        if failures and not successes:
            raise BatchIndexingError(
                f"All {len(failures)} documents failed to index.",
                successes=successes, failures=failures,
            )

        return successes, failures

    def remove_document(self, folder_name: str, filename: str) -> FolderIndex:
        """Remove a document. Cleans up index and PDF files."""
        lock = self._get_folder_lock(folder_name)
        folder_index = self.load_folder(folder_name)
        folder_dir = self.base_dir / folder_name

        entry = folder_index.get_document(filename)
        if not entry:
            raise DocumentNotFoundError(
                f"'{filename}' not found in folder '{folder_name}'. "
                f"Available: {', '.join(d.filename for d in folder_index.documents) or 'none'}"
            )

        # Delete files (ignore errors — best effort)
        # Check both pdfs/ and files/ dirs for backward compat
        paths_to_delete = [entry.index_path]
        for subdir in ["pdfs", "files"]:
            paths_to_delete.append(str(folder_dir / subdir / filename))
        for path_str in paths_to_delete:
            try:
                p = Path(path_str)
                if p.exists():
                    p.unlink()
            except OSError as e:
                console.print(f"[yellow]Warning: Could not delete {path_str}: {e}[/yellow]")

        with lock:
            folder_index = self.load_folder(folder_name)
            folder_index.remove_document(filename)
            folder_index.save(folder_dir / "folder_index.json")

        console.print(f"[red]Removed from '{folder_name}':[/red] {filename}")
        return folder_index

    # =========================================================================
    # Health Check & Maintenance
    # =========================================================================

    def health_check(self, folder_name: str) -> dict:
        """
        Check folder integrity. Detects:
        - Missing PDF files (meta-tree references deleted PDFs)
        - Missing index files (manually deleted)
        - Stale entries (file hash changed)
        - Orphaned index files (no meta-tree entry)
        """
        folder_index = self.load_folder(folder_name)
        folder_dir = self.base_dir / folder_name

        issues = {
            "missing_pdfs": [],       # Meta-tree entry but PDF gone
            "missing_indices": [],    # Meta-tree entry but index JSON gone
            "stale_entries": [],      # File hash changed
            "orphaned_indices": [],   # Index file exists but no meta-tree entry
            "healthy": [],            # Everything OK
        }

        # Check each document entry
        for entry in folder_index.documents:
            pdf_ok = entry.pdf_path and Path(entry.pdf_path).exists()
            idx_ok = entry.index_path and Path(entry.index_path).exists()

            if not pdf_ok:
                issues["missing_pdfs"].append(entry.filename)
            elif not idx_ok:
                issues["missing_indices"].append(entry.filename)
            else:
                current_hash = compute_file_hash(entry.pdf_path)
                if current_hash != entry.file_hash:
                    issues["stale_entries"].append(entry.filename)
                else:
                    issues["healthy"].append(entry.filename)

        # Check for orphaned index files
        index_dir = folder_dir / "indices"
        if index_dir.exists():
            known_indices = {Path(d.index_path).name for d in folder_index.documents if d.index_path}
            for f in index_dir.iterdir():
                if f.suffix == ".json" and f.name not in known_indices:
                    issues["orphaned_indices"].append(f.name)

        return issues

    def repair_folder(self, folder_name: str, remove_broken: bool = False) -> FolderIndex:
        """
        Repair folder based on health check.
        - Re-index stale entries
        - Optionally remove entries with missing PDFs
        - Clean up orphaned index files
        """
        issues = self.health_check(folder_name)
        folder_dir = self.base_dir / folder_name

        console.print(f"\n[bold]Repairing folder: {folder_name}[/bold]")
        console.print(f"  Healthy: {len(issues['healthy'])}")
        console.print(f"  Missing PDFs: {len(issues['missing_pdfs'])}")
        console.print(f"  Missing indices: {len(issues['missing_indices'])}")
        console.print(f"  Stale (changed): {len(issues['stale_entries'])}")
        console.print(f"  Orphaned indices: {len(issues['orphaned_indices'])}")

        # Handle missing PDFs
        if issues["missing_pdfs"]:
            if remove_broken:
                for fname in issues["missing_pdfs"]:
                    console.print(f"  [red]Removing broken entry: {fname}[/red]")
                    self.remove_document(folder_name, fname)
            else:
                for fname in issues["missing_pdfs"]:
                    console.print(f"  [yellow]Warning: PDF missing for '{fname}' — use --remove-broken to clean up[/yellow]")

        # Re-index entries with missing indices
        for fname in issues["missing_indices"]:
            folder_index = self.load_folder(folder_name)
            entry = folder_index.get_document(fname)
            if entry and entry.pdf_path and Path(entry.pdf_path).exists():
                console.print(f"  [cyan]Re-indexing (missing index): {fname}[/cyan]")
                try:
                    self.add_document(folder_name, entry.pdf_path, copy_pdf=False)
                except Exception as e:
                    console.print(f"  [red]Failed to re-index {fname}: {e}[/red]")

        # Re-index stale entries
        for fname in issues["stale_entries"]:
            folder_index = self.load_folder(folder_name)
            entry = folder_index.get_document(fname)
            if entry and entry.pdf_path and Path(entry.pdf_path).exists():
                console.print(f"  [cyan]Re-indexing (changed): {fname}[/cyan]")
                try:
                    self.add_document(folder_name, entry.pdf_path, copy_pdf=False)
                except Exception as e:
                    console.print(f"  [red]Failed to re-index {fname}: {e}[/red]")

        # Clean orphaned index files
        for orphan in issues["orphaned_indices"]:
            orphan_path = folder_dir / "indices" / orphan
            console.print(f"  [dim]Removing orphaned index: {orphan}[/dim]")
            try:
                orphan_path.unlink()
            except OSError as e:
                console.print(f"[yellow]Warning: Could not remove orphaned index {orphan}: {e}[/yellow]")

        return self.load_folder(folder_name)

    def refresh_folder(self, folder_name: str) -> FolderIndex:
        """Convenience wrapper for repair_folder."""
        return self.repair_folder(folder_name, remove_broken=False)

    # =========================================================================
    # Index Loading (for search)
    # =========================================================================

    def load_document_index(self, entry: FolderDocEntry) -> DocumentIndex:
        """Load a document's full tree index. Validates existence first."""
        if not entry.index_path:
            raise IndexNotFoundError(f"No index path for '{entry.filename}'. Re-add the document.")

        idx_path = Path(entry.index_path)
        if not idx_path.exists():
            raise IndexNotFoundError(
                f"Index file missing for '{entry.filename}' (expected: {idx_path}). "
                f"Run 'folder refresh' or re-add the document."
            )

        doc_index = DocumentIndex.load(idx_path)

        # Verify PDF still exists (needed for deep reads)
        if entry.pdf_path and not Path(entry.pdf_path).exists():
            console.print(
                f"[yellow]Warning: PDF missing for '{entry.filename}'. "
                f"Search will work but deep reads will be skipped.[/yellow]"
            )

        if doc_index.root:
            self._set_parent_refs(doc_index.root)
        return doc_index

    def load_document_indices(self, entries: list[FolderDocEntry]) -> list[DocumentIndex]:
        """Load multiple indices. Skips broken ones with warnings."""
        indices = []
        for entry in entries:
            try:
                indices.append(self.load_document_index(entry))
            except (IndexNotFoundError, IndexCorruptError) as e:
                console.print(f"[yellow]Skipping {entry.filename}: {e}[/yellow]")
        return indices

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def _validate_folder_name(self, name: str):
        """Validate folder name characters."""
        if not name or not name.strip():
            raise ValueError("Folder name cannot be empty.")
        invalid_chars = set('/\\:*?"<>|')
        found = [c for c in name if c in invalid_chars]
        if found:
            raise ValueError(f"Folder name contains invalid characters: {''.join(found)}")
        if len(name) > 200:
            raise ValueError(f"Folder name too long ({len(name)} chars, max 200).")

    def _set_parent_refs(self, node):
        for c in node.children:
            c.parent = node
            self._set_parent_refs(c)
