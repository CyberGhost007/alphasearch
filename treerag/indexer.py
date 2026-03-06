"""
Document Indexer — builds tree index from PDF with cleanup on failure.
If indexing fails midway, any partial files are cleaned up.
"""

import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import TreeRAGConfig
from .models import TreeNode, DocumentIndex, compute_file_hash
from .llm_client import LLMClient
from .pdf_processor import PDFProcessor, LoadedPDF
from .exceptions import IndexingFailedError

console = Console()

SYSTEM_PROMPT_ANALYZE = """You are a document structure analyst. You receive images of PDF pages and must identify ALL logical sections — every page must be covered.

CRITICAL RULES:
1. Every single page must belong to at least one section. Do NOT skip pages.
2. If a page is a standalone slide, it IS its own section.
3. If pages are part of a presentation/deck, each slide or group of related slides is a section.
4. Title pages, cover pages, appendix dividers, and contact pages are all sections too.
5. Summaries must capture the ACTUAL CONTENT — specific data, names, numbers, key points — not just "this page discusses..."

Output JSON:
{
  "sections": [
    {
      "title": "Descriptive section title",
      "start_page": <0-indexed>,
      "end_page": <0-indexed, inclusive>,
      "level": <1=major section/chapter, 2=sub-section, 3=detail>,
      "summary": "2-3 sentences with SPECIFIC information from this section. Include key data points, names, figures, decisions.",
      "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
    }
  ],
  "continues_previous": true/false,
  "continues_next": true/false
}

IMPORTANT: The total page coverage of all sections must span from the first to the last page. If you see 11 pages, you should identify sections that together cover pages 0 through 10. Missing pages = failed analysis."""

SYSTEM_PROMPT_BUILD_TREE = """You are a document architect. Given a flat list of sections, build a properly nested tree structure.

Output a single JSON object for the root node:
{
  "node_id": "0000",
  "title": "Document title",
  "summary": "Overall document summary (2-3 sentences with specifics)",
  "keywords": ["doc-level", "keywords"],
  "start_page": 0,
  "end_page": <last page number>,
  "level": 0,
  "children": [
    {
      "node_id": "0001",
      "title": "Section title",
      "summary": "Section summary",
      "keywords": [...],
      "start_page": ...,
      "end_page": ...,
      "level": 1,
      "children": [ ... sub-sections ... ]
    }
  ]
}

Rules:
- node_id must be unique zero-padded 4-digit strings: "0000", "0001", "0002"...
- Root node (level 0) represents the entire document
- Nest sections by level: level 2 goes under parent level 1
- EVERY input section must appear somewhere in the tree — do not drop any sections
- Preserve original summaries and keywords exactly — do not rewrite them
- The root's start_page should be 0 and end_page should be the last page"""

SYSTEM_PROMPT_VALIDATE = """Check if a node's summary accurately captures the page content.
Output JSON: {"is_accurate": true/false, "revised_summary": "..." or null, "revised_keywords": [...] or null}"""

SYSTEM_PROMPT_DOC_SUMMARY = """You are summarizing a document for search purposes. Given images of the first several pages, generate:

{
  "summary": "3-5 sentences about the ENTIRE document. Be SPECIFIC — include company names, project names, dates, technologies, dollar amounts, key findings. This summary will be used to decide if this document is relevant to a user's query.",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6", "keyword7", "keyword8"]
}

Keywords should be specific nouns someone might search for, not generic terms. Include company names, product names, technologies, dates."""


class Indexer:
    def __init__(self, config: TreeRAGConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.pdf_processor = PDFProcessor(
            dpi=config.indexer.image_dpi,
            max_pages=config.indexer.max_pages,
        )

    def index_document(
        self, pdf_path: str | Path,
        save_path: Optional[str | Path] = None,
        skip_validation: bool = False,
    ) -> DocumentIndex:
        """
        Build a complete tree index. On failure, cleans up any partial save_path.
        """
        pdf_path = Path(pdf_path)
        save_path = Path(save_path) if save_path else None

        # Validate PDF first (raises on bad files)
        self.pdf_processor.validate(pdf_path)

        console.print(f"\n[bold blue]Indexing:[/bold blue] {pdf_path.name}")
        pdf = self.pdf_processor.load(pdf_path)
        console.print(f"  Pages: {pdf.total_pages}")

        try:
            # Step 1: Analyze
            console.print("\n[bold]Step 1:[/bold] Analyzing document structure...")
            all_sections = self._analyze_pages(pdf)
            console.print(f"  Found {len(all_sections)} sections")

            # Step 2: Build tree
            console.print("\n[bold]Step 2:[/bold] Building tree index...")
            tree = self._build_tree(all_sections, pdf.total_pages)

            # Step 3: Validate
            if not skip_validation:
                console.print("\n[bold]Step 3:[/bold] Validating summaries...")
                self._validate_summaries(tree, pdf)

        except Exception as e:
            pdf.close()
            # Cleanup partial index file
            if save_path and save_path.exists():
                save_path.unlink()
            raise IndexingFailedError(
                f"Indexing failed for '{pdf_path.name}': {e}"
            ) from e

        file_hash = compute_file_hash(pdf_path)
        doc_index = DocumentIndex(
            document_id=file_hash[:12], filename=pdf_path.name,
            total_pages=pdf.total_pages,
            description=tree.summary if tree else "",
            root=tree, file_hash=file_hash, pdf_path=str(pdf_path),
        )

        if save_path:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                doc_index.save(save_path)
                console.print(f"\n[green]Index saved:[/green] {save_path}")
            except OSError as e:
                raise IndexingFailedError(
                    f"Failed to save index for '{pdf_path.name}': {e}. "
                    f"Check disk space and permissions."
                ) from e

        pdf.close()
        console.print(f"[bold green]Indexing complete![/bold green] (LLM calls: {self.llm.total_calls})")
        return doc_index

    def generate_document_summary(self, pdf_path: str | Path) -> tuple[str, list[str]]:
        """Generate document-level summary for the meta-tree."""
        pdf_path = Path(pdf_path)
        pdf = self.pdf_processor.load(pdf_path)
        num_pages = min(5, pdf.total_pages)
        images = pdf.get_page_images_batch(0, num_pages - 1)

        prompt = f"""Document: "{pdf_path.name}" ({pdf.total_pages} pages).
You see the first {num_pages} pages. Summarize what this document is about."""

        try:
            response = self.llm.complete_with_images(
                prompt=prompt, images=images,
                model=self.config.model.indexing_model,
                system_prompt=SYSTEM_PROMPT_DOC_SUMMARY,
                json_mode=True, max_tokens=1024,
            )
            pdf.close()
            result = json.loads(response)
            return result.get("summary", ""), result.get("keywords", [])
        except Exception:
            pdf.close()
            return f"Document: {pdf_path.name}", []

    # --- Internal methods (same logic, with better error handling) ---

    def _analyze_pages(self, pdf: LoadedPDF) -> list[dict]:
        batch_size = self.config.indexer.batch_size
        overlap = self.config.indexer.overlap_pages
        all_sections = []
        total_batches = max(1, (pdf.total_pages + batch_size - 1) // batch_size)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Analyzing pages...", total=total_batches)
            batch_num, start = 0, 0

            while start < pdf.total_pages:
                end = min(start + batch_size - 1, pdf.total_pages - 1)
                images = pdf.get_page_images_batch(start, end)

                if not images:
                    console.print(f"  [yellow]Warning: No renderable pages in batch {batch_num + 1}[/yellow]")
                    start = end + 1
                    batch_num += 1
                    progress.advance(task)
                    continue

                prompt = f"""Analyze these PDF pages and identify ALL logical sections. Every page must be covered.

Document: {pdf.total_pages} total pages. This batch: pages {start+1} to {end+1} (batch {batch_num+1} of {total_batches}).
You are seeing {end - start + 1} page images. Page numbers in output must be 0-indexed (page 1 = index 0).

{"This is the FIRST batch — the document starts here." if batch_num == 0 else "This is a CONTINUATION — previous pages were already analyzed."}
{"This is the LAST batch — the document ends here." if batch_num + 1 >= total_batches else "More pages will follow in the next batch."}

IMPORTANT: 
- Identify a section for EVERY page. Do not skip any pages.
- If a page is a title slide, cover page, divider, or contact page — it is still a section.
- If this is a presentation/slide deck, each slide (or small group of related slides) should be its own section.
- Summaries must include SPECIFIC content: numbers, names, data points, key findings — not just topic descriptions.

Output valid JSON only."""

                try:
                    response = self.llm.complete_with_images(
                        prompt=prompt, images=images,
                        model=self.config.model.indexing_model,
                        system_prompt=SYSTEM_PROMPT_ANALYZE,
                        json_mode=True, max_tokens=4096,
                    )
                    sections = json.loads(response).get("sections", [])
                    all_sections.extend(sections)
                except (json.JSONDecodeError, RuntimeError) as e:
                    console.print(f"  [yellow]Warning: Batch {batch_num+1} failed: {e}[/yellow]")

                start = end + 1 - overlap if end + 1 < pdf.total_pages else pdf.total_pages
                batch_num += 1
                progress.advance(task)

        if not all_sections:
            console.print("  [yellow]Warning: No sections found. Falling back to per-page analysis.[/yellow]")
            all_sections = self._analyze_per_page(pdf)

        # Coverage check: ensure sections cover most pages
        all_sections = self._ensure_coverage(all_sections, pdf)

        return self._deduplicate(all_sections)

    def _ensure_coverage(self, sections: list[dict], pdf: LoadedPDF) -> list[dict]:
        """
        Check that sections cover all pages. If large gaps exist, 
        do targeted per-page analysis on uncovered pages.
        """
        if not sections:
            return self._analyze_per_page(pdf)

        total = pdf.total_pages
        covered = set()
        for s in sections:
            for p in range(s.get("start_page", 0), s.get("end_page", 0) + 1):
                covered.add(p)

        coverage = len(covered) / total if total > 0 else 0
        console.print(f"  Page coverage: {coverage:.0%} ({len(covered)}/{total} pages)")

        if coverage >= 0.7:
            return sections

        # Coverage too low — analyze uncovered pages individually
        console.print(f"  [yellow]Low coverage ({coverage:.0%}). Analyzing uncovered pages...[/yellow]")
        uncovered = sorted(set(range(total)) - covered)

        # Group consecutive uncovered pages into batches
        batches = []
        batch_start = uncovered[0]
        for i in range(1, len(uncovered)):
            if uncovered[i] != uncovered[i - 1] + 1:
                batches.append((batch_start, uncovered[i - 1]))
                batch_start = uncovered[i]
        batches.append((batch_start, uncovered[-1]))

        for start, end in batches:
            new_sections = self._analyze_page_range(pdf, start, end)
            sections.extend(new_sections)

        return sections

    def _analyze_per_page(self, pdf: LoadedPDF) -> list[dict]:
        """Fallback: analyze each page individually."""
        console.print("  [cyan]Per-page analysis (fallback)...[/cyan]")
        all_sections = []
        for page_num in range(pdf.total_pages):
            sections = self._analyze_page_range(pdf, page_num, page_num)
            if sections:
                all_sections.extend(sections)
            else:
                # Even fallback failed — create a minimal entry
                all_sections.append({
                    "title": f"Page {page_num + 1}",
                    "start_page": page_num,
                    "end_page": page_num,
                    "level": 1,
                    "summary": f"Content on page {page_num + 1}",
                    "keywords": [],
                })
        return all_sections

    def _analyze_page_range(self, pdf: LoadedPDF, start: int, end: int) -> list[dict]:
        """Analyze a specific page range."""
        images = pdf.get_page_images_batch(start, end)
        if not images:
            return []

        page_desc = f"page {start + 1}" if start == end else f"pages {start + 1}-{end + 1}"
        prompt = f"""Analyze {page_desc} of a {pdf.total_pages}-page document.
Identify ALL sections on these pages. Each page must be covered.
If a page is a single slide, it is its own section.
Output valid JSON with specific, detailed summaries — include actual data, names, numbers found on the page."""

        try:
            response = self.llm.complete_with_images(
                prompt=prompt, images=images,
                model=self.config.model.indexing_model,
                system_prompt=SYSTEM_PROMPT_ANALYZE,
                json_mode=True, max_tokens=2048,
            )
            return json.loads(response).get("sections", [])
        except Exception as e:
            console.print(f"  [yellow]Warning: Failed to analyze {page_desc}: {e}[/yellow]")
            return []

    def _build_tree(self, sections: list[dict], total_pages: int) -> TreeNode:
        if not sections:
            return TreeNode(node_id="0000", title="Document", summary="Full document content",
                            start_page=0, end_page=total_pages - 1)

        prompt = f"""Here are {len(sections)} sections identified in a {total_pages}-page document:

{json.dumps(sections, indent=2)}

Build a properly nested tree structure from these sections.
- The root node (level 0) should represent the entire document (pages 0 to {total_pages - 1}).
- Group related sections under parent nodes where appropriate.
- EVERY section above must appear in the tree — do not drop any.
- If sections are all at the same level (e.g. slides in a presentation), make them all direct children of the root.

Output valid JSON only."""
        try:
            response = self.llm.complete(
                prompt=prompt, model=self.config.model.indexing_model,
                system_prompt=SYSTEM_PROMPT_BUILD_TREE, json_mode=True, max_tokens=8192,
            )
            tree = TreeNode.from_dict(json.loads(response))
            self._set_parents(tree)
            return tree
        except Exception as e:
            console.print(f"  [yellow]Warning: Tree building failed, using flat structure: {e}[/yellow]")
            return self._flat_tree(sections, total_pages)

    def _validate_summaries(self, tree: TreeNode, pdf: LoadedPDF, max_validations: int = 10):
        nodes = []
        def _collect(n):
            if n.is_leaf or (n.end_page - n.start_page) >= 3:
                nodes.append(n)
            for c in n.children:
                _collect(c)
        _collect(tree)
        revised = 0
        for node in nodes[:max_validations]:
            start, end = node.start_page, min(node.end_page, node.start_page + 4)
            images = pdf.get_page_images_batch(start, end)
            if not images:
                continue
            prompt = f"Node: {node.title} (pages {node.start_page+1}-{node.end_page+1})\nSummary: {node.summary}\nKeywords: {', '.join(node.keywords)}\nAccurate?"
            try:
                r = json.loads(self.llm.complete_with_images(
                    prompt=prompt, images=images, model=self.config.model.indexing_model,
                    system_prompt=SYSTEM_PROMPT_VALIDATE, json_mode=True, max_tokens=1024,
                ))
                if not r.get("is_accurate", True):
                    if r.get("revised_summary"): node.summary = r["revised_summary"]
                    if r.get("revised_keywords"): node.keywords = r["revised_keywords"]
                    revised += 1
            except Exception as e:
                console.print(f"  [dim]Validation skipped for '{node.title}': {e}[/dim]")
        if revised:
            console.print(f"  Revised {revised} summaries")

    def _deduplicate(self, sections):
        seen, out = set(), []
        for s in sections:
            key = (s.get("title", ""), s.get("start_page", 0))
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def _flat_tree(self, sections, total_pages):
        root = TreeNode(node_id="0000", title="Document", summary="Full document", start_page=0, end_page=total_pages - 1)
        for i, s in enumerate(sections):
            c = TreeNode(node_id=f"{i+1:04d}", title=s.get("title", f"Section {i+1}"),
                         summary=s.get("summary", ""), keywords=s.get("keywords", []),
                         start_page=s.get("start_page", 0), end_page=s.get("end_page", 0), level=1, parent=root)
            root.children.append(c)
        return root

    def _set_parents(self, node):
        for c in node.children:
            c.parent = node
            self._set_parents(c)
