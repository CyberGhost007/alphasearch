"""
Core data structures for TreeRAG.
All save operations use atomic writes (write to temp, then rename) to prevent corruption.
"""

import json
import time
import math
import hashlib
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .exceptions import IndexCorruptError


def _atomic_write(path: Path, data: str):
    """Write to a temp file then atomically rename. Prevents corrupt files on crash/disk-full."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp_path, str(path))  # Atomic on POSIX
    except Exception:
        # Cleanup temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def compute_file_hash(file_path: str | Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Tree Node
# =============================================================================

@dataclass
class TreeNode:
    node_id: str
    title: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    start_page: int = 0
    end_page: int = 0
    level: int = 0
    children: list["TreeNode"] = field(default_factory=list)
    content: Optional[str] = None

    visit_count: int = 0
    total_reward: float = 0.0
    parent: Optional["TreeNode"] = None
    pruned: bool = False

    @property
    def average_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        if self.visit_count == 0:
            return float("inf")
        if self.parent is None or self.parent.visit_count == 0:
            return self.average_reward
        return self.average_reward + exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )

    def reset_mcts_state(self):
        self.visit_count = 0
        self.total_reward = 0.0
        self.pruned = False
        for c in self.children:
            c.reset_mcts_state()

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def page_range(self) -> str:
        if self.start_page == self.end_page:
            return f"p.{self.start_page + 1}"
        return f"pp.{self.start_page + 1}-{self.end_page + 1}"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "title": self.title, "summary": self.summary,
            "keywords": self.keywords, "start_page": self.start_page,
            "end_page": self.end_page, "level": self.level,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict, parent: Optional["TreeNode"] = None) -> "TreeNode":
        node = cls(
            node_id=data.get("node_id", "0000"), title=data.get("title", "Unknown"),
            summary=data.get("summary", ""), keywords=data.get("keywords", []),
            start_page=data.get("start_page", 0), end_page=data.get("end_page", 0),
            level=data.get("level", 0), parent=parent,
        )
        node.children = [cls.from_dict(c, parent=node) for c in data.get("children", [])]
        return node

    def pretty_print(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}[{self.node_id}] {self.title} ({self.page_range})"]
        lines.append(f"{prefix}  Summary: {self.summary[:80]}...")
        if self.keywords:
            lines.append(f"{prefix}  Keywords: {', '.join(self.keywords)}")
        for c in self.children:
            lines.append(c.pretty_print(indent + 1))
        return "\n".join(lines)


# =============================================================================
# Document Index
# =============================================================================

@dataclass
class DocumentIndex:
    document_id: str
    filename: str
    total_pages: int
    description: str = ""
    root: Optional[TreeNode] = None
    created_at: float = field(default_factory=time.time)
    file_hash: str = ""
    pdf_path: str = ""

    def save(self, path: str | Path):
        data = {
            "document_id": self.document_id, "filename": self.filename,
            "total_pages": self.total_pages, "description": self.description,
            "created_at": self.created_at, "file_hash": self.file_hash,
            "pdf_path": self.pdf_path,
            "tree": self.root.to_dict() if self.root else None,
        }
        _atomic_write(Path(path), json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DocumentIndex":
        path = Path(path)
        if not path.exists():
            from .exceptions import IndexNotFoundError
            raise IndexNotFoundError(f"Index file not found: {path}")
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise IndexCorruptError(f"Index file corrupt: {path} — {e}")

        index = cls(
            document_id=data.get("document_id", "unknown"),
            filename=data.get("filename", "unknown"),
            total_pages=data.get("total_pages", 0),
            description=data.get("description", ""),
            created_at=data.get("created_at", 0),
            file_hash=data.get("file_hash", ""),
            pdf_path=data.get("pdf_path", ""),
        )
        if data.get("tree"):
            index.root = TreeNode.from_dict(data["tree"])
        return index

    def get_all_nodes(self) -> list[TreeNode]:
        nodes = []
        def _collect(n):
            nodes.append(n)
            for c in n.children:
                _collect(c)
        if self.root:
            _collect(self.root)
        return nodes

    def find_node(self, node_id: str) -> Optional[TreeNode]:
        for n in self.get_all_nodes():
            if n.node_id == node_id:
                return n
        return None


# =============================================================================
# Folder Index
# =============================================================================

@dataclass
class FolderDocEntry:
    document_id: str
    filename: str
    total_pages: int
    summary: str
    keywords: list[str] = field(default_factory=list)
    index_path: str = ""
    pdf_path: str = ""
    file_hash: str = ""
    indexed_at: float = field(default_factory=time.time)
    source_path: str = ""   # Original path the file was added from (for duplicate detection)

    visit_count: int = 0
    total_reward: float = 0.0
    pruned: bool = False

    @property
    def average_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    def ucb1(self, parent_visits: int, exploration_constant: float = 1.414) -> float:
        if self.visit_count == 0:
            return float("inf")
        if parent_visits == 0:
            return self.average_reward
        return self.average_reward + exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )

    def reset_mcts_state(self):
        self.visit_count = 0
        self.total_reward = 0.0
        self.pruned = False

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id, "filename": self.filename,
            "total_pages": self.total_pages, "summary": self.summary,
            "keywords": self.keywords, "index_path": self.index_path,
            "pdf_path": self.pdf_path, "file_hash": self.file_hash,
            "indexed_at": self.indexed_at, "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FolderDocEntry":
        return cls(
            document_id=data.get("document_id", ""),
            filename=data.get("filename", ""),
            total_pages=data.get("total_pages", 0),
            summary=data.get("summary", ""),
            keywords=data.get("keywords", []),
            index_path=data.get("index_path", ""),
            pdf_path=data.get("pdf_path", ""),
            file_hash=data.get("file_hash", ""),
            indexed_at=data.get("indexed_at", 0),
            source_path=data.get("source_path", ""),
        )


@dataclass
class FolderIndex:
    folder_name: str
    folder_path: str
    documents: list[FolderDocEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    @property
    def total_pages(self) -> int:
        return sum(d.total_pages for d in self.documents)

    def add_document(self, entry: FolderDocEntry):
        self.documents = [d for d in self.documents if d.filename != entry.filename]
        self.documents.append(entry)
        self.updated_at = time.time()

    def remove_document(self, filename: str) -> Optional[FolderDocEntry]:
        """Remove and return the entry, or None if not found."""
        entry = self.get_document(filename)
        if entry:
            self.documents = [d for d in self.documents if d.filename != filename]
            self.updated_at = time.time()
        return entry

    def get_document(self, filename: str) -> Optional[FolderDocEntry]:
        for d in self.documents:
            if d.filename == filename:
                return d
        return None

    def get_document_by_id(self, doc_id: str) -> Optional[FolderDocEntry]:
        for d in self.documents:
            if d.document_id == doc_id:
                return d
        return None

    def needs_reindex(self, filename: str, current_hash: str) -> bool:
        entry = self.get_document(filename)
        return entry is None or entry.file_hash != current_hash

    def reset_mcts_state(self):
        for d in self.documents:
            d.reset_mcts_state()

    def save(self, path: str | Path):
        data = {
            "folder_name": self.folder_name, "folder_path": self.folder_path,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "documents": [d.to_dict() for d in self.documents],
        }
        _atomic_write(Path(path), json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FolderIndex":
        path = Path(path)
        if not path.exists():
            from .exceptions import FolderNotFoundError
            raise FolderNotFoundError(f"Folder index not found: {path}")
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise IndexCorruptError(f"Folder index corrupt: {path} — {e}")

        folder = cls(
            folder_name=data.get("folder_name", "unknown"),
            folder_path=data.get("folder_path", ""),
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
        )
        folder.documents = [FolderDocEntry.from_dict(d) for d in data.get("documents", [])]
        return folder

    def pretty_print(self) -> str:
        lines = [
            f"Folder: {self.folder_name} ({self.folder_path})",
            f"Documents: {self.total_documents} | Total pages: {self.total_pages}", "",
        ]
        for doc in self.documents:
            lines.append(f"  [{doc.document_id[:8]}] {doc.filename} ({doc.total_pages} pages)")
            lines.append(f"    Summary: {doc.summary[:100]}...")
            if doc.keywords:
                lines.append(f"    Keywords: {', '.join(doc.keywords)}")
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# Search Results
# =============================================================================

@dataclass
class SearchResult:
    node: TreeNode
    relevance_score: float
    visit_count: int
    document_filename: str = ""
    document_id: str = ""
    content: str = ""
    reasoning: str = ""

    @property
    def citation(self) -> str:
        if self.document_filename:
            return f"[{self.document_filename} → {self.node.title}, {self.node.page_range}]"
        return f"[{self.node.title}, {self.node.page_range}]"


@dataclass
class DocumentScore:
    entry: FolderDocEntry
    relevance_score: float
    visit_count: int


@dataclass
class SearchStats:
    """Detailed statistics about an adaptive MCTS search run."""
    iterations_used: int = 0
    iterations_max: int = 0
    converged: bool = False
    convergence_reason: str = ""
    convergence_iteration: int = 0
    total_nodes: int = 0
    visited_nodes: int = 0
    coverage_pct: float = 0.0
    pruned_branches: int = 0
    pruned_at_iterations: list[int] = field(default_factory=list)
    exploration_start: float = 0.0
    exploration_end: float = 0.0
    mean_reward: float = 0.0
    reward_variance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "iterations_used": self.iterations_used,
            "iterations_max": self.iterations_max,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "convergence_iteration": self.convergence_iteration,
            "total_nodes": self.total_nodes,
            "visited_nodes": self.visited_nodes,
            "coverage_pct": round(self.coverage_pct, 1),
            "pruned_branches": self.pruned_branches,
            "mean_reward": round(self.mean_reward, 4),
            "reward_variance": round(self.reward_variance, 4),
        }


@dataclass
class QueryResult:
    query: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    documents_searched: list[str] = field(default_factory=list)
    total_mcts_iterations: int = 0
    total_llm_calls: int = 0
    latency_seconds: float = 0.0
    phase1_time: float = 0.0
    phase2_time: float = 0.0
    phase1_stats: Optional["SearchStats"] = None
    phase2_stats: list["SearchStats"] = field(default_factory=list)
