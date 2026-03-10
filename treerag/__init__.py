"""
TreeRAG — MCTS-powered Vectorless Document Retrieval
Two-phase search with folder management, caching, and parallel processing.
Supports PDF, CSV, and Excel files.
"""
__version__ = "0.3.0"
__author__ = "KnightOwl"

from .config import TreeRAGConfig
from .pipeline import TreeRAGPipeline, ChatMessage
from .models import DocumentIndex, FolderIndex, TreeNode, SearchResult, QueryResult
from .tabular_processor import SUPPORTED_TABULAR_EXTENSIONS

__all__ = [
    "TreeRAGConfig",
    "TreeRAGPipeline",
    "ChatMessage",
    "DocumentIndex",
    "FolderIndex",
    "TreeNode",
    "SearchResult",
    "QueryResult",
    "SUPPORTED_TABULAR_EXTENSIONS",
]
