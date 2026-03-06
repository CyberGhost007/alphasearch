"""
Custom exceptions for TreeRAG.
Each exception maps to a specific failure mode with actionable messages.
"""


class TreeRAGError(Exception):
    """Base exception for all TreeRAG errors."""
    pass


# --- File Validation Errors ---

class InvalidFileError(TreeRAGError):
    """File is not a valid PDF or is unsupported."""
    pass


class CorruptPDFError(TreeRAGError):
    """PDF file is corrupt and cannot be opened."""
    pass


class EmptyPDFError(TreeRAGError):
    """PDF has zero pages."""
    pass


class FileTooLargeError(TreeRAGError):
    """PDF exceeds the configured page limit."""
    pass


# --- Folder Errors ---

class FolderNotFoundError(TreeRAGError):
    """Folder does not exist. User must create it first."""
    pass


class FolderAlreadyExistsError(TreeRAGError):
    """Folder already exists."""
    pass


class DocumentNotFoundError(TreeRAGError):
    """Document not found in the folder."""
    pass


class DuplicateFilenameWarning(TreeRAGError):
    """Same filename being added from a different source path."""
    pass


# --- Index Errors ---

class IndexNotFoundError(TreeRAGError):
    """Index JSON file is missing from disk (manually deleted or corrupt)."""
    pass


class IndexCorruptError(TreeRAGError):
    """Index JSON exists but cannot be parsed."""
    pass


class PDFMissingError(TreeRAGError):
    """Meta-tree references a PDF that no longer exists on disk."""
    pass


# --- Indexing Pipeline Errors ---

class IndexingFailedError(TreeRAGError):
    """Indexing pipeline failed (LLM error, parse error, etc.)."""
    pass


class BatchIndexingError(TreeRAGError):
    """Some files in a batch add failed. Contains successes and failures."""
    def __init__(self, message: str, successes: list[str] = None, failures: list[tuple[str, str]] = None):
        super().__init__(message)
        self.successes = successes or []
        self.failures = failures or []  # [(filename, error_message)]


# --- Search Errors ---

class SearchError(TreeRAGError):
    """Search pipeline failed."""
    pass


class EmptyFolderError(TreeRAGError):
    """Folder has no documents to search."""
    pass
