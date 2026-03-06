"""Tests for custom exceptions."""

from treerag.exceptions import (
    InvalidFileError, CorruptPDFError, EmptyPDFError, FileTooLargeError,
    FolderNotFoundError, FolderAlreadyExistsError, DocumentNotFoundError,
    IndexNotFoundError, IndexCorruptError, PDFMissingError,
    IndexingFailedError, BatchIndexingError, EmptyFolderError,
)


class TestExceptions:
    def test_all_exceptions_are_importable(self):
        """Verify all 13 custom exceptions exist and can be instantiated."""
        exceptions = [
            InvalidFileError, CorruptPDFError, EmptyPDFError, FileTooLargeError,
            FolderNotFoundError, FolderAlreadyExistsError, DocumentNotFoundError,
            IndexNotFoundError, IndexCorruptError, PDFMissingError,
            IndexingFailedError, BatchIndexingError, EmptyFolderError,
        ]
        assert len(exceptions) == 13
        for exc_class in exceptions:
            exc = exc_class("test message")
            assert str(exc) == "test message"

    def test_batch_indexing_error_fields(self):
        exc = BatchIndexingError(
            "Batch failed",
            successes=["a.pdf"],
            failures=[("b.pdf", "corrupt")],
        )
        assert exc.successes == ["a.pdf"]
        assert exc.failures == [("b.pdf", "corrupt")]
