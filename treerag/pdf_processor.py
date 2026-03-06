"""
PDF Processor — converts PDF pages to images using PyMuPDF.

Validates:
- File exists and is accessible
- File extension is .pdf
- File has valid PDF magic bytes (%PDF-)
- PDF can be opened without errors
- PDF has at least 1 page
- Page rendering doesn't crash
"""

from pathlib import Path
import fitz  # PyMuPDF

from .exceptions import (
    InvalidFileError, CorruptPDFError, EmptyPDFError, FileTooLargeError,
)

# PDF magic bytes
PDF_MAGIC = b"%PDF-"

# Safety limit — warn above this
MAX_PAGES_WARNING = 1000
MAX_PAGES_HARD_LIMIT = 5000


class PDFProcessor:
    def __init__(self, dpi: int = 200, max_pages: int = MAX_PAGES_HARD_LIMIT):
        self.dpi = dpi
        self.zoom = dpi / 72
        self.max_pages = max_pages

    def validate(self, pdf_path: str | Path) -> dict:
        """
        Validate a PDF file without fully loading it.
        Returns dict with file info or raises on fatal errors.
        """
        pdf_path = Path(pdf_path)

        # 1. File exists
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # 2. Is a file, not directory
        if not pdf_path.is_file():
            raise InvalidFileError(f"Not a file: {pdf_path}")

        # 3. Extension check
        if pdf_path.suffix.lower() != ".pdf":
            raise InvalidFileError(
                f"Not a PDF file: '{pdf_path.name}' (extension: {pdf_path.suffix}). "
                f"Only .pdf files are supported."
            )

        # 4. Not empty on disk
        file_size = pdf_path.stat().st_size
        if file_size == 0:
            raise InvalidFileError(f"File is empty (0 bytes): {pdf_path}")

        # 5. Magic bytes check
        try:
            with open(pdf_path, "rb") as f:
                header = f.read(5)
            if header != PDF_MAGIC:
                raise InvalidFileError(
                    f"File is not a valid PDF (bad header): {pdf_path.name}. "
                    f"Expected PDF magic bytes, got: {header!r}"
                )
        except PermissionError:
            raise InvalidFileError(f"Permission denied: {pdf_path}")

        # 6. Try opening with PyMuPDF
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise CorruptPDFError(
                f"Cannot open PDF '{pdf_path.name}': {e}. "
                f"The file may be corrupt, encrypted, or password-protected."
            )

        # 7. Page count
        page_count = len(doc)
        doc.close()

        if page_count == 0:
            raise EmptyPDFError(f"PDF has 0 pages: {pdf_path.name}")

        if page_count > self.max_pages:
            raise FileTooLargeError(
                f"PDF has {page_count} pages, exceeding the limit of {self.max_pages}. "
                f"Consider splitting the document."
            )

        return {
            "path": str(pdf_path),
            "filename": pdf_path.name,
            "file_size_bytes": file_size,
            "page_count": page_count,
            "is_large": page_count > MAX_PAGES_WARNING,
        }

    def load(self, pdf_path: str | Path) -> "LoadedPDF":
        """Load a validated PDF. Calls validate() first."""
        pdf_path = Path(pdf_path)
        info = self.validate(pdf_path)

        if info["is_large"]:
            import warnings
            warnings.warn(
                f"Large PDF: {info['filename']} has {info['page_count']} pages. "
                f"Indexing may take several minutes and cost $2+.",
                stacklevel=2,
            )

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise CorruptPDFError(f"Failed to open '{pdf_path.name}': {e}")

        return LoadedPDF(doc, pdf_path.name, self.zoom)


class LoadedPDF:
    def __init__(self, doc: fitz.Document, filename: str, zoom: float):
        self.doc = doc
        self.filename = filename
        self.zoom = zoom
        self._closed = False

    @property
    def total_pages(self) -> int:
        return len(self.doc)

    def get_page_image(self, page_num: int) -> bytes:
        self._check_closed()
        self._check_page_range(page_num)
        try:
            page = self.doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
            return pix.tobytes("png")
        except Exception as e:
            raise CorruptPDFError(
                f"Failed to render page {page_num + 1} of '{self.filename}': {e}"
            )

    def get_page_images_batch(self, start: int, end: int) -> list[bytes]:
        self._check_closed()
        start = max(0, start)
        end = min(end, self.total_pages - 1)
        if start > end:
            return []
        images = []
        for i in range(start, end + 1):
            try:
                images.append(self.get_page_image(i))
            except CorruptPDFError:
                # Skip corrupt pages, log warning
                import warnings
                warnings.warn(f"Skipping corrupt page {i + 1} in '{self.filename}'")
                continue
        return images

    def get_page_text(self, page_num: int) -> str:
        self._check_closed()
        self._check_page_range(page_num)
        try:
            return self.doc[page_num].get_text()
        except Exception:
            return f"[Error extracting text from page {page_num + 1}]"

    def get_pages_text_batch(self, start: int, end: int) -> str:
        self._check_closed()
        start = max(0, start)
        end = min(end, self.total_pages - 1)
        parts = []
        for i in range(start, end + 1):
            text = self.get_page_text(i)
            parts.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(parts)

    def _check_closed(self):
        if self._closed:
            raise RuntimeError(f"PDF '{self.filename}' is already closed.")

    def _check_page_range(self, page_num: int):
        if page_num < 0 or page_num >= self.total_pages:
            raise IndexError(
                f"Page {page_num} out of range for '{self.filename}' "
                f"(0 to {self.total_pages - 1})"
            )

    def close(self):
        if not self._closed:
            self.doc.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        # Safety net — close if user forgot
        try:
            self.close()
        except Exception:
            pass
