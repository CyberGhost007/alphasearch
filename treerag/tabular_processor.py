"""
Tabular Processor — reads CSV/Excel files and serves markdown table pages.

This is a thin I/O layer. It:
1. Validates CSV/Excel files (corrupt, empty, too large)
2. Reads them into a DataFrame
3. Chunks rows into pages of markdown tables
4. Exposes the same interface as LoadedPDF (total_pages, get_page_text,
   get_page_image, etc.) so the existing PDF Indexer can handle tabular
   data identically to PDFs.

All intelligence (summarization, tree building, keyword extraction) lives
in the Indexer. All search optimization (BM25, PUCT) lives in mcts.py
and bm25.py.
"""

import io
import math
from pathlib import Path
from typing import Optional

from .exceptions import InvalidTabularError, EmptyTabularError, FileTooLargeError

# Lazy imports — pandas/matplotlib only needed when actually processing tabular files.
# This prevents crashing the server if pandas isn't installed but no CSV/Excel is used.
pd = None
plt = None

def _ensure_pandas():
    global pd
    if pd is None:
        try:
            import pandas as _pd
            pd = _pd
        except ImportError:
            raise ImportError(
                "CSV/Excel support requires pandas. Install with: "
                "pip install pandas openpyxl matplotlib"
            )
    return pd

def _ensure_matplotlib():
    global plt
    if plt is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            plt = _plt
        except ImportError:
            pass
    return plt

SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}

MAX_ROWS_WARNING = 50_000
MAX_ROWS_HARD_LIMIT = 100_000

# Rendering limits
MAX_DISPLAY_COLUMNS = 20
MAX_COL_WIDTH = 40
MAX_RENDER_ROWS = 30      # matplotlib table rendering cap per page


class TabularProcessor:
    """Reads and validates CSV/Excel files, serves pages of markdown tables."""

    def __init__(self, rows_per_page: int = 50, max_rows: int = MAX_ROWS_HARD_LIMIT):
        self.rows_per_page = rows_per_page
        self.max_rows = max_rows

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self, file_path: str | Path) -> dict:
        """Validate a tabular file without fully loading it."""
        _pd = _ensure_pandas()
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise InvalidTabularError(f"Not a file: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_TABULAR_EXTENSIONS:
            raise InvalidTabularError(
                f"Unsupported file type: '{file_path.name}' (extension: {ext}). "
                f"Supported: {', '.join(sorted(SUPPORTED_TABULAR_EXTENSIONS))}"
            )

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise InvalidTabularError(f"File is empty (0 bytes): {file_path}")

        # Quick parse check
        try:
            if ext == ".csv":
                df = _pd.read_csv(file_path, nrows=5)
            else:
                df = _pd.read_excel(file_path, nrows=5)
        except Exception as e:
            raise InvalidTabularError(
                f"Cannot parse '{file_path.name}': {e}. "
                f"Check the file format and encoding."
            )

        if len(df.columns) == 0:
            raise EmptyTabularError(f"File has no columns: {file_path.name}")

        # Get row count without loading everything
        try:
            if ext == ".csv":
                row_count = sum(
                    1 for _ in open(file_path, encoding="utf-8", errors="ignore")
                ) - 1
            else:
                full_df = _pd.read_excel(file_path, sheet_name=0)
                row_count = len(full_df)
        except Exception:
            row_count = -1  # Unknown

        if row_count == 0:
            raise EmptyTabularError(f"File has no data rows: {file_path.name}")

        if row_count > self.max_rows:
            raise FileTooLargeError(
                f"File has {row_count:,} rows, exceeding the limit of "
                f"{self.max_rows:,}. Consider splitting the file."
            )

        return {
            "path": str(file_path),
            "filename": file_path.name,
            "file_size_bytes": file_size,
            "row_count": row_count,
            "column_count": len(df.columns),
            "is_large": row_count > MAX_ROWS_WARNING,
        }

    # =========================================================================
    # Loading
    # =========================================================================

    def load(self, file_path: str | Path) -> "LoadedTabular":
        """Load a tabular file and return a LoadedTabular (same interface as LoadedPDF)."""
        _pd = _ensure_pandas()
        file_path = Path(file_path)
        self.validate(file_path)

        ext = file_path.suffix.lower()
        sheets = {}

        if ext == ".csv":
            df = _pd.read_csv(file_path, low_memory=False)
            sheets["Sheet1"] = df
        else:
            xls = _pd.read_excel(file_path, sheet_name=None)
            sheets = xls if isinstance(xls, dict) else {"Sheet1": xls}

        # Combine all sheets with sheet name tracking
        all_dfs = []
        sheet_boundaries: list[tuple[str, int, int]] = []
        current_row = 0

        for sheet_name, sdf in sheets.items():
            if sdf.empty:
                continue
            sdf = sdf.copy()
            sdf["__sheet__"] = sheet_name
            all_dfs.append(sdf)
            sheet_boundaries.append(
                (sheet_name, current_row, current_row + len(sdf) - 1)
            )
            current_row += len(sdf)

        if not all_dfs:
            raise EmptyTabularError(f"All sheets are empty: {file_path.name}")

        combined = _pd.concat(all_dfs, ignore_index=True)

        return LoadedTabular(
            df=combined,
            filename=file_path.name,
            rows_per_page=self.rows_per_page,
            sheet_boundaries=sheet_boundaries,
        )


class LoadedTabular:
    """
    Loaded tabular file — same interface as LoadedPDF.

    Pages are chunks of rows rendered as markdown tables.
    The existing PDF Indexer reads these pages identically to PDF text pages.
    """

    def __init__(
        self,
        df,  # pandas DataFrame
        filename: str,
        rows_per_page: int,
        sheet_boundaries: list[tuple[str, int, int]],
    ):
        self.df = df
        self.filename = filename
        self.rows_per_page = rows_per_page
        self.sheet_boundaries = sheet_boundaries
        self._closed = False

        # Data columns (exclude internal __sheet__ tracker)
        self._data_cols = [c for c in df.columns if c != "__sheet__"]
        # Display columns (cap width for rendering)
        self._display_cols = self._data_cols[:MAX_DISPLAY_COLUMNS]

    @property
    def total_pages(self) -> int:
        return max(1, math.ceil(len(self.df) / self.rows_per_page))

    # =========================================================================
    # Page text — clean markdown table (what Indexer + BM25 read)
    # =========================================================================

    def get_page_text(self, page_num: int) -> str:
        """
        Get a page as a markdown table.

        Format:
            File: sales_data.csv | Sheet: Sheet1 | Rows 1-50 of 2,400

            | Region | Product | Revenue | Date |
            |--------|---------|---------|------|
            | NA     | Widget  | 5000    | 2024-01-15 |
            ...

        This is what the LLM reads during indexing and what BM25 tokenizes.
        """
        _pd = _ensure_pandas()
        self._check_closed()
        self._check_page_range(page_num)

        start_row = page_num * self.rows_per_page
        end_row = min(start_row + self.rows_per_page, len(self.df))
        chunk = self.df.iloc[start_row:end_row][self._data_cols]

        # Header with context
        parts = [
            f"File: {self.filename} | "
            f"Rows {start_row + 1}-{end_row} of {len(self.df):,}"
        ]

        # Sheet info for multi-sheet Excel files
        if len(self.sheet_boundaries) > 1:
            for sheet_name, s_start, s_end in self.sheet_boundaries:
                if s_start <= start_row <= s_end:
                    parts[0] = (
                        f"File: {self.filename} | Sheet: {sheet_name} | "
                        f"Rows {start_row + 1}-{end_row} of {len(self.df):,}"
                    )
                    break

        parts.append("")

        # Markdown table
        display = chunk[self._display_cols].copy()
        # Truncate wide string values
        for col in display.columns:
            if display[col].dtype == object:
                display[col] = display[col].astype(str).str[:MAX_COL_WIDTH]

        # Build markdown manually for clean output
        headers = list(display.columns)
        parts.append("| " + " | ".join(str(h) for h in headers) + " |")
        parts.append("| " + " | ".join("---" for _ in headers) + " |")

        for _, row in display.iterrows():
            vals = []
            for h in headers:
                v = row[h]
                # Clean NaN display
                if _pd.isna(v):
                    vals.append("")
                else:
                    vals.append(str(v))
            parts.append("| " + " | ".join(vals) + " |")

        return "\n".join(parts)

    # =========================================================================
    # Page images — for vision-based processing (optional)
    # =========================================================================

    def get_page_image(self, page_num: int) -> bytes:
        """Render a page as a table image (PNG)."""
        self._check_closed()
        self._check_page_range(page_num)

        _plt = _ensure_matplotlib()
        if _plt is None:
            raise RuntimeError(
                "matplotlib is required for table rendering. "
                "Install with: pip install matplotlib"
            )

        start_row = page_num * self.rows_per_page
        end_row = min(start_row + self.rows_per_page, len(self.df))
        chunk = self.df.iloc[start_row:end_row][self._display_cols]

        # Cap rows for rendering performance
        if len(chunk) > MAX_RENDER_ROWS:
            chunk = chunk.head(MAX_RENDER_ROWS)

        # Truncate cell values for display
        display_data = []
        for _, row in chunk.iterrows():
            display_data.append([str(v)[:30] for v in row])

        col_labels = [str(c)[:20] for c in chunk.columns]

        fig_height = max(2, 0.35 * len(display_data) + 1.5)
        fig_width = max(8, 0.8 * len(col_labels))
        fig, ax = _plt.subplots(
            figsize=(min(fig_width, 20), min(fig_height, 15))
        )
        ax.axis("off")

        title = (
            f"{self.filename} — Page {page_num + 1} "
            f"(Rows {start_row + 1}-{end_row})"
        )
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=10)

        if display_data:
            table = ax.table(
                cellText=display_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.auto_set_column_width(range(len(col_labels)))

            # Style header row
            for j in range(len(col_labels)):
                table[0, j].set_facecolor("#4472C4")
                table[0, j].set_text_props(color="white", fontweight="bold")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.2)
        _plt.close(fig)
        buf.seek(0)
        return buf.read()

    # =========================================================================
    # Batch accessors (match LoadedPDF interface exactly)
    # =========================================================================

    def get_page_images_batch(self, start: int, end: int) -> list[bytes]:
        """Get page images for a range of pages."""
        self._check_closed()
        start = max(0, start)
        end = min(end, self.total_pages - 1)
        if start > end:
            return []
        images = []
        for i in range(start, end + 1):
            try:
                images.append(self.get_page_image(i))
            except Exception:
                continue
        return images

    def get_pages_text_batch(self, start: int, end: int) -> str:
        """Get concatenated text for a range of pages."""
        self._check_closed()
        start = max(0, start)
        end = min(end, self.total_pages - 1)
        parts = []
        for i in range(start, end + 1):
            parts.append(self.get_page_text(i))
        return "\n\n".join(parts)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _check_closed(self):
        if self._closed:
            raise RuntimeError(f"File '{self.filename}' is already closed.")

    def _check_page_range(self, page_num: int):
        if page_num < 0 or page_num >= self.total_pages:
            raise IndexError(
                f"Page {page_num} out of range for '{self.filename}' "
                f"(0 to {self.total_pages - 1})"
            )

    def close(self):
        if not self._closed:
            self.df = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
