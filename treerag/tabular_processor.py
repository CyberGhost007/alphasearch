"""
Tabular Processor — reads CSV/Excel files with smart data-aware sectioning.

Unlike PDFs where structure must be inferred, tabular data has explicit
column types and natural groupings. This processor:
1. Detects column types (numeric, categorical, date)
2. Finds natural grouping columns (low-cardinality categoricals)
3. Computes per-section statistics (sum, mean, min/max)
4. Builds sections based on data patterns, not blind row chunks
"""

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "CSV/Excel support requires pandas. Install with: "
        "pip install pandas openpyxl matplotlib"
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError:
    plt = None

from .exceptions import InvalidTabularError, EmptyTabularError, FileTooLargeError

SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}

MAX_ROWS_WARNING = 50000
MAX_ROWS_HARD_LIMIT = 100000

# Max columns to display in rendered tables (prevents ultra-wide images)
MAX_DISPLAY_COLUMNS = 20
# Max column width in characters for text rendering
MAX_COL_WIDTH = 40


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    is_numeric: bool = False
    is_categorical: bool = False
    is_date: bool = False
    stats: dict = field(default_factory=dict)  # min, max, mean, median, sum for numeric
    top_values: list = field(default_factory=list)  # top 5 value counts for categorical


@dataclass
class TabularAnalysis:
    filename: str
    total_rows: int
    total_columns: int
    columns: list[ColumnInfo] = field(default_factory=list)
    grouping_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    sheet_names: list[str] = field(default_factory=list)
    suggested_sections: list[dict] = field(default_factory=list)
    global_stats: dict = field(default_factory=dict)


class TabularProcessor:
    def __init__(self, rows_per_page: int = 50, max_rows: int = MAX_ROWS_HARD_LIMIT):
        self.rows_per_page = rows_per_page
        self.max_rows = max_rows

    def validate(self, file_path: str | Path) -> dict:
        """Validate a tabular file without fully loading it."""
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
                df = pd.read_csv(file_path, nrows=5)
            else:
                df = pd.read_excel(file_path, nrows=5)
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
                row_count = sum(1 for _ in open(file_path, encoding="utf-8", errors="ignore")) - 1
            else:
                full_df = pd.read_excel(file_path, sheet_name=0)
                row_count = len(full_df)
        except Exception:
            row_count = -1  # Unknown

        if row_count == 0:
            raise EmptyTabularError(f"File has no data rows: {file_path.name}")

        if row_count > self.max_rows:
            raise FileTooLargeError(
                f"File has {row_count:,} rows, exceeding the limit of {self.max_rows:,}. "
                f"Consider splitting the file."
            )

        return {
            "path": str(file_path),
            "filename": file_path.name,
            "file_size_bytes": file_size,
            "row_count": row_count,
            "column_count": len(df.columns),
            "is_large": row_count > MAX_ROWS_WARNING,
        }

    def load(self, file_path: str | Path) -> "LoadedTabular":
        """Load and analyze a tabular file."""
        file_path = Path(file_path)
        self.validate(file_path)

        ext = file_path.suffix.lower()
        sheets = {}

        if ext == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
            sheets["Sheet1"] = df
        else:
            xls = pd.read_excel(file_path, sheet_name=None)
            sheets = xls if isinstance(xls, dict) else {"Sheet1": xls}

        # Combine all sheets with sheet tracking
        all_dfs = []
        sheet_boundaries = []  # (sheet_name, start_row, end_row)
        current_row = 0
        for sheet_name, sdf in sheets.items():
            if sdf.empty:
                continue
            sdf = sdf.copy()
            sdf["__sheet__"] = sheet_name
            all_dfs.append(sdf)
            sheet_boundaries.append((sheet_name, current_row, current_row + len(sdf) - 1))
            current_row += len(sdf)

        if not all_dfs:
            raise EmptyTabularError(f"All sheets are empty: {file_path.name}")

        combined = pd.concat(all_dfs, ignore_index=True)

        # Run analysis
        analysis = self._analyze(combined, file_path.name, list(sheets.keys()), sheet_boundaries)

        # Build sections based on analysis
        sections = self._build_sections(combined, analysis, sheet_boundaries)
        analysis.suggested_sections = sections

        return LoadedTabular(
            df=combined,
            filename=file_path.name,
            rows_per_page=self.rows_per_page,
            analysis=analysis,
            sections=sections,
            sheet_boundaries=sheet_boundaries,
        )

    def analyze(self, file_path: str | Path) -> TabularAnalysis:
        """Run analysis without fully loading into LoadedTabular."""
        loaded = self.load(file_path)
        analysis = loaded.analysis
        loaded.close()
        return analysis

    def _analyze(
        self, df: "pd.DataFrame", filename: str,
        sheet_names: list[str], sheet_boundaries: list,
    ) -> TabularAnalysis:
        """Analyze dataframe structure, detect types and groupings."""
        # Drop the __sheet__ column for analysis
        data_cols = [c for c in df.columns if c != "__sheet__"]
        analysis = TabularAnalysis(
            filename=filename,
            total_rows=len(df),
            total_columns=len(data_cols),
            sheet_names=sheet_names,
        )

        for col_name in data_cols:
            series = df[col_name]
            col_info = ColumnInfo(
                name=col_name,
                dtype=str(series.dtype),
                null_count=int(series.isna().sum()),
                null_pct=round(series.isna().mean() * 100, 1),
                unique_count=int(series.nunique()),
            )

            # Detect column type
            if pd.api.types.is_numeric_dtype(series):
                col_info.is_numeric = True
                non_null = series.dropna()
                if len(non_null) > 0:
                    col_info.stats = {
                        "min": float(non_null.min()),
                        "max": float(non_null.max()),
                        "mean": round(float(non_null.mean()), 2),
                        "median": round(float(non_null.median()), 2),
                        "sum": float(non_null.sum()),
                        "std": round(float(non_null.std()), 2) if len(non_null) > 1 else 0.0,
                    }
                analysis.numeric_columns.append(col_name)

            elif pd.api.types.is_datetime64_any_dtype(series):
                col_info.is_date = True
                non_null = series.dropna()
                if len(non_null) > 0:
                    col_info.stats = {
                        "min": str(non_null.min()),
                        "max": str(non_null.max()),
                    }
                analysis.date_columns.append(col_name)

            else:
                # Try to parse as date
                if series.dtype == object and len(series.dropna()) > 0:
                    sample = series.dropna().head(20)
                    try:
                        parsed = pd.to_datetime(sample, format="mixed", dayfirst=False)
                        if parsed.notna().sum() >= len(sample) * 0.8:
                            col_info.is_date = True
                            full_parsed = pd.to_datetime(series, errors="coerce")
                            non_null = full_parsed.dropna()
                            if len(non_null) > 0:
                                col_info.stats = {
                                    "min": str(non_null.min()),
                                    "max": str(non_null.max()),
                                }
                            analysis.date_columns.append(col_name)
                    except (ValueError, TypeError):
                        pass

                if not col_info.is_date:
                    # Categorical check: low cardinality
                    if col_info.unique_count <= 20 and col_info.unique_count > 0:
                        col_info.is_categorical = True
                        vc = series.value_counts().head(5)
                        col_info.top_values = [
                            {"value": str(v), "count": int(c)} for v, c in vc.items()
                        ]
                        analysis.grouping_columns.append(col_name)

            analysis.columns.append(col_info)

        # Global stats
        analysis.global_stats = {
            "total_rows": len(df),
            "total_columns": len(data_cols),
            "numeric_columns": len(analysis.numeric_columns),
            "categorical_columns": len(analysis.grouping_columns),
            "date_columns": len(analysis.date_columns),
            "total_nulls": int(df[data_cols].isna().sum().sum()),
        }

        return analysis

    def _build_sections(
        self, df: "pd.DataFrame", analysis: TabularAnalysis,
        sheet_boundaries: list,
    ) -> list[dict]:
        """Build smart sections based on data patterns."""
        data_cols = [c for c in df.columns if c != "__sheet__"]
        multi_sheet = len(sheet_boundaries) > 1

        # Strategy 1: Multiple sheets → each sheet is a section
        if multi_sheet:
            sections = []
            for sheet_name, start_row, end_row in sheet_boundaries:
                sheet_df = df.iloc[start_row:end_row + 1]
                section = self._make_section(
                    sheet_df, data_cols, analysis,
                    title=f"Sheet: {sheet_name}",
                    start_row=start_row, end_row=end_row, level=1,
                )
                sections.append(section)
                # Add sub-sections within each sheet
                sub_sections = self._build_subsections(
                    sheet_df, data_cols, analysis,
                    base_start=start_row, level=2,
                )
                sections.extend(sub_sections)
            return sections

        # Strategy 2: Grouping column found → group by it
        if analysis.grouping_columns:
            group_col = analysis.grouping_columns[0]  # Use the first one
            sections = []
            for group_val, group_df in df.groupby(group_col, sort=True):
                if group_df.empty:
                    continue
                start_row = int(group_df.index[0])
                end_row = int(group_df.index[-1])
                section = self._make_section(
                    group_df, data_cols, analysis,
                    title=f"{group_col}: {group_val}",
                    start_row=start_row, end_row=end_row, level=1,
                )
                sections.append(section)

                # Sub-sections by date or chunks
                sub_sections = self._build_subsections(
                    group_df, data_cols, analysis,
                    base_start=start_row, level=2,
                )
                sections.extend(sub_sections)
            return sections

        # Strategy 3: Date column → group by time period
        if analysis.date_columns:
            return self._build_date_sections(df, data_cols, analysis)

        # Strategy 4: Fallback — row chunks with stats
        return self._build_chunk_sections(df, data_cols, analysis)

    def _build_subsections(
        self, df: "pd.DataFrame", data_cols: list, analysis: TabularAnalysis,
        base_start: int, level: int,
    ) -> list[dict]:
        """Build sub-sections within a group (by date or chunks)."""
        if len(df) <= self.rows_per_page:
            return []  # Too small to subdivide

        # Try date-based sub-sections
        if analysis.date_columns:
            date_col = analysis.date_columns[0]
            try:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                if dates.notna().sum() >= len(df) * 0.5:
                    df_with_date = df.copy()
                    df_with_date["__period__"] = dates.dt.to_period("Q")
                    sections = []
                    for period, period_df in df_with_date.groupby("__period__", sort=True):
                        if period_df.empty or pd.isna(period):
                            continue
                        start_row = int(period_df.index[0])
                        end_row = int(period_df.index[-1])
                        sections.append(self._make_section(
                            period_df, data_cols, analysis,
                            title=str(period), start_row=start_row,
                            end_row=end_row, level=level,
                        ))
                    if len(sections) > 1:
                        return sections
            except Exception:
                pass

        # Fallback: chunks
        sections = []
        for chunk_start in range(0, len(df), self.rows_per_page):
            chunk_end = min(chunk_start + self.rows_per_page - 1, len(df) - 1)
            chunk_df = df.iloc[chunk_start:chunk_end + 1]
            abs_start = base_start + chunk_start
            abs_end = base_start + chunk_end
            sections.append(self._make_section(
                chunk_df, data_cols, analysis,
                title=f"Rows {abs_start + 1}-{abs_end + 1}",
                start_row=abs_start, end_row=abs_end, level=level,
            ))
        return sections

    def _build_date_sections(
        self, df: "pd.DataFrame", data_cols: list, analysis: TabularAnalysis,
    ) -> list[dict]:
        """Group by date periods."""
        date_col = analysis.date_columns[0]
        sections = []
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            df_copy = df.copy()
            df_copy["__period__"] = dates.dt.to_period("Q")
            for period, period_df in df_copy.groupby("__period__", sort=True):
                if period_df.empty or pd.isna(period):
                    continue
                start_row = int(period_df.index[0])
                end_row = int(period_df.index[-1])
                sections.append(self._make_section(
                    period_df, data_cols, analysis,
                    title=str(period), start_row=start_row,
                    end_row=end_row, level=1,
                ))
        except Exception:
            return self._build_chunk_sections(df, data_cols, analysis)

        return sections if sections else self._build_chunk_sections(df, data_cols, analysis)

    def _build_chunk_sections(
        self, df: "pd.DataFrame", data_cols: list, analysis: TabularAnalysis,
    ) -> list[dict]:
        """Fallback: row chunks with stats."""
        sections = []
        for chunk_start in range(0, len(df), self.rows_per_page):
            chunk_end = min(chunk_start + self.rows_per_page - 1, len(df) - 1)
            chunk_df = df.iloc[chunk_start:chunk_end + 1]
            sections.append(self._make_section(
                chunk_df, data_cols, analysis,
                title=f"Rows {chunk_start + 1}-{chunk_end + 1}",
                start_row=chunk_start, end_row=chunk_end, level=1,
            ))
        return sections

    def _make_section(
        self, section_df: "pd.DataFrame", data_cols: list,
        analysis: TabularAnalysis, title: str,
        start_row: int, end_row: int, level: int,
    ) -> dict:
        """Create a section dict with statistical summary and keywords."""
        row_count = len(section_df)

        # Compute numeric stats for this section
        stat_parts = [f"{row_count} rows."]
        keywords = []

        for col_name in analysis.numeric_columns:
            if col_name not in section_df.columns:
                continue
            series = section_df[col_name].dropna()
            if len(series) == 0:
                continue
            total = float(series.sum())
            mean = float(series.mean())
            # Format large numbers
            if abs(total) >= 1_000_000:
                stat_parts.append(f"{col_name}: total ${total/1e6:.1f}M, mean ${mean/1e3:.1f}K")
            elif abs(total) >= 1_000:
                stat_parts.append(f"{col_name}: total {total:,.0f}, mean {mean:,.1f}")
            else:
                stat_parts.append(f"{col_name}: total {total:.1f}, mean {mean:.2f}")

        # Top categorical values in this section
        for col_name in analysis.grouping_columns:
            if col_name not in section_df.columns:
                continue
            top = section_df[col_name].value_counts().head(3)
            if len(top) > 0:
                vals = [str(v) for v in top.index]
                keywords.extend(vals[:3])

        # Date range
        for col_name in analysis.date_columns:
            if col_name not in section_df.columns:
                continue
            try:
                dates = pd.to_datetime(section_df[col_name], errors="coerce").dropna()
                if len(dates) > 0:
                    stat_parts.append(f"Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
            except Exception:
                pass

        # Null info
        null_count = int(section_df[data_cols].isna().sum().sum()) if data_cols else 0
        if null_count > 0:
            stat_parts.append(f"{null_count} null values")

        # Keywords: column names + categorical values
        keywords.extend(data_cols[:5])
        keywords = list(dict.fromkeys(keywords))[:8]  # Dedupe, limit to 8

        summary = " ".join(stat_parts)

        return {
            "title": title,
            "summary": summary,
            "keywords": keywords,
            "start_page": start_row // max(self.rows_per_page, 1),
            "end_page": end_row // max(self.rows_per_page, 1),
            "start_row": start_row,
            "end_row": end_row,
            "level": level,
            "row_count": row_count,
        }

    # =========================================================================
    # Static helper: format numeric value for display
    # =========================================================================

    @staticmethod
    def _fmt_num(val: float) -> str:
        if abs(val) >= 1_000_000:
            return f"{val/1e6:.1f}M"
        if abs(val) >= 1_000:
            return f"{val/1e3:.1f}K"
        return f"{val:.1f}"


class LoadedTabular:
    """Loaded tabular file — implements same interface as LoadedPDF."""

    def __init__(
        self, df: "pd.DataFrame", filename: str,
        rows_per_page: int, analysis: TabularAnalysis,
        sections: list[dict], sheet_boundaries: list,
    ):
        self.df = df
        self.filename = filename
        self.rows_per_page = rows_per_page
        self.analysis = analysis
        self.sections = sections
        self.sheet_boundaries = sheet_boundaries
        self._closed = False

        # Data columns (exclude internal __sheet__)
        self._data_cols = [c for c in df.columns if c != "__sheet__"]
        # Limit display columns
        self._display_cols = self._data_cols[:MAX_DISPLAY_COLUMNS]

    @property
    def total_pages(self) -> int:
        return max(1, math.ceil(len(self.df) / self.rows_per_page))

    def get_page_text(self, page_num: int) -> str:
        """Get formatted text for a page (chunk of rows)."""
        self._check_closed()
        self._check_page_range(page_num)

        start_row = page_num * self.rows_per_page
        end_row = min(start_row + self.rows_per_page, len(self.df))
        chunk = self.df.iloc[start_row:end_row][self._data_cols]

        # Header with context
        parts = [f"--- Page {page_num + 1} (Rows {start_row + 1}-{end_row}) ---"]

        # Sheet info if multi-sheet
        if len(self.sheet_boundaries) > 1:
            for sheet_name, s_start, s_end in self.sheet_boundaries:
                if s_start <= start_row <= s_end:
                    parts.append(f"Sheet: {sheet_name}")
                    break

        # Column stats for this chunk
        for col in self.analysis.numeric_columns:
            if col in chunk.columns:
                series = chunk[col].dropna()
                if len(series) > 0:
                    parts.append(
                        f"  {col}: sum={TabularProcessor._fmt_num(series.sum())}, "
                        f"mean={TabularProcessor._fmt_num(series.mean())}, "
                        f"min={TabularProcessor._fmt_num(series.min())}, "
                        f"max={TabularProcessor._fmt_num(series.max())}"
                    )

        # Table data (truncate wide columns)
        display = chunk[self._display_cols].copy()
        for col in display.columns:
            if display[col].dtype == object:
                display[col] = display[col].astype(str).str[:MAX_COL_WIDTH]

        parts.append("")
        parts.append(display.to_string(index=False))

        return "\n".join(parts)

    def get_page_image(self, page_num: int) -> bytes:
        """Render a page as a table image (PNG)."""
        self._check_closed()
        self._check_page_range(page_num)

        if plt is None:
            raise RuntimeError(
                "matplotlib is required for table rendering. "
                "Install with: pip install matplotlib"
            )

        start_row = page_num * self.rows_per_page
        end_row = min(start_row + self.rows_per_page, len(self.df))
        chunk = self.df.iloc[start_row:end_row][self._display_cols]

        # Limit rows for rendering (matplotlib tables get slow with many rows)
        max_render_rows = 30
        if len(chunk) > max_render_rows:
            chunk = chunk.head(max_render_rows)

        # Truncate cell values
        display_data = []
        for _, row in chunk.iterrows():
            display_data.append([str(v)[:30] for v in row])

        col_labels = [str(c)[:20] for c in chunk.columns]

        fig_height = max(2, 0.35 * len(display_data) + 1.5)
        fig_width = max(8, 0.8 * len(col_labels))
        fig, ax = plt.subplots(figsize=(min(fig_width, 20), min(fig_height, 15)))
        ax.axis("off")

        title = f"{self.filename} — Page {page_num + 1} (Rows {start_row + 1}-{end_row})"
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

            # Style header
            for j in range(len(col_labels)):
                table[0, j].set_facecolor("#4472C4")
                table[0, j].set_text_props(color="white", fontweight="bold")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

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
            except Exception:
                continue
        return images

    def get_pages_text_batch(self, start: int, end: int) -> str:
        self._check_closed()
        start = max(0, start)
        end = min(end, self.total_pages - 1)
        parts = []
        for i in range(start, end + 1):
            parts.append(self.get_page_text(i))
        return "\n\n".join(parts)

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
