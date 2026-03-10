"""Tests for tabular processing and indexing (CSV/Excel support)."""

import tempfile
import os
from pathlib import Path

import pandas as pd
import pytest

from treerag.tabular_processor import (
    TabularProcessor, LoadedTabular, TabularAnalysis,
    ColumnInfo, SUPPORTED_TABULAR_EXTENSIONS,
)
from treerag.exceptions import (
    InvalidTabularError, EmptyTabularError, FileTooLargeError,
)


# =========================================================================
# Fixtures — create temp CSV/Excel files for testing
# =========================================================================

@pytest.fixture
def sample_csv(tmp_path):
    """Simple CSV with numeric, categorical, and date columns."""
    df = pd.DataFrame({
        "Region": ["North", "North", "South", "South", "East", "East", "West", "West"] * 10,
        "Product": ["Widget", "Gadget"] * 40,
        "Revenue": [100.0 + i * 10 for i in range(80)],
        "Quantity": [5 + i % 20 for i in range(80)],
        "Date": pd.date_range("2024-01-01", periods=80, freq="D"),
    })
    path = tmp_path / "sales.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_xlsx(tmp_path):
    """Excel file with two sheets."""
    path = tmp_path / "report.xlsx"
    with pd.ExcelWriter(path) as writer:
        df1 = pd.DataFrame({
            "Name": ["Alice", "Bob", "Charlie"],
            "Score": [95, 87, 72],
        })
        df1.to_excel(writer, sheet_name="Scores", index=False)
        df2 = pd.DataFrame({
            "Item": ["Laptop", "Mouse", "Keyboard"],
            "Price": [999.99, 29.99, 79.99],
        })
        df2.to_excel(writer, sheet_name="Inventory", index=False)
    return path


@pytest.fixture
def empty_csv(tmp_path):
    """CSV with headers only, no data rows."""
    path = tmp_path / "empty.csv"
    path.write_text("col_a,col_b,col_c\n")
    return path


@pytest.fixture
def numeric_only_csv(tmp_path):
    """CSV with only numeric columns (no grouping column)."""
    df = pd.DataFrame({
        "temperature": [20.1, 21.3, 19.8, 22.5, 23.1, 18.9, 20.0, 21.7, 22.2, 19.5],
        "humidity": [45, 50, 55, 42, 38, 60, 52, 48, 44, 58],
        "pressure": [1013.2, 1012.8, 1014.0, 1011.5, 1013.7, 1012.0, 1013.5, 1012.3, 1014.1, 1011.8],
    })
    path = tmp_path / "weather.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def processor():
    return TabularProcessor(rows_per_page=20)


# =========================================================================
# TabularProcessor.validate() tests
# =========================================================================

class TestTabularProcessorValidate:
    def test_validate_csv(self, processor, sample_csv):
        result = processor.validate(sample_csv)
        assert result["filename"] == "sales.csv"
        assert result["row_count"] == 80
        assert result["column_count"] == 5

    def test_validate_xlsx(self, processor, sample_xlsx):
        result = processor.validate(sample_xlsx)
        assert result["filename"] == "report.xlsx"
        assert result["column_count"] == 2  # First sheet

    def test_validate_empty_csv(self, processor, empty_csv):
        with pytest.raises(EmptyTabularError):
            processor.validate(empty_csv)

    def test_validate_nonexistent(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.validate("/tmp/nonexistent.csv")

    def test_validate_bad_extension(self, processor, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("hello")
        with pytest.raises(InvalidTabularError, match="Unsupported"):
            processor.validate(bad)

    def test_validate_zero_byte_file(self, processor, tmp_path):
        zero = tmp_path / "zero.csv"
        zero.write_text("")
        with pytest.raises(InvalidTabularError, match="empty"):
            processor.validate(zero)


# =========================================================================
# TabularAnalysis tests
# =========================================================================

class TestTabularAnalysis:
    def test_analyze_detects_grouping(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        analysis = loaded.analysis
        assert "Region" in analysis.grouping_columns
        loaded.close()

    def test_analyze_detects_dates(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        analysis = loaded.analysis
        assert len(analysis.date_columns) >= 1
        loaded.close()

    def test_analyze_numeric_stats(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        analysis = loaded.analysis
        assert "Revenue" in analysis.numeric_columns
        assert "Quantity" in analysis.numeric_columns
        # Check Revenue column stats
        revenue_col = next(c for c in analysis.columns if c.name == "Revenue")
        assert revenue_col.is_numeric
        assert revenue_col.stats["min"] == 100.0
        assert revenue_col.stats["max"] == 890.0
        loaded.close()

    def test_analyze_sections_generated(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        # Should have sections (grouped by Region)
        assert len(loaded.sections) > 0
        loaded.close()

    def test_analyze_column_info(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        analysis = loaded.analysis
        assert analysis.total_rows == 80
        assert analysis.total_columns == 5
        assert len(analysis.columns) == 5
        loaded.close()

    def test_analyze_no_grouping(self, processor, numeric_only_csv):
        loaded = processor.load(numeric_only_csv)
        analysis = loaded.analysis
        # No categorical columns with < 20 unique values that aren't numeric
        assert len(analysis.numeric_columns) == 3
        loaded.close()


# =========================================================================
# LoadedTabular tests
# =========================================================================

class TestLoadedTabular:
    def test_total_pages(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        # 80 rows / 20 rows_per_page = 4 pages
        assert loaded.total_pages == 4
        loaded.close()

    def test_get_page_text(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        text = loaded.get_page_text(0)
        assert "Page 1" in text
        assert "Rows 1-20" in text
        # Should contain column data
        assert "Revenue" in text or "Region" in text
        loaded.close()

    def test_get_page_text_last_page(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        text = loaded.get_page_text(loaded.total_pages - 1)
        assert "Page" in text
        loaded.close()

    def test_get_page_text_out_of_range(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        with pytest.raises(IndexError):
            loaded.get_page_text(100)
        loaded.close()

    def test_get_page_image(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        img = loaded.get_page_image(0)
        assert isinstance(img, bytes)
        assert len(img) > 0
        # PNG magic bytes
        assert img[:4] == b'\x89PNG'
        loaded.close()

    def test_get_page_images_batch(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        images = loaded.get_page_images_batch(0, 1)
        assert len(images) == 2
        for img in images:
            assert isinstance(img, bytes)
            assert len(img) > 0
        loaded.close()

    def test_get_pages_text_batch(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        text = loaded.get_pages_text_batch(0, 1)
        assert "Page 1" in text
        assert "Page 2" in text
        loaded.close()

    def test_close_prevents_access(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        loaded.close()
        with pytest.raises(RuntimeError, match="closed"):
            loaded.get_page_text(0)

    def test_context_manager(self, processor, sample_csv):
        with processor.load(sample_csv) as loaded:
            assert loaded.total_pages > 0
            text = loaded.get_page_text(0)
            assert len(text) > 0

    def test_excel_multi_sheet(self, processor, sample_xlsx):
        loaded = processor.load(sample_xlsx)
        analysis = loaded.analysis
        assert len(analysis.sheet_names) == 2
        assert "Scores" in analysis.sheet_names
        assert "Inventory" in analysis.sheet_names
        # Sections should include sheet-based sections
        section_titles = [s["title"] for s in loaded.sections]
        assert any("Scores" in t for t in section_titles)
        assert any("Inventory" in t for t in section_titles)
        loaded.close()


# =========================================================================
# Smart sectioning tests
# =========================================================================

class TestSmartSectioning:
    def test_grouping_sections(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        # Should create sections by Region (North, South, East, West)
        level1_sections = [s for s in loaded.sections if s.get("level") == 1]
        titles = [s["title"] for s in level1_sections]
        assert len(level1_sections) == 4
        assert any("North" in t for t in titles)
        assert any("South" in t for t in titles)
        loaded.close()

    def test_section_summaries_have_stats(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        for section in loaded.sections:
            summary = section.get("summary", "")
            # Every section should have row count in summary
            assert "rows" in summary.lower() or "row" in summary.lower()
        loaded.close()

    def test_section_keywords(self, processor, sample_csv):
        loaded = processor.load(sample_csv)
        for section in loaded.sections:
            keywords = section.get("keywords", [])
            assert len(keywords) > 0
        loaded.close()

    def test_chunk_fallback(self, processor, numeric_only_csv):
        """No grouping or date columns → should fall back to row chunks."""
        loaded = processor.load(numeric_only_csv)
        sections = loaded.sections
        assert len(sections) >= 1
        # Should have row-based titles
        assert any("Rows" in s["title"] for s in sections)
        loaded.close()

    def test_multi_sheet_sectioning(self, processor, sample_xlsx):
        loaded = processor.load(sample_xlsx)
        level1 = [s for s in loaded.sections if s.get("level") == 1]
        # Each sheet should be a level 1 section
        assert len(level1) == 2
        loaded.close()


# =========================================================================
# SUPPORTED_TABULAR_EXTENSIONS constant
# =========================================================================

class TestSupportedExtensions:
    def test_csv_supported(self):
        assert ".csv" in SUPPORTED_TABULAR_EXTENSIONS

    def test_xlsx_supported(self):
        assert ".xlsx" in SUPPORTED_TABULAR_EXTENSIONS

    def test_xls_supported(self):
        assert ".xls" in SUPPORTED_TABULAR_EXTENSIONS
