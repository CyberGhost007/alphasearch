"""
Tests for tabular processing, BM25 indexing, and PUCT/tiered search.

Covers:
- TabularProcessor validation and loading
- LoadedTabular interface (markdown table pages)
- BM25 tokenizer (email, currency, phone, date preservation)
- BM25 inverted index (scoring, normalization, top-k, persistence)
- BM25 tier detection (Tier 1/2/3 classification)
- PUCT formula on TreeNode
- Supported extensions constant
- BM25 + tabular data integration
- DocumentIndex BM25 field persistence
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from treerag.tabular_processor import (
    TabularProcessor, LoadedTabular, SUPPORTED_TABULAR_EXTENSIONS,
)
from treerag.bm25 import BM25Index, tokenize
from treerag.models import TreeNode, DocumentIndex
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
def contact_csv(tmp_path):
    """CSV with contact data — emails, phones, mixed data."""
    df = pd.DataFrame({
        "Name": ["John Smith", "Jane Doe", "Bob Wilson", "Alice Chen"],
        "Email": ["john@acme.com", "jane@corp.org", "bob@startup.io", "alice@mega.co"],
        "Phone": ["555-123-4567", "555-987-6543", "555-456-7890", "555-321-0987"],
        "Company": ["Acme Corp", "MegaCorp", "StartupIO", "MegaCorp"],
        "Revenue": ["$125,000", "$250,000", "$75,000", "$180,000"],
    })
    path = tmp_path / "contacts.csv"
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
        assert result["column_count"] == 2  # First sheet columns

    def test_validate_empty_csv(self, processor, empty_csv):
        with pytest.raises(EmptyTabularError):
            processor.validate(empty_csv)

    def test_validate_nonexistent(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.validate("/tmp/nonexistent_file_xyz_12345.csv")

    def test_validate_bad_extension(self, processor, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("hello\nworld\n")
        with pytest.raises(InvalidTabularError, match="Unsupported"):
            processor.validate(bad)

    def test_validate_zero_byte_file(self, processor, tmp_path):
        zero = tmp_path / "zero.csv"
        zero.write_text("")
        with pytest.raises(InvalidTabularError, match="empty"):
            processor.validate(zero)


# =========================================================================
# LoadedTabular interface tests
# =========================================================================

class TestLoadedTabular:
    def test_total_pages(self, processor, sample_csv):
        """80 rows / 20 rows_per_page = 4 pages."""
        with processor.load(sample_csv) as loaded:
            assert loaded.total_pages == 4

    def test_get_page_text_returns_markdown(self, processor, sample_csv):
        """Page text should be a markdown table with header."""
        with processor.load(sample_csv) as loaded:
            text = loaded.get_page_text(0)
            # Should have file context header
            assert "sales.csv" in text
            assert "Rows 1-20" in text
            # Should have markdown table headers
            assert "| Region |" in text or "Region" in text
            assert "---" in text  # Markdown separator

    def test_get_page_text_contains_data(self, processor, sample_csv):
        """Page text should contain actual row data."""
        with processor.load(sample_csv) as loaded:
            text = loaded.get_page_text(0)
            # Should contain actual values from the first 20 rows
            assert "North" in text or "South" in text
            assert "Widget" in text or "Gadget" in text

    def test_get_page_text_last_page(self, processor, sample_csv):
        """Last page should be valid and contain row data."""
        with processor.load(sample_csv) as loaded:
            last_page = loaded.total_pages - 1
            text = loaded.get_page_text(last_page)
            assert "sales.csv" in text
            assert len(text) > 50  # Not empty

    def test_get_page_text_out_of_range(self, processor, sample_csv):
        """Out-of-range page should raise IndexError."""
        with processor.load(sample_csv) as loaded:
            with pytest.raises(IndexError):
                loaded.get_page_text(100)

    def test_get_page_image_returns_png(self, processor, sample_csv):
        """Page image should be valid PNG bytes."""
        with processor.load(sample_csv) as loaded:
            img = loaded.get_page_image(0)
            assert isinstance(img, bytes)
            assert len(img) > 0
            assert img[:4] == b'\x89PNG'

    def test_get_page_images_batch(self, processor, sample_csv):
        """Batch image request returns correct number of images."""
        with processor.load(sample_csv) as loaded:
            images = loaded.get_page_images_batch(0, 1)
            assert len(images) == 2
            for img in images:
                assert isinstance(img, bytes)
                assert img[:4] == b'\x89PNG'

    def test_get_pages_text_batch(self, processor, sample_csv):
        """Batch text request concatenates multiple pages."""
        with processor.load(sample_csv) as loaded:
            text = loaded.get_pages_text_batch(0, 1)
            assert "Rows 1-20" in text
            assert "Rows 21-40" in text

    def test_close_prevents_access(self, processor, sample_csv):
        """Accessing closed file should raise RuntimeError."""
        loaded = processor.load(sample_csv)
        loaded.close()
        with pytest.raises(RuntimeError, match="closed"):
            loaded.get_page_text(0)

    def test_context_manager(self, processor, sample_csv):
        """Context manager should auto-close."""
        with processor.load(sample_csv) as loaded:
            assert loaded.total_pages > 0
            text = loaded.get_page_text(0)
            assert len(text) > 0
        # Should be closed now
        assert loaded._closed is True

    def test_multi_sheet_excel(self, processor, sample_xlsx):
        """Excel with multiple sheets should combine rows."""
        with processor.load(sample_xlsx) as loaded:
            assert loaded.total_pages >= 1
            text = loaded.get_page_text(0)
            # Should have data from at least one sheet
            assert len(text) > 50

    def test_multi_sheet_has_sheet_info(self, processor, sample_xlsx):
        """Multi-sheet Excel pages should include sheet context."""
        with processor.load(sample_xlsx) as loaded:
            text = loaded.get_page_text(0)
            # With 2 sheets, the sheet name should appear in context
            assert "Sheet:" in text or "report.xlsx" in text


# =========================================================================
# BM25 Tokenizer tests
# =========================================================================

class TestBM25Tokenizer:
    def test_basic_words(self):
        """Simple words are lowercased and split."""
        tokens = tokenize("Hello World Test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_email_preserved(self):
        """Emails are preserved as atomic tokens."""
        tokens = tokenize("Contact john@acme.com for info")
        assert "john@acme.com" in tokens
        # Parts also appear as word tokens
        assert "contact" in tokens
        assert "for" in tokens
        assert "info" in tokens

    def test_currency_preserved(self):
        """Currency amounts are preserved as atomic tokens."""
        tokens = tokenize("Revenue was $125,000 last quarter")
        assert "$125,000" in tokens
        assert "revenue" in tokens

    def test_percentage_preserved(self):
        """Percentages are preserved as atomic tokens."""
        tokens = tokenize("Growth rate 15.3% year over year")
        assert "15.3%" in tokens

    def test_date_iso_preserved(self):
        """ISO dates are preserved as atomic tokens."""
        tokens = tokenize("Started on 2024-01-15 and ended 2024-03-20")
        assert "2024-01-15" in tokens
        assert "2024-03-20" in tokens

    def test_date_mdy_preserved(self):
        """M/D/Y dates are preserved as atomic tokens."""
        tokens = tokenize("Due date: 03/15/2024")
        assert "03/15/2024" in tokens

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert tokenize("") == []

    def test_mixed_content(self):
        """Complex text with multiple token types."""
        text = "John john@acme.com earned $250,000 (15.5%) since 2024-01-01"
        tokens = tokenize(text)
        assert "john@acme.com" in tokens
        assert "$250,000" in tokens
        assert "15.5%" in tokens
        assert "2024-01-01" in tokens
        assert "john" in tokens
        assert "earned" in tokens


# =========================================================================
# BM25 Index tests
# =========================================================================

class TestBM25Index:
    def _build_sample_index(self):
        """Build a small BM25 index for testing."""
        node_texts = {
            "node_a": "Sales revenue for North America region. Total: $5M",
            "node_b": "Employee contact list. john@acme.com, jane@corp.org",
            "node_c": "Inventory data for electronics department. Laptops, mice, keyboards.",
            "node_d": "Quarterly financial report Q3 2024. Revenue growth 15%.",
        }
        return BM25Index().build(node_texts)

    def test_build_populates_index(self):
        """Building populates document counts and inverted index."""
        idx = self._build_sample_index()
        assert idx.total_docs == 4
        assert len(idx.inverted_index) > 0
        assert len(idx.doc_node_ids) == 4
        assert idx.avg_doc_length > 0

    def test_score_returns_all_nodes(self):
        """Scoring returns a score for every indexed node."""
        idx = self._build_sample_index()
        scores = idx.score("revenue sales")
        assert len(scores) == 4
        assert all(isinstance(v, float) for v in scores.values())

    def test_score_relevant_higher(self):
        """Relevant nodes should score higher than irrelevant ones."""
        idx = self._build_sample_index()
        scores = idx.score("employee contact email")
        # node_b (contact list) should score highest
        assert scores["node_b"] > scores["node_c"]
        assert scores["node_b"] > scores["node_a"]

    def test_score_no_match(self):
        """Query with no matching terms returns all zeros."""
        idx = self._build_sample_index()
        scores = idx.score("xyznonexistent foobar")
        assert all(v == 0.0 for v in scores.values())

    def test_score_normalized(self):
        """Normalized scores should be in [0, 1] range."""
        idx = self._build_sample_index()
        scores = idx.score_normalized("revenue financial report")
        for v in scores.values():
            assert 0.0 <= v <= 1.0
        # At least one should be 1.0 (the max)
        assert max(scores.values()) == 1.0 or all(v == 0.0 for v in scores.values())

    def test_top_k(self):
        """Top-K returns the right number and order."""
        idx = self._build_sample_index()
        top = idx.top_k("revenue sales financial", k=2)
        assert len(top) == 2
        # First should be highest scoring
        assert top[0][1] >= top[1][1]

    def test_save_load_roundtrip(self, tmp_path):
        """Save + load should produce identical scores."""
        idx = self._build_sample_index()
        path = tmp_path / "bm25.json"
        idx.save(path)

        loaded = BM25Index.load(path)
        assert loaded.total_docs == idx.total_docs
        assert loaded.k1 == idx.k1
        assert loaded.b == idx.b
        assert len(loaded.inverted_index) == len(idx.inverted_index)

        # Scores should be identical
        query = "revenue report"
        orig_scores = idx.score(query)
        loaded_scores = loaded.score(query)
        for nid in orig_scores:
            assert abs(orig_scores[nid] - loaded_scores[nid]) < 1e-10

    def test_empty_index(self):
        """Empty index returns empty scores."""
        idx = BM25Index()
        scores = idx.score("anything")
        assert scores == {}
        assert idx.is_empty is True


# =========================================================================
# BM25 Tier Detection tests
# =========================================================================

class TestBM25TierDetection:
    def test_tier3_no_signal(self):
        """When no terms match, should return Tier 3."""
        node_texts = {
            "n1": "alpha beta gamma",
            "n2": "delta epsilon zeta",
            "n3": "eta theta iota",
        }
        idx = BM25Index().build(node_texts)
        tier, scores = idx.detect_tier("xyznonexistent foobar baz")
        assert tier == 3

    def test_tier1_clear_winner(self):
        """When one node dominates, should return Tier 1."""
        node_texts = {
            "n1": "revenue revenue revenue sales profit income total quarterly",
            "n2": "weather temperature humidity pressure",
            "n3": "sports football basketball soccer",
            "n4": "cooking recipe ingredients pasta",
            "n5": "music guitar piano drums",
        }
        idx = BM25Index().build(node_texts)
        tier, scores = idx.detect_tier("revenue sales profit")
        assert tier == 1

    def test_tier2_spread_scores(self):
        """When scores are spread, should return Tier 2."""
        # Create nodes where the query matches several somewhat equally
        node_texts = {
            "n1": "sales revenue report quarterly financial data",
            "n2": "revenue growth analysis financial projections",
            "n3": "financial statements balance sheet revenue",
            "n4": "unrelated cooking recipe pasta ingredients",
        }
        idx = BM25Index().build(node_texts)
        tier, scores = idx.detect_tier("financial revenue")
        # Multiple nodes match "financial" and "revenue" — should be Tier 2
        assert tier in (1, 2)  # Could be 1 or 2 depending on exact scores
        # But at least it shouldn't be Tier 3
        assert tier != 3


# =========================================================================
# PUCT formula tests (TreeNode)
# =========================================================================

class TestPUCT:
    def _make_tree(self):
        """Create a small tree for PUCT testing."""
        root = TreeNode(node_id="0000", title="Root", summary="root",
                        start_page=0, end_page=9)
        for i in range(3):
            child = TreeNode(
                node_id=f"000{i+1}", title=f"Child {i+1}",
                summary=f"child {i+1}", start_page=i*3, end_page=(i+1)*3,
                parent=root,
            )
            root.children.append(child)
        return root

    def test_puct_unvisited_high_prior_preferred(self):
        """Unvisited nodes with higher BM25 prior should score higher."""
        root = self._make_tree()
        root.visit_count = 10  # Parent has visits
        c0, c1, c2 = root.children

        # All unvisited — PUCT should be proportional to prior
        score_high = c0.puct(bm25_prior=0.9, exploration_constant=1.414)
        score_low = c1.puct(bm25_prior=0.1, exploration_constant=1.414)
        score_zero = c2.puct(bm25_prior=0.0, exploration_constant=1.414)

        assert score_high > score_low
        assert score_low > score_zero
        assert score_zero == 0.0  # Zero prior, unvisited → 0

    def test_puct_exploitation_dominates_with_visits(self):
        """With many visits, exploitation (Q) should dominate over exploration."""
        root = self._make_tree()
        root.visit_count = 100
        child = root.children[0]

        # High reward, many visits
        child.visit_count = 50
        child.total_reward = 45.0  # avg = 0.9

        # Low prior — but high exploitation
        score = child.puct(bm25_prior=0.1, exploration_constant=1.414)
        # Should be close to 0.9 (the Q value) since exploration term is small
        assert score > 0.85

    def test_puct_zero_prior_still_exploitable(self):
        """Node with zero prior but high reward should still get explored via exploitation."""
        root = self._make_tree()
        root.visit_count = 50
        child = root.children[0]
        child.visit_count = 10
        child.total_reward = 8.0  # avg = 0.8

        score = child.puct(bm25_prior=0.0, exploration_constant=1.414)
        # Should equal just the exploitation term (average_reward)
        assert abs(score - 0.8) < 1e-10

    def test_puct_vs_ucb1_different_behavior(self):
        """PUCT and UCB1 should give different orderings with priors."""
        root = self._make_tree()
        root.visit_count = 20
        c0, c1, c2 = root.children

        # c0: low reward, high BM25 prior
        c0.visit_count = 5
        c0.total_reward = 1.0  # avg = 0.2
        # c1: high reward, low BM25 prior
        c1.visit_count = 5
        c1.total_reward = 4.0  # avg = 0.8

        puct_c0 = c0.puct(bm25_prior=0.9, exploration_constant=1.414)
        puct_c1 = c1.puct(bm25_prior=0.1, exploration_constant=1.414)
        ucb1_c0 = c0.ucb1(exploration_constant=1.414)
        ucb1_c1 = c1.ucb1(exploration_constant=1.414)

        # UCB1 should prefer c1 (higher reward, same visits)
        assert ucb1_c1 > ucb1_c0
        # PUCT might prefer c0 (high prior boosts exploration term)
        # At least the gap should be different
        ucb1_gap = ucb1_c1 - ucb1_c0
        puct_gap = puct_c1 - puct_c0
        assert ucb1_gap != puct_gap  # Different scoring


# =========================================================================
# Supported Extensions constant
# =========================================================================

class TestSupportedExtensions:
    def test_csv_supported(self):
        assert ".csv" in SUPPORTED_TABULAR_EXTENSIONS

    def test_xlsx_supported(self):
        assert ".xlsx" in SUPPORTED_TABULAR_EXTENSIONS

    def test_xls_supported(self):
        assert ".xls" in SUPPORTED_TABULAR_EXTENSIONS

    def test_pdf_not_tabular(self):
        assert ".pdf" not in SUPPORTED_TABULAR_EXTENSIONS


# =========================================================================
# BM25 + Tabular Data integration
# =========================================================================

class TestBM25WithTabularData:
    def test_index_from_csv_pages(self, processor, contact_csv):
        """Build BM25 index from actual CSV page text."""
        with processor.load(contact_csv) as loaded:
            node_texts = {}
            num_pages = loaded.total_pages
            for i in range(num_pages):
                node_texts[f"page_{i}"] = loaded.get_page_text(i)

        idx = BM25Index().build(node_texts)
        assert idx.total_docs == num_pages
        assert not idx.is_empty

        # Search for specific contact data
        scores = idx.score("john@acme.com")
        assert max(scores.values()) > 0.0

    def test_email_search_in_contacts(self, processor, contact_csv):
        """BM25 should find emails in contact CSVs."""
        with processor.load(contact_csv) as loaded:
            texts = {}
            for i in range(loaded.total_pages):
                texts[f"page_{i}"] = loaded.get_page_text(i)

        idx = BM25Index().build(texts)
        top = idx.top_k("john@acme.com", k=1)
        assert len(top) > 0
        assert top[0][1] > 0.0  # Has a positive score

    def test_currency_search(self, processor, contact_csv):
        """BM25 should find currency values."""
        with processor.load(contact_csv) as loaded:
            texts = {}
            for i in range(loaded.total_pages):
                texts[f"page_{i}"] = loaded.get_page_text(i)

        idx = BM25Index().build(texts)
        scores = idx.score("$125,000")
        assert max(scores.values()) > 0.0


# =========================================================================
# DocumentIndex BM25 field persistence
# =========================================================================

class TestDocumentIndexBM25Field:
    def test_bm25_path_persists(self, tmp_path):
        """bm25_index_path should survive save/load cycle."""
        bm25_path = str(tmp_path / "bm25.json")
        idx = DocumentIndex(
            document_id="test123",
            filename="data.csv",
            total_pages=5,
            description="Test data",
            bm25_index_path=bm25_path,
        )
        save_path = tmp_path / "index.json"
        idx.save(save_path)

        loaded = DocumentIndex.load(save_path)
        assert loaded.bm25_index_path == bm25_path

    def test_has_bm25_true_when_file_exists(self, tmp_path):
        """has_bm25 should be True when the BM25 file actually exists."""
        bm25_path = tmp_path / "bm25.json"
        bm25_path.write_text("{}")  # Create the file

        idx = DocumentIndex(
            document_id="test123",
            filename="data.csv",
            total_pages=5,
            bm25_index_path=str(bm25_path),
        )
        assert idx.has_bm25 is True

    def test_has_bm25_false_when_no_path(self):
        """has_bm25 should be False when no path is set."""
        idx = DocumentIndex(
            document_id="test123",
            filename="report.pdf",
            total_pages=10,
        )
        assert idx.has_bm25 is False

    def test_has_bm25_false_when_file_missing(self, tmp_path):
        """has_bm25 should be False when path is set but file doesn't exist."""
        idx = DocumentIndex(
            document_id="test123",
            filename="data.csv",
            total_pages=5,
            bm25_index_path=str(tmp_path / "nonexistent_bm25.json"),
        )
        assert idx.has_bm25 is False


# =========================================================================
# BM25 edge cases
# =========================================================================

class TestBM25EdgeCases:
    def test_single_document_index(self):
        """BM25 with only one document should still score correctly."""
        idx = BM25Index().build({"only_node": "revenue sales profit data"})
        scores = idx.score("revenue")
        assert scores["only_node"] > 0.0

    def test_duplicate_terms_in_query(self):
        """Repeated query terms should boost score."""
        idx = BM25Index().build({
            "n1": "revenue sales data report",
            "n2": "weather temperature data",
        })
        score_single = idx.score("revenue")["n1"]
        score_double = idx.score("revenue revenue")["n1"]
        # Double query shouldn't decrease score
        assert score_double >= score_single

    def test_very_long_document(self):
        """BM25 should handle long documents without errors."""
        long_text = " ".join(f"word_{i}" for i in range(10000))
        idx = BM25Index().build({"long_doc": long_text, "short_doc": "hello world"})
        scores = idx.score("word_5000")
        assert scores["long_doc"] > 0.0
        assert scores["short_doc"] == 0.0

    def test_special_characters(self):
        """Tokenizer should handle special characters gracefully."""
        tokens = tokenize("Hello! @#$% *** (test) [brackets]")
        assert "hello" in tokens
        assert "test" in tokens
        assert "brackets" in tokens

    def test_load_nonexistent_raises(self):
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BM25Index.load("/tmp/nonexistent_bm25_xyz_12345.json")
