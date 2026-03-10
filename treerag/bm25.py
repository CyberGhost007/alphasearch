"""
BM25 Inverted Index for tabular data search.

Provides fast keyword-based scoring of tree nodes against a query.
Used as the "policy network" prior in PUCT-guided MCTS for CSV/Excel files.

Features:
- Custom tokenizer that preserves emails, currency, phone numbers, dates as atomic tokens
- Standard BM25 (Okapi BM25) scoring with configurable k1 and b parameters
- Serializable to/from JSON for persistence alongside tree indices
- Score propagation: internal nodes scored from their own summary text
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Custom Tokenizer — preserves structured tokens common in tabular data
# ---------------------------------------------------------------------------

# Patterns for atomic tokens (order matters: matched first → preserved whole)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"https?://[^\s,)]+")
_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[\s\-]?)?"               # optional country code
    r"(?:\(?\d{1,4}\)?[\s\-]?)?"            # optional area code
    r"\d{2,4}[\s\-]\d{2,4}(?:[\s\-]\d{2,4})?"  # number groups
)
_CURRENCY_RE = re.compile(
    r"[\$\u00a3\u20ac\u00a5]"               # currency symbol ($£€¥)
    r"\s?"
    r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?"       # amount with optional commas/decimals
    r"(?:[KMBkmb])?"                         # optional suffix
)
_PERCENTAGE_RE = re.compile(r"-?\d+(?:\.\d+)?%")
_DATE_ISO_RE = re.compile(r"\d{4}[\-/]\d{1,2}[\-/]\d{1,2}")
_DATE_MDY_RE = re.compile(r"\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}")
_NUMBER_WITH_COMMAS_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_DECIMAL_RE = re.compile(r"-?\d+\.\d+")

# Order: longer/more-specific patterns first
_ATOMIC_PATTERNS = [
    _EMAIL_RE,
    _URL_RE,
    _CURRENCY_RE,
    _PERCENTAGE_RE,
    _DATE_ISO_RE,
    _DATE_MDY_RE,
    _NUMBER_WITH_COMMAS_RE,
    _PHONE_RE,
    _DECIMAL_RE,
]

# Word splitting: letters, digits, underscores kept together; everything else splits
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.

    1. Extract atomic tokens (emails, currency, dates, phone numbers) — preserved whole.
    2. Split remaining text into word tokens.
    3. Lowercase everything.
    4. Return combined list (atomic + word tokens).

    Example:
        "John john@acme.com earned $125,000 in Q3 2024"
        → ["john@acme.com", "$125,000", "john", "john", "acme", "com",
           "earned", "125", "000", "in", "q3", "2024"]

    The atomic tokens ensure that searching for "john@acme.com" matches exactly,
    while the split tokens ensure partial searches like "acme" also match.
    """
    if not text:
        return []

    text_lower = text.lower()
    atomic_tokens = []
    # Track character positions consumed by atomic patterns
    consumed = set()

    for pattern in _ATOMIC_PATTERNS:
        for match in pattern.finditer(text_lower):
            start, end = match.start(), match.end()
            # Don't double-match overlapping patterns
            if any(pos in consumed for pos in range(start, end)):
                continue
            token = match.group().strip()
            if token:
                atomic_tokens.append(token)
                consumed.update(range(start, end))

    # Split remaining text into word tokens
    word_tokens = []
    for match in _WORD_RE.finditer(text_lower):
        start, end = match.start(), match.end()
        # Skip if fully consumed by an atomic token
        if all(pos in consumed for pos in range(start, end)):
            continue
        word = match.group()
        if len(word) >= 1:  # Keep even single-char tokens (IDs, codes)
            word_tokens.append(word)

    return atomic_tokens + word_tokens


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Okapi BM25 inverted index.

    Build from tree nodes, score queries against all indexed nodes.
    Each node is indexed by its text content (summaries for internal nodes,
    raw markdown table for leaf nodes).

    Parameters:
        k1: Term frequency saturation parameter. Higher = more weight on
            repeated terms. Default 1.5.
        b: Document length normalization. 0 = no normalization, 1 = full.
            Default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Inverted index: term → {node_id: term_frequency}
        self.inverted_index: dict[str, dict[str, int]] = {}

        # Document metadata
        self.doc_lengths: dict[str, int] = {}       # node_id → total tokens
        self.doc_node_ids: list[str] = []            # ordered list of node IDs
        self.total_docs: int = 0
        self.avg_doc_length: float = 0.0

        # Forward index for debugging: node_id → token list
        self._node_tokens: dict[str, list[str]] = {}

    def build(self, node_texts: dict[str, str]) -> "BM25Index":
        """
        Build the index from a mapping of node_id → text content.

        Args:
            node_texts: {node_id: text_content} for every node in the tree.
                Leaf nodes should have raw markdown table text.
                Internal nodes should have summary + keywords text.

        Returns:
            self (for chaining)
        """
        self.inverted_index.clear()
        self.doc_lengths.clear()
        self.doc_node_ids.clear()
        self._node_tokens.clear()

        total_length = 0

        for node_id, text in node_texts.items():
            tokens = tokenize(text)
            self.doc_node_ids.append(node_id)
            self.doc_lengths[node_id] = len(tokens)
            self._node_tokens[node_id] = tokens
            total_length += len(tokens)

            # Count term frequencies for this document
            tf = Counter(tokens)
            for term, count in tf.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][node_id] = count

        self.total_docs = len(node_texts)
        self.avg_doc_length = total_length / max(self.total_docs, 1)

        return self

    def score(self, query: str) -> dict[str, float]:
        """
        Score all indexed nodes against a query.

        Returns:
            {node_id: bm25_score} for all nodes. Unmatched nodes get 0.0.
        """
        if self.total_docs == 0:
            return {}

        query_tokens = tokenize(query)
        if not query_tokens:
            return {nid: 0.0 for nid in self.doc_node_ids}

        scores: dict[str, float] = {nid: 0.0 for nid in self.doc_node_ids}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            posting = self.inverted_index[term]
            doc_freq = len(posting)

            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            # The +1 inside log prevents negative IDF for very common terms
            idf = math.log(
                (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )

            for node_id, tf in posting.items():
                dl = self.doc_lengths[node_id]
                # BM25 term score: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * dl / max(self.avg_doc_length, 1.0)
                )
                scores[node_id] += idf * numerator / denominator

        return scores

    def score_normalized(self, query: str) -> dict[str, float]:
        """
        Score all nodes and normalize to [0, 1] range.

        Uses min-max normalization. If all scores are zero, returns zeros.
        """
        raw_scores = self.score(query)
        if not raw_scores:
            return {}

        max_score = max(raw_scores.values())
        if max_score <= 0.0:
            return {nid: 0.0 for nid in raw_scores}

        return {nid: s / max_score for nid, s in raw_scores.items()}

    def top_k(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Return top-K (node_id, score) pairs, sorted by score descending.
        """
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------
    # Tier detection — decides which MCTS strategy to use
    # ------------------------------------------------------------------

    def detect_tier(
        self,
        query: str,
        tier1_z_threshold: float = 2.0,
        tier3_max_score: float = 0.1,
    ) -> tuple[int, dict[str, float]]:
        """
        Score the query and determine the execution tier.

        Returns:
            (tier, normalized_scores) where tier is 1, 2, or 3.

        Tier 1: Clear winner — top score is > tier1_z_threshold standard
                deviations above the mean. Direct deep read, zero MCTS.
        Tier 2: Spread scores — multiple candidates. PUCT-guided MCTS.
        Tier 3: Flat/zero scores — no keyword signal. Full UCB1 MCTS.
        """
        raw_scores = self.score(query)
        if not raw_scores:
            return 3, {}

        values = list(raw_scores.values())
        max_score = max(values)
        normalized = self.score_normalized(query)

        # Tier 3: No meaningful BM25 signal
        if max_score <= tier3_max_score:
            return 3, normalized

        # Compute z-score of the top result
        n = len(values)
        mean = sum(values) / max(n, 1)
        variance = sum((v - mean) ** 2 for v in values) / max(n, 1)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std > 0:
            z_top = (max_score - mean) / std
        else:
            # All scores identical (shouldn't happen with real data)
            z_top = 0.0

        # Tier 1: Dominant winner
        if z_top >= tier1_z_threshold:
            return 1, normalized

        # Tier 2: Spread across candidates
        return 2, normalized

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        """Save the BM25 index to a JSON file."""
        path = Path(path)
        data = {
            "k1": self.k1,
            "b": self.b,
            "total_docs": self.total_docs,
            "avg_doc_length": self.avg_doc_length,
            "doc_lengths": self.doc_lengths,
            "doc_node_ids": self.doc_node_ids,
            "inverted_index": self.inverted_index,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Load a BM25 index from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")

        data = json.loads(path.read_text())
        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        index.total_docs = data.get("total_docs", 0)
        index.avg_doc_length = data.get("avg_doc_length", 0.0)
        index.doc_lengths = data.get("doc_lengths", {})
        index.doc_node_ids = data.get("doc_node_ids", [])
        index.inverted_index = data.get("inverted_index", {})
        return index

    @property
    def is_empty(self) -> bool:
        return self.total_docs == 0

    def __repr__(self) -> str:
        return (
            f"BM25Index(docs={self.total_docs}, "
            f"terms={len(self.inverted_index)}, "
            f"avg_len={self.avg_doc_length:.0f})"
        )
