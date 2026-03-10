# Tabular Data: Design & Implementation

## The Problem

AlphaSearch indexes PDFs by reading pages, building tree summaries, and searching via MCTS. Adding CSV/Excel support means handling fundamentally different data — rows and columns instead of paragraphs and pages.

Every dataset is different — sales data, contact lists, task boards, server logs, survey responses. Hardcoding summary strategies per type doesn't scale.

## Approaches We Considered (and Why We Rejected Them)

### 1. Heuristic Analysis (pandas detects everything)
Pandas infers column types by cardinality and dtype. Sections are built by groupby on detected categories. Summaries are template-driven.

**Rejected:** Pandas can't tell a zip code from a revenue figure. Summaries are template-driven (`f"Total revenue: ${sum:,.0f}"`) and produce empty results for non-analytical data like contacts.

### 2. Full LLM Analysis (LLM classifies, pandas executes)
LLM classifies each column semantically, picks a grouping strategy. Pandas executes the groupby. Code writes stat-heavy summaries.

**Rejected:** Still hardcoded summary templates. A contact list grouped by company still produces "15 contacts at Acme Corp" — thin, unsearchable. Adding per-dataset-type summary logic doesn't scale.

### 3. Narrative Conversion (LLM converts rows to prose)
Convert tabular rows into natural language paragraphs. Feed prose into the existing PDF indexer.

**Rejected:** Adds a round of LLM calls (one per chunk) that duplicates what the indexer already does — reading text and writing summaries. Extra cost for something the indexer handles natively. Also risks data fidelity issues (LLM may round numbers, scramble emails).

### 4. Vision-based (render tables as images)
Render rows as table images, feed into existing vision pipeline.

**Rejected:** Round-trip for no reason — text to image back to text. Loses precision. Vision tokens are expensive (~1000 tokens for an image vs ~400-500 for the same data as text).

## Chosen Approach: Markdown Tables + Existing Indexer + BM25

### Core Insight

The existing PDF indexer already reads text pages and writes LLM-generated summaries. Markdown tables ARE text. The indexer reads them, understands the data, and writes appropriate summaries automatically — no special tabular logic needed.

For search, we add BM25 (a classical text search algorithm) as a fast scoring layer. Tabular data is full of exact searchable terms (names, emails, amounts, dates) where BM25 is instant and precise. This mirrors AlphaGo's architecture: BM25 is the "policy network" (fast, guides exploration), LLM is the "value network" (slow, confirms).

### Architecture

**PDF path (unchanged):**
```
PDF → pages → Indexer (LLM summaries) → MCTS (UCB1) → LLM evaluation
```

**Tabular path (new):**
```
CSV/Excel → pandas → markdown table pages → same Indexer (LLM summaries)
                                           → BM25 index on raw text
                                           → MCTS (PUCT with BM25 prior) → LLM evaluation
```

### Indexing Flow

1. **Pandas reads the file** — I/O only, chunks into ~40-50 row pages as markdown tables
2. **Existing PDF Indexer** — LLM reads each markdown table page, writes summaries and keywords, builds tree. Same code path as PDFs, no changes needed.
3. **BM25 index** — built on the raw markdown table text. Fast, no LLM, stores inverted index of terms alongside the tree index.

### Search Flow (PUCT — the AlphaGo formula)

Standard UCB1 (current, used for PDFs):
```
UCB1 = Q/N + C * sqrt(ln(parent_N) / N)
```

PUCT with BM25 prior (new, used for tabular):
```
PUCT = Q/N + C * P * sqrt(parent_N) / (1 + N)
```

Where `P` = normalized BM25 score for that node against the query.

**How it works during search:**
1. Query comes in ("john@acme.com" or "Q3 revenue North America")
2. BM25 scores every leaf node instantly → these become the prior `P` for each node
3. MCTS selection uses PUCT — explores high-BM25 branches first
4. LLM only evaluates the most promising nodes (fewer API calls)
5. Backpropagation updates scores as usual

### Why BM25 + MCTS for Tabular Data

| Query type | BM25 alone | LLM alone | BM25 + MCTS (PUCT) |
|---|---|---|---|
| Exact lookup ("john@acme.com") | Instant, perfect | Slow, overkill | BM25 handles it, zero LLM calls |
| Keyword match ("Q3 revenue North America") | Good | Good but slow | BM25 guides, LLM confirms |
| Semantic ("who leads engineering") | Misses | Gets it, slow | BM25 narrows, LLM evaluates fewer nodes |
| Fuzzy ("Jon Smith" for "John Smith") | Partial | Gets it | BM25 gets close, LLM resolves |

BM25 is especially effective for tabular data because every cell value is a searchable term — names, emails, numbers, dates, status codes. A 10,000-row CSV produces a rich inverted index.

PDFs stay on pure UCB1 because prose content benefits more from semantic LLM evaluation than keyword matching.

## What This Means for Code

### New Code
- `TabularProcessor` — simplified to just file I/O + markdown table chunking
- `BM25Index` — inverted index builder + scorer (small, well-understood algorithm)
- PUCT selection mode in MCTS engine (alongside existing UCB1)

### Eliminated Code
- `TabularIndexer` — gone (reuses existing PDF Indexer)
- `SYSTEM_PROMPT_TABULAR_ANALYZE` — gone
- Column classification logic — gone
- Grouping strategy logic — gone
- Stat summary templates — gone
- Dataset type detection — gone

### Unchanged Code
- PDF Indexer — handles both PDFs and markdown table pages
- MCTS engine — UCB1 for PDFs, PUCT for tabular (flag-based)
- Pipeline, folder management — just routing by file extension

## Summary

| Aspect | Previous approach (LLM Analysis) | Final approach (Indexer + BM25) |
|--------|----------------------------------|--------------------------------|
| Tabular-specific indexer | Yes (~600 lines) | No (reuses PDF indexer) |
| Special LLM prompts | 2 tabular-specific | None (indexer handles it) |
| Text search | None | BM25 with PUCT |
| Handles any data type | Needs per-type logic | Indexer adapts automatically |
| Search: exact lookups | LLM only (slow, expensive) | BM25 instant, zero LLM calls |
| Search: semantic queries | LLM via MCTS | BM25 guides + LLM confirms |
| LLM calls during search | Every MCTS iteration | Only promising nodes |
| Code complexity | High | Low |
| AlphaGo parallel | Value network only | Policy (BM25) + Value (LLM) + MCTS |

---

## Implementation Details

This section documents what was actually built, key decisions made during implementation, and how the pieces fit together.

### Files Created

**`treerag/tabular_processor.py`** (~420 lines)

`TabularProcessor` validates and loads CSV/Excel files. `LoadedTabular` mirrors the `LoadedPDF` interface exactly:

- `total_pages` — computed from `ceil(rows / rows_per_page)` (default 50 rows/page)
- `get_page_text(page_num)` — returns markdown table with header context (filename, sheet, row range)
- `get_page_image(page_num)` — matplotlib table rendering (for vision if needed)
- `get_page_images_batch()` / `get_pages_text_batch()` — batch accessors
- Context manager + close-state detection

Validation catches: missing file, wrong extension, empty file, unparseable content, no columns, no data rows, row count exceeding 100K limit. Three custom exceptions: `InvalidTabularError`, `EmptyTabularError`, `FileTooLargeError`.

Multi-sheet Excel: all sheets concatenated with `__sheet__` tracker column. Sheet boundaries tracked for page header display. Empty sheets silently skipped.

**Key decision — lazy imports:** `pandas` and `matplotlib` are imported on first use, not at module level. This means the server starts fine without pandas installed — it only fails when someone actually uploads a CSV/Excel file. PDF-only users are unaffected.

```python
pd = None  # module level

def _ensure_pandas():
    global pd
    if pd is None:
        import pandas as _pd
        pd = _pd
    return pd
```

**`treerag/bm25.py`** (~200 lines)

Okapi BM25 implementation with tabular-aware tokenization:

- `BM25Index.build(node_texts: dict[str, str])` — builds inverted index from leaf node text
- `BM25Index.score_all(query: str)` — returns `{node_id: normalized_score}` for all nodes
- `BM25Index.save(path)` / `BM25Index.load(path)` — JSON persistence as `.bm25.json`

BM25 parameters: k1=1.5, b=0.75 (standard Okapi defaults).

Custom tokenizer designed for tabular content:
- Splits on markdown pipe characters (`|`)
- Preserves email addresses and URLs as single tokens
- Normalizes currency ($1,234.56 → searchable)
- Removes English stop words
- Lowercase normalization

Scores normalized to [0, 1] range for use as PUCT prior `P`.

### Files Modified

**`treerag/mcts.py`** — Added PUCT selection + three-tier execution

The `search_document()` function now accepts an optional `bm25_index` parameter. When provided (tabular files), it runs the tiered search:

```
Tier 1: BM25 scores all leaves. If top score ≥ 2× runner-up → return immediately (0 LLM calls)
Tier 2: BM25 spread detected → run MCTS with PUCT selection (BM25 as prior P)
Tier 3: All BM25 scores near zero → fall back to standard UCB1 (same as PDF path)
```

PUCT selection (`_select_puct`) replaces UCB1 during tree traversal:
```python
puct = (q / n) + C * P * math.sqrt(parent_n) / (1 + n)
```

Where P is the max BM25 score among the node's descendants (propagated up from leaves).

**Key decision — BM25 prior propagation:** BM25 only scores leaf nodes (they have the actual text). Parent nodes get `P = max(child P values)`. This ensures the PUCT formula explores branches that contain high-scoring leaves.

**`treerag/pipeline.py`** — File type dispatch + load-once BM25

- `index()` detects file extension, dispatches to PDF processor or tabular processor
- `_build_bm25_index()` loads the file once, extracts text from all leaf nodes, builds BM25 index
- `query_document()` loads BM25 index for tabular files, passes to MCTS
- `_deep_read_single()` / `_deep_read()` uses text mode for tabular, vision for PDF

**Key decision — load file once for BM25:** Original implementation opened the CSV/Excel file once per leaf node during BM25 index building. For a file with 50 pages, that's 50 file opens + pandas parses. Fixed to load once, extract all pages, close.

**`treerag/folder_manager.py`** — BM25 lifecycle management

- `add_document()` routes to PDF or tabular processor/indexer by extension
- Files stored in `files/` dir (CSV/Excel) or `pdfs/` dir (PDFs)
- BM25 index built after tree index for tabular files, saved as `.bm25.json`
- `remove_document()` deletes BM25 index alongside tree index
- `health_check()` detects missing BM25 indices for tabular files (new `missing_bm25` category)
- `repair_folder()` rebuilds missing BM25 indices

**`treerag/exceptions.py`** — Three new exceptions

- `TreeRAGError` — base class for all custom exceptions
- `InvalidTabularError` — CSV/Excel cannot be parsed
- `EmptyTabularError` — CSV/Excel has no data rows

**`server.py`** — Auto-detection for single mode

- `single_upload` passes `save_path` so BM25 index gets built
- `single_chat` auto-detects tabular files and sets `use_vision=False`
- UI title changed from "Single PDF" to "Single File"

### Rendering Details

Markdown table page format (what the indexer and BM25 see):

```
File: sales_data.csv | Rows 1-50 of 2,400

| Region | Product | Revenue | Date |
| --- | --- | --- | --- |
| NA | Widget | 5000 | 2024-01-15 |
| EU | Gadget | 3200 | 2024-01-16 |
...
```

Display limits: 20 columns max, 40 chars per cell value. Multi-sheet Excel shows sheet name in header. NaN values rendered as empty strings.

Image rendering (matplotlib): capped at 30 rows per page for performance, blue header row, auto-sized columns. Optional — only used if vision mode is requested.

### Test Coverage

75 tests total, all passing. Tabular-specific tests cover:

- TabularProcessor validation (valid CSV, valid Excel, empty file, bad extension, corrupt file)
- LoadedTabular interface (page count, page text format, page images, batch accessors)
- Multi-sheet Excel handling (sheet boundaries, combined page count)
- BM25Index (build, score, save/load, tokenizer edge cases)
- PUCT selection (tier 1 instant match, tier 2 guided search, tier 3 fallback)
- Pipeline integration (index CSV, query CSV, BM25 cache)
- Folder manager (add CSV, remove with BM25 cleanup, health check missing BM25)

### Backward Compatibility

| Concern | Resolution |
|---------|-----------|
| Existing `pdfs/` directories | Still created. Lookup checks both `pdfs/` and `files/`. |
| `pdf_path` field in JSON indices | Kept as-is — renaming breaks serialized data. |
| PDF indexing behavior | Completely unchanged — separate code path. |
| pandas dependency | Lazy import — PDF-only users never need it. |
| BM25 index missing for old tabular files | `health_check()` detects, `repair_folder()` rebuilds. |
