# AlphaSearch v0.3 — Technical Documentation

**AlphaGo-inspired MCTS for Document Retrieval**

---

## Table of Contents

1. [Overview](#overview)
2. [Two Modes](#two-modes)
3. [Architecture](#architecture)
4. [The Search Phases](#the-search-phases)
5. [MCTS Algorithm Deep Dive](#mcts-algorithm-deep-dive)
6. [Folder System](#folder-system)
7. [Edge Case Handling](#edge-case-handling)
8. [Caching & Incremental Updates](#caching--incremental-updates)
9. [Parallel Processing](#parallel-processing)
10. [API Reference](#api-reference)
11. [Module Reference](#module-reference)
12. [CLI Reference](#cli-reference)
13. [Configuration Guide](#configuration-guide)
14. [Full Walkthrough](#full-walkthrough)
15. [Cost Analysis](#cost-analysis)
16. [Comparison](#comparison)
17. [Limitations & Future Work](#limitations--future-work)

---

## Overview

AlphaSearch replaces vector database retrieval with Monte Carlo Tree Search (MCTS). Instead of embedding chunks into vectors and doing cosine similarity, it builds a hierarchical tree index from documents and uses LLM reasoning to navigate it. Supports PDFs, CSVs, and Excel files.

The core insight: **similarity ≠ relevance**. Vector search finds text that *looks* similar. AlphaSearch finds text that actually *answers the question* — by reasoning about document structure like a human expert would.

For tabular data (CSV/Excel), a second insight applies: **BM25 + MCTS mirrors AlphaGo's dual-network architecture**. BM25 acts as the "policy network" (fast keyword scoring to guide exploration), while the LLM acts as the "value network" (slower semantic evaluation). Combined via the PUCT formula, this searches tabular data efficiently — exact lookups resolve instantly via BM25, semantic queries benefit from LLM-guided MCTS.

---

## Two Modes

### Single File Mode (`/single`)

Upload one file (PDF, CSV, or Excel), ask questions. No folders, no routing, no setup.

```
Upload file → Index → Ask question → Phase 2 (MCTS section search) → Deep Read → Answer

PDF:       GPT-4o reads page images → tree index → UCB1 MCTS → vision deep read
CSV/Excel: pandas → markdown tables → same indexer → BM25 index → PUCT MCTS → text deep read
```

Best for: Quick analysis of individual documents, one-off queries, demos.

### Folder Mode (`/folders`)

Organize documents into folders. Supports mixed file types (PDFs alongside CSVs and Excel files). The system auto-routes queries to the right folder, picks the right document, finds the right section.

```
Create folders → Upload files (PDF, CSV, Excel) → Ask question
    → Phase 0 (route to folder) → Phase 1 (select docs)
    → Phase 2 (find sections, parallel) → Deep Read → Answer
```

Best for: Teams with multiple projects, cross-document search, ongoing use.

### What's Shared

Both modes use the exact same core engine:
- Same MCTS algorithm (UCB1 for PDFs, PUCT with BM25 prior for tabular)
- Same indexer (GPT-4o reads content → tree with coverage check)
- Same deep read (vision for PDFs, text for CSV/Excel)
- Same answer generation (citations with doc/section/page)

Single mode just skips Phase 0 and Phase 1. File type (PDF vs CSV/Excel) determines the processing path, not the mode.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     http://localhost:8000                             │
│                                                                     │
│  Landing Page (/)                                                   │
│  ┌──────────────────┐  ┌──────────────────┐                         │
│  │  📄 Single File   │  │  📁 Folders       │                         │
│  │  /single          │  │  /folders         │                         │
│  └────────┬─────────┘  └────────┬─────────┘                         │
│           │                      │                                   │
│  ┌────────┴──────────────────────┴─────────────────────────────┐     │
│  │                    FastAPI Server (server.py)                │     │
│  │                                                             │     │
│  │  /api/single/upload    /api/folders                         │     │
│  │  /api/single/chat      /api/folders/{name}/documents        │     │
│  │  /api/single/status    /api/folders/chat                    │     │
│  │  /api/single/reset     /api/folders/chat/reset              │     │
│  └───────────────────────────┬─────────────────────────────────┘     │
│                              │                                       │
│  ┌───────────────────────────┴─────────────────────────────────┐     │
│  │                    AlphaSearch Pipeline                       │     │
│  │                                                             │     │
│  │  Single Mode              Folder Mode                       │     │
│  │  ───────────              ───────────                       │     │
│  │                           Phase 0 — Router Agent            │     │
│  │                           (scores folder summaries)         │     │
│  │                                │                            │     │
│  │                           Phase 1 — Doc Selection           │     │
│  │                           (MCTS on meta-tree)               │     │
│  │                                │                            │     │
│  │  Phase 2 ────────────── Phase 2 (parallel)                  │     │
│  │  MCTS on doc tree       MCTS on each selected doc           │     │
│  │  (UCB1 for PDF,         (UCB1 for PDF,                      │     │
│  │   PUCT+BM25 for CSV)    PUCT+BM25 for CSV)                 │     │
│  │       │                      │                              │     │
│  │  Deep Read ──────────── Deep Read                           │     │
│  │  (vision for PDF,       (vision for PDF,                    │     │
│  │   text for CSV/Excel)    text for CSV/Excel)                │     │
│  │       │                      │                              │     │
│  │  Answer ────────────── Answer                               │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │  Storage (.treerag_data/)                                   │     │
│  │  ├── uploads/              ← temp upload directory          │     │
│  │  └── folders/                                               │     │
│  │      ├── project_zenith/                                    │     │
│  │      │   ├── folder_index.json  ← meta-tree                │     │
│  │      │   ├── indices/           ← per-doc trees + .bm25    │     │
│  │      │   ├── pdfs/              ← uploaded PDFs             │     │
│  │      │   └── files/             ← uploaded CSV/Excel        │     │
│  │      └── client_onboarding/                                 │     │
│  └─────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
alphasearch/
├── server.py              FastAPI (landing + /single + /folders + API)
├── chat_ui.jsx            React frontend (reads APP_MODE from server)
├── main.py                CLI entry point
├── requirements.txt       Dependencies
├── .env.example           Config template
├── LICENSE                MIT
├── README.md              Quick-start guide
├── DOCUMENTATION.md       This file
└── treerag/               Core package
    ├── __init__.py         Public API exports
    ├── config.py           ModelConfig, MCTSConfig, IndexerConfig, FolderConfig
    ├── exceptions.py       16 custom exceptions (PDF + tabular)
    ├── models.py           TreeNode, DocumentIndex, FolderIndex, SearchResult
    ├── llm_client.py       OpenAI wrapper (text + vision, retry, tracking)
    ├── pdf_processor.py    PDF validation + page rendering
    ├── tabular_processor.py CSV/Excel → markdown table pages (LoadedTabular)
    ├── bm25.py             BM25 inverted index for tabular text search
    ├── indexer.py          Content → tree (handles both PDF and tabular pages)
    ├── mcts.py             Two-phase MCTS (UCB1 for PDF, PUCT+BM25 for tabular)
    ├── router.py           Phase 0 folder routing agent
    ├── folder_manager.py   CRUD, caching, health check, BM25 lifecycle, thread safety
    └── pipeline.py         Orchestration, chat(), query_folder(), query_document()
```

---

## The Search Phases

### Phase 0 — Router Agent (Folder Mode Only)

Scores all folder summaries against the query + chat history. Uses GPT-4o-mini. Picks the best folder with a confidence score. If the user asks a follow-up, chat history keeps the context so it routes to the same folder.

Cost: ~1-3 GPT-4o-mini calls (~$0.001)

### Phase 1 — Document Selection (Folder Mode Only)

MCTS on the meta-tree (flat list of document summaries). 15 iterations. UCB1 selects documents, GPT-4o-mini scores relevance, backpropagation updates scores. Picks top-K documents (default 3).

Cost: ~15 GPT-4o-mini calls (~$0.002)

### Phase 2 — Section Search (Both Modes)

Full MCTS on each document's internal tree. 25 iterations. In folder mode, runs in parallel across selected documents via ThreadPoolExecutor.

**For PDFs:** Standard UCB1 selection → random expansion → LLM simulation → backpropagation. Early stopping when confident.

**For CSV/Excel:** Three-tier execution based on BM25 signal strength:

| Tier | Condition | Behavior | LLM Calls |
|------|-----------|----------|-----------|
| **Tier 1: BM25 Clear Winner** | Top score ≥ 2× runner-up | Skip MCTS entirely, return BM25 result | 0 |
| **Tier 2: BM25 Spread** | Scores vary, no clear winner | PUCT-guided MCTS (BM25 as prior P) | Reduced |
| **Tier 3: No BM25 Signal** | All scores near zero | Standard UCB1 MCTS (same as PDF) | Full 25 |

The tiered approach means exact lookups (names, emails, IDs) resolve instantly via Tier 1, keyword queries use Tier 2 with fewer LLM calls, and semantic queries fall through to full MCTS.

Cost: ~0-25 GPT-4o-mini calls per document (~$0.00-$0.003/doc)

### Deep Read + Answer (Both Modes)

Top 3 matched sections → extract specific data → synthesize answer with citations.

**For PDFs:** Send actual page images to GPT-4o (vision mode).
**For CSV/Excel:** Send markdown table text to GPT-4o (text mode — cheaper, faster, lossless).

Cost: ~4 GPT-4o calls (~$0.04 for PDF, ~$0.01 for CSV/Excel)

### Cost Summary

| Mode | Phases | LLM Calls | Cost |
|------|--------|-----------|------|
| Single PDF | Phase 2 (UCB1) + Deep Read (vision) | ~29 | ~$0.04 |
| Single CSV/Excel (exact match) | BM25 Tier 1 + Deep Read (text) | ~4 | ~$0.01 |
| Single CSV/Excel (keyword) | BM25 Tier 2 (PUCT) + Deep Read (text) | ~15 | ~$0.02 |
| Folder (3 docs) | Phase 0+1+2 + Deep Read | ~97 | ~$0.05 |

---

## MCTS Algorithm Deep Dive

### UCB1 Formula

```
UCB1(node) = Q(node) + C × √(ln(N_parent) / N_node)
             ├─ exploit ─┤  ├─── explore ───┤
```

- **Q(node):** Average relevance score. Favors nodes that scored well.
- **C (default √2):** Exploration constant. Higher = more exploration.
- **N_parent / N_node:** Visit counts. Unvisited nodes = ∞ (always explored first).

### The Four Operations

**1. Select** — Walk down tree using UCB1. Balances exploitation and exploration.

**2. Expand** — At a node with unvisited children, pick one randomly.

**3. Simulate** — GPT-4o-mini scores "how likely is this section to contain the answer?" (0.0-1.0). This replaces AlphaGo's random playouts.

**4. Backpropagate** — Propagate score upward to all ancestors. If subsection scores 0.9, parent chapter gets +0.9, making siblings more likely to be explored.

### PUCT Formula (CSV/Excel)

For tabular data, BM25 provides a prior probability `P` that guides MCTS exploration:

```
PUCT(node) = Q(node)/N(node) + C × P × √(N_parent) / (1 + N_node)
              ├─ exploit ──┤   ├────────── explore ──────────┤
```

- **Q/N:** Average relevance score (same as UCB1 exploit term).
- **P:** Normalized BM25 score for this node against the query. High P = BM25 thinks this node's text matches.
- **C × P × √(N_parent) / (1+N):** Exploration term modulated by BM25 prior. Nodes with high BM25 scores get explored first.

This mirrors AlphaGo's architecture exactly:
- **Policy network** → BM25 (fast, ~1ms, suggests promising moves)
- **Value network** → LLM (slow, ~200ms, evaluates positions accurately)
- **MCTS** → Combines both via PUCT (same formula AlphaGo uses)

### Why MCTS Beats Alternatives

| Approach | Weakness | MCTS Advantage |
|----------|----------|----------------|
| Greedy top-down | One wrong turn = miss everything | UCB1 explores multiple branches |
| Beam search (K=3) | Commits at each level, can't backtrack | Backpropagation revises beliefs |
| Exhaustive search | Scores every node = expensive | Focuses on promising areas |
| Vector similarity | Similarity ≠ relevance | LLM reasons about meaning |

### Early Stopping

Stops when 3+ leaf nodes have been visited 3+ times with scores above confidence threshold (default 0.7).

---

## Folder System

Each folder has a lightweight meta-tree (document summaries) and independent per-document trees.

| Operation | What Happens |
|-----------|-------------|
| Create folder | Directory + folder_index.json + `pdfs/` + `files/` + `indices/` |
| Add PDF | Validate → copy to `pdfs/` → vision index → summary → meta-tree update |
| Add CSV/Excel | Validate → copy to `files/` → text index → BM25 index → summary → meta-tree |
| Add unchanged file | Hash match → skipped (cached) |
| Add changed file | Hash mismatch → re-index (+ rebuild BM25 if tabular) → replace entry |
| Remove document | Meta-tree entry + index + BM25 index + source file deleted |
| Health check | Detects missing files, stale indices, orphans, missing BM25 for tabular |
| Repair | Re-indexes stale, removes orphans, rebuilds missing BM25 |

---

## Edge Case Handling

### PDF Validation

| Edge Case | Exception |
|-----------|-----------|
| File not found | `FileNotFoundError` |
| Not a file | `InvalidFileError` |
| Wrong extension | `InvalidFileError` |
| Empty (0 bytes) | `InvalidFileError` |
| Bad magic bytes | `InvalidFileError` |
| Corrupt/encrypted | `CorruptPDFError` |
| 0 pages | `EmptyPDFError` |
| >5000 pages | `FileTooLargeError` |
| >1000 pages | Warning (continues) |
| Corrupt page | Skipped with warning |
| Access after close | `RuntimeError` |

### CSV/Excel Validation

| Edge Case | Exception |
|-----------|-----------|
| File not found | `FileNotFoundError` |
| Not a file | `InvalidTabularError` |
| Unsupported extension | `InvalidTabularError` |
| Empty (0 bytes) | `InvalidTabularError` |
| Cannot parse | `InvalidTabularError` |
| No columns | `EmptyTabularError` |
| No data rows | `EmptyTabularError` |
| All sheets empty (Excel) | `EmptyTabularError` |
| >100,000 rows | `FileTooLargeError` |
| >50,000 rows | Warning (`is_large` flag) |
| pandas not installed | `ImportError` (lazy — only when CSV/Excel used) |
| Access after close | `RuntimeError` |

### Folder Operations

| Edge Case | Handling |
|-----------|---------|
| Create existing folder | `FolderAlreadyExistsError` |
| Load missing folder | `FolderNotFoundError` |
| Invalid folder name | `ValueError` |
| Add to non-existent folder | `FolderNotFoundError` |
| Remove non-existent doc | `DocumentNotFoundError` |
| Duplicate filename | Warning, replaces |
| Batch: partial failure | Continues, reports all |
| Disk full | Atomic write, cleanup |
| Index manually deleted | Detected by health check |
| Source file deleted | Detected by health check |
| BM25 index missing (tabular) | Detected by health check, rebuilt on repair |
| Concurrent writes | Per-folder thread locks |
| Empty folder search | `EmptyFolderError` |
| Deep read missing file | Warning, uses summaries |

---

## Caching & Incremental Updates

Every document tracked by MD5 hash:
- Same hash → skip (instant)
- Different hash → re-index only that file (+ rebuild BM25 if tabular)
- New file → index only the new file

BM25 indices stored alongside tree indices as `.bm25.json` files. Rebuilt automatically when the source file changes.

Folder summaries (Router Agent) cached in memory per session.

Storage: ~5-20KB per document tree index, ~2-10KB per BM25 index (tabular only).

---

## Parallel Processing

Phase 2 uses `ThreadPoolExecutor` (max 4 workers):

```
Sequential: 25 iters × 3 docs × ~200ms = ~15s
Parallel:   25 iters × 1 doc  × ~200ms = ~5s (3x speedup)
```

Disable with `PARALLEL_PHASE2=false`.

---

## API Reference

### Landing & UI

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page (pick Single or Folder mode) |
| `/single` | GET | Single File chat UI |
| `/folders` | GET | Folder mode chat UI |

### Single Mode

**`POST /api/single/upload`** — Upload + index a file (PDF, CSV, or Excel)
```
Content-Type: multipart/form-data
Body: file=@document.pdf  (or data.csv, report.xlsx)
Response: {"filename": "doc.pdf", "pages": 42, "description": "...", "status": "indexed"}
```

**`POST /api/single/chat`** — Query the uploaded document
```json
Request:  {"query": "What are the key findings?"}
Response: {"answer": "...", "sources": [...], "stats": {"total": "3.2s", "calls": 29, "cost": "$0.04"}}
```

**`GET /api/single/status`** — Check if a document is loaded
```json
Response: {"loaded": true, "filename": "doc.pdf", "pages": 42, "description": "..."}
```

**`POST /api/single/reset`** — Clear document + chat
```json
Response: {"status": "cleared"}
```

### Folder Mode

**`GET /api/folders`** — List folders
```json
Response: {"folders": [{"name": "Project Zenith", "docs": [...], "total_pages": 170}]}
```

**`POST /api/folders`** — Create folder
```json
Request: {"name": "Project Zenith"}
```

**`DELETE /api/folders/{name}`** — Delete folder

**`POST /api/folders/{name}/documents`** — Upload + index file (PDF, CSV, Excel) into folder
```
Content-Type: multipart/form-data
Body: file=@document.pdf
```

**`DELETE /api/folders/{name}/documents/{filename}`** — Remove document

**`POST /api/folders/chat`** — Chat with auto-routing (Phase 0→1→2)
```json
Request:  {"query": "What was the budget?"}
Response: {"answer": "...", "folder": "Project Zenith", "sources": [...], "stats": {...}}
```

**`POST /api/folders/chat/reset`** — Clear chat history

### System

**`GET /api/health`**
```json
{"status": "ok", "folders": 3, "single_loaded": true, "usage": {"total_calls": 150, "estimated_cost_usd": 0.23}}
```

---

## Module Reference

### server.py
Landing page at `/`. Injects `APP_MODE` ("single" or "folders") into chat_ui.jsx. Separate state and APIs for each mode. Serves UI via Babel standalone (no build step). Auto-detects tabular files and uses text-mode deep reads (skips vision) for CSV/Excel.

### chat_ui.jsx
Reads `APP_MODE` to switch between `SingleSidebar` (upload dropzone, doc info, reset) and `FolderSidebar` (folder CRUD, multi-doc). Shared components: Msg, Source, Phase, Md, Welcome. API client switches endpoints based on mode.

### main.py
CLI commands: `chat` (unified auto-route), `interactive` (folder-specific), `folder` (CRUD), `search`, `search-doc`, `query` (standalone), `inspect`.

### config.py
`ModelConfig`, `MCTSConfig`, `IndexerConfig`, `FolderConfig`. All loadable from `.env`.

### exceptions.py
16 exceptions: `InvalidFileError`, `CorruptPDFError`, `EmptyPDFError`, `FileTooLargeError`, `FolderNotFoundError`, `FolderAlreadyExistsError`, `DocumentNotFoundError`, `IndexNotFoundError`, `IndexCorruptError`, `PDFMissingError`, `IndexingFailedError`, `BatchIndexingError`, `EmptyFolderError`, `InvalidTabularError`, `EmptyTabularError`, `TreeRAGError` (base).

### models.py
`TreeNode` (UCB1, MCTS state), `DocumentIndex`, `FolderIndex`, `FolderDocEntry`, `SearchResult`, `QueryResult`. Atomic writes via temp+rename.

### llm_client.py
OpenAI wrapper: `complete()` (text), `complete_with_images()` (vision), retry with backoff, token tracking.

### pdf_processor.py
`validate()` (magic bytes, pages, size), `load()` → `LoadedPDF` (safe page access, corrupt page skip, close-state detection).

### tabular_processor.py
`TabularProcessor`: `validate()` (extension, parse check, row count, size), `load()` → `LoadedTabular`. Same interface as `LoadedPDF` (`total_pages`, `get_page_text()`, `get_page_image()`, batch accessors, context manager). Pandas/matplotlib are lazy-imported — only loaded when a CSV/Excel file is actually processed. Server starts fine without pandas if only PDFs are used.

### bm25.py
`BM25Index`: Inverted index with Okapi BM25 scoring (k1=1.5, b=0.75). `build(node_texts)` builds the index from leaf node text. `score_all(query)` returns normalized scores for all nodes. `save()`/`load()` for persistence as `.bm25.json`. Custom tokenizer handles tabular content (splits on pipes, preserves emails/URLs, normalizes numbers).

### indexer.py
`index_document()`: Batch pages → GPT-4o → sections → tree. Works identically for both PDF pages and markdown table pages. Coverage check (70% threshold). Per-page fallback for slides/presentations. `generate_document_summary()` for meta-tree.

### mcts.py
`search_meta()`: Phase 1. `search_document()`: Phase 2. `search_documents_parallel()`: Parallel Phase 2 with timeout. Core MCTS: select, expand, simulate, backpropagate. Supports two selection modes: UCB1 (PDFs) and PUCT with BM25 prior (tabular). Three-tier execution for tabular: Tier 1 (BM25 instant), Tier 2 (PUCT-guided), Tier 3 (standard UCB1).

### router.py
`route()`: Phase 0. Scores folder summaries + chat history. Cached summaries. Keyword fallback.

### folder_manager.py
CRUD with validation, atomic writes, hash caching, batch add, health check (6 issue types including missing BM25), repair, thread-safe. Dispatches to PDF or tabular processor by file extension. Manages BM25 index lifecycle (create on add, delete on remove, rebuild on repair).

### pipeline.py
`chat()`: Unified (Route → query_folder). `query_folder()`: Phase 1+2+deep read. `query_document()`: Phase 2+deep read (used by single mode). Dispatches processor by file extension. Loads file once for BM25 index building (not per-leaf). `ChatMessage` dataclass. Lazy init.

---

## CLI Reference

```bash
python server.py                  # Web UI at http://localhost:8000

python main.py chat               # Unified chat (auto-routes)
python main.py interactive "Proj" # Folder-specific interactive

python main.py folder create "Proj"
python main.py folder add "Proj" a.pdf b.pdf data.csv report.xlsx
python main.py folder list
python main.py folder info "Proj"
python main.py folder health "Proj"
python main.py folder repair "Proj"
python main.py folder refresh "Proj"
python main.py folder delete "Proj"

python main.py search "Proj" "budget?"
python main.py search-doc "Proj" budget.pdf "Phase 2?"
python main.py search-doc "Proj" sales.csv "Q3 revenue?"

python main.py query report.pdf "Summary?"
python main.py query sales_data.csv "Top customers?"
python main.py query financials.xlsx "Total expenses?"
python main.py inspect path/to/index.json
```

---

## Configuration Guide

### .env

```bash
OPENAI_API_KEY=sk-...          # Required

INDEXING_MODEL=gpt-4o
SEARCH_MODEL=gpt-4o-mini
ANSWER_MODEL=gpt-4o

MCTS_ITERATIONS=25
MCTS_META_ITERATIONS=15
UCB_EXPLORATION_CONSTANT=1.414
CONFIDENCE_THRESHOLD=0.7
TOP_K_DOCUMENTS=3
PARALLEL_PHASE2=true

BATCH_SIZE=15
MAX_PAGES=5000

TREERAG_DATA_DIR=.treerag_data
AUTO_REINDEX=true
```

### Presets

**Accuracy:** `MCTS_ITERATIONS=40, MCTS_META_ITERATIONS=20, UCB_EXPLORATION_CONSTANT=1.6, CONFIDENCE_THRESHOLD=0.8, BATCH_SIZE=8`

**Speed:** `MCTS_ITERATIONS=15, MCTS_META_ITERATIONS=10, CONFIDENCE_THRESHOLD=0.6, TOP_K_DOCUMENTS=2`

**Cost:** `INDEXING_MODEL=gpt-4o-mini, MCTS_ITERATIONS=15, TOP_K_DOCUMENTS=2`

---

## Full Walkthrough

### Single Mode — PDF
```bash
python server.py → http://localhost:8000/single
# Upload report.pdf (50 pages) → Indexed in ~30s
# Ask: "What were the Q3 results?"
# → Phase 2: MCTS (UCB1) finds "Q3 Financial Results" (pp.12-15)
# → Deep Read: GPT-4o reads pages 12-15 images
# → Answer: "Q3 revenue was $4.2M, up 15%..."
# → Source: [report.pdf → Q3 Financial Results, pp.12-15] [92%]
# → Stats: 3.2s | 29 calls | $0.04
```

### Single Mode — CSV
```bash
python server.py → http://localhost:8000/single
# Upload sales_data.csv (2,400 rows) → Indexed in ~15s + BM25 built
# Ask: "john@acme.com purchase history"
# → BM25 Tier 1: Clear match on rows mentioning john@acme.com
# → Deep Read: GPT-4o reads matching markdown table (text mode)
# → Answer: "John Smith (john@acme.com) made 12 purchases totaling $45K..."
# → Stats: 1.1s | 4 calls | $0.01
```

### Folder Mode
```bash
python server.py → http://localhost:8000/folders
# Create "Project Zenith" → Upload proposal.pdf, budget.pdf, expenses.csv
# Ask: "What was the total project cost?"
# → Phase 0: Routes to "Project Zenith" (94%)
# → Phase 1: Selects budget.pdf, expenses.csv
# → Phase 2: Finds "Cost Summary" (pp.5-8) in PDF, BM25 matches in CSV
# → Deep Read: vision for PDF, text for CSV
# → Answer: "Total cost is $812,000 across 3 phases..."
# → Source: [budget.pdf → Cost Summary, pp.5-8] [94%]
# → Stats: 5.1s | 34 calls | $0.06
```

---

## Cost Analysis

| Operation | Cost |
|-----------|------|
| Index 10-page PDF | ~$0.10 |
| Index 50-page PDF | ~$0.35 |
| Index 100-page PDF | ~$0.60 |
| Index 500-row CSV | ~$0.04 |
| Index 2,000-row CSV | ~$0.12 |
| Index 10,000-row CSV | ~$0.50 |
| Query PDF (single doc) | ~$0.04 |
| Query CSV (exact match, Tier 1) | ~$0.01 |
| Query CSV (keyword, Tier 2) | ~$0.02 |
| Query CSV (semantic, Tier 3) | ~$0.03 |
| Query (folder, 3 docs) | ~$0.05 |
| Infrastructure | **$0** |

---

## Comparison

| | Vector RAG | PageIndex | **AlphaSearch** |
|---|-----------|-----------|-----------------|
| Modes | Single | Single | **Single + Folder** |
| File types | PDF only | PDF only | **PDF + CSV + Excel** |
| Retrieval | Cosine similarity | Greedy tree | **MCTS (UCB1 + PUCT w/ BM25)** |
| Text search | None | None | **BM25 inverted index (tabular)** |
| Multi-doc | Flat index | Limited | **Meta-tree + auto-routing** |
| Chat context | None | None | **History-aware routing** |
| Parallel | No | No | **ThreadPoolExecutor** |
| Caching | Embedding | None | **MD5 per document** |
| Error handling | Varies | Basic | **16 exceptions + health check** |
| UI | None | Paid | **React chat + landing page** |
| Infrastructure | Vector DB ($70/mo+) | Paid API | **JSON files ($0)** |

---

## Limitations & Future Work

### Current
1. OpenAI-only (architecture supports any vision LLM)
2. Sequential MCTS iterations (documents parallelized, not iterations)
3. No cross-document reference following
4. No query result caching
5. Server-side chat history (resets on restart)
6. BM25 is English-optimized (tokenizer assumes English stop words)

### Planned
1. Multi-model support (Gemini, Claude, local)
2. Cross-document reference following
3. Query result caching
4. Streaming answers
5. BM25 for PDFs (currently tabular-only — would benefit keyword-heavy PDF queries)
6. Nested folder hierarchy
7. Persistent chat history (SQLite)
8. WebSocket for real-time phase progress
9. Authentication and multi-user
10. Multi-language BM25 tokenization

---

## License

MIT
