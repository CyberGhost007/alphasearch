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

AlphaSearch replaces vector database retrieval with Monte Carlo Tree Search (MCTS). Instead of embedding chunks into vectors and doing cosine similarity, it builds a hierarchical tree index from documents and uses LLM reasoning to navigate it.

The core insight: **similarity ≠ relevance**. Vector search finds text that *looks* similar. AlphaSearch finds text that actually *answers the question* — by reasoning about document structure like a human expert would.

---

## Two Modes

### Single PDF Mode (`/single`)

Upload one PDF, ask questions. No folders, no routing, no setup.

```
Upload PDF → Index (GPT-4o reads pages) → Ask question
    → Phase 2 (MCTS section search) → Deep Read → Answer
```

Best for: Quick analysis of individual documents, one-off queries, demos.

### Folder Mode (`/folders`)

Organize documents into folders. The system auto-routes queries to the right folder, picks the right document, finds the right section.

```
Create folders → Upload PDFs → Ask question
    → Phase 0 (route to folder) → Phase 1 (select docs)
    → Phase 2 (find sections, parallel) → Deep Read → Answer
```

Best for: Teams with multiple projects, cross-document search, ongoing use.

### What's Shared

Both modes use the exact same core engine:
- Same MCTS algorithm (UCB1 select → expand → simulate → backpropagate)
- Same indexer (GPT-4o vision → tree with coverage check)
- Same deep read (actual page images → GPT-4o extraction)
- Same answer generation (citations with doc/section/page)

Single mode just skips Phase 0 and Phase 1.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     http://localhost:8000                             │
│                                                                     │
│  Landing Page (/)                                                   │
│  ┌──────────────────┐  ┌──────────────────┐                         │
│  │  📄 Single PDF    │  │  📁 Folders       │                         │
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
│  │       │                      │                              │     │
│  │  Deep Read ──────────── Deep Read                           │     │
│  │  GPT-4o vision          GPT-4o vision                       │     │
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
│  │      │   ├── indices/           ← per-doc trees             │     │
│  │      │   └── pdfs/              ← uploaded originals        │     │
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
    ├── __init__.py
    ├── config.py           ModelConfig, MCTSConfig, IndexerConfig, FolderConfig
    ├── exceptions.py       13 custom exceptions
    ├── models.py           TreeNode, DocumentIndex, FolderIndex, SearchResult
    ├── llm_client.py       OpenAI wrapper (text + vision, retry, tracking)
    ├── pdf_processor.py    PDF validation + page rendering
    ├── indexer.py          PDF → tree (coverage check, per-page fallback)
    ├── mcts.py             Two-phase MCTS (meta + per-doc, parallel)
    ├── router.py           Phase 0 folder routing agent
    ├── folder_manager.py   CRUD, caching, health check, thread safety
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

This is where the core MCTS algorithm operates — UCB1 selection, random expansion, LLM simulation, backpropagation. Early stopping when confident.

Cost: ~25 GPT-4o-mini calls per document (~$0.003/doc)

### Deep Read + Answer (Both Modes)

Top 3 matched sections → send their actual page images to GPT-4o → extract specific data. A final GPT-4o call synthesizes the answer with citations.

Cost: ~4 GPT-4o calls (~$0.04)

### Cost Summary

| Mode | Phases | LLM Calls | Cost |
|------|--------|-----------|------|
| Single PDF | Phase 2 + Deep Read | ~29 | ~$0.04 |
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
| Create folder | Directory + folder_index.json |
| Add document | Validate → copy → index → summary → meta-tree update |
| Add unchanged file | Hash match → skipped (cached) |
| Add changed file | Hash mismatch → re-index → replace entry |
| Remove document | Meta-tree entry + index + PDF deleted |
| Health check | Detects missing PDFs, stale indices, orphans |
| Repair | Re-indexes stale, removes orphans |

---

## Edge Case Handling

### File Validation

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
| PDF deleted | Detected by health check |
| Concurrent writes | Per-folder thread locks |
| Empty folder search | `EmptyFolderError` |
| Deep read missing PDF | Warning, uses summaries |

---

## Caching & Incremental Updates

Every document tracked by MD5 hash:
- Same hash → skip (instant)
- Different hash → re-index only that file
- New file → index only the new file

Folder summaries (Router Agent) cached in memory per session.

Storage: ~5-20KB per document index.

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
| `/single` | GET | Single PDF chat UI |
| `/folders` | GET | Folder mode chat UI |

### Single Mode

**`POST /api/single/upload`** — Upload + index a PDF
```
Content-Type: multipart/form-data
Body: file=@document.pdf
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

**`POST /api/folders/{name}/documents`** — Upload + index PDF into folder
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
Landing page at `/`. Injects `APP_MODE` ("single" or "folders") into chat_ui.jsx. Separate state and APIs for each mode. Serves UI via Babel standalone (no build step).

### chat_ui.jsx
Reads `APP_MODE` to switch between `SingleSidebar` (upload dropzone, doc info, reset) and `FolderSidebar` (folder CRUD, multi-doc). Shared components: Msg, Source, Phase, Md, Welcome. API client switches endpoints based on mode.

### main.py
CLI commands: `chat` (unified auto-route), `interactive` (folder-specific), `folder` (CRUD), `search`, `search-doc`, `query` (standalone), `inspect`.

### config.py
`ModelConfig`, `MCTSConfig`, `IndexerConfig`, `FolderConfig`. All loadable from `.env`.

### exceptions.py
13 exceptions: `InvalidFileError`, `CorruptPDFError`, `EmptyPDFError`, `FileTooLargeError`, `FolderNotFoundError`, `FolderAlreadyExistsError`, `DocumentNotFoundError`, `IndexNotFoundError`, `IndexCorruptError`, `PDFMissingError`, `IndexingFailedError`, `BatchIndexingError`, `EmptyFolderError`.

### models.py
`TreeNode` (UCB1, MCTS state), `DocumentIndex`, `FolderIndex`, `FolderDocEntry`, `SearchResult`, `QueryResult`. Atomic writes via temp+rename.

### llm_client.py
OpenAI wrapper: `complete()` (text), `complete_with_images()` (vision), retry with backoff, token tracking.

### pdf_processor.py
`validate()` (magic bytes, pages, size), `load()` → `LoadedPDF` (safe page access, corrupt page skip, close-state detection).

### indexer.py
`index_document()`: Batch pages → GPT-4o vision → sections → tree. Coverage check (70% threshold). Per-page fallback for slides/presentations. `generate_document_summary()` for meta-tree.

### mcts.py
`search_meta()`: Phase 1. `search_document()`: Phase 2. `search_documents_parallel()`: Parallel Phase 2 with timeout. Core MCTS: select, expand, simulate, backpropagate.

### router.py
`route()`: Phase 0. Scores folder summaries + chat history. Cached summaries. Keyword fallback.

### folder_manager.py
CRUD with validation, atomic writes, hash caching, batch add, health check (5 issue types), repair, thread-safe.

### pipeline.py
`chat()`: Unified (Route → query_folder). `query_folder()`: Phase 1+2+deep read. `query_document()`: Phase 2+deep read (used by single mode). `ChatMessage` dataclass. Lazy init.

---

## CLI Reference

```bash
python server.py                  # Web UI at http://localhost:8000

python main.py chat               # Unified chat (auto-routes)
python main.py interactive "Proj" # Folder-specific interactive

python main.py folder create "Proj"
python main.py folder add "Proj" a.pdf b.pdf
python main.py folder list
python main.py folder info "Proj"
python main.py folder health "Proj"
python main.py folder repair "Proj"
python main.py folder refresh "Proj"
python main.py folder delete "Proj"

python main.py search "Proj" "budget?"
python main.py search-doc "Proj" budget.pdf "Phase 2?"

python main.py query report.pdf "Summary?"
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

### Single Mode
```bash
python server.py → http://localhost:8000/single
# Upload report.pdf (50 pages) → Indexed in ~30s
# Ask: "What were the Q3 results?"
# → Phase 2: MCTS finds "Q3 Financial Results" (pp.12-15)
# → Deep Read: GPT-4o reads pages 12-15 images
# → Answer: "Q3 revenue was $4.2M, up 15%..."
# → Source: [report.pdf → Q3 Financial Results, pp.12-15] [92%]
# → Stats: 3.2s | 29 calls | $0.04
```

### Folder Mode
```bash
python server.py → http://localhost:8000/folders
# Create "Project Zenith" → Upload proposal.pdf, budget.pdf
# Ask: "What was the total project cost?"
# → Phase 0: Routes to "Project Zenith" (94%)
# → Phase 1: Selects budget.pdf, proposal.pdf
# → Phase 2: Finds "Cost Summary" (pp.5-8), parallel
# → Deep Read: GPT-4o reads pages 5-8 images
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
| Query (single doc) | ~$0.04 |
| Query (folder, 3 docs) | ~$0.05 |
| Infrastructure | **$0** |

---

## Comparison

| | Vector RAG | PageIndex | **AlphaSearch** |
|---|-----------|-----------|-----------------|
| Modes | Single | Single | **Single + Folder** |
| Retrieval | Cosine similarity | Greedy tree | **MCTS (explore+exploit+backtrack)** |
| Multi-doc | Flat index | Limited | **Meta-tree + auto-routing** |
| Chat context | None | None | **History-aware routing** |
| Parallel | No | No | **ThreadPoolExecutor** |
| Caching | Embedding | None | **MD5 per document** |
| Error handling | Varies | Basic | **13 exceptions + health check** |
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

### Planned
1. Multi-model support (Gemini, Claude, local)
2. Cross-document reference following
3. Query result caching
4. Streaming answers
5. Keyword pre-filter before MCTS
6. Nested folder hierarchy
7. Persistent chat history (SQLite)
8. WebSocket for real-time phase progress
9. Authentication and multi-user

---

## License

MIT
