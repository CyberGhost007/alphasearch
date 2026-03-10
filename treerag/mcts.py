"""
Two-Phase MCTS Engine.
Phase 1: Score document summaries → pick top-K docs
Phase 2: Search within selected docs → find sections (parallel)

For tabular documents (CSV/Excel), Phase 2 uses BM25-guided tiered execution:
  Tier 1: Clear BM25 winner → skip MCTS, direct deep read
  Tier 2: Spread BM25 scores → PUCT-guided MCTS (BM25 as policy prior)
  Tier 3: No BM25 signal → standard UCB1 MCTS (same as PDFs)
"""

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from rich.console import Console
from .config import MCTSConfig
from .models import TreeNode, DocumentIndex, FolderIndex, FolderDocEntry, SearchResult, DocumentScore
from .llm_client import LLMClient
from .bm25 import BM25Index

console = Console()

SYSTEM_PROMPT_SIMULATE = """You are a relevance scorer. Output JSON only:
{"relevance_score": <0.0 to 1.0>, "reasoning": "<one sentence>"}
Scoring: 0.0-0.2 unrelated, 0.2-0.4 tangential, 0.4-0.6 partial, 0.6-0.8 likely, 0.8-1.0 direct answer."""

SYSTEM_PROMPT_SIMULATE_DOC = """You are a document relevance scorer. Output JSON only:
{"relevance_score": <0.0 to 1.0>, "reasoning": "<one sentence>"}
Scoring: 0.0-0.2 wrong document, 0.2-0.4 tangential, 0.4-0.6 partial info, 0.6-0.8 likely, 0.8-1.0 almost certain."""

SYSTEM_PROMPT_DEEP_READ = """Extract ALL information relevant to the query from these pages.
Output JSON: {"relevant_content": "...", "relevance_confirmed": true/false, "key_facts": [...], "references_other_sections": [...]}"""

SYSTEM_PROMPT_ANSWER = """Answer using ONLY the provided sources. Cite as [filename → Section, pages].
If sources don't fully answer, say what's missing. Be specific with numbers, dates, names."""


class MCTSEngine:
    def __init__(self, config: MCTSConfig, llm: LLMClient, search_model: str = "gpt-4o-mini"):
        self.config = config
        self.llm = llm
        self.search_model = search_model

    # =========================================================================
    # Phase 1: Meta-tree search
    # =========================================================================

    def search_meta(self, query: str, folder_index: FolderIndex, verbose: bool = True) -> list[DocumentScore]:
        if not folder_index.documents:
            return []

        folder_index.reset_mcts_state()
        total_visits = 0
        iterations = min(self.config.meta_iterations, len(folder_index.documents) * 5)

        if verbose:
            console.print(f"\n[bold cyan]Phase 1 — Document Selection[/bold cyan]")
            console.print(f"  Scoring {folder_index.total_documents} documents ({iterations} iterations)")

        for _ in range(iterations):
            selected = max(folder_index.documents, key=lambda d: d.ucb1(total_visits, self.config.exploration_constant))
            try:
                score = self._simulate_document(query, selected)
            except Exception:
                score = 0.3  # Neutral on LLM failure
            selected.visit_count += 1
            selected.total_reward += score
            total_visits += 1

        scored = sorted(
            [DocumentScore(entry=d, relevance_score=d.average_reward, visit_count=d.visit_count)
             for d in folder_index.documents if d.visit_count > 0],
            key=lambda x: x.relevance_score, reverse=True,
        )
        top_docs = scored[:self.config.top_k_documents]

        if verbose:
            for d in top_docs:
                console.print(f"    → {d.entry.filename} (score: {d.relevance_score:.3f}, visits: {d.visit_count})")
        return top_docs

    def _simulate_document(self, query: str, doc: FolderDocEntry) -> float:
        prompt = f"Query: {query}\nDocument: {doc.filename} ({doc.total_pages} pages)\nSummary: {doc.summary}\nKeywords: {', '.join(doc.keywords) if doc.keywords else 'none'}\nRelevance?"
        response = self.llm.complete(prompt=prompt, model=self.search_model, system_prompt=SYSTEM_PROMPT_SIMULATE_DOC, json_mode=True, max_tokens=256)
        try:
            return max(0.0, min(1.0, float(json.loads(response).get("relevance_score", 0.0))))
        except (json.JSONDecodeError, ValueError):
            return 0.3

    # =========================================================================
    # Phase 2: Per-document search
    # =========================================================================

    def search_document(self, query: str, doc_index: DocumentIndex, verbose: bool = True) -> list[SearchResult]:
        if not doc_index.root:
            return []
        root = doc_index.root
        root.reset_mcts_state()

        if verbose:
            console.print(f"\n  [dim]Searching: {doc_index.filename}[/dim]")

        for _ in range(self.config.iterations):
            selected = self._select(root)
            expanded = self._expand(selected)
            target = expanded or selected
            try:
                reward = self._simulate_section(query, target)
            except Exception:
                reward = 0.3
            self._backpropagate(target, reward)
            if self._should_stop_early(root):
                break

        return self._collect_results(root, doc_index.filename, doc_index.document_id)

    def search_documents_parallel(self, query: str, doc_indices: list[DocumentIndex], verbose: bool = True) -> list[SearchResult]:
        """Phase 2 parallel. Gracefully handles per-document failures."""
        all_results = []
        if verbose:
            console.print(f"\n[bold cyan]Phase 2 — Searching {len(doc_indices)} documents (parallel)[/bold cyan]")

        with ThreadPoolExecutor(max_workers=min(len(doc_indices), 4)) as executor:
            futures = {executor.submit(self.search_document, query, idx, False): idx for idx in doc_indices}
            for future in as_completed(futures):
                doc_idx = futures[future]
                try:
                    results = future.result(timeout=120)  # 2 min timeout per doc
                    if verbose:
                        console.print(f"    {doc_idx.filename}: {len(results)} sections found")
                    all_results.extend(results)
                except TimeoutError:
                    console.print(f"    [yellow]{doc_idx.filename}: Timed out — skipping[/yellow]")
                except Exception as e:
                    console.print(f"    [red]{doc_idx.filename}: Error — {e}[/red]")

        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return all_results[:self.config.top_k_results]

    def search_documents_sequential(self, query: str, doc_indices: list[DocumentIndex], verbose: bool = True) -> list[SearchResult]:
        all_results = []
        if verbose:
            console.print(f"\n[bold cyan]Phase 2 — Searching {len(doc_indices)} documents[/bold cyan]")
        for idx in doc_indices:
            try:
                results = self.search_document(query, idx, verbose)
                all_results.extend(results)
            except Exception as e:
                console.print(f"    [red]{idx.filename}: Error — {e}[/red]")
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return all_results[:self.config.top_k_results]

    # =========================================================================
    # BM25-Guided Tiered Search (for tabular documents)
    # =========================================================================

    def search_document_bm25(
        self,
        query: str,
        doc_index: DocumentIndex,
        bm25_index: BM25Index,
        verbose: bool = True,
    ) -> list[SearchResult]:
        """
        Search a tabular document using BM25 + tiered MCTS execution.

        Tier 1: Direct deep read on BM25 winner (zero MCTS iterations)
        Tier 2: PUCT-guided MCTS using BM25 priors
        Tier 3: Standard UCB1 MCTS (same as PDFs)
        """
        if not doc_index.root:
            return []

        root = doc_index.root
        root.reset_mcts_state()

        # Score all nodes with BM25
        tier, bm25_scores = bm25_index.detect_tier(query)

        if verbose:
            top_score = max(bm25_scores.values()) if bm25_scores else 0
            console.print(
                f"\n  [dim]Searching: {doc_index.filename} "
                f"(Tier {tier}, BM25 top={top_score:.3f})[/dim]"
            )

        if tier == 1:
            # Direct: return top BM25 leaves without MCTS
            return self._tier1_direct(root, bm25_scores, doc_index)

        if tier == 2:
            # PUCT-guided MCTS
            return self._tier2_puct(
                query, root, bm25_scores, doc_index, verbose
            )

        # Tier 3: Standard UCB1 (same as search_document)
        return self.search_document(query, doc_index, verbose)

    def _tier1_direct(
        self,
        root: TreeNode,
        bm25_scores: dict[str, float],
        doc_index: DocumentIndex,
    ) -> list[SearchResult]:
        """Tier 1: Return top BM25 leaf nodes directly, zero MCTS iterations."""
        leaves = self._get_all_leaves(root)

        # Score and rank leaves by BM25
        scored_leaves = []
        for leaf in leaves:
            score = bm25_scores.get(leaf.node_id, 0.0)
            scored_leaves.append((leaf, score))
        scored_leaves.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_pages: set[tuple[int, int]] = set()
        for leaf, score in scored_leaves:
            if len(results) >= self.config.top_k_results:
                break
            if score <= 0.0:
                break
            pr = (leaf.start_page, leaf.end_page)
            if any(pr[0] <= e and pr[1] >= s for s, e in seen_pages):
                continue
            seen_pages.add(pr)
            # Use BM25 score as relevance (scaled to 0-1)
            results.append(SearchResult(
                node=leaf,
                relevance_score=min(score, 1.0),
                visit_count=1,
                document_filename=doc_index.filename,
                document_id=doc_index.document_id,
            ))

        return results

    def _tier2_puct(
        self,
        query: str,
        root: TreeNode,
        bm25_scores: dict[str, float],
        doc_index: DocumentIndex,
        verbose: bool,
    ) -> list[SearchResult]:
        """Tier 2: MCTS with PUCT selection using BM25 priors."""
        for _ in range(self.config.iterations):
            selected = self._select_puct(root, bm25_scores)
            expanded = self._expand(selected)
            target = expanded or selected
            try:
                reward = self._simulate_section(query, target)
            except Exception:
                reward = 0.3
            self._backpropagate(target, reward)
            if self._should_stop_early(root):
                break

        return self._collect_results(
            root, doc_index.filename, doc_index.document_id
        )

    def _select_puct(
        self, node: TreeNode, bm25_scores: dict[str, float]
    ) -> TreeNode:
        """Select node using PUCT (BM25 prior) instead of UCB1."""
        current, depth = node, 0
        while not current.is_leaf and depth < self.config.max_depth:
            unvisited = [c for c in current.children if c.visit_count == 0]
            if unvisited:
                return current
            # PUCT selection: use BM25 score as prior
            current = max(
                current.children,
                key=lambda c: c.puct(
                    bm25_scores.get(c.node_id, 0.0),
                    self.config.exploration_constant,
                ),
            )
            depth += 1
        return current

    # =========================================================================
    # Core MCTS
    # =========================================================================

    def _select(self, node: TreeNode) -> TreeNode:
        current, depth = node, 0
        while not current.is_leaf and depth < self.config.max_depth:
            unvisited = [c for c in current.children if c.visit_count == 0]
            if unvisited:
                return current
            current = max(current.children, key=lambda c: c.ucb1(self.config.exploration_constant))
            depth += 1
        return current

    def _expand(self, node: TreeNode) -> Optional[TreeNode]:
        if node.is_leaf:
            return None
        unvisited = [c for c in node.children if c.visit_count == 0]
        return random.choice(unvisited) if unvisited else None

    def _simulate_section(self, query: str, node: TreeNode) -> float:
        prompt = f"Query: {query}\nSection: {node.title}\nPages: {node.start_page+1}-{node.end_page+1}\nSummary: {node.summary}\nKeywords: {', '.join(node.keywords) if node.keywords else 'none'}\nRelevance?"
        response = self.llm.complete(prompt=prompt, model=self.search_model, system_prompt=SYSTEM_PROMPT_SIMULATE, json_mode=True, max_tokens=256)
        try:
            return max(0.0, min(1.0, float(json.loads(response).get("relevance_score", 0.0))))
        except (json.JSONDecodeError, ValueError):
            return 0.3

    def _backpropagate(self, node: TreeNode, reward: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent

    def _should_stop_early(self, root: TreeNode) -> bool:
        leaves = self._get_all_leaves(root)
        return len([l for l in leaves if l.visit_count >= 3 and l.average_reward >= self.config.confidence_threshold]) >= self.config.top_k_results

    def _collect_results(self, root, doc_filename="", doc_id=""):
        all_nodes = []
        def _collect(n):
            if n.visit_count > 0:
                all_nodes.append(n)
            for c in n.children:
                _collect(c)
        _collect(root)
        all_nodes.sort(key=lambda n: n.average_reward * math.log(n.visit_count + 1), reverse=True)
        results, seen = [], set()
        for node in all_nodes:
            if len(results) >= self.config.top_k_results:
                break
            pr = (node.start_page, node.end_page)
            if any(pr[0] <= e and pr[1] >= s for s, e in seen):
                continue
            seen.add(pr)
            results.append(SearchResult(node=node, relevance_score=node.average_reward,
                visit_count=node.visit_count, document_filename=doc_filename, document_id=doc_id))
        return results

    def _get_all_leaves(self, node):
        if node.is_leaf:
            return [node]
        leaves = []
        for c in node.children:
            leaves.extend(self._get_all_leaves(c))
        return leaves
