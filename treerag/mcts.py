"""
Two-Phase MCTS Engine with Adaptive Search.
Phase 1: Score document summaries → pick top-K docs
Phase 2: Search within selected docs → find sections (parallel)

Adaptive features:
- Dynamic iteration bounds (min/max instead of fixed)
- Multi-signal convergence detection (top-k stability, variance stability, confidence)
- Branch pruning (skip consistently low-scoring subtrees)
- Exploration constant decay (high C → low C over search lifetime)
"""

import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from rich.console import Console
from .config import MCTSConfig
from .models import TreeNode, DocumentIndex, FolderIndex, FolderDocEntry, SearchResult, DocumentScore, SearchStats
from .llm_client import LLMClient

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
    # Adaptive helpers
    # =========================================================================

    def _get_exploration_constant(self, iteration: int, max_iterations: int) -> float:
        """Compute decaying exploration constant for this iteration."""
        if not self.config.adaptive or not self.config.adaptive_exploration:
            return self.config.exploration_constant

        progress = iteration / max(max_iterations - 1, 1)
        c_start = self.config.exploration_start
        c_end = self.config.exploration_end

        if self.config.exploration_decay == "cosine":
            return c_end + (c_start - c_end) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return c_start + (c_end - c_start) * progress

    def _check_convergence(self, root: TreeNode, iteration: int, tracker: dict) -> tuple[bool, str]:
        """Multi-signal convergence detection.

        Signals (any one triggers stop):
        1. Top-k node IDs unchanged for top_k_stable_rounds consecutive iterations
        2. Reward variance stabilized across convergence_window
        3. Original confidence check (3+ leaves with high scores)
        """
        if not self.config.adaptive:
            if self._should_stop_early_legacy(root):
                return True, "confidence_met"
            return False, ""

        min_iters = tracker.get("min_iterations", self.config.min_iterations)
        if iteration < min_iters:
            return False, ""

        # Collect visited, non-pruned nodes
        all_nodes = []
        def _collect_visited(n):
            if n.visit_count > 0 and not n.pruned:
                all_nodes.append(n)
            for c in n.children:
                if not c.pruned:
                    _collect_visited(c)
        _collect_visited(root)

        # Signal 1: Top-K stability
        all_nodes.sort(key=lambda n: n.average_reward * math.log(n.visit_count + 1), reverse=True)
        current_top_k = tuple(n.node_id for n in all_nodes[:self.config.top_k_results])

        prev_top_k = tracker.get("prev_top_k")
        if prev_top_k == current_top_k:
            tracker["top_k_stable_count"] = tracker.get("top_k_stable_count", 0) + 1
        else:
            tracker["top_k_stable_count"] = 0
        tracker["prev_top_k"] = current_top_k

        if tracker["top_k_stable_count"] >= self.config.top_k_stable_rounds:
            return True, "top_k_stable"

        # Signal 2: Variance stability
        rewards = [n.average_reward for n in all_nodes if n.visit_count > 0]
        if len(rewards) >= 2:
            current_variance = statistics.variance(rewards)
            variance_history = tracker.setdefault("variance_history", [])
            variance_history.append(current_variance)

            window = self.config.convergence_window
            if len(variance_history) >= window * 2:
                old_window = variance_history[-(window * 2):-window]
                new_window = variance_history[-window:]
                old_mean = sum(old_window) / len(old_window)
                new_mean = sum(new_window) / len(new_window)
                if abs(new_mean - old_mean) < self.config.convergence_variance_threshold:
                    return True, "variance_stable"

        # Signal 3: Original confidence check
        leaves = self._get_all_leaves(root)
        confident = [l for l in leaves if l.visit_count >= 3
                     and l.average_reward >= self.config.confidence_threshold
                     and not l.pruned]
        if len(confident) >= self.config.top_k_results:
            return True, "confidence_met"

        return False, ""

    def _apply_virtual_loss(self, node: TreeNode):
        """Add a fake visit with 0 reward to discourage re-selection in the same batch."""
        current = node
        while current is not None:
            current.visit_count += 1
            current = current.parent

    def _remove_virtual_loss(self, node: TreeNode):
        """Remove the fake visit added by _apply_virtual_loss."""
        current = node
        while current is not None:
            current.visit_count -= 1
            current = current.parent

    def _simulate_batch(self, query: str, targets: list[TreeNode]) -> list[float]:
        """Run multiple simulations in parallel using threads."""
        if len(targets) == 1:
            try:
                return [self._simulate_section(query, targets[0])]
            except Exception:
                return [0.3]

        results = [0.3] * len(targets)
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            futures = {executor.submit(self._simulate_section, query, t): i for i, t in enumerate(targets)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=30)
                except Exception:
                    results[idx] = 0.3
        return results

    def _prune_branches(self, root: TreeNode, iteration: int, stats: SearchStats):
        """Mark consistently low-scoring internal nodes as pruned."""
        if not self.config.adaptive or not self.config.pruning_enabled:
            return

        def _prune_recursive(node: TreeNode):
            if node.pruned:
                return
            if (node.visit_count >= self.config.pruning_min_visits
                    and node.average_reward < self.config.pruning_reward_threshold
                    and not node.is_leaf):
                node.pruned = True
                stats.pruned_branches += 1
                stats.pruned_at_iterations.append(iteration)
                # Mark entire subtree
                def _mark_subtree(n):
                    n.pruned = True
                    for c in n.children:
                        _mark_subtree(c)
                for c in node.children:
                    _mark_subtree(c)
                return
            for c in node.children:
                _prune_recursive(c)

        _prune_recursive(root)

    # =========================================================================
    # Phase 1: Meta-tree search
    # =========================================================================

    def search_meta(self, query: str, folder_index: FolderIndex, verbose: bool = True) -> tuple[list[DocumentScore], SearchStats]:
        if not folder_index.documents:
            return [], SearchStats()

        folder_index.reset_mcts_state()
        total_visits = 0
        num_docs = len(folder_index.documents)

        if self.config.adaptive:
            min_iter = min(self.config.meta_min_iterations, num_docs * 2)
            max_iter = min(self.config.meta_max_iterations, num_docs * 5)
        else:
            max_iter = min(self.config.meta_iterations, num_docs * 5)
            min_iter = max_iter

        stats = SearchStats(
            iterations_max=max_iter,
            total_nodes=num_docs,
            exploration_start=self.config.exploration_start if self.config.adaptive_exploration else self.config.exploration_constant,
        )

        if verbose:
            console.print(f"\n[bold cyan]Phase 1 — Document Selection[/bold cyan]")
            console.print(f"  Scoring {num_docs} documents (up to {max_iter} iterations)")

        convergence_tracker = {"min_iterations": min_iter}

        for i in range(max_iter):
            c = self._get_exploration_constant(i, max_iter)

            active_docs = [d for d in folder_index.documents if not d.pruned]
            if not active_docs:
                break

            selected = max(active_docs, key=lambda d: d.ucb1(total_visits, c))
            try:
                score = self._simulate_document(query, selected)
            except Exception:
                score = 0.3
            selected.visit_count += 1
            selected.total_reward += score
            total_visits += 1

            # Prune low-scoring docs
            if self.config.adaptive and self.config.pruning_enabled and i >= min_iter and i % 3 == 0:
                for d in folder_index.documents:
                    if (not d.pruned
                            and d.visit_count >= self.config.pruning_min_visits
                            and d.average_reward < self.config.pruning_reward_threshold):
                        d.pruned = True
                        stats.pruned_branches += 1
                        stats.pruned_at_iterations.append(i)

            # Convergence: top-k doc stability
            if self.config.adaptive and i >= min_iter:
                scored_docs = sorted(
                    [d for d in folder_index.documents if d.visit_count > 0 and not d.pruned],
                    key=lambda d: d.average_reward, reverse=True,
                )
                current_top = tuple(d.document_id for d in scored_docs[:self.config.top_k_documents])
                prev_top = convergence_tracker.get("prev_top_docs")
                if prev_top == current_top:
                    convergence_tracker["doc_stable_count"] = convergence_tracker.get("doc_stable_count", 0) + 1
                else:
                    convergence_tracker["doc_stable_count"] = 0
                convergence_tracker["prev_top_docs"] = current_top

                if convergence_tracker["doc_stable_count"] >= self.config.top_k_stable_rounds:
                    stats.converged = True
                    stats.convergence_reason = "top_k_stable"
                    stats.convergence_iteration = i
                    stats.iterations_used = i + 1
                    if verbose:
                        console.print(f"    [dim]Converged at iteration {i+1}/{max_iter} ({stats.convergence_reason})[/dim]")
                    break
        else:
            stats.iterations_used = max_iter
            stats.convergence_reason = "max_reached"

        if not stats.iterations_used:
            stats.iterations_used = total_visits

        # Final stats
        visited_docs = [d for d in folder_index.documents if d.visit_count > 0]
        stats.visited_nodes = len(visited_docs)
        stats.coverage_pct = (len(visited_docs) / num_docs * 100) if num_docs else 0.0
        if visited_docs:
            rewards = [d.average_reward for d in visited_docs]
            stats.mean_reward = statistics.mean(rewards)
            stats.reward_variance = statistics.variance(rewards) if len(rewards) > 1 else 0.0

        scored = sorted(
            [DocumentScore(entry=d, relevance_score=d.average_reward, visit_count=d.visit_count)
             for d in folder_index.documents if d.visit_count > 0 and not d.pruned],
            key=lambda x: x.relevance_score, reverse=True,
        )
        top_docs = scored[:self.config.top_k_documents]

        if verbose:
            for d in top_docs:
                console.print(f"    → {d.entry.filename} (score: {d.relevance_score:.3f}, visits: {d.visit_count})")
            if stats.converged:
                console.print(f"    [dim]{stats.iterations_used}/{max_iter} iters, {stats.pruned_branches} docs pruned[/dim]")

        stats.exploration_end = c if total_visits > 0 else stats.exploration_start
        return top_docs, stats

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

    def search_document(self, query: str, doc_index: DocumentIndex, verbose: bool = True) -> tuple[list[SearchResult], SearchStats]:
        if not doc_index.root:
            return [], SearchStats()

        root = doc_index.root
        root.reset_mcts_state()

        if self.config.adaptive:
            min_iter = self.config.min_iterations
            max_iter = self.config.max_iterations
        else:
            min_iter = self.config.iterations
            max_iter = self.config.iterations

        all_nodes = doc_index.get_all_nodes()
        stats = SearchStats(
            iterations_max=max_iter,
            total_nodes=len(all_nodes),
            exploration_start=self.config.exploration_start if self.config.adaptive_exploration else self.config.exploration_constant,
        )

        batch_size = self.config.simulation_batch_size

        if verbose:
            mode = f"batch={batch_size}" if batch_size > 1 else "sequential"
            console.print(f"\n  [dim]Searching: {doc_index.filename} (up to {max_iter} iterations, {mode})[/dim]")

        convergence_tracker = {"min_iterations": min_iter}
        c = self.config.exploration_constant
        i = 0
        converged = False

        while i < max_iter:
            c = self._get_exploration_constant(i, max_iter)

            # Select up to batch_size nodes using virtual loss for diversity
            batch_targets = []
            remaining = min(batch_size, max_iter - i)
            for _ in range(remaining):
                selected = self._select(root, exploration_c=c)
                expanded = self._expand(selected)
                target = expanded or selected

                if target.pruned:
                    break

                self._apply_virtual_loss(target)
                batch_targets.append(target)

            if not batch_targets:
                break

            # Parallel simulate all targets
            rewards = self._simulate_batch(query, batch_targets)

            # Remove virtual loss, then backpropagate real rewards
            for target, reward in zip(batch_targets, rewards):
                self._remove_virtual_loss(target)
                self._backpropagate(target, reward)

            i += len(batch_targets)

            # Periodic pruning
            if i >= min_iter and i % 5 == 0:
                self._prune_branches(root, i, stats)

            # Convergence check
            should_stop, reason = self._check_convergence(root, i, convergence_tracker)
            if should_stop:
                stats.converged = True
                stats.convergence_reason = reason
                stats.convergence_iteration = i
                stats.iterations_used = i
                if verbose:
                    console.print(f"    [dim]Converged at iteration {i}/{max_iter} ({reason})[/dim]")
                converged = True
                break

        if not converged:
            stats.iterations_used = i if i else max_iter
            stats.convergence_reason = "max_reached"

        # Final stats
        visited = [n for n in all_nodes if n.visit_count > 0]
        stats.visited_nodes = len(visited)
        stats.coverage_pct = (len(visited) / len(all_nodes) * 100) if all_nodes else 0.0
        stats.exploration_end = c

        if visited:
            rewards = [n.average_reward for n in visited]
            stats.mean_reward = statistics.mean(rewards)
            stats.reward_variance = statistics.variance(rewards) if len(rewards) > 1 else 0.0

        results = self._collect_results(root, doc_index.filename, doc_index.document_id)
        return results, stats

    def search_documents_parallel(self, query: str, doc_indices: list[DocumentIndex], verbose: bool = True) -> tuple[list[SearchResult], list[SearchStats]]:
        """Phase 2 parallel. Gracefully handles per-document failures."""
        all_results = []
        all_stats = []
        if verbose:
            console.print(f"\n[bold cyan]Phase 2 — Searching {len(doc_indices)} documents (parallel)[/bold cyan]")

        with ThreadPoolExecutor(max_workers=min(len(doc_indices), 4)) as executor:
            futures = {executor.submit(self.search_document, query, idx, False): idx for idx in doc_indices}
            for future in as_completed(futures):
                doc_idx = futures[future]
                try:
                    results, stats = future.result(timeout=120)
                    if verbose:
                        status = f"converged@{stats.convergence_iteration}" if stats.converged else f"{stats.iterations_used}iters"
                        console.print(f"    {doc_idx.filename}: {len(results)} sections ({status}, {stats.pruned_branches} pruned)")
                    all_results.extend(results)
                    all_stats.append(stats)
                except TimeoutError:
                    console.print(f"    [yellow]{doc_idx.filename}: Timed out — skipping[/yellow]")
                    all_stats.append(SearchStats(convergence_reason="timeout"))
                except Exception as e:
                    console.print(f"    [red]{doc_idx.filename}: Error — {e}[/red]")
                    all_stats.append(SearchStats(convergence_reason="error"))

        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return all_results[:self.config.top_k_results], all_stats

    def search_documents_sequential(self, query: str, doc_indices: list[DocumentIndex], verbose: bool = True) -> tuple[list[SearchResult], list[SearchStats]]:
        all_results = []
        all_stats = []
        if verbose:
            console.print(f"\n[bold cyan]Phase 2 — Searching {len(doc_indices)} documents[/bold cyan]")
        for idx in doc_indices:
            try:
                results, stats = self.search_document(query, idx, verbose)
                all_results.extend(results)
                all_stats.append(stats)
            except Exception as e:
                console.print(f"    [red]{idx.filename}: Error — {e}[/red]")
                all_stats.append(SearchStats(convergence_reason="error"))
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return all_results[:self.config.top_k_results], all_stats

    # =========================================================================
    # Core MCTS
    # =========================================================================

    def _select(self, node: TreeNode, exploration_c: float = None) -> TreeNode:
        c = exploration_c if exploration_c is not None else self.config.exploration_constant
        current, depth = node, 0
        while not current.is_leaf and depth < self.config.max_depth:
            active_children = [ch for ch in current.children if not ch.pruned]
            if not active_children:
                break
            unvisited = [ch for ch in active_children if ch.visit_count == 0]
            if unvisited:
                return current
            current = max(active_children, key=lambda ch: ch.ucb1(c))
            depth += 1
        return current

    def _expand(self, node: TreeNode) -> Optional[TreeNode]:
        if node.is_leaf:
            return None
        unvisited = [c for c in node.children if c.visit_count == 0 and not c.pruned]
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

    def _should_stop_early_legacy(self, root: TreeNode) -> bool:
        """Original early stopping logic, used when adaptive=False."""
        leaves = self._get_all_leaves(root)
        return len([l for l in leaves if l.visit_count >= 3 and l.average_reward >= self.config.confidence_threshold]) >= self.config.top_k_results

    def _collect_results(self, root, doc_filename="", doc_id=""):
        all_nodes = []
        def _collect(n):
            if n.visit_count > 0 and not n.pruned:
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
