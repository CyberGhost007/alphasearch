"""
TreeRAG Pipeline — orchestrates two-phase search with graceful error handling.
Handles: missing PDFs during deep read, search on empty folders, corrupt indices.
"""

import json
import time
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dataclasses import dataclass, field

from .config import TreeRAGConfig
from .models import DocumentIndex, FolderIndex, SearchResult, QueryResult, SearchStats
from .llm_client import LLMClient
from .indexer import Indexer
from .mcts import MCTSEngine, SYSTEM_PROMPT_DEEP_READ, SYSTEM_PROMPT_ANSWER
from .folder_manager import FolderManager
from .router import RouterAgent
from .pdf_processor import PDFProcessor
from .exceptions import EmptyFolderError, FolderNotFoundError


@dataclass
class ChatMessage:
    """Single message in the chat history."""
    role: str                       # "user" | "assistant" | "system"
    content: str
    folder_name: str = ""           # Which folder this message relates to
    sources: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_history_entry(self) -> dict:
        return {"role": self.role, "content": self.content}

console = Console()


class TreeRAGPipeline:
    """
    End-to-end pipeline.
    
    Folder search:
        pipeline.folder.create_folder("project")
        pipeline.folder.add_document("project", "doc.pdf")
        result = pipeline.query_folder("question?", "project")
    
    Single doc:
        doc_idx = pipeline.index("doc.pdf")
        result = pipeline.query_document("question?", doc_idx)
    """

    def __init__(self, config: TreeRAGConfig):
        self.config = config
        self._llm = None
        self._indexer = None
        self._mcts = None
        self._router = None
        self.pdf_processor = PDFProcessor(dpi=config.indexer.image_dpi)

        # Folder manager can be used without API key for CRUD operations
        # It lazily gets LLM when indexing is needed
        self._folder = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMClient(self.config.model)
        return self._llm

    @property
    def indexer(self):
        if self._indexer is None:
            self._indexer = Indexer(self.config, self.llm)
        return self._indexer

    @property
    def mcts(self):
        if self._mcts is None:
            self._mcts = MCTSEngine(self.config.mcts, self.llm, search_model=self.config.model.search_model)
        return self._mcts

    @property
    def folder(self):
        if self._folder is None:
            self._folder = FolderManager(self.config, self)
        return self._folder

    @property
    def router(self):
        if self._router is None:
            self._router = RouterAgent(self.config, self.llm, self.folder)
        return self._router

    # =========================================================================
    # Unified Chat — Phase 0 (Route) → Phase 1 → Phase 2 → Answer
    # =========================================================================

    def chat(
        self,
        query: str,
        chat_history: list[ChatMessage] = None,
        use_vision: bool = True,
    ) -> ChatMessage:
        """
        Unified chat interface. Automatically routes to the right folder,
        searches, and returns an answer.
        
        Flow:
        1. Phase 0 — Router Agent picks the best folder based on query + history
        2. Phase 1 — MCTS selects relevant documents in that folder
        3. Phase 2 — MCTS finds relevant sections (parallel)
        4. Deep read + answer generation
        
        Args:
            query: User's question
            chat_history: Previous ChatMessages for context-aware routing
            use_vision: Use page images for deep read
            
        Returns:
            ChatMessage with answer, folder_name, sources, and stats
        """
        start_time = time.time()
        llm_before = self.llm.total_calls

        # Convert ChatMessages to history format for router
        history_entries = []
        if chat_history:
            history_entries = [m.to_history_entry() for m in chat_history[-8:]]

        # Phase 0: Route to folder
        console.print(f"\n[bold cyan]Phase 0 — Routing...[/bold cyan]")
        route_result = self.router.route(query, history_entries)

        # Handle no folders
        if route_result.no_folders_available:
            return ChatMessage(
                role="assistant",
                content="No folders available. Create a folder and add documents first.",
                stats={"latency": f"{time.time() - start_time:.1f}s"},
            )

        # Handle needs clarification
        if route_result.needs_clarification:
            return ChatMessage(
                role="assistant",
                content=route_result.clarification_question or "Could you clarify which project or topic you're asking about?",
                stats={"latency": f"{time.time() - start_time:.1f}s", "phase": "routing"},
            )

        # Handle no match
        if not route_result.has_match:
            return ChatMessage(
                role="assistant",
                content="I couldn't find a relevant folder for your question. Could you be more specific about which project or topic?",
                stats={"latency": f"{time.time() - start_time:.1f}s"},
            )

        # Route to the top folder
        folder_name = route_result.top_folder
        confidence = route_result.top_confidence
        phase0_time = route_result.latency

        console.print(f"  [green]→ {folder_name}[/green] (confidence: {confidence:.0%})")

        # Phase 1 + 2 + Deep Read + Answer (reuse existing query_folder)
        try:
            result = self.query_folder(query, folder_name, use_vision=use_vision)
        except EmptyFolderError:
            return ChatMessage(
                role="assistant",
                content=f"The folder '{folder_name}' is empty. Add documents to it first.",
                folder_name=folder_name,
                stats={"latency": f"{time.time() - start_time:.1f}s"},
            )
        except Exception as e:
            return ChatMessage(
                role="assistant",
                content=f"Search error: {str(e)}",
                folder_name=folder_name,
                stats={"latency": f"{time.time() - start_time:.1f}s"},
            )

        # Build ChatMessage response
        total_time = time.time() - start_time
        total_calls = self.llm.total_calls - llm_before

        source_entries = []
        for s in result.sources:
            source_entries.append({
                "document": s.document_filename,
                "section": s.node.title,
                "pages": s.node.page_range,
                "score": round(s.relevance_score, 3),
            })

        return ChatMessage(
            role="assistant",
            content=result.answer,
            folder_name=folder_name,
            sources=source_entries,
            stats={
                "folder": folder_name,
                "folder_confidence": f"{confidence:.0%}",
                "phase0_time": f"{phase0_time:.1f}s",
                "phase1_time": f"{result.phase1_time:.1f}s",
                "phase2_time": f"{result.phase2_time:.1f}s",
                "total_time": f"{total_time:.1f}s",
                "llm_calls": total_calls,
                "mcts_iterations": result.total_mcts_iterations,
                "cost": f"${self.llm._estimate_cost():.4f}",
                "docs_searched": result.documents_searched,
                "phase1_search_stats": result.phase1_stats.to_dict() if result.phase1_stats else None,
                "phase2_search_stats": [s.to_dict() for s in result.phase2_stats] if result.phase2_stats else [],
            },
        )

    # =========================================================================
    # Folder Query (Two-Phase) — called by chat() or directly
    # =========================================================================

    def query_folder(self, query: str, folder_name: str, use_vision: bool = True) -> QueryResult:
        start_time = time.time()
        llm_before = self.llm.total_calls

        # Load and validate folder
        folder_index = self.folder.load_folder(folder_name)
        if not folder_index.documents:
            raise EmptyFolderError(
                f"Folder '{folder_name}' has no documents. "
                f"Add documents first: folder add {folder_name} file.pdf"
            )

        console.print(Panel(
            f"[bold]{query}[/bold]\n\nSearching: {folder_name} ({folder_index.total_documents} docs, {folder_index.total_pages} pages)",
            title="Query", border_style="cyan",
        ))

        # Phase 1
        p1_start = time.time()
        doc_scores, phase1_stats = self.mcts.search_meta(query, folder_index)
        p1_time = time.time() - p1_start

        if not doc_scores:
            return QueryResult(
                query=query, answer="No relevant documents found in this folder.",
                latency_seconds=time.time() - start_time, phase1_time=p1_time,
                phase1_stats=phase1_stats,
            )

        # Load selected document indices (gracefully skips broken ones)
        selected_entries = [ds.entry for ds in doc_scores]
        doc_indices = self.folder.load_document_indices(selected_entries)

        if not doc_indices:
            return QueryResult(
                query=query,
                answer="Selected documents have missing or corrupt indices. Run 'folder refresh' to repair.",
                documents_searched=[e.filename for e in selected_entries],
                latency_seconds=time.time() - start_time, phase1_time=p1_time,
                phase1_stats=phase1_stats,
            )

        # Phase 2
        p2_start = time.time()
        if self.config.mcts.parallel_phase2 and len(doc_indices) > 1:
            search_results, phase2_stats = self.mcts.search_documents_parallel(query, doc_indices)
        else:
            search_results, phase2_stats = self.mcts.search_documents_sequential(query, doc_indices)
        p2_time = time.time() - p2_start

        if not search_results:
            return QueryResult(
                query=query, answer="No relevant sections found in selected documents.",
                documents_searched=[e.filename for e in selected_entries],
                latency_seconds=time.time() - start_time,
                phase1_time=p1_time, phase2_time=p2_time,
                phase1_stats=phase1_stats, phase2_stats=phase2_stats,
            )

        # Deep Read
        console.print("\n[bold]Deep reading matched sections...[/bold]")
        search_results = self._deep_read(query, search_results, folder_name, use_vision)

        # Answer
        console.print("\n[bold]Generating answer...[/bold]")
        answer = self._generate_answer(query, search_results)

        total_iters = (phase1_stats.iterations_used if phase1_stats else 0) + sum(s.iterations_used for s in phase2_stats)
        result = QueryResult(
            query=query, answer=answer, sources=search_results,
            documents_searched=[e.filename for e in selected_entries],
            total_mcts_iterations=total_iters,
            total_llm_calls=self.llm.total_calls - llm_before,
            latency_seconds=time.time() - start_time,
            phase1_time=p1_time, phase2_time=p2_time,
            phase1_stats=phase1_stats, phase2_stats=phase2_stats,
        )
        self._print_result(result)
        return result

    # =========================================================================
    # Single-Document Query
    # =========================================================================

    def query_document(self, query: str, doc_index: DocumentIndex, use_vision: bool = True) -> QueryResult:
        start_time = time.time()
        llm_before = self.llm.total_calls

        console.print(Panel(
            f"[bold]{query}[/bold]\n\nSearching: {doc_index.filename} ({doc_index.total_pages} pages)",
            title="Query", border_style="cyan",
        ))

        search_results, doc_stats = self.mcts.search_document(query, doc_index)

        if not search_results:
            return QueryResult(
                query=query, answer="No relevant sections found.",
                documents_searched=[doc_index.filename],
                latency_seconds=time.time() - start_time,
                phase2_stats=[doc_stats],
            )

        console.print("\n[bold]Deep reading...[/bold]")
        search_results = self._deep_read_single(query, search_results, doc_index, use_vision)

        console.print("\n[bold]Generating answer...[/bold]")
        answer = self._generate_answer(query, search_results)

        result = QueryResult(
            query=query, answer=answer, sources=search_results,
            documents_searched=[doc_index.filename],
            total_mcts_iterations=doc_stats.iterations_used,
            total_llm_calls=self.llm.total_calls - llm_before,
            latency_seconds=time.time() - start_time,
            phase2_stats=[doc_stats],
        )
        self._print_result(result)
        return result

    # =========================================================================
    # Index helpers
    # =========================================================================

    def index(self, pdf_path, save_path=None):
        return self.indexer.index_document(pdf_path, save_path)

    def load_index(self, index_path):
        doc_index = DocumentIndex.load(index_path)
        if doc_index.root:
            self._set_parent_refs(doc_index.root)
        return doc_index

    # =========================================================================
    # Deep Read — with graceful PDF-missing handling
    # =========================================================================

    def _deep_read(self, query, results, folder_name, use_vision=True):
        """Deep read for folder-level search. Finds PDFs via folder entries."""
        for result in results:
            pdf_path = self._find_pdf(result, folder_name)
            if not pdf_path:
                console.print(f"  [yellow]Skipping deep read for {result.node.title}: PDF missing[/yellow]")
                continue
            self._deep_read_node(query, result, pdf_path, use_vision)

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def _deep_read_single(self, query, results, doc_index, use_vision=True):
        """Deep read for single-document search."""
        pdf_path = doc_index.pdf_path
        if not pdf_path or not Path(pdf_path).exists():
            console.print("[yellow]PDF not found — skipping deep read, using summaries only[/yellow]")
            return results

        for result in results:
            self._deep_read_node(query, result, pdf_path, use_vision)

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def _deep_read_node(self, query, result, pdf_path, use_vision):
        """Deep read a single search result node."""
        node = result.node
        try:
            pdf = self.pdf_processor.load(pdf_path)
        except Exception as e:
            console.print(f"  [yellow]Cannot open PDF for deep read: {e}[/yellow]")
            return

        try:
            if use_vision:
                images = pdf.get_page_images_batch(node.start_page, node.end_page)
                if not images:
                    pdf.close()
                    return
                prompt = f"Query: {query}\nSection: {node.title} (Pages {node.start_page+1}-{node.end_page+1})\nDocument: {result.document_filename}\nExtract ALL relevant information."
                response = self.llm.complete_with_images(
                    prompt=prompt, images=images, model=self.config.model.answer_model,
                    system_prompt=SYSTEM_PROMPT_DEEP_READ, json_mode=True, max_tokens=4096,
                )
            else:
                text = pdf.get_pages_text_batch(node.start_page, node.end_page)
                prompt = f"Query: {query}\nSection: {node.title} (Pages {node.start_page+1}-{node.end_page+1})\nDocument: {result.document_filename}\nContent:\n{text}\n\nExtract ALL relevant information."
                response = self.llm.complete(
                    prompt=prompt, model=self.config.model.answer_model,
                    system_prompt=SYSTEM_PROMPT_DEEP_READ, json_mode=True, max_tokens=4096,
                )

            r = json.loads(response)
            result.content = r.get("relevant_content", "")
            result.reasoning = json.dumps(r.get("key_facts", []))
            if not r.get("relevance_confirmed", True):
                result.relevance_score *= 0.3

        except json.JSONDecodeError:
            result.content = response if isinstance(response, str) else ""
        except Exception as e:
            console.print(f"  [yellow]Deep read failed for {node.title}: {e}[/yellow]")
        finally:
            pdf.close()

    def _find_pdf(self, result, folder_name):
        """Find PDF path for a search result via folder entries."""
        try:
            fi = self.folder.load_folder(folder_name)
            entry = fi.get_document_by_id(result.document_id)
            if entry and entry.pdf_path and Path(entry.pdf_path).exists():
                return entry.pdf_path
            # Try by filename
            entry = fi.get_document(result.document_filename)
            if entry and entry.pdf_path and Path(entry.pdf_path).exists():
                return entry.pdf_path
        except Exception:
            pass
        return None

    # =========================================================================
    # Answer Generation
    # =========================================================================

    def _generate_answer(self, query, results):
        source_texts = []
        for i, r in enumerate(results):
            src = f"--- Source {i+1}: {r.document_filename} → {r.node.title} ({r.node.page_range}) ---\n"
            src += f"{r.content or r.node.summary}\nRelevance: {r.relevance_score:.2f}"
            source_texts.append(src)

        prompt = f"Query: {query}\n\nSource Material:\n{chr(10).join(source_texts)}\n\nAnswer using ONLY the sources above. Cite as [filename → Section, pages]."
        return self.llm.complete(prompt=prompt, model=self.config.model.answer_model, system_prompt=SYSTEM_PROMPT_ANSWER, max_tokens=4096)

    # =========================================================================
    # Display
    # =========================================================================

    def _print_result(self, result):
        console.print()
        console.print(Panel(result.answer, title="[bold green]Answer[/bold green]", border_style="green"))

        if result.sources:
            table = Table(title="Sources", show_header=True)
            table.add_column("Document", style="blue")
            table.add_column("Section", style="cyan")
            table.add_column("Pages", style="dim")
            table.add_column("Score", justify="right")
            for s in result.sources:
                table.add_row(s.document_filename, s.node.title, s.node.page_range, f"{s.relevance_score:.3f}")
            console.print(table)

        timing = f"Phase 1: {result.phase1_time:.1f}s | Phase 2: {result.phase2_time:.1f}s | " if result.phase1_time else ""
        console.print(f"\n[dim]{timing}Total: {result.latency_seconds:.1f}s | LLM calls: {result.total_llm_calls} | Iterations: {result.total_mcts_iterations} | Cost: ~${self.llm._estimate_cost():.4f}[/dim]")
        if result.documents_searched:
            console.print(f"[dim]Searched: {', '.join(result.documents_searched)}[/dim]")

        # Adaptive search stats
        if result.phase1_stats and result.phase1_stats.iterations_used > 0:
            s = result.phase1_stats
            conv = f"converged@{s.convergence_iteration+1} ({s.convergence_reason})" if s.converged else f"{s.iterations_used} iters"
            console.print(f"[dim]Phase 1: {conv}, {s.pruned_branches} docs pruned, coverage: {s.coverage_pct:.0f}%[/dim]")
        for i, s in enumerate(result.phase2_stats):
            if s.iterations_used > 0:
                conv = f"converged@{s.convergence_iteration+1} ({s.convergence_reason})" if s.converged else f"{s.iterations_used} iters"
                console.print(f"[dim]Phase 2 doc {i+1}: {conv}, coverage: {s.coverage_pct:.0f}%, pruned: {s.pruned_branches}[/dim]")

    def _set_parent_refs(self, node):
        for c in node.children:
            c.parent = node
            self._set_parent_refs(c)

    @property
    def usage(self):
        return self.llm.usage_summary
