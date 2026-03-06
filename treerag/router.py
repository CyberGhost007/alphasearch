"""
Router Agent — Phase 0: Automatic folder selection.

Given a user query and chat history, scores all folder summaries
to determine which folder(s) to search. Uses chat history for
context-aware routing (follow-up queries use previous folder context).

Pipeline position:
  Query → PHASE 0 (Router) → PHASE 1 (Doc Selection) → PHASE 2 (Section Search)
"""

import json
import time
from typing import Optional
from rich.console import Console

from .config import TreeRAGConfig
from .models import FolderIndex
from .llm_client import LLMClient
from .folder_manager import FolderManager

console = Console()


SYSTEM_PROMPT_ROUTE = """You are a folder routing agent. Given a user query, chat history, and a list of available folders with their descriptions, determine which folder(s) are most relevant.

Output JSON only:
{
  "selected_folders": [
    {"folder_name": "exact folder name", "confidence": <0.0 to 1.0>, "reasoning": "one sentence"}
  ],
  "needs_clarification": false,
  "clarification_question": null
}

Rules:
- Select 1-2 folders maximum. Only select 2 if the query explicitly spans multiple projects/topics.
- If the chat history shows the user was just discussing a specific folder, strongly prefer that folder for follow-up queries.
- If no folder is relevant (e.g. general questions unrelated to any folder), return empty selected_folders.
- If the query is ambiguous and could match multiple folders equally, set needs_clarification=true and provide a question.
- Confidence: 0.9+ for clear matches, 0.5-0.8 for reasonable guesses, below 0.5 means uncertain.
- Match based on content relevance, not just keyword overlap."""


SYSTEM_PROMPT_FOLDER_SUMMARY = """You are summarizing a document folder. Given the folder name and its document summaries, create a concise folder-level summary.

Output JSON:
{
  "summary": "2-3 sentences describing what this folder contains, its purpose, key topics, and scope",
  "keywords": ["keyword1", "keyword2", ..., "keyword8"]
}

The summary should help a routing agent decide if a user's query is relevant to this folder."""


class RouterAgent:
    """
    Routes user queries to the most relevant folder(s).
    
    Uses chat history for context-aware follow-up routing:
    - "What was the budget?" → routes to the relevant project folder
    - "What about the timeline?" → uses history to stay in the same folder
    
    Generates and caches folder-level summaries for scoring.
    """

    def __init__(self, config: TreeRAGConfig, llm: LLMClient, folder_manager: FolderManager):
        self.config = config
        self.llm = llm
        self.folder_manager = folder_manager
        
        # Cache folder summaries: {folder_name: {"summary": ..., "keywords": [...]}}
        self._folder_summaries: dict[str, dict] = {}

    def route(
        self,
        query: str,
        chat_history: list[dict] = None,
        verbose: bool = True,
    ) -> "RouteResult":
        """
        Route a query to the best folder(s).
        
        Args:
            query: User's question
            chat_history: List of {"role": "user"|"assistant", "content": "..."} messages
            verbose: Print routing info
            
        Returns:
            RouteResult with selected folders, confidence, and reasoning
        """
        start_time = time.time()
        
        # Get all available folders
        folder_names = self.folder_manager.list_folders()
        if not folder_names:
            return RouteResult(
                selected_folders=[],
                needs_clarification=False,
                no_folders_available=True,
                latency=time.time() - start_time,
            )

        # Build folder descriptions
        folder_descriptions = self._get_folder_descriptions(folder_names)
        
        if not folder_descriptions:
            return RouteResult(
                selected_folders=[],
                needs_clarification=False,
                latency=time.time() - start_time,
            )

        # Build the routing prompt with chat history
        prompt = self._build_route_prompt(query, folder_descriptions, chat_history)

        # Ask LLM to route
        response = self.llm.complete(
            prompt=prompt,
            model=self.config.model.search_model,  # Use cheap model for routing
            system_prompt=SYSTEM_PROMPT_ROUTE,
            json_mode=True,
            max_tokens=512,
        )

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: if parsing fails, try to match by keyword
            return self._fallback_route(query, folder_names, time.time() - start_time)

        # Build RouteResult
        selected = []
        for entry in result.get("selected_folders", []):
            fname = entry.get("folder_name", "")
            # Validate folder name exists
            if fname in folder_names:
                selected.append(SelectedFolder(
                    folder_name=fname,
                    confidence=float(entry.get("confidence", 0.5)),
                    reasoning=entry.get("reasoning", ""),
                ))

        route_result = RouteResult(
            selected_folders=selected,
            needs_clarification=result.get("needs_clarification", False),
            clarification_question=result.get("clarification_question"),
            latency=time.time() - start_time,
        )

        if verbose and selected:
            for sf in selected:
                console.print(
                    f"  [dim]Routed → {sf.folder_name} "
                    f"(confidence: {sf.confidence:.0%})[/dim]"
                )

        return route_result

    def _get_folder_descriptions(self, folder_names: list[str]) -> dict[str, str]:
        """Get or generate summaries for all folders."""
        descriptions = {}
        
        for name in folder_names:
            # Check cache
            if name in self._folder_summaries:
                descriptions[name] = self._folder_summaries[name]["summary"]
                continue

            # Generate summary from folder's document summaries
            try:
                folder_index = self.folder_manager.load_folder(name)
                if not folder_index.documents:
                    descriptions[name] = f"Empty folder: {name}"
                    continue

                # Combine document summaries
                doc_summaries = []
                for doc in folder_index.documents:
                    doc_summaries.append(
                        f"- {doc.filename} ({doc.total_pages} pages): {doc.summary}"
                    )

                prompt = f"""Folder: {name}
Documents:
{chr(10).join(doc_summaries)}

Create a folder-level summary."""

                response = self.llm.complete(
                    prompt=prompt,
                    model=self.config.model.search_model,
                    system_prompt=SYSTEM_PROMPT_FOLDER_SUMMARY,
                    json_mode=True,
                    max_tokens=512,
                )

                try:
                    parsed = json.loads(response)
                    summary = parsed.get("summary", f"Folder containing {len(folder_index.documents)} documents")
                    self._folder_summaries[name] = {
                        "summary": summary,
                        "keywords": parsed.get("keywords", []),
                    }
                    descriptions[name] = summary
                except json.JSONDecodeError:
                    descriptions[name] = f"Folder '{name}' with {len(folder_index.documents)} documents"

            except Exception:
                descriptions[name] = f"Folder: {name}"

        return descriptions

    def _build_route_prompt(
        self,
        query: str,
        folder_descriptions: dict[str, str],
        chat_history: list[dict] = None,
    ) -> str:
        """Build the routing prompt with folder descriptions and chat history."""
        # Format folder list
        folder_list = []
        for name, desc in folder_descriptions.items():
            folder_list.append(f'- "{name}": {desc}')
        folders_text = "\n".join(folder_list)

        # Format chat history (last 6 messages for context)
        history_text = ""
        if chat_history:
            recent = chat_history[-6:]
            history_parts = []
            for msg in recent:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")[:200]  # Truncate long messages
                history_parts.append(f"{role}: {content}")
            history_text = f"""

Recent chat history:
{chr(10).join(history_parts)}

Use this history to understand context. If the user seems to be asking a follow-up question about a topic discussed before, route to the same folder."""

        return f"""User query: {query}
{history_text}

Available folders:
{folders_text}

Which folder(s) should this query be routed to?"""

    def _fallback_route(
        self,
        query: str,
        folder_names: list[str],
        latency: float,
    ) -> "RouteResult":
        """Simple keyword-based fallback if LLM routing fails."""
        query_lower = query.lower()
        
        for name in folder_names:
            # Check if folder name appears in query
            if name.lower() in query_lower:
                return RouteResult(
                    selected_folders=[SelectedFolder(
                        folder_name=name,
                        confidence=0.7,
                        reasoning="Folder name matched in query (fallback)",
                    )],
                    latency=latency,
                )

        # No match — route to most recent folder
        if folder_names:
            return RouteResult(
                selected_folders=[SelectedFolder(
                    folder_name=folder_names[0],
                    confidence=0.3,
                    reasoning="No clear match, using most recent folder (fallback)",
                )],
                latency=latency,
            )

        return RouteResult(selected_folders=[], latency=latency)

    def invalidate_cache(self, folder_name: str = None):
        """Clear cached folder summaries."""
        if folder_name:
            self._folder_summaries.pop(folder_name, None)
        else:
            self._folder_summaries.clear()


# ============================================================================
# Data classes
# ============================================================================

class SelectedFolder:
    def __init__(self, folder_name: str, confidence: float, reasoning: str = ""):
        self.folder_name = folder_name
        self.confidence = confidence
        self.reasoning = reasoning


class RouteResult:
    def __init__(
        self,
        selected_folders: list[SelectedFolder] = None,
        needs_clarification: bool = False,
        clarification_question: str = None,
        no_folders_available: bool = False,
        latency: float = 0.0,
    ):
        self.selected_folders = selected_folders or []
        self.needs_clarification = needs_clarification
        self.clarification_question = clarification_question
        self.no_folders_available = no_folders_available
        self.latency = latency

    @property
    def has_match(self) -> bool:
        return len(self.selected_folders) > 0

    @property
    def top_folder(self) -> Optional[str]:
        if self.selected_folders:
            return self.selected_folders[0].folder_name
        return None

    @property
    def top_confidence(self) -> float:
        if self.selected_folders:
            return self.selected_folders[0].confidence
        return 0.0
