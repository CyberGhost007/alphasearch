"""
Tabular Indexer — builds tree index from CSV/Excel with data-aware sectioning.

Flow:
1. Auto-analyze: pandas detects types, groupings, computes stats, builds sections
2. LLM refine: sends analysis + sample rows to LLM for better titles/summaries
3. Build tree: converts refined sections into TreeNode hierarchy
4. Validate: quick LLM check on a few sections
"""

import json
from pathlib import Path
from typing import Optional
from rich.console import Console

from .config import TreeRAGConfig
from .models import TreeNode, DocumentIndex, compute_file_hash
from .llm_client import LLMClient
from .tabular_processor import TabularProcessor, LoadedTabular, TabularAnalysis
from .exceptions import IndexingFailedError

console = Console()

SYSTEM_PROMPT_TABULAR_REFINE = """You are a data analyst. Given auto-detected sections of a tabular dataset
with statistics, refine the section titles and summaries to be search-friendly.

Rules:
- Keep ALL statistics (numbers, counts, sums, means) — these are critical for search
- Improve titles to be descriptive and searchable
- Add context about what the data represents
- If you can infer what the columns measure (revenue, quantity, temperature, etc.), say so
- Keep keywords data-driven: column names, top values, date ranges

Output JSON:
{
  "title": "Refined document title",
  "summary": "2-3 sentences about the entire dataset with key statistics",
  "keywords": ["keyword1", ...],
  "sections": [
    {
      "title": "Refined section title",
      "summary": "Section summary with all statistics preserved",
      "keywords": ["keyword1", ...],
      "start_row": <int>,
      "end_row": <int>,
      "level": <int>
    }
  ]
}"""

SYSTEM_PROMPT_TABULAR_DOC_SUMMARY = """You are summarizing a tabular dataset for search purposes.
Given the column structure, statistics, and sample data, generate:

{
  "summary": "3-5 sentences about this dataset. Be SPECIFIC: include column names, row counts,
  numeric ranges, date spans, categories found. This summary will decide if this file is relevant to queries.",
  "keywords": ["keyword1", "keyword2", ..., "keyword8"]
}

Keywords should be specific: column names, category values, date ranges, metric names."""


class TabularIndexer:
    def __init__(self, config: TreeRAGConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.processor = TabularProcessor(
            rows_per_page=50,
            max_rows=config.indexer.max_pages * 50,  # Approximate row limit
        )

    def index_document(
        self, file_path: str | Path,
        save_path: Optional[str | Path] = None,
        skip_validation: bool = False,
    ) -> DocumentIndex:
        """Build a tree index from a tabular file."""
        file_path = Path(file_path)
        save_path = Path(save_path) if save_path else None

        console.print(f"\n[bold blue]Indexing:[/bold blue] {file_path.name}")

        # Step 1: Load and auto-analyze
        console.print("\n[bold]Step 1:[/bold] Analyzing tabular structure...")
        loaded = self.processor.load(file_path)
        analysis = loaded.analysis
        sections = loaded.sections

        console.print(f"  Rows: {analysis.total_rows:,} | Columns: {analysis.total_columns}")
        console.print(f"  Numeric: {len(analysis.numeric_columns)} | Categorical: {len(analysis.grouping_columns)} | Date: {len(analysis.date_columns)}")
        console.print(f"  Auto-detected {len(sections)} sections")

        if analysis.grouping_columns:
            console.print(f"  Grouping by: {analysis.grouping_columns[0]}")

        try:
            # Step 2: LLM refine
            console.print("\n[bold]Step 2:[/bold] Refining sections with LLM...")
            refined = self._refine_sections(loaded, analysis, sections)

            # Step 3: Build tree
            console.print("\n[bold]Step 3:[/bold] Building tree index...")
            tree = self._build_tree(refined, analysis, loaded)

        except Exception as e:
            loaded.close()
            if save_path and save_path.exists():
                save_path.unlink()
            raise IndexingFailedError(
                f"Indexing failed for '{file_path.name}': {e}"
            ) from e

        file_hash = compute_file_hash(file_path)
        doc_index = DocumentIndex(
            document_id=file_hash[:12],
            filename=file_path.name,
            total_pages=loaded.total_pages,
            description=tree.summary if tree else "",
            root=tree,
            file_hash=file_hash,
            pdf_path=str(file_path),
        )

        if save_path:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                doc_index.save(save_path)
                console.print(f"\n[green]Index saved:[/green] {save_path}")
            except OSError as e:
                raise IndexingFailedError(
                    f"Failed to save index for '{file_path.name}': {e}."
                ) from e

        loaded.close()
        console.print(f"[bold green]Indexing complete![/bold green] (LLM calls: {self.llm.total_calls})")
        return doc_index

    def generate_document_summary(self, file_path: str | Path) -> tuple[str, list[str]]:
        """Generate document-level summary for the meta-tree."""
        file_path = Path(file_path)
        loaded = self.processor.load(file_path)
        analysis = loaded.analysis

        # Build a rich text description of the dataset
        col_descriptions = []
        for col in analysis.columns[:15]:
            desc = f"  - {col.name} ({col.dtype})"
            if col.is_numeric and col.stats:
                desc += f": min={col.stats['min']}, max={col.stats['max']}, mean={col.stats['mean']}"
            elif col.is_categorical and col.top_values:
                vals = [tv["value"] for tv in col.top_values[:3]]
                desc += f": {', '.join(vals)}"
            elif col.is_date and col.stats:
                desc += f": {col.stats.get('min', '?')} to {col.stats.get('max', '?')}"
            col_descriptions.append(desc)

        # Sample rows as text
        sample = loaded.get_page_text(0) if loaded.total_pages > 0 else ""

        prompt = f"""Dataset: "{file_path.name}"
Rows: {analysis.total_rows:,} | Columns: {analysis.total_columns}

Column Details:
{chr(10).join(col_descriptions)}

Sample Data (first rows):
{sample[:2000]}

Summarize this dataset for search purposes."""

        try:
            response = self.llm.complete(
                prompt=prompt,
                model=self.config.model.indexing_model,
                system_prompt=SYSTEM_PROMPT_TABULAR_DOC_SUMMARY,
                json_mode=True, max_tokens=1024,
            )
            loaded.close()
            result = json.loads(response)
            return result.get("summary", ""), result.get("keywords", [])
        except Exception:
            loaded.close()
            # Fallback: auto-generated summary
            summary = (
                f"Tabular dataset: {file_path.name}. "
                f"{analysis.total_rows:,} rows, {analysis.total_columns} columns. "
                f"Columns: {', '.join(c.name for c in analysis.columns[:8])}."
            )
            keywords = [c.name for c in analysis.columns[:8]]
            return summary, keywords

    def _refine_sections(
        self, loaded: LoadedTabular, analysis: TabularAnalysis,
        sections: list[dict],
    ) -> dict:
        """Send auto-generated sections to LLM for refinement."""
        # Build context for LLM
        col_info = []
        for col in analysis.columns[:20]:
            info = {"name": col.name, "type": col.dtype, "nulls": col.null_count}
            if col.stats:
                info["stats"] = col.stats
            if col.top_values:
                info["top_values"] = col.top_values[:3]
            col_info.append(info)

        # Sample rows per section (first 3 rows each)
        section_samples = []
        for s in sections[:15]:  # Limit sections sent to LLM
            start = s.get("start_row", 0)
            end = min(start + 2, len(loaded.df) - 1) if loaded.df is not None else start
            data_cols = [c for c in loaded.df.columns if c != "__sheet__"]
            sample_rows = loaded.df.iloc[start:end + 1][data_cols].head(3).to_dict(orient="records")
            section_samples.append({
                **s,
                "sample_rows": sample_rows,
            })

        prompt = f"""Dataset: "{analysis.filename}" ({analysis.total_rows:,} rows, {analysis.total_columns} columns)

Columns:
{json.dumps(col_info, indent=2, default=str)}

Auto-detected sections:
{json.dumps(section_samples, indent=2, default=str)}

Refine these sections for better search. Keep all statistics. Improve titles and summaries."""

        try:
            response = self.llm.complete(
                prompt=prompt,
                model=self.config.model.indexing_model,
                system_prompt=SYSTEM_PROMPT_TABULAR_REFINE,
                json_mode=True, max_tokens=4096,
            )
            refined = json.loads(response)
            console.print(f"  LLM refined {len(refined.get('sections', []))} sections")
            return refined
        except Exception as e:
            console.print(f"  [yellow]LLM refinement failed: {e}. Using auto-generated sections.[/yellow]")
            return {
                "title": analysis.filename,
                "summary": f"Tabular data: {analysis.total_rows:,} rows, {analysis.total_columns} columns",
                "keywords": [c.name for c in analysis.columns[:8]],
                "sections": sections,
            }

    def _build_tree(
        self, refined: dict, analysis: TabularAnalysis,
        loaded: LoadedTabular,
    ) -> TreeNode:
        """Convert refined sections into TreeNode hierarchy."""
        sections = refined.get("sections", [])

        # Root node
        root_summary = refined.get("summary", f"{analysis.filename}: {analysis.total_rows:,} rows")
        root_keywords = refined.get("keywords", [c.name for c in analysis.columns[:8]])
        total_pages = loaded.total_pages

        root = TreeNode(
            node_id="0000",
            title=refined.get("title", analysis.filename),
            summary=root_summary,
            keywords=root_keywords,
            start_page=0,
            end_page=total_pages - 1,
            level=0,
        )

        if not sections:
            return root

        # Build children from sections
        # Separate level 1 (primary) and level 2+ (sub-sections)
        level1_sections = [s for s in sections if s.get("level", 1) == 1]
        level2_sections = [s for s in sections if s.get("level", 1) >= 2]

        node_id = 1
        for s in level1_sections:
            start_page = s.get("start_page", s.get("start_row", 0) // loaded.rows_per_page)
            end_page = s.get("end_page", s.get("end_row", 0) // loaded.rows_per_page)

            child = TreeNode(
                node_id=f"{node_id:04d}",
                title=s.get("title", f"Section {node_id}"),
                summary=s.get("summary", ""),
                keywords=s.get("keywords", [])[:8],
                start_page=start_page,
                end_page=end_page,
                level=1,
                parent=root,
            )
            root.children.append(child)

            # Attach level 2 children that fall within this section's row range
            s_start = s.get("start_row", start_page * loaded.rows_per_page)
            s_end = s.get("end_row", (end_page + 1) * loaded.rows_per_page - 1)

            for sub in level2_sections:
                sub_start = sub.get("start_row", 0)
                sub_end = sub.get("end_row", 0)
                if sub_start >= s_start and sub_end <= s_end:
                    node_id += 1
                    sub_start_page = sub.get("start_page", sub_start // loaded.rows_per_page)
                    sub_end_page = sub.get("end_page", sub_end // loaded.rows_per_page)
                    grandchild = TreeNode(
                        node_id=f"{node_id:04d}",
                        title=sub.get("title", f"Sub-section {node_id}"),
                        summary=sub.get("summary", ""),
                        keywords=sub.get("keywords", [])[:8],
                        start_page=sub_start_page,
                        end_page=sub_end_page,
                        level=2,
                        parent=child,
                    )
                    child.children.append(grandchild)

            node_id += 1

        nodes = root.children if root.children else [root]
        console.print(f"  Tree: {len(root.children)} top-level sections, {sum(len(c.children) for c in root.children)} sub-sections")
        return root
