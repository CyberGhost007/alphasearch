#!/usr/bin/env python3
"""
TreeRAG CLI — Folder-based, MCTS-powered document retrieval.

Folder management:
    python main.py folder create "Project Alpha"
    python main.py folder add "Project Alpha" doc1.pdf doc2.pdf
    python main.py folder list
    python main.py folder info "Project Alpha"
    python main.py folder remove "Project Alpha" doc1.pdf
    python main.py folder health "Project Alpha"
    python main.py folder repair "Project Alpha"
    python main.py folder delete "Project Alpha"

Search:
    python main.py search "Project Alpha" "What was the budget?"
    python main.py search-doc "Project Alpha" budget.pdf "Phase 2 costs"
    python main.py interactive "Project Alpha"

Standalone:
    python main.py query report.pdf "What was Q3 revenue?"
    python main.py inspect .treerag_data/folders/project/indices/doc_tree.json
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def _handle_errors(func):
    """Decorator to catch TreeRAG exceptions and print friendly messages."""
    def wrapper(args):
        try:
            return func(args)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
            sys.exit(1)
        except Exception as e:
            error_type = type(e).__name__
            console.print(f"\n[bold red]Error ({error_type}):[/bold red] {e}")
            sys.exit(1)
    return wrapper


# =============================================================================
# Folder commands
# =============================================================================

@_handle_errors
def cmd_folder(args):
    from treerag.config import TreeRAGConfig
    from treerag.pipeline import TreeRAGPipeline

    config = TreeRAGConfig.from_env()
    pipeline = TreeRAGPipeline(config)
    fm = pipeline.folder

    if args.action == "create":
        if not args.name:
            console.print("[red]Usage: folder create <name>[/red]")
            return
        fm.create_folder(args.name)

    elif args.action == "add":
        if not args.name or not args.files:
            console.print("[red]Usage: folder add <name> file1.pdf file2.pdf ...[/red]")
            return
        if len(args.files) == 1:
            fm.add_document(args.name, args.files[0])
        else:
            fm.add_documents_batch(args.name, args.files)

    elif args.action == "remove":
        if not args.name or not args.files:
            console.print("[red]Usage: folder remove <name> filename.pdf[/red]")
            return
        for f in args.files:
            fm.remove_document(args.name, f)

    elif args.action == "list":
        folders = fm.list_folders()
        if not folders:
            console.print("[dim]No folders found. Create one: folder create <name>[/dim]")
            return
        table = Table(title="Folders")
        table.add_column("Name", style="cyan")
        table.add_column("Docs", justify="right")
        table.add_column("Pages", justify="right")
        for name in folders:
            try:
                fi = fm.load_folder(name)
                table.add_row(name, str(fi.total_documents), str(fi.total_pages))
            except Exception:
                table.add_row(name, "?", "?")
        console.print(table)

    elif args.action == "info":
        if not args.name:
            console.print("[red]Usage: folder info <name>[/red]")
            return
        fi = fm.load_folder(args.name)
        console.print(fi.pretty_print())

    elif args.action == "health":
        if not args.name:
            console.print("[red]Usage: folder health <name>[/red]")
            return
        issues = fm.health_check(args.name)
        console.print(f"\n[bold]Health Check: {args.name}[/bold]")
        console.print(f"  [green]Healthy: {len(issues['healthy'])}[/green]")
        if issues["missing_pdfs"]:
            console.print(f"  [red]Missing PDFs: {', '.join(issues['missing_pdfs'])}[/red]")
        if issues["missing_indices"]:
            console.print(f"  [yellow]Missing indices: {', '.join(issues['missing_indices'])}[/yellow]")
        if issues["stale_entries"]:
            console.print(f"  [cyan]Changed files: {', '.join(issues['stale_entries'])}[/cyan]")
        if issues["orphaned_indices"]:
            console.print(f"  [dim]Orphaned indices: {', '.join(issues['orphaned_indices'])}[/dim]")
        if not any(issues[k] for k in ["missing_pdfs", "missing_indices", "stale_entries", "orphaned_indices"]):
            console.print("  [green]Everything looks good![/green]")
        else:
            console.print(f"\n  Run 'folder repair {args.name}' to fix issues.")

    elif args.action == "repair":
        if not args.name:
            console.print("[red]Usage: folder repair <name>[/red]")
            return
        remove_broken = "--remove-broken" in (args.files or [])
        fm.repair_folder(args.name, remove_broken=remove_broken)

    elif args.action == "refresh":
        if not args.name:
            console.print("[red]Usage: folder refresh <name>[/red]")
            return
        fm.refresh_folder(args.name)

    elif args.action == "delete":
        if not args.name:
            console.print("[red]Usage: folder delete <name>[/red]")
            return
        fm.delete_folder(args.name)


# =============================================================================
# Search commands
# =============================================================================

@_handle_errors
def cmd_search(args):
    from treerag.config import TreeRAGConfig
    from treerag.pipeline import TreeRAGPipeline
    config = TreeRAGConfig.from_env()
    pipeline = TreeRAGPipeline(config)
    pipeline.query_folder(args.query, args.folder_name, use_vision=not args.no_vision)


@_handle_errors
def cmd_search_doc(args):
    from treerag.config import TreeRAGConfig
    from treerag.pipeline import TreeRAGPipeline
    config = TreeRAGConfig.from_env()
    pipeline = TreeRAGPipeline(config)
    fm = pipeline.folder

    fi = fm.load_folder(args.folder_name)
    entry = fi.get_document(args.filename)
    if not entry:
        available = ', '.join(d.filename for d in fi.documents) or 'none'
        console.print(f"[red]'{args.filename}' not found in '{args.folder_name}'. Available: {available}[/red]")
        return

    doc_index = fm.load_document_index(entry)
    pipeline.query_document(args.query, doc_index, use_vision=not args.no_vision)


@_handle_errors
def cmd_interactive(args):
    from treerag.config import TreeRAGConfig
    from treerag.pipeline import TreeRAGPipeline, ChatMessage
    config = TreeRAGConfig.from_env()
    pipeline = TreeRAGPipeline(config)

    if args.mode == "folder":
        # Folder-specific interactive mode (existing)
        fi = pipeline.folder.load_folder(args.target)
        console.print(f"\n[bold green]Interactive mode[/bold green] — {args.target}")
        console.print(f"  Documents: {fi.total_documents} | Pages: {fi.total_pages}")
        console.print("  Commands: 'quit', 'info', 'health'\n")

        while True:
            try:
                query = console.input("[bold cyan]Query>[/bold cyan] ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    break
                if query.lower() == "info":
                    console.print(fi.pretty_print())
                    continue
                if not query:
                    continue
                try:
                    pipeline.query_folder(query, args.target, use_vision=not args.no_vision)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                console.print()
            except KeyboardInterrupt:
                break
    else:
        # Unified chat mode (new) — auto-routes to folders
        console.print(f"\n[bold green]TreeRAG Chat[/bold green] — auto-routes across all folders")
        folders = pipeline.folder.list_folders()
        console.print(f"  Available folders: {', '.join(folders) if folders else 'none'}")
        console.print("  Commands: 'quit', 'folders'\n")

        chat_history = []

        while True:
            try:
                query = console.input("[bold cyan]You>[/bold cyan] ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    break
                if query.lower() == "folders":
                    for f in pipeline.folder.list_folders():
                        fi = pipeline.folder.load_folder(f)
                        console.print(f"  {f}: {fi.total_documents} docs, {fi.total_pages} pages")
                    continue
                if not query:
                    continue

                chat_history.append(ChatMessage(role="user", content=query))

                try:
                    response = pipeline.chat(query, chat_history, use_vision=not args.no_vision)
                    chat_history.append(response)

                    if response.folder_name:
                        console.print(f"\n[dim]📁 {response.folder_name}[/dim]")
                    console.print(f"\n{response.content}")
                    if response.sources:
                        console.print(f"\n[dim]Sources:[/dim]")
                        for s in response.sources:
                            console.print(f"  [dim]{s['document']} → {s['section']} ({s['pages']}) [{s['score']:.0%}][/dim]")
                    if response.stats:
                        console.print(f"[dim]{response.stats.get('total_time', '')} | {response.stats.get('llm_calls', '')} calls | {response.stats.get('cost', '')}[/dim]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                console.print()
            except KeyboardInterrupt:
                break

    console.print(f"\n[dim]Session: {pipeline.usage}[/dim]")


# =============================================================================
# Standalone commands
# =============================================================================

@_handle_errors
def cmd_query(args):
    from treerag.config import TreeRAGConfig
    from treerag.pipeline import TreeRAGPipeline
    config = TreeRAGConfig.from_env()
    pipeline = TreeRAGPipeline(config)

    if args.index:
        doc_index = pipeline.load_index(args.index)
    else:
        doc_index = pipeline.index(args.pdf_path)

    pipeline.query_document(args.query, doc_index, use_vision=not args.no_vision)


@_handle_errors
def cmd_inspect(args):
    from treerag.models import DocumentIndex
    doc = DocumentIndex.load(args.index_path)
    console.print(f"\n[bold]Document:[/bold] {doc.filename} ({doc.total_pages} pages)")
    console.print(f"[bold]Hash:[/bold] {doc.file_hash}")
    console.print(f"[bold]Description:[/bold] {doc.description}")
    if doc.root:
        console.print(f"\n[bold]Tree:[/bold]")
        console.print(doc.root.pretty_print())
        nodes = doc.get_all_nodes()
        console.print(f"\nNodes: {len(nodes)} | Leaves: {len([n for n in nodes if n.is_leaf])}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TreeRAG — MCTS-powered Vectorless Document Retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # folder
    p = subparsers.add_parser("folder", help="Manage folders")
    p.add_argument("action", choices=["create", "add", "remove", "list", "info", "health", "repair", "refresh", "delete"])
    p.add_argument("name", nargs="?")
    p.add_argument("files", nargs="*")
    p.set_defaults(func=cmd_folder)

    # search
    p = subparsers.add_parser("search", help="Search a folder")
    p.add_argument("folder_name"); p.add_argument("query"); p.add_argument("--no-vision", action="store_true")
    p.set_defaults(func=cmd_search)

    # search-doc
    p = subparsers.add_parser("search-doc", help="Search a specific document")
    p.add_argument("folder_name"); p.add_argument("filename"); p.add_argument("query"); p.add_argument("--no-vision", action="store_true")
    p.set_defaults(func=cmd_search_doc)

    # chat (unified)
    p = subparsers.add_parser("chat", help="Unified chat — auto-routes to folders")
    p.add_argument("--no-vision", action="store_true")
    p.set_defaults(func=cmd_interactive, mode="chat", target=None)

    # interactive (folder-specific)
    p = subparsers.add_parser("interactive", help="Interactive search on a specific folder")
    p.add_argument("folder_name")
    p.add_argument("--no-vision", action="store_true")
    p.set_defaults(func=lambda a: cmd_interactive(
        argparse.Namespace(mode="folder", target=a.folder_name, no_vision=a.no_vision)
    ))

    # query (standalone)
    p = subparsers.add_parser("query", help="Standalone: index + query a PDF")
    p.add_argument("pdf_path"); p.add_argument("query"); p.add_argument("--index"); p.add_argument("--no-vision", action="store_true")
    p.set_defaults(func=cmd_query)

    # inspect
    p = subparsers.add_parser("inspect", help="Inspect a saved index")
    p.add_argument("index_path")
    p.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
