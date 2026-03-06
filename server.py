"""
AlphaSearch API Server

Routes:
  /           → Landing page (pick Single or Folder mode)
  /single     → Single PDF mode (upload one PDF, ask questions)
  /folders    → Folder mode (multi-doc, auto-routing, full MCTS)

Run: python server.py
"""

import os
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from treerag.config import TreeRAGConfig
from treerag.pipeline import TreeRAGPipeline, ChatMessage
from treerag.models import DocumentIndex


# ============================================================================
# State
# ============================================================================
pipeline: Optional[TreeRAGPipeline] = None

folder_chat_history: list[ChatMessage] = []

single_doc_index: Optional[DocumentIndex] = None
single_pdf_path: Optional[str] = None
single_chat_history: list[ChatMessage] = []

UPLOAD_DIR = Path(".treerag_data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    config = TreeRAGConfig.from_env()
    pipeline = TreeRAGPipeline(config)
    port = os.getenv("PORT", "8000")
    print(f"\n  AlphaSearch ready")
    print(f"  Landing:     http://localhost:{port}")
    print(f"  Single mode: http://localhost:{port}/single")
    print(f"  Folder mode: http://localhost:{port}/folders\n")
    yield


app = FastAPI(title="AlphaSearch", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    query: str

class FolderCreateRequest(BaseModel):
    name: str


# ============================================================================
# SINGLE MODE APIs
# ============================================================================

@app.post("/api/single/upload")
async def single_upload(file: UploadFile = File(...)):
    global single_doc_index, single_pdf_path, single_chat_history
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())
        single_doc_index = pipeline.index(str(save_path))
        single_pdf_path = str(save_path)
        single_chat_history = []
        return {"filename": file.filename, "pages": single_doc_index.total_pages, "description": single_doc_index.description, "status": "indexed"}
    except Exception as e:
        raise HTTPException(400, f"Failed to index: {e}")


@app.post("/api/single/chat")
def single_chat(req: ChatRequest):
    if not single_doc_index:
        return {"answer": "No document uploaded yet. Upload a PDF first.", "sources": [], "stats": {}}
    try:
        result = pipeline.query_document(req.query, single_doc_index, use_vision=True)
        sources = [{"doc": s.document_filename, "section": s.node.title, "pages": s.node.page_range, "score": round(s.relevance_score, 3)} for s in result.sources]
        return {"answer": result.answer, "sources": sources, "stats": {"total": f"{result.latency_seconds:.1f}s", "calls": result.total_llm_calls, "cost": f"${pipeline.llm._estimate_cost():.4f}"}}
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": [], "stats": {}}


@app.get("/api/single/status")
def single_status():
    if single_doc_index:
        return {"loaded": True, "filename": single_doc_index.filename, "pages": single_doc_index.total_pages, "description": single_doc_index.description}
    return {"loaded": False}


@app.post("/api/single/reset")
def single_reset():
    global single_doc_index, single_pdf_path, single_chat_history
    single_doc_index = None
    single_pdf_path = None
    single_chat_history = []
    return {"status": "cleared"}


# ============================================================================
# FOLDER MODE APIs
# ============================================================================

@app.get("/api/folders")
def list_folders():
    folders = []
    for name in pipeline.folder.list_folders():
        try:
            fi = pipeline.folder.load_folder(name)
            docs = [{"name": d.filename, "pages": d.total_pages, "time": _time_ago(d.indexed_at), "status": "indexed"} for d in fi.documents]
            folders.append({"name": fi.folder_name, "docs": docs, "total_pages": fi.total_pages})
        except Exception:
            folders.append({"name": name, "docs": [], "total_pages": 0})
    return {"folders": folders}


@app.post("/api/folders")
def create_folder(req: FolderCreateRequest):
    try:
        fi = pipeline.folder.create_folder(req.name)
        return {"name": fi.folder_name, "docs": [], "total_pages": 0}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.delete("/api/folders/{folder_name}")
def delete_folder(folder_name: str):
    pipeline.folder.delete_folder(folder_name)
    return {"deleted": folder_name}


@app.post("/api/folders/{folder_name}/documents")
async def upload_folder_document(folder_name: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    temp_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        fi = pipeline.folder.add_document(folder_name, str(temp_path))
        entry = fi.get_document(file.filename)
        return {"name": entry.filename, "pages": entry.total_pages, "time": "just now", "status": "indexed"}
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.delete("/api/folders/{folder_name}/documents/{filename}")
def remove_folder_document(folder_name: str, filename: str):
    try:
        pipeline.folder.remove_document(folder_name, filename)
        return {"removed": filename}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/folders/chat")
def folder_chat(req: ChatRequest):
    global folder_chat_history
    folder_chat_history.append(ChatMessage(role="user", content=req.query))
    try:
        response = pipeline.chat(req.query, folder_chat_history, use_vision=True)
        folder_chat_history.append(response)
        return {"answer": response.content, "folder": response.folder_name, "sources": response.sources, "stats": response.stats}
    except Exception as e:
        return {"answer": f"Error: {e}", "folder": "", "sources": [], "stats": {}}


@app.post("/api/folders/chat/reset")
def folder_chat_reset():
    global folder_chat_history
    folder_chat_history = []
    return {"status": "cleared"}


# ============================================================================
# Health
# ============================================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "folders": len(pipeline.folder.list_folders()), "single_loaded": single_doc_index is not None, "usage": pipeline.usage}


# ============================================================================
# Landing Page
# ============================================================================

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlphaSearch</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Instrument+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#09090B;color:#FAFAFA;font-family:'DM Sans',system-ui,sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center}
.ctr{text-align:center;max-width:640px;padding:40px 24px}
.logo{width:64px;height:64px;border-radius:18px;background:linear-gradient(135deg,#3B82F6,#8B5CF6);display:inline-flex;align-items:center;justify-content:center;margin-bottom:24px;box-shadow:0 12px 40px rgba(59,130,246,.25)}
.logo svg{width:32px;height:32px;stroke:#fff;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
h1{font-family:'Instrument Sans',system-ui,sans-serif;font-size:36px;font-weight:700;letter-spacing:-.5px;margin-bottom:8px}
.tag{color:#71717A;font-size:15px;line-height:1.6;margin-bottom:48px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#111113;border:1px solid #27272A;border-radius:16px;padding:32px 24px;cursor:pointer;text-decoration:none;color:#FAFAFA;transition:all .2s;text-align:left}
.card:hover{border-color:#3B82F6;background:rgba(59,130,246,.04);transform:translateY(-2px);box-shadow:0 8px 24px rgba(59,130,246,.1)}
.ci{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;margin-bottom:16px}
.ci svg{width:22px;height:22px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.card h2{font-family:'Instrument Sans',system-ui,sans-serif;font-size:18px;font-weight:600;margin-bottom:8px}
.card p{color:#71717A;font-size:13px;line-height:1.5}
.badge{display:inline-block;margin-top:12px;padding:3px 10px;border-radius:20px;font-size:11px;font-family:'JetBrains Mono',monospace}
.ft{margin-top:48px;font-size:12px;color:#52525B;font-family:'JetBrains Mono',monospace}
.ft a{color:#3B82F6;text-decoration:none}
</style>
</head>
<body>
<div class="ctr">
<div class="logo"><svg viewBox="0 0 24 24"><path d="M12 3v9m0 0l-4 4m4-4l4 4M4 20h16"/></svg></div>
<h1>AlphaSearch</h1>
<p class="tag">AlphaGo-inspired MCTS for document retrieval.<br>No vectors. No embeddings. Just reasoning.</p>
<div class="cards">
<a href="/single" class="card">
<div class="ci" style="background:rgba(59,130,246,.12);color:#3B82F6"><svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6"/></svg></div>
<h2>Single PDF</h2>
<p>Upload one document, ask questions instantly. No setup needed.</p>
<span class="badge" style="background:rgba(59,130,246,.12);color:#3B82F6">Quick start</span>
</a>
<a href="/folders" class="card">
<div class="ci" style="background:rgba(139,92,246,.12);color:#8B5CF6"><svg viewBox="0 0 24 24"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg></div>
<h2>Folders</h2>
<p>Organize documents into folders. Auto-routes queries across all docs.</p>
<span class="badge" style="background:rgba(139,92,246,.12);color:#8B5CF6">Full power</span>
</a>
</div>
<div class="ft">MIT License · Built by KnightOwl</div>
</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def landing():
    return HTMLResponse(LANDING_HTML)


# ============================================================================
# Serve Chat UIs
# ============================================================================

def _serve_jsx(mode: str) -> HTMLResponse:
    jsx_path = Path(__file__).parent / "chat_ui.jsx"
    if not jsx_path.exists():
        return HTMLResponse("<h1>chat_ui.jsx not found</h1>")
    jsx_code = f'const APP_MODE = "{mode}";\n' + jsx_path.read_text()
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AlphaSearch — {"Single PDF" if mode == "single" else "Folders"}</title>
<style>body{{margin:0}}#root{{height:100vh}}</style></head><body><div id="root"></div>
<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script type="text/babel">{jsx_code}</script></body></html>""")


@app.get("/single", response_class=HTMLResponse)
def serve_single():
    return _serve_jsx("single")

@app.get("/folders", response_class=HTMLResponse)
def serve_folders():
    return _serve_jsx("folders")


def _time_ago(ts: float) -> str:
    if not ts: return "unknown"
    d = time.time() - ts
    if d < 60: return "just now"
    if d < 3600: return f"{int(d/60)}m ago"
    if d < 86400: return f"{int(d/3600)}h ago"
    return f"{int(d/86400)}d ago"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
