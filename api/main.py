"""FastAPI entrypoint.

Endpoint: POST /invoke {"query": "..."}
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from pydantic import BaseModel
from inference.run import invoke_agent, stream_agent
from model.schemas import AgentRequest, AgentResponse
from services.mongo.mongo_repo import insert_many
from typing import List, Dict, Any, Optional
import uuid
from services.milvus.milvus_repo import index_text_file
from services.api.conversation_service import get_history, append_messages, clear_history

app = FastAPI(title="LangGraph Base Agent", version="0.1.0")

# Conversation storage is persisted in Mongo via services/api/conversation_service

# Mount static directory (js, css, images)
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.post("/invoke", response_model=AgentResponse)
def invoke(req: AgentRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query rỗng")
    session_id = req.session_id or "default"
    history = get_history(session_id, limit=12)
    # Build history text from last N messages
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-12:]]) if history else ""
    augmented_query = f"Lịch sử trước đó:\n{history_text}\n\nNgười dùng hỏi: {req.query}" if history_text else req.query
    try:
        result = invoke_agent(augmented_query)
        output = result.get("output", "")
    except Exception as e:
        import traceback

        print(f"[api.invoke] error invoking agent for session={session_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"agent error: {e}")
    # Persist messages
    append_messages(session_id, [{"role": "user", "content": req.query}, {"role": "assistant", "content": output}])
    return AgentResponse(**result, session_id=session_id)


@app.get("/health")
def health():
    return {"status": "ok"}


class InsertManyRequest(BaseModel):
    col: str
    documents: List[Dict[str, Any]]


class InsertManyResponse(BaseModel):
    inserted_ids: List[str]


@app.post("/insert_many", response_model=InsertManyResponse)
def mongo_insert_many(req: InsertManyRequest):
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents rỗng")
    print(f"Insert {len(req.documents)} documents")
    ids = insert_many(req.col, req.documents)
    return InsertManyResponse(inserted_ids=ids)


class UploadTextResponse(BaseModel):
    doc_id: str
    collection: str
    chunk_count: int
    inserted_ids: List[int]


@app.post("/upload_text", response_model=UploadTextResponse)
async def upload_text(
    file: UploadFile = File(..., description="File .txt chứa văn bản dài"),
    doc_id: Optional[str] = Form(None),
    chunk_size: int = Form(800),
    overlap: int = Form(100),
    collection: Optional[str] = Form(None),
):
    """Nhận 1 file txt, tách chunk, embed và lưu vào Milvus.

    Trả về doc_id + thông tin số chunk.
    """
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .txt")
    try:
        content_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không đọc được file: {e}")
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content_bytes.decode("utf-16")
        except Exception:
            text = content_bytes.decode(errors="ignore")
    if not text.strip():
        raise HTTPException(status_code=400, detail="File rỗng")
    the_doc_id = doc_id or str(uuid.uuid4())
    try:
        result = index_text_file(
            doc_id=the_doc_id,
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            collection=collection,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index lỗi: {e}")
    return UploadTextResponse(
        doc_id=result["doc_id"],
        collection=result["collection"],
        chunk_count=result["chunk_count"],
        inserted_ids=result.get("inserted_ids", []),
    )


@app.get("/", include_in_schema=False)
def ui_index():
    """Serve the frontend HTML (templates/index.html)."""
    index_path = Path(__file__).resolve().parent.parent / "templates" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"UI not found: {index_path}")
    return FileResponse(str(index_path))


@app.get("/invoke/stream", include_in_schema=False)
def invoke_stream(q: str, request: Request, session_id: str | None = None):
    if not q.strip():
        raise HTTPException(status_code=400, detail="query rỗng")
    sid = session_id or "default"
    history = get_history(sid, limit=12)
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-12:]]) if history else ""
    augmented_query = f"Lịch sử trước đó:\n{history_text}\n\nNgười dùng hỏi: {q}" if history_text else q

    async def event_gen():
        collected = []
        try:
            for token in stream_agent(augmented_query):
                if await request.is_disconnected():
                    break
                collected.append(token)
                yield f"data: {token}\n\n"
            else:
                if not await request.is_disconnected():
                    yield "event: done\ndata: [DONE]\n\n"
        except (ConnectionResetError, BrokenPipeError):
            return
        except Exception as e:
            import traceback

            print(f"[api.invoke_stream] error streaming for session={sid}: {e}")
            traceback.print_exc()
            # propagate a server error event and stop
            try:
                yield f"event: error\ndata: {str(e)}\n\n"
            except Exception:
                pass
            return
        # Lưu vào db nếu không disconnect sớm
        if collected:
            answer = "".join(collected)
            append_messages(sid, [{"role": "user", "content": q}, {"role": "assistant", "content": answer}])

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/history/{session_id}")
def get_history_endpoint(session_id: str, limit: int = 50):
    msgs = get_history(session_id, limit=limit)
    return {"session_id": session_id, "messages": msgs}


@app.delete("/history/{session_id}")
def clear_history_endpoint(session_id: str):
    clear_history(session_id)
    return {"session_id": session_id, "cleared": True}
