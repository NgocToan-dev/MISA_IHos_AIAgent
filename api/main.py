"""FastAPI entrypoint.

Endpoint: POST /invoke {"query": "..."}
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from pydantic import BaseModel
from inference.run import invoke_agent, stream_agent
from model.schemas import AgentRequest, AgentResponse

app = FastAPI(title="LangGraph Base Agent", version="0.1.0")

# In-memory conversation memory: session_id -> list of (role, content)
CONVERSATIONS: dict[str, list[dict[str, str]]] = {}

# Mount static directory (js, css, images)
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.post("/invoke", response_model=AgentResponse)
def invoke(req: AgentRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query rỗng")
    session_id = req.session_id or "default"
    history = CONVERSATIONS.setdefault(session_id, [])
    # Ghép history vào prompt đầu vào đơn giản
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-6:]])  # last 6 turns
    augmented_query = f"Lịch sử trước đó:\n{history_text}\n\nNgười dùng hỏi: {req.query}" if history else req.query
    result = invoke_agent(augmented_query)
    output = result.get("output", "")
    # Cập nhật history
    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": output})
    return AgentResponse(**result, session_id=session_id)


@app.get("/health")
def health():
    return {"status": "ok"}


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
    history = CONVERSATIONS.setdefault(sid, [])
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-6:]])
    augmented_query = f"Lịch sử trước đó:\n{history_text}\n\nNgười dùng hỏi: {q}" if history else q

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
        # Lưu vào memory nếu không disconnect sớm
        if collected:
            answer = "".join(collected)
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
