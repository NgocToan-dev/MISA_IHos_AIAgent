"""FastAPI entrypoint.

Endpoint: POST /invoke {"query": "..."}
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference.run import invoke_agent
from model.schemas import AgentRequest, AgentResponse

app = FastAPI(title="LangGraph Base Agent", version="0.1.0")


@app.post("/invoke", response_model=AgentResponse)
def invoke(req: AgentRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query rỗng")
    result = invoke_agent(req.query)
    return AgentResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}
