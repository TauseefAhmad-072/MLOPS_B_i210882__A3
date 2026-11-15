"""Agent FastAPI interface: POST /chat, GET /tools, GET /health."""
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException

from agent.config import get_settings
from agent.orchestrator import Orchestrator
from agent.schemas import ChatRequest, ChatResponse
from agent.tools import REGISTRY

app = FastAPI(title="MLOps AI Agent", version="1.0.0")
_orch = Orchestrator()


def auth(x_api_key: str | None = Header(default=None)):
    key = get_settings().agent_api_key
    if key and x_api_key != key:
        raise HTTPException(401, "invalid or missing X-API-Key")


@app.get("/health")
def health():
    return {"status": "ok", "tools": len(REGISTRY)}


@app.get("/tools", dependencies=[Depends(auth)])
def tools():
    return [{"name": t.name, "module": t.module, "mutating": t.mutating, "description": t.description,
             "input_schema": t.input_schema} for t in REGISTRY.values()]


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(auth)])
def chat(req: ChatRequest):
    return _orch.handle(req.message, confirm=req.confirm, request_id=uuid.uuid4().hex[:12])
