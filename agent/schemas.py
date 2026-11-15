from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["ok", "error", "timeout", "denied", "invalid", "unavailable"]


class ToolResult(BaseModel):
    tool: str
    status: Status
    data: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0
    request_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    confirm: bool = False  # explicit user confirmation for state-changing operations


class ChatResponse(BaseModel):
    request_id: str
    mode: str
    answer: str
    tool_calls: list[ToolResult] = []
    facts: list[str] = []
    uncertainties: list[str] = []
    recommendations: list[str] = []
