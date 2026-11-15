"""Tool registry + safe executor (validation, permissions, timeout, structured logging)."""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Callable

from agent.config import get_settings
from agent.logging_utils import log_tool_call, redact_text
from agent.schemas import ToolResult


class ToolError(Exception):
    """Expected operational failure (service reachable but request failed)."""


class ServiceUnavailable(ToolError):
    """The backing service could not be reached."""


class PermissionDenied(Exception):
    """Operation rejected by the safety layer."""


_TYPES = {"string": str, "integer": int, "boolean": bool, "number": (int, float), "array": list, "object": dict}


@dataclass
class Tool:
    name: str
    description: str
    module: str
    properties: dict[str, dict]
    required: tuple[str, ...]
    fn: Callable[..., dict]
    mutating: bool = False

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": self.properties,
                "required": list(self.required), "additionalProperties": False}


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, module: str, properties: dict | None = None,
         required: tuple = (), mutating: bool = False):
    def deco(fn):
        REGISTRY[name] = Tool(name, description, module, properties or {}, tuple(required), fn, mutating)
        return fn
    return deco


def validate_args(t: Tool, args: dict) -> str | None:
    if not isinstance(args, dict):
        return "arguments must be an object"
    for r in t.required:
        if r not in args:
            return f"missing required argument '{r}'"
    for k, v in args.items():
        if k not in t.properties:
            return f"unknown argument '{k}'"
        want = _TYPES.get(t.properties[k].get("type", "string"))
        if want is int and isinstance(v, bool):
            return f"argument '{k}' must be integer"
        if want and not isinstance(v, want):
            return f"argument '{k}' must be {t.properties[k].get('type')}"
    return None


def execute_tool(name: str, args: dict | None = None, request_id: str | None = None,
                 timeout: float | None = None) -> ToolResult:
    request_id = request_id or uuid.uuid4().hex[:12]
    args = args or {}
    start = time.perf_counter()

    def done(status, data=None, error=None):
        ms = (time.perf_counter() - start) * 1000
        error = redact_text(error) if error else None
        log_tool_call(request_id, name, status, ms, error)
        return ToolResult(tool=name, status=status, data=data, error=error,
                          duration_ms=round(ms, 1), request_id=request_id)

    t = REGISTRY.get(name)
    if t is None:
        return done("invalid", error=f"unknown tool '{name}'")
    problem = validate_args(t, args)
    if problem:
        return done("invalid", error=problem)
    if t.mutating and args.get("confirm") is not True:
        return done("denied", error="state-changing operation requires explicit confirmation (confirm=true)")

    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(t.fn, **args)
    try:
        data = fut.result(timeout=timeout or get_settings().tool_timeout_s)
        return done("ok", data=data)
    except FutureTimeout:
        return done("timeout", error=f"tool exceeded {timeout or get_settings().tool_timeout_s}s")
    except PermissionDenied as e:
        return done("denied", error=str(e))
    except ServiceUnavailable as e:
        return done("unavailable", error=str(e))
    except ToolError as e:
        return done("error", error=str(e))
    except Exception as e:  # noqa: BLE001 - surface any adapter failure as a structured error
        return done("error", error=f"{type(e).__name__}: {e}")
    finally:
        pool.shutdown(wait=False)
