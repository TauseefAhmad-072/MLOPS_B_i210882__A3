"""Structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import re
import sys
import time

SENSITIVE_KEYS = ("password", "token", "secret", "api_key", "apikey", "authorization", "credential")
_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)((?:password|token|secret|api[_-]?key)\s*[=:]\s*)[^\s,;\"']+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),
]


def redact_text(text: str) -> str:
    for p in _PATTERNS:
        text = p.sub(lambda m: (m.group(1) if m.groups() else "") + "***REDACTED***", text)
    return text


def redact(obj):
    if isinstance(obj, dict):
        return {k: ("***REDACTED***" if any(s in str(k).lower() for s in SENSITIVE_KEYS) else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
                   "level": record.levelname, "logger": record.name, "msg": redact_text(record.getMessage())}
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(redact(extra))
        return json.dumps(payload, default=str)


def get_logger(name: str = "mlops-agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = True
    return logger


def log_tool_call(request_id: str, tool: str, status: str, duration_ms: float, error: str | None = None):
    get_logger().info("tool_call", extra={"fields": {
        "event": "tool_call", "request_id": request_id, "tool": tool,
        "status": status, "duration_ms": round(duration_ms, 1), "error": error}})
