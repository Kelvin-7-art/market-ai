"""Append-only audit logging for local MLOps writes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from market_ai.mlops.paths import AUDIT_PATH, ensure_storage


def append_audit(
    action: str,
    entity_type: str,
    entity_id: str,
    details: dict[str, Any] | None = None,
    actor: str = "demo-user",
) -> dict[str, Any]:
    """Append one auditable write event and return it."""
    ensure_storage()
    event = {
        "event_id": f"audit_{uuid4().hex[:10]}",
        "action": action,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "actor": actor,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str) + "\n")
    return event


def read_audit(limit: int = 100) -> list[dict[str, Any]]:
    """Read the newest audit events."""
    if not AUDIT_PATH.exists():
        return []
    lines = AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()][::-1]