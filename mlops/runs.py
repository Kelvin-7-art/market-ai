"""Append-only training run tracking."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from market_ai.mlops.audit import append_audit
from market_ai.mlops.paths import RUNS_PATH, ensure_storage


def append_run(run: dict[str, Any]) -> dict[str, Any]:
    """Append a training run record and audit the write."""
    ensure_storage()
    record = {"run_id": run.get("run_id", f"run_{uuid4().hex[:10]}"), **run}
    with RUNS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    append_audit("run.appended", "run", str(record["run_id"]), {"model_id": record.get("model_id"), "status": record.get("status")})
    return record


def list_runs(model_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Read newest training runs, optionally scoped to one model."""
    if not RUNS_PATH.exists():
        return []
    records = [json.loads(line) for line in RUNS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if model_id:
        records = [record for record in records if record.get("model_id") == model_id]
    return records[-limit:][::-1]