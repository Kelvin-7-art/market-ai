"""CRUD operations for the local model registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_ai.mlops.audit import append_audit
from market_ai.mlops.paths import REGISTRY_PATH, ensure_storage


def load_registry() -> list[dict[str, Any]]:
    """Load registered models, returning an empty list when uninitialized."""
    if not REGISTRY_PATH.exists():
        return []
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return payload if isinstance(payload, list) else []


def save_registry(models: list[dict[str, Any]], action: str = "registry.updated", entity_id: str = "registry") -> None:
    """Atomically save registry state and audit the write."""
    ensure_storage()
    temporary = Path(f"{REGISTRY_PATH}.tmp")
    temporary.write_text(json.dumps(models, indent=2, default=str), encoding="utf-8")
    temporary.replace(REGISTRY_PATH)
    append_audit(action, "registry", entity_id, {"model_count": len(models)})


def get_model(model_id: str) -> dict[str, Any] | None:
    """Return one model by ID."""
    return next((model for model in load_registry() if model.get("model_id") == model_id), None)


def upsert_model(model: dict[str, Any], action: str = "model.upserted") -> dict[str, Any]:
    """Create or replace one model record."""
    models = load_registry()
    existing = next((index for index, item in enumerate(models) if item.get("model_id") == model.get("model_id")), None)
    if existing is None:
        models.append(model)
    else:
        models[existing] = model
    save_registry(models, action=action, entity_id=str(model.get("model_id", "unknown")))
    return model


def register_version(
    model_id: str,
    metrics: dict[str, float],
    hyperparameters: dict[str, Any],
    artifact_path: str,
) -> dict[str, Any]:
    """Register the next semantic version for an existing model."""
    models = load_registry()
    model = next((item for item in models if item.get("model_id") == model_id), None)
    if model is None:
        raise ValueError(f"Unknown model: {model_id}")
    current = [str(version.get("version", "v0.0.0")) for version in model.get("versions", [])]
    next_number = len(current) + 1
    version = {
        "version": f"v1.0.{next_number}",
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "artifact_path": artifact_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    model.setdefault("versions", []).append(version)
    model["metrics"] = metrics
    model["last_trained"] = version["created_at"]
    save_registry(models, action="model.version_registered", entity_id=model_id)
    return version