"""Model stage transitions and lifecycle validation."""

from __future__ import annotations

from typing import Any

from market_ai.mlops.registry import get_model, save_registry, load_registry


STAGES = ("Development", "Staging", "Production", "Archived")
ALLOWED_TRANSITIONS = {
    "Development": {"Staging", "Archived"},
    "Staging": {"Development", "Production", "Archived"},
    "Production": {"Staging", "Archived"},
    "Archived": {"Development"},
}


def promote_model(model_id: str, target_stage: str) -> dict[str, Any]:
    """Move a model through a validated lifecycle transition."""
    if target_stage not in STAGES:
        raise ValueError(f"Invalid stage: {target_stage}")
    models = load_registry()
    model = next((item for item in models if item.get("model_id") == model_id), None)
    if model is None:
        raise ValueError(f"Unknown model: {model_id}")
    current = str(model.get("stage", "Development"))
    if target_stage not in ALLOWED_TRANSITIONS.get(current, set()):
        raise ValueError(f"Cannot move {current} directly to {target_stage}")
    model["stage"] = target_stage
    model["archived"] = target_stage == "Archived"
    save_registry(models, action="model.stage_changed", entity_id=model_id)
    return model


def archive_model(model_id: str) -> dict[str, Any]:
    """Soft-archive a model while retaining its versions and history."""
    return promote_model(model_id, "Archived")