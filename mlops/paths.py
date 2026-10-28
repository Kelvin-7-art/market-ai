"""Centralized paths for local MLOps persistence."""

from __future__ import annotations

from pathlib import Path


MLOPS_DIR = Path(__file__).resolve().parents[1] / "data" / "mlops"
REGISTRY_PATH = MLOPS_DIR / "registry.json"
RUNS_PATH = MLOPS_DIR / "runs.jsonl"
AUDIT_PATH = MLOPS_DIR / "audit.jsonl"
PREDICTIONS_PATH = MLOPS_DIR / "predictions.jsonl"


def ensure_storage() -> None:
    """Create the local MLOps directory on first use."""
    MLOPS_DIR.mkdir(parents=True, exist_ok=True)