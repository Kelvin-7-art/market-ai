"""Seed realistic local MLOps records for the seven model families."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from market_ai.mlops.audit import append_audit
from market_ai.mlops.paths import PREDICTIONS_PATH, REGISTRY_PATH, ensure_storage
from market_ai.mlops.registry import save_registry
from market_ai.mlops.runs import append_run


def demo_models() -> list[dict[str, Any]]:
    """Return deterministic model registry records."""
    specs = [
        ("model_rf_base", "Random Forest Base", "Random Forest", "Production", 0.834, "Aisha", ["AAPL", "baseline"]),
        ("model_xgb_adv", "XGBoost Advanced", "XGBoost", "Staging", 0.912, "Maya", ["SPY", "factors"]),
        ("model_lgbm_opt", "LightGBM Optimized", "LightGBM", "Production", 0.887, "Ravi", ["macro", "production"]),
        ("model_svm_base", "SVM Base", "SVM", "Development", 0.801, "Kai", ["classification"]),
        ("model_nn_deep", "Neural Network Deep", "Neural Network", "Staging", 0.864, "Aisha", ["sentiment"]),
        ("model_lstm_tuned", "LSTM Tuned", "LSTM", "Production", 0.923, "Maya", ["forecasting", "BTC"]),
        ("model_transformer", "Transformer Fine-tuned", "Transformer", "Archived", 0.779, "Ravi", ["NLP", "legacy"]),
    ]
    created = datetime(2026, 9, 12, 8, tzinfo=timezone.utc)
    records = []
    for index, (model_id, name, model_type, stage, accuracy, owner, tags) in enumerate(specs):
        trained = created - timedelta(days=index)
        version = f"v{2 if index < 3 else 1}.0.{index + 1}"
        metrics = {"accuracy": accuracy, "auc": min(0.98, accuracy + 0.06), "rmse": round((1 - accuracy) * 0.35, 4)}
        records.append(
            {
                "model_id": model_id,
                "name": name,
                "model_type": model_type,
                "framework": "scikit-learn" if model_type in {"Random Forest", "SVM"} else "PyTorch" if model_type in {"Neural Network", "LSTM", "Transformer"} else "XGBoost",
                "stage": stage,
                "metrics": metrics,
                "last_trained": trained.isoformat(),
                "owner": owner,
                "tags": tags,
                "description": f"{name} for financial market research.",
                "features": ["MA20", "RSI_14", "MACD", "Volume_Ratio"],
                "training_snapshot": {"rows": 1208 - index * 31, "date_range": "2020-01-01 → 2026-09-12", "feature_count": 4, "dataset": "market_factors_v1"},
                "versions": [{"version": version, "metrics": metrics, "hyperparameters": {"random_state": 42, "n_estimators": 100 + index * 25}, "artifact_path": f"artifacts/{model_id}/{version}/model.joblib", "created_at": trained.isoformat()}],
                "artifacts": [{"name": "model.joblib", "path": f"artifacts/{model_id}/{version}/model.joblib", "type": "pickle/joblib"}, {"name": "feature_schema.json", "path": f"artifacts/{model_id}/{version}/feature_schema.json", "type": "schema"}],
                "created_at": created.isoformat(),
                "archived": stage == "Archived",
            }
        )
    return records


def seed_demo(force: bool = False) -> None:
    """Create demo registry, run history, and predictions on first use."""
    ensure_storage()
    if REGISTRY_PATH.exists() and not force:
        try:
            if json.loads(REGISTRY_PATH.read_text(encoding="utf-8")):
                return
        except json.JSONDecodeError:
            pass
    models = demo_models()
    save_registry(models, action="demo.seeded", entity_id="demo")
    for index, model in enumerate(models):
        append_run(
            {
                "run_id": f"run_demo_{index + 1:02d}",
                "model_id": model["model_id"],
                "started_at": model["last_trained"],
                "ended_at": model["last_trained"],
                "duration_seconds": 4.2 + index * 0.7,
                "status": "Succeeded" if index != 4 else "Failed",
                "metrics": model["metrics"],
                "hyperparameters": model["versions"][0]["hyperparameters"],
                "git_commit": f"demo{index + 1:05d}",
            }
        )
    with PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
        for index in range(30):
            handle.write(json.dumps({"model_id": "model_rf_base", "timestamp": (datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(), "volume": 80 + index * 3, "psi": round(0.04 + index * 0.0015, 4), "health": "Healthy" if index < 22 else "Warning"}) + "\n")
    append_audit("predictions.seeded", "predictions", "demo", {"rows": 30})


def ensure_demo_seeded() -> None:
    """Initialize demo records without overwriting existing local state."""
    seed_demo(force=False)