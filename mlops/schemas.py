"""Dataclasses used by the local MLOps persistence layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelVersion:
    """A versioned model artifact and its evaluation metadata."""

    version: str
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    artifact_path: str = ""
    created_at: str = ""


@dataclass
class RegistryModel:
    """A model entry tracked through its lifecycle."""

    model_id: str
    name: str
    model_type: str
    framework: str
    stage: str
    metrics: dict[str, float] = field(default_factory=dict)
    last_trained: str = ""
    owner: str = ""
    tags: list[str] = field(default_factory=list)
    description: str = ""
    features: list[str] = field(default_factory=list)
    training_snapshot: dict[str, Any] = field(default_factory=dict)
    versions: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    archived: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model record."""
        return asdict(self)


@dataclass
class TrainingRun:
    """A single tracked training run."""

    run_id: str
    model_id: str
    started_at: str
    ended_at: str = ""
    duration_seconds: float = 0.0
    status: str = "Running"
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run record."""
        return asdict(self)


@dataclass
class AuditEvent:
    """An append-only lifecycle or persistence event."""

    event_id: str
    action: str
    entity_type: str
    entity_id: str
    actor: str
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit event."""
        return asdict(self)