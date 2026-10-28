"""Shared, self-describing JSON exports for analytical workbenches."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import date, datetime, timezone
from numbers import Integral, Real
from typing import Any


SCHEMA_VERSION = "1.0"
_WORKBENCH_TYPES = {"forecast", "classification", "sentiment", "risk", "backtest"}


def _utc_iso(value: datetime) -> str:
    """Serialize a datetime in UTC with an ISO-8601 Z suffix."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    """Convert common pandas/NumPy values into JSON-native values."""
    if value is None or isinstance(value, (str, bool)):
        return value

    # Check pandas values before general numeric/date handling.
    try:
        import numpy as np
        import pandas as pd

        if value is pd.NaT:
            return None
        if isinstance(value, pd.DataFrame):
            return _json_safe(value.to_dict(orient="records"))
        if isinstance(value, (pd.Series, pd.Index, np.ndarray)):
            return _json_safe(value.tolist())
        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, pd.Timestamp):
            return _utc_iso(value.to_pydatetime())
    except ImportError:
        pass

    if isinstance(value, datetime):
        return _utc_iso(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def build_envelope(
    *,
    workbench: str,
    model_name: str,
    model_variant: str | None,
    request: dict[str, Any],
    inputs: dict[str, Any],
    results: dict[str, Any],
    metrics: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None,
    caveats: list[str],
    app_name: str = "AI Market Assistant",
    app_version: str = "2.0.0",
) -> dict[str, Any]:
    """Build the common export envelope, normalizing values for JSON consumers."""
    if workbench not in _WORKBENCH_TYPES:
        raise ValueError(f"Unsupported workbench type: {workbench}")
    if not model_name.strip():
        raise ValueError("model_name must not be empty.")
    normalized_caveats = [str(item).strip() for item in caveats if str(item).strip()]
    if not normalized_caveats:
        raise ValueError("Every analysis export must include at least one caveat.")

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_iso(datetime.now(timezone.utc)),
        "app": {
            "name": app_name,
            "version": app_version,
            "workbench": workbench,
            "model_name": model_name,
            "model_variant": model_variant,
        },
        "request": request,
        "inputs": inputs,
        "results": results,
        "metrics": metrics or {},
        "diagnostics": diagnostics or {},
        "caveats": normalized_caveats,
    }
    return _json_safe(envelope)


def to_json(envelope: dict[str, Any]) -> str:
    """Serialize a normalized envelope as readable, Unicode-preserving JSON."""
    return json.dumps(
        _json_safe(envelope),
        indent=2,
        default=str,
        ensure_ascii=False,
        allow_nan=False,
    )


def significance_check(n: int, r: float, alpha: float = 0.05) -> dict[str, Any]:
    """Return the two-tailed Pearson-correlation critical value and decision."""
    sample_count = int(n)
    if sample_count != n or sample_count < 0:
        raise ValueError("n must be a non-negative integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    coefficient = float(r)
    if not math.isfinite(coefficient):
        return {
            "n": sample_count,
            "r": None,
            "r_critical": None,
            "significant": False,
        }
    if not -1 <= coefficient <= 1:
        raise ValueError("r must be between -1 and 1.")
    if sample_count < 3:
        return {
            "n": sample_count,
            "r": coefficient,
            "r_critical": None,
            "significant": False,
        }

    # SciPy is already a transitive dependency of scikit-learn in this project.
    from scipy.stats import t

    degrees_of_freedom = sample_count - 2
    t_critical = float(t.ppf(1 - alpha / 2, degrees_of_freedom))
    r_critical = t_critical / math.sqrt(t_critical**2 + degrees_of_freedom)
    return {
        "n": sample_count,
        "r": coefficient,
        "r_critical": float(r_critical),
        "significant": abs(coefficient) >= r_critical,
    }


def significance_caveat(n: int, r: float, alpha: float = 0.05) -> str:
    """Create an LLM-readable caveat containing the exact significance result."""
    result = significance_check(n, r, alpha)
    decision = (
        "statistically significant"
        if result["significant"]
        else "not statistically significant"
    )
    serialized_result = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return (
        f"Pearson correlation is {decision} at n={result['n']} "
        f"(two-tailed alpha={alpha:g}); significance_check output: "
        f"{serialized_result}."
    )


# ===== Compact LLM export =====

def build_llm_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    """Create a compact copy; never mutate the full analytical export.

    Long arrays are represented by their count and endpoints. Classification
    prediction records and tuning trials are omitted explicitly. No metrics
    are recomputed and no missing values are invented.
    """
    source = _json_safe(envelope)
    omissions: list[str] = []
    skip_keys = {"test_predictions", "cv_results", "signature"}

    def compact(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            out = {}
            for name, item in value.items():
                child = f"{path}.{name}"
                if name in skip_keys:
                    if item:
                        count = f" ({len(item)} items)" if isinstance(item, list) else ""
                        omissions.append(child + count)
                    continue
                out[name] = compact(item, child)
            return out
        if isinstance(value, list):
            if len(value) > 20:
                omissions.append(f"{path}: summarized {len(value)} items using first and last only")
                return {"total_items": len(value), "representation": "endpoints_only",
                        "first": compact(value[0], path + "[0]"),
                        "last": compact(value[-1], path + f"[{len(value)-1}]")}
            return [compact(item, f"{path}[{i}]") for i, item in enumerate(value)]
        if isinstance(value, str) and len(value) > 2000:
            omissions.append(f"{path}: text shortened to 2000 characters")
            return value[:2000] + " [truncated; see full export]"
        return value

    output = {name: compact(source.get(name, {}), name)
              for name in ("app", "request", "inputs", "results", "metrics", "diagnostics")}
    # Promote the recorded importance convention to the visible diagnostics.
    diagnostics = source.get("diagnostics", {})
    selection = diagnostics.get("selection", {}) if isinstance(diagnostics, dict) else {}
    params = selection.get("final_parameters", {}) if isinstance(selection, dict) else {}
    if isinstance(params, dict) and params.get("importance_type") is not None:
        output["diagnostics"].setdefault("importance_type", params["importance_type"])
    output["caveats"] = list(source.get("caveats", []))
    if omissions:
        output["caveats"].append(
            "Compact LLM export. Omitted or summarized: " + "; ".join(omissions)
            + ". Use the full analysis JSON for row-level questions; endpoints are not extrema."
        )
    return output


def render_download_button(
    envelope: dict[str, Any],
    *,
    key: str,
    label: str = "⬇️ Download analysis JSON",
) -> None:
    """Render the common download button with a ticker/workbench/date filename."""
    import streamlit as st

    request = envelope.get("request", {})
    app = envelope.get("app", {})
    ticker = request.get("ticker") or envelope.get("inputs", {}).get("ticker") or "analysis"
    date_stamp = str(envelope.get("generated_at", datetime.now(timezone.utc).isoformat()))[:10]
    filename_parts = (ticker, app.get("workbench", "analysis"), date_stamp)
    filename = "_".join(
        re.sub(r"[^A-Za-z0-9._-]+", "-", str(part)).strip("._-") or "analysis"
        for part in filename_parts
    )
    st.download_button(
        label=label,
        data=to_json(envelope),
        file_name=f"{filename}.json",
        mime="application/json",
        key=key,
    )
    compact_json = to_json(build_llm_envelope(envelope))
    st.download_button(
        label="Download LLM-ready JSON",
        data=compact_json,
        file_name=f"{filename}_llm.json",
        mime="application/json",
        key=f"{key}_llm",
        help="Compact results for an LLM. Full row-level data remains in the original download.",
    )
    st.caption("LLM-ready JSON keeps key results and notes omitted detail. Provider token limits still apply.")


__all__ = [
    "build_llm_envelope",
    "SCHEMA_VERSION",
    "build_envelope",
    "render_download_button",
    "significance_caveat",
    "significance_check",
    "to_json",
]
