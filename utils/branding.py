"""Shared branding image for the sidebar and Dashboard header."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def logo_data_uri() -> str:
    """Read the bundled PNG using a path independent of the working directory."""
    path = Path(__file__).resolve().parents[1] / "assets" / "ai_framework_logo.png"
    if not path.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def logo_html(size: int = 48) -> str:
    """Return an image tag without stretching or cropping the original artwork."""
    source = logo_data_uri()
    if not source:
        return ""
    return (
        f'<img src="{source}" alt="TradeGPT logo" '
        f'width="{size}" height="{size}" '
        f'style="width:{size}px;height:{size}px;object-fit:contain;'
        'flex-shrink:0;display:block;border-radius:10px;" />'
    )
