"""Resolve the existing modular folders when deployed from the repository root."""
from pathlib import Path
__path__.append(str(Path(__file__).resolve().parent.parent))
