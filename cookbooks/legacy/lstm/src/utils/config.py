"""Minimal YAML config loader (falls back to a tiny parser if PyYAML is absent)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _tiny_yaml(text: str) -> dict[str, Any]:
    """Parse a strict subset of YAML: nested mappings via indentation, scalars only."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        val = val.strip()
        if val == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce(val)
    return root


def _coerce(val: str) -> Any:
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    if val.lower() in {"null", "none", "~"}:
        return None
    try:
        if any(c in val for c in ".eE") and val.replace("-", "").replace(".", "").replace("e", "").replace("E", "").replace("+", "").isdigit():
            return float(val)
        return int(val)
    except ValueError:
        return val.strip('"').strip("'")


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        return _tiny_yaml(text)
