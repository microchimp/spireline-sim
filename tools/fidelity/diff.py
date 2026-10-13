"""Field-level differences between two canonical documents, first difference first."""

from __future__ import annotations


def diff(a, b, path: str = "", out: list | None = None, limit: int = 50) -> list:
    """Every (json path, left, right) that differs, depth first, in document order."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for key in list(a) + [k for k in b if k not in a]:
            if key not in a or key not in b:
                out.append((f"{path}.{key}", a.get(key, "<missing>"), b.get(key, "<missing>")))
            else:
                diff(a[key], b[key], f"{path}.{key}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            if i >= len(a) or i >= len(b):
                out.append((f"{path}[{i}]", a[i] if i < len(a) else "<missing>", b[i] if i < len(b) else "<missing>"))
            else:
                diff(a[i], b[i], f"{path}[{i}]", out, limit)
    elif a != b:
        out.append((path or ".", a, b))
    return out


def first_divergence(left: list, right: list, key: str = "hash") -> int | None:
    """Index of the first step whose `key` differs between two traces, or None if they agree."""
    for i in range(max(len(left), len(right))):
        if i >= len(left) or i >= len(right) or left[i][key] != right[i][key]:
            return i
    return None
