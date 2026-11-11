"""Deliverable 2a: export, load, export again, and diff the two documents (doc-only, no replay).

    python -I doc_start_roundtrip.py DOCS_DIR
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_start_lib as L  # noqa: E402,F401


def norm(d):
    """The unordered draw middle is a multiset: sort it, so the round trip is compared up to that."""
    d = json.loads(json.dumps(d))
    u = d.get("piles", {}).get("draw", {}).get("unordered")
    if isinstance(u, list):
        d["piles"]["draw"]["unordered"] = sorted(u, key=lambda c: json.dumps(c, sort_keys=True))
    return d


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append((f"{path}/{k}", "<absent>", b[k]))
            elif k not in b:
                out.append((f"{path}/{k}", a[k], "<absent>"))
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path + "[len]", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append((path, a, b))
    return out


def main():
    from sts2_gym import native

    rows = []
    for p in sorted(Path(sys.argv[1]).glob("*.json")):
        d = json.loads(p.read_text())
        doc = d["doc"]
        h = native.load_combat_snapshot(doc, 11)
        try:
            doc2 = native.get_state_json(h)
        finally:
            native.destroy(h)
        df = diff(norm(doc), norm(doc2))
        rows.append((p.name, df))
        print(p.name, "differences:", len(df))
        for path, x, y in df[:40]:
            print("   ", path, json.dumps(x)[:80], "->", json.dumps(y)[:80])


if __name__ == "__main__":
    main()
