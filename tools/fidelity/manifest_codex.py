"""Counts of the Codex runs that touch each game model: the one reader of the Codex dataset the port manifest uses.

    python -I tools/fidelity/manifest_codex.py <runs.jsonl.gz> <out.json> [--character IRONCLAD] [--ascension 10]

The Codex dataset is untrusted data from outside. This file is self-contained (standard library only, no import
from the repo) so it runs under ``python -I``, reads each line as JSON and nothing else, and writes AGGREGATES ONLY:
how many runs matched and, per model key (``kind:ID``), how many of those runs name it. No seed, run id, floor or
other per-run value ever reaches the output. manifest.py starts it as a child process.
"""

from __future__ import annotations

import gzip
import json
import re
import sys

FORMAT = 1
# The id prefixes the dataset uses and the model kind each one names. Anything else is ignored.
PREFIX_KIND = {"CARD": "card", "RELIC": "relic", "POTION": "potion", "MONSTER": "monster",
               "ENCHANTMENT": "enchantment", "AFFLICTION": "affliction", "POWER": "power", "ORB": "orb"}
ID = re.compile(r"^([A-Z]+)\.([A-Z0-9_]+)$")


def models_of(run) -> set:
    """The model keys (``kind:ID``) a run names anywhere in its values (not its keys)."""
    out: set = set()
    stack = [run]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, str) and len(x) < 96:
            m = ID.match(x)
            if m and m.group(1) in PREFIX_KIND:
                out.add(f"{PREFIX_KIND[m.group(1)]}:{m.group(2)}")
    return out


def matches(run, character: str, ascension: int | None) -> bool:
    if not isinstance(run, dict):
        return False
    if ascension is not None and run.get("ascension") != ascension:
        return False
    players = run.get("players")
    return isinstance(players, list) and len(players) == 1 and isinstance(players[0], dict) \
        and players[0].get("character") == f"CHARACTER.{character}"


def aggregate(lines, character: str = "IRONCLAD", ascension: int | None = 10) -> dict:
    """``{"format", "runs", "scanned", "by_model": {key: runs naming it}}`` over an iterable of JSON lines."""
    by_model: dict = {}
    runs = scanned = 0
    needle = f"CHARACTER.{character}"
    for line in lines:
        scanned += 1
        if needle not in line:
            continue
        try:
            run = json.loads(line)
        except ValueError:
            continue
        if not matches(run, character, ascension):
            continue
        runs += 1
        for key in models_of(run):
            by_model[key] = by_model.get(key, 0) + 1
    return {"format": FORMAT, "runs": runs, "scanned": scanned, "by_model": dict(sorted(by_model.items()))}


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    opts = {"--character": "IRONCLAD", "--ascension": "10"}
    pos = []
    i = 0
    while i < len(a):
        if a[i] in opts and i + 1 < len(a):
            opts[a[i]] = a[i + 1]
            i += 2
        else:
            pos.append(a[i])
            i += 1
    if len(pos) != 2:
        print(__doc__)
        return 2
    src, dst = pos
    opener = gzip.open if src.endswith(".gz") else open
    with opener(src, "rt", encoding="utf-8", errors="replace") as f:
        doc = aggregate(f, opts["--character"], int(opts["--ascension"]))
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=0)
    print(f"codex: {doc['runs']} runs matched of {doc['scanned']} lines, {len(doc['by_model'])} models named")
    return 0


if __name__ == "__main__":
    sys.exit(main())
