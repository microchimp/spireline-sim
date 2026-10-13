#!/usr/bin/env python3
"""Make a private game copy for a referee session without copying 2.8 GB: hard links to the verified copy.

    python referee/linkcopy.py --src D:/spireline/referee/game --dst D:/spireline/referee/game-<name>

Every file is hard-linked (same volume, no extra disk), except mods/ and override.cfg, which are the
session's own (run_referee.install writes them). The game only reads its own files, so the verified
copy is untouched; a session that rebuilds the driver can then install it without waiting for other
sessions that hold the shared copy's mods/SpirelineReferee.dll open.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args()
    src, dst = a.src.resolve(), a.dst.resolve()
    skip = {"mods", "override.cfg"}
    n = 0
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        if rel == Path("."):
            dirs[:] = [d for d in dirs if d not in skip]
            files = [f for f in files if f not in skip]
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            target = dst / rel / f
            if not target.exists():
                os.link(Path(root) / f, target)
                n += 1
    print(f"linked {n} files into {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
