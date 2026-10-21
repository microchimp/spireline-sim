"""Spec files for the speed spike (referee/design/real-game-speed.md), written into the spike folder.

    python referee/spike/mkspecs.py <a lab fights batch spec, e.g. LAB/runs/<job>/fights/<name>-b00.json> [--out D:/spireline/referee/spike]

fights50.json: the batch's fights with the game's per-action checksums on; fights50_nl.json: the same on the nodeless
host with no frame sleep (full-host replay on error); runs4.json: four act-1 whole runs (runs.py's seeds from 1048).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runs  # noqa: E402  (referee/runs.py: the seed list)

ap = argparse.ArgumentParser()
ap.add_argument("batch", type=Path)
ap.add_argument("--out", type=Path, default=Path("D:/spireline/referee/spike"))
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
src = json.loads(a.batch.read_text(encoding="utf-8"))["runs"]
for s in src:
    s["checksums"] = True
(a.out / "fights50.json").write_text(json.dumps({"runs": src}), encoding="utf-8")
(a.out / "fights50_nl.json").write_text(json.dumps({"runs": src, "nodeless": True, "frame_sleep_usec": 0,
                                                    "nodeless_on_error": "full"}), encoding="utf-8")
rs = [{"kind": "run", "seed": s, "character": "IRONCLAD", "ascension": 10, "chooser_seed": 1048 + i, "buff_max_hp": 2000,
       "avoid": [], "stop_after_act": 1, "max_decisions": 1500, "potions": False} for i, s in enumerate(runs.seeds(4, 1048))]
(a.out / "runs4.json").write_text(json.dumps({"runs": rs}), encoding="utf-8")
print(f"{len(src)} fights, {len(rs)} runs -> {a.out}")
