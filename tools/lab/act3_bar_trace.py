"""Replay one judged run on the copy and print its boss fights' enemy health, decision by decision (the Act 3 week's lane 1).

This is how the Test Subject's three forms were seen: at the first Act 3 boss the health bar's fraction reads 1.0 when the first form
falls to 0 and the boss then stands at its second form's full health. One process, BelowNormal priority, the CPU prior path (a replay is
exact there only: it reproduces the judged row's floor, end and bar). Needs the native library (`out/` of this tree, or `--lib`); about 30 s
to a few minutes a run. Never while the lab's nightly block runs (`LAB/nightly/active`) or another lane holds QUIET.lock.

    D:/spireline/src/sts2/.venv/Scripts/python.exe -I tools/lab/act3_bar_trace.py --gen-dir <agent>/gen18/candidate-fight-v3 \
        --seed <seed of a judged row> --ix <its ix> --out trace.json

`--seed` and `--ix` are the row's own (`runs.jsonl`); `--config` is the arm's search config (default `a3-v1`, the agent's own stakes schedule
applies). Prints, for each boss decision, the floor and the enemy slots as (slot, health, max health).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen-dir", required=True, help="the agent folder that played the row (or a current.json)")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--ix", type=int, required=True)
    ap.add_argument("--config", default="a3-v1")
    ap.add_argument("--lib", help="the native library folder (STS2_LIB_PATH)")
    ap.add_argument("--out", help="write {row, trace} as JSON")
    a = ap.parse_args(argv)
    if os.name == "nt":  # BelowNormal through the Windows priority class
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    os.environ["STS2_LIB_PATH"] = a.lib or os.environ.get("STS2_LIB_PATH") or str(ROOT / "out")
    sys.path.insert(0, str(ROOT / "rl"))
    import gen0  # noqa: PLC0415

    gen0._setup()
    import gen_loop  # noqa: PLC0415
    import run_fields  # noqa: PLC0415
    from run_train import engine  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    trace: list[dict] = []

    def hook(k, root, session, pos):  # called at each combat decision before its search; it only reads
        if session.info["node_type"] == engine.vocab.NODE_BOSS:
            enemies = run_fields.obs_enemies(session.env._run_obs_buf, native.OBS_ENEMY_OFFSET, native.OBS_ENEMY_SLOT_SIZE, native.MAX_ENEMIES)
            trace.append({"k": k, "floor": int(session.info["floor"]), "enemies": [list(e) for e in enemies]})
        return False

    t0 = time.time()
    row = gen0.play_emu_run(a.seed, a.ix, a.config, 10, hook=hook, agent=gen_loop.load_agent(a.gen_dir))
    row = {k: v for k, v in row.items() if k in ("ix", "win", "floor", "end", "decisions", "wall", "boss_damage_acts", "nodes")}
    print(json.dumps(row))
    for t in trace:
        print(t["k"], t["floor"], t["enemies"])
    if a.out:
        Path(a.out).write_text(json.dumps({"row": row, "trace": trace, "wall": round(time.time() - t0, 1)}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
