"""The held halves of the information-parity blocks for the human shards (D59 lane D, follow-up; `docs/rl/HUMAN_PRIOR.md` "v4").

The human shards (`human-prior/decisions-v3`, `shard-NNNNN.npz`) store each row's afterstate as counts: the base vector's blocks E (deck copies, base and upgraded), F (relics) and G (potions) as uint8,
1,513 columns. The held halves of P1, P2 and P4 are linear in those counts (`rl/parity_blocks.py` `held_from_counts`), so no replay and no engine are needed:

    python rl/human_parity.py --src <human-prior>/decisions-v3 --out <human-prior>/decisions-v3-parity

writes `shard-NNNNN.rows_parity.npz` in a **separate folder** (the source is never written): `relics_held` (n, 45: the effect tags of the relics held, summed), `roles_deck` (n, 10: the deck's role counts and
its size) and `potions_held` (n, 34), float32, rows in the source shard's order, and `parity-manifest.json` (counts, widths, layout version). Arrays only: no run id is copied. A shard that has a file of the
right length is left (idempotent, resumable); a shard whose runs fall in the exam fifth by the fixed hash rule is refused (`human_starts.is_exam`, the count is reported and no id is printed; the ids file sits under the corpus folder and is not
read: the shards were built with the full rule, this is the second guard). One process at BelowNormal, reading and writing only; no QUIET line (a re-encode never takes one).
The option's own relic, card and potion halves and the event payoff need each row's identity (`ident`, the offer kind, the event key) and are a later step.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import human_starts  # noqa: E402
import parity_blocks as pb  # noqa: E402
import parity_potions as pp  # noqa: E402
import parity_relics as pr  # noqa: E402

KEYS = ("relics_held", "roles_deck", "potions_held")
WIDTHS = {"relics_held": pr.RELIC_TAG_WIDTH, "roles_deck": 10, "potions_held": pp.POTION_TAG_WIDTH}
SUFFIX = ".rows_parity.npz"
MANIFEST = "parity-manifest.json"


def _finished(f: Path, n_rows: int) -> bool:
    """Whether `f` is a complete output for a shard of `n_rows` rows."""
    if not f.is_file():
        return False
    try:
        with np.load(f, allow_pickle=False) as z:
            return all(k in z.files and z[k].shape == (n_rows, WIDTHS[k]) and z[k].dtype == np.float32 for k in KEYS)
    except (OSError, ValueError):
        return False


def reencode(src: Path | str, out: Path | str, *, log=None) -> dict:  # noqa: ANN001
    """The held halves of every `shard-*.npz` of `src` into `out` (created). Returns {shards, rows, skipped, seconds}. Raises ValueError for a shard of the wrong width or `src` = `out`, and
    AssertionError for a shard with a run of the exam fifth (nothing is written for that shard)."""
    src, out = Path(src), Path(out)
    if src.resolve() == out.resolve():
        raise ValueError("the output folder must not be the source folder")
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("shard-*.npz"))
    t0 = time.time()
    rows = skipped = 0
    for k, p in enumerate(shards):
        target = out / (p.stem + SUFFIX)
        with np.load(p, allow_pickle=False) as z:
            n = int(z["ident"].shape[0])
            bad = sum(1 for x in z["run_ids"] if human_starts.is_exam(str(x), frozenset()))
            if bad:  # the count only: a run id is never printed
                raise AssertionError(f"{p.name}: {bad} runs belong to the exam fifth by the fixed hash rule: nothing is written for this shard")
            rows += n
            if _finished(target, n):
                skipped += 1
                continue
            held = pb.held_from_counts(z["counts"])
        if len(held["relics"]) != n:
            raise ValueError(f"{p.name}: {len(held['relics'])} rows of counts against {n} rows")
        tmp = out / (p.stem + ".tmp.npz")
        np.savez_compressed(tmp, relics_held=held["relics"], roles_deck=held["roles"], potions_held=held["potions"])
        os.replace(tmp, target)
        if log and (k + 1) % 25 == 0:
            log(f"human parity: {k + 1} of {len(shards)} shards, {time.time() - t0:.0f} s")
    res = {"shards": len(shards), "rows": rows, "skipped": skipped, "seconds": round(time.time() - t0, 1)}
    doc = {"kind": "human-parity-held", "parity_version": pb.PARITY_VERSION, "source": src.name, "blocks": WIDTHS, "shards": len(shards), "rows": rows,
           "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "note": "held halves of P1 (relic tags), P2 (deck role counts and size) and P4 (potion tags) from the shards' counts; no run id is stored; the exam fifth is not in the source"}
    tmpm = out / (MANIFEST + ".tmp")
    tmpm.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmpm, out / MANIFEST)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="the human shards' folder (decisions-v3)")
    ap.add_argument("--out", required=True, help="a separate folder for the parity files")
    a = ap.parse_args(argv)
    with contextlib.suppress(Exception):
        from sts2_gym import procutil  # noqa: PLC0415

        procutil.lower_priority()  # BelowNormal (Windows priority class)
    res = reencode(a.src, a.out, log=lambda m: print(m, flush=True))
    print("human parity: " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
