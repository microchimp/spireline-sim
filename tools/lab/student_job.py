"""The lab's `student` job command: lane 3's fit, legs and candidate (`rl/run_student.py`), with the run loop's freeze honoured.

    D:/spireline/venv-gpu/Scripts/python.exe tools/lab/student_job.py --lab LAB --shards-root run-teacher --prior <incumbent's prior> \
        --human-shards human-prior/decisions-v1 --out <job>/student --base agent/current.json --agent-root agent --target N --tag student \
        --records run-teacher --result <job>/result.json

It is `run_student.py round` with three differences the loop needs, and no other: (1) a kind in `LAB/student/frozen.json` (it failed the human bar in two
rounds running, `tools/lab/jobs.update_human_bar`) is left out of the fit (the loader's `kinds`) and of the candidate; (2) it writes
`result.json` (the rounds' verdicts per kind, the candidate's folder under the agent root, the labelled screens the fit used: what
`jobs.student_finish` reads to write the marker, count the failures and freeze), and `run-bar.md`, the run-level human bar beside it (the human
records' figures and the copy's from the teacher's recorded runs, `rl/run_human_bar.py`); (3) exit 0 when a candidate was built, 3 when no kind
passed both legs (a finished round, not a failure: the runner's `ok_rcs` for the kind). The marker is not written here: the runner writes it once
this process has exited cleanly. BelowNormal; one process; the fit is on the card, the legs are CPU for under half an hour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.lab import jobs  # noqa: E402


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def run_bar_text(records_root: str | None, copy_dirs: list[Path] | None = None) -> str:
    """The run-level human bar beside the copy's figures from the teacher's recorded runs; a text that says why when there is none. Never raises."""
    try:
        import run_human_bar as hb  # noqa: PLC0415

        root = Path(records_root) if records_root else None
        dirs = copy_dirs if copy_dirs is not None else ([] if root is None else
                                                        [*root.glob("*/jobs/*/shards"), *root.glob("*/*/jobs/*/shards"), *root.glob("jobs/*/shards")])
        copy = hb.copy_from_records(dirs) if dirs else None
        return hb.beside_text(hb.read_human(), copy, label="copy (the teacher's recorded runs)")
    except Exception as e:  # noqa: BLE001
        return f"run-level human bar: not available ({type(e).__name__})\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lab", required=True)
    ap.add_argument("--shards-root", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--human-shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--agent-root", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--tag", default=jobs.STUDENT_TAG)
    ap.add_argument("--why", default="run student: the teacher's labels and the human choices (D56)")
    ap.add_argument("--records", help="the teacher store root: its jobs/<job>/shards hold the copy's recorded runs (the run-level bar's copy side)")
    ap.add_argument("--device")
    ap.add_argument("--result", required=True)
    a = ap.parse_args(argv)

    import run_student as rs  # noqa: PLC0415 -- the loop's lane 3: imports numpy, torch only where it fits

    rs._priority()  # noqa: SLF001
    result_path, out = Path(a.result), Path(a.out)
    frozen = jobs.read_frozen(a.lab)
    kinds = tuple(k for k in rs.KINDS if k not in frozen)
    result: dict[str, Any] = {"frozen": sorted(frozen), "fitted_kinds": list(kinds), "built": False, "candidate": None, "kinds_passing": {}, "verdict": {},
                              "target": a.target, "tag": a.tag}
    if not kinds:
        result["why"] = "every kind is frozen"
        _write(result_path, result)
        print("every kind is frozen: nothing to fit")
        return 3
    orig = rs.load_shards
    rs.load_shards = lambda root, **kw: orig(root, **{**kw, "kinds": kinds})  # the frozen kinds are not read, so not fitted
    try:
        legs = rs.run_fit(a.shards_root, a.prior, out, human_dir=a.human_shards, cfg=rs.FitConfig(), device=a.device)
    finally:
        rs.load_shards = orig
    fit_doc = json.loads((out / rs.FIT_FILE).read_text(encoding="utf-8"))
    result["decisions"] = int((fit_doc.get("data") or {}).get("decisions") or 0)
    result["kinds_passing"] = {k: v for k, v in legs["kinds_passing"].items() if k in kinds}
    result["verdict"] = {k: {key: v.get(key) for key in ("pass", "leg1", "leg2", "lambda", "why")} for k, v in legs["verdict"].items()}
    cells = ((legs.get("leg2") or {}).get("kinds")) or {}
    for k in rs.HUMAN_KINDS:  # lane 3 calls an unmeasured human bar a failed one (leg2 False); the loop's stops count only a bar that was measured
        if k in result["verdict"]:
            result["verdict"][k]["leg2_measured"] = bool((cells.get(k) or {}).get("won") and (cells.get(k) or {}).get("all"))
    bar = run_bar_text(a.records)
    (result_path.parent / "run-bar.md").write_text(bar, encoding="utf-8")
    print(json.dumps({"kinds_passing": result["kinds_passing"], "decisions": result["decisions"]}))
    if not result["kinds_passing"]:
        result["why"] = "no kind passed both legs with a measured lambda"
        _write(result_path, result)
        print("no kind passed both legs: no candidate is built")
        return 3
    cand = rs.build_student_candidate(out, a.base, a.agent_root, a.target, why=a.why, tag=a.tag, kinds=kinds)
    try:
        rel = Path(cand).resolve().relative_to(Path(a.agent_root).resolve()).as_posix()
    except ValueError:
        rel = f"gen{a.target}/candidate-{a.tag}"
    result.update(built=True, candidate=rel, generation=a.target)
    _write(result_path, result)
    print(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
