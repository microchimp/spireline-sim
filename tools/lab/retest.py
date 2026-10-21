"""The retest strategy: a group judged host-suspect only by gaps since fixed is played again on the current tree.

tools/fidelity/lab_groups.py marks such a group ``retest`` and gives it ``retest.spec``: the fight spec of its smallest
example (the shrunk setup, the example's seed). A retest job plays the specs of the groups with no verdict yet (the
oldest first, at most LIMIT per job) and judges each fight against the game:

  diverges   the fight still differs from the game: the host gap was not the cause, the group is real
  identical  the fight matches: the group came from the fixed gap and is stale
  unclear    an error, a mirror gap or a divergence in another field; tried again until MAX_ATTEMPTS

LAB/retests.json maps group id -> {verdict, utc, evidence, attempts, main_sha}. A diverges or identical verdict clears the
group's host suspicion at the next regroup (the verdict stays with the id; a group whose id changes is judged anew).
"""

from __future__ import annotations

import json
from pathlib import Path

TAG = "retest:"
LIMIT = 8
MAX_ATTEMPTS = 3
DECIDED = ("diverges", "identical")
JUDGED = "retest-judged.json"


def read_results(lab: Path) -> dict:
    try:
        doc = json.loads((Path(lab) / "retests.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def write_results(lab: Path, new: dict) -> dict:
    cur = read_results(lab)
    for gid, r in new.items():
        if (cur.get(gid) or {}).get("verdict") in DECIDED and r["verdict"] not in DECIDED:
            continue
        cur[gid] = r
    p = Path(lab) / "retests.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(dict(sorted(cur.items())), indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)
    return cur


def pending(groups: list[dict], results: dict, limit: int = LIMIT) -> list[dict]:
    """[{id, spec}] of the groups waiting for a retest: marked retest, not decided, not out of attempts."""
    out = []
    for g in groups:
        rt = g.get("retest") or {}
        r = results.get(g["id"]) or {}
        if rt.get("spec") and r.get("verdict") not in DECIDED and r.get("attempts", 0) < MAX_ATTEMPTS:
            out.append({"id": g["id"], "spec": rt["spec"], "first_seen_utc": g.get("first_seen_utc") or ""})
    out.sort(key=lambda x: (x["first_seen_utc"], x["id"]))
    return [{"id": x["id"], "spec": x["spec"]} for x in out[:limit]]


def to_specs(items: list[dict]) -> list[dict]:
    return [{**it["spec"], "tag": TAG + it["id"]} for it in items]


def classify(verdicts: list[dict]) -> tuple[str, str]:
    if not verdicts:
        return "unclear", "no fight produced a verdict"
    v = verdicts[0]
    if v.get("error"):
        return "unclear", "the fight gave no comparable result"
    if v.get("identical"):
        return ("unclear", "identical, but the setup mirror had gaps") if v.get("mirror_gap") else ("identical", "the fight matches the game")
    if v.get("mirror_gap"):
        return "unclear", "the setup mirror had gaps"
    return "diverges", f"the fight first differs in {v.get('field')} at step {v.get('step')}"


def judge_all(items: list[dict], verdicts: list[dict], evidence: str, utc: str, sha: str, prior: dict) -> dict:
    by_id: dict[str, list[dict]] = {}
    for v in verdicts:
        tag = str(v.get("tag") or "")
        if tag.startswith(TAG):
            by_id.setdefault(tag[len(TAG):], []).append(v)
    out = {}
    for it in items:
        if it["id"] in by_id:
            verdict, reason = classify(by_id[it["id"]])
            out[it["id"]] = {"verdict": verdict, "reason": reason, "utc": utc, "evidence": evidence, "main_sha": sha,
                             "attempts": (prior.get(it["id"]) or {}).get("attempts", 0) + 1}
    return out


def collect(job_dir: Path, lab: Path, utc: str, sha: str) -> int:
    """After a retest batch: judge the job's verdicts files not judged before into LAB/retests.json (each file once)."""
    job_dir = Path(job_dir)
    try:
        items = json.loads((job_dir / "retest-in" / "pending.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    try:
        done = set(json.loads((job_dir / JUDGED).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        done = set()
    written = 0
    for vf in sorted(job_dir.rglob("*.verdicts.json")):
        rel = vf.relative_to(job_dir).as_posix()
        if rel in done:
            continue
        try:
            doc = json.loads(vf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        verdicts = [v for v in doc if isinstance(v, dict)] if isinstance(doc, list) else []
        new = judge_all(items, verdicts, f"runs/{job_dir.name}/{rel}", utc, sha, read_results(lab))
        write_results(lab, new)
        written += len(new)
        done.add(rel)
        (job_dir / JUDGED).write_text(json.dumps(sorted(done), indent=1) + "\n", encoding="utf-8")
    return written
