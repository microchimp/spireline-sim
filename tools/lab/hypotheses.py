"""The hypotheses strategy: static audit findings, tested by the real game on idle CPU.

A setup file is ``<HYP>/<source>/<name>.json`` (HYP = referee/hypotheses beside the backlog folder), one object or a
list of objects:

  id        unique over all sources, e.g. "order-M1"
  source    the audit it came from (the folder name)
  claim     one line: what the emulator is said to do differently
  spec      the fight spec, as scenarios.py's aimed fights take it ("setup" list, optional character, ascension, neow,
            chooser_seed, max_actions, script, use_potions); "seed" is given or derived
  seeds     how many fights to play (default 8; each gets its own seed), or a list of seed strings
  compare   {"fields": [".hand", ".draw_pile"], "from_step": 0}: the claim holds when a fight first diverges in one of
            these field groups (reanchor_compare.group names), at or after from_step
  testable  false for a finding no setup fight can show; "why" says what it would need (not run, listed as not testable)

LAB/hypotheses.json maps id -> {status, reason, claim, source, fields, fights, evidence, utc, attempts}. status:
confirmed (a fight diverges in a compared field), refuted (every fight is identical), not testable (the setup cannot
show it: an error, a mirror gap, a divergence elsewhere, or testable false). confirmed and refuted are final; a
not testable one is tried again until MAX_ATTEMPTS, unless testable is false.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
DEFAULT_SEEDS = 8
MAX_ATTEMPTS = 3
TAG = "hyp:"


def _seeds(hid: str, seeds: object) -> list[str]:
    if isinstance(seeds, list):
        return [str(s) for s in seeds]
    n = DEFAULT_SEEDS if seeds is None else int(seeds)
    rng = random.Random("hyp:" + hid)
    return ["".join(rng.choice(ALPHABET) for _ in range(10)) for _ in range(n)]


def parse_setup(doc: object, source: str) -> tuple[dict | None, str | None]:
    """(setup, problem) for one object of a setup file; the setup is normalised."""
    if not isinstance(doc, dict):
        return None, "not an object"
    hid = doc.get("id")
    if not isinstance(hid, str) or not hid.strip() or any(ch in hid for ch in " /\\:"):
        return None, f"bad id {hid!r}"
    if not isinstance(doc.get("claim"), str) or not doc["claim"].strip():
        return None, f"{hid}: no claim"
    s = {"id": hid, "source": str(doc.get("source") or source), "claim": doc["claim"].strip()}
    if doc.get("testable") is False:
        s.update(testable=False, why=str(doc.get("why") or "no reason given"))
        return s, None
    spec, cmp_ = doc.get("spec"), doc.get("compare")
    if not isinstance(spec, dict) or not isinstance(spec.get("setup"), list) or not spec["setup"]:
        return None, f"{hid}: spec needs a non-empty setup list"
    fields = cmp_.get("fields") if isinstance(cmp_, dict) else None
    if not isinstance(fields, list) or not fields or not all(isinstance(f, str) and f.startswith(".") for f in fields):
        return None, f"{hid}: compare.fields must be a list of field groups such as \".hand\""
    seeds = doc.get("seeds")
    ok_list = isinstance(seeds, list) and seeds and all(isinstance(x, str) for x in seeds)
    ok_count = isinstance(seeds, int) and not isinstance(seeds, bool) and 1 <= seeds <= 200
    if seeds is not None and not ok_list and not ok_count:
        return None, f"{hid}: seeds must be a count (1-200) or a list of seed strings"
    frm = cmp_.get("from_step", 0)
    if isinstance(frm, bool) or not isinstance(frm, int) or frm < 0:
        return None, f"{hid}: compare.from_step must be a number of at least 0"
    s.update(testable=True, spec=spec, seeds=seeds, fields=list(fields), from_step=frm)
    return s, None


def load_setups(root: Path) -> tuple[list[dict], list[str]]:
    """Every setup under root/<source>/*.json, and the problems found (a bad file never stops the others)."""
    out: dict[str, dict] = {}
    problems: list[str] = []
    for f in sorted(Path(root).glob("*/*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            problems.append(f"{f.parent.name}/{f.name}: unreadable ({type(e).__name__})")
            continue
        for item in doc if isinstance(doc, list) else [doc]:
            s, why = parse_setup(item, f.parent.name)
            if s is None:
                problems.append(f"{f.parent.name}/{f.name}: {why}")
            elif s["id"] in out:
                problems.append(f"{f.parent.name}/{f.name}: duplicate id {s['id']}")
            else:
                out[s["id"]] = s
    return list(out.values()), problems


def to_specs(setups: list[dict]) -> list[dict]:
    """The fight specs of the testable setups, one per seed, tagged ``hyp:<id>`` (scenarios.py --mode hypotheses)."""
    specs = []
    for s in setups:
        if not s.get("testable"):
            continue
        for i, seed in enumerate(_seeds(s["id"], s["seeds"])):
            base = {"kind": "combat", "character": "IRONCLAD", "ascension": 10, "neow": i % 3, "chooser_seed": 7000 + i,
                    "max_actions": 150, "use_potions": True}
            specs.append({**base, **s["spec"], "seed": seed, "tag": TAG + s["id"]})
    return specs


def pending(setups: list[dict], results: dict) -> list[dict]:
    """The testable setups still to run: not decided and not out of attempts."""
    out = []
    for s in setups:
        r = results.get(s["id"]) or {}
        if s.get("testable") and r.get("status") not in ("confirmed", "refuted") and r.get("attempts", 0) < MAX_ATTEMPTS:
            out.append(s)
    return out


def classify(setup: dict, verdicts: list[dict]) -> tuple[str, str]:
    """(status, reason) of one setup from the verdicts of its fights."""
    if not verdicts:
        return "not testable", "no fight of this setup produced a verdict"
    fields, frm = setup["fields"], setup["from_step"]
    other: dict[str, int] = {}
    same = errors = gaps = 0
    for v in verdicts:
        if v.get("error"):
            errors += 1
            continue
        gaps += 1 if v.get("mirror_gap") else 0
        if v.get("identical"):
            same += 1
        elif v.get("field") in fields and v.get("step", 0) >= frm:
            return "confirmed", f"a fight first diverges in {v['field']} at step {v.get('step')} (seed {v.get('seed')})"
        else:
            other[str(v.get("field"))] = other.get(str(v.get("field")), 0) + 1
    if other:
        return "not testable", f"fights diverge elsewhere ({', '.join(sorted(other))}), not in {', '.join(fields)}"
    if same == len(verdicts) and not gaps:
        return "refuted", f"all {same} fights are identical"
    if same == len(verdicts):
        return "not testable", f"{same} fights identical but the setup mirror had gaps"
    return "not testable", f"{errors} of {len(verdicts)} fights gave no comparable result"


def judge_all(setups: list[dict], verdicts: list[dict], evidence: str, utc: str, prior: dict) -> dict:
    """id -> result entry: testable setups with fights in these verdicts, and the not-testable ones as such."""
    by_id: dict[str, list[dict]] = {}
    for v in verdicts:
        tag = str(v.get("tag") or "")
        if tag.startswith(TAG):
            by_id.setdefault(tag[len(TAG):], []).append(v)
    out = {}
    for s in setups:
        head = {"claim": s["claim"], "source": s["source"], "utc": utc}
        if not s.get("testable"):
            out[s["id"]] = {"status": "not testable", "reason": s["why"], **head, "attempts": 0}
        elif s["id"] in by_id:
            status, reason = classify(s, by_id[s["id"]])
            out[s["id"]] = {"status": status, "reason": reason, **head, "fields": s["fields"], "fights": len(by_id[s["id"]]),
                            "evidence": evidence, "attempts": (prior.get(s["id"]) or {}).get("attempts", 0) + 1}
    return out


def read_results(lab: Path) -> dict:
    try:
        doc = json.loads((Path(lab) / "hypotheses.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def write_results(lab: Path, new: dict) -> dict:
    """Merge `new` into LAB/hypotheses.json (a decided entry is never replaced by another status) and return the whole."""
    cur = read_results(lab)
    for hid, r in new.items():
        if (cur.get(hid) or {}).get("status") in ("confirmed", "refuted") and r["status"] != cur[hid]["status"]:
            continue
        cur[hid] = r
    p = Path(lab) / "hypotheses.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(dict(sorted(cur.items())), indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)
    return cur


def collect(job_dir: Path, setups_dir: Path, lab: Path, utc: str) -> int:
    """After a hypotheses job: judge its verdicts and update LAB/hypotheses.json. Returns the number of entries written."""
    setups, _ = load_setups(setups_dir)
    verdicts: list[dict] = []
    ev = ""
    for vf in sorted(Path(job_dir).rglob("*.verdicts.json")):
        try:
            doc = json.loads(vf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, list):
            verdicts += [v for v in doc if isinstance(v, dict)]
            ev = ev or f"runs/{Path(job_dir).name}/{vf.relative_to(job_dir).as_posix()}"
    new = judge_all(setups, verdicts, ev, utc, read_results(lab))
    write_results(lab, new)
    return len(new)


def doc_of(setup: dict) -> dict:
    """A parsed setup back in the file format (the job's copy of what is pending)."""
    if not setup.get("testable"):
        return {"id": setup["id"], "source": setup["source"], "claim": setup["claim"], "testable": False, "why": setup["why"]}
    return {"id": setup["id"], "source": setup["source"], "claim": setup["claim"], "spec": setup["spec"], "seeds": setup["seeds"],
            "compare": {"fields": setup["fields"], "from_step": setup["from_step"]}}
