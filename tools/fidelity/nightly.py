"""Nightly regression check: main tonight against main's previous run, and a bisect of every fight that got worse.

    python -m tools.fidelity.nightly run [--repo D:/spireline/fork/sts2] [--ref main] [--state D:/spireline/referee/nightly]
        [--procs 16] [--bisect 5] [--corpus D:/spireline/referee/corpus]
    python -m tools.fidelity.nightly bisect --fight <corpus id> --good <commit> --bad <commit> [...]
    python -m tools.fidelity.nightly judge --fight <corpus id> --baseline <verdict.json>      (git bisect run's test)

`run` builds EmuDump at the ref's commit in its own worktree (<state>/wt-night, detached; the main
checkout is never touched), replays the whole corpus with it (corpus.py run: plain replay and
re-anchoring, the same judge as the gate) and checks it with the gate itself (corpus.check) against
the previous main: last night's run (<state>/last.json), or on a first night the corpus baseline of
the nearest ancestor (corpus/baselines/<commit>.json). So a night FAILS as the gate fails: a fight
that got worse (a plain replay that diverges earlier, a new emulator exception, a re-anchored step
that matched and no longer does or now diverges in another field), a fight the previous run had and
tonight's did not compare, nothing compared, a stale build or EmuDump failures. Fights only tonight
has are counted. Each fight that got worse, up to --bisect of them, is bisected over the previous
commit..tonight's with `git bisect run` in a second worktree (<state>/wt-bisect): at each commit
EmuDump is built and that one fight replayed and judged against its verdict at the good commit (a
commit whose build fails is skipped, exit 125); both ends are judged first, and a fight that is not worse
at tonight's commit under that judge, or already worse at the previous one, is reported as not
bisectable instead of blaming tonight's tip. The report goes to <state>/<date>-<commit>.md (five
numbers and the bisects) and .json; <state>/last.json then points at tonight's run (not when it had
EmuDump problems or a stale build). Exit 1 when the gate failed, 2 when the night could not be run or
had nothing to compare with.

Schedule it with the owner's go (it is a standing task): e.g. Windows Task Scheduler running
`D:/spireline/src/sts2/.venv/Scripts/python.exe -m tools.fidelity.nightly run` from a checkout of the
fork at 03:00 (the processes run at BelowNormal priority).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

DEFAULT_REPO = Path("D:/spireline/fork/sts2")
DEFAULT_STATE = Path(os.environ.get("SPIRELINE_NIGHTLY", "D:/spireline/referee/nightly"))
PY = sys.executable
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
SKIP = 125  # git bisect run: this commit cannot be tested
FIRST_BAD = re.compile(r"^([0-9a-f]{40}) is the first '?bad'? commit")


def git(repo: Path, *args, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def worktree_at(repo: Path, path: Path, commit: str) -> Path:
    """A detached worktree of repo at commit (made once, then moved)."""
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        git(repo, "worktree", "add", "--detach", str(path), commit)
    else:
        git(path, "checkout", "--detach", "--force", commit)
    return path


def build_emudump(tree: Path, log: Path) -> Path | None:
    """EmuDump Release in this tree, offline from the NuGet cache; its dll, or None when the build fails."""
    cache = os.environ.get("NUGET_PACKAGES", str(Path.home() / ".nuget" / "packages"))
    env = dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1")
    with log.open("ab") as out:
        rc = subprocess.run(["dotnet", "build", str(tree / "tools" / "emudump" / "EmuDump.csproj"), "-c", "Release",
                             "--source", cache, "-v", "q"], stdout=out, stderr=subprocess.STDOUT, env=env, check=False,
                            creationflags=BELOW_NORMAL).returncode
    dll = tree / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
    return dll if rc == 0 and dll.exists() else None


def worse(base: dict, now: dict) -> list:
    """The gate's own rule (corpus.compare) on one fight's verdicts: what got worse, empty if nothing."""
    from tools.fidelity import corpus as cp  # noqa: PLC0415
    return cp.compare({"f": base}, {"f": now})["regressions"]


def as_baseline(run_doc: dict, commit: str) -> dict:
    """A corpus run (results.json) in the baseline's shape, so corpus.check judges against it."""
    return {"format": run_doc.get("format"), "commit": commit, "run": run_doc.get("label"),
            "judge_sha256": (run_doc.get("emulator") or {}).get("judge_sha256"), "totals": run_doc.get("totals", {}),
            "fights": run_doc.get("fights", {})}


def previous(repo: Path, state: Path, corpus_dir: Path, commit: str) -> tuple[dict, str] | None:
    """The run to compare tonight with, as a baseline doc and where it came from: last night's run, else the
    corpus baseline of tonight's nearest ancestor (the parent first; tonight's own only when no older one)."""
    last_path = state / "last.json"
    if last_path.exists():
        last = json.loads(last_path.read_text(encoding="utf-8"))
        if Path(last["run"]).exists():
            doc = json.loads(Path(last["run"]).read_text(encoding="utf-8"))
            return as_baseline(doc, last["commit"]), f"last night's run {Path(last['run']).parent.name}"
    bdir = corpus_dir / "baselines"
    parents = git(repo, "rev-list", "--max-count=2000", f"{commit}^", check=False).split()
    for c in [*parents, commit]:
        p = bdir / f"{c}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), f"corpus baseline {c[:10]}"
    return None


def gate(corpus_dir: Path, base: dict, run_doc: dict, work: Path) -> tuple[int, str, dict]:
    """corpus.check (the gate) of tonight's run against the previous one: its exit status, what it printed, and
    the comparison (corpus.compare) for the report and the bisects."""
    from tools.fidelity import corpus as cp  # noqa: PLC0415
    work.mkdir(parents=True, exist_ok=True)
    bfile = work / "previous.baseline.json"
    bfile.write_text(json.dumps(base), encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cp.check(cp.Corpus(corpus_dir), run_doc, bfile, show=20)
    return code, out.getvalue(), cp.compare(base["fights"], run_doc["fights"])


# ------------------------------------------------------------------------------------- judge / bisect

def replay_one(fid: str, corpus_dir: Path, dll: Path, work: Path) -> dict:
    """One corpus fight through EmuDump fight and reanchor, judged as the corpus run judges it."""
    from tools.fidelity import corpus as cp  # noqa: PLC0415
    work.mkdir(parents=True, exist_ok=True)
    verdict = {"n": len(cp.Corpus(corpus_dir).read(fid)["steps"])}
    for mode in ("fight", "reanchor"):
        r = cp.replay_part(mode, [fid], str(corpus_dir), str(dll), str(work))
        if r["problem"]:
            raise RuntimeError(r["problem"])
        verdict.update(r["fights"][fid])
    return verdict


def cmd_judge(a) -> int:
    """git bisect run's test at the commit the bisect worktree stands on: 0 good, 1 worse, 125 untestable."""
    tree = Path(a.tree or os.getcwd())
    base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
    dll = build_emudump(tree, Path(a.baseline).with_suffix(".build.log"))
    if dll is None:
        return SKIP
    try:
        now = replay_one(a.fight, Path(a.corpus), dll, Path(tempfile.mkdtemp(prefix="nightly_judge_")))
    except Exception as ex:  # noqa: BLE001  any failure to replay is a commit that cannot be judged
        print(f"judge: {type(ex).__name__}: {ex}", file=sys.stderr)
        return SKIP
    bad = worse(base, now)
    print(f"judge {git(tree, 'rev-parse', '--short', 'HEAD')}: {'worse' if bad else 'not worse'}")
    return 1 if bad else 0


def bisect(repo: Path, state: Path, fid: str, good: str, bad: str, corpus_dir: Path) -> dict:
    """The first commit between good and bad at which this fight gets worse (git bisect run)."""
    tree = worktree_at(repo, state / "wt-bisect", good)
    log = state / "bisect.log"
    dll = build_emudump(tree, log)
    if dll is None:
        return {"fight": fid, "error": "EmuDump does not build at the good commit"}
    try:
        base = replay_one(fid, corpus_dir, dll, Path(tempfile.mkdtemp(prefix="nightly_base_")))
    except Exception as ex:  # noqa: BLE001
        return {"fight": fid, "error": f"the fight does not replay at the good commit: {ex}"[:300]}
    bfile = state / f"bisect-{fid}.baseline.json"
    bfile.write_text(json.dumps(base), encoding="utf-8")
    test = [PY, str(ROOT / "tools" / "fidelity" / "nightly.py"), "judge", "--fight", fid, "--baseline", str(bfile),
            "--corpus", str(corpus_dir), "--tree", str(tree)]
    return {"fight": fid, **bisect_run(tree, good, bad, test)}


def end_check(tree: Path, good: str, bad: str, test: list) -> str | None:
    """Why good..bad cannot be bisected with this test, or None: the test must fail (exit 1..127, not 125) at
    the bad end and pass at the good one. git bisect takes both on trust, so a regression that does not
    reproduce under the bisect's judge (the judge changed since the night, a flaky timeout) would make every
    tested commit good and blame the bad end, which was never tested."""
    for commit, want in ((bad, "bad"), (good, "good")):
        git(tree, "checkout", "--detach", "--force", commit)
        rc = subprocess.run(test, cwd=str(tree), capture_output=True, text=True, check=False,
                            creationflags=BELOW_NORMAL).returncode
        if rc == SKIP:
            return f"the {want} end {commit[:10]} cannot be tested (exit {SKIP}: EmuDump does not build or replay)"
        if want == "bad" and rc == 0:
            return (f"not reproduced: the bad end {commit[:10]} is not worse under the bisect's judge (the judge changed, "
                    "or the night's result was a flake); nothing to bisect")
        if want == "good" and rc != 0:
            return f"the good end {commit[:10]} is already worse under the bisect's judge (exit {rc}); nothing to bisect"
    return None


def bisect_run(tree: Path, good: str, bad: str, test: list) -> dict:
    """`git bisect run test` over good..bad in tree: the first bad commit (and its subject), or None.

    Both ends are tested first (end_check); when they are not bad and good, nothing is bisected and the
    result says why ("error"), with first_bad None.
    """
    git(tree, "bisect", "reset", check=False)
    here = git(tree, "symbolic-ref", "-q", "--short", "HEAD", check=False) or git(tree, "rev-parse", "HEAD")
    try:
        why = end_check(tree, good, bad, test)
    finally:
        git(tree, "checkout", "--force", here)
    if why:
        return {"first_bad": None, "error": why, "log": []}
    git(tree, "bisect", "start", bad, good)
    try:
        r = subprocess.run(["git", "-C", str(tree), "bisect", "run", *test], capture_output=True, text=True, check=False,
                           creationflags=BELOW_NORMAL)
        out = r.stdout + r.stderr
        # "<sha> is the first bad commit" (newer git: "first 'bad' commit")
        first = next((m.group(1) for ln in out.splitlines() for m in [FIRST_BAD.match(ln.strip())] if m), None)
        result = {"first_bad": first, "log": out.strip().splitlines()[-12:]}
        if first:
            result["subject"] = git(tree, "log", "-1", "--format=%s", first)
    finally:
        git(tree, "bisect", "reset", check=False)
    return result


def cmd_bisect(a) -> int:
    res = bisect(Path(a.repo), Path(a.state), a.fight, a.good, a.bad, Path(a.corpus))
    print(json.dumps(res, indent=1))
    return 0 if res.get("first_bad") else 1


# ------------------------------------------------------------------------------------------- run

def corpus_run(tree: Path, dll: Path, corpus_dir: Path, procs: int, label: str, log: Path) -> Path | None:
    """corpus.py run from that tree (its own judge and commit id) with its own EmuDump."""
    with log.open("ab") as out:
        rc = subprocess.run([PY, str(tree / "tools" / "fidelity" / "corpus.py"), "--corpus", str(corpus_dir), "run",
                             "--emudump", str(dll), "--procs", str(procs), "--label", label, "--no-scan"],
                            stdout=out, stderr=subprocess.STDOUT, check=False, cwd=str(tree)).returncode
    res = corpus_dir / "runs" / label / "results.json"
    return res if rc == 0 and res.exists() else None


def report_md(rep: dict) -> str:
    """The night in five numbers, then the bisects and the gate's own lines."""
    c = rep.get("compare") or {}
    t, pt = rep.get("totals") or {}, (rep.get("previous_totals") or {})
    verdict = {0: "ok", None: "no comparison"}.get(rep.get("gate_exit"), "FAIL")
    lines = [f"# Nightly {rep['date']}: main {rep['commit'][:10]} against {rep.get('previous', 'nothing')}: {verdict}", "",
             f"1. identical fights: {pt.get('identical', '-')} -> {t.get('identical', '-')} of {t.get('fights', '-')}",
             f"2. re-anchored steps matched: {pt.get('reanchor_matched', '-')} -> {t.get('reanchor_matched', '-')} "
             f"of {t.get('reanchored', '-')}",
             f"3. fights worse: {len(c.get('regressions', []))}; better: {len(c.get('improvements', []))}",
             f"4. not compared: {len(c.get('missing', []))}; new tonight: {len(c.get('new', []))}; emulator errors: "
             f"{t.get('emu_errors', 0)}",
             f"5. commits since: {rep.get('commits', '-')}; bisected: {len(rep.get('bisects', []))}"]
    for b in rep.get("bisects", []):
        lines.append(f"- {b['fight']}: first bad commit {(b.get('first_bad') or '?')[:10]} {b.get('subject', '')} "
                     f"{b.get('error', '')}".rstrip())
    if rep.get("gate_text"):
        lines += ["", "```", rep["gate_text"].rstrip()[-4000:], "```"]
    return "\n".join(lines) + "\n"


def night(repo: Path, state: Path, corpus_dir: Path, commit: str, res_path: Path, bisect_max: int) -> tuple[int, dict]:
    """Tonight's run (res_path) against the previous main, bisected; the exit status and the report."""
    cur = json.loads(res_path.read_text(encoding="utf-8"))
    rep = {"date": datetime.date.today().isoformat(), "commit": commit, "run": str(res_path), "totals": cur.get("totals"),
           "gate_exit": None}
    prev = previous(repo, state, corpus_dir, commit)
    code = 2
    if prev:
        base, where = prev
        rep.update(previous=where, previous_commit=base.get("commit"), previous_totals=base.get("totals"))
        code, text, cmp = gate(corpus_dir, base, cur, state / "work")
        rep.update(gate_exit=code, gate_text=text)
        rep["compare"] = {k: [x if isinstance(x, str) else {"id": x["id"], "plain": x.get("plain"), "steps": x.get("steps")}
                              for x in v] for k, v in cmp.items()}
        if base.get("commit"):
            rep["commits"] = int(git(repo, "rev-list", "--count", f"{base['commit']}..{commit}", check=False) or 0)
        worse_ids = [r["id"] for r in cmp["regressions"]]
        if worse_ids and bisect_max and base.get("commit") and base["commit"] != commit:
            rep["bisects"] = [bisect(repo, state, fid, base["commit"], commit, corpus_dir) for fid in worse_ids[:bisect_max]]
        code = 0 if code == 0 else 1
    else:
        rep["previous"] = "nothing (no last night, no corpus baseline among the ancestors): this run is the first"
    out = state / f"{rep['date']}-{commit[:10]}.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text(report_md(rep), encoding="utf-8")
    if cur.get("problems") or (cur.get("emulator") or {}).get("stale_build") or cur.get("subset"):
        rep["not_kept"] = "tonight's run is not the next night's comparison (EmuDump problems, a stale build or a partial run)"
    else:
        (state / "last.json").write_text(json.dumps({"commit": commit, "run": str(res_path), "date": rep["date"]}),
                                         encoding="utf-8")
    return code, rep


def cmd_run(a) -> int:
    repo, state, corpus_dir = Path(a.repo), Path(a.state), Path(a.corpus)
    state.mkdir(parents=True, exist_ok=True)
    commit = git(repo, "rev-parse", a.ref)
    log = state / f"{datetime.date.today().isoformat()}-{commit[:10]}.log"
    tree = worktree_at(repo, state / "wt-night", commit)
    dll = build_emudump(tree, log)
    if dll is None:
        print(f"nightly: EmuDump does not build at {commit[:10]} (log {log})")
        return 2
    res_path = corpus_run(tree, dll, corpus_dir, a.procs, f"nightly-{datetime.date.today().isoformat()}-{commit[:10]}", log)
    if res_path is None:
        print(f"nightly: the corpus run failed (log {log})")
        return 2
    code, rep = night(repo, state, corpus_dir, commit, res_path, a.bisect)
    print(report_md(rep) + (f"\n{rep['not_kept']}\n" if rep.get("not_kept") else ""))
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "bisect", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", default=os.environ.get("SPIRELINE_CORPUS", "D:/spireline/referee/corpus"))
        if name != "judge":
            p.add_argument("--repo", default=str(DEFAULT_REPO))
            p.add_argument("--state", default=str(DEFAULT_STATE))
    r = sub.choices["run"]
    r.add_argument("--ref", default="main")
    r.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    r.add_argument("--bisect", type=int, default=5, help="bisect at most this many fights that got worse (0: none)")
    b = sub.choices["bisect"]
    b.add_argument("--fight", required=True)
    b.add_argument("--good", required=True)
    b.add_argument("--bad", required=True)
    j = sub.choices["judge"]
    j.add_argument("--fight", required=True)
    j.add_argument("--baseline", required=True)
    j.add_argument("--tree", help="the bisect worktree (default: the current directory)")
    a = ap.parse_args(argv)
    if shutil.which("git") is None:
        print("nightly: git not found")
        return 2
    return {"run": cmd_run, "bisect": cmd_bisect, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
