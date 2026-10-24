"""The old behaviour suites run twice, once with the test switches off and once on (docs/rebuild/MIGRATION_F08.md,
"The old suites under the switch"; the F08 review's recommendation 2).

    python -m tools.fidelity.suites_switch --out DIR [--switches F08,F10] [--no-build] [--suites A,B,...]

The suites are xunit classes of src/Sts2Emulator.Tests. The first run has STS2_TEST_SWITCHES unset (today's engine); the
second sets it, so src/Sts2Emulator.Tests/Support/TestSwitches.cs binds every combat the suites build to a Core2 kernel
with those switches (at its first step) and steps it through RunEngine.StepCombat. The second run also sets
STS2_TEST_SWITCHES_COVERAGE: the fixture writes one line per kernel-bound step (the test method that made it). Each test
gets a verdict: "pass both", "fail on only", "fail off only", "fail both", "not covered" (passed both ways but made no
kernel-bound step with the switch on, so it proves nothing about the switch; a theory's cases share their method's
count), "skipped", or "missing off" / "missing on" (absent from one run: that run crashed or was cut short). Writes DIR/suites_switch.json (every test) and
DIR/suites_switch.md (the counts and the tests that are not "pass both"). The trx files dotnet writes carry the machine's
name, so they are read from a temporary directory and deleted; the outputs hold test names, verdicts and the first line
of each failure message (paths removed). Exit 1 when a run wrote no trx, was aborted (a crashed test host) or lost
tests the other has; the table is still written, with "crashed" naming the run.

Builds with --no-restore (the tree must be restored, as tools/build_and_test.sh does); --no-build reuses the last build.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = "src/Sts2Emulator.Tests"
VARIABLE = "STS2_TEST_SWITCHES"
COVERAGE = "STS2_TEST_SWITCHES_COVERAGE"

# The ten the review's section 2.3 item 3 names (DeathCheckTests is a file of six classes).
NAMED = (
    "PotionKillTests", "TurnStartDeathTests", "SelectionAnswerDeathTests", "AfterPlayDeathTests", "FightHeldOpenTests",
    "AutoPlayDeathTests",  # DeathCheckTests.cs
    "CombatEndingGateTests", "CorpseRemovalTests", "WaterfallGiantTests", "IllusionAndMinionTests", "PhrogParasiteTests",
    "MercPowerListTests", "HeistGoldRewardTests", "LethalHitHookTests", "HeadbuttTests",
)
# The other old-engine suites about damage, death, HP or the combat's end.
ADDED = (
    "CalculatedDamageTests", "CombatEndRewardPowerTests", "CombatEndsWithTheEnemyTurnTests", "CorpseSlugsNormalTests",
    "DamageAndPowerModifierRelicTests", "DamageReceivedRelicTests", "DeadlyPoisonTests", "DeathMarchTests",
    "DeathbringerTests", "DeathsDoorTests", "HunterKillerTests", "LethalityTests", "MinionDiveBombTests",
    "MinionSacrificeTests", "MinionStrikeTests", "OstyDamageSinkTests", "PlayerHpLossTests", "RunHpLossTests",
    "PoisonedStabTests", "TestSubjectTests", "ThornsOrderTests", "TurnEndSelfDamageTests", "UnpoweredDamageHookTests",
    "TwinStrikeTests", "WhirlwindTests", "GremlinHornTests", "FeedTests",
)
SUITES = NAMED + ADDED

_NS = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
_PATH = re.compile(r"([A-Za-z]:[\\/]|/)[^\s:'\"]*[\\/][^\s:'\"]*")


def test_filter(suites) -> str:
    """The dotnet test filter of the suites: each class by its dotted name."""
    return "|".join(f"FullyQualifiedName~.{s}." for s in suites)


def _scrub(text: str) -> str:
    return _PATH.sub("<path>", text)


def parse_trx(path) -> dict:
    """{test name: {"suite", "outcome", "message"}} from a trx file (outcome Passed, Failed, NotExecuted, ...)."""
    root = ET.parse(path).getroot()
    suite_of = {}
    for ut in root.iterfind("t:TestDefinitions/t:UnitTest", _NS):
        m = ut.find("t:TestMethod", _NS)
        if m is not None:
            suite_of[ut.get("id")] = m.get("className", "").rsplit(".", 1)[-1]
    out = {}
    for r in root.iterfind("t:Results/t:UnitTestResult", _NS):
        msg = r.find("t:Output/t:ErrorInfo/t:Message", _NS)
        first = (msg.text or "").strip().splitlines()[0] if msg is not None and (msg.text or "").strip() else ""
        out[r.get("testName")] = {
            "suite": suite_of.get(r.get("testId"), ""),
            "outcome": r.get("outcome", ""),
            "message": _scrub(first)[:240],
        }
    return out


def aborted(path) -> bool:
    """Whether the run behind a trx was aborted (dotnet's run info names a crashed or aborted test host)."""
    root = ET.parse(path).getroot()
    texts = " ".join((t.text or "") for t in root.iterfind("t:ResultSummary/t:RunInfos/t:RunInfo/t:Text", _NS))
    return bool(re.search(r"aborted|crashed", texts, re.I))


def method_key(test_name: str) -> str:
    """A test's method (Namespace.Class.Method), its theory arguments dropped: the fixture's coverage key."""
    return test_name.split("(", 1)[0]


def read_coverage(path) -> dict:
    """{method: kernel-bound steps} from the fixture's coverage file (absent: no test stepped the kernel)."""
    out: dict[str, int] = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out[line.strip()] = out.get(line.strip(), 0) + 1
    return out


def verdict(off: str | None, on: str | None) -> str:
    """One test's verdict from its two outcomes."""
    if off is None or on is None:
        return "missing " + ("off" if off is None else "on")
    if off == "NotExecuted" or on == "NotExecuted":
        return "skipped"
    a, b = off == "Passed", on == "Passed"
    return {(True, True): "pass both", (True, False): "fail on only", (False, True): "fail off only"}.get(
        (a, b), "fail both")


def compare(off: dict, on: dict, coverage: dict | None = None) -> list[dict]:
    rows = []
    for name in sorted(set(off) | set(on)):
        o, n = off.get(name), on.get(name)
        v = verdict(o["outcome"] if o else None, n["outcome"] if n else None)
        steps = None if coverage is None else coverage.get(method_key(name), 0)
        if v == "pass both" and steps == 0:
            v = "not covered"
        rows.append({
            "test": name,
            "suite": (o or n)["suite"],
            "off": o["outcome"] if o else None,
            "on": n["outcome"] if n else None,
            "verdict": v,
            "bound_steps": steps,
            "message": (n or {}).get("message", "") if not (n and n["outcome"] == "Passed") else "",
        })
    return rows


def counts(rows) -> dict:
    c: dict[str, int] = {}
    for r in rows:
        c[r["verdict"]] = c.get(r["verdict"], 0) + 1
    return dict(sorted(c.items()))


COLUMNS = ("pass both", "not covered", "fail on only", "fail off only", "fail both")


def uncovered(rows) -> dict:
    """{suite: (not covered, tests)} for the suites with a test that never stepped the kernel."""
    per: dict[str, list[int]] = {}
    for r in rows:
        c = per.setdefault(r["suite"], [0, 0])
        c[1] += 1
        c[0] += r["verdict"] == "not covered"
    return {s: (c[0], c[1]) for s, c in sorted(per.items()) if c[0]}


def markdown(switches: str, suites, rows, crashed=()) -> str:
    per: dict[str, dict[str, int]] = {}
    for r in rows:
        per.setdefault(r["suite"], {}).setdefault(r["verdict"], 0)
        per[r["suite"]][r["verdict"]] += 1
    lines = [f"# Old suites, switches off vs `{switches}`", "",
             f"{len(suites)} suites, {len(rows)} tests: " + ", ".join(f"{k} {v}" for k, v in counts(rows).items()), ""]
    if crashed:
        lines += [f"**Crashed or incomplete run: {', '.join(crashed)}**; the table below is partial.", ""]
    lines += ["| suite | tests | " + " | ".join(COLUMNS) + " |", "|---|---|" + "---|" * len(COLUMNS)]
    for s in sorted(per):
        v = per[s]
        lines.append(f"| {s} | {sum(v.values())} | " + " | ".join(
            str(v.get(k, 0)) for k in COLUMNS) + " |")
    unc = uncovered(rows)
    if unc:
        lines += ["", "Not covered (no kernel-bound step with the switch on): " + ", ".join(
            f"{s} ({n} of {m})" for s, (n, m) in unc.items())]
    odd = [r for r in rows if r["verdict"] not in ("pass both", "not covered")]
    if odd:
        lines += ["", "| test | verdict | on: first line |", "|---|---|---|"]
        lines += [f"| {r['test']} | {r['verdict']} | {r['message'].replace('|', '/')} |" for r in odd]
    return "\n".join(lines) + "\n"


def _env(switches: str | None, coverage: Path | None = None) -> dict:
    env = dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1")
    env.pop(VARIABLE, None)
    env.pop(COVERAGE, None)
    if switches is not None:
        env[VARIABLE] = switches
    if coverage is not None:
        env[COVERAGE] = str(coverage)
    return env


def run_suites(tree: Path, suites, switches: str | None, trx_dir: Path, name: str, coverage: Path | None = None):
    """(results, aborted) of one dotnet test run; results None when it wrote no trx (a crash before the logger)."""
    cmd = ["dotnet", "test", PROJECT, "--no-build", "--no-restore", "--filter", test_filter(suites),
           "--logger", f"trx;LogFileName={name}.trx", "--results-directory", str(trx_dir)]
    subprocess.run(cmd, cwd=tree, env=_env(switches, coverage), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    trx = trx_dir / f"{name}.trx"
    if not trx.exists():
        return None, True
    return parse_trx(trx), aborted(trx)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--switches", default="F08,F10")
    ap.add_argument("--tree", type=Path, default=ROOT)
    ap.add_argument("--suites", default=None, help="comma-separated xunit class names (default: NAMED + ADDED)")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args(argv)
    suites = tuple(s for s in a.suites.split(",") if s) if a.suites else SUITES
    if not a.no_build:
        b = subprocess.run(["dotnet", "build", PROJECT, "--no-restore"], cwd=a.tree, env=_env(None),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if b.returncode != 0:
            raise SystemExit("suites_switch: the test project did not build")
    with tempfile.TemporaryDirectory(prefix="suites-switch-") as tmp:
        cov = Path(tmp) / "coverage.txt"
        off, off_bad = run_suites(a.tree, suites, None, Path(tmp), "off")
        on, on_bad = run_suites(a.tree, suites, a.switches, Path(tmp), "on", cov)
        coverage = read_coverage(cov)
    if off is None and on is None:
        raise SystemExit("suites_switch: neither run wrote a trx (did the test project build?)")
    rows = compare(off or {}, on or {}, coverage)
    crashed = [n for n, bad in (("off", off_bad), ("on", on_bad)) if bad]
    crashed += [f"{n} lost tests" for n in ("off", "on")
                if n not in crashed and any(r["verdict"] == f"missing {n}" for r in rows)]
    a.out.mkdir(parents=True, exist_ok=True)
    doc = {"switches": a.switches, "suites": list(suites), "tests": len(rows), "counts": counts(rows),
           "uncovered": {s: list(v) for s, v in uncovered(rows).items()}, "crashed": crashed, "rows": rows}
    (a.out / "suites_switch.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (a.out / "suites_switch.md").write_text(markdown(a.switches, suites, rows, crashed), encoding="utf-8")
    print(json.dumps({"tests": len(rows), "counts": doc["counts"], "crashed": crashed}))
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
