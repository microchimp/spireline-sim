"""Engine-neutral scenario tests (docs/rebuild/SCENARIOS.md): runs a directory of scenario files (tests/scenarios) through
`EmuDump scenario` on each engine and judges the outcomes.

    python tools/fidelity/scenarios.py [--dir tests/scenarios] [--engines old core2] [--dll EmuDump.dll] [--json]

A scenario is the setup in the game's terms, the actions and the expected fields at named points (the format is in
SCENARIOS.md); EmuDump runs it through IEngine only. Per scenario and engine the outcome is pass, fail, xfail (a fail the file
lists for that engine under `xfail_engines`, with its reason), xpass (listed as an xfail and passed), skip (`engines` leaves the
engine out) or invalid (the file breaks the format). Exit 1 when any engine has a fail, an xpass or an invalid one; 2 when
EmuDump cannot run. One EmuDump process runs one engine over the whole directory (one emulator per process).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DLL = ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
DEFAULT_DIR = ROOT / "tests" / "scenarios"
ENGINES = ("old", "core2")
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


class EmuDumpError(RuntimeError):
    """EmuDump did not produce a report (it exited with a code other than 0 or 1, or wrote no JSON)."""


def run_dir(directory: Path, engine: str, dll: Path = DEFAULT_DLL, timeout: float = 900) -> dict:
    """The report of `EmuDump scenario <directory> --engine <engine> --json`: {engine, summary, results[], exit}."""
    p = subprocess.run(["dotnet", str(dll), "scenario", str(directory), "--engine", engine, "--json"], capture_output=True,
                       timeout=timeout, creationflags=BELOW_NORMAL)
    if p.returncode not in (0, 1):
        raise EmuDumpError(f"EmuDump scenario exited {p.returncode}: " + p.stderr.decode("utf-8", "replace")[:400])
    try:
        report = json.loads(p.stdout.decode("utf-8"))
    except ValueError as e:
        raise EmuDumpError("EmuDump scenario wrote no JSON report: " + str(e)) from e
    report["exit"] = p.returncode
    return report


def unexpected(report: dict) -> list[dict]:
    """The results that fail the run: a fail, an xpass (an xfail row that now passes: take it out) or an invalid file."""
    return [r for r in report["results"] if r["unexpected"]]


def by_status(report: dict, status: str) -> list[str]:
    return sorted(r["name"] for r in report["results"] if r["status"] == status)


def render(report: dict) -> str:
    lines = []
    for r in report["results"]:
        if r["status"] == "pass":
            continue
        note = f" ({r['reason']})" if r["status"] in ("xfail", "xpass") and r.get("reason") else ""
        lines.append(f"{r['status'].upper():7}{r['file']}: {r['name']}{note}")
        if r["status"] in ("fail", "invalid", "xpass"):
            lines += ["    " + d for d in r["diffs"]]
    s = report["summary"]
    lines.append(f"{report['engine']}: {s['pass']} pass, {s['xfail']} xfail, {s['fail']} fail, {s['xpass']} xpass, "
                 f"{s['invalid']} invalid, {s['skip']} skip")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--engines", nargs="+", default=list(ENGINES), choices=ENGINES)
    ap.add_argument("--dll", type=Path, default=DEFAULT_DLL)
    ap.add_argument("--json", action="store_true", help="write the reports as JSON")
    args = ap.parse_args(argv)
    if not args.dll.is_file():
        print("EmuDump is not built (dotnet build tools/emudump/EmuDump.csproj -c Release)", file=sys.stderr)
        return 2
    reports, bad = [], 0
    for engine in args.engines:
        try:
            report = run_dir(args.dir, engine, args.dll)
        except EmuDumpError as e:
            print(f"{engine}: {e}", file=sys.stderr)
            return 2
        reports.append(report)
        bad += len(unexpected(report))
        if not args.json:
            print(render(report))
    if args.json:
        print(json.dumps(reports, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
