"""The fix lab's jobs as data: each job is a list of steps, each step a few commands (one process each).

A plan is built from the job's recorded parameters, so an interrupted job resumes with the same plan. A command
is written symbolically (``{tree}``, ``{lab}``, ``{job}`` ...); the symbolic form is what job.json and the logs
keep, so they hold no absolute path of the user's. A command whose tool is missing on the tree, whose input is
absent, or that ``--skip`` names is skipped with a one-line reason, never an error.

Jobs, in the runner's order of value (runner.ROTATION): presort, fights, probes, mutation, invariants.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tools.lab.winsys import NO_WINDOW

TOKEN = re.compile(r"\{(\w+)\}")
EMUDUMP = "tools/emudump/bin/Release/net9.0/EmuDump.dll"
DRIVER_DLL = "referee/driver/bin/Release/net9.0/SpirelineReferee.dll"
# scenarios.py before it had --jobs: its output folder is a module constant, so set it first (an existing mode's
# behaviour is unchanged; only where its files go). No braces: the string goes through the token expander.
SCENARIOS_SHIM = ("import sys; sys.path.insert(0, 'referee'); import scenarios; from pathlib import Path; "
                  "scenarios.JOBS = Path(sys.argv[1]); sys.argv = ['scenarios.py'] + sys.argv[2:]; sys.exit(scenarios.main())")
# Mixed in at run time for the native build: Visual Studio's installer folder (vswhere) must be on PATH.
VS_INSTALLER = "C:/Program Files (x86)/Microsoft Visual Studio/Installer"


class Unavailable(Exception):
    """A job type that cannot run now (its tool or its input is missing); the runner skips it with this reason."""


@dataclasses.dataclass
class Ctx:
    lab: Path
    tree: Path
    game: Path
    fork: Path
    py: str
    sha: str
    job_dir: Path
    job_name: str
    params: dict[str, Any]
    backlog_dir: Path
    backlog: Path | None
    codex: Path
    decompiled: Path
    records: Path
    game_src: Path
    nuget: Path
    skip: tuple[str, ...] = ()

    def tokens(self) -> dict[str, str]:
        f = lambda p: str(p).replace("\\", "/")  # noqa: E731
        return {"lab": f(self.lab), "tree": f(self.tree), "game": f(self.game), "game_src": f(self.game_src),
                "game_parent": f(self.game.parent), "fork": f(self.fork), "py": self.py, "sha": self.sha,
                "sha7": self.sha[:7], "job": f(self.job_dir), "jobname": self.job_name,
                "backlog": f(self.backlog) if self.backlog else "", "backlog_dir": f(self.backlog_dir),
                "codex": f(self.codex), "decompiled": f(self.decompiled), "records": f(self.records),
                "nuget": f(self.nuget), "vs_installer": VS_INSTALLER}

    def expand(self, s: str) -> str:
        t = self.tokens()
        return TOKEN.sub(lambda m: t.get(m.group(1), m.group(0)), s)

    def path(self, s: str) -> Path:
        return Path(self.expand(s))


@dataclasses.dataclass
class Cmd:
    name: str
    argv: list[str]
    cwd: str = "{tree}"
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    msvc: bool = False  # add Visual Studio's installer folder to PATH and drop Git's usr/bin (its link.exe shadows MSVC's)
    stdout: str | None = None  # default: logs/<name>.log (stderr merged)
    stderr: str | None = None
    needs: list[str] = dataclasses.field(default_factory=list)  # paths that must exist, else the command is skipped
    when: Callable[[Ctx], str | None] | None = None  # a reason to skip, or None to run (evaluated just before it runs)
    ok_rcs: tuple[int, ...] = (0,)
    timeout: float | None = None
    mkdirs: list[str] = dataclasses.field(default_factory=list)
    cleanup: list[str] = dataclasses.field(default_factory=list)  # globs removed before a (re)run
    after_cleanup: list[str] = dataclasses.field(default_factory=list)  # globs removed once it ended, whatever its rc
    outputs: dict[str, str] = dataclasses.field(default_factory=dict)  # role -> path relative to the job folder
    pre: Callable[[Ctx], None] | None = None
    post: Callable[[Ctx], str | None] | None = None  # a problem; "HALT: ..." stops the lab (a safety check failed)
    core: bool = False  # real work of its job: a job with no runnable core command is not run at all
    counts: dict[str, int] = dataclasses.field(default_factory=dict)  # added to the lab's counters when it succeeds (fights played)

    def skip_reason(self, c: Ctx, step: str = "") -> str | None:
        for pre in c.skip:
            if self.name.startswith(pre) or (step and step.startswith(pre)):
                return f"skipped by --skip {pre}"
        for n in self.needs:
            if not c.path(n).exists():
                return "missing on this tree: " + _rel(c.expand(n), c)
        return self.when(c) if self.when else None

    def symbolic(self) -> str:
        return " ".join(self.argv)


@dataclasses.dataclass
class Step:
    name: str
    cmds: list[Cmd]
    mode: str = "parallel"  # parallel: one process each at once; serial: in order, stopping at the first failure
    required: bool = False  # a failure stops the job (and the sha is not presorted)


def _rel(s: str, c: Ctx) -> str:
    for k in ("tree", "lab", "game", "fork"):
        base = c.tokens()[k]
        if s.startswith(base):
            return "{" + k + "}" + s[len(base):]
    return s


def git_head(tree: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
                              creationflags=NO_WINDOW).stdout.strip()
    except OSError:
        return ""


def _assets_missing(rel: str) -> Callable[[Ctx], str | None]:
    return lambda c: None if not (c.tree / rel / "obj" / "project.assets.json").exists() else "restore not needed (assets present)"


def source_has(path: Path, *needles: str) -> list[bool]:
    """Which of `needles` appear in a source file (how the lab learns what a tool on the tree can do, without running it)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [False] * len(needles)
    return [n in text for n in needles]


# ---------------------------------------------------------------- presort

def plan_presort(c: Ctx) -> list[Step]:
    p = c.params
    restore = [Cmd(f"restore-{d.replace('/', '-')}", ["dotnet", "restore", d, "--source", "{nuget}", "-v", "q"], when=_assets_missing(d),
                   needs=[f"{{tree}}/{d}"]) for d in ("tools/emudump", "referee/driver")]
    walk = [Cmd(f"codex-s{i}", ["dotnet", "{tree}/" + EMUDUMP, "codex", "{codex}", f"{{job}}/codex/s{i}.jsonl", "--decompiled", "{decompiled}",
                                "--character", "IRONCLAD", "--ascension", "10", "--shard", f"{i}/4"],
                 stderr=f"{{job}}/codex/s{i}.err", needs=["{tree}/" + EMUDUMP, "{codex}", "{decompiled}"], mkdirs=["{job}/codex"],
                 cleanup=[f"{{job}}/codex/s{i}.*"], outputs={"codex_walk": f"codex/s{i}.jsonl", "codex_walk_err": f"codex/s{i}.err"},
                 core=True) for i in range(4)]
    # Both keep their results outside the job (the refset's scores folder, the corpus's runs folder); a copy goes
    # into the job folder under the roles lab_groups reads, so their symptoms reach the groups.
    refset = Cmd("refset", ["{py}", "-m", "tools.fidelity.refset_score", "--emu", "{tree}", "--procs", str(p["procs_refset"]),
                            "--label", "lab-{sha7}"], needs=["{tree}/tools/fidelity/refset_score.py", "{tree}/" + EMUDUMP],
                 ok_rcs=(0, 1), outputs={"refset_score": "refset/score.json"}, core=True, post=_keep_refset)
    corpus = Cmd("corpus", ["{py}", "tools/fidelity/corpus.py", "check", "--no-scan", "--procs", str(p["procs_corpus"]), "--label", "lab-{sha7}"],
                 needs=["{tree}/tools/fidelity/corpus.py", "{tree}/" + EMUDUMP], ok_rcs=(0, 1),
                 outputs={"corpus": "corpus/results.json"}, core=True, post=_keep_corpus)
    return [
        Step("tree", [Cmd("worktree-add", ["git", "-C", "{fork}", "worktree", "add", "--detach", "{tree}", "{sha}"], cwd="{fork}",
                          when=lambda c: "tree exists" if (c.tree / ".git").exists() else None),
                      Cmd("checkout", ["git", "-C", "{tree}", "checkout", "--detach", "--force", "{sha}"], cwd="{fork}",
                          when=lambda c: "tree already at main" if git_head(c.tree) == c.sha else None)],
             mode="serial", required=True),
        Step("restore", restore, mode="serial"),
        Step("build", [Cmd("build-emudump", ["dotnet", "build", "-c", "Release", "--no-restore", "-v", "q"], cwd="{tree}/tools/emudump",
                           needs=["{tree}/tools/emudump"])], mode="serial", required=True),
        Step("driver", [Cmd("build-driver", ["dotnet", "build", "referee/driver/SpirelineReferee.csproj", "-c", "Release", "--no-restore", "-v", "q",
                                             "-p:GameData={game_src}/data_sts2_windows_x86_64"],
                            needs=["{tree}/referee/driver/SpirelineReferee.csproj", "{game_src}/data_sts2_windows_x86_64"])], mode="serial"),
        Step("native", [Cmd("publish-native", ["dotnet", "publish", "src/Sts2Emulator/Sts2Emulator.csproj", "-c", "Release", "-r", "win-x64",
                                               "--self-contained", "-o", "out", "--source", "{nuget}", "-v", "q"], msvc=True,
                            needs=["{tree}/src/Sts2Emulator/Sts2Emulator.csproj"])], mode="serial"),
        Step("game", [game_cmd()], mode="serial"),
        Step("checks", walk + [refset]),
        Step("corpus", [corpus]),
        *report_steps(c),
    ]


def _copy_in(src: Path | None, dst: Path, what: str) -> str | None:
    if src is None or not src.is_file():
        return f"no {what} to copy into the job folder"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    except OSError as e:
        return f"{what} not copied into the job folder: {type(e).__name__}"
    return None


def _keep_refset(c: Ctx) -> str | None:
    """refset_score ends its log with '  stored <score file>'."""
    src = None
    try:
        for line in (c.job_dir / "logs" / "refset.log").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("  stored ") and line.rstrip().endswith(".json"):
                src = Path(line[len("  stored "):].strip())
    except OSError:
        pass
    return _copy_in(src, c.job_dir / "refset" / "score.json", "refset score")


def corpus_root(tree: Path) -> Path | None:
    """The corpus the tree's corpus.py uses: $SPIRELINE_CORPUS, else the default its source names."""
    if os.environ.get("SPIRELINE_CORPUS"):
        return Path(os.environ["SPIRELINE_CORPUS"])
    try:
        m = re.search(r'"SPIRELINE_CORPUS",\s*r?"([^"]+)"', (tree / "tools/fidelity/corpus.py").read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return Path(m.group(1)) if m else None


def _keep_corpus(c: Ctx) -> str | None:
    root = corpus_root(c.tree)
    src = root / "runs" / f"lab-{c.sha[:7]}" / "results.json" if root else None
    return _copy_in(src, c.job_dir / "corpus" / "results.json", "corpus results")


def game_cmd() -> Cmd:
    return Cmd("game-copy", ["{py}", "{tree}/referee/linkcopy.py", "--src", "{game_src}", "--dst", "{game}"],
               needs=["{tree}/referee/linkcopy.py", "{game_src}"], when=lambda c: "game copy exists" if c.game.exists() else None)


def report_steps(c: Ctx) -> list[Step]:
    """Groups and kits (other branches' tools; skipped where the tree does not have them yet)."""
    bl = ["--backlog", "{backlog}"] if c.backlog else []
    # This job has no done.rc while its groups step runs: name it, so its own fights and checks are grouped now.
    own = ["--job", "{job}"] if source_has(c.tree / "tools/fidelity/lab_groups.py", '"--job"')[0] else []
    groups = Cmd("groups", ["{py}", "tools/fidelity/lab_groups.py", "--lab", "{lab}", *bl, *own, "--out", "{lab}/groups.json", "--summary", "{lab}/summary.md"],
                 needs=["{tree}/tools/fidelity/lab_groups.py"])
    kits = Cmd("kits", ["{py}", "-m", "tools.fidelity.packet", "--top", "20", "--groups", "{lab}/groups.json", *bl,
                        "--out", "{backlog_dir}/packets"],
               needs=["{tree}/tools/fidelity/packet.py", "{lab}/groups.json"], when=_kits_ready)
    return [Step("groups", [groups]), Step("kits", [kits])]


def _kits_ready(c: Ctx) -> str | None:
    if not c.backlog:
        return "no backlog-*.json to rank the kits by"
    if not source_has(c.tree / "tools/fidelity/packet.py", "--groups")[0]:
        return "packet.py on this tree has no --groups yet"
    return None


# ---------------------------------------------------------------- fights

def fights_mode(tree: Path) -> tuple[str, bool]:
    """('targets' | 'coverage', whether scenarios.py takes --jobs), from the tree's sources."""
    has_targets, has_jobs = source_has(tree / "referee/scenarios.py", '"targets"', '"--jobs"')
    targets_tool = (tree / "tools/fidelity/lab_targets.py").exists()
    return ("targets" if has_targets and targets_tool else "coverage"), has_jobs


def _halt_on_unsafe(name: str) -> Callable[[Ctx], str | None]:
    def check(c: Ctx) -> str | None:
        f = c.job_dir / "fights" / f"{name}.summary.txt"
        try:
            first = f.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            return f"no summary line for {name}"
        net = re.search(r"network (\[.*?\]|None|\{.*?\})", first)
        prof = re.search(r"real profile touched (\w+)", first)
        if net and net.group(1) not in ("[]", "None"):
            return f"HALT: the game opened network connections in {name}: {net.group(1)[:120]}"
        if prof and prof.group(1) == "True":
            return f"HALT: the real game profile was touched in {name}"
        return None
    return check


def _cov_in(c: Ctx) -> None:
    src, dst = c.lab / "coverage-setup.json", c.job_dir / "fights" / "coverage-setup.json"
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def _cov_out(c: Ctx) -> str | None:
    src, dst = c.job_dir / "fights" / "coverage-setup.json", c.lab / "coverage-setup.json"
    if src.exists():
        dst.write_bytes(src.read_bytes())
    return None


def plan_fights(c: Ctx) -> list[Step]:
    p = c.params
    mode, has_jobs = fights_mode(c.tree)
    base = ["--mode", mode] + (["--targets", "{lab}/targets.json"] if mode == "targets" else [])
    steps: list[Step] = [Step("game", [game_cmd()], mode="serial")]
    if mode == "targets":
        bl = ["--backlog", "{backlog}"] if c.backlog else []
        steps.append(Step("targets", [Cmd("targets", ["{py}", "tools/fidelity/lab_targets.py", "--lab", "{lab}", "--out", "{lab}/targets.json", *bl,
                                                      "--codex", "{codex}"], needs=["{tree}/tools/fidelity/lab_targets.py", "{codex}"])],
                          mode="serial"))
    for i in range(p["batches"]):
        name = f"lab{c.job_name.split('-')[0]}-b{i:02d}"
        args = ["--name", name, *base, "--fights", str(p["batch"]), "--start", str(p["start"] + i), "--servers", str(p["servers"]),
                "--game", "{game}", "--shrink", str(p["shrink"])]
        argv = ["{py}", "referee/scenarios.py", *args, "--jobs", "{job}/fights"] if has_jobs else \
            ["{py}", "-c", SCENARIOS_SHIM, "{job}/fights", *args]
        cmd = Cmd(f"scen-{i:02d}", argv, needs=["{tree}/referee/scenarios.py", "{tree}/" + EMUDUMP, "{tree}/" + DRIVER_DLL],
                  mkdirs=["{job}/fights"], cleanup=[f"{{job}}/fights/{name}.*", f"{{game_parent}}/serve/setup-{name}*"],
                  after_cleanup=[f"{{game_parent}}/serve/setup-{name}*"], post=_halt_on_unsafe(name), core=True,
                  counts={"fights": p["batch"]}, outputs={"fights": f"fights/{name}.verdicts.json"})
        if mode == "coverage":
            cmd.pre, cmd.post = _cov_in, (lambda c, h=_halt_on_unsafe(name): _cov_out(c) or h(c))
        steps.append(Step(f"batch-{i:02d}", [cmd], mode="serial"))
    return steps + report_steps(c)


# ---------------------------------------------------------------- probes, mutation, invariants

def plan_probes(c: Ctx) -> list[Step]:
    ids = c.params.get("probe_ids") or []
    if not ids:
        raise Unavailable("no fix kit names a probe (grades.json lists none): probes.py takes a probe id list, not a queue")
    return [Step("game", [game_cmd()], mode="serial"),
            Step("probes", [Cmd("probes", ["{py}", "referee/probe.py", "run", "--name", "lab" + c.job_name.split("-")[0], "--only", ",".join(ids),
                                          "--game", "{game}", "--servers", "2", "--dir", "{job}/probes"],
                                needs=["{tree}/referee/probe.py", "{game}"], mkdirs=["{job}/probes"], core=True,
                                outputs={"probes": "probes"})], mode="serial"),
            *report_steps(c)]


def probe_ids(backlog_dir: Path) -> list[str]:
    """Probe ids the kits ask for: grades.json entries missing a rule that name `probes` (nothing else maps item to probe)."""
    try:
        grades = json.loads((backlog_dir / "packets" / "grades.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[str] = []
    for g in grades.values() if isinstance(grades, dict) else []:
        if isinstance(g, dict) and any("rule" in str(m).lower() for m in g.get("missing", [])):
            out += [str(x) for x in g.get("probes", []) if str(x) not in out]
    return out


def plan_mutation(c: Ctx) -> list[Step]:
    src = c.params.get("mutation_source")
    if not src:
        raise Unavailable("no finished fights job with a verdict file to prepare mutation from")
    o, e, r = (f"{{lab}}/{src}{s}" for s in (".out.json", ".emu.json", ".reanchor.json"))
    mk = lambda n, a, **kw: Cmd(n, ["{py}", "-m", "tools.fidelity.mutate", *a], needs=["{tree}/tools/fidelity/mutate.py", o, e, r], **kw)  # noqa: E731
    return [Step("mutation", [
        mk("mut-prepare", ["prepare", "--out-json", o, "--emu", e, "--dir", "{job}/mut"], core=True),
        mk("mut-schemata", ["schemata", "--dir", "{job}/mut", "--reanchor", r, "--mutants", str(c.params["mutants"])], timeout=3 * 3600),
        mk("mut-report", ["report", "--dir", "{job}/mut", "--results", "{job}/mut/schemata/results.jsonl", "--json", "{job}/mutation.json"],
           outputs={"mutation_reach": "mutation.json"}),
    ], mode="serial")]


def plan_invariants(c: Ctx) -> list[Step]:
    seeds = c.params["seeds"]
    common = ["--jobs", "{records}", "--emudump", "{tree}/" + EMUDUMP]
    need = ["{tree}/tools/fidelity/invariants.py", "{tree}/" + EMUDUMP, "{records}"]
    return [Step("invariants", [
        Cmd("meta", ["{py}", "-m", "tools.fidelity.invariants", "meta", *common, "--work", "{job}/work/meta", "--seeds", str(seeds),
                     "--limit", "300", "--parallel", str(c.params["procs"]), "--json", "{job}/meta.json"],
            needs=need, ok_rcs=(0, 1), core=True, mkdirs=["{job}/work"], outputs={"metamorphic": "meta.json"}),
        Cmd("invariants", ["{py}", "-m", "tools.fidelity.invariants", "run", *common, "--work", "{job}/work/run", "--json", "{job}/invariants.json"],
            needs=need, ok_rcs=(0, 1), mkdirs=["{job}/work"], outputs={"invariants": "invariants.json"}),
    ], mode="serial"), *report_steps(c)]


PLANS: dict[str, Callable[[Ctx], list[Step]]] = {
    "presort": plan_presort, "fights": plan_fights, "probes": plan_probes, "mutation": plan_mutation, "invariants": plan_invariants,
}


def plan(job: str, c: Ctx) -> list[Step]:
    return PLANS[job](c)


def runnable(steps: list[Step], c: Ctx) -> str | None:
    """None when the plan has a core command that can run, else the reason it cannot."""
    core = [(s.name, cmd) for s in steps for cmd in s.cmds if cmd.core]
    if not core:
        return None
    reasons = [cmd.skip_reason(c, sn) for sn, cmd in core]
    if all(r is not None for r in reasons):
        return reasons[0]
    return None
