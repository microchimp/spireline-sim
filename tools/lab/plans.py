"""The fix lab's jobs as data: each job is a list of steps, each step a few commands (one process each).

A plan is built from the job's recorded parameters, so an interrupted job resumes with the same plan. A command
is written symbolically (``{tree}``, ``{lab}``, ``{job}`` ...); the symbolic form is what job.json and the logs
keep, so they hold no absolute path of the user's. A command whose tool is missing on the tree, whose input is
absent, or that ``--skip`` names is skipped with a one-line reason, never an error.

Jobs, in the runner's order of value (runner.ROTATION): presort, fights (one of the strategies of tools/lab/mix.py), runs
(the whole-runs strategy), probes, mutation, invariants.
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

from tools.lab import hypotheses as hyp
from tools.lab import jobs as jobkinds
from tools.lab import retest
from tools.lab import stepcache as sc
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
    report_beside: bool = False  # groups and kits run beside the next job (runner.Report), not as steps of this one
    off: tuple[str, ...] = ()  # step names LAB/jobs.json `reports` switches off now (checks, corpus-import, groups, kits): their commands are skipped

    def tokens(self) -> dict[str, str]:
        f = lambda p: str(p).replace("\\", "/")  # noqa: E731
        return {"lab": f(self.lab), "tree": f(self.tree), "game": f(self.game), "game_src": f(self.game_src),
                "game_parent": f(self.game.parent), "fork": f(self.fork), "py": self.py, "sha": self.sha,
                "sha7": self.sha[:7], "job": f(self.job_dir), "jobname": self.job_name,
                "backlog": f(self.backlog) if self.backlog else "", "backlog_dir": f(self.backlog_dir),
                "codex": f(self.codex), "decompiled": f(self.decompiled), "records": f(self.records),
                "nuget": f(self.nuget), "python": self.py, "vs_installer": VS_INSTALLER,
                "hypotheses": f(self.backlog_dir.parent / "hypotheses")}

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
    append_log: bool = False  # a restarted command adds to its log (after a restart marker line) instead of replacing it
    mem_share: float = 0.0  # the share of the machine's RAM its processes may commit together (a Job Object limit; an allocation past it fails)

    def skip_reason(self, c: Ctx, step: str = "") -> str | None:
        for name in c.off:
            if step == name or self.name == name:
                return f"off in LAB/jobs.json reports.{name}"
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
                 core=True, when=_unchanged(f"codex-s{i}"), post=_stamp_after(f"codex-s{i}", [f"codex/s{i}.jsonl"])) for i in range(4)]
    # Both keep their results outside the job (the refset's scores folder, the corpus's runs folder); a copy goes
    # into the job folder under the roles lab_groups reads, so their symptoms reach the groups.
    refset = Cmd("refset", ["{py}", "-m", "tools.fidelity.refset_score", "--emu", "{tree}", "--procs", str(p["procs_refset"]),
                            "--label", "lab-{sha7}"], needs=["{tree}/tools/fidelity/refset_score.py", "{tree}/" + EMUDUMP],
                 ok_rcs=(0, 1), outputs={"refset_score": "refset/score.json"}, core=True, when=_unchanged("refset"),
                 post=_stamp_after("refset", ["refset/score.json"], _keep_refset))
    corpus = Cmd("corpus", ["{py}", "tools/fidelity/corpus.py", "check", "--no-scan", "--procs", str(p["procs_corpus"]), "--label", "lab-{sha7}"],
                 needs=["{tree}/tools/fidelity/corpus.py", "{tree}/" + EMUDUMP], ok_rcs=(0, 1),
                 outputs={"corpus": "corpus/results.json"}, core=True, when=_unchanged("corpus"),
                 post=_stamp_after("corpus", ["corpus/results.json"], _keep_corpus))
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


# ---- skip a check whose inputs have not changed (tools/lab/stepcache.py; reset order D44 step D)

CORPUS_CODE = ("tools/fidelity/corpus.py", "tools/fidelity/fight_compare.py", "tools/fidelity/reanchor_compare.py")
REFSET_CODE = ("tools/fidelity/refset_score.py", "tools/fidelity/fight_compare.py", "tools/fidelity/reanchor_compare.py", "referee/refset.py")


def check_parts(c: Ctx, step: str) -> dict[str, str]:
    """The named inputs of a presort check, each as a short digest. The tree's sha is not one of them: what the checks read of
    it is the engine (emulator and EmuDump sources, the data they copy) and the check's own code, so a landing that changes
    neither leaves the digest as it was. Read-only; evaluated just before the check would run, after the tree step."""
    t = c.tree
    data = [t / "tools/fidelity" / n for n in ("allowed_diff.json", "fuzz_explained.json", "hook_dispatch.json", "hook_families.json")]
    if step.startswith("codex-s"):  # EmuDump codex <dataset> --decompiled <dir> --character IRONCLAD --ascension 10 --shard i/4
        return {"engine": sc.engine_digest(t), "codex": sc.stat_sig(c.codex), "decompiled": sc.decompiled_sig(c.decompiled),
                "args": f"IRONCLAD A10 shard {step[len('codex-s'):]}/4"}
    if step == "corpus":  # the corpus check reads the corpus index (the fights) and replays them with EmuDump
        root = corpus_root(t)
        return {"engine": sc.engine_digest(t), "code": sc.files_hash([t / r for r in CORPUS_CODE] + data, t),
                "corpus": sc.file_hash(root / "index.json") if root else sc.MISSING}
    if step == "refset":  # the reference set's records and manifest, scored with EmuDump
        ref = c.backlog_dir.parent / "refset" / "v2"
        return {"engine": sc.engine_digest(t), "code": sc.files_hash([t / r for r in REFSET_CODE] + data, t),
                "set": sc.digest({n: sc.stat_sig(ref / n) for n in ("manifest.json", "refs-all.json", "refs-pilot.json", "record")})}
    raise KeyError(step)


def _unchanged(step: str) -> Callable[[Ctx], str | None]:
    """The check's `when`: skipped when its last stamp holds the digest of its inputs now and its output is still as written."""
    return sc.when_unchanged(lambda c: c.lab, step, lambda c: sc.digest(check_parts(c, step)))


def _stamp_after(step: str, outputs: list[str], then: Callable[[Ctx], str | None] | None = None) -> Callable[[Ctx], str | None]:
    """The check's `post`: what `then` does, and when that finds no problem the stamp that lets the next presort skip the check."""
    def post(c: Ctx) -> str | None:
        problem = then(c) if then else None
        if problem:
            return problem
        try:
            parts = check_parts(c, step)
            sc.write_stamp(c.lab, step, sc.digest(parts), [c.job_dir / o for o in outputs], job=c.job_name, parts=parts)
        except Exception as e:  # noqa: BLE001  (a stamp that cannot be written only means the next presort runs the check)
            return f"no skip stamp for {step}: {type(e).__name__}"
        return None
    return post


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


def report_cmds(c: Ctx) -> list[Cmd]:
    """Groups and kits (other branches' tools; skipped where the tree does not have them yet). Both skip themselves when nothing they
    read has changed (tools/lab/stepcache.py) and say so on a line starting 'step skipped'."""
    bl = ["--backlog", "{backlog}"] if c.backlog else []
    # This job has no done.rc while its groups step runs: name it, so its own fights and checks are grouped now.
    own = ["--job", "{job}"] if source_has(c.tree / "tools/fidelity/lab_groups.py", '"--job"')[0] else []
    groups = Cmd("groups", ["{py}", "tools/fidelity/lab_groups.py", "--lab", "{lab}", *bl, *own, "--out", "{lab}/groups.json", "--summary", "{lab}/summary.md"],
                 needs=["{tree}/tools/fidelity/lab_groups.py"])
    kits = Cmd("kits", ["{py}", "-m", "tools.fidelity.packet", "--top", "20", "--groups", "{lab}/groups.json", *bl,
                        "--out", "{backlog_dir}/packets"],
               needs=["{tree}/tools/fidelity/packet.py", "{lab}/groups.json"], when=_kits_ready)
    return [groups, kits]


def report_steps(c: Ctx) -> list[Step]:
    """The job's groups and kits steps; none when they run beside the next job (the runner's report, runner.Report)."""
    if c.report_beside:
        return []
    groups, kits = report_cmds(c)
    return [Step("groups", [groups]), Step("kits", [kits])]


def _kits_ready(c: Ctx) -> str | None:
    if not c.backlog:
        return "no backlog-*.json to rank the kits by"
    if not source_has(c.tree / "tools/fidelity/packet.py", "--groups")[0]:
        return "packet.py on this tree has no --groups yet"
    return None


# ---------------------------------------------------------------- fights

def fights_mode(tree: Path) -> tuple[str, bool]:
    """('targets' | 'coverage', whether scenarios.py takes --jobs), from the tree's sources (jobs without a strategy)."""
    has_targets, has_jobs = source_has(tree / "referee/scenarios.py", '"targets"', '"--jobs"')
    targets_tool = (tree / "tools/fidelity/lab_targets.py").exists()
    return ("targets" if has_targets and targets_tool else "coverage"), has_jobs


def _halt_on_unsafe(name: str) -> Callable[[Ctx], str | None]:
    """scenarios.py puts the safety report on the first line of its summary, batch.py on its second."""
    def check(c: Ctx) -> str | None:
        f = c.job_dir / "fights" / f"{name}.summary.txt"
        try:
            text = "\n".join(f.read_text(encoding="utf-8", errors="replace").splitlines()[:6])
        except OSError:
            return f"no summary line for {name}"
        net = re.search(r"network (\[.*?\]|None|\{.*?\})", text)
        prof = re.search(r"real profile touched (\w+)", text)
        if not net and not prof:
            return f"no summary line for {name}"
        if net and net.group(1) not in ("[]", "None"):
            return f"HALT: the game opened network connections in {name}: {net.group(1)[:120]}"
        if prof and prof.group(1) == "True":
            return f"HALT: the real game profile was touched in {name}"
        return None
    return check


def _halt_on_unsafe_runs(name: str) -> Callable[[Ctx], str | None]:
    """runs.py logs each session's report in its summary; a failed safety check or a report that does not show the
    network and the real profile untouched halts the lab. A crashed session is only noted (its runs are kept)."""
    def check(c: Ctx) -> str | None:
        f = c.job_dir / "runs" / f"{name}.summary.txt"
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return f"no summary for {name}"
        if "SAFETY CHECK FAILED" in text:
            return f"HALT: a game session's safety check failed in {name}"
        for net in re.findall(r"'network_connections': (\[.*?\]|None)", text):
            if net != "[]":
                return f"HALT: the game opened network connections in {name}: {net[:120]}"
        if re.search(r"'real_profile_touched': True", text):
            return f"HALT: the real game profile was touched in {name}"
        failed = len(re.findall(r"^session \d+ FAILED", text, re.M))
        return f"{failed} game session(s) crashed; the runs they wrote and the other sessions' runs were kept" if failed else None
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


def _hyp_in(c: Ctx) -> None:
    setups, _ = hyp.load_setups(c.path("{hypotheses}"))
    todo = hyp.pending(setups, hyp.read_results(c.lab))
    dst = c.job_dir / "hyp-in" / "pending"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "pending.json").write_text(json.dumps([hyp.doc_of(s) for s in todo + [s for s in setups if not s.get("testable")]], indent=1), encoding="utf-8")


def _groups_doc(c: Ctx) -> list:
    try:
        return json.loads((c.lab / "groups.json").read_text(encoding="utf-8")).get("groups", [])
    except (OSError, ValueError):
        return []


def _retest_in(c: Ctx) -> None:
    todo = retest.pending(_groups_doc(c), retest.read_results(c.lab))
    dst = c.job_dir / "retest-in"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "pending.json").write_text(json.dumps(todo, indent=1), encoding="utf-8")


def _retest_out(c: Ctx, safety: str | None) -> str | None:
    from tools.lab.mix import now_utc
    if not (safety or "").startswith("HALT"):
        retest.collect(c.job_dir, c.lab, now_utc(), git_head(c.tree))
    return safety


def _hyp_out(c: Ctx, safety: str | None) -> str | None:
    """Judge the verdicts into LAB/hypotheses.json unless the safety check halts the lab; its message is passed on."""
    from tools.lab.mix import now_utc
    if not (safety or "").startswith("HALT"):
        hyp.collect(c.job_dir, c.job_dir / "hyp-in", c.lab, now_utc())
    return safety


# strategy -> (scenarios.py mode, extra arguments, tool needs)
SCENARIO_STRATEGIES = {
    "aimed-bugs": ("targets", ["--aim", "bugs"]),
    "aimed-gaps": ("targets", ["--aim", "gaps"]),
    "choice-edges": ("choices", ["--features", "{lab}/cache/model_features.json", "--decompiled", "{decompiled}",
                      "--act1-seeds", "{lab}/cache/act1_seeds.json"]),
    "dense": ("dense", ["--act1-seeds", "{lab}/cache/act1_seeds.json"]),
    "under-tested": ("coverage", ["--act1-seeds", "{lab}/cache/act1_seeds.json"]),
    "hypotheses": ("hypotheses", ["--hypotheses", "{job}/hyp-in", "--shrink", "0", "--trace", "0"]),
    "retest": ("retest", ["--retest", "{job}/retest-in", "--shrink", "0", "--trace", "0"]),
}


def tree_can(tree: Path) -> dict[str, bool]:
    """What the tree's tools can do, read from their sources (a job type that cannot run is skipped with the reason)."""
    sc = tree / "referee/scenarios.py"
    aim, choices, analyze, budgets = source_has(sc, '"--aim"', '"choices"', '"analyze"', '"--shrink-cap"')
    return {"aim": aim, "choices": choices, "analyze": analyze, "budgets": budgets,
            "driver_edges": source_has(tree / "referee/driver/Combat.cs", "choice_edges")[0],
            "batch": (tree / "referee/batch.py").exists(), "runs": source_has(tree / "referee/runs.py", '"--jobs"')[0],
            "targets": (tree / "tools/fidelity/lab_targets.py").exists()}


def strategy_unavailable(strategy: str, c: Ctx) -> str | None:
    """Why a strategy cannot run on this tree now, or None. Checks the tools and the inputs, not the machine."""
    can = tree_can(c.tree)
    if not (c.tree / "referee/scenarios.py").exists() and strategy != "whole-runs":
        return "referee/scenarios.py is missing on this tree"
    if strategy in ("aimed-bugs", "aimed-gaps"):
        if not can["aim"]:
            return "scenarios.py on this tree has no --aim"
        if not can["targets"]:
            return "tools/fidelity/lab_targets.py is missing on this tree"
        if not c.codex.exists() and not (c.lab / "targets.json").exists():
            return "no Codex dataset and no targets.json to aim at"
        if strategy == "aimed-bugs" and not c.backlog:
            return "no backlog-*.json: aimed-bugs aims at what its open items name"
    elif strategy == "choice-edges":
        if not can["choices"]:
            return "scenarios.py on this tree has no choices mode"
        if not can["driver_edges"]:
            return "the driver on this tree has no choice_edges"
    elif strategy == "random":
        if not can["batch"] or not can["analyze"]:
            return "batch.py or scenarios.py --mode analyze is missing on this tree"
    elif strategy == "whole-runs":
        if not can["runs"]:
            return "runs.py on this tree has no --jobs"
    elif strategy == "retest":
        if not source_has(c.tree / "referee/scenarios.py", '"retest"')[0]:
            return "scenarios.py on this tree has no retest mode"
        if not retest.pending(_groups_doc(c), retest.read_results(c.lab)):
            return "no group waits for a retest (none marked in LAB/groups.json, or all decided)"
    elif strategy == "hypotheses":
        if not source_has(c.tree / "referee/scenarios.py", '"hypotheses"')[0]:
            return "scenarios.py on this tree has no hypotheses mode"
        setups, _ = hyp.load_setups(c.path("{hypotheses}"))
        if not hyp.pending(setups, hyp.read_results(c.lab)):
            return "no hypothesis setup is waiting (none in the folder, or all decided or out of attempts)"
    return None


def _scenarios_cmd(c: Ctx, name: str, i: int, mode: str, extra: list[str], batches: int | None = None) -> Cmd:
    """One scenarios.py call (a batch of aimed or setup fights, shrunk and traced) with the job's budgets."""
    p = c.params
    n = max(1, p["batches"] if batches is None else batches)
    per = lambda total, floor=1: str(max(floor, -(-int(total) // n)))  # noqa: E731  (a job's budget split over its batches)
    args = ["--name", name, "--mode", mode, "--fights", str(p["batch"]), "--start", str(p["start"] + i), "--servers", str(p["servers"]),
            "--game", "{game}", "--shrink", str(p["shrink"])]
    if mode == "targets":
        args += ["--targets", "{lab}/targets.json"]
    can = tree_can(c.tree)["budgets"]
    if can:  # budgets exist on this tree: every diverging fight is shrunk and traced, within the job's time
        args += ["--shrink-cap", str(p.get("shrink_cap", 160)), "--shrink-budget", per(p.get("shrink_budget", 6000), 100),
                 "--shrink-minutes", per(p.get("shrink_minutes", 45), 2), "--trace", str(p.get("trace", 1000)),
                 "--trace-minutes", per(p.get("trace_minutes", 20), 1), "--trace-servers", str(min(8, p["servers"]))]
    args += extra
    has_jobs = fights_mode(c.tree)[1]
    argv = ["{py}", "referee/scenarios.py", *args, "--jobs", "{job}/fights"] if has_jobs else ["{py}", "-c", SCENARIOS_SHIM, "{job}/fights", *args]
    cmd = Cmd(f"scen-{i:02d}", argv, needs=["{tree}/referee/scenarios.py", "{tree}/" + EMUDUMP, "{tree}/" + DRIVER_DLL],
              mkdirs=["{job}/fights"], cleanup=[f"{{job}}/fights/{name}.*", f"{{game_parent}}/serve/setup-{name}*"],
              after_cleanup=[f"{{game_parent}}/serve/setup-{name}*"], post=_halt_on_unsafe(name), core=True,
              counts={"fights": p["batch"]}, outputs={"fights": f"fights/{name}.verdicts.json"})
    if mode in ("coverage", "dense", "choices"):  # these weigh by what the lab's setup fights have covered
        cmd.pre, cmd.post = _cov_in, (lambda c, h=_halt_on_unsafe(name): _cov_out(c) or h(c))
    if mode == "retest":  # the groups waiting, one fight each; the verdicts clear or confirm the groups
        cmd.pre, cmd.post = _retest_in, (lambda c, h=_halt_on_unsafe(name): _retest_out(c, h(c)))
        cmd.counts = {"fights": len(retest.pending(_groups_doc(c), retest.read_results(c.lab)))}
    if mode == "hypotheses":  # only the setups still waiting are played, in the job's one batch; the verdicts decide them
        cmd.pre, cmd.post = _hyp_in, (lambda c, h=_halt_on_unsafe(name): _hyp_out(c, h(c)))
        setups, _ = hyp.load_setups(c.path("{hypotheses}"))
        cmd.counts = {"fights": len(hyp.to_specs(hyp.pending(setups, hyp.read_results(c.lab))))}
    return cmd


def _random_cmds(c: Ctx, name: str, i: int) -> list[Cmd]:
    """batch.py's random decks (its streaming pool), then scenarios.py --mode analyze: shrink, trace, verdicts."""
    p = c.params
    n = max(1, p["batches"])
    per = lambda total, floor=1: str(max(floor, -(-int(total) // n)))  # noqa: E731
    batch = Cmd(f"batch-{i:02d}", ["{py}", "referee/batch.py", "--name", name, "--fights", str(p["batch"]), "--start", str(p["start"] + i),
                                   "--parallel", str(p["servers"]), "--game", "{game}", "--jobs", "{job}/fights",
                                   "--coverage", "{lab}/coverage-random.json", "--timeout", "3600"],
                needs=["{tree}/referee/batch.py", "{tree}/" + EMUDUMP, "{tree}/" + DRIVER_DLL], mkdirs=["{job}/fights"],
                cleanup=[f"{{job}}/fights/{name}.*", f"{{game_parent}}/serve/{name}*"], after_cleanup=[f"{{game_parent}}/serve/{name}*"],
                ok_rcs=(0, 1), post=_halt_on_unsafe(name), core=True, counts={"fights": p["batch"]})  # rc 1: stopped early, what was played stays
    args = ["--name", name, "--mode", "analyze", "--jobs", "{job}/fights", "--servers", str(p["servers"]), "--game", "{game}",
            "--shrink", str(p["shrink"]), "--shrink-cap", str(p.get("shrink_cap", 160)), "--shrink-budget", per(p.get("shrink_budget", 6000), 100),
            "--shrink-minutes", per(p.get("shrink_minutes", 45), 2), "--trace", str(p.get("trace", 1000)),
            "--trace-minutes", per(p.get("trace_minutes", 20), 1), "--trace-servers", str(min(8, p["servers"]))]
    gone = [f"{{job}}/fights/{name}.verdicts.json", f"{{job}}/fights/{name}.shrink*", f"{{job}}/fights/{name}.packs", f"{{job}}/fights/{name}.trace.log"]
    analyze = Cmd(f"analyze-{i:02d}", ["{py}", "referee/scenarios.py", *args],
                  needs=["{tree}/referee/scenarios.py", "{tree}/" + EMUDUMP, "{tree}/" + DRIVER_DLL, f"{{job}}/fights/{name}.reanchor.json"],
                  cleanup=gone, after_cleanup=[f"{{game_parent}}/serve/setup-{name}*"], core=True,
                  outputs={"fights": f"fights/{name}.verdicts.json"})
    return [batch, analyze]


def plan_fights(c: Ctx) -> list[Step]:
    p = c.params
    strategy = p.get("strategy")
    if strategy:
        why = strategy_unavailable(strategy, c)
        if why:
            raise Unavailable(why)
        if strategy not in SCENARIO_STRATEGIES and strategy != "random":
            raise Unavailable(f"{strategy} is not a fights strategy")
    mode, _ = fights_mode(c.tree)
    steps: list[Step] = [Step("game", [game_cmd()], mode="serial")]
    needs_targets = (strategy is None and mode == "targets") or (strategy in ("aimed-bugs", "aimed-gaps"))
    if needs_targets:
        bl = ["--backlog", "{backlog}"] if c.backlog else []
        steps.append(Step("targets", [Cmd("targets", ["{py}", "tools/fidelity/lab_targets.py", "--lab", "{lab}", "--out", "{lab}/targets.json", *bl,
                                                      "--codex", "{codex}"], needs=["{tree}/tools/fidelity/lab_targets.py", "{codex}"])],
                          mode="serial"))
    # hypotheses: one batch per job, so each waiting setup is played once and judged once (more batches would replay
    # the not testable ones and spend their attempts inside one job)
    nb = 1 if strategy in ("hypotheses", "retest") else p["batches"]
    for i in range(nb):
        name = f"lab{c.job_name.split('-')[0]}-b{i:02d}"
        if strategy == "random":
            steps.append(Step(f"batch-{i:02d}", _random_cmds(c, name, i), mode="serial"))
        elif strategy:
            smode, extra = SCENARIO_STRATEGIES[strategy]
            steps.append(Step(f"batch-{i:02d}", [_scenarios_cmd(c, name, i, smode, extra, nb)], mode="serial"))
        else:  # a job from before strategies: targets (else coverage) mode as it was
            base = ["--mode", mode] + (["--targets", "{lab}/targets.json"] if mode == "targets" else [])
            args = ["--name", name, *base, "--fights", str(p["batch"]), "--start", str(p["start"] + i), "--servers", str(p["servers"]),
                    "--game", "{game}", "--shrink", str(p["shrink"])]
            argv = ["{py}", "referee/scenarios.py", *args, "--jobs", "{job}/fights"] if fights_mode(c.tree)[1] else \
                ["{py}", "-c", SCENARIOS_SHIM, "{job}/fights", *args]
            cmd = Cmd(f"scen-{i:02d}", argv, needs=["{tree}/referee/scenarios.py", "{tree}/" + EMUDUMP, "{tree}/" + DRIVER_DLL],
                      mkdirs=["{job}/fights"], cleanup=[f"{{job}}/fights/{name}.*", f"{{game_parent}}/serve/setup-{name}*"],
                      after_cleanup=[f"{{game_parent}}/serve/setup-{name}*"], post=_halt_on_unsafe(name), core=True,
                      counts={"fights": p["batch"]}, outputs={"fights": f"fights/{name}.verdicts.json"})
            if mode == "coverage":
                cmd.pre, cmd.post = _cov_in, (lambda c, h=_halt_on_unsafe(name): _cov_out(c) or h(c))
            steps.append(Step(f"batch-{i:02d}", [cmd], mode="serial"))
    return steps + report_steps(c) + [import_step()]


def import_step() -> Step:
    """Add the job's exact and divergent fights to the corpus (after groups has read them; prune then drops the raw file)."""
    imp = Cmd("corpus-import", ["{py}", "tools/lab/corpus_import.py", "--job", "{job}", "--budget-gb", "2", "--max-fights", "100000"],
              needs=["{tree}/tools/lab/corpus_import.py"])
    return Step("corpus-import", [imp])


def plan_runs(c: Ctx) -> list[Step]:
    """Whole seeded runs: the game plays them in a few sessions, the emulator replays and compares (runs.py)."""
    p = c.params
    why = strategy_unavailable("whole-runs", c)
    if why:
        raise Unavailable(why)
    steps: list[Step] = [Step("game", [game_cmd()], mode="serial")]
    repro = all(source_has(c.tree / "referee/scenarios.py", '"--runs-from"')) and (c.tree / "referee/run_repro.py").exists()
    for i in range(p["batches"]):
        name = f"lab{c.job_name.split('-')[0]}-w{i:02d}"
        cmd = Cmd(f"runs-{i:02d}", ["{py}", "referee/runs.py", "--name", name, "--runs", str(p["run_batch"]), "--start", str(p["start"] + i),
                                    "--procs", str(p["procs"]), "--stop-after-act", str(p.get("stop_after_act", 3)), "--game", "{game}",
                                    "--jobs", "{job}/runs"],
                  needs=["{tree}/referee/runs.py", "{tree}/" + EMUDUMP, "{tree}/" + DRIVER_DLL], mkdirs=["{job}/runs"],
                  cleanup=[f"{{job}}/runs/{name}.*", f"{{job}}/runs/scratch/{name}-*"], ok_rcs=(0, 1, 3, 4),  # 3: a session crashed, 4: halted below
                  post=_halt_on_unsafe_runs(name), core=True, counts={"runs": p["run_batch"]}, outputs={"runs": f"runs/{name}.runs.json"})
        cmds = [cmd]
        if repro:  # each run whose first divergence is in a fight, played again as a setup fight from the save before it
            rn = name + "-repro"
            cmds.append(Cmd(f"repro-{i:02d}", ["{py}", "referee/scenarios.py", "--name", rn, "--mode", "runs",
                                               "--runs-from", f"{{job}}/runs/{name}.runs.json", "--jobs", "{job}/runs",
                                               "--servers", str(p.get("repro_servers", 4)), "--game", "{game}",
                                               "--shrink", "1000", "--shrink-cap", str(p.get("repro_shrink_cap", 60)),
                                               "--shrink-budget", str(p.get("repro_shrink_budget", 300)),
                                               "--shrink-minutes", str(p.get("repro_shrink_minutes", 20)),
                                               "--trace", "1000", "--trace-minutes", str(p.get("repro_trace_minutes", 5))],
                            needs=["{tree}/referee/scenarios.py", "{tree}/referee/run_repro.py", f"{{job}}/runs/{name}.runs.json"],
                            cleanup=[f"{{job}}/runs/{rn}.*"], after_cleanup=[f"{{game_parent}}/serve/setup-{rn}*"]))
        steps.append(Step(f"runs-{i:02d}", cmds, mode="serial"))
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
    """Probe ids the kits ask for: the `probes` that tools/fidelity/packet.py writes into grades.json for a kit still
    missing its probe answer (a catalog probe that matches the kit's item and has no verdict yet)."""
    try:
        grades = json.loads((backlog_dir / "packets" / "grades.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[str] = []
    for g in grades.values() if isinstance(grades, dict) else []:
        if isinstance(g, dict) and "probe answer" in [str(m) for m in g.get("missing", [])]:
            out += [str(x) for x in g.get("probes", []) if str(x) not in out]
    return out


# A mutant can loop without end and allocate as it goes (a flipped loop bound), and the engine it runs in then holds tens of GB
# until a stall timer kills it: 2026-11-05 that took the machine's virtual memory. The mutation commands share this part of the
# RAM, each host of tools.fidelity.mutate under its own slice of it (mutate.MEM_SHARE, the same number).
MUTATION_MEM_SHARE = 0.5


def plan_mutation(c: Ctx) -> list[Step]:
    src = c.params.get("mutation_source")
    if not src:
        raise Unavailable("no finished fights job with a verdict file to prepare mutation from")
    o, e, r = (f"{{lab}}/{src}{s}" for s in (".out.json", ".emu.json", ".reanchor.json"))
    mk = lambda n, a, **kw: Cmd(n, ["{py}", "-m", "tools.fidelity.mutate", *a], needs=["{tree}/tools/fidelity/mutate.py", o, e, r], **kw)  # noqa: E731
    return [Step("mutation", [
        mk("mut-prepare", ["prepare", "--out-json", o, "--emu", e, "--dir", "{job}/mut"], core=True, mem_share=MUTATION_MEM_SHARE),
        mk("mut-schemata", ["schemata", "--dir", "{job}/mut", "--reanchor", r, "--mutants", str(c.params["mutants"])], timeout=3 * 3600,
           mem_share=MUTATION_MEM_SHARE),
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


# ---------------------------------------------------------------- the agent-era kinds (tools/lab/jobs.py)

def _open_score_window(c: Ctx) -> None:
    (c.lab / "score").mkdir(parents=True, exist_ok=True)
    (c.lab / "score" / "active").write_text("1\n")


def _open_nightly_window(c: Ctx) -> None:
    (c.lab / "nightly").mkdir(parents=True, exist_ok=True)
    (c.lab / "nightly" / "active").write_text("1\n")


def plan_kind(kind: str, c: Ctx) -> list[Step]:
    """One command: the kind's configured command (params `cmd`, `cwd`, `timeout_min`, `needs`), core and required. Whether the
    kind is ready to run (marker, schedule, switch) is decided before the job is made (runner.unavailable), not here, so a
    job that is resumed after a stop is never refused. The command may write {job}/result.json (see jobs.py)."""
    p = c.params
    cmd = Cmd(kind, list(p["cmd"]), cwd=p.get("cwd", "{tree}"), needs=list(p.get("needs") or []), core=True,
              timeout=float(p.get("timeout_min", 60)) * 60, outputs={"result": "result.json"},
              post=lambda cx, k=kind: jobkinds.finish(k, cx.lab, cx.job_dir, cx.params))
    if kind == "score-real":  # the ranking of the lockstep divergences (tools/lab/rank.py) runs before the markers are consumed
        fin = cmd.post
        cmd.post = lambda cx, f=fin: jobkinds.rank_score_real(cx.job_dir) or f(cx)
    if kind == "score-real":  # the window marker keeps other fights runners off the game while it is open
        cmd.pre = _open_score_window
        cmd.after_cleanup = ["{lab}/score/active"]
    if kind == "nightly-real":  # the marker the next pick reads (jobs.nightly_blocked); the driver's own ranking replaces rank_score_real
        cmd.append_log = True  # the night's story must survive a restart (a stuck first attempt's log was once overwritten)
        cmd.pre = _open_nightly_window
        cmd.after_cleanup = ["{lab}/nightly/active"]
    return [Step(kind, [cmd], mode="serial", required=True)]


PLANS: dict[str, Callable[[Ctx], list[Step]]] = {
    "presort": plan_presort, "fights": plan_fights, "runs": plan_runs, "probes": plan_probes, "mutation": plan_mutation,
    "invariants": plan_invariants,
    **{k: (lambda c, k=k: plan_kind(k, c)) for k in jobkinds.NEW_KINDS},
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
