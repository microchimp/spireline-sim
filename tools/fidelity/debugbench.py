"""Debugging benchmark from the fork's own history, SWE-bench style: past engine fixes undone on today's main.

    python -m tools.fidelity.debugbench --truth-store <dir> harvest [--ref main] [--jobs 3] [--only <sha>...] [--limit N]
    python -m tools.fidelity.debugbench list [--split dev|heldout|all] [--truth [--heldout]]
    python -m tools.fidelity.debugbench signal (<task>... | --all) [--gates corpus,invariants,refset,codex] [--missing]
    python -m tools.fidelity.debugbench prepare <task> --out <dir> [--heldout]
    python -m tools.fidelity.debugbench --truth-store <dir> run <task> --run <name> --fixer "<command>" --sandbox <dir>
    python -m tools.fidelity.debugbench --truth-store <dir> score <task> --repo <repo> --ref <branch> --run <name>
        [--report <fixer report .json/.md>] [--verdict accept|reject] [--round N] [--gates corpus] [--quick]
    python -m tools.fidelity.debugbench --truth-store <dir> report [--run <name>] [--split dev|heldout|all]
    python -m tools.fidelity.debugbench --truth-store <dir> migrate

Run from the fork's root with the upstream venv's Python. Everything it writes is local and never committed. Two
places: the bench (--bench, default D:/spireline/referee/debugbench or SPIRELINE_DEBUGBENCH: the public half of
each task, its signal, main's gate results, worktrees under <bench>/wt) and the truth store (--truth-store or
SPIRELINE_DEBUGBENCH_TRUTH, no default, never under the bench: each task's fix, message, hidden tests, sites,
mechanism and diffs, the harvest log, harvest and score logs, every score). A fixer gets neither: `run` puts its
repo in a sandbox clear of both and of the fork (main is the answer), strips the bench's variables, denies it
those folders and voids an attempt whose own tool calls name one.

harvest  A candidate is a non-merge commit on --ref that changed the engine (src/Sts2Emulator/) and a test
         file (src/Sts2Emulator.Tests/, tools/emudump.Tests/). It becomes a task when
         - it is a fix: no conventional feat/refactor/style/perf/chore/docs/test/build/ci prefix, at most
           --max-lines changed engine lines, not the same engine patch (patch-id) as a newer candidate;
         - its engine change reverts cleanly on today's main: the reverse diff of src/Sts2Emulator/ only,
           applied 3-way (git revert restricted to the engine; everything else stays current);
         - a test fails without it: at the fix commit with the engine its parent had (the parent plus the
           fix's tests and any non-engine change) a test of the classes the fix touched fails, and it
           passes at the fix (fail-to-pass; those tests must build without the engine change);
         - on the base (main with the engine change reverted) those tests still fail and, once they are
           taken out, every test main passes passes. Hidden: the fix's own fail-to-pass tests. Collateral:
           other main tests that fail on the base (they would point at the site). Both are taken out of the
           base (the test methods, or a test file the fix added once nothing of it is left) and run at
           scoring. Outside the engine, the base loses every line that names the answer (scrub_hints: in
           docs the line or list item, in test sources a whole-line comment, naming the fix's commit, its
           hidden or collateral tests, or code the gold patch adds that the base lacks), and its one commit
           says "base".
         Ground truth: the gold patch (base -> main, engine only), its files and sites (file::Type.Member
         holding a changed line of code), the mechanism (each site's callers and callees on main, and the
         hook table's (model, hook) pairs whose dispatch points the base changes) and the fix's commit
         message. A task's id is a hash of its fix commit; the split is by that hash too (HELDOUT_PER_MILLE),
         so a task never changes side. Held-out tasks are not prepared or listed with their truth unless
         --heldout.
signal   What the gates show on the broken base, without the truth, each against main's own result: the
         corpus check (every recorded fight replayed; corpus.compare, the gate's rule), the oracle-free
         invariants, the reference set v2 (refset_score's gate per cell, fight, class and coverage) and the
         Spire Codex whole-run replay (the same refset_score run with --codex: every Ironclad A10 run walked,
         a run that stops earlier than on main is worse). refset_score is the base's own, run with
         python -I, its scores kept in a private copy of the reference set's folder (the shared history is
         never written). Per gate: detected (or not judged), what it names, seconds. A gate not run keeps
         its last result.
run      One fixer attempt (the harness): setup (the sandbox, a one-commit repo of the base, the brief: the
         gates' lines and output only, the deny profile) and scoring are timed apart from the fixer, whose
         wall time is its process and whose tokens are read from its own output (Claude Code's json or
         stream-json usage; unknown is never 0). Its tool calls are audited: naming the fork, the bench, the
         truth store or the bench's own tool contaminates the attempt. Fixers run on detected tasks; a
         silent one only with --silent-ok (blind, scored apart).
score    A candidate by behaviour, never by its diff against the gold patch: main's hidden and collateral
         tests (main's test files put back), main's other tests (unless --quick), the --gates against main,
         and how much of the mechanism it covers (sibling_credit: a site whole when it or a member it calls
         changes, else by the share of its callers that change; a hook pair by its dispatch points). fixed:
         all of it. partial: the hidden tests pass with no gate worse, a sibling is left. disputed: no gate
         worse and one better than main while a test of main fails (a fix better than main, for review).
         wrong: the rest. Judged with the base's judge (tools/fidelity, data/id_map.json put back).
report   Without --run: the task set and what each gate detects (and what the added gates catch of what the
         first ones miss). With --run: detection, then the fix rate of the detected tasks only (silent tasks
         apart, contaminated attempts nowhere), mechanism credit, beats main, localisation, first-round
         acceptance, regressions added, cost as the harness measured it.

A fixer's report is JSON {"suspects": ["src/Sts2Emulator/X.cs::Type.Member", ...]} or Markdown whose section
headed "Suspects" lists them in order; hours and tokens in it are kept as reported, never used.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import concurrent.futures
import contextlib
import datetime
import hashlib
import io
import json
import os
import queue
import re
import secrets
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools.fidelity import corpus as cp  # noqa: E402

DEFAULT_FORK = Path(os.environ.get("SPIRELINE_FORK", "D:/spireline/fork/sts2"))
DEFAULT_BENCH = Path(os.environ.get("SPIRELINE_DEBUGBENCH", "D:/spireline/referee/debugbench"))
DEFAULT_CORPUS = Path(os.environ.get("SPIRELINE_CORPUS", "D:/spireline/referee/corpus"))
DEFAULT_JOBS = Path(os.environ.get("SPIRELINE_JOBS", "D:/spireline/referee/jobs"))
REFSET_DIR = Path(os.environ.get("SPIRELINE_REFSET", "D:/spireline/referee/refset"))
REFSET_VERSION = "v2"
# The gates a signal or a score can run, in order. The first signal run (2026-10-14) ran the first two only.
GATES = ("corpus", "invariants", "refset", "codex")
LEGACY_GATES = ("corpus", "invariants")
FORK_BASE = "b7452170ce1df91de29d3c94acb11d87c7efc708"  # FORK.md: every commit after it is ours or a reviewed pick
PY = sys.executable
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
CLOCK = time.perf_counter  # the harness's clock (a test moves it)

ENGINE = "src/Sts2Emulator/"
TEST_DIRS = {"src/Sts2Emulator.Tests/": "src/Sts2Emulator.Tests/Sts2Emulator.Tests.csproj",
             "tools/emudump.Tests/": "tools/emudump.Tests/EmuDump.Tests.csproj"}
JUDGE_PATHS = ("tools/fidelity/", "data/id_map.json")
FINDINGS = "referee/FINDINGS.md"
# Held out: this many per mille of task ids, by the id's own hash. Fixed: changing it moves tasks a fixer has
# already seen into the held-out set. 425 holds out 30 of the 80 tasks of the 2026-10-14 harvest (main 3fc51181).
HELDOUT_PER_MILLE = 425
NOT_A_FIX = re.compile(r"^(feat|refactor|style|perf|chore|docs|test|tests|build|ci)(\([^)]*\))?!?:", re.I)
TRAILER = re.compile(r"^\s*(co-authored-by|signed-off-by|reviewed-by|reported-by):|^\s*\(cherry picked from commit", re.I)
CHERRY = re.compile(r"\(cherry picked from commit ([0-9a-f]{7,40})\)")
BOT = {"GIT_AUTHOR_NAME": "debugbench", "GIT_AUTHOR_EMAIL": "debugbench@example.invalid",
       "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000", "GIT_COMMITTER_NAME": "debugbench",
       "GIT_COMMITTER_EMAIL": "debugbench@example.invalid", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000"}


class BenchError(RuntimeError):
    pass


class CorpusRunFailed(BenchError):
    """corpus.py run itself failed on this tree (an EmuDump crash takes the whole run down): the gate fails."""


def say(*a) -> None:
    print(*a, flush=True)


def scrub(text: str) -> str:
    """No user paths in anything stored: the home directory becomes ~."""
    home = str(Path.home())
    for h in {home, home.replace("\\", "/"), home.replace("/", "\\")}:
        text = text.replace(h, "~")
    return text


# ------------------------------------------------------------------------------------------------ git

def _git(repo: Path, args, inp: bytes | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], input=inp, capture_output=True,
                          env=dict(os.environ, **(env or {})), check=False)


def git(repo: Path, *args: str, inp: bytes | None = None, env: dict | None = None, check: bool = True) -> str:
    r = _git(repo, args, inp, env)
    if check and r.returncode != 0:
        raise BenchError(f"git {' '.join(args)[:200]}: {r.stderr.decode('utf-8', 'replace').strip()[:400]}")
    return r.stdout.decode("utf-8", "replace").strip()


def git_bytes(repo: Path, *args: str, inp: bytes | None = None, env: dict | None = None) -> bytes:
    r = _git(repo, args, inp, env)
    if r.returncode != 0:
        raise BenchError(f"git {' '.join(args)[:200]}: {r.stderr.decode('utf-8', 'replace').strip()[:400]}")
    return r.stdout


def git_ok(repo: Path, *args: str, inp: bytes | None = None, env: dict | None = None) -> bool:
    return _git(repo, args, inp, env).returncode == 0


def blob(repo: Path, rev: str, path: str) -> bytes | None:
    r = _git(repo, ["cat-file", "blob", f"{rev}:{path}"])
    return r.stdout if r.returncode == 0 else None


def tree_of(repo: Path, rev: str) -> str:
    return git(repo, "rev-parse", f"{rev}^{{tree}}")


def tree_blobs(repo: Path, tree: str, prefix: str, suffix: str = ".cs") -> dict:
    """path -> text of every file under prefix ending in suffix, read from the object store (no checkout)."""
    names = [n for n in git(repo, "ls-tree", "-r", "--name-only", tree, "--", prefix).splitlines() if n.endswith(suffix)]
    if not names:
        return {}
    out = git_bytes(repo, "cat-file", "--batch", inp="".join(f"{tree}:{n}\n" for n in names).encode("utf-8"))
    res, i = {}, 0
    for n in names:
        j = out.index(b"\n", i)
        size = int(out[i:j].split()[2])
        res[n] = out[j + 1:j + 1 + size].decode("utf-8", "replace")
        i = j + 1 + size + 1
    return res


def has_object(repo: Path, sha: str, kind: str) -> bool:
    return git_ok(repo, "cat-file", "-e", f"{sha}^{{{kind}}}")


@contextlib.contextmanager
def scratch_index(repo: Path, tree: str):
    """A private index holding tree: tree edits without a working tree (no checkout is touched)."""
    d = tempfile.mkdtemp(prefix="debugbench_idx_")
    try:
        env = {"GIT_INDEX_FILE": str(Path(d) / "index")}
        git(repo, "read-tree", tree, env=env)
        yield env
    finally:
        shutil.rmtree(d, ignore_errors=True)


def diff_patch(repo: Path, a: str, b: str, paths=(".",)) -> bytes:
    return git_bytes(repo, "diff", "--binary", "--full-index", "--no-renames", a, b, "--", *paths)


def apply_to_tree(repo: Path, tree: str, patch: bytes, three_way: bool) -> str | None:
    """tree with patch applied (index only), or None when it does not apply cleanly (3-way: no conflict)."""
    if not patch.strip():
        return tree
    with scratch_index(repo, tree) as env:
        args = ["apply", "--cached", *(["--3way"] if three_way else []), "-"]
        if not git_ok(repo, *args, inp=patch, env=env):
            return None
        r = _git(repo, ["write-tree"], env=env)
        return r.stdout.decode().strip() if r.returncode == 0 else None


def edit_tree(repo: Path, tree: str, changes: dict) -> str:
    """tree with these files replaced (bytes) or removed (None)."""
    if not changes:
        return tree
    with scratch_index(repo, tree) as env:
        for path, data in sorted(changes.items()):
            if data is None:
                git(repo, "update-index", "--force-remove", "--", path, env=env)
                continue
            sha = git(repo, "hash-object", "-w", "--stdin", inp=data)
            ls = git(repo, "ls-tree", tree, "--", path)
            mode = ls.split()[0] if ls else "100644"
            git(repo, "update-index", "--add", "--cacheinfo", f"{mode},{sha},{path}", env=env)
        return git(repo, "write-tree", env=env)


def commit_tree(repo: Path, tree: str, message: str) -> str:
    """A parentless commit of tree with a fixed identity and date: the same tree and message give the same sha."""
    return git(repo, "commit-tree", "--no-gpg-sign", tree, "-m", message, env=BOT)


def changed(repo: Path, a: str, b: str, paths=(".",)) -> list:
    out = git(repo, "diff", "--name-status", "--no-renames", a, b, "--", *paths)
    return [(ln.split("\t", 1)[0][0], ln.split("\t", 1)[1]) for ln in out.splitlines() if "\t" in ln]


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def changed_lines(repo: Path, a: str, b: str, path: str) -> tuple[set, set]:
    """Lines of path removed or changed on a's side and added or changed on b's side."""
    old, new = set(), set()
    for ln in git(repo, "diff", "-U0", "--no-renames", a, b, "--", path).splitlines():
        m = HUNK.match(ln)
        if m:
            os_, oc = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            ns, nc = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
            old.update(range(os_, os_ + oc))
            new.update(range(ns, ns + nc))
    return old, new


def worktree(repo: Path, path: Path, commit: str) -> Path:
    """A detached worktree of repo at commit (made once, then moved; ignored build output is kept)."""
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        git(repo, "worktree", "add", "--detach", "--force", str(path), commit)
    else:
        git(path, "checkout", "--detach", "--force", "-q", commit)
        git(path, "clean", "-fdq")
    return path


def park(ctx, slots: list) -> None:
    """Leave the bench's worktrees on main, so no base (a held-out one least of all) stays checked out."""
    for wt in slots:
        if (wt / ".git").exists():
            with contextlib.suppress(BenchError):
                worktree(ctx.fork, wt, ctx.main)


def overlay(tree: Path, files: dict) -> None:
    for rel, data in files.items():
        p = tree / rel
        if data is None:
            p.unlink(missing_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)


# ------------------------------------------------------------------------------------------- C# outline

def _char_end(t: str, i: int) -> int:
    k = i + 1
    if k < len(t) and t[k] == "\\":
        k += 2
    else:
        k += 1
    j = t.find("'", k)
    return j + 1 if 0 <= j and j - i < 12 else i + 1


def _hole_end(t: str, k: int) -> int:
    """End of an interpolation hole starting at k (just after its '{')."""
    depth, n = 0, len(t)
    while k < n:
        ch = t[k]
        if ch in "\"$@":
            e = _string_end(t, k)
            if e is not None:
                k = e
                continue
        elif ch == "'":
            k = _char_end(t, k)
            continue
        elif ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return k + 1
            depth -= 1
        k += 1
    return n


def _string_end(t: str, i: int) -> int | None:
    """End (exclusive) of the string literal starting at i ("", @"", $"", $@"", raw), or None if none starts."""
    n, k, dollars, verbatim = len(t), i, 0, False
    while k < n and t[k] in "$@" and k - i < 6:
        dollars += t[k] == "$"
        verbatim |= t[k] == "@"
        k += 1
    if k >= n or t[k] != '"':
        return None
    if t.startswith('"""', k):
        q = k
        while q < n and t[q] == '"':
            q += 1
        quotes = q - k
        j = t.find('"' * quotes, q)
        return n if j < 0 else j + quotes
    k += 1
    while k < n:
        ch = t[k]
        if verbatim and ch == '"':
            if t.startswith('""', k):
                k += 2
                continue
            return k + 1
        if not verbatim:
            if ch == "\\":
                k += 2
                continue
            if ch == '"':
                return k + 1
            if ch == "\n":
                return k
        if dollars and ch == "{":
            if t.startswith("{{", k):
                k += 2
                continue
            k = _hole_end(t, k + 1)
            continue
        k += 1
    return n


def cs_mask(text: str) -> str:
    """text with comments, strings and char literals blanked (newlines kept), so its braces can be counted."""
    out, n, i = list(text), len(text), 0

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "
    while i < n:
        ch = text[i]
        if ch == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif ch == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif ch == "'":
            j = _char_end(text, i)
            blank(i, j)
            i = j
        elif ch in "\"$@":
            j = _string_end(text, i)
            if j is None:
                i += 1
                continue
            blank(i, j)
            i = j
        else:
            i += 1
    return "".join(out)


TYPE_KW = re.compile(r"\b(class|struct|interface|record|enum)\s+(\w+)")
NS_KW = re.compile(r"^\s*namespace\s+([\w.]+)", re.M)
KEYWORDS = {"if", "for", "foreach", "while", "switch", "using", "lock", "catch", "return", "new", "typeof", "nameof",
            "base", "sizeof", "default", "checked", "unchecked", "fixed", "when", "await", "throw", "is", "as", "in"}
MODIFIERS = {"public", "private", "protected", "internal", "static", "readonly", "override", "virtual", "abstract",
             "sealed", "async", "extern", "unsafe", "partial", "const", "volatile", "ref", "out", "params", "this",
             "implicit", "explicit", "void", "file", "required"}
TEST_ATTR = re.compile(r"\[\s*(?:Xunit\.)?\w*(?:Fact|Theory)\w*\b")  # [Fact], [Theory], [AuditFact(..)], [SkippableFact]
CASE = re.compile(r"^\s*case\s+(.+?)\s*(?:\bwhen\b.*)?:(?!:)")


def _strip_attrs(h: str) -> str:
    h = h.strip()
    while h.startswith("["):
        depth = 0
        for k, ch in enumerate(h):
            depth += ch == "["
            depth -= ch == "]"
            if depth == 0:
                h = h[k + 1:].lstrip()
                break
        else:
            return ""
    return h


def _top(h: str) -> str:
    """h with the contents of (), [] removed, so '=' and names are read at the top level."""
    out, depth = [], 0
    for ch in h:
        if ch in "([":
            depth += 1
            if depth == 1:
                out.append(ch)
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            if depth == 0:
                out.append(ch)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


ASSIGN = re.compile(r"(?<![=!<>+\-*/%&|^?])=(?![=>])")


def member_name(header: str) -> str | None:
    h = _strip_attrs(header)
    t = _top(h)
    cut = len(t)
    for m in (re.search(r"=>", t), ASSIGN.search(t)):
        if m:
            cut = min(cut, m.start())
    decl = t[:cut].strip()
    if not decl or decl.startswith(("=", ":")):
        return None
    m = re.search(r"operator\s*([^\s(]+)\s*\(", decl)
    if m:
        return "operator" + m.group(1)
    if "(" in decl:
        # The name is the first identifier before a parameter list that is no modifier or keyword, so a tuple
        # return type `(int a, int b) M()` and a constructor's `: base(x)` come out right.
        for p in (k for k, ch in enumerate(decl) if ch == "("):
            m = re.search(r"(\w+)\s*(?:<[^()]*>)?\s*$", decl[:p])
            if m and m.group(1) not in KEYWORDS and m.group(1) not in MODIFIERS:
                return m.group(1)
        return None
    words = re.findall(r"\w+", decl)
    return words[-1] if words and words[-1] not in KEYWORDS else None


def cs_outline(text: str) -> dict:
    """The namespace, types and members of a C# file: (qualified name, first line, last line), 1-based.

    A member's first line is its first attribute or modifier; Type.Member names nest types with '.'."""
    m = cs_mask(text)
    starts = [0] + [k + 1 for k, ch in enumerate(m) if ch == "\n"]

    def line(idx: int) -> int:
        return bisect.bisect_right(starts, idx)

    def first(a: int, b: int) -> int:
        seg = m[a:b]
        off = len(seg) - len(seg.lstrip())
        return a + off if off < len(seg) else b
    ns = NS_KW.search(m)
    types, members, stack = [], [], []
    hstart, paren = 0, 0

    def qual(name: str) -> str:
        return ".".join([s["name"] for s in stack if s["kind"] == "type"] + [name])
    for i, ch in enumerate(m):
        if ch in "([":
            paren += 1
        elif ch in ")]":
            paren = max(0, paren - 1)
        elif ch == "{":
            header = m[hstart:i]
            parent = stack[-1]["kind"] if stack else "top"
            kind, name = "block", None
            if paren > 0:
                kind = "expr"
            elif parent in ("top", "ns", "type"):
                h = _strip_attrs(header)
                ht = _top(h)
                pre = ht.split("(", 1)[0]
                tm = TYPE_KW.search(pre)
                if re.match(r"\s*namespace\b", h):
                    kind = "ns"
                elif tm:
                    kind, name = "type", tm.group(2)
                elif parent == "type":
                    if "=>" in ht or ASSIGN.search(ht):
                        kind = "expr"
                    else:
                        name = member_name(header)
                        kind = "member" if name else "block"
            stack.append({"kind": kind, "name": name, "start": line(first(hstart, i)), "hstart": hstart,
                          "paren": paren, "test": bool(TEST_ATTR.search(header))})
            paren = 0
            if kind != "expr":
                hstart = i + 1
        elif ch == "}":
            if not stack:
                continue
            top = stack.pop()
            paren = top["paren"]
            if top["kind"] == "type":
                types.append((qual(top["name"]), top["start"], line(i)))
            elif top["kind"] == "member":
                members.append((qual(top["name"]), top["start"], line(i), top["test"]))
            hstart = top["hstart"] if top["kind"] == "expr" else i + 1
        elif ch == ";" and paren == 0:
            parent = stack[-1]["kind"] if stack else "top"
            if parent == "type":
                header = m[hstart:i]
                name = member_name(header)
                if name and not TYPE_KW.search(_top(_strip_attrs(header)).split("(", 1)[0]):
                    members.append((qual(name), line(first(hstart, i)), line(i), bool(TEST_ATTR.search(header))))
            hstart = i + 1
    depth, d = [0], 0
    for ln in m.split("\n"):
        depth.append(d)  # depth[k]: braces open at the start of line k (1-based)
        d += ln.count("{") - ln.count("}")
    cases = {}
    for k, (raw, masked) in enumerate(zip(text.split("\n"), m.split("\n")), 1):
        if masked.lstrip().startswith("case "):
            cm = CASE.match(raw)
            if cm:
                cases[k] = re.sub(r"\s+", " ", cm.group(1))[:80]
    return {"namespace": ns.group(1) if ns else "", "types": types, "members": members, "depth": depth, "cases": cases}


def site_of(outline: dict, ln: int) -> str:
    """Type.Member holding line ln, refined to Type.Member[case X] inside a switch section (one card's branch of
    a big switch is its own site), else the innermost type, else (file)."""
    best = None
    for q, s, e, *_ in outline["members"]:
        if s <= ln <= e and (best is None or e - s < best[2] - best[1]):
            best = (q, s, e)
    if best is None:
        for q, s, e in outline["types"]:
            if s <= ln <= e and (best is None or e - s < best[2] - best[1]):
                best = (q, s, e)
        return best[0] if best else "(file)"
    depth, cases = outline.get("depth"), outline.get("cases") or {}
    if depth and cases and ln < len(depth):
        low = depth[ln]
        for j in range(ln, best[1] - 1, -1):
            low = min(low, depth[j])
            if j in cases and depth[j] <= low:
                return f"{best[0]}[case {cases[j]}]"
    return best[0]


def member_of(site: str) -> str:
    """A site without its switch-section refinement: file::Type.Member."""
    return re.sub(r"\[case .*\]$", "", site)


def sites_between(repo: Path, a: str, b: str, paths: list) -> list:
    """file::Type.Member for every member holding a line of code changed between a and b (C# files; others:
    file::(file)). A line that is only a comment or blank is no site: a doc comment above a member is outside it,
    and nothing a fix must touch."""
    out = set()
    for path in paths:
        old, new = changed_lines(repo, a, b, path)
        for rev, lines in ((a, old), (b, new)):
            if not lines:
                continue
            data = blob(repo, rev, path)
            if data is None:
                continue
            if not path.endswith(".cs"):
                out.add(f"{path}::(file)")
                continue
            text = data.decode("utf-8", "replace")
            o, code = cs_outline(text), cs_mask(text).split("\n")
            out.update(f"{path}::{site_of(o, ln)}" for ln in lines if ln - 1 < len(code) and code[ln - 1].strip())
    return sorted(out)


def test_fqns(text: str) -> dict:
    """Namespace.Type.Member -> (first, last line) for the test methods of a C# file."""
    o = cs_outline(text)
    pre = o["namespace"] + "." if o["namespace"] else ""
    return {pre + q: (s, e) for q, s, e, is_test in o["members"] if is_test}


def class_filters(text: str) -> list:
    """Namespace.TopType of every top-level type in a C# file (dotnet test filters)."""
    o = cs_outline(text)
    pre = o["namespace"] + "." if o["namespace"] else ""
    return sorted({pre + q.split(".")[0] for q, _, _ in o["types"]})


def strip_members(text: str, spans: list) -> str:
    """text without these (first, last) line spans, nor the comment lines right above each."""
    lines = text.splitlines(keepends=True)
    drop = set()
    for s, e in spans:
        k = s - 1
        while k - 1 >= 0 and lines[k - 1].lstrip().startswith("//"):
            k -= 1
        drop.update(range(k, e))
        if e < len(lines) and not lines[e].strip() and (k == 0 or not lines[k - 1].strip()):
            drop.add(e)  # no double blank line where the method was
    return "".join(ln for j, ln in enumerate(lines) if j not in drop)


def test_key(name: str) -> str:
    """A trx test name (Ns.Type.Method(args), nested types with '+') as Ns.Type.Method."""
    return name.split("(", 1)[0].strip().replace("+", ".")


# ------------------------------------------------------------------------------------------------ hints

NAME = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b")
BULLET = re.compile(r"\s*(?:[-*+]|\d+[.)])\s")
DOCS = (".md", ".txt")
# Never scrubbed: the engine is the task itself; the judge and its data are main's again at scoring.
NO_SCRUB = (ENGINE, "tools/fidelity/", "data/")


def _named(tok: str) -> bool:
    """An identifier that names something (CamelCase, UPPER_SNAKE), not a word of prose."""
    return sum(c.isupper() for c in tok) >= 2 or ("_" in tok.strip("_") and any(c.isupper() for c in tok))


def hint_names(ctx: "Ctx", base_tree: str, hidden: list, collateral: list) -> list:
    """Names that point at the answer from outside the engine: identifiers the gold patch adds that the base's
    engine does not have (they name code the base lacks), and the hidden and collateral tests (the method, and the
    class when the base's tests have no class of that name). Only the base's code counts as having a name: one
    in a comment or a string is the hint itself."""
    def code_words(prefix: str) -> set:
        return set(re.findall(r"\w+", "\n".join(cs_mask(t) for t in tree_blobs(ctx.fork, base_tree, prefix).values())))
    eng = code_words(ENGINE)
    names = set()
    for _, f in changed(ctx.fork, base_tree, ctx.main_tree, [ENGINE]):
        _, new = changed_lines(ctx.fork, base_tree, ctx.main_tree, f)
        data = blob(ctx.fork, ctx.main_tree, f)
        if not new or data is None:
            continue
        code = cs_mask(data.decode("utf-8", "replace")).split("\n")  # the gold's code: a game name in a comment is no hint
        names.update(t for ln in new if ln - 1 < len(code) for t in NAME.findall(code[ln - 1])
                     if _named(t) and t not in eng)
    tests = set().union(*(code_words(d) for d in TEST_DIRS))
    for fqn in list(hidden) + list(collateral):
        names.update(p for p in fqn.split(".")[-2:] if p not in tests and len(p) >= 6)
    return sorted(names)


def _drop_doc_lines(text: str, hit) -> tuple[str, int]:
    """text without the lines hit names; a list item goes with its continuation lines and sub-items."""
    lines = text.splitlines(keepends=True)
    out, i, n = [], 0, 0
    while i < len(lines):
        ln = lines[i]
        if not hit(ln):
            out.append(ln)
            i += 1
            continue
        n += 1
        ind = len(ln) - len(ln.lstrip())
        i += 1
        if BULLET.match(ln):
            while i < len(lines) and lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) > ind:
                i += 1
                n += 1
    return "".join(out), n


def _drop_comment_lines(text: str, hit) -> tuple[str, int]:
    """C# text without the whole-line comments hit names (code, strings and trailing comments stay)."""
    raw, masked = text.split("\n"), cs_mask(text).split("\n")
    keep = [r for r, m in zip(raw, masked) if not (r.lstrip().startswith("//") and not m.strip() and hit(r))]
    return "\n".join(keep), len(raw) - len(keep)


def scrub_hints(ctx: "Ctx", tree: str, names: list, shas: list) -> tuple[str, dict]:
    """tree without the lines outside the engine that name the answer (a hint name, or the fix's commit): in docs
    (*.md, *.txt) the line, a list item with its continuation and sub-items; in test sources a whole-line comment.
    A name left in code, in a string or in another file is listed ("left"), for the truth."""
    shas = [s for s in shas if s]
    pats = [re.compile(rf"(?<!\w){re.escape(n)}(?!\w)") for n in names] + [re.compile(re.escape(s)) for s in shas]
    info = {"names": list(names), "dropped": {}, "left": []}
    if not pats:
        return tree, info

    def hit(line: str) -> bool:
        return any(p.search(line) for p in pats)
    with tempfile.TemporaryDirectory() as d:
        pf = Path(d) / "patterns"
        pf.write_text("\n".join([*names, *shas]) + "\n", encoding="utf-8")
        r = _git(ctx.fork, ["grep", "-l", "-I", "-F", "-f", str(pf), tree, "--", "."])
    paths = sorted({ln.split(":", 1)[1] for ln in r.stdout.decode("utf-8", "replace").splitlines() if ":" in ln})
    changes = {}
    for path in paths:
        data = blob(ctx.fork, tree, path) or b""
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new, n = text, 0
        if not path.startswith(NO_SCRUB):
            if path.endswith(DOCS):
                new, n = _drop_doc_lines(text, hit)
            elif path.endswith(".cs") and proj_of(path):
                new, n = _drop_comment_lines(text, hit)
        if n:
            changes[path] = new.encode("utf-8")
            info["dropped"][path] = n
        if any(hit(ln) for ln in new.splitlines()):
            info["left"].append(path)
    return edit_tree(ctx.fork, tree, changes), info


# -------------------------------------------------------------------------------------------- mechanism

def engine_outlines(texts: dict) -> dict:
    """path -> (outline, masked text, line starts) of every C# file of an engine."""
    out = {}
    for p, t in texts.items():
        m = cs_mask(t)
        out[p] = (cs_outline(t), m, [0] + [k + 1 for k, ch in enumerate(m) if ch == "\n"])
    return out


def _span(outline: dict, q: str):
    return next(((s, e) for qq, s, e, *_ in outline["members"] if qq == q), None)


def callers_of(path: str, q: str, outl: dict, decl: collections.Counter, skip: set) -> list:
    """The engine members that use Type.Member q of path (by name: Type.Member anywhere, or the bare name inside
    its own type, or any use when no other member has that name), its own body and the members in skip apart."""
    typ, name = q.rsplit(".", 1)
    tname = typ.rsplit(".", 1)[-1]
    own = _span(outl[path][0], q)
    unique = decl[name] == 1
    rx = re.compile(rf"(?<![\w])(?:(\w+)\s*\??\.\s*)?{re.escape(name)}(?!\w)")
    out = set()
    for p, (o, m, starts) in outl.items():
        if name not in m:
            continue
        for h in rx.finditer(m):
            ln = bisect.bisect_right(starts, h.end() - len(name))
            if p == path and own and own[0] <= ln <= own[1]:
                continue
            site = site_of(o, ln)
            pre = h.group(1)
            if pre and pre not in ("this", "base"):
                if pre != tname and not unique:
                    continue
            elif not unique and not (site == typ or site.startswith(typ + ".")):
                continue
            c = member_of(f"{p}::{site}")
            if c.endswith("::(file)") or "." not in c.split("::", 1)[1] or c in skip:
                continue
            out.add(c)
    return sorted(out)


def callees_of(path: str, q: str, outl: dict) -> list:
    """The members of q's own type that q calls by bare name (one level): a fix there repairs q too."""
    o, m, starts = outl[path]
    own = _span(o, q)
    if not own:
        return []
    typ = q.rsplit(".", 1)[0]
    body = m[starts[own[0] - 1]:starts[own[1]] if own[1] < len(starts) else len(m)]
    same = {qq.rsplit(".", 1)[1]: qq for qq, *_ in o["members"] if qq.rsplit(".", 1)[0] == typ and qq != q}
    called = dict.fromkeys(c for c in re.findall(r"(?<![\w.])(\w+)\s*\(", body) if c in same)
    return [f"{path}::{same[c]}" for c in called]


DISPATCH_AT = re.compile(r"^\w+\.\w+(?:\[\w+\])?:\d+$")


def hook_rows_at(engine_root: Path) -> dict | None:
    """(Model.Hook) -> the emulator dispatch points that serve it, by the hook table (tools/fidelity/hook_table.py),
    for an engine on disk; None when the table does not apply (no hook table here, or no local decompile)."""
    try:
        from tools.fidelity import hook_table as ht
    except ImportError:
        return None
    dec = os.environ.get("SPIRELINE_DECOMPILED", "D:/spireline/referee/decompiled")
    if not (Path(dec) / "MegaCrit.Sts2.Core.Hooks" / "Hook.cs").exists():
        return None
    engine = ht.Engine.load(Path(engine_root))
    open_missing = {k.split(" ", 1)[1] for k in ht.read_open() if k.startswith("missing ")}
    rows = ht.classify(str(dec), engine, ht.load_json(ht.DISPATCH), ht.load_json(ht.UNMODELLED), open_missing)
    return {f"{ht.short(r.model)}.{r.hook}": sorted({w.rsplit(":", 1)[0] for w in r.where if DISPATCH_AT.match(w)})
            for r in rows}


def hook_rows_of(ctx: "Ctx", tree: str) -> dict | None:
    """hook_rows_at for the engine of a git tree (its Core/ written to a scratch folder), cached per tree."""
    if ctx.hooks is not None:
        return ctx.hooks(ctx, tree)
    if tree not in ctx.hook_cache:
        with tempfile.TemporaryDirectory(prefix="debugbench_eng_") as d:
            root = Path(d) / "engine"
            for p, text in tree_blobs(ctx.fork, tree, ENGINE + "Core/").items():
                f = root / p[len(ENGINE):]
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(text, encoding="utf-8", newline="")
            ctx.hook_cache[tree] = hook_rows_at(root)
    return ctx.hook_cache[tree]


def hook_units(main_rows: dict | None, base_rows: dict | None) -> list:
    """The (model, hook) pairs whose dispatch points differ between the base and main: the planted bug's place in
    the hook table, each with every dispatch point main serves it at."""
    if main_rows is None or base_rows is None:
        return []
    return [{"hook": k, "main": main_rows.get(k, []), "base": base_rows.get(k, [])}
            for k in sorted(set(main_rows) | set(base_rows)) if set(main_rows.get(k, [])) != set(base_rows.get(k, []))]


def mechanism(ctx: "Ctx", base_tree: str, main_tree: str, sites: list) -> dict:
    """The planted mechanism's siblings, for the credit of a fix: per truth site its callers on main (the members
    that reach it) and its callees (members of its type it calls), and the hook table's (model, hook) pairs whose
    dispatch points the base changes."""
    outl = engine_outlines(tree_blobs(ctx.fork, main_tree, ENGINE))
    decl = collections.Counter(q.rsplit(".", 1)[-1] for o, _, _ in outl.values() for q, *_ in o["members"])
    gold = {member_of(s) for s in sites}
    units = []
    for s in sites:
        path, _, q = s.partition("::")
        u = {"site": s, "callers": [], "callees": []}
        if path in outl and "[case " not in q and "." in q:
            u["callers"] = callers_of(path, q, outl, decl, gold)
            u["callees"] = callees_of(path, q, outl)
        units.append(u)
    main_rows, base_rows = hook_rows_of(ctx, main_tree), hook_rows_of(ctx, base_tree)
    return {"units": units, "hooks": hook_units(main_rows, base_rows),
            "hook_table": "applied" if main_rows is not None and base_rows is not None else "not applied"}


def sibling_credit(mech: dict, cand_sites: set, cand_hooks: dict | None) -> dict:
    """How much of the planted mechanism a candidate repairs, by place, never by diff equality: a truth site counts
    whole when the candidate changes it (its member) or a member it calls, else by the share of its callers the
    candidate changes; a hook pair by the share of its dispatch points that are main's again."""
    members = {member_of(s) for s in cand_sites}
    rows = []
    for u in mech.get("units", []):
        s = u["site"]
        if s in cand_sites or ("[case " not in s and member_of(s) in members) or any(c in members for c in u.get("callees", [])):
            c, how = 1.0, "changed"
        elif u.get("callers"):
            hit = [x for x in u["callers"] if x in members]
            c, how = len(hit) / len(u["callers"]), f"callers {len(hit)}/{len(u['callers'])}"
        else:
            c, how = 0.0, "missed"
        rows.append({"unit": s, "credit": round(c, 3), "how": how})
    for h in mech.get("hooks", []):
        main_set = set(h["main"])
        diff = main_set ^ set(h["base"])
        if cand_hooks is None:
            rows.append({"unit": f"hook {h['hook']}", "credit": None, "how": "hook table not applied"})
            continue
        cand = set(cand_hooks.get(h["hook"], []))
        ok = [d for d in diff if (d in cand) == (d in main_set)]
        rows.append({"unit": f"hook {h['hook']}", "credit": round(len(ok) / len(diff), 3) if diff else 1.0,
                     "how": f"dispatch points {len(ok)}/{len(diff)}"})
    vals = [r["credit"] for r in rows if r["credit"] is not None]
    return {"units": len(rows), "credit": round(sum(vals) / len(vals), 3) if vals else None,
            "complete": sum(1 for v in vals if v >= 0.999), "missed": [r["unit"] for r in rows if r["credit"] == 0.0],
            "rows": rows}


# ----------------------------------------------------------------------------------------- toolchain

def parse_trx(path: Path) -> dict:
    rank = {"fail": 2, "pass": 1, "skip": 0}
    out = {}
    for el in ET.parse(path).getroot().iter():
        if not el.tag.endswith("UnitTestResult"):
            continue
        v = {"Passed": "pass", "Failed": "fail", "NotExecuted": "skip"}.get(el.get("outcome", ""), "fail")
        k = test_key(el.get("testName", ""))
        if k not in out or rank[v] > rank[out[k]]:
            out[k] = v
    return out


class Dotnet:
    """Builds and runs the fork's C# suites offline from the NuGet cache (the gate's own toolchain)."""

    def __init__(self) -> None:
        self.cache = os.environ.get("NUGET_PACKAGES", str(Path.home() / ".nuget" / "packages"))
        self.env = dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_NOLOGO="1", PYTHONDONTWRITEBYTECODE="1")

    def _run(self, args: list, log: Path, timeout: float) -> int:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.env,
                             creationflags=BELOW_NORMAL)
        try:
            out, _ = p.communicate(timeout=timeout)
            text, rc = out.decode("utf-8", "replace"), p.returncode
        except subprocess.TimeoutExpired:
            # The test host is a child of dotnet: end the whole tree, or it keeps running (and holding the build).
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, check=False)
            p.kill()
            p.communicate()
            text, rc = f"timed out after {timeout:.0f} s\n", 124
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(Path(a).name if os.sep in a or '/' in a else a for a in args[:3])} ... rc {rc}\n")
            f.write(scrub(text)[-20000:] + "\n")
        return rc

    def build(self, tree: Path, proj: str, log: Path) -> bool:
        return self._run(["dotnet", "build", str(tree / proj), "--source", self.cache, "-v", "q", "--nologo"],
                         log, 1800) == 0

    def tests(self, tree: Path, proj: str, filters: list | None, work: Path, log: Path) -> dict | None:
        """Per test (Ns.Type.Method): pass, fail or skip; None when the run left no results."""
        work.mkdir(parents=True, exist_ok=True)
        name = f"r{time.time_ns()}.trx"
        args = ["dotnet", "test", str(tree / proj), "--no-build", "--nologo", "--logger", f"trx;LogFileName={name}",
                "--results-directory", str(work)]
        if filters:
            args += ["--filter", "|".join(f"FullyQualifiedName~{f}" for f in filters)]
        self._run(args, log, 900 if not filters else 600)  # the whole suite takes about a minute
        trx = work / name
        if not trx.exists():
            return None
        try:
            return parse_trx(trx)
        finally:
            trx.unlink(missing_ok=True)

    def emudump(self, tree: Path, log: Path) -> Path | None:
        rc = self._run(["dotnet", "build", str(tree / "tools" / "emudump" / "EmuDump.csproj"), "-c", "Release",
                        "--source", self.cache, "-v", "q", "--nologo"], log, 1800)
        dll = tree / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
        if rc != 0 or not dll.exists():
            return None
        # When only the engine changed, msbuild rebuilds and copies Sts2Emulator.dll but skips EmuDump.dll's own
        # compile (the engine's reference assembly is the same), so EmuDump.dll keeps an older time than the
        # sources though this folder is this tree's build: it is marked built now (refset_score checks the time).
        os.utime(dll)
        return dll


class Gates:
    """The gates a fixer runs, each from the tree under test (its own judge, as the real gate)."""

    def __init__(self, corpus: Path, jobs: Path, procs: int) -> None:
        self.corpus, self.jobs, self.procs = corpus, jobs, procs

    def corpus_run(self, tree: Path, dll: Path, label: str, keep: Path) -> dict:
        """corpus.py run from tree with dll; its results doc (the run folder is moved under keep)."""
        r = subprocess.run([PY, str(tree / "tools" / "fidelity" / "corpus.py"), "--corpus", str(self.corpus), "run",
                            "--emudump", str(dll), "--procs", str(self.procs), "--label", label, "--no-scan"],
                           capture_output=True, text=True, check=False, cwd=str(tree),
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        rdir = self.corpus / "runs" / label
        res = rdir / "results.json"
        if r.returncode != 0 or not res.exists():
            text = scrub(r.stdout + r.stderr)
            keep.mkdir(parents=True, exist_ok=True)
            (keep / "corpus-run.log").write_text(text, encoding="utf-8")
            shutil.rmtree(rdir, ignore_errors=True)
            last = [ln.strip() for ln in text.splitlines() if ln.strip()][-1:] or ["no output"]
            raise CorpusRunFailed(f"corpus run failed (exit {r.returncode}): {last[0][:300]}")
        doc = json.loads(res.read_text(encoding="utf-8"))
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res, keep / "corpus-results.json")
        shutil.rmtree(rdir, ignore_errors=True)
        return doc

    def invariants(self, tree: Path, dll: Path, work: Path) -> dict:
        out = work / "invariants.json"
        w = work / "inv-work"
        r = subprocess.run([PY, "-m", "tools.fidelity.invariants", "run", "--jobs", str(self.jobs), "--emudump", str(dll),
                            "--work", str(w), "--json", str(out)], capture_output=True, text=True, check=False,
                           cwd=str(tree), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), creationflags=BELOW_NORMAL)
        shutil.rmtree(w, ignore_errors=True)
        if r.returncode != 0 or not out.exists():
            text = scrub(r.stdout + r.stderr)
            (work / "invariants.log").write_text(text, encoding="utf-8")
            last = [ln.strip() for ln in text.splitlines() if ln.strip()][-1:] or ["no output"]
            return {"error": f"invariants run failed (exit {r.returncode}): {last[0][:300]}"}
        doc = json.loads(out.read_text(encoding="utf-8"))
        return {"emulator": doc["emulator"].get("violations", {}), "live": {k: v.get("violations", 0) for k, v in
                                                                            (doc.get("live", {}).get("checks") or {}).items()},
                "replay_errors": doc.get("live", {}).get("replay_errors", 0),
                "notes": doc.get("notes", {}), "examples": {k: v[:3] for k, v in doc["emulator"].get("examples", {}).items()}}

    def refset(self, tree: Path, label: str, view: Path, against: Path | None, codex: bool, work: Path) -> dict:
        """The tree's own refset_score (its judge), with --codex the Spire Codex whole-run replay too, its score kept
        in view (a private copy of the reference set's folder: the shared score history is never written). With
        against (main's score): the tree's own gate against it, {"worse", "unsound", "better", ...}; else {"score"}.
        Python -I: the walk reads third-party run histories (through EmuDump), and nothing on the path is the
        tree's but the tree itself."""
        work.mkdir(parents=True, exist_ok=True)
        runner = work / "refset_runner.py"
        runner.write_text(REFSET_RUNNER, encoding="utf-8")
        gate_out = work / "refset-gate.json"
        gate_out.unlink(missing_ok=True)
        args = ["--version", REFSET_VERSION, "--emu", str(tree), "--procs", str(self.procs)] + (["--codex"] if codex else [])
        r = subprocess.run([PY, "-I", str(runner), str(tree), str(view), REFSET_VERSION, label, str(against or "-"),
                            str(gate_out), *args], capture_output=True, text=True, check=False, cwd=str(tree),
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), creationflags=BELOW_NORMAL)
        text = scrub(r.stdout + r.stderr)
        (work / "refset.log").write_text(text[-30000:], encoding="utf-8")
        if r.returncode != 0 or (against and not gate_out.exists()):
            last = [ln.strip() for ln in text.splitlines() if ln.strip()][-1:] or ["no output"]
            return {"error": f"refset_score failed (exit {r.returncode}): {last[0][:300]}"}
        if against:
            return json.loads(gate_out.read_text(encoding="utf-8"))
        return {"score": str(view / REFSET_VERSION / "scores" / f"all-{label}.json")}


# Run with the tree's Python modules: argv tree view version label against|- gate_out refset_score-args...
REFSET_RUNNER = '''"""debugbench: refset_score of the tree in argv[1], its scores kept in a private reference-set folder."""
import json
import os
import sys
from pathlib import Path

tree, view, version, label, against, gate_out = sys.argv[1:7]
sys.path.insert(0, tree)
os.chdir(tree)
from tools.fidelity import refset_score as rs  # noqa: E402

rs.REFSET = Path(view)
sys.argv = ["refset_score", "--label", label, *sys.argv[7:]]
rc = rs.main()
if rc != 0:
    sys.exit(rc)
if against != "-":
    score = Path(view) / version / "scores" / f"all-{label}.json"
    new = json.loads(score.read_text(encoding="utf-8"))
    g = rs.gate(json.loads(Path(against).read_text(encoding="utf-8")), new)
    g["minutes"] = new.get("minutes")
    g["codex_runs"] = (new.get("codex") or {}).get("runs")
    Path(gate_out).write_text(json.dumps(g), encoding="utf-8")
    score.unlink()  # a task's score is a few MB; its gate is what the bench keeps
sys.exit(0)
'''


# ------------------------------------------------------------------------------------------- context

class Ctx:
    def __init__(self, fork: Path, bench: Path, ref: str = "main", toolchain=None, gates=None, truth=None,
                 hooks=None) -> None:
        self.fork, self.bench = Path(fork), Path(bench)
        self.main = git(self.fork, "rev-parse", ref)
        self.main_tree = tree_of(self.fork, self.main)
        self.tc = toolchain or Dotnet()
        self.gates = gates
        self.truth = Path(truth) if truth else None
        self.hooks = hooks  # tests: (ctx, tree) -> hook rows; None: the hook table (hook_rows_of)
        self.hook_cache: dict = {}
        self.lock = threading.Lock()
        self._index = None

    @property
    def tasks_dir(self) -> Path:
        return self.bench / "tasks"

    @property
    def main_dir(self) -> Path:
        return self.bench / "main" / self.main[:12]

    def slot(self, name: str) -> Path:
        return self.bench / "wt" / name

    def need_truth(self) -> Path:
        """The truth store: the scorer's side (answers, hidden tests, harvest log, scores), never under the bench."""
        if self.truth is None:
            raise BenchError("this needs the truth store (--truth-store or SPIRELINE_DEBUGBENCH_TRUTH); it has no "
                             "default, so a fixer running this tool from its own repo cannot find it")
        if inside(self.truth, self.bench) or inside(self.bench, self.truth):
            raise BenchError(f"the truth store {self.truth} and the bench {self.bench} must not hold each other")
        return self.truth

    def truth_dir(self, tid: str) -> Path:
        return self.need_truth() / "tasks" / tid

    def main_index(self) -> dict:
        """Ns.Type.Method -> test file path, for every test method on main."""
        if self._index is None:
            idx = {}
            files = git(self.fork, "ls-tree", "-r", "--name-only", self.main, "--", *TEST_DIRS).splitlines()
            for f in files:
                if f.endswith(".cs"):
                    data = blob(self.fork, self.main, f) or b""
                    for k in test_fqns(data.decode("utf-8", "replace")):
                        idx.setdefault(k, f)
            self._index = idx
        return self._index


def inside(p: Path, root: Path) -> bool:
    """p is root or below it (case-insensitively where the file system is)."""
    a, b = (Path(os.path.normcase(os.path.abspath(x))) for x in (p, root))
    return a == b or b in a.parents


@contextlib.contextmanager
def bench_lock(bench: Path, name: str):
    """One harvest, signal or score at a time per bench: they share its worktrees (wt/h*, wt/s*, wt/score)."""
    bench.mkdir(parents=True, exist_ok=True)
    p = bench / f".{name}.lock"
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise BenchError(f"another {name} runs on this bench ({p}; delete it if that run is gone)") from None
    try:
        os.write(fd, f"{os.getpid()} {datetime.datetime.now().isoformat(timespec='seconds')}\n".encode())
        os.close(fd)
        yield
    finally:
        p.unlink(missing_ok=True)


def task_id(fix: str) -> str:
    return "db-" + hashlib.sha256(f"debugbench:{fix}".encode()).hexdigest()[:10]


def split_of(tid: str) -> str:
    h = int(hashlib.sha256(f"debugbench-split:{tid}".encode()).hexdigest()[:8], 16)
    return "heldout" if h % 1000 < HELDOUT_PER_MILLE else "dev"


def projects_in(tree: Path) -> list:
    return [proj for proj in TEST_DIRS.values() if (tree / proj).exists()]


def proj_of(path: str) -> str | None:
    return next((proj for d, proj in TEST_DIRS.items() if path.startswith(d)), None)


def main_tests(ctx: Ctx) -> dict:
    """Every test's outcome on main (both suites), run once per main commit."""
    p = ctx.main_dir / "tests.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    wt = worktree(ctx.fork, ctx.slot("main"), ctx.main)
    out = {}
    for proj in projects_in(wt):
        if not ctx.tc.build(wt, proj, ctx.main_dir / "build.log"):
            raise BenchError(f"main does not build ({proj})")
        res = ctx.tc.tests(wt, proj, None, ctx.main_dir / "trx", ctx.main_dir / "test.log")
        if res is None:
            raise BenchError(f"main's {proj} run left no results")
        out[proj] = res
    ctx.main_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    return out


def main_corpus(ctx: Ctx) -> dict:
    """Main's own corpus run: the baseline every signal and score is judged against."""
    p = ctx.main_dir / "corpus-results.json"
    with ctx.lock:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        wt = worktree(ctx.fork, ctx.slot("main"), ctx.main)
        dll = ctx.tc.emudump(wt, ctx.main_dir / "emudump.log")
        if dll is None:
            raise BenchError("EmuDump does not build on main")
        return ctx.gates.corpus_run(wt, dll, f"debugbench-main-{ctx.main[:10]}", ctx.main_dir)


def main_invariants(ctx: Ctx) -> dict:
    p = ctx.main_dir / "invariants.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    wt = worktree(ctx.fork, ctx.slot("main"), ctx.main)
    dll = ctx.tc.emudump(wt, ctx.main_dir / "emudump.log")
    work = ctx.main_dir / "inv"
    work.mkdir(parents=True, exist_ok=True)
    res = ctx.gates.invariants(wt, dll, work)
    p.write_text(json.dumps(res), encoding="utf-8")
    return res


def refset_view(ctx: Ctx) -> Path:
    """A private copy of the reference set's folder for this main (its record pointer and manifest, a scores/ of its
    own), copied once: every task of the main is scored on the same record, and the shared history is never written."""
    v = ctx.main_dir / "refset" / REFSET_VERSION
    if not (v / "manifest.json").exists():
        src = REFSET_DIR / REFSET_VERSION
        if not (src / "manifest.json").exists():
            raise BenchError(f"no reference set {REFSET_VERSION} at {src}")
        (v / "scores").mkdir(parents=True, exist_ok=True)
        for name in ("refs-all.json", "refs-runs.json", "manifest.json"):  # the manifest last: it marks a full copy
            if (src / name).exists():
                shutil.copyfile(src / name, v / name)
    return v.parent


def main_refset(ctx: Ctx, codex: bool) -> Path:
    """Main's reference-set score (with the codex walk when codex): what the refset and codex gates judge against."""
    view = refset_view(ctx)
    label = f"debugbench-main-{ctx.main[:10]}" + ("-codex" if codex else "")
    with ctx.lock:
        for lab in ([label] if codex else [label + "-codex", label]):  # a score with the walk serves the refset gate
            p = view / REFSET_VERSION / "scores" / f"all-{lab}.json"
            if p.exists():
                return p
        wt = worktree(ctx.fork, ctx.slot("main"), ctx.main)
        if ctx.tc.emudump(wt, ctx.main_dir / "emudump.log") is None:
            raise BenchError("EmuDump does not build on main")
        r = ctx.gates.refset(wt, label, view, None, codex, ctx.main_dir / "refset-work")
        if "error" in r:
            raise BenchError(f"main's reference-set score: {r['error']}")
        return Path(r["score"])


CODEX_LINE = re.compile(r"^codex run (\d+): .*?at floor (\d+) \(([^)]*)\)")


def judge_refset(gate: dict, want: set) -> dict:
    """The tree's refset gate against main's score, split into the two signals: the codex whole-run replay
    (lines `codex ...`) and the reference set itself (cells, fights, classes, coverage, whole runs). An unsound
    comparison is an error of the signal, never a detection."""
    if "error" in gate:
        return {g: {"error": gate["error"], "detected": None} for g in want}
    worse, unsound, better = gate.get("worse", []), gate.get("unsound", []), gate.get("better", [])
    out = {}
    for g in ("refset", "codex"):
        if g not in want:
            continue
        mine = [w for w in worse if w.startswith("codex") == (g == "codex")]
        uns = [u for u in unsound if g == "codex" or "codex" not in u]
        res = {"worse": len(mine), "lines": mine[:25], "better": sum(1 for b in better if b.startswith("codex") == (g == "codex")),
               "minutes": gate.get("minutes")}
        if g == "codex":
            stops = collections.Counter(m.group(3) for w in mine if (m := CODEX_LINE.match(w)))
            res.update(runs=gate.get("codex_runs"), stops=dict(stops.most_common(8)))
        else:
            res["kinds"] = dict(collections.Counter(w.split(" ", 1)[0] for w in mine))
        if uns:
            res.update(detected=None, error="; ".join(uns)[:400])
        else:
            res["detected"] = bool(mine)
        out[g] = res
    return out


def gate_line(name: str, g: dict) -> str:
    """One line per gate for the brief: what the gate shows, in the gate's words, against the reference (main)."""
    if name == "corpus":
        if g.get("crashed"):
            return f"corpus check: the run fails on this repo ({g['crashed']})"
        return (f"corpus check against the reference run: {g.get('regressed')} fights worse"
                + (f" (fields {', '.join(list(g.get('fields') or {})[:6])})" if g.get("fields") else "")
                + f", {g.get('improved')} better; identical {(g.get('main_totals') or {}).get('identical')} -> "
                  f"{(g.get('totals') or {}).get('identical')}")
    if name == "invariants":
        return "invariants: " + (", ".join(f"{k} {a}->{b}" for k, (a, b) in (g.get("worse") or {}).items())
                                 or g.get("error") or "no new violation")
    if name == "refset":
        if g.get("error"):
            return f"reference set {REFSET_VERSION} gate: not judged ({g['error'][:200]})"
        return (f"reference set {REFSET_VERSION} gate against the reference score: {g['worse']} worse"
                + (f" ({'; '.join(g['lines'][:3])})" if g.get("lines") else ""))
    if name == "codex":
        if g.get("error"):
            return f"codex whole-run replay: not judged ({g['error'][:200]})"
        return (f"codex whole-run replay against the reference walk: {g['worse']} runs worse of {g.get('runs')}"
                + (f" (stops: {', '.join(f'{k} {v}' for k, v in g['stops'].items())})" if g.get("stops") else ""))
    return f"{name}: {g}"


# ------------------------------------------------------------------------------------------- harvest

def candidates(ctx: Ctx) -> list:
    eng = git(ctx.fork, "rev-list", "--no-merges", ctx.main, "--", ENGINE).split()
    tst = set(git(ctx.fork, "rev-list", "--no-merges", ctx.main, "--", *TEST_DIRS).split())
    return [c for c in eng if c in tst]


def message_of(ctx: Ctx, c: str) -> tuple[str, str]:
    raw = git(ctx.fork, "log", "-1", "--format=%B", c)
    lines = [ln for ln in raw.splitlines() if not TRAILER.match(ln)]
    subject = lines[0] if lines else ""
    return subject, "\n".join(lines).strip()


def screen(ctx: Ctx, c: str, max_lines: int) -> dict:
    """The cheap checks (no build): a fix, small enough, its engine revert applies on main."""
    parents = git(ctx.fork, "rev-list", "--parents", "-n", "1", c).split()[1:]
    subject, message = message_of(ctx, c)
    rec = {"fix": c, "subject": subject}
    if len(parents) != 1:
        return {**rec, "skip": "a root or merge commit"}
    p = parents[0]
    rec["parent"] = p
    m = NOT_A_FIX.match(subject)
    if m:
        return {**rec, "skip": f"not a fix ({m.group(1).lower()})"}
    files = changed(ctx.fork, p, c)
    tests = [f for s, f in files if proj_of(f)]
    if not any(f.endswith(".cs") for f in tests):
        return {**rec, "skip": "no test source changed"}
    num = git(ctx.fork, "diff", "--numstat", "--no-renames", p, c, "--", ENGINE)
    lines = sum(int(a) + int(b) for a, b, *_ in (ln.split("\t") for ln in num.splitlines()) if a.isdigit() and b.isdigit())
    rec["engine_lines"] = lines
    if lines > max_lines:
        return {**rec, "skip": f"engine change of {lines} lines (over --max-lines)"}
    patch = diff_patch(ctx.fork, p, c, [ENGINE])
    rec["patch_id"] = (git(ctx.fork, "patch-id", "--stable", inp=patch).split() or [""])[0]
    bwh = apply_to_tree(ctx.fork, ctx.main_tree, diff_patch(ctx.fork, c, p, [ENGINE]), three_way=True)
    if bwh is None:
        return {**rec, "skip": "the engine revert does not apply cleanly on main"}
    if bwh == ctx.main_tree:
        return {**rec, "skip": "reverting the engine change changes nothing on main"}
    cherry = CHERRY.search(git(ctx.fork, "log", "-1", "--format=%B", c))
    rec.update(message=message, bwh_tree=bwh, test_files=tests, added={f for s, f in files if s == "A"},
               engine_files=[f for _, f in files if f.startswith(ENGINE)],
               date=git(ctx.fork, "log", "-1", "--format=%as", c), cherry=cherry.group(1) if cherry else None,
               ours=git_ok(ctx.fork, "merge-base", "--is-ancestor", FORK_BASE, c))
    return rec


def _outcomes(ctx: Ctx, wt: Path, projs: list, filters: dict | None, work: Path, log: Path) -> dict | None:
    """Build the projects in wt and run them (per project: filters, or everything): outcomes, or None on a failed build."""
    out = {}
    for proj in projs:
        if not ctx.tc.build(wt, proj, log):
            return None
        res = ctx.tc.tests(wt, proj, (filters or {}).get(proj), work, log)
        if res is None:
            raise BenchError(f"the {proj} run left no results (crashed?)")
        out.update(res)
    return out


def _broken(main_t: dict, got: dict) -> set:
    """Tests main passes that fail here."""
    passing = {t for res in main_t.values() for t, v in res.items() if v == "pass"}
    return {t for t, v in got.items() if v == "fail" and t in passing}


def hide(ctx: Ctx, tree: str, tests: set, added: set, remove_files: bool) -> tuple[str, dict, list]:
    """tree without these test methods (main's files): the new tree, the hidden files (path -> main bytes), removed files."""
    by_file = collections.defaultdict(list)
    idx = ctx.main_index()
    for t in tests:
        if t not in idx:
            raise BenchError(f"cannot find {t} on main")
        by_file[idx[t]].append(t)
    changes, hidden, removed = {}, {}, []
    for f, names in sorted(by_file.items()):
        data = blob(ctx.fork, ctx.main, f)
        text = data.decode("utf-8")
        spans = test_fqns(text)
        missing = [n for n in names if n not in spans]
        if missing:
            raise BenchError(f"cannot locate {missing[0]} in {f}")
        hidden[f] = data
        left = set(spans) - set(names)
        if remove_files and f in added and not left:
            changes[f] = None
            removed.append(f)
        else:
            changes[f] = strip_members(text, [spans[n] for n in names]).encode("utf-8")
    return edit_tree(ctx.fork, tree, changes), hidden, removed


def verify(ctx: Ctx, cand: dict, wt_path: Path) -> dict:
    """The expensive checks of one screened candidate (builds and test runs in its slot's worktree): the task, or a skip."""
    c, p, fork = cand["fix"], cand["parent"], ctx.fork
    tid = task_id(c)
    work = ctx.need_truth() / "work" / tid  # its logs name the hidden tests: the scorer's side
    log = work / "harvest.log"
    if log.exists():
        log.unlink()
    rec = {"fix": c, "subject": cand["subject"], "id": tid}
    # 1. The fix's tests without its engine change: the fix commit with the parent's engine.
    pprime_tree = apply_to_tree(fork, tree_of(fork, c), diff_patch(fork, c, p, [ENGINE]), three_way=False)
    if pprime_tree is None:
        return {**rec, "skip": "cannot take the engine change out of the fix commit"}
    pprime = commit_tree(fork, pprime_tree, f"debugbench: {c[:10]} without its engine change")
    filters = collections.defaultdict(list)
    for f in cand["test_files"]:
        data = blob(fork, c, f)
        if f.endswith(".cs") and data is not None:
            filters[proj_of(f)] += class_filters(data.decode("utf-8", "replace"))
    filters = {k: sorted(set(v)) for k, v in filters.items() if v}
    if not filters:
        return {**rec, "skip": "the fix's test files hold no type"}
    wt = worktree(fork, wt_path, pprime)
    projs = [pj for pj in filters if (wt / pj).exists()]
    out_p = _outcomes(ctx, wt, projs, filters, work / "trx", log)
    if out_p is None:
        return {**rec, "skip": "the fix's tests do not build without its engine change (they need what it added)"}
    wt = worktree(fork, wt_path, c)
    out_c = _outcomes(ctx, wt, projs, filters, work / "trx", log)
    if out_c is None:
        return {**rec, "skip": "the fix commit does not build"}
    f2p = sorted(t for t, v in out_c.items() if v == "pass" and out_p.get(t) == "fail")
    if not f2p:
        return {**rec, "skip": "no test of the fix fails without it"}
    # 2. The base: main with the engine change reverted. The fix's tests must still fail there.
    main_t = main_tests(ctx)
    idx = ctx.main_index()
    on_main = [t for t in f2p if t in idx]
    if not on_main:
        return {**rec, "skip": "the fix's failing tests are gone from main", "f2p": f2p}
    bwh = cand["bwh_tree"]
    wt = worktree(fork, wt_path, commit_tree(fork, bwh, f"debugbench: {tid} with its hidden tests"))
    got = _outcomes(ctx, wt, projects_in(wt), None, work / "trx", log)
    if got is None:
        return {**rec, "skip": "the base does not build"}
    broken = _broken(main_t, got)
    hidden = sorted(t for t in on_main if t in broken)
    if not hidden:
        return {**rec, "skip": "the fix's tests pass on today's base (later code covers the change)", "f2p": f2p}
    collateral = sorted(broken - set(hidden))
    # 3. Take the hidden and collateral tests out; the rest of main's suite must pass on the base.
    removed, files = [], {}
    base = None
    for _round in range(3):
        for remove_files in (True, False):
            try:
                tree, files, removed = hide(ctx, bwh, set(hidden) | set(collateral), cand["added"], remove_files)
            except BenchError as ex:
                return {**rec, "skip": f"hiding the tests: {ex}"}
            commit = commit_tree(fork, tree, f"debugbench base {tid}")
            wt = worktree(fork, wt_path, commit)
            got = _outcomes(ctx, wt, projects_in(wt), None, work / "trx", log)
            if got is not None:
                break
            if not removed:
                return {**rec, "skip": "taking the hidden tests out breaks the build"}
        if got is None:
            return {**rec, "skip": "taking the hidden tests out breaks the build"}
        more = _broken(main_t, got)
        if not more:
            base = tree
            break
        collateral = sorted(set(collateral) | more)
    if base is None:
        return {**rec, "skip": "the base's suite does not pass once the failing tests are out (flaky?)"}
    # 4. Nothing outside the engine names the answer (docs and test comments: names the base lacks, the fix's
    # commit), and the base commit says nothing ("base"). Lines only: the suite that passed is unchanged.
    names = hint_names(ctx, base, hidden, collateral)
    tree, hints = scrub_hints(ctx, base, names, [c[:7], (cand.get("cherry") or "")[:7]])
    commit = commit_tree(fork, tree, "base")
    task = make_task(ctx, tid, c, p, cand, tree, commit, hidden, collateral, f2p, files, removed, hints)
    write_task(ctx, task, {"hidden": files, "gold.diff": diff_patch(fork, tree, ctx.main_tree, [ENGINE]),
                           "base.diff": diff_patch(fork, ctx.main_tree, tree), "fix.diff": diff_patch(fork, p, c)},
               fresh=True)
    return {**rec, "task": tid, "split": task["split"], "hidden": len(hidden), "collateral": len(collateral)}


def make_task(ctx: Ctx, tid: str, c: str, p: str, cand: dict, tree: str, commit: str, hidden: list, collateral: list,
              f2p: list, files: dict, removed: list, hints: dict) -> dict:
    """The task with its ground truth: the gold patch (base -> main, engine only), its sites and the mechanism's
    siblings (mechanism), the fix's commit message."""
    fork = ctx.fork
    eng = [f for _, f in changed(fork, tree, ctx.main_tree, [ENGINE])]
    num = git(fork, "diff", "--numstat", "--no-renames", tree, ctx.main_tree, "--", ENGINE)
    ins = sum(int(ln.split("\t")[0]) for ln in num.splitlines() if ln.split("\t")[0].isdigit())
    dels = sum(int(ln.split("\t")[1]) for ln in num.splitlines() if ln.split("\t")[1].isdigit())
    sites = sites_between(fork, tree, ctx.main_tree, eng)
    eng = sorted({s.split("::", 1)[0] for s in sites}) or eng  # a file the gold only comments in holds no site
    return {"id": tid, "split": split_of(tid), "main": ctx.main, "fix": c, "parent": p, "cherry_of": cand.get("cherry"),
            "subject": cand["subject"], "message": cand["message"], "date": cand["date"], "ours": cand["ours"],
            "base_tree": tree, "base_commit": commit, "hidden": hidden, "collateral": collateral,
            "fail_to_pass_at_fix": f2p, "test_files": sorted(files), "removed_files": removed,
            "truth": {"files": eng, "sites": sites, "insertions": ins, "deletions": dels},
            "mechanism": mechanism(ctx, tree, ctx.main_tree, sites), "hints": hints,
            "made": datetime.datetime.now().isoformat(timespec="seconds")}


PUBLIC = ("id", "split", "main", "base_tree", "base_commit", "made")


def write_task(ctx: Ctx, task: dict, files: dict, fresh: bool) -> None:
    """A task in two halves. The bench holds what a signal or a prepare needs and names no answer (tasks/<split>/
    <id>/task.json: the base, how many tests are hidden). The truth store holds the rest (tasks/<id>/: truth.json
    with the fix, its message, tests, sites and mechanism; gold.diff, base.diff, fix.diff; hidden/, main's test
    files): the scorer's side, never in a fixer's reach."""
    tdir_t = ctx.truth_dir(task["id"])
    if tdir_t.exists():
        shutil.rmtree(tdir_t)
    (tdir_t / "hidden").mkdir(parents=True)
    for f, data in files["hidden"].items():
        (tdir_t / "hidden" / f).parent.mkdir(parents=True, exist_ok=True)
        (tdir_t / "hidden" / f).write_bytes(data)
    for name in ("gold.diff", "base.diff", "fix.diff"):
        (tdir_t / name).write_bytes(files[name])
    (tdir_t / "truth.json").write_text(json.dumps(task, indent=1), encoding="utf-8")
    tdir = ctx.tasks_dir / task["split"] / task["id"]
    if fresh and tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir(parents=True, exist_ok=True)
    pub = {k: task[k] for k in PUBLIC} | {"format": 2, "hidden_count": len(task["hidden"]),
                                         "collateral_count": len(task["collateral"])}
    (tdir / "task.json").write_text(json.dumps(pub, indent=1), encoding="utf-8")


def load_truth(ctx: Ctx, tid: str) -> tuple[dict, Path]:
    d = ctx.truth_dir(tid)
    p = d / "truth.json"
    if not p.exists():
        pub, _ = load_task(ctx.bench, tid)
        if "fix" in pub:
            raise BenchError(f"{tid} keeps its truth in the bench (made before the truth store): "
                             "`debugbench --truth-store <dir> migrate`")
        raise BenchError(f"no truth for {tid} in {ctx.truth}")
    return json.loads(p.read_text(encoding="utf-8")), d


def cmd_harvest(a) -> int:
    with bench_lock(Path(a.bench), "harvest"):
        ctx = Ctx(a.fork, a.bench, a.ref, truth=a.truth_store)
        ctx.need_truth()
        return harvest(ctx, a.ref, a.only, a.limit, a.max_lines, a.jobs, a.force)


def harvest(ctx: Ctx, ref: str, only: list | None, limit: int, max_lines: int, jobs: int, force: bool) -> int:
    # One bench holds one main's tasks: a task's id is its fix's, so another main's harvest would overwrite them.
    other = {t["main"] for t in all_tasks(ctx.bench)} - {ctx.main}
    if other:
        raise BenchError(f"{ctx.bench} holds the tasks of main {sorted(other)[0][:10]}; harvest {ref} "
                         f"({ctx.main[:10]}) into another --bench, or --ref {sorted(other)[0][:10]} to add to them")
    state = ctx.need_truth() / "harvest" / f"{ctx.main[:12]}.jsonl"  # fix commits and subjects: the truth's
    state.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if state.exists() and not force:
        for ln in state.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            done[r["fix"]] = r
    cands = candidates(ctx)
    if only:
        cands = [c for c in cands if any(c.startswith(o) for o in only)]
    say(f"harvest: {len(cands)} candidates on {ref} {ctx.main[:10]} (engine and test changed, no merges)")
    firstp = set(git(ctx.fork, "rev-list", "--first-parent", ctx.main).split())
    cands.sort(key=lambda c: c not in firstp)  # a duplicate patch keeps its first-parent copy
    screened, seen = [], {}
    for c in cands:
        if c in done:
            continue
        s = screen(ctx, c, max_lines)
        if "skip" not in s:
            if s["patch_id"] in seen:
                s = {**s, "skip": f"same engine patch as {seen[s['patch_id']][:10]}"}
            else:
                seen[s["patch_id"]] = c
        screened.append(s)
    for s in screened:
        if "skip" in s:
            _log(state, {"fix": s["fix"], "subject": s["subject"], "skip": s["skip"], "stage": "screen"})
    passed = [s for s in screened if "skip" not in s]
    todo = passed[: limit or None]
    say(f"harvest: {len(passed)} pass the screen (fix, size, unique, reverts cleanly), {len(todo)} verified now; "
        f"{len(done)} judged before")
    if not todo:
        return write_index(ctx, state)
    main_tests(ctx)
    ctx.main_index()
    slots = queue.Queue()
    for k in range(jobs):
        slots.put(worktree(ctx.fork, ctx.slot(f"h{k}"), ctx.main))

    def one(cand: dict) -> dict:
        wt = slots.get()
        t0 = time.time()
        try:
            r = verify(ctx, cand, wt)
        except BenchError as ex:
            r = {"fix": cand["fix"], "subject": cand["subject"], "skip": f"error: {ex}"[:400]}
        finally:
            slots.put(wt)
        r.update(stage="verify", seconds=round(time.time() - t0))
        _log(state, {k: v for k, v in r.items() if k in ("fix", "subject", "skip", "stage", "task", "split", "hidden",
                                                         "collateral", "seconds", "f2p")})
        say(f"  {cand['fix'][:10]} {('TASK ' + r['task'] + ' ' + r['split']) if 'task' in r else 'skip: ' + r['skip']}"
            f" ({r['seconds']} s) {cand['subject'][:70]}")
        return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(one, todo))
    park(ctx, [ctx.slot(f"h{k}") for k in range(jobs)])
    return write_index(ctx, state)


_LOG_LOCK = threading.Lock()


def _log(state: Path, rec: dict) -> None:
    with _LOG_LOCK, state.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def write_index(ctx: Ctx, state: Path) -> int:
    recs = {}
    for ln in state.read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        recs[r["fix"]] = r
    skips = collections.Counter(re.sub(r"[0-9a-f]{10}|\d+ lines", "#", r["skip"]) for r in recs.values() if "skip" in r)
    tasks = []
    for t in all_tasks(ctx.bench):
        if t["main"] != ctx.main:
            continue
        want = split_of(t["id"])
        if t["split"] != want:  # the rule is the id's hash: a task made under another HELDOUT_PER_MILLE moves
            src = ctx.tasks_dir / t["split"] / t["id"]
            t["split"] = want
            (src / "task.json").write_text(json.dumps(t, indent=1), encoding="utf-8")
            (ctx.tasks_dir / want).mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(ctx.tasks_dir / want / t["id"]))
            tp = ctx.truth_dir(t["id"]) / "truth.json"
            if tp.exists():
                tt = json.loads(tp.read_text(encoding="utf-8"))
                tt["split"] = want
                tp.write_text(json.dumps(tt, indent=1), encoding="utf-8")
        tasks.append(t)
    idx = {"main": ctx.main, "made": datetime.datetime.now().isoformat(timespec="seconds"),
           "heldout_per_mille": HELDOUT_PER_MILLE,
           "tasks": sorted(({"id": t["id"], "split": t["split"]} for t in tasks), key=lambda x: x["id"]),
           "candidates": len(recs), "skipped": dict(skips.most_common())}
    ctx.tasks_dir.mkdir(parents=True, exist_ok=True)
    (ctx.tasks_dir / "index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")
    n = collections.Counter(t["split"] for t in idx["tasks"])
    say(f"harvest: {len(tasks)} tasks ({n['dev']} dev, {n['heldout']} held out) of {len(recs)} candidates")
    for why, k in skips.most_common():
        say(f"  skipped {k:4}  {why}")
    return 0


# ------------------------------------------------------------------------------------------- tasks

def load_task(bench: Path, tid: str) -> tuple[dict, Path]:
    for split in ("dev", "heldout"):
        d = bench / "tasks" / split / tid
        if (d / "task.json").exists():
            return json.loads((d / "task.json").read_text(encoding="utf-8")), d
    raise BenchError(f"no task {tid} under {bench / 'tasks'}")


def all_tasks(bench: Path, split: str = "all") -> list:
    out = []
    for s in ("dev", "heldout"):
        if split not in ("all", s):
            continue
        for p in sorted((bench / "tasks" / s).glob("*/task.json")):
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def ensure_base(ctx: Ctx, task: dict, tdir: Path) -> str:
    """The task's base commit in the fork (made again from its tree, or from the truth's base.diff on main, if gc
    took it)."""
    fork = ctx.fork
    if has_object(fork, task["base_commit"], "commit"):
        return task["base_commit"]
    tree = task["base_tree"]
    if not has_object(fork, tree, "tree"):
        tree = apply_to_tree(fork, tree_of(fork, task["main"]), (ctx.truth_dir(task["id"]) / "base.diff").read_bytes(),
                             three_way=False)
        if tree != task["base_tree"]:
            raise BenchError(f"base.diff does not rebuild {task['id']}'s base on {task['main'][:10]}")
    c = commit_tree(fork, tree, "base")
    if c != task["base_commit"]:
        raise BenchError(f"{task['id']}: the base commit came out as {c[:10]}, not {task['base_commit'][:10]}")
    return c


def cmd_list(a) -> int:
    if a.split in ("heldout", "all") and a.truth and not a.heldout:
        say("list: the held-out tasks' truth is shown only with --heldout")
        return 2
    ctx = Ctx(a.fork, a.bench, truth=a.truth_store) if a.truth else None
    for t in all_tasks(Path(a.bench), a.split):
        if t["split"] == "heldout" and a.split == "all" and not a.heldout:
            say(f"{t['id']}  heldout")
            continue
        sig = Path(a.bench) / "tasks" / t["split"] / t["id"] / "signal.json"
        det, per = "-", ""
        if sig.exists():
            s = json.loads(sig.read_text(encoding="utf-8"))
            det = {True: "detected", False: "silent"}.get(s.get("detected"), "unknown")
            per = " ".join(f"{g}{'+' if v.get('detected') else '-' if v.get('detected') is False else '?'}"
                           for g, v in (s.get("gates") or {}).items())
        line = f"{t['id']}  {t['split']:7}  {det:8}  {per:34}  hidden {t.get('hidden_count', len(t.get('hidden', [])))}"
        if ctx is not None:
            tr, _ = load_truth(ctx, t["id"])
            line += (f"  gold +{tr['truth']['insertions']}/-{tr['truth']['deletions']}  {tr['fix'][:10]}  "
                     f"{', '.join(tr['truth']['sites'][:3])}  {tr['subject'][:70]}")
        say(line)
    return 0


def cmd_prepare(a) -> int:
    return prepare(Ctx(a.fork, a.bench, truth=a.truth_store), a.task, Path(a.out), a.heldout)


DENY_DEFAULT = [p for p in os.environ.get(
    "SPIRELINE_DEBUGBENCH_DENY", os.pathsep.join(["D:/spireline/src", "D:/spireline/spireline",
                                                  "D:/spireline/referee/backlog"])).split(os.pathsep) if p]


def protected_roots(ctx: Ctx) -> list:
    """What a fixer must not reach: the fork and every worktree of it (main is the answer), the bench, the truth
    store, the upstream checkout and the project's notes (DENY_DEFAULT, SPIRELINE_DEBUGBENCH_DENY)."""
    roots = [ctx.fork, ctx.bench] + ([ctx.truth] if ctx.truth else []) + [Path(p) for p in DENY_DEFAULT]
    for ln in git(ctx.fork, "worktree", "list", "--porcelain").splitlines():
        if ln.startswith("worktree "):
            roots.append(Path(ln[len("worktree "):]))
    out, seen = [], set()
    for r in roots:
        k = os.path.normcase(os.path.abspath(r))
        if k not in seen:
            seen.add(k)
            out.append(Path(os.path.abspath(r)))
    return out


def neutral(text: str, task: dict) -> str:
    """Gate output as a fixer may see it: main's commit and the bench's run labels read "reference"."""
    for k in (task["main"], task["main"][:12], task["main"][:10]):
        text = text.replace(k, "reference")
    return re.sub(r"debugbench-[\w-]+", "reference", text)


def brief_text(task: dict, tdir: Path) -> str:
    """The fixer's brief: what the gates show on its repo, in the gates' own words, and nothing else (no task id,
    no commit, no path, no word of the fix)."""
    lines = ["# Debugging task", "",
             "This repository's emulator diverges from the game. Find the cause and fix it on a branch: the root "
             "cause, and every other place that uses the same mechanism.", ""]
    sig = tdir / "signal.json"
    if not sig.exists():
        return "\n".join(lines + ["No gate result is recorded for this repository."]) + "\n"
    s = json.loads(sig.read_text(encoding="utf-8"))
    gates = s.get("gates") or {}
    lines += ["## What the gates show on this repository", ""] + [f"- {gate_line(n, gates[n])}" for n in GATES if n in gates]
    for n in ("refset", "codex"):
        if (gates.get(n) or {}).get("lines"):
            lines += ["", f"{n} gate, the first lines it fails on:", "```", *gates[n]["lines"][:15], "```"]
    if (tdir / "corpus-check.txt").exists():
        lines += ["", "corpus check:", "```", (tdir / "corpus-check.txt").read_text(encoding="utf-8")[-6000:], "```"]
    return neutral("\n".join(lines) + "\n", task)


def prepare(ctx: Ctx, tid: str, out: Path, heldout: bool, brief: Path | None = None) -> int:
    task, tdir = load_task(ctx.bench, tid)
    if task["split"] == "heldout" and not heldout:
        say(f"prepare: {task['id']} is held out; prepare it only for a held-out evaluation (--heldout)")
        return 2
    for r in protected_roots(ctx):
        if inside(out, r) or inside(r, out):
            say(f"prepare: {out} is inside {r} (or holds it): a fixer's repo stays clear of the fork, the bench and "
                "the truth store")
            return 2
    if out.exists() and any(out.iterdir()):
        say(f"prepare: {out} is not empty")
        return 2
    commit = ensure_base(ctx, task, tdir)
    out.mkdir(parents=True, exist_ok=True)
    git(out, "init", "-q")
    git(ctx.fork, "push", "-q", str(out), f"{commit}:refs/heads/base")
    git(out, "checkout", "-q", "-f", "base")
    bp = Path(brief) if brief else out.parent / f"{out.name}.brief.md"
    bp.write_text(scrub(brief_text(task, tdir)), encoding="utf-8")
    say(f"prepare: {out} (branch base, one commit); brief {bp}")
    return 0


# -------------------------------------------------------------------------------------------- signal

def regression_rows(items: list, metas: dict) -> list:
    rows = []
    for it in items:
        now = it["now"]
        fields = set()
        if it.get("plain") and now.get("field"):
            fields.add(now["field"])
        for j in it.get("steps", []):
            v = now.get("dp", {}).get(str(j))
            fields.update([v] if isinstance(v, str) else (v or []))
        for v in it.get("fields", {}).values():
            fields.update(v)
        rows.append({"id": it["id"], "seed": metas.get(it["id"], {}).get("seed"), "plain": it.get("plain"),
                     "card": now.get("card"), "first_field": now.get("field"), "steps": len(it.get("steps", [])),
                     "errors": [e[0] for e in it.get("errors", [])], "fields": sorted(fields)})
    return rows


def judge_corpus(ctx: Ctx, run_doc: dict, base: dict, out_txt: Path | None) -> dict:
    """The gate's rule (corpus.compare / corpus.check) on run_doc against main's run."""
    cmp = cp.compare(base["fights"], run_doc["fights"])
    corpus = cp.Corpus(ctx.gates.corpus)
    metas = {m["id"]: m for m in corpus.load_index()["fights"]}
    if out_txt is not None:
        bfile = out_txt.with_suffix(".baseline.json")
        # The baseline is "the reference": its commit and label would name the answer's commit to a fixer.
        bfile.write_text(json.dumps({"format": run_doc.get("format"), "commit": "reference", "run": "reference run",
                                     "judge_sha256": (base.get("emulator") or {}).get("judge_sha256"),
                                     "totals": base.get("totals", {}), "fights": base["fights"]}), encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cp.check(corpus, {**run_doc, "label": "(not kept)"}, bfile, show=25)
        bfile.unlink(missing_ok=True)
        out_txt.write_text(scrub(buf.getvalue()), encoding="utf-8")
    rows = regression_rows(cmp["regressions"], metas)
    fields = collections.Counter(f for r in rows for f in r["fields"])
    return {"regressed": len(rows), "improved": len(cmp["improvements"]), "missing": len(cmp["missing"]),
            "new": len(cmp["new"]), "fields": dict(fields.most_common(12)), "fights": rows[:40],
            "regressed_ids": sorted(r["id"] for r in rows), "problems": len(run_doc.get("problems") or []),
            "totals": run_doc.get("totals"), "main_totals": base.get("totals")}


def judge_invariants(main_inv: dict, inv: dict) -> dict:
    if "error" in inv or "error" in main_inv:
        return {"error": inv.get("error") or main_inv.get("error")}
    worse = {}
    for part in ("emulator", "live"):
        for k, v in inv.get(part, {}).items():
            if v > main_inv.get(part, {}).get(k, 0):
                worse[f"{part}:{k}"] = [main_inv.get(part, {}).get(k, 0), v]
    if inv.get("replay_errors", 0) > main_inv.get("replay_errors", 0):
        worse["live:replay_errors"] = [main_inv.get("replay_errors", 0), inv["replay_errors"]]
    for k, v in (inv.get("notes") or {}).items():
        if v > (main_inv.get("notes") or {}).get(k, 0):
            worse[f"note:{k}"] = [(main_inv.get("notes") or {}).get(k, 0), v]
    return {"detected": bool(worse), "worse": worse, "examples": inv.get("examples", {})}


def run_gates(ctx: Ctx, tree: Path, dll: Path, gates: tuple, label: str, work: Path, check_txt: Path | None,
              build_s: float) -> dict:
    """The named gates on a built tree, each judged against main's own result (the gate's rule): name -> result,
    with run_s and seconds_to_detect (build + gate). refset and codex share one refset_score run."""
    out = {}
    if "corpus" in gates:
        base = main_corpus(ctx)
        t1 = time.time()
        try:
            with ctx.lock:  # one corpus run at a time: its wall time is the gate's
                doc = ctx.gates.corpus_run(tree, dll, f"{label}-corpus", work)
        except CorpusRunFailed as ex:
            # Main's run went through, so the tree takes the gate down: the fixer sees the check fail.
            g = {"detected": True, "crashed": str(ex), "regressed": 0, "regressed_ids": [], "fields": {}, "fights": [],
                 "problems": 1, "improved": 0}
        else:
            g = judge_corpus(ctx, doc, base, check_txt)
            g["detected"] = bool(g["regressed"]) or g["problems"] > 0
        g["run_s"] = round(time.time() - t1, 1)
        out["corpus"] = g
    if "invariants" in gates:
        t2 = time.time()
        g = judge_invariants(main_invariants(ctx), ctx.gates.invariants(tree, dll, work))
        g["run_s"] = round(time.time() - t2, 1)
        out["invariants"] = g
    want = {g for g in ("refset", "codex") if g in gates}
    if want:
        main_score = main_refset(ctx, "codex" in want)
        t3 = time.time()
        with ctx.lock:
            gate = ctx.gates.refset(tree, label, main_score.parent.parent.parent, main_score, "codex" in want,
                                    work / "refset")
        run_s = round(time.time() - t3, 1)
        for name, g in judge_refset(gate, want).items():
            g["run_s"] = run_s  # one run for both: its wall time is each one's
            out[name] = g
    for g in out.values():
        if "detected" in g:
            g["seconds_to_detect"] = round(build_s + g["run_s"], 1) if g["detected"] else None
    return out


def finish_signal(res: dict) -> dict:
    gates = res["gates"]
    res["detected"] = detected_by(res, list(gates))  # silent only when every gate run judged and none sees it
    det =[g.get("seconds_to_detect") for g in gates.values() if g.get("seconds_to_detect")]
    res["seconds_to_detect"] = min(det) if det else None
    res["lines"] = [gate_line(n, gates[n]) for n in GATES if n in gates]
    return res


def signal_one(ctx: Ctx, task: dict, tdir: Path, wt_path: Path, gates: tuple) -> dict:
    """The gates on the task's base, merged into its signal: a gate not run now keeps its last result."""
    commit = ensure_base(ctx, task, tdir)
    sig = tdir / "signal.json"
    prev = json.loads(sig.read_text(encoding="utf-8")) if sig.exists() else {}
    kept = {k: v for k, v in (prev.get("gates") or {}).items() if k not in gates}
    work = ctx.bench / "work" / task["id"] / "signal"
    for g in gates:
        shutil.rmtree(work / g, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    wt = worktree(ctx.fork, wt_path, commit)
    t0 = time.time()
    dll = ctx.tc.emudump(wt, work / "emudump.log")
    build_s = round(time.time() - t0, 1)
    res = {"task": task["id"], "main": ctx.main, "base_commit": commit, "build_s": build_s, "gates": kept}
    if dll is None:
        res.update(detected=True, lines=["EmuDump does not build"], error="EmuDump does not build on the base")
        sig.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return res
    res["gates"].update(run_gates(ctx, wt, dll, gates, f"debugbench-{task['id']}-signal", work,
                                  tdir / "corpus-check.txt", build_s))
    res["gates"] = {k: res["gates"][k] for k in GATES if k in res["gates"]}
    finish_signal(res)
    sig.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def parse_gates(text: str) -> tuple:
    names = tuple(dict.fromkeys(x.strip() for x in text.split(",") if x.strip()))
    bad = [n for n in names if n not in GATES]
    if bad or not names:
        raise BenchError(f"--gates: {', '.join(bad) or 'none named'} (gates: {', '.join(GATES)})")
    return tuple(n for n in GATES if n in names)


def cmd_signal(a) -> int:
    ctx = Ctx(a.fork, a.bench, gates=Gates(Path(a.corpus), Path(a.jobs_dir), a.procs), truth=a.truth_store)
    gates = parse_gates(a.gates)
    tasks = all_tasks(ctx.bench) if a.all else [load_task(ctx.bench, t)[0] for t in a.tasks]
    if a.missing:
        def has(t):
            p = ctx.tasks_dir / t["split"] / t["id"] / "signal.json"
            return p.exists() and all(g in json.loads(p.read_text(encoding="utf-8")).get("gates", {}) for g in gates)
        tasks = [t for t in tasks if not has(t)]
    with bench_lock(ctx.bench, "signal"):
        return run_signals(ctx, tasks, a.jobs, gates)


def run_signals(ctx: Ctx, tasks: list, jobs: int, gates: tuple = LEGACY_GATES) -> int:
    mains = {t["main"] for t in tasks}
    if len(mains) > 1:
        raise BenchError(f"the tasks were harvested on {len(mains)} mains; signal one main's tasks at a time")
    if mains:
        ctx.main = mains.pop()
        ctx.main_tree = tree_of(ctx.fork, ctx.main)
    if "corpus" in gates:
        main_corpus(ctx)
    if "invariants" in gates:
        main_invariants(ctx)
    if "refset" in gates or "codex" in gates:
        main_refset(ctx, "codex" in gates)
    slots = queue.Queue()
    for k in range(jobs):
        slots.put(ctx.slot(f"s{k}"))

    def one(t: dict) -> dict:
        wt = slots.get()
        t0 = time.time()
        try:
            r = signal_one(ctx, t, ctx.tasks_dir / t["split"] / t["id"], wt, gates)
        except BenchError as ex:
            r = {"task": t["id"], "error": str(ex)[:300], "detected": None, "lines": []}
        finally:
            slots.put(wt)
        state = {True: "DETECTED", False: "silent"}.get(r.get("detected"), "error")
        now = ", ".join(f"{g} {'yes' if r['gates'][g].get('detected') else 'no' if r['gates'][g].get('detected') is False else 'error'}"
                        for g in gates if g in r.get("gates", {}))
        say(f"  {t['id']} {state} ({now}; {round(time.time() - t0)} s)  {r.get('error', '')}"[:300])
        return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        res = list(ex.map(one, tasks))
    park(ctx, [ctx.slot(f"s{k}") for k in range(jobs)])
    n = sum(1 for r in res if r.get("detected"))
    say(f"signal: {n} of {len(res)} tasks detected by any gate ({', '.join(gates)} run now)")
    return 0


# --------------------------------------------------------------------------------------------- score

SUSPECT = re.compile(r"([\w./\\-]+\.cs)(?:\s*(?:::|#)\s*([A-Za-z_][\w.<>]*))?")


def parse_report(path: Path | None) -> dict:
    if path is None:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    try:
        doc = json.loads(text)
        if isinstance(doc, dict):
            return doc
    except ValueError:
        pass
    sec, grab = [], False
    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            grab = "suspect" in ln.lower()
            continue
        if grab:
            sec.append(ln)
    body = "\n".join(sec) if sec else text
    sus = []
    for m in SUSPECT.finditer(body):
        s = m.group(1).replace("\\", "/") + (f"::{m.group(2)}" if m.group(2) else "")
        if s not in sus:
            sus.append(s)
    return {"suspects": sus}


def _file_match(sus_file: str, truth_file: str) -> bool:
    s = sus_file.replace("\\", "/").lstrip("./")
    return s == truth_file or truth_file.endswith("/" + s)


def _member_match(named: str, site: str) -> bool:
    """A suspect's member against a truth site's (Type.Member, Member or Type.Member[case X] on either side)."""
    a, b = member_of(named), member_of(site)
    return a == b or b.endswith("." + a) or a.endswith("." + b)


def localise(suspects: list, truth: dict) -> dict:
    """Top-1 / top-3 of the suspects (in order) against the truth's files, and its sites where a suspect names a member."""
    files, sites = [], []
    for s in suspects:
        f, _, member = s.partition("::")
        hit_f = any(_file_match(f, tf) for tf in truth["files"])
        if f not in [x[0] for x in files]:
            files.append((f, hit_f))
        if member:
            sites.append(any(_file_match(f, ts.split("::", 1)[0]) and _member_match(member, ts.split("::", 1)[1])
                             for ts in truth["sites"]))
    return {"suspects": len(suspects),
            "top1_file": bool(files) and files[0][1], "top3_file": any(h for _, h in files[:3]),
            "top1_site": (sites[0] if sites else None), "top3_site": (any(sites[:3]) if sites else None)}


def candidate_tree(ctx: Ctx, repo: Path, commit: str, slot: Path) -> Path:
    """The candidate's commit checked out in a scratch repo of the bench (the candidate's repo gets no worktree)."""
    if not (slot / ".git").exists():
        slot.mkdir(parents=True, exist_ok=True)
        git(slot, "init", "-q")
    git(repo, "push", "-q", "--force", str(slot), f"{commit}:refs/heads/candidate")
    git(slot, "checkout", "-q", "--force", "--detach", commit)
    git(slot, "clean", "-fdq")
    return slot


def score_one(ctx: Ctx, task: dict, tdir: Path, repo: Path, ref: str, run: str, report: dict, quick: bool,
              slot: Path, gates: tuple = ("corpus",), extra: dict | None = None) -> dict:
    """A candidate judged by behaviour: main's hidden and collateral tests, main's suite and the gates against
    main; the share of the planted mechanism it repairs (sibling_credit). Never by its diff against the gold patch
    (that is reported apart, as differs_from_gold)."""
    truth, trdir = load_truth(ctx, task["id"])
    commit = git(repo, "rev-parse", f"{ref}^{{commit}}")
    base_tree = task["base_tree"]
    if not has_object(repo, base_tree, "tree"):
        raise BenchError(f"{repo} does not hold {task['id']}'s base (tree {base_tree[:10]}): fix on a repo made by "
                         "`debugbench prepare`")
    diff = changed(repo, base_tree, commit)
    eng = [f for _, f in diff if f.startswith(ENGINE)]
    cand_sites = set(sites_between(repo, base_tree, commit, eng))
    gold = truth["truth"]
    res = {"task": task["id"], "split": task["split"], "run": run, "candidate": commit, "main": task["main"],
           "changed_files": [f for _, f in diff], "engine_files": eng, "sites": sorted(cand_sites),
           "differs_from_gold": {"missed_sites": sorted(set(gold["sites"]) - cand_sites),
                                 "missed_files": sorted(set(gold["files"]) - set(eng)),
                                 "extra_sites": sorted(cand_sites - set(gold["sites"]))},
           "outside_engine": [f for _, f in diff if not f.startswith(ENGINE) and not proj_of(f)], **(extra or {})}
    work = ctx.need_truth() / "work" / task["id"] / f"score-{run}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    tree = candidate_tree(ctx, repo, commit, slot)
    # The base's judge, whatever the branch did to it.
    judge = {f: blob(ctx.fork, task["main"], f) for s, f in diff if f.startswith(JUDGE_PATHS)}
    overlay(tree, judge)
    mech = truth.get("mechanism") or {"units": [{"site": s} for s in gold["sites"]], "hooks": []}
    res["siblings"] = sibling_credit(mech, cand_sites, hook_rows_at(tree / ENGINE) if mech.get("hooks") else None)
    dll = ctx.tc.emudump(tree, work / "emudump.log")
    if dll is None:
        res.update(builds=False, hidden_pass=False, collateral_pass=False, gates={})
        return finish(ctx, res, report, run, truth)
    res["builds"] = True
    g = run_gates(ctx, tree, dll, gates, f"debugbench-{task['id']}-{run}", work, work / "corpus-check.txt", 0)
    res["gates"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("fights", "lines")} for k, v in g.items()}
    if "corpus" in g:
        c = g["corpus"]
        sig = tdir / "signal.json"
        was = set(((json.loads(sig.read_text(encoding="utf-8")).get("gates") or {}).get("corpus") or {})
                  .get("regressed_ids", [])) if sig.exists() else set()
        c["restored"] = len(was - set(c.get("regressed_ids", []))) if not c.get("crashed") else 0
        c["base_regressed"] = len(was)
        res["corpus"] = {k: v for k, v in c.items() if k != "fights"} | {"fights": c.get("fights", [])[:10]}
    # Hidden and collateral tests: main's test files over the candidate's.
    hidden_files = {str(p.relative_to(trdir / "hidden")).replace("\\", "/"): p.read_bytes()
                    for p in (trdir / "hidden").rglob("*") if p.is_file()}
    overlay(tree, hidden_files)
    want = set(truth["hidden"]) | set(truth["collateral"])
    filters = collections.defaultdict(set)
    for f, data in hidden_files.items():
        filters[proj_of(f)].update(class_filters(data.decode("utf-8", "replace")))
    try:
        got = _outcomes(ctx, tree, sorted(filters), {k: sorted(v) for k, v in filters.items()}, work / "trx",
                        work / "tests.log")
    except BenchError as ex:
        got, res["tests_error"] = None, str(ex)
    if got is None:
        res.update(hidden_pass=False, collateral_pass=False, hidden_builds=False)
    else:
        res["hidden_builds"] = True
        res["hidden"] = {t: got.get(t, "missing") for t in truth["hidden"]}
        res["collateral"] = {t: got.get(t, "missing") for t in truth["collateral"]}
        res["hidden_pass"] = all(v == "pass" for v in res["hidden"].values())
        res["collateral_pass"] = all(v == "pass" for v in res["collateral"].values())
    if not quick and got is not None:
        full = _outcomes(ctx, tree, projects_in(tree), None, work / "trx", work / "tests.log")
        res["unit_regressions"] = sorted(_broken(main_tests_for(ctx, task), full or {}) - want) if full is not None else None
    return finish(ctx, res, report, run, truth)


def gate_worse(name: str, g: dict) -> bool:
    """The candidate is worse than main on this gate, or the gate could not tell (no worse is not shown)."""
    if name == "corpus":
        return g.get("regressed") is None or g.get("regressed") > 0 or (g.get("problems") or 0) > 0
    return g.get("detected") is not False


def gate_better(name: str, g: dict) -> bool:
    return (g.get("improved") or 0) > 0 if name == "corpus" else (g.get("better") or 0) > 0


def outcome_of(res: dict) -> str:
    """fixed: main's hidden, collateral and other tests pass, no gate is worse than main, and the whole mechanism
    is covered. partial: the hidden tests pass with no gate worse, but a sibling is left (a collateral test fails,
    or sites, callers or hook dispatch points are not covered). disputed: no gate worse and one better than main
    while a test of main fails: a fix better than main, against a test that pins main's own behaviour, for a
    reviewer to settle (never counted wrong). wrong: anything else."""
    gates = res.get("gates") or {}
    res["gates_no_worse"] = bool(gates) and not any(gate_worse(k, v) for k, v in gates.items())
    res["beats_main"] = res["gates_no_worse"] and any(gate_better(k, v) for k, v in gates.items())
    if not res.get("builds"):
        return "wrong"
    credit = (res.get("siblings") or {}).get("credit")
    units_ok = credit is None or credit >= 0.999
    clean = res.get("hidden_pass") and not res.get("unit_regressions") and res["gates_no_worse"]
    if clean and res.get("collateral_pass"):
        return "fixed" if units_ok else "partial"
    if clean:
        return "partial"
    if res["beats_main"]:
        return "disputed"
    return "wrong"


def main_tests_for(ctx: Ctx, task: dict) -> dict:
    p = ctx.bench / "main" / task["main"][:12] / "tests.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return main_tests(ctx)


def finish(ctx: Ctx, res: dict, report: dict, run: str, truth: dict) -> dict:
    res["localisation"] = localise(report.get("suspects", []), truth["truth"])
    res["report_suspects"] = report.get("suspects", [])[:10]
    for k in ("hours", "tokens", "verdict", "round"):
        res[k] = report.get(k)
    res["outcome"] = outcome_of(res)
    res["correct"] = res["outcome"] == "fixed"
    res["scored"] = datetime.datetime.now().isoformat(timespec="seconds")
    out = ctx.need_truth() / "runs" / run / f"{res['task']}.json"  # scores name the truth: the scorer's side
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def cmd_score(a) -> int:
    ctx = Ctx(a.fork, a.bench, gates=Gates(Path(a.corpus), Path(a.jobs_dir), a.procs), truth=a.truth_store)
    task, tdir = load_task(ctx.bench, a.task)
    ctx.main, ctx.main_tree = task["main"], tree_of(ctx.fork, task["main"])
    report = parse_report(Path(a.report) if a.report else None)
    for k in ("hours", "tokens", "verdict", "round"):
        if getattr(a, k) is not None:
            report[k] = getattr(a, k)
    with bench_lock(ctx.bench, "score"):
        res = score_one(ctx, task, tdir, Path(a.repo), a.ref, a.run, report, a.quick, ctx.slot("score"),
                        parse_gates(a.gates), {"cost_source": "reported"})
    say(score_line(res))
    return 0 if res["correct"] else 1


def score_line(res: dict) -> str:
    c = res.get("corpus") or {}
    loc, sib = res["localisation"], res.get("siblings") or {}
    return (f"score {res['task']} ({res['run']}): {res['outcome'].upper()}"
            f"{' (beats main)' if res.get('beats_main') else ''}: hidden {'pass' if res.get('hidden_pass') else 'FAIL'}, "
            f"collateral {'pass' if res.get('collateral_pass') else 'FAIL'}, corpus {c.get('regressed', '?')} worse than "
            f"main ({c.get('restored', '?')} of {c.get('base_regressed', '?')} restored, {c.get('improved', '?')} better), "
            f"unit regressions {len(res.get('unit_regressions') or [])}, mechanism {sib.get('complete', '?')} of "
            f"{sib.get('units', '?')} places (credit {sib.get('credit')}), localisation top-1 {loc['top1_file']} top-3 "
            f"{loc['top3_file']}")


# ------------------------------------------------------------------------------------------- harness

PROMPT = """You are debugging the repository at {repo}. It holds a game emulator (src/Sts2Emulator) that diverges from
the game; the brief below is what its gates show.

{brief}
Work in {repo} only. Fix the bug on a git branch named `fix` and commit it there. When you are done, write {report}:
JSON {{"suspects": ["src/Sts2Emulator/<file>.cs::<Type.Member>", ...]}}, the places you believe hold the bug, most
likely first.
"""


def fixer_env() -> dict:
    """The fixer's environment: the bench's own variables taken out (they would name the truth store)."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith(("SPIRELINE_DEBUGBENCH", "SPIRELINE_FORK"))}


def _rule_path(p: Path) -> str:
    s = Path(os.path.abspath(p)).as_posix()
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    return f"//{m.group(1).lower()}/{m.group(2)}" if m else "/" + s.lstrip("/")


def deny_settings(roots: list) -> dict:
    """Claude Code settings for the fixer: no reading or editing below a protected root."""
    return {"permissions": {"deny": [f"{tool}({_rule_path(r)}/**)" for r in roots for tool in ("Read", "Edit")]}}


def _json_docs(text: str) -> list:
    docs = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            with contextlib.suppress(ValueError):
                docs.append(json.loads(ln))
    if not docs:
        with contextlib.suppress(ValueError):
            d = json.loads(text)
            docs = d if isinstance(d, list) else [d]
    return [d for d in docs if isinstance(d, dict)]


USAGE = (("input", "input_tokens"), ("output", "output_tokens"), ("cache_creation", "cache_creation_input_tokens"),
         ("cache_read", "cache_read_input_tokens"))


def parse_usage(text: str) -> dict:
    """Token use from the fixer's output (Claude Code's --output-format json or stream-json): the final result
    events' usage, else every assistant message's once (stream-json repeats a message per content block). None
    when the output holds no usage: unknown is never 0."""
    docs = _json_docs(text)
    results = [d for d in docs if d.get("type") == "result" and isinstance(d.get("usage"), dict)]
    if results:
        usages, source = [d["usage"] for d in results], "result"
    else:
        by_id = {}
        for d in docs:
            msg = d.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("usage"), dict):
                by_id[msg.get("id") or len(by_id)] = msg["usage"]
        usages, source = list(by_id.values()), "messages"
    if not usages:
        return {"source": "none", "total": None}
    out = {k: sum(int(u.get(f) or 0) for u in usages) for k, f in USAGE}
    out["total"] = sum(out[k] for k, _ in USAGE)
    out["source"] = source
    cost = [d.get("total_cost_usd") for d in results if isinstance(d.get("total_cost_usd"), (int, float))]
    if cost:
        out["cost_usd"] = round(sum(cost), 4)
    return out


def _forms(p: Path) -> list:
    s = Path(os.path.abspath(p)).as_posix().lower()
    m = re.match(r"^([a-z]):/(.*)$", s)
    return [s] + ([f"/{m.group(1)}/{m.group(2)}", f"/mnt/{m.group(1)}/{m.group(2)}"] if m else [])


def audit(text: str, roots: list, allow: list) -> list:
    """What in the fixer's own actions (its tool calls; the whole output when it has none) names a protected root
    or the bench's tool: each a reason the attempt is contaminated (it may have read the answer)."""
    docs = _json_docs(text)
    calls = [json.dumps(c.get("input")) for d in docs if isinstance(d.get("message"), dict)
             for c in (d["message"].get("content") if isinstance(d["message"].get("content"), list) else [])
             if isinstance(c, dict) and c.get("type") == "tool_use"]
    hay = "\n".join(calls) if docs else text
    norm = hay.replace("\\\\", "/").replace("\\", "/").lower()
    for a in allow:
        for f in _forms(a):
            norm = norm.replace(f, "<allowed>")
    out = []
    for r in roots:
        if any(re.search(re.escape(f) + r"(?![\w.-])", norm) for f in _forms(r)):
            out.append(f"named {Path(r).as_posix()}")
    if re.search(r"debugbench", norm):
        out.append("used the bench's own tool (debugbench)")
    return out


def _launch(args: list) -> list:
    exe = shutil.which(args[0]) or args[0]
    return (["cmd", "/c", exe] if exe.lower().endswith((".cmd", ".bat")) else [exe]) + args[1:]


def run_fixer(ctx: Ctx, tid: str, run: str, fixer: str, sandbox_root: Path, timeout: float, heldout: bool,
              silent_ok: bool, gates: tuple, quick: bool) -> dict:
    """One fixer attempt. Setup (the sandbox, its repo and brief, the deny profile) and scoring are timed apart;
    the fixer's wall time is its process alone, its tokens are read from its own output, never from its word."""
    t_setup = CLOCK()
    task, tdir = load_task(ctx.bench, tid)
    ctx.main, ctx.main_tree = task["main"], tree_of(ctx.fork, task["main"])
    ctx.need_truth()
    sig = tdir / "signal.json"
    detected = json.loads(sig.read_text(encoding="utf-8")).get("detected") if sig.exists() else None
    if detected is not True and not silent_ok:
        raise BenchError(f"{tid}: no gate detects its bug; fixers run on detected tasks (--silent-ok: a blind "
                         "attempt, scored apart from the fix rate)")
    sb = Path(sandbox_root) / run / secrets.token_hex(4)
    repo, brief, report_p = sb / "repo", sb / "BRIEF.md", sb / "REPORT.json"
    if prepare(ctx, tid, repo, heldout, brief=brief) != 0:
        raise BenchError(f"{tid}: prepare refused {repo}")
    hw = ctx.need_truth() / "work" / tid / f"run-{run}"
    shutil.rmtree(hw, ignore_errors=True)
    hw.mkdir(parents=True)
    roots = protected_roots(ctx)
    settings = hw / "settings.json"
    settings.write_text(json.dumps(deny_settings(roots), indent=1), encoding="utf-8")
    fmt = {"sandbox": sb.as_posix(), "repo": repo.as_posix(), "brief": brief.as_posix(), "report": report_p.as_posix(),
           "settings": settings.as_posix()}
    prompt = PROMPT.format(repo=fmt["repo"], brief=brief.read_text(encoding="utf-8"), report=fmt["report"])
    args = _launch([x.format(**fmt) for x in shlex.split(fixer)])
    setup_s = CLOCK() - t_setup
    with (hw / "transcript.jsonl").open("wb") as out, (hw / "stderr.log").open("wb") as err:
        t0 = CLOCK()
        p = subprocess.Popen(args, cwd=str(sb), env=fixer_env(), stdin=subprocess.PIPE, stdout=out, stderr=err,
                             creationflags=BELOW_NORMAL)
        timed_out = False
        try:
            p.communicate(prompt.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, check=False)
            p.kill()
            p.communicate()
        fixer_s = CLOCK() - t0
    text = (hw / "transcript.jsonl").read_text(encoding="utf-8", errors="replace")
    usage = parse_usage(text)
    harness = {"setup_s": round(setup_s, 2), "fixer_s": round(fixer_s, 2), "timed_out": timed_out,
               "exit": p.returncode, "tokens": usage, "attempt": sb.relative_to(Path(sandbox_root)).as_posix(),
               "blind": detected is not True}
    contaminated = [scrub(x) for x in audit(text, roots, [Path(sys.prefix)])]
    report = {}
    for name in ("REPORT.json", "REPORT.md"):
        if (sb / name).exists():
            report = parse_report(sb / name)
            break
    harness["reported"] = {k: report.pop(k, None) for k in ("hours", "tokens")}
    report.update(hours=round(fixer_s / 3600, 4), tokens=usage.get("total"))
    extra = {"harness": harness, "cost_source": "harness"} | ({"contaminated": contaminated} if contaminated else {})
    ref = "fix" if git_ok(repo, "rev-parse", "--verify", "-q", "refs/heads/fix") else "HEAD"
    t2 = CLOCK()
    if git(repo, "rev-parse", f"{ref}^{{tree}}") == task["base_tree"]:
        truth, _ = load_truth(ctx, tid)
        res = finish(ctx, {"task": tid, "split": task["split"], "run": run, "main": task["main"], "builds": False,
                           "no_candidate": True, **extra}, report, run, truth)
    else:
        with bench_lock(ctx.bench, "score"):
            res = score_one(ctx, task, tdir, repo, ref, run, report, quick, ctx.slot("score"), gates, extra)
    res["harness"]["score_s"] = round(CLOCK() - t2, 1)
    (ctx.need_truth() / "runs" / run / f"{tid}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def cmd_run(a) -> int:
    ctx = Ctx(a.fork, a.bench, gates=Gates(Path(a.corpus), Path(a.jobs_dir), a.procs), truth=a.truth_store)
    res = run_fixer(ctx, a.task, a.run, a.fixer, Path(a.sandbox), a.timeout, a.heldout, a.silent_ok,
                    parse_gates(a.gates), a.quick)
    h = res["harness"]
    say(("NO CANDIDATE " if res.get("no_candidate") else score_line(res) + "; ")
        + f"fixer {h['fixer_s']:.0f} s, tokens {h['tokens'].get('total')} ({h['tokens']['source']}), setup "
          f"{h['setup_s']:.0f} s, score {h['score_s']:.0f} s"
        + (f"; CONTAMINATED: {'; '.join(res['contaminated'])}" if res.get("contaminated") else ""))
    return 0 if res.get("correct") else 1


# -------------------------------------------------------------------------------------------- report

def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}% ({n}/{d})" if d else "-"


def detected_by(s: dict, gates) -> bool | None:
    """True when one of these gates detects the bug, False when all of them ran and none does, else None."""
    vals = [((s.get("gates") or {}).get(g) or {}).get("detected") for g in gates]
    if any(v is True for v in vals):
        return True
    return False if vals and all(v is False for v in vals) else None


def detection_lines(sigs: list) -> list:
    """Per gate: detected of judged (errors apart); the first run's gates against the two added ones, and how many
    of the tasks the first ones miss the added ones catch."""
    lines = [f"- detected by any gate: {pct(sum(1 for s in sigs if s.get('detected')), len(sigs))}"]
    for g in GATES:
        vals = [((s.get("gates") or {}).get(g) or {}) for s in sigs]
        ran = [v for v in vals if v]
        if not ran:
            continue
        judged = [v for v in ran if v.get("detected") is not None]
        lines.append(f"  - {g}: {pct(sum(1 for v in judged if v['detected']), len(judged))}"
                     + (f"; not judged {len(ran) - len(judged)}" if len(ran) > len(judged) else ""))
    added = [g for g in GATES if g not in LEGACY_GATES and any(g in (s.get("gates") or {}) for s in sigs)]
    if added:
        missed = [s for s in sigs if detected_by(s, LEGACY_GATES) is False]
        lines.append(f"- the first run's gates ({', '.join(LEGACY_GATES)}) miss {len(missed)}; of those the added gates catch:")
        for g in added:
            lines.append(f"  - {g}: {sum(1 for s in missed if detected_by(s, (g,)))}")
        lines.append(f"  - any of {', '.join(added)}: {sum(1 for s in missed if detected_by(s, added))}"
                     f"; still silent {sum(1 for s in missed if detected_by(s, added) is False)}"
                     f", not judged {sum(1 for s in missed if detected_by(s, added) is None)}")
    return lines


def task_set_report(bench: Path, split: str, ctx: Ctx | None = None) -> list:
    tasks = all_tasks(bench, split)
    lines = [f"# debugbench task set ({split})", ""]
    by = collections.Counter(t["split"] for t in tasks)
    line = f"- tasks: {len(tasks)} (dev {by['dev']}, held out {by['heldout']})"
    if ctx is not None and ctx.truth is not None:
        ours = sum(bool(load_truth(ctx, t["id"])[0].get("ours")) for t in tasks)
        line += f"; ours {ours}, upstream {len(tasks) - ours}"
    lines.append(line)
    sigs = []
    for t in tasks:
        p = bench / "tasks" / t["split"] / t["id"] / "signal.json"
        if p.exists():
            sigs.append((t, json.loads(p.read_text(encoding="utf-8"))))
    lines.append(f"- signal run on {len(sigs)}")
    lines += detection_lines([s for _, s in sigs])
    secs = [s["seconds_to_detect"] for _, s in sigs if s.get("seconds_to_detect")]
    if secs:
        lines.append(f"- seconds to detect (build + fastest detecting gate): median {statistics.median(secs):.0f}, "
                     f"max {max(secs):.0f}")
    fields = collections.Counter(f for _, s in sigs for f in (s.get("gates", {}).get("corpus", {}).get("fields") or {}))
    if fields:
        lines.append("- fields the corpus check names most: " + ", ".join(f"{k} ({v})" for k, v in fields.most_common(8)))
    stops = collections.Counter()
    for _, s in sigs:
        stops.update((s.get("gates", {}).get("codex") or {}).get("stops") or {})
    if stops:
        lines.append("- codex stops the bases add most: " + ", ".join(f"{k} ({v})" for k, v in stops.most_common(8)))
    silent = [t["id"] for t, s in sigs if s.get("detected") is False]
    if silent:
        lines.append(f"- silent (no gate sees the bug): {len(silent)}: {', '.join(silent)}")
    return lines


def run_report(ctx: Ctx, run: str, split: str) -> list:
    """A run's numbers. Detection first; the fix rate is of the detected tasks only (a silent task's attempt is
    blind and reported apart); a contaminated attempt (the fixer reached a protected root) counts nowhere; cost is
    the harness's measurement only."""
    rows = []
    for p in sorted((ctx.need_truth() / "runs" / run).glob("db-*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if split in ("all", r.get("split")):
            rows.append(r)
    lines = [f"# debugbench run {run} ({split}): {len(rows)} tasks scored", ""]
    if not rows:
        return lines + ["no scored tasks"]

    def detected(r):
        p = ctx.tasks_dir / r["split"] / r["task"] / "signal.json"
        return json.loads(p.read_text(encoding="utf-8")).get("detected") if p.exists() else None

    def outcome(r):
        o = r.get("outcome") or ("fixed" if r.get("correct") else "wrong")
        return "fixed" if o == "disputed" and str(r.get("verdict") or "").lower().startswith("accept") else o
    dirty = [r for r in rows if r.get("contaminated")]
    clean = [r for r in rows if not r.get("contaminated")]
    det = [r for r in clean if detected(r) is True]
    blind = [r for r in clean if detected(r) is not True]
    oc = collections.Counter(outcome(r) for r in det)
    credit = [((r.get("siblings") or {}).get("credit")) for r in det]
    credit = [c if c is not None else (1.0 if outcome(r) == "fixed" else 0.0) for c, r in zip(credit, det)]
    lines += [f"- detected by a gate: {len(det)} scored; silent (no gate sees the bug, scored apart, never in the fix "
              f"rate): {len(blind)}; contaminated (the fixer reached a protected root, counted nowhere): {len(dirty)}"]
    if det:
        verdicts = [r for r in det if r.get("verdict")]
        first = [r for r in verdicts if str(r["verdict"]).lower().startswith("accept") and int(r.get("round") or 1) == 1]
        with_sus = [r for r in det if (r.get("localisation") or {}).get("suspects")]
        site_rows = [r for r in det if (r.get("localisation") or {}).get("top1_site") is not None]
        regs = sum((r.get("corpus") or {}).get("regressed") or 0 for r in det)
        unit = sum(len(r.get("unit_regressions") or []) for r in det)
        lines += [f"- fixed given detection: {pct(oc['fixed'], len(det))}; partial {oc['partial']}, disputed "
                  f"{oc['disputed']} (better than main on the gates, a main test fails: for review), wrong {oc['wrong']}",
                  f"- mechanism credit given detection (the share of sites, callers and hook dispatch points repaired): "
                  f"mean {statistics.mean(credit):.2f}",
                  f"- beats main (no gate worse, one better): {sum(1 for r in det if r.get('beats_main'))}",
                  f"- localisation, file: top-1 {pct(sum(bool((r.get('localisation') or {}).get('top1_file')) for r in det), len(det))}, "
                  f"top-3 {pct(sum(bool((r.get('localisation') or {}).get('top3_file')) for r in det), len(det))} "
                  f"(with a report: {len(with_sus)})",
                  f"- localisation, site: top-1 {pct(sum(bool(r['localisation']['top1_site']) for r in site_rows), len(site_rows))}, "
                  f"top-3 {pct(sum(bool(r['localisation']['top3_site']) for r in site_rows), len(site_rows))}",
                  f"- first-round acceptance: {pct(len(first), len(verdicts))}",
                  f"- regressions added: {regs} corpus fights worse than main, {unit} unit tests"]
        measured = [r for r in det if r.get("cost_source") == "harness" and (r.get("harness") or {}).get("fixer_s") is not None]
        fixed_m = [r for r in measured if outcome(r) == "fixed"]
        hours = sum(r["harness"]["fixer_s"] for r in measured) / 3600
        toks = [r["harness"]["tokens"].get("total") for r in measured]
        known = [t for t in toks if t is not None]
        lines.append(f"- cost (harness: the fixer's own wall time and tokens): {len(measured)} of {len(det)} measured, "
                     f"{hours:.2f} h, " + (f"{sum(known)} tokens over {len(known)}" if known else "tokens unknown")
                     + (f"; per fixed task {hours / len(fixed_m):.2f} h" + (f", {sum(known) / len(fixed_m):.0f} tokens"
                                                                           if len(known) == len(measured) else "")
                        if fixed_m else "") + (f"; {len(det) - len(measured)} self-reported only (not counted)"
                                               if len(measured) < len(det) else ""))
    if blind:
        ob = collections.Counter(outcome(r) for r in blind)
        lines.append(f"- silent tasks attempted blind: {len(blind)}: fixed {ob['fixed']}, partial {ob['partial']}, "
                     f"wrong {ob['wrong'] + ob['disputed']} (not in the fix rate)")
    if dirty:
        lines.append(f"- contaminated: {', '.join(r['task'] + ' (' + '; '.join(r['contaminated'])[:80] + ')' for r in dirty)}")
    lines += ["", "| task | detected | outcome | credit | hidden | corpus worse | better | restored | top-1 | top-3 | fixer h | tokens | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, loc, h = r.get("corpus") or {}, r.get("localisation") or {}, r.get("harness") or {}
        lines.append(f"| {r['task']} | {'yes' if detected(r) else 'no'} | {outcome(r)}{' (contaminated)' if r.get('contaminated') else ''} | "
                     f"{(r.get('siblings') or {}).get('credit', '-')} | {'pass' if r.get('hidden_pass') else 'fail'} | "
                     f"{c.get('regressed', '-')} | {c.get('improved', '-')} | {c.get('restored', '-')}/{c.get('base_regressed', '-')} | "
                     f"{'y' if loc.get('top1_file') else 'n'} | {'y' if loc.get('top3_file') else 'n'} | "
                     f"{round(h['fixer_s'] / 3600, 2) if h.get('fixer_s') is not None else '-'} | "
                     f"{(h.get('tokens') or {}).get('total') if h else '-'} | {r.get('verdict') or '-'} {r.get('round') or ''} |")
    return lines


def cmd_report(a) -> int:
    bench = Path(a.bench)
    ctx = Ctx(a.fork, a.bench, truth=a.truth_store)
    if a.run:
        lines = run_report(ctx, a.run, a.split)
    else:
        lines = task_set_report(bench, a.split, ctx if ctx.truth else None)
    text = "\n".join(lines) + "\n"
    if a.run:
        (ctx.need_truth() / "runs" / a.run / "report.md").write_text(text, encoding="utf-8")
    else:
        (bench / "tasks" / f"report-{a.split}.md").write_text(text, encoding="utf-8")
    say(text)
    return 0


# ------------------------------------------------------------------------------------------- migrate

def migrate(ctx: Ctx) -> int:
    """Tasks made before the truth store: their truth moves out of the bench, their base loses the lines that name
    the answer (scrub_hints; the engine and the test code are unchanged, so their signals stand), their mechanism is
    worked out, and the harvest log, runs and logs that name the truth move to the store."""
    truth = ctx.need_truth()
    n, dropped, left = 0, 0, 0
    for split in ("dev", "heldout"):
        for p in sorted((ctx.tasks_dir / split).glob("*/task.json")):
            t, tdir = json.loads(p.read_text(encoding="utf-8")), p.parent
            if t.get("format") == 2:
                continue
            ctx.main, ctx.main_tree = t["main"], tree_of(ctx.fork, t["main"])
            names = hint_names(ctx, t["base_tree"], t["hidden"], t["collateral"])
            tree, hints = scrub_hints(ctx, t["base_tree"], names, [t["fix"][:7], (t.get("cherry_of") or "")[:7]])
            eng = ENGINE.rstrip("/")
            if git(ctx.fork, "rev-parse", f"{tree}:{eng}") != git(ctx.fork, "rev-parse", f"{t['base_tree']}:{eng}"):
                raise BenchError(f"{t['id']}: the scrub changed the engine")
            commit = commit_tree(ctx.fork, tree, "base")
            hidden = {str(f.relative_to(tdir / "hidden")).replace("\\", "/"): f.read_bytes()
                      for f in (tdir / "hidden").rglob("*") if f.is_file()}
            sites = sites_between(ctx.fork, tree, ctx.main_tree, [f for _, f in changed(ctx.fork, tree, ctx.main_tree, [ENGINE])])
            t.update(base_tree=tree, base_commit=commit, hints=hints, migrated=datetime.datetime.now().isoformat(timespec="seconds"),
                     mechanism=mechanism(ctx, tree, ctx.main_tree, sites))
            t["truth"]["sites"] = sites
            t["truth"]["files"] = sorted({s.split("::", 1)[0] for s in sites}) or t["truth"]["files"]
            write_task(ctx, t, {"hidden": hidden, "gold.diff": diff_patch(ctx.fork, tree, ctx.main_tree, [ENGINE]),
                                "base.diff": diff_patch(ctx.fork, ctx.main_tree, tree),
                                "fix.diff": (tdir / "fix.diff").read_bytes() if (tdir / "fix.diff").exists()
                                else diff_patch(ctx.fork, t["parent"], t["fix"])}, fresh=False)
            for name in ("gold.diff", "base.diff", "fix.diff"):
                (tdir / name).unlink(missing_ok=True)
            shutil.rmtree(tdir / "hidden", ignore_errors=True)
            sig = tdir / "signal.json"
            if sig.exists():
                s = json.loads(sig.read_text(encoding="utf-8"))
                s.update(base_commit=commit, base_note="the base's docs and test comments lost the lines naming the "
                         "answer after this signal; its engine and test code are the same")
                sig.write_text(json.dumps(s, indent=1), encoding="utf-8")
            w = ctx.bench / "work" / t["id"]
            for name in ("harvest.log", "trx"):
                if (w / name).exists():
                    (truth / "work" / t["id"]).mkdir(parents=True, exist_ok=True)
                    shutil.move(str(w / name), str(truth / "work" / t["id"] / name))
            for d in w.glob("score-*"):
                (truth / "work" / t["id"]).mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(truth / "work" / t["id"] / d.name))
            n += 1
            dropped += sum(hints["dropped"].values())
            left += bool(hints["left"])
    for sub in ("harvest", "runs"):
        src = ctx.bench / sub
        if src.exists():
            for f in sorted(src.iterdir()):
                (truth / sub).mkdir(parents=True, exist_ok=True)
                if not (truth / sub / f.name).exists():
                    shutil.move(str(f), str(truth / sub / f.name))
            with contextlib.suppress(OSError):
                src.rmdir()
    say(f"migrate: {n} tasks' truth moved to the truth store; {dropped} lines naming the answer dropped from their "
        f"bases; {left} bases keep a name in code or other files (listed in their truth)")
    return 0


def cmd_migrate(a) -> int:
    ctx = Ctx(a.fork, a.bench, truth=a.truth_store)
    with bench_lock(ctx.bench, "signal"), bench_lock(ctx.bench, "harvest"):
        return migrate(ctx)


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--fork", default=str(DEFAULT_FORK), help="the fork's repo (history, main, the base commits)")
    ap.add_argument("--bench", default=str(DEFAULT_BENCH))
    ap.add_argument("--truth-store", default=os.environ.get("SPIRELINE_DEBUGBENCH_TRUTH"),
                    help="the truth store (SPIRELINE_DEBUGBENCH_TRUTH): answers, hidden tests, harvest log, scores; "
                         "outside the bench, and never given to a fixer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--ref", default="main")
    h.add_argument("--jobs", type=int, default=3, help="candidates verified at once (one worktree each)")
    h.add_argument("--only", nargs="*")
    h.add_argument("--limit", type=int, default=0)
    h.add_argument("--max-lines", type=int, default=300, help="largest engine change (added + deleted lines) taken")
    h.add_argument("--force", action="store_true", help="redo candidates already judged for this main")
    ls = sub.add_parser("list")
    ls.add_argument("--split", choices=("dev", "heldout", "all"), default="dev")
    ls.add_argument("--truth", action="store_true", help="also the fix, its sites and subject (ground truth)")
    ls.add_argument("--heldout", action="store_true", help="allow the held-out tasks' truth")
    pr = sub.add_parser("prepare")
    pr.add_argument("task")
    pr.add_argument("--out", required=True)
    pr.add_argument("--heldout", action="store_true", help="allow a held-out task (held-out evaluation only)")
    for name in ("signal", "score", "run"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
        p.add_argument("--jobs-dir", default=str(DEFAULT_JOBS), help="the referee's jobs folder (invariants)")
        p.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    s = sub.choices["signal"]
    s.add_argument("tasks", nargs="*")
    s.add_argument("--all", action="store_true")
    s.add_argument("--missing", action="store_true", help="only tasks without a result of every --gates gate yet")
    s.add_argument("--jobs", type=int, default=2, help="tasks at once (corpus runs still go one at a time)")
    s.add_argument("--gates", default=",".join(LEGACY_GATES),
                   help=f"comma-separated, of {', '.join(GATES)} (refset and codex: one refset_score run, about 4 min "
                        "a task with the codex walk); a gate not run keeps its last result in the task's signal")
    sc = sub.choices["score"]
    sc.add_argument("task")
    sc.add_argument("--repo", required=True, help="the candidate's repo (holds the task's base)")
    sc.add_argument("--ref", required=True, help="the candidate branch or commit")
    sc.add_argument("--run", required=True, help="the run this score belongs to (runs/<run>/)")
    sc.add_argument("--report", help="the fixer's report (.json or .md)")
    sc.add_argument("--hours", type=float)
    sc.add_argument("--tokens", type=int)
    sc.add_argument("--verdict", choices=("accept", "reject"))
    sc.add_argument("--round", type=int)
    sc.add_argument("--quick", action="store_true", help="skip the full unit suites")
    sc.add_argument("--gates", default="corpus", help=f"the gates judged against main, of {', '.join(GATES)}")
    rn = sub.choices["run"]
    rn.add_argument("task")
    rn.add_argument("--run", required=True, help="the run this attempt belongs to")
    rn.add_argument("--fixer", required=True,
                    help="the fixer's command (prompt on stdin; {sandbox} {repo} {brief} {report} {settings} are "
                         "filled in, forward slashes), e.g. claude -p --output-format stream-json --verbose "
                         "--settings {settings} --permission-mode acceptEdits")
    rn.add_argument("--sandbox", required=True, help="where attempts go (<sandbox>/<run>/<random>/repo): outside the "
                                                     "fork, the bench and the truth store")
    rn.add_argument("--timeout", type=float, default=4 * 3600)
    rn.add_argument("--heldout", action="store_true", help="allow a held-out task (held-out evaluation only)")
    rn.add_argument("--silent-ok", action="store_true", help="attempt a task no gate detects (blind, scored apart)")
    rn.add_argument("--quick", action="store_true", help="skip the full unit suites at scoring")
    rn.add_argument("--gates", default="corpus", help=f"the gates scoring judges against main, of {', '.join(GATES)}")
    sub.add_parser("migrate", help="move a bench's tasks made before the truth store into it (and scrub their bases)")
    rp = sub.add_parser("report")
    rp.add_argument("--run")
    rp.add_argument("--split", choices=("dev", "heldout", "all"), default="all")
    a = ap.parse_args(argv)
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(errors="replace")  # a commit subject the console cannot encode never stops a run
    if a.cmd == "signal" and not a.tasks and not a.all:
        ap.error("signal: name tasks or --all")
    try:
        return {"harvest": cmd_harvest, "list": cmd_list, "prepare": cmd_prepare, "signal": cmd_signal,
                "score": cmd_score, "run": cmd_run, "migrate": cmd_migrate, "report": cmd_report}[a.cmd](a)
    except BenchError as ex:
        say(f"debugbench: {ex}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
