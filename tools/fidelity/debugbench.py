"""Debugging benchmark from the fork's own history, SWE-bench style: past engine fixes undone on today's main.

    python -m tools.fidelity.debugbench harvest [--ref main] [--jobs 3] [--only <sha>...] [--limit N] [--max-lines 300]
    python -m tools.fidelity.debugbench list [--split dev|heldout|all] [--truth]
    python -m tools.fidelity.debugbench prepare <task> --out <dir> [--heldout]
    python -m tools.fidelity.debugbench signal (<task>... | --all) [--jobs 2] [--no-invariants] [--refset <main score>]
    python -m tools.fidelity.debugbench score <task> --repo <candidate repo> --ref <branch> --run <name>
        [--report <fixer report .json/.md>] [--hours H] [--tokens N] [--verdict accept|reject] [--round N] [--quick]
    python -m tools.fidelity.debugbench report [--run <name>] [--split dev|heldout|all]

Run from the fork's root with the upstream venv's Python. Everything it writes is local (--bench, default
D:/spireline/referee/debugbench or SPIRELINE_DEBUGBENCH) and never committed: tasks, ground truth, signals,
scores. The main checkout is never touched: builds run in worktrees under <bench>/wt.

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
           scoring; referee/FINDINGS.md lines naming the fix commit are dropped too.
         Ground truth: the gold patch (base -> main, engine only), its files and sites (file::Type.Member
         holding a changed line), and the fix's commit message (the mechanism). A task's id is a hash of
         its fix commit, so the id names no commit; the split is by that hash too (HELDOUT_PER_MILLE), so a
         task never changes side as tasks are added. Held-out tasks are not prepared or listed with their
         truth unless --heldout.
signal   What the gates a fixer runs show on the broken base, without the truth: the corpus check (every
         recorded fight replayed, judged against main's own corpus run: corpus.compare, the gate's rule),
         the oracle-free invariants (emulator replay and live-state violations against main's), and with
         --refset the reference-set gate (refset_score --against a main score: about 3 min a task, and it
         appends to the refset score history). Per gate: detected, the fights and fields, seconds.
score    A candidate fix (a branch of a repo that holds the task's base: `prepare` makes one): the hidden
         and collateral tests pass (main's test files put back over the candidate's), the corpus check
         against main's run has 0 regressions (and how many of the base's regressed fights it restores),
         no test main passes fails (unless --quick), sibling sites (the truth's sites the candidate did
         not touch), localisation (the report's suspects against the truth's files and sites, top-1 and
         top-3), cost (hours, tokens), reviewer verdict and round. The candidate is judged with the base's
         judge (tools/fidelity, data/id_map.json put back), whatever the branch did to them.
report   Without --run: the task set (dev/held-out, how many each gate detects). With --run: detection %,
         localisation top-1/top-3, fix correctness %, first-round acceptance, missed siblings per task,
         regressions added, cost per correct fix.

A fixer's report (score --report) is JSON: {"suspects": ["src/Sts2Emulator/X.cs::Type.Member", ...],
"hours": 1.5, "tokens": 123456, "verdict": "accept", "round": 1}, or Markdown whose section headed
"Suspects" lists them in order (any *.cs path, optionally ::Member); --hours/--tokens/--verdict/--round override.
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
FORK_BASE = "b7452170ce1df91de29d3c94acb11d87c7efc708"  # FORK.md: every commit after it is ours or a reviewed pick
PY = sys.executable
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)

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
    """file::Type.Member for every member holding a line changed between a and b (C# files; others: file::(file))."""
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
            o = cs_outline(data.decode("utf-8", "replace"))
            out.update(f"{path}::{site_of(o, ln)}" for ln in lines)
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
        return dll if rc == 0 and dll.exists() else None


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

    def refset(self, tree: Path, label: str, against: str) -> dict:
        r = subprocess.run([PY, "-m", "tools.fidelity.refset_score", "--emu", str(tree), "--procs", str(self.procs),
                            "--label", label, "--against", against], capture_output=True, text=True, check=False,
                           cwd=str(tree), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        return {"exit": r.returncode, "tail": scrub(r.stdout)[-3000:]}


# ------------------------------------------------------------------------------------------- context

class Ctx:
    def __init__(self, fork: Path, bench: Path, ref: str = "main", toolchain=None, gates=None) -> None:
        self.fork, self.bench = Path(fork), Path(bench)
        self.main = git(self.fork, "rev-parse", ref)
        self.main_tree = tree_of(self.fork, self.main)
        self.tc = toolchain or Dotnet()
        self.gates = gates
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


def drop_findings(ctx: Ctx, tree: str, shas: list) -> str:
    data = blob(ctx.fork, ctx.main, FINDINGS)
    if data is None:
        return tree
    keys = [s[:7] for s in shas if s]
    lines = data.decode("utf-8").splitlines(keepends=True)
    kept = [ln for ln in lines if not any(k in ln for k in keys)]
    return tree if len(kept) == len(lines) else edit_tree(ctx.fork, tree, {FINDINGS: "".join(kept).encode("utf-8")})


def verify(ctx: Ctx, cand: dict, wt_path: Path) -> dict:
    """The expensive checks of one screened candidate (builds and test runs in its slot's worktree): the task, or a skip."""
    c, p, fork = cand["fix"], cand["parent"], ctx.fork
    tid = task_id(c)
    work = ctx.bench / "work" / tid
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
    bwh = drop_findings(ctx, cand["bwh_tree"], [c, cand.get("cherry")])
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
            base = (tree, commit)
            break
        collateral = sorted(set(collateral) | more)
    if base is None:
        return {**rec, "skip": "the base's suite does not pass once the failing tests are out (flaky?)"}
    tree, commit = base
    # 4. Ground truth: the gold patch (base -> main, engine only) and its sites.
    eng = [f for _, f in changed(fork, tree, ctx.main_tree, [ENGINE])]
    num = git(fork, "diff", "--numstat", "--no-renames", tree, ctx.main_tree, "--", ENGINE)
    ins = sum(int(ln.split("\t")[0]) for ln in num.splitlines() if ln.split("\t")[0].isdigit())
    dels = sum(int(ln.split("\t")[1]) for ln in num.splitlines() if ln.split("\t")[1].isdigit())
    task = {"id": tid, "split": split_of(tid), "main": ctx.main, "fix": c, "parent": p, "cherry_of": cand.get("cherry"),
            "subject": cand["subject"], "message": cand["message"], "date": cand["date"], "ours": cand["ours"],
            "base_tree": tree, "base_commit": commit, "hidden": hidden, "collateral": collateral,
            "fail_to_pass_at_fix": f2p, "test_files": sorted(files), "removed_files": removed,
            "truth": {"files": eng, "sites": sites_between(fork, tree, ctx.main_tree, eng),
                      "insertions": ins, "deletions": dels},
            "made": datetime.datetime.now().isoformat(timespec="seconds")}
    tdir = ctx.tasks_dir / task["split"] / tid
    if tdir.exists():
        shutil.rmtree(tdir)
    (tdir / "hidden").mkdir(parents=True)
    for f, data in files.items():
        (tdir / "hidden" / f).parent.mkdir(parents=True, exist_ok=True)
        (tdir / "hidden" / f).write_bytes(data)
    (tdir / "gold.diff").write_bytes(diff_patch(fork, tree, ctx.main_tree, [ENGINE]))
    (tdir / "base.diff").write_bytes(diff_patch(fork, ctx.main_tree, tree))
    (tdir / "fix.diff").write_bytes(diff_patch(fork, p, c))
    (tdir / "task.json").write_text(json.dumps(task, indent=1), encoding="utf-8")
    return {**rec, "task": tid, "split": task["split"], "hidden": len(hidden), "collateral": len(collateral)}


def cmd_harvest(a) -> int:
    with bench_lock(Path(a.bench), "harvest"):
        return harvest(Ctx(a.fork, a.bench, a.ref), a.ref, a.only, a.limit, a.max_lines, a.jobs, a.force)


def harvest(ctx: Ctx, ref: str, only: list | None, limit: int, max_lines: int, jobs: int, force: bool) -> int:
    # One bench holds one main's tasks: a task's id is its fix's, so another main's harvest would overwrite them.
    other = {t["main"] for t in all_tasks(ctx.bench)} - {ctx.main}
    if other:
        raise BenchError(f"{ctx.bench} holds the tasks of main {sorted(other)[0][:10]}; harvest {ref} "
                         f"({ctx.main[:10]}) into another --bench, or --ref {sorted(other)[0][:10]} to add to them")
    state = ctx.bench / "harvest" / f"{ctx.main[:12]}.jsonl"
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
    """The task's base commit in the fork (made again from its tree, or from base.diff on main, if gc took it)."""
    fork = ctx.fork
    if has_object(fork, task["base_commit"], "commit"):
        return task["base_commit"]
    tree = task["base_tree"]
    if not has_object(fork, tree, "tree"):
        tree = apply_to_tree(fork, tree_of(fork, task["main"]), (tdir / "base.diff").read_bytes(), three_way=False)
        if tree != task["base_tree"]:
            raise BenchError(f"base.diff does not rebuild {task['id']}'s base on {task['main'][:10]}")
    c = commit_tree(fork, tree, f"debugbench base {task['id']}")
    if c != task["base_commit"]:
        raise BenchError(f"{task['id']}: the base commit came out as {c[:10]}, not {task['base_commit'][:10]}")
    return c


def cmd_list(a) -> int:
    if a.split in ("heldout", "all") and a.truth and not a.heldout:
        say("list: the held-out tasks' truth is shown only with --heldout")
        return 2
    for t in all_tasks(Path(a.bench), a.split):
        if t["split"] == "heldout" and a.split == "all" and not a.heldout:
            say(f"{t['id']}  heldout")
            continue
        sig = Path(a.bench) / "tasks" / t["split"] / t["id"] / "signal.json"
        det = "-"
        if sig.exists():
            det = "detected" if json.loads(sig.read_text(encoding="utf-8")).get("detected") else "silent"
        line = f"{t['id']}  {t['split']:7}  {det:8}  hidden {len(t['hidden'])}  gold +{t['truth']['insertions']}/-{t['truth']['deletions']}"
        if a.truth:
            line += f"  {t['fix'][:10]}  {', '.join(t['truth']['sites'][:3])}  {t['subject'][:70]}"
        say(line)
    return 0


def cmd_prepare(a) -> int:
    return prepare(Ctx(a.fork, a.bench), a.task, Path(a.out), a.heldout)


def prepare(ctx: Ctx, tid: str, out: Path, heldout: bool) -> int:
    task, tdir = load_task(ctx.bench, tid)
    if task["split"] == "heldout" and not heldout:
        say(f"prepare: {task['id']} is held out; prepare it only for a held-out evaluation (--heldout)")
        return 2
    if out.exists() and any(out.iterdir()):
        say(f"prepare: {out} is not empty")
        return 2
    commit = ensure_base(ctx, task, tdir)
    out.mkdir(parents=True, exist_ok=True)
    git(out, "init", "-q")
    git(ctx.fork, "push", "-q", str(out), f"{commit}:refs/heads/base")
    git(out, "checkout", "-q", "-f", "base")
    sig = tdir / "signal.json"
    brief = [f"# Debugging task {task['id']}", "",
             "A repo whose emulator has one engine divergence from the game that today's main does not have. "
             "Find the cause, fix it on a branch of this repo (root cause and every sibling site), and write a report.",
             "", f"Score: python -m tools.fidelity.debugbench score {task['id']} --repo {out.as_posix()} --ref <branch> "
             "--run <run name> --report <report.json>",
             'Report: {"suspects": ["src/Sts2Emulator/<file>.cs::<Type.Member>", ...], "hours": 0.0, "tokens": 0}', ""]
    if sig.exists():
        s = json.loads(sig.read_text(encoding="utf-8"))
        brief += ["## What the gates show on this repo", ""] + [f"- {ln}" for ln in s.get("lines", [])]
        if (tdir / "corpus-check.txt").exists():
            brief += ["", "```", (tdir / "corpus-check.txt").read_text(encoding="utf-8")[-6000:], "```"]
    bp = out.parent / f"{out.name}.brief.md"
    bp.write_text(scrub("\n".join(brief) + "\n"), encoding="utf-8")
    say(f"prepare: {task['id']} at {out} (branch base, one commit, tree {task['base_tree'][:10]}); brief {bp}")
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
        bfile.write_text(json.dumps({"format": run_doc.get("format"), "commit": ctx.main, "run": base.get("label"),
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


def signal_one(ctx: Ctx, task: dict, tdir: Path, wt_path: Path, invariants: bool, refset: str | None) -> dict:
    commit = ensure_base(ctx, task, tdir)
    work = ctx.bench / "work" / task["id"] / "signal"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    wt = worktree(ctx.fork, wt_path, commit)
    t0 = time.time()
    dll = ctx.tc.emudump(wt, work / "emudump.log")
    build_s = round(time.time() - t0, 1)
    res = {"task": task["id"], "main": ctx.main, "base_commit": commit, "build_s": build_s, "gates": {}}
    if dll is None:
        res.update(detected=True, lines=["EmuDump does not build"], error="EmuDump does not build on the base")
        return res
    base = main_corpus(ctx)
    t1 = time.time()
    try:
        with ctx.lock:  # one corpus run at a time: its wall time is the gate's
            doc = ctx.gates.corpus_run(wt, dll, f"debugbench-{task['id']}-signal", work)
    except CorpusRunFailed as ex:
        # Main's run went through, so the base takes the gate down: the fixer sees the check fail.
        g = {"detected": True, "crashed": str(ex), "regressed": 0, "regressed_ids": [], "fields": {}, "fights": [],
             "run_s": round(time.time() - t1, 1)}
        g["seconds_to_detect"] = round(build_s + g["run_s"], 1)
        res["gates"]["corpus"] = g
        lines = [f"corpus check against main: the run fails on this repo ({ex})"]
    else:
        g = judge_corpus(ctx, doc, base, tdir / "corpus-check.txt")
        g["run_s"] = round(time.time() - t1, 1)
        g["detected"] = bool(g["regressed"]) or g["problems"] > 0
        g["seconds_to_detect"] = round(build_s + g["run_s"], 1) if g["detected"] else None
        res["gates"]["corpus"] = g
        lines = [f"corpus check against main: {g['regressed']} fights worse"
                 + (f" (fields {', '.join(list(g['fields'])[:6])})" if g["fields"] else "")
                 + f", {g['improved']} better; identical {g['main_totals'].get('identical')} -> {g['totals'].get('identical')}"]
    if invariants:
        t2 = time.time()
        inv = judge_invariants(main_invariants(ctx), ctx.gates.invariants(wt, dll, work))
        inv["run_s"] = round(time.time() - t2, 1)
        if "detected" in inv:
            inv["seconds_to_detect"] = round(build_s + inv["run_s"], 1) if inv["detected"] else None
        res["gates"]["invariants"] = inv
        lines.append("invariants: " + (", ".join(f"{k} {a}->{b}" for k, (a, b) in inv.get("worse", {}).items()) or
                                       inv.get("error", "no new violation")))
    if refset:
        t3 = time.time()
        r = ctx.gates.refset(wt, f"debugbench-{task['id']}", refset)
        r.update(detected=r["exit"] == 1, run_s=round(time.time() - t3, 1))
        res["gates"]["refset"] = r
        lines.append(f"reference set gate: {'worse' if r['detected'] else 'no worse'} (exit {r['exit']})")
    res["detected"] = any(gv.get("detected") for gv in res["gates"].values())
    det = [gv.get("seconds_to_detect") for gv in res["gates"].values() if gv.get("seconds_to_detect")]
    res["seconds_to_detect"] = min(det) if det else None
    res["lines"] = lines
    (tdir / "signal.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def cmd_signal(a) -> int:
    ctx = Ctx(a.fork, a.bench, gates=Gates(Path(a.corpus), Path(a.jobs_dir), a.procs))
    tasks = all_tasks(ctx.bench) if a.all else [load_task(ctx.bench, t)[0] for t in a.tasks]
    if a.missing:
        tasks = [t for t in tasks if not (ctx.tasks_dir / t["split"] / t["id"] / "signal.json").exists()]
    with bench_lock(ctx.bench, "signal"):
        return run_signals(ctx, tasks, a.jobs, not a.no_invariants, a.refset)


def run_signals(ctx: Ctx, tasks: list, jobs: int, invariants: bool, refset: str | None) -> int:
    mains = {t["main"] for t in tasks}
    if len(mains) > 1:
        raise BenchError(f"the tasks were harvested on {len(mains)} mains; signal one main's tasks at a time")
    if mains:
        ctx.main = mains.pop()
        ctx.main_tree = tree_of(ctx.fork, ctx.main)
    main_corpus(ctx)
    if invariants:
        main_invariants(ctx)
    slots = queue.Queue()
    for k in range(jobs):
        slots.put(ctx.slot(f"s{k}"))

    def one(t: dict) -> dict:
        wt = slots.get()
        try:
            r = signal_one(ctx, t, ctx.tasks_dir / t["split"] / t["id"], wt, invariants, refset)
        except BenchError as ex:
            r = {"task": t["id"], "error": str(ex)[:300], "detected": None, "lines": []}
        finally:
            slots.put(wt)
        state = {True: "DETECTED", False: "silent"}.get(r.get("detected"), "error")
        say(f"  {t['id']} {state} {r.get('seconds_to_detect') or ''}  "
            f"{'; '.join(r.get('lines', [])) or r.get('error', '')}"[:300])
        return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        res = list(ex.map(one, tasks))
    park(ctx, [ctx.slot(f"s{k}") for k in range(jobs)])
    n = sum(1 for r in res if r.get("detected"))
    say(f"signal: {n} of {len(res)} tasks detected by today's gates")
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
              slot: Path) -> dict:
    commit = git(repo, "rev-parse", f"{ref}^{{commit}}")
    base_tree = task["base_tree"]
    if not has_object(repo, base_tree, "tree"):
        raise BenchError(f"{repo} does not hold {task['id']}'s base (tree {base_tree[:10]}): fix on a repo made by "
                         "`debugbench prepare`")
    diff = changed(repo, base_tree, commit)
    eng = [f for _, f in diff if f.startswith(ENGINE)]
    cand_sites = set(sites_between(repo, base_tree, commit, eng))
    truth = task["truth"]
    missed = sorted(set(truth["sites"]) - cand_sites)
    res = {"task": task["id"], "split": task["split"], "run": run, "candidate": commit, "main": task["main"],
           "changed_files": [f for _, f in diff], "engine_files": eng, "sites": sorted(cand_sites),
           "missed_sites": missed, "missed_files": sorted(set(truth["files"]) - set(eng)),
           "missed_members": sorted({member_of(s) for s in truth["sites"]} - {member_of(s) for s in cand_sites}),
           "extra_sites": sorted(cand_sites - set(truth["sites"])),
           "outside_engine": [f for _, f in diff if not f.startswith(ENGINE) and not proj_of(f)]}
    work = ctx.bench / "work" / task["id"] / f"score-{run}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    tree = candidate_tree(ctx, repo, commit, slot)
    # The base's judge, whatever the branch did to it.
    judge = {f: blob(ctx.fork, task["main"], f) for s, f in diff if f.startswith(JUDGE_PATHS)}
    overlay(tree, judge)
    dll = ctx.tc.emudump(tree, work / "emudump.log")
    if dll is None:
        res.update(builds=False, correct=False, corpus=None, hidden_pass=False)
        return finish(ctx, res, report, run)
    res["builds"] = True
    try:
        with ctx.lock:
            doc = ctx.gates.corpus_run(tree, dll, f"debugbench-{task['id']}-{run}", work)
    except CorpusRunFailed as ex:
        doc = None
        g = {"crashed": str(ex), "regressed": None, "regressed_ids": [], "problems": 1, "fights": []}
    if doc is not None:
        g = judge_corpus(ctx, doc, main_corpus(ctx), work / "corpus-check.txt")
    sig = tdir / "signal.json"
    was = set(json.loads(sig.read_text(encoding="utf-8"))["gates"]["corpus"].get("regressed_ids", [])) if sig.exists() else set()
    g["restored"] = len(was - set(g["regressed_ids"])) if doc is not None else 0
    g["base_regressed"] = len(was)
    res["corpus"] = {k: v for k, v in g.items() if k != "fights"} | {"fights": g["fights"][:10]}
    # Hidden and collateral tests: main's test files over the candidate's.
    hidden_files = {str(p.relative_to(tdir / "hidden")).replace("\\", "/"): p.read_bytes()
                    for p in (tdir / "hidden").rglob("*") if p.is_file()}
    overlay(tree, hidden_files)
    want = set(task["hidden"]) | set(task["collateral"])
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
        res["hidden"] = {t: got.get(t, "missing") for t in task["hidden"]}
        res["collateral"] = {t: got.get(t, "missing") for t in task["collateral"]}
        res["hidden_pass"] = all(v == "pass" for v in res["hidden"].values())
        res["collateral_pass"] = all(v == "pass" for v in res["collateral"].values())
    if not quick and got is not None:
        full = _outcomes(ctx, tree, projects_in(tree), None, work / "trx", work / "tests.log")
        res["unit_regressions"] = sorted(_broken(main_tests_for(ctx, task), full or {}) - want) if full is not None else None
    res["correct"] = bool(res.get("hidden_pass") and res.get("collateral_pass") and g["regressed"] == 0
                          and g["problems"] == 0 and not res.get("unit_regressions"))
    return finish(ctx, res, report, run)


def main_tests_for(ctx: Ctx, task: dict) -> dict:
    p = ctx.bench / "main" / task["main"][:12] / "tests.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return main_tests(ctx)


def finish(ctx: Ctx, res: dict, report: dict, run: str) -> dict:
    task, _ = load_task(ctx.bench, res["task"])
    res["localisation"] = localise(report.get("suspects", []), task["truth"])
    res["report_suspects"] = report.get("suspects", [])[:10]
    for k in ("hours", "tokens", "verdict", "round"):
        res[k] = report.get(k)
    res["scored"] = datetime.datetime.now().isoformat(timespec="seconds")
    out = ctx.bench / "runs" / run / f"{res['task']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def cmd_score(a) -> int:
    ctx = Ctx(a.fork, a.bench, gates=Gates(Path(a.corpus), Path(a.jobs_dir), a.procs))
    task, tdir = load_task(ctx.bench, a.task)
    ctx.main, ctx.main_tree = task["main"], tree_of(ctx.fork, task["main"])
    report = parse_report(Path(a.report) if a.report else None)
    for k in ("hours", "tokens", "verdict", "round"):
        if getattr(a, k) is not None:
            report[k] = getattr(a, k)
    with bench_lock(ctx.bench, "score"):
        res = score_one(ctx, task, tdir, Path(a.repo), a.ref, a.run, report, a.quick, ctx.slot("score"))
    c = res.get("corpus") or {}
    loc = res["localisation"]
    say(f"score {task['id']} ({a.run}): {'CORRECT' if res['correct'] else 'not correct'}: hidden "
        f"{'pass' if res.get('hidden_pass') else 'FAIL'}, collateral {'pass' if res.get('collateral_pass') else 'FAIL'}, "
        f"corpus {c.get('regressed', '?')} worse than main ({c.get('restored', '?')} of {c.get('base_regressed', '?')} "
        f"restored), unit regressions {len(res.get('unit_regressions') or [])}, missed sites {len(res['missed_sites'])} "
        f"of {len(task['truth']['sites'])}, localisation top-1 {loc['top1_file']} top-3 {loc['top3_file']}")
    return 0 if res["correct"] else 1


# -------------------------------------------------------------------------------------------- report

def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}% ({n}/{d})" if d else "-"


def task_set_report(bench: Path, split: str) -> list:
    tasks = all_tasks(bench, split)
    lines = [f"# debugbench task set ({split})", ""]
    by = collections.Counter(t["split"] for t in tasks)
    lines.append(f"- tasks: {len(tasks)} (dev {by['dev']}, held out {by['heldout']}); ours {sum(t['ours'] for t in tasks)}, "
                 f"upstream {sum(not t['ours'] for t in tasks)}")
    sigs = []
    for t in tasks:
        p = bench / "tasks" / t["split"] / t["id"] / "signal.json"
        if p.exists():
            sigs.append((t, json.loads(p.read_text(encoding="utf-8"))))
    lines.append(f"- signal run on {len(sigs)}")
    gates = collections.defaultdict(lambda: [0, 0])
    for _, s in sigs:
        for g, v in s.get("gates", {}).items():
            if v.get("detected") is not None:
                gates[g][1] += 1
                gates[g][0] += bool(v.get("detected"))
    lines.append(f"- detected by any gate: {pct(sum(1 for _, s in sigs if s.get('detected')), len(sigs))}")
    for g, (n, d) in sorted(gates.items()):
        lines.append(f"  - {g}: {pct(n, d)}")
    secs = [s["seconds_to_detect"] for _, s in sigs if s.get("seconds_to_detect")]
    if secs:
        lines.append(f"- seconds to detect (build + gate): median {statistics.median(secs):.0f}, max {max(secs):.0f}")
    fields = collections.Counter(f for _, s in sigs for f in (s.get("gates", {}).get("corpus", {}).get("fields") or {}))
    if fields:
        lines.append("- fields the corpus check names most: " + ", ".join(f"{k} ({v})" for k, v in fields.most_common(8)))
    silent = [t["id"] for t, s in sigs if s.get("detected") is False]
    if silent:
        lines.append(f"- silent (no gate sees the bug): {len(silent)}: {', '.join(silent[:40])}")
    return lines


def run_report(bench: Path, run: str, split: str) -> list:
    rows = []
    for p in sorted((bench / "runs" / run).glob("db-*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if split in ("all", r.get("split")):
            rows.append(r)
    lines = [f"# debugbench run {run} ({split}): {len(rows)} tasks", ""]
    if not rows:
        return lines + ["no scored tasks"]
    det = []
    for r in rows:
        p = bench / "tasks" / r["split"] / r["task"] / "signal.json"
        if p.exists():
            det.append(bool(json.loads(p.read_text(encoding="utf-8")).get("detected")))
    correct = [r for r in rows if r.get("correct")]
    with_sus = [r for r in rows if r["localisation"]["suspects"]]
    site_rows = [r for r in rows if r["localisation"]["top1_site"] is not None]
    verdicts = [r for r in rows if r.get("verdict")]
    first = [r for r in verdicts if str(r["verdict"]).lower().startswith("accept") and int(r.get("round") or 1) == 1]
    hours = sum(float(r.get("hours") or 0) for r in rows)
    tokens = sum(int(r.get("tokens") or 0) for r in rows)
    regs = sum((r.get("corpus") or {}).get("regressed", 0) for r in rows)
    unit = sum(len(r.get("unit_regressions") or []) for r in rows)
    missed = [len(r["missed_sites"]) for r in rows]
    lines += [f"- detection (today's gates on the base): {pct(sum(det), len(det))}",
              f"- localisation, file: top-1 {pct(sum(r['localisation']['top1_file'] for r in rows), len(rows))}, "
              f"top-3 {pct(sum(r['localisation']['top3_file'] for r in rows), len(rows))} (with a report: {len(with_sus)})",
              f"- localisation, site: top-1 {pct(sum(bool(r['localisation']['top1_site']) for r in site_rows), len(site_rows))}, "
              f"top-3 {pct(sum(bool(r['localisation']['top3_site']) for r in site_rows), len(site_rows))}",
              f"- fix correctness: {pct(len(correct), len(rows))} (hidden {pct(sum(bool(r.get('hidden_pass')) for r in rows), len(rows))}"
              f", corpus 0 worse {pct(sum((r.get('corpus') or {}).get('regressed', 1) == 0 for r in rows), len(rows))})",
              f"- first-round acceptance: {pct(len(first), len(verdicts))}",
              f"- missed sibling sites per task: mean {statistics.mean(missed):.1f}, tasks with any {sum(1 for m in missed if m)}",
              f"- regressions added: {regs} corpus fights worse than main, {unit} unit tests",
              f"- cost per correct fix: {hours / len(correct):.2f} h, {tokens / len(correct):.0f} tokens" if correct and (hours or tokens)
              else f"- cost: {hours:.2f} h, {tokens} tokens, {len(correct)} correct", "",
              "| task | correct | hidden | corpus worse | restored | missed sites | top-1 | top-3 | h | tokens | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r.get("corpus") or {}
        loc = r["localisation"]
        lines.append(f"| {r['task']} | {'yes' if r.get('correct') else 'no'} | {'pass' if r.get('hidden_pass') else 'fail'} | "
                     f"{c.get('regressed', '-')} | {c.get('restored', '-')}/{c.get('base_regressed', '-')} | "
                     f"{len(r['missed_sites'])} | {'y' if loc['top1_file'] else 'n'} | {'y' if loc['top3_file'] else 'n'} | "
                     f"{r.get('hours') or '-'} | {r.get('tokens') or '-'} | {r.get('verdict') or '-'} {r.get('round') or ''} |")
    return lines


def cmd_report(a) -> int:
    bench = Path(a.bench)
    lines = run_report(bench, a.run, a.split) if a.run else task_set_report(bench, a.split)
    text = "\n".join(lines) + "\n"
    if a.run:
        (bench / "runs" / a.run / "report.md").write_text(text, encoding="utf-8")
    else:
        (bench / "tasks" / f"report-{a.split}.md").write_text(text, encoding="utf-8")
    say(text)
    return 0


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--fork", default=str(DEFAULT_FORK), help="the fork's repo (history, main, the base commits)")
    ap.add_argument("--bench", default=str(DEFAULT_BENCH))
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
    for name in ("signal", "score"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
        p.add_argument("--jobs-dir", default=str(DEFAULT_JOBS), help="the referee's jobs folder (invariants)")
        p.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    s = sub.choices["signal"]
    s.add_argument("tasks", nargs="*")
    s.add_argument("--all", action="store_true")
    s.add_argument("--missing", action="store_true", help="only tasks without a signal yet")
    s.add_argument("--jobs", type=int, default=2, help="tasks at once (corpus runs still go one at a time)")
    s.add_argument("--no-invariants", action="store_true")
    s.add_argument("--refset", metavar="MAIN_SCORE", help="also the reference-set gate against this main score")
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
    rp = sub.add_parser("report")
    rp.add_argument("--run")
    rp.add_argument("--split", choices=("dev", "heldout", "all"), default="all")
    a = ap.parse_args(argv)
    if a.cmd == "signal" and not a.tasks and not a.all:
        ap.error("signal: name tasks or --all")
    try:
        return {"harvest": cmd_harvest, "list": cmd_list, "prepare": cmd_prepare, "signal": cmd_signal,
                "score": cmd_score, "report": cmd_report}[a.cmd](a)
    except BenchError as ex:
        say(f"debugbench: {ex}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
