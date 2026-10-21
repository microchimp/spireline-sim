"""The recorded-fight corpus: every fight the referee ever recorded in the game, replayed as a regression gate.

    python tools/fidelity/corpus.py build [--jobs <dir>]        scan every *.out.json under jobs and jobs-*
    python tools/fidelity/corpus.py add <out.json> [...]        add one batch's recordings, from anywhere
    python tools/fidelity/corpus.py run [--procs 16]            replay the corpus (EmuDump fight + reanchor)
    python tools/fidelity/corpus.py baseline [--run <results>]  store a run as the baseline of this commit
    python tools/fidelity/corpus.py check [--run <results>] [--baseline <commit|file>]
    python tools/fidelity/corpus.py status
    python tools/fidelity/corpus.py identity --a <old EmuDump.dll> --b <new EmuDump.dll>   byte-safety of a refactor
    python tools/fidelity/corpus.py identity --runs <results.json> <results.json>     two `run --state-hash` runs

Run from the fork's root (or anywhere: it finds the fork from its own path). The corpus lives outside
the repo (default D:/spireline/referee/corpus, or $SPIRELINE_CORPUS) and is never committed: it holds
game recordings.

- ``build`` reads every *.out.json under the jobs folder and its sibling jobs-* folders, at any depth.
  Each folder has an entry in POLICY (include, include with a tag, or exclude, with the reason); a
  folder without one is read and its fights tagged ``unreviewed``.
- A fight is kept once, by a hash of its spec (seed, character, ascension, Neow pick, deck additions,
  room, and the newer spec options when a recording sets them) and of its game side (each step's action
  and state, and the choices). Everything else in a step (cost, trace, checksums) is instrumentation and
  is dropped before hashing and storing, as are the run's timing, cost and serve-mode fields. The same
  fight recorded in a shard, a wave and the merged batch file is one entry with all its sources (paths).
- A fight the game broke off (an error after some steps) is kept up to its last good step and flagged
  ``truncated``: up to the first step that changed nothing (the driver acted while the game was still
  setting up the turn and the game parked it, fork fix/referee-stalls), without the end of turn before
  it, and without a trailing end of turn. One that is a prefix of a complete fight of the same spec is
  only another source of it.
- Each fight is tagged with the game build (sts2.dll sha256 when the recording names one, else the
  build the referee has always required), the recording format (1: card choices not tied to a step;
  2: choices carry the step they belong to; 3: refused plays recorded as played=false, fork 732aacc)
  and flags. Before 732aacc the driver aimed RandomEnemy cards, the game refused the play and the
  recording did not say so, so a format 1-2 fight with such a play is flagged driver-artifact-risk
  (kept: the gate only needs it to behave the same from one emulator commit to the next).
- ``tools/lab/corpus_import.py`` adds the lab's recordings the same way (one fight per setup, a size budget):
  fights the emulator matched exactly plain, fights it diverged on tagged ``lab-diverged``. The tag is
  provenance (what the emulator of the day did), not a promise: the baselines hold the verdicts and `check`
  flags such a fight only when it gets worse. A stored record is the game's side only, so the corpus
  replays unchanged on any emulator that reads the same recordings, a rebuilt one included.
- A fight whose spec needs an EmuDump feature (enemy_hp_mult, a console setup) is replayed only by an
  EmuDump whose source has it; until then ``run`` skips it and says so.
- ``run`` first picks up new or changed recordings (``--no-scan`` turns it off), then replays every
  fight in both EmuDump modes, split across processes (each EmuDump call has a timeout), and writes one
  result per fight: identical or the first divergence (step, field, card), any emulator exception, and
  the re-anchored verdict of every step (M match, D diverge, E error, U unjudged, - not re-anchored).
- ``identity`` replays every fight with two EmuDump builds and compares the full-state hash after every
  step (EmuDump fight --state-hash: every field of the run and combat state by reflection, the fields
  Clone copies, not the dump's bytes), naming the fields that differ at the first differing step. A
  structure-only refactor must leave every hash the same.
- ``check`` compares a run with the baseline of this commit or its nearest ancestor that has one. A
  regression is a fight whose plain replay now diverges earlier than before (identical before and not
  now included), a new emulator exception in either replay, or a re-anchored step that matched and does
  not now, now throws, or now diverges in a field it did not before. A full check also fails when a
  baseline fight was not compared. Exit 1 on any, and when the check compared no baseline fight, ran an
  EmuDump older than the emulator sources, or lost fights to EmuDump failures (timeouts, crashes).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":  # run as a script: import the package so pool workers can find this module
    sys.path.insert(0, str(ROOT))

DEFAULT_CORPUS = Path(os.environ.get("SPIRELINE_CORPUS", r"D:\spireline\referee\corpus"))
DEFAULT_JOBS = Path(os.environ.get("SPIRELINE_JOBS", r"D:\spireline\referee\jobs"))
EMUDUMP = ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
# The only build referee/run_referee.py has ever accepted (KNOWN_BUILDS): v0.107.1 / 23811903.
KNOWN_BUILD = "a1f9e653f1e28e4076558fee1e60d218619cb7e057b887c6417f62c62c6d7a52"
BUILD_KEYS = ("sts2_sha256", "sts2_dll_sha256", "game_sha256", "build_sha256")
# What drives a replay (EmuDump reads these and the steps and choices).
SPEC_KEYS = ("seed", "character", "ascension", "neow_pick", "deck_add", "room")
# Spec options of newer drivers: part of a fight's identity only when a recording sets them, so the id of
# a fight recorded without them never changes.
EXTRA_SPEC_KEYS = ("enemy_hp_mult", "setup", "use_potions", "script", "fight", "room_choices_from")
# A step's game side. Anything else in a step (cost, trace, hidden, checksums, net) is instrumentation.
STEP_KEYS = ("action", "state")
# Not game state: timing and cost (instrumentation), the save file, serve.py's per-line tags.
DROP_KEYS = {"phase_ms", "timing_ms", "cost", "cost_setup", "checksum_cost", "checksums_tail", "settle_audit",
             "idle_probe", "save", "_id", "_pid", "_t"}
SERVE_KEYS = {"_id", "_pid", "_t"}
INDEX_FORMAT = 2  # of index.json (2: sources are paths, fights carry their spec, tags and needs)
RESULTS_FORMAT = 2  # of run results and baselines (2: ferr/rerr apart, dp holds every diverging field)
VERDICT = {"match": "M", "diverge": "D", "error": "E", "unjudged": "U"}
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
# Spec options an EmuDump must apply for a replay to mean anything, and the text its source has when it does.
FEATURES = {"enemy_hp_mult": '"enemy_hp_mult"', "setup": '["setup"]'}
# What a replay is judged with: a change here changes results without any emulator change.
JUDGE_PATHS = ("tools/fidelity", "data/id_map.json")

# Every folder of recordings (relative to the jobs folder's parent) and how its fights enter the corpus:
# (action, tag, reason). Matched on the file itself, then its folder and the folders above it ("jobs"
# itself only for the files directly in it). The counts in the reasons were measured on 2026-10-14.
POLICY = {
    "jobs": ("include", None, "the referee's batches"),
    "jobs/stalls": ("include", None, "fix/referee-stalls: stalled fights re-run"),
    "jobs/runref": ("include", None, "whole-run reference jobs: their fight records are ordinary ones"),
    "jobs/frames": ("include", "settle-trial", ("speed/frames settle trials (frame or flag settle, fast tweens, frame sleep or none): "
                    "every copy of a fight the stock settle also recorded is step-identical to it (513 of 513 outside "
                    "s0_frames); tagged so a regression seen only in them can be weighed")),
    "jobs/frames/s0_frames.out.json": ("exclude", None, ("frame-count settle with no frame sleep: step 0 was recorded "
                                       "before the opening draw finished (16 of 20 fights hold 3-4 cards where the "
                                       "stock recording holds 5)")),
    "jobs-stream": ("include", None, "speed/streaming: the same driver in a streaming server pool"),
    "jobs-checksum": ("include", None, "feat/checksum-tap: per-step checksums are instrumentation, dropped"),
    "jobs-trace": ("include", None, "feat/divergence-trace: per-step traces dropped; enemy intents kept (flag intents)"),
    "jobs-windowed": ("include", None, "check/windowed: the normal windowed game"),
    "jobs-setup": ("include", None, "feat/console-setup: a fight with a setup is replayed once EmuDump applies it"),
    "jobs-longfights": ("include", None, "speed/long-fights: an enemy_hp_mult fight is replayed once EmuDump applies it"),
    "jobs-nodeless": ("include", "nodeless-trial", ("speed/nodeless trials, the nodeless and the full host mixed (no "
                      "recording says which): per file, 88-100% of the fights also recorded elsewhere match a stock "
                      "copy, as in the full host's own files (91-100%)")),
    "jobs-pickorder": ("include", None, ("fix/turn-start-choices-2: every corpus fight with a turn-start screen (87 specs) "
                       'recorded again with pick_timing "ui" (a turn-start screen answered where the UI shows it), '
                       "step-identical in 87 of 87 to the same driver's 'instant' recordings; and the probe and "
                       "turn-start cases where the pick timing decides the outcome (P19, P20, Bone Tea with Toolbox, "
                       "Gambling Chip and Choices Paradox, Tyranny with Furnace)")),
    "jobs-nodeless/fpm_full3s0.out.json": ("exclude", None, ("full host, frame-count settle with no frame sleep: 181 of "
                                           "295 fights match a stock copy (61%), against 88-100% in the folder's "
                                           "other files")),
}
UNREVIEWED = ("include", "unreviewed", "no entry in corpus.POLICY yet")

# Single fights the gate leaves out, by fight id, each with its reason: a reference the game or the
# driver got wrong, which no emulator should match. The fight stays in the corpus, flagged "excluded"
# (status lists them); run does not replay it and check compares neither side of it.
_FROZE = ("the game froze at {who}'s auto-play before the turn-start screen ({what}) and the driver went on "
          "acting: from step {step} on, {noop} of the {after} steps change nothing (no-op-step); a bad reference, "
          "not an emulator regression (fix/turn-start-choices-2)")
FIGHT_POLICY: dict[str, str] = {
    "4fbfad918b1f5ff8": _FROZE.format(who="Hellraiser", what="Stratagem", step=19, noop=11, after=14),  # 6V7YRDLB2L
    "3e3439d56fce03b6": _FROZE.format(who="Hellraiser", what="Entropy", step=15, noop=11, after=14),    # ET9FHLWRS7
    "7b21e9f8e63f57aa": _FROZE.format(who="Mayhem", what="Brand", step=27, noop=29, after=35),          # AN1KDHMVHL
    "495f56a1a3899974": _FROZE.format(who="Mayhem", what="Entropy", step=15, noop=15, after=17),        # V3LDXLDHW2
    "684a7fde207495a1": _FROZE.format(who="Hellraiser", what="Entropy", step=11, noop=15, after=25),    # TCU1SMLQHY
    "790cf1a5af4c3df0": _FROZE.format(who="Mayhem", what="Entropy", step=15, noop=16, after=18),        # V3LDXLDHW2, nodeless
}


def apply_fight_policy(meta: dict) -> dict:
    """Flag a fight FIGHT_POLICY leaves out ("excluded", with its reason under "excluded"), and only those."""
    reason = FIGHT_POLICY.get(meta.get("id", ""))
    meta["flags"] = [f for f in meta.get("flags", []) if f != "excluded"] + (["excluded"] if reason else [])
    if reason:
        meta["excluded"] = reason
    else:
        meta.pop("excluded", None)
    return meta


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spec_of(r: dict) -> dict:
    d = {k: r.get(k) for k in SPEC_KEYS}
    d.update({k: r[k] for k in EXTRA_SPEC_KEYS if r.get(k) not in (None, [], {}, False)})
    if d.get("enemy_hp_mult") == 1:  # a plain fight
        del d["enemy_hp_mult"]
    return d


def game_steps(steps):
    return [{k: s[k] for k in STEP_KEYS if k in s} if isinstance(s, dict) else s for s in steps or []]


def spec_id(r: dict) -> str:
    return sha(canon(spec_of(r)))[:16]


def fight_id(r: dict) -> str:
    spec = sha(canon(spec_of(r)))
    side = sha(canon({"steps": game_steps(r.get("steps")), "choices": r.get("choices", [])}))
    return sha(spec + side)[:16]


def clean(r) -> bool:
    """A fight the game finished: no error, steps recorded, and the run fields a replay needs."""
    return isinstance(r, dict) and not r.get("error") and bool(r.get("steps")) and r.get("ascension") is not None


def needs_of(r: dict) -> list:
    out = []
    if r.get("enemy_hp_mult") not in (None, 1):
        out.append("enemy_hp_mult")
    if r.get("setup") or r.get("use_potions"):
        out.append("setup")
    return out


def emudump_features(root: Path = ROOT) -> set:
    """The spec options this EmuDump's source applies (FEATURES)."""
    text = "".join(p.read_text(encoding="utf-8", errors="replace") for p in (root / "tools" / "emudump").glob("*.cs"))
    return {f for f, marker in FEATURES.items() if marker in text}


def random_enemy_cards(root: Path = ROOT) -> set:
    """Cards the game targets at a random enemy (TargetType.RandomEnemy), from the emulator's card table."""
    src = root / "src" / "Sts2Emulator" / "Generated" / "Cards.g.cs"
    text = src.read_text(encoding="utf-8") if src.exists() else ""
    return {m.group(1) for line in text.splitlines() if "CardTarget.RandomEnemy" in line
            for m in [re.search(r'Entry: "([A-Z0-9_]+)"', line)] if m}


def _kind(step: dict) -> str | None:
    return (step.get("action") or {}).get("kind")


def _plays(r: dict) -> list:
    return [s.get("action") or {} for s in r.get("steps", []) if (s.get("action") or {}).get("kind") == "play"]


def noop_steps(steps: list) -> list:
    """Steps whose play or end of turn changed nothing the snapshot shows.

    A play the game refused and the recording marks played=false is not one.
    """
    return [j for j in range(1, len(steps)) if (steps[j].get("action") or {}).get("played") is not False
            and steps[j].get("state") == steps[j - 1].get("state")]


def last_good(steps: list) -> int:
    """How many steps of a fight the game broke off can be trusted.

    Up to the first step that did nothing: the driver acted while the game was still setting up the turn
    and the game parked the request (fork fix/referee-stalls), so the end of turn before it was recorded
    before that turn's start had run, and it goes too. A trailing end of turn goes for the same reason.
    """
    nops = noop_steps(steps)
    n = nops[0] if nops else len(steps)
    if n > 1 and _kind(steps[n - 1]) == "end_turn":
        n -= 1
    return n


def cut_choices(choices: list, n: int) -> list:
    """The choices of the first n steps (a choice that names no step is kept: format 1 cannot tell)."""
    return [c for c in choices if not (isinstance(c, dict) and isinstance(c.get("step"), int) and c["step"] >= n)]


def evidence(r: dict) -> dict:
    """What a recording says about the driver that made it."""
    return {"played": any("played" in a for a in _plays(r)),
            "choice_step": any(isinstance(c, dict) and "step" in c for c in r.get("choices", []) or []),
            "serve": bool(SERVE_KEYS & set(r))}


def file_evidence(results: list) -> dict:
    ev = [evidence(r) for r in results if isinstance(r, dict)]
    return {k: any(e[k] for e in ev) for k in ("played", "choice_step", "serve")}


def recording_format(r: dict, file_ev: dict | None = None) -> int:
    """1: choices not tied to steps; 2: choices carry their step; 3: refused plays recorded (732aacc).

    One file is one driver, so what any fight of the file shows holds for the others (a fight with no
    card choice says nothing about the tags by itself).
    """
    ev, fe = evidence(r), file_ev or {}
    if ev["played"] or fe.get("played"):
        return 3
    return 2 if ev["choice_step"] or fe.get("choice_step") else 1


def flags_of(r: dict, fmt: int, rand_cards: set) -> list:
    flags = []
    if fmt < 3 and any(a.get("card") in rand_cards for a in _plays(r)):
        flags.append("driver-artifact-risk")
    if noop_steps(r.get("steps", [])):
        flags.append("no-op-step")
    if "game_error" in r:
        flags.append("truncated")
    if any("intents" in e for s in r.get("steps", []) for e in (s.get("state") or {}).get("enemies", []) or []):
        flags.append("intents")
    return flags


def build_of(r: dict, doc: dict) -> str:
    for src in (r, doc):
        for k in BUILD_KEYS:
            v = src.get(k)
            if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v):
                return v
    return KNOWN_BUILD


def stripped(r: dict) -> dict:
    out = {k: v for k, v in r.items() if k not in DROP_KEYS}
    if "steps" in out:
        out["steps"] = game_steps(out["steps"])
    return out


def _first_line(err) -> str:
    return str(err).strip().splitlines()[0][:200] if err else ""


def prepare(r) -> tuple[dict | None, str]:
    """The game side of one result worth keeping, and what it is (clean, truncated) or why it is not kept."""
    if not isinstance(r, dict) or not r.get("steps") or r.get("ascension") is None:
        return None, "no_fight"
    s0 = r["steps"][0].get("state") if isinstance(r["steps"][0], dict) else None
    if not isinstance(s0, dict) or "snapshot" in s0:  # an emulator's output, not the game's
        return None, "not_game"
    rec = stripped(r)
    if not r.get("error"):
        return rec, "clean"
    n = last_good(rec["steps"])
    if n < 2:
        return None, "error_before_a_good_step"
    rec.pop("error")
    rec["steps"] = rec["steps"][:n]
    if "choices" in rec:
        rec["choices"] = cut_choices(rec["choices"] or [], n)
    rec["game_error"] = {"error": _first_line(r["error"]), "recorded_steps": len(r["steps"]), "kept": n}
    return rec, "truncated"


def policy_of(rel: str) -> tuple:
    """(name, action, tag, reason) of a recording at ``rel`` (posix, relative to the jobs folder's parent)."""
    if rel in POLICY:
        return (rel, *POLICY[rel])
    parts = rel.split("/")
    for i in range(len(parts) - 1, 0, -1):
        folder = "/".join(parts[:i])
        if folder == "jobs" and i != len(parts) - 1:
            continue
        if folder in POLICY:
            return (folder, *POLICY[folder])
    return ("/".join(parts[:-1]) or ".", *UNREVIEWED)


def locate(f: Path, jobs: Path) -> tuple[str, str | None]:
    """A recording's source key and its policy path.

    The key is its path, relative to the jobs folder's parent when it is under it; the policy path is the
    same with the jobs folder named "jobs", None when the file is outside the jobs folders.
    """
    f, jobs = Path(f).resolve(), Path(jobs).resolve()
    try:
        rel = f.relative_to(jobs.parent).as_posix()
    except ValueError:
        home = str(Path.home())
        p = str(f)
        return ("~" + p[len(home):] if p.lower().startswith(home.lower()) else p).replace("\\", "/"), None
    top, _, rest = rel.partition("/")
    if top == jobs.name:
        return rel, "jobs/" + rest
    if top.startswith(jobs.name + "-"):
        return rel, "jobs" + top[len(jobs.name):] + "/" + rest
    return rel, None


def recordings(jobs: Path) -> list:
    """Every *.out.json under the jobs folder and its sibling jobs-* folders, oldest first."""
    jobs = Path(jobs)
    roots = [jobs, *sorted(p for p in jobs.parent.glob(jobs.name + "-*") if p.is_dir())]
    files = {f for r in roots if r.is_dir() for f in r.rglob("*.out.json") if f.is_file()}
    return sorted(files, key=lambda p: (p.stat().st_mtime_ns, str(p)))


# ---------------------------------------------------------------------------------------- storage


class Corpus:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.fights_dir = self.path / "fights"
        self.index_path = self.path / "index.json"

    def load_index(self) -> dict:
        if self.index_path.exists():
            idx = json.loads(self.index_path.read_text(encoding="utf-8"))
            idx = self._migrate(idx) if idx.get("format", 1) < INDEX_FORMAT else idx
            for m in idx["fights"]:
                apply_fight_policy(m)
            return idx
        return {"format": INDEX_FORMAT, "fights": [], "scanned": {}, "excluded": {}}

    def _migrate(self, idx: dict) -> dict:
        """Bring a format 1 index up to date.

        Format 1 keyed sources and scanned files by bare file name: forget both (the next scan reads every
        file again and adds each source by path) and give each fight its spec, tags and needs.
        """
        for m in idx["fights"]:
            rec = self.read(m["id"])
            m.update(spec=spec_id(rec), sources=[], tags=[], needs=needs_of(rec))
        idx.update(format=INDEX_FORMAT, scanned={}, excluded={})
        return idx

    def save_index(self, idx: dict) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(idx, indent=0, sort_keys=False), encoding="utf-8")
        tmp.replace(self.index_path)

    def record_path(self, fid: str) -> Path:
        return self.fights_dir / f"{fid}.json.gz"

    def read(self, fid: str) -> dict:
        with gzip.open(self.record_path(fid), "rt", encoding="utf-8") as f:
            return json.load(f)

    def write(self, fid: str, rec: dict) -> None:
        p = self.record_path(fid)
        tmp = p.with_suffix(".tmp")
        with tmp.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(canon(rec).encode("utf-8"))
        tmp.replace(p)

    @contextlib.contextmanager
    def lock(self, timeout: float = 600):
        """One writer of the index at a time (batches may add while a check scans)."""
        self.path.mkdir(parents=True, exist_ok=True)
        lk = self.path / "index.lock"
        t0 = time.time()
        while True:
            try:
                fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.time() - t0 > timeout:
                    raise SystemExit(f"corpus locked by another writer ({lk.name}); remove it if no corpus.py is running") from None
                time.sleep(0.5)
        try:
            yield
        finally:
            os.close(fd)
            lk.unlink(missing_ok=True)

    def add_files(self, files: list, rand_cards: set | None = None, scan: bool = False, quiet: bool = False,
                  jobs: Path = DEFAULT_JOBS, labels: dict | None = None) -> dict:
        """Add every usable fight of each referee out.json; returns counts.

        ``labels`` maps str(file) to (source key, tag) for files that are not under the jobs folders (the lab's
        import writes a filtered copy of a job's recordings and names the original as the source).

        ``scan`` skips files unchanged since the last scan and refreshes the summary of excluded folders.
        """
        rand_cards = random_enemy_cards() if rand_cards is None else rand_cards
        self.fights_dir.mkdir(parents=True, exist_ok=True)
        counts = collections.Counter()
        with self.lock():
            idx = self.load_index()
            by_id = {m["id"]: m for m in idx["fights"]}
            by_spec = collections.defaultdict(list)
            for m in idx["fights"]:
                by_spec[m["spec"]].append(m["id"])
            scanned = idx.setdefault("scanned", {})
            if scan:
                idx["excluded"] = {}
            today = datetime.date.today().isoformat()
            cache: dict = {}

            def stored(fid):
                if fid not in cache:
                    cache[fid] = self.read(fid)
                return cache[fid]

            def source(m, key, tag):
                """A fight's tags are those every one of its sources has (seen in a stock batch too: no trial tag)."""
                if key not in m["sources"]:
                    mine = {tag} if tag else set()
                    m["tags"] = sorted(mine & set(m["tags"]) if m["sources"] else mine)
                    m["sources"].append(key)

            def put(rec, r, fmt, build, serve, key, tag, truncated):
                fid = fight_id(rec)
                m = by_id.get(fid)
                if m is None and truncated:  # a prefix of a fight already kept is that fight
                    n, ch = len(rec["steps"]), rec.get("choices") or []
                    for cid in by_spec[spec_id(rec)]:
                        c = stored(cid)
                        if len(c["steps"]) > n and c["steps"][:n] == rec["steps"] \
                                and cut_choices(c.get("choices") or [], n)[:len(ch)] == ch:
                            counts["truncated_known"] += 1
                            source(by_id[cid], key, tag)
                            return
                if m is None:
                    self.write(fid, rec)
                    m = {"id": fid, "spec": spec_id(rec), **{k: rec.get(k) for k in SPEC_KEYS}, "steps": len(rec["steps"]),
                         "plays": len(_plays(rec)), "format": fmt, "build": build, "flags": flags_of(rec, fmt, rand_cards),
                         "tags": [], "needs": needs_of(rec), "sources": [], "added": today}
                    if truncated:
                        m["cut"] = rec["game_error"]
                    if serve:
                        m["serve"] = True
                    idx["fights"].append(m)
                    by_id[fid] = m
                    by_spec[m["spec"]].append(fid)
                    counts["truncated_new" if truncated else "new"] += 1
                else:
                    counts["known"] += 1
                    m["format"] = max(m["format"], fmt)  # the same record seen in a file that says more
                    # Same steps, so the same flags, except "truncated": the stored record's, not this copy's.
                    m["flags"] = [f for f in flags_of(rec, m["format"], rand_cards) if f != "truncated"] +                         (["truncated"] if "cut" in m else [])
                source(m, key, tag)

            pending = []  # broken-off fights go last, so a prefix of a complete fight is never a fight of its own
            for f in files:
                f = Path(f)
                key, ppath = locate(f, jobs)
                name, action, tag, reason = policy_of(ppath) if ppath else ("(added)", "include", None, "")
                if labels and str(f) in labels:
                    key, tag = labels[str(f)]
                    action = "include"
                if action == "exclude":
                    counts["files_excluded"] += 1
                    if not scan and not quiet:
                        print(f"excluded {key}: {reason}", file=sys.stderr)
                    if scan:
                        e = idx["excluded"].setdefault(name, {"files": 0, "reason": reason})
                        e["files"] += 1
                    continue
                if tag == "unreviewed":
                    counts["files_unreviewed"] += 1
                st = f.stat()
                stamp = [st.st_size, st.st_mtime_ns, action, tag]
                if scan and scanned.get(key) == stamp:
                    counts["files_unchanged"] += 1
                    continue
                try:
                    doc = json.loads(f.read_text(encoding="utf-8"))
                    results = doc["results"]
                    assert isinstance(results, list)
                except (ValueError, KeyError, TypeError, AssertionError, UnicodeDecodeError) as ex:  # still being written, or not a referee file
                    counts["files_unreadable"] += 1
                    if not quiet:
                        print(f"skipped {key}: {type(ex).__name__}", file=sys.stderr)
                    continue
                counts["files"] += 1
                fe = file_evidence(results)
                for r in results:
                    counts["results"] += 1
                    rec, why = prepare(r)
                    if rec is None:
                        counts["not_kept"] += 1
                        continue
                    item = (rec, r, recording_format(r, fe), build_of(r, doc), evidence(r)["serve"] or fe["serve"], key, tag)
                    if why == "truncated":
                        pending.append(item)
                    else:
                        put(*item, truncated=False)
                scanned[key] = stamp
            for item in pending:
                put(*item, truncated=True)
            idx["format"] = INDEX_FORMAT
            self.save_index(idx)
        counts["fights"] = len(by_id)
        return dict(counts)

    def specs(self) -> set:
        """The spec ids (seed, character, setup, ...) of every fight in the corpus."""
        return {m["spec"] for m in self.load_index()["fights"]}

    def size(self) -> int:
        """Bytes of the stored fight records."""
        return sum(f.stat().st_size for f in self.fights_dir.glob("*.json.gz")) if self.fights_dir.is_dir() else 0

    def scan(self, jobs: Path, quiet: bool = False) -> dict:
        return self.add_files(recordings(jobs), scan=True, quiet=quiet, jobs=jobs)


# ------------------------------------------------------------------------------------ judging


def card_of(step: dict | None, j: int) -> str:
    if j == 0 or step is None:
        return "start"
    a = step.get("action") or {}
    return "end_turn" if a.get("kind") == "end_turn" else str(a.get("card"))


def judge_fight(g: dict, e: dict, nm: dict) -> dict:
    """Judge the plain replay.

    The first step whose state differs (or where the emulator's fight broke off), and any emulator
    exception (``ferr``), also one after the last step.
    """
    from .fight_compare import first_difference  # noqa: PLC0415
    from .reanchor_compare import group  # noqa: PLC0415

    gsteps = g.get("steps", [])
    fd = first_difference(g, e, nm)
    idx, path = (None, None) if fd is None else (fd[0], fd[1][0][0])
    out = {"fd": idx}
    if idx is not None:
        out.update(field=group(path) if path.startswith(".") else path, path=path,
                   card=card_of(gsteps[idx] if idx < len(gsteps) else None, idx))
    if e.get("error"):
        out["ferr"] = _first_line(e["error"])
    return out


def judge_reanchor(g: dict, r: dict, nm: dict, keys) -> dict:
    """Judge the re-anchored replay.

    One verdict letter per transition (step 1 on), the stop, every diverging field group per step
    (``dp``) and any emulator exception (``rerr``).
    """
    from .reanchor_compare import analyse, group  # noqa: PLC0415

    an = analyse(g, r, nm, "stop", keys)
    by_j = {s["j"]: s for s in an["steps"]}
    n = len(g.get("steps", []))
    rv = "".join(VERDICT.get(by_j[j]["verdict"], "?") if j in by_j else "-" for j in range(1, n))
    dp = {}
    for s in an["steps"]:
        if s["j"] > 0 and s["verdict"] in ("diverge", "error"):
            fields = {group(p) for p in s["paths"]}
            if s["choice_problems"]:
                fields.add("(choice)")
            dp[str(s["j"])] = sorted(fields) or ["(error)"]
    out = {"rv": rv}
    if dp:
        out["dp"] = dp
    if an["stop"]:
        out["stop"] = an["stop"]["why"][:120]
    if r.get("error"):
        out["rerr"] = _first_line(r["error"])
    return out


def _below_normal() -> None:
    if os.name == "nt":
        import ctypes  # noqa: PLC0415
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS


def emudump_command(emudump) -> list:
    """How to start an EmuDump: the JIT build through dotnet, the NativeAOT build (tools/emudump/aot.sh) as it is."""
    return [str(emudump)] if str(emudump).lower().endswith(".exe") else ["dotnet", str(emudump)]


def part_timeout(steps: int) -> float:
    """Seconds one EmuDump call over ``steps`` steps may take: about 15 times what it takes on a busy PC."""
    return 120 + 0.1 * steps


def replay_part(mode: str, ids: list, corpus_dir: str, emudump: str, work: str, timeout: float | None = None,
                state_hash: bool = False) -> dict:
    """One worker: EmuDump ``mode`` over these fights (one emulator in its own process), judged here."""
    from .fight_compare import names  # noqa: PLC0415
    from .reanchor_compare import keys_of  # noqa: PLC0415

    c = Corpus(Path(corpus_dir))
    games = [c.read(i) for i in ids]
    timeout = timeout or part_timeout(sum(len(g["steps"]) for g in games))
    nm = names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    src, dst, err = (Path(work) / f"{mode}-{ids[0]}{ext}" for ext in (".in.json", ".out.json", ".err.txt"))
    src.write_text(json.dumps({"results": games}), encoding="utf-8")
    t0 = time.time()
    with dst.open("wb") as so, err.open("wb") as se:
        # --compact: a quarter less CPU than the indented default (older builds ignore it)
        extra = (["--compact"] + (["--state-hash"] if state_hash else [])) if mode == "fight" else []
        try:
            rc = subprocess.run([*emudump_command(emudump), mode, str(src), *extra], stdout=so, stderr=se, check=False,
                                creationflags=BELOW_NORMAL, timeout=timeout).returncode
        except subprocess.TimeoutExpired:  # run() has killed it
            rc = None
    secs = time.time() - t0
    doc, res = {}, []
    if rc is not None:
        with contextlib.suppress(ValueError):
            doc = json.loads(dst.read_text(encoding="utf-8"))
            res = doc.get("results", [])
    problem = None
    if rc is None:
        problem = f"EmuDump {mode} timed out after {timeout:.0f} s over {len(games)} fights (killed)"
    elif rc != 0 or len(res) != len(games):
        problem = f"EmuDump {mode} exit {rc}, {len(res)} of {len(games)} results: " + \
            _first_line(err.read_text(encoding="utf-8", errors="replace"))
    out = {}
    keys = keys_of(doc) if mode == "reanchor" and doc else None
    for n, (fid, g) in enumerate(zip(ids, games)):
        e = res[n] if n < len(res) and isinstance(res[n], dict) else {"error": problem or "no emulator result", "steps": []}
        out[fid] = judge_fight(g, e, nm) if mode == "fight" else judge_reanchor(g, e, nm, keys)
        if state_hash and mode == "fight":
            out[fid]["sh"] = [st.get("state_hash") for st in e.get("steps", [])]
    for f in (src, dst, err):
        with contextlib.suppress(OSError):
            f.unlink()
    return {"mode": mode, "fights": out, "seconds": secs, "problem": problem}


def balance(metas: list, k: int) -> list:
    """Split fights into k parts of about equal steps (longest first, each to the lightest part)."""
    bins = [[0, []] for _ in range(max(1, min(k, len(metas))))]
    for m in sorted(metas, key=lambda m: -m["steps"]):
        b = min(bins, key=lambda b: b[0])
        b[0] += m["steps"]
        b[1].append(m["id"])
    return [ids for _, ids in sorted(bins, key=lambda b: -b[0]) if ids]


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False).stdout.strip()


def judge_sha256(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    files = [*sorted((root / "tools" / "fidelity").glob("*.py")), root / "data" / "id_map.json"]
    for p in files:
        if p.exists():
            h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def emulator_id(emudump: Path) -> dict:
    """The code a run tested: commit, uncommitted changes to the emulator or the judge, the EmuDump binary."""
    dirty = bool(git("status", "--porcelain", "--", "src", "tools/emudump", *JUDGE_PATHS))
    dll = Path(emudump)
    info = {"commit": git("rev-parse", "HEAD"), "dirty": dirty, "judge_sha256": judge_sha256(),
            "emudump_sha256": hashlib.sha256(dll.read_bytes()).hexdigest() if dll.exists() else None}
    # A stale build tests old code: warn when a source file is newer than the binary.
    if dll.exists():
        newest = max((p.stat().st_mtime for d in ("src/Sts2Emulator", "tools/emudump") for pat in ("*.cs", "*.csproj")
                      for p in (ROOT / d).rglob(pat) if "obj" not in p.parts and "bin" not in p.parts), default=0)
        info["stale_build"] = newest > dll.stat().st_mtime
    return info


def totals(fights: dict) -> dict:
    t = collections.Counter()
    for r in fights.values():
        t["fights"] += 1
        n = r["n"]
        t["transitions"] += n - 1
        if r["fd"] is None:
            t["identical"] += 1
        t["plain_verified"] += max(0, lead(r) - 1)
        t["emu_errors"] += 1 if r.get("ferr") or r.get("rerr") else 0
        rv = r.get("rv", "")
        t["reanchored"] += sum(1 for ch in rv if ch in "MDE")
        t["reanchor_matched"] += rv.count("M")
    return dict(t)


def replayable(metas: list, features: set) -> tuple[list, dict]:
    """The fights this EmuDump can replay, and how many it cannot by the features they need."""
    keep, skipped = [], collections.Counter()
    for m in metas:
        miss = [n for n in m.get("needs", []) if n not in features]
        if miss:
            skipped[",".join(miss)] += 1
        else:
            keep.append(m)
    return keep, dict(skipped)


def run(corpus: Corpus, procs: int, emudump: Path, label: str | None = None, only: set | None = None,
        limit: int | None = None, timeout: float | None = None, state_hash: bool = False) -> Path:
    idx = corpus.load_index()
    metas = [m for m in idx["fights"] if not only or m["id"] in only or m["seed"] in only][: limit or None]
    left_out = sum(1 for m in metas if "excluded" in m["flags"])
    metas, skipped = replayable([m for m in metas if "excluded" not in m["flags"]], emudump_features())
    if left_out:
        skipped["excluded"] = left_out
    if not metas:
        raise SystemExit("corpus is empty: run `corpus.py build` first")
    if skipped:
        print(f"skipped, this EmuDump does not apply what they need: {skipped}")
    emu = emulator_id(emudump)
    if emu.get("stale_build"):
        print("warning: emulator sources are newer than EmuDump.dll; rebuild tools/emudump first", file=sys.stderr)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = label or f"{stamp}-{emu['commit'][:7]}{'-dirty' if emu['dirty'] else ''}"
    rdir = corpus.path / "runs" / label
    work = rdir / "work"
    work.mkdir(parents=True, exist_ok=True)
    parts = balance(metas, 2 * procs)  # more parts than processes, so the last ones fill the gaps
    tasks = [(mode, ids) for ids in parts for mode in ("fight", "reanchor")]
    t0 = time.time()
    got = collections.defaultdict(dict)
    problems, busy = [], collections.Counter()
    with ProcessPoolExecutor(max_workers=procs, mp_context=get_context("spawn"), initializer=_below_normal) as ex:
        futs = [ex.submit(replay_part, mode, ids, str(corpus.path), str(emudump), str(work), timeout, state_hash)
                for mode, ids in tasks]
        for f in futs:
            r = f.result()
            busy[r["mode"]] += r["seconds"]
            if r["problem"]:
                problems.append(r["problem"])
            for fid, v in r["fights"].items():
                got[fid].update(v)
    wall = time.time() - t0
    with contextlib.suppress(OSError):
        work.rmdir()
    fights = {m["id"]: {"n": m["steps"], **got[m["id"]]} for m in metas}
    doc = {"format": RESULTS_FORMAT, "label": label, "emulator": emu, "procs": procs, "wall_s": round(wall, 1),
           "subset": bool(only or limit), "skipped": skipped,
           "emudump_s": {k: round(v, 1) for k, v in busy.items()}, "problems": problems[:20], "totals": totals(fights), "fights": fights}
    out = rdir / "results.json"
    out.write_text(json.dumps(doc), encoding="utf-8")
    print(run_line(doc))
    return out


def run_line(doc: dict) -> str:
    t = doc["totals"]
    return (f"corpus run {doc['label']}: {t['fights']} fights on {doc['procs']} processes in {doc['wall_s']} s: "
            f"identical {t.get('identical', 0)}, plain replay verified {t.get('plain_verified', 0)} of "
            f"{t.get('transitions', 0)} transitions, re-anchored {t.get('reanchor_matched', 0)} matched of "
            f"{t.get('reanchored', 0)} tried, emulator errors {t.get('emu_errors', 0)}"
            + (f", skipped {sum(doc['skipped'].values())}" if doc.get("skipped") else "")
            + (f", {len(doc['problems'])} EmuDump problems" if doc.get("problems") else ""))


# ----------------------------------------------------------------------------- baseline / check


def lead(r: dict) -> int:
    """Steps the plain replay reproduced from the start: all of them if identical."""
    return r["n"] if r["fd"] is None else r["fd"]


def _fields(r: dict, j: int) -> set:
    v = r.get("dp", {}).get(str(j))
    return {v} if isinstance(v, str) else set(v or [])


def compare(base: dict, cur: dict) -> dict:
    """Per fight: what got worse since ``base`` (regressions) and what got better (improvements).

    Worse: the plain replay diverges earlier, either replay has an emulator exception it did not have, or
    a re-anchored step that matched does not, a step now throws (E) that did not, or a step that diverged
    now also diverges in another field.
    """
    reg, imp, missing = [], [], []
    for fid, b in base.items():
        c = cur.get(fid)
        if c is None:
            missing.append(fid)
            continue
        rb, rc_ = b.get("rv", ""), c.get("rv", "")
        lost, won, fields = [], [], {}
        for j, (x, y) in enumerate(zip(rb, rc_), 1):
            if (x == "M" and y != "M") or (y == "E" and x != "E"):
                lost.append(j)
            elif x == "D" and y == "D" and _fields(c, j) - _fields(b, j):
                lost.append(j)
                fields[j] = sorted(_fields(c, j) - _fields(b, j))
            elif x != "M" and y == "M":
                won.append(j)
        errors = [(w, c[k]) for w, k in (("plain replay", "ferr"), ("re-anchoring", "rerr")) if c.get(k) and not b.get(k)]
        gone = [w for w, k in (("plain replay", "ferr"), ("re-anchoring", "rerr")) if b.get(k) and not c.get(k)]
        lb, lc = lead(b), lead(c)
        if lc < lb or lost or errors:
            reg.append({"id": fid, "plain": [lb, lc] if lc < lb else None, "steps": lost, "fields": fields,
                        "errors": errors, "was": b, "now": c})
        if lc > lb or won or gone:
            imp.append({"id": fid, "plain": [lb, lc] if lc > lb else None, "steps": won, "fields": {},
                        "errors": [(w, "no exception now") for w in gone], "was": b, "now": c})
    new = [fid for fid in cur if fid not in base]
    return {"regressions": reg, "improvements": imp, "missing": missing, "new": new}


def describe(item: dict, meta: dict, rec: dict | None, worse: bool) -> str:
    was, now = item["was"], item["now"]
    steps = rec.get("steps", []) if rec else []
    bits = []
    if item["plain"]:
        def where(r):
            return "identical" if r["fd"] is None else f"diverged at step {r['fd']} ({r.get('field')} after {r.get('card')})"
        bits.append(f"plain replay {where(was)}, now {where(now)}")
    for w, msg in item.get("errors", []):
        bits.append(f"{w}: {'new emulator exception: ' if worse else ''}{msg}")
    if item["steps"]:
        shown = []
        for j in item["steps"][:6]:
            f = item.get("fields", {}).get(j)
            if f:
                field = "also " + ",".join(f)
            else:
                v = (now if worse else was).get("dp", {}).get(str(j), "")
                field = ",".join(v) if isinstance(v, list) else v
            letters = f"{was.get('rv', '')[j - 1:j]}>{now.get('rv', '')[j - 1:j]}"
            shown.append(f"{j} {card_of(steps[j] if j < len(steps) else None, j)} {letters}{(' ' + field) if field else ''}")
        more = f" +{len(item['steps']) - 6}" if len(item["steps"]) > 6 else ""
        bits.append(f"re-anchored steps {'worse' if worse else 'now matching'}: {', '.join(shown)}{more}")
    marks = meta.get("flags", []) + meta.get("tags", [])
    flags = f" [{','.join(marks)}]" if marks else ""
    src = (meta.get("sources") or ["?"])[0]
    return f"{item['id']} {meta.get('seed')} ({src}, format {meta.get('format')}){flags}: " + "; ".join(bits)


def baseline_for(corpus: Corpus, ref: str | None) -> Path | None:
    bdir = corpus.path / "baselines"
    if ref:
        p = Path(ref)
        if p.exists():
            return p
        hits = sorted(bdir.glob(f"{ref}*.json"))
        return hits[0] if hits else None
    for c in git("rev-list", "--max-count=2000", "HEAD").split():
        p = bdir / f"{c}.json"
        if p.exists():
            return p
    return None


def store_baseline(corpus: Corpus, run_doc: dict, force: bool) -> Path:
    emu = run_doc["emulator"]
    if run_doc.get("format", 1) < RESULTS_FORMAT:
        raise SystemExit("the run predates this corpus.py's results format; run it again")
    refusals = []
    if emu["dirty"]:
        refusals.append("the emulator or the judge has uncommitted changes; commit first so the baseline names its code")
    if emu.get("stale_build"):
        refusals.append("EmuDump.dll is older than the emulator sources; rebuild tools/emudump first")
    if run_doc.get("subset"):
        refusals.append("the run covers part of the corpus (--only/--limit); a baseline must cover all of it")
    if run_doc.get("problems"):
        refusals.append(f"EmuDump failed in this run ({run_doc['problems'][0]})")
    if refusals and not force:
        raise SystemExit("; ".join(refusals) + " (or --force)")
    bdir = corpus.path / "baselines"
    bdir.mkdir(parents=True, exist_ok=True)
    keep = ("n", "fd", "field", "card", "rv", "dp", "ferr", "rerr")
    doc = {"format": RESULTS_FORMAT, "commit": emu["commit"], "made": datetime.datetime.now().isoformat(timespec="seconds"),
           "run": run_doc["label"], "emudump_sha256": emu["emudump_sha256"], "judge_sha256": emu.get("judge_sha256"),
           "skipped": run_doc.get("skipped", {}), "totals": run_doc["totals"],
           "fights": {fid: {k: v for k, v in r.items() if k in keep} for fid, r in run_doc["fights"].items()}}
    p = bdir / f"{emu['commit']}.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def check(corpus: Corpus, run_doc: dict, base_path: Path, show: int = 40) -> int:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("format", 1) < RESULTS_FORMAT or run_doc.get("format", 1) < RESULTS_FORMAT:
        print(f"check: baseline {base_path.name} or the run predates results format {RESULTS_FORMAT}; "
              "store a new baseline with `corpus.py baseline` on the baseline's commit")
        return 2
    subset = run_doc.get("subset", False)
    metas = {m["id"]: m for m in corpus.load_index()["fights"]}
    # FIGHT_POLICY's fights are left out of both sides, and the totals counted over the rest.
    out = {fid for fid, m in metas.items() if "excluded" in m["flags"]}
    base_fights = {fid: r for fid, r in base["fights"].items() if fid not in out}
    cur_fights = {fid: r for fid, r in run_doc["fights"].items() if fid not in out}
    res = compare(base_fights, cur_fights)
    bt = totals(base_fights) if len(base_fights) < len(base["fights"]) else base["totals"]
    ct = totals(cur_fights) if len(cur_fights) < len(run_doc["fights"]) else run_doc["totals"]
    print(f"baseline {base['commit'][:10]} ({base['run']}): identical {bt.get('identical', 0)}, "
          f"re-anchored matched {bt.get('reanchor_matched', 0)}")
    left = sorted(fid for fid in out if fid in base["fights"] or fid in run_doc["fights"])
    if left:
        print(f"excluded (corpus.FIGHT_POLICY), compared on neither side: {len(left)} fights")
        for fid in left[:show]:
            print(f"  {fid} {metas[fid].get('seed')}: {metas[fid]['excluded']}")
    if base.get("judge_sha256") and base["judge_sha256"] != run_doc["emulator"].get("judge_sha256"):
        print("note: the judge (tools/fidelity, data/id_map.json) changed since the baseline; a difference may be the judge's")
    for title, items, worse in (("REGRESSIONS", res["regressions"], True), ("IMPROVEMENTS", res["improvements"], False)):
        print(f"{title}: {len(items)} fights")
        for it in items[:show]:
            rec = None
            with contextlib.suppress(OSError, ValueError):
                rec = corpus.read(it["id"])
            print("  " + describe(it, metas.get(it["id"], {}), rec, worse))
        if len(items) > show:
            print(f"  ... {len(items) - show} more in check.json")
    if res["new"]:
        print(f"new fights (no baseline yet): {len(res['new'])}")
    missing_fail = bool(res["missing"]) and not subset
    if res["missing"]:
        if subset:
            print(f"baseline fights not in this partial run (--only/--limit): {len(res['missing'])}")
        else:
            why = collections.Counter("not in the corpus" if fid not in metas else
                                      ("needs " + ",".join(metas[fid].get("needs", [])) if metas[fid].get("needs") else "not replayed")
                                      for fid in res["missing"])
            print(f"NOT COMPARED: {len(res['missing'])} baseline fights are missing from this run ({dict(why)}); "
                  "a full check compares every one")
    rdir = corpus.path / "runs" / run_doc["label"]
    if rdir.exists():
        slim = {k: [{x: y for x, y in it.items() if x not in ("was", "now")} for it in v] if k in ("regressions", "improvements") else v
                for k, v in res.items()}
        (rdir / "check.json").write_text(json.dumps({"baseline": base["commit"], **slim}), encoding="utf-8")
    lost = sum(len(it["steps"]) for it in res["regressions"])
    # A check that compared nothing, ran stale code or lost fights to EmuDump failures vouches for nothing.
    compared = len(base["fights"]) - len(res["missing"])
    unsound = []
    if compared == 0:
        unsound.append("no baseline fight was compared")
    if run_doc.get("emulator", {}).get("stale_build"):
        unsound.append("EmuDump.dll is older than the emulator sources (rebuild tools/emudump)")
    if run_doc.get("problems"):
        unsound.append(f"EmuDump failed in this run ({len(run_doc['problems'])} problems, first: {run_doc['problems'][0][:120]})")
    for u in unsound:
        print(f"UNSOUND: {u}")
    failed = bool(res["regressions"]) or missing_fail or bool(unsound)
    print(f"check: {'FAIL' if failed else 'ok'}: {len(res['regressions'])} regressed fights ({lost} re-anchored steps), "
          f"{len(res['missing']) if missing_fail else 0} not compared, {len(res['improvements'])} improved, "
          f"{compared} compared{', unsound' if unsound else ''}; "
          f"identical {bt.get('identical', 0)} -> {ct.get('identical', 0)}, "
          f"re-anchored matched {bt.get('reanchor_matched', 0)} -> {ct.get('reanchor_matched', 0)}")
    return 1 if failed else 0


# ------------------------------------------------------------------------- identity (byte-safety)


def hash_part(ids: list, corpus_dir: str, emudump: str, work: str, fields: bool = False, timeout: float | None = None) -> dict:
    """One worker: `EmuDump fight --state-hash` over these fights; per fight its step hashes (and field hashes)."""
    c = Corpus(Path(corpus_dir))
    games = [c.read(i) for i in ids]
    timeout = timeout or part_timeout(sum(len(g["steps"]) for g in games))
    tag = hashlib.sha256((emudump + ids[0]).encode()).hexdigest()[:10]
    src, dst = Path(work) / f"hash-{tag}.in.json", Path(work) / f"hash-{tag}.out.json"
    src.write_text(json.dumps({"results": games}), encoding="utf-8")
    flag = "--state-hash=fields" if fields else "--state-hash"
    res, problem = [], None
    try:
        with dst.open("wb") as so:
            rc = subprocess.run(["dotnet", emudump, "fight", str(src), "--compact", flag], stdout=so, stderr=subprocess.DEVNULL,
                                check=False, creationflags=BELOW_NORMAL, timeout=timeout).returncode
        res = json.loads(dst.read_text(encoding="utf-8")).get("results", [])
        if rc != 0 or len(res) != len(games):
            problem = f"EmuDump fight exit {rc}, {len(res)} of {len(games)} results"
    except (subprocess.TimeoutExpired, ValueError) as ex:
        problem = f"EmuDump fight {type(ex).__name__}"
    for f in (src, dst):
        with contextlib.suppress(OSError):
            f.unlink()
    out = {}
    for n, fid in enumerate(ids):
        e = res[n] if n < len(res) else {"error": problem or "no result", "steps": []}
        out[fid] = {"sh": [st.get("state_hash") for st in e.get("steps", [])], "err": _first_line(e.get("error"))}
        if fields:
            out[fid]["fields"] = [st.get("state_fields") for st in e.get("steps", [])]
    return {"fights": out, "problem": problem}


def hash_run(corpus: Corpus, ids: list, emudump: Path, procs: int, work: Path, fields: bool = False) -> dict:
    metas = {m["id"]: m for m in corpus.load_index()["fights"]}
    parts = balance([metas[i] for i in ids], 2 * procs)
    got, problems = {}, []
    with ProcessPoolExecutor(max_workers=procs, mp_context=get_context("spawn"), initializer=_below_normal) as ex:
        for r in [f.result() for f in [ex.submit(hash_part, p, str(corpus.path), str(emudump), str(work), fields) for p in parts]]:
            got.update(r["fights"])
            if r["problem"]:
                problems.append(r["problem"])
    return {"fights": got, "problems": problems}


def compare_hashes(a: dict, b: dict) -> dict:
    """Per fight: the first step whose full-state hash differs, or a replay error on one side only."""
    res = {"same": 0, "differ": [], "errors": [], "missing": []}
    for fid, x in a.items():
        y = b.get(fid)
        if y is None or (not y.get("sh") and not y.get("err")):
            res["missing"].append(fid)
            continue
        if bool(x.get("err")) != bool(y.get("err")) or (x.get("err") and x["err"] != y["err"]):
            res["errors"].append((fid, x.get("err"), y.get("err")))
            continue
        sa, sb = x.get("sh") or [], y.get("sh") or []
        if None in sa or None in sb:
            res["missing"].append(fid)  # a build without --state-hash
            continue
        step = next((j for j, (p, q) in enumerate(zip(sa, sb)) if p != q), None)
        if step is None and len(sa) != len(sb):
            step = min(len(sa), len(sb))
        if step is None:
            res["same"] += 1
        else:
            res["differ"].append((fid, step))
    return res


def identity(corpus: Corpus, a: dict) -> int:
    """Two builds (or two stored runs) agree on every field of the state after every step of every fight."""
    if a.runs:
        ra, rb = (json.loads(Path(p).read_text(encoding="utf-8"))["fights"] for p in a.runs)
        res = compare_hashes(ra, rb)
        explain = {}
    else:
        if not (a.a and a.b):
            raise SystemExit("identity: --a and --b (two EmuDump.dll builds), or --runs A B")
        idx = corpus.load_index()["fights"]
        metas = [m for m in idx if not a.only or m["id"] in a.only or m["seed"] in a.only]
        metas, skipped = replayable(metas[: a.limit or None], emudump_features())
        ids = [m["id"] for m in metas]
        work = corpus.path / "runs" / f"identity-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        work.mkdir(parents=True, exist_ok=True)
        ha, hb = (hash_run(corpus, ids, Path(p), a.procs, work) for p in (a.a, a.b))
        res = compare_hashes(ha["fights"], hb["fights"])
        for side, h in (("a", ha), ("b", hb)):
            for p in h["problems"][:3]:
                print(f"EmuDump problem ({side}): {p}")
        res["problems"] = ha["problems"] + hb["problems"]
        explain = {}
        if res["differ"] and a.explain:
            # The fields behind the first differing step, from both builds' per-field hashes.
            few = [fid for fid, _ in res["differ"][: a.explain]]
            fa, fb = (hash_run(corpus, few, Path(p), min(a.procs, len(few)), work, fields=True)["fights"] for p in (a.a, a.b))
            for fid, step in res["differ"][: a.explain]:
                xa = (fa.get(fid, {}).get("fields") or [None] * (step + 1))
                xb = (fb.get(fid, {}).get("fields") or [None] * (step + 1))
                pa = xa[step] if step < len(xa) else None
                pb = xb[step] if step < len(xb) else None
                explain[fid] = sorted(k for k in set(pa or {}) | set(pb or {}) if (pa or {}).get(k) != (pb or {}).get(k))
        with contextlib.suppress(OSError):
            work.rmdir()
        if skipped:
            print(f"skipped, this EmuDump does not apply what they need: {skipped}")
    metas = {m["id"]: m for m in corpus.load_index()["fights"]}
    for fid, step in res["differ"][: a.show]:
        m = metas.get(fid, {})
        why = f": fields {', '.join(explain[fid][:8])}" if explain.get(fid) else ""
        print(f"  {fid} {m.get('seed')}: state differs from step {step}{why}")
    for fid, ea, eb in res["errors"][: a.show]:
        print(f"  {fid}: replay error a: {ea or 'none'} | b: {eb or 'none'}")
    bad = len(res["differ"]) + len(res["errors"]) + len(res["missing"]) + len(res.get("problems", []))
    total = res["same"] + len(res["differ"]) + len(res["errors"]) + len(res["missing"])
    print(f"identity: {'FAIL' if bad or not total else 'ok'}: {total} fights, {res['same']} identical in every field at every step, "
          f"{len(res['differ'])} differ, {len(res['errors'])} replay errors on one side, {len(res['missing'])} without hashes")
    return 1 if bad or not total else 0


def status(corpus: Corpus) -> None:
    idx = corpus.load_index()
    fs = idx["fights"]
    by_fmt = collections.Counter(m["format"] for m in fs)
    flags = collections.Counter(f for m in fs for f in m["flags"])
    tags = collections.Counter(t for m in fs for t in m.get("tags", []))
    needs = collections.Counter(n for m in fs for n in m.get("needs", []))
    builds = collections.Counter(m["build"][:8] for m in fs)
    size = sum(p.stat().st_size for p in corpus.fights_dir.glob("*.json.gz")) if corpus.fights_dir.exists() else 0
    print(f"corpus {corpus.path}: {len(fs)} fights, {sum(m['steps'] for m in fs)} steps, "
          f"{size / 1e6:.1f} MB, from {len(idx.get('scanned', {}))} scanned files")
    print(f"  formats {dict(sorted(by_fmt.items()))}, flags {dict(flags)}, tags {dict(tags)}, builds {dict(builds)}")
    if needs:
        have = emudump_features()
        print("  need an EmuDump feature: " + ", ".join(f"{n} {c} ({'replayed' if n in have else 'skipped: not in this EmuDump'})"
                                                         for n, c in needs.items()))
    for name, e in sorted(idx.get("excluded", {}).items()):
        print(f"  excluded {name} ({e['files']} files): {e['reason']}")
    for m in fs:
        if m.get("excluded"):
            print(f"  excluded fight {m['id']} {m.get('seed')}: {m['excluded']}")
    bl = sorted((corpus.path / "baselines").glob("*.json"))
    print(f"  baselines: {', '.join(p.stem[:10] for p in bl) or 'none'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="add every *.out.json under the jobs folder and its jobs-* siblings")
    b.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    b.add_argument("--rescan", action="store_true", help="read every file, changed or not")
    ad = sub.add_parser("add", help="add referee out.json files")
    ad.add_argument("files", nargs="+", type=Path)
    for name in ("run", "baseline", "check"):
        p = sub.add_parser(name)
        p.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
        p.add_argument("--emudump", type=Path, default=EMUDUMP)
        p.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
        p.add_argument("--no-scan", action="store_true", help="do not pick up new recordings from the jobs folders first")
        p.add_argument("--only", nargs="*", help="fight ids or seeds")
        p.add_argument("--limit", type=int)
        p.add_argument("--label")
        p.add_argument("--timeout", type=float, help="seconds per EmuDump call (default: from its steps)")
        if name == "run":
            p.add_argument("--state-hash", action="store_true",
                           help="also store each step's full-state hash (EmuDump fight --state-hash), for `identity --runs`")
        if name != "run":
            p.add_argument("--run", type=Path, help="an existing run's results.json instead of a new run")
        if name == "baseline":
            p.add_argument("--force", action="store_true")
        if name == "check":
            p.add_argument("--baseline", help="a baseline file or commit (default: this commit or its nearest ancestor with one)")
            p.add_argument("--show", type=int, default=40)
    sub.add_parser("status")
    idn = sub.add_parser("identity", help="two builds agree on every state field after every step (byte-safety of a refactor)")
    idn.add_argument("--a", help="EmuDump.dll of the old build")
    idn.add_argument("--b", help="EmuDump.dll of the new build")
    idn.add_argument("--runs", nargs=2, type=Path, help="two `run --state-hash` results.json instead of running builds")
    idn.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    idn.add_argument("--only", nargs="*", help="fight ids or seeds")
    idn.add_argument("--limit", type=int)
    idn.add_argument("--explain", type=int, default=5, help="name the differing fields of the first N differing fights")
    idn.add_argument("--show", type=int, default=40)
    a = ap.parse_args(argv)
    corpus = Corpus(a.corpus)
    if a.cmd == "build":
        if a.rescan:
            with corpus.lock():
                idx = corpus.load_index()
                idx["scanned"] = {}
                corpus.save_index(idx)
        print("build:", corpus.scan(a.jobs))
        status(corpus)
        return 0
    if a.cmd == "add":
        print("add:", corpus.add_files(a.files))
        return 0
    if a.cmd == "status":
        status(corpus)
        return 0
    if a.cmd == "identity":
        return identity(corpus, a)
    base_path = None
    if a.cmd == "check":
        base_path = baseline_for(corpus, a.baseline)
        if base_path is None:
            print("check: no baseline for this commit or its ancestors; store one with `corpus.py baseline` on main")
            return 2
    if getattr(a, "run", None):
        run_path = a.run
    else:
        if not a.no_scan and a.jobs.exists():
            got = corpus.scan(a.jobs, quiet=True)
            n_new = got.get("new", 0) + got.get("truncated_new", 0)
            if n_new:
                print(f"picked up {n_new} new fights from {got.get('files', 0)} new or changed recordings")
        run_path = run(corpus, a.procs, a.emudump, a.label, set(a.only or []), a.limit, a.timeout,
                       state_hash=getattr(a, "state_hash", False))
    run_doc = json.loads(Path(run_path).read_text(encoding="utf-8"))
    if a.cmd == "baseline":
        print("baseline:", store_baseline(corpus, run_doc, a.force).name)
        return 0
    if a.cmd == "check":
        return check(corpus, run_doc, base_path, a.show)
    return 0


if __name__ == "__main__":
    from tools.fidelity import corpus as _corpus  # noqa: PLW0406  the package copy, whose functions pool workers can import

    sys.exit(_corpus.main())
