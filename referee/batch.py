#!/usr/bin/env python3
"""One fidelity batch end to end: make fights, play them in the game, replay them in the emulator, compare.

    python referee/batch.py --name <name> --fights 100 [--start 0] [--parallel 6] [--chunk 48] [--timeout 5400]
                            [--game <copy>] [--jobs <dir>] [--coverage <file>]

Run from the fork's root. One pool of long-lived game servers (serve.Pool) plays the whole batch, with
no barrier anywhere: the inbox is kept topped up with fights whose decks are drawn from the coverage
as it is at that moment; finished fights are replayed in the emulator in chunks of about --chunk
(EmuDump fight and reanchor, split across processes) on a background thread and compared in
process, and each chunk updates coverage while the servers go on playing. The canary (--canary fights per server) is the first chunk; a
systemic failure there stops the batch.

Writes <jobs>/<name>.{json,out.json,emu.json,reanchor.json,summary.txt}, <name>.packs/ (context_pack)
and a metrics.v2.csv row, and prints the summary: fights identical, fights diverging grouped by the
first differing field, and errors. Results are in spec order. Seeds, Neow choices and chooser seeds
are deterministic from --start; decks depend on the coverage when each fight was drawn, and
<name>.json records the specs as played, so `serve.py --specs <name>.json` re-runs a batch exactly.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NL = chr(10)
DEFAULT_JOBS = Path(r"D:\spireline\referee\jobs")
DEFAULT_GAME = Path(r"D:\spireline\referee\game")
ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
EMUDUMP = ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"

sys.path.insert(0, str(ROOT / "referee"))
import serve  # noqa: E402


def canary_failure(results: list) -> str:
    """Why the canary says the setup is broken, or "" when it is fine. Systemic only: a stalled fight
    is recorded and skipped, so a few are tolerated; any other error type, or more than a quarter of
    the canary failing, stops the batch."""
    bad = [r for r in results if r.get("error")]
    other = [r for r in bad if not str(r.get("error", "")).startswith("stalled")]
    if other or len(bad) > len(results) // 4:
        return f"{len(bad)} of {len(results)} fights with errors"
    return ""


class Stream:
    """Plays `fights` specs on one pool and replays finished fights in chunks, with no barrier.

    draw(n, offset) makes n specs (offset = how many were drawn before); it is called as the inbox
    has room, so each deck sees the coverage of every chunk finished by then. process(tag, ids,
    games) replays one chunk on a worker thread and returns {"fights": {id: data}, ...}; the next
    chunk starts when it is done and takes every fight finished meanwhile (at least `chunk`; near
    the end, at least as many as are still to play). The canary fights are the first chunk and are checked before it starts.
    on_chunk(tag, ids, result) runs on this thread after each chunk. run() returns "" when every fight
    was played and replayed, else why the batch stopped early ("canary failed: ...", the pool's
    reason, "network connection ...", "timeout"); after an early stop the pool is closed, unplayed
    specs become "no result" errors and what was played is still replayed, except after a failed
    canary, which stops at once as before.
    """

    def __init__(self, pool, draw, fights: int, process, *, canary: int = 0, chunk: int = 48,
                 timeout: float = 5400, on_chunk=None, tick: float = 0.25):
        self.pool, self.draw, self.fights, self.process = pool, draw, fights, process
        self.canary_n = canary if canary and fights > canary else 0
        self.chunk, self.timeout, self.on_chunk, self.tick = max(1, chunk), timeout, on_chunk, tick
        self.specs: list = []
        self.ids: list[str] = []
        self.canary_ids: list[str] = []
        self.canary_bad: list = []
        self.canary_ok = not self.canary_n
        self.finished: dict[str, dict] = {}  # id -> result, timing stripped
        self.ready: list[str] = []  # finished, not yet replayed (arrival order)
        self.data: dict[str, dict] = {}  # id -> what process() returned for it
        self.chunks: list[tuple[str, list]] = []  # (tag, ids) in the order they were replayed
        self.reason = ""
        self.t0 = time.time()

    def run(self) -> str:
        with ThreadPoolExecutor(max_workers=1) as ex:
            try:
                self._run(ex)
            finally:
                if not self.pool.closed:
                    self._take(self.pool.close(abandon=True))
        return self.reason

    def _run(self, ex: ThreadPoolExecutor) -> None:
        if self.canary_n:
            self.canary_ids = self._submit(self.canary_n)
        self._top_up()
        self.pool.start()
        fut: Future | None = None
        while True:
            self._take(self.pool.poll())
            if self.pool.failed:
                self.reason = self.pool.failed
                break
            if self.pool.seen_net:
                self.reason = f"network connection {sorted(self.pool.seen_net)}"
                break
            if not self.canary_ok and all(i in self.finished for i in self.canary_ids):
                games = [self.finished[i] for i in self.canary_ids]
                why = canary_failure(games)
                if why:
                    self.canary_bad = [r for r in games if r.get("error")]
                    self.reason = "canary failed: " + why
                    return
                self.canary_ok = True
                fut = self._start(ex, "canary", self.canary_ids)  # nothing else can have started
            fut = self._finish(fut, wait=False)
            # Near the end a chunk also goes once it is at least as big as what is still to play, so the
            # chunks halve towards the end and the replay left after the last fight is small (not one
            # dotnet start per fight: each chunk costs two process starts).
            if fut is None and self.canary_ok and self.ready and (
                    len(self.ready) >= min(self.chunk, max(1, self.fights - len(self.finished)))):
                fut = self._start(ex, f"c{len(self.chunks)}", list(self.ready))
            if len(self.finished) >= self.fights:
                break
            if time.time() - self.t0 > self.timeout:
                self.reason = "timeout"
                break
            self._top_up()
            time.sleep(self.tick)
        # Every fight is in (or the batch stopped): let the servers go, then replay the rest.
        self._take(self.pool.close(abandon=bool(self.reason)))
        for rid in self.ids:
            if rid not in self.finished:
                self.finished[rid] = {"seed": self.pool.specs[rid].get("seed"), "steps": [], "error": "no result"}
                self.ready.append(rid)
        fut = self._finish(fut, wait=True)
        if self.ready:
            self._finish(self._start(ex, f"c{len(self.chunks)}", list(self.ready)), wait=True)

    def _submit(self, n: int) -> list[str]:
        new = self.draw(n, len(self.specs))
        self.specs += new
        ids = self.pool.submit(new)
        self.ids += ids
        self.pool.more = self.fights - len(self.specs)
        return ids

    def _top_up(self) -> None:
        # One spec at a time, only as the inbox has room, so every deck is drawn as late as possible.
        while len(self.specs) < self.fights and self.pool.unclaimed < self.pool.window:
            self._submit(1)

    def _take(self, new: list) -> None:
        for rid, r in new:
            if rid not in self.finished:
                self.finished[rid] = serve.strip(r)
                self.ready.append(rid)

    def _start(self, ex: ThreadPoolExecutor, tag: str, ids: list[str]) -> Future:
        taken = set(ids)
        self.ready = [i for i in self.ready if i not in taken]
        self.chunks.append((tag, ids))
        fut = ex.submit(self.process, tag, ids, [self.finished[i] for i in ids])
        fut.tag, fut.ids = tag, ids
        return fut

    def _finish(self, fut: Future | None, wait: bool) -> Future | None:
        if fut is None or (not wait and not fut.done()):
            return fut
        res = fut.result()  # a worker exception ends the batch here (the pool is closed on the way out)
        self.data.update(res["fights"])
        if self.on_chunk:
            self.on_chunk(fut.tag, fut.ids, res)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--fights", type=int, default=100)
    ap.add_argument("--start", type=int, default=0, help="seed stream offset")
    ap.add_argument("--timeout", type=float, default=5400, help="whole batch, seconds")
    ap.add_argument("--parallel", type=int, default=6, help="game servers running at once")
    ap.add_argument("--deck-cards", type=int, default=5, help="random Ironclad/colourless cards added per fight")
    ap.add_argument("--canary", type=int, default=2, help="fights per server checked before the rest is replayed (0 = off)")
    ap.add_argument("--chunk", type=int, default=48, help="finished fights per emulator replay; coverage is updated after each")
    ap.add_argument("--waves", type=int, default=None, help=argparse.SUPPRESS)  # gone: the batch streams; still accepted
    ap.add_argument("--max-restarts", type=int, default=None, help="server restarts before giving up (default 3*parallel+3+fights/50)")
    ap.add_argument("--game", type=Path, default=DEFAULT_GAME, help="game copy the servers run")
    ap.add_argument("--jobs", type=Path, default=DEFAULT_JOBS, help="folder for the job, out, emu and summary files")
    ap.add_argument("--coverage", type=Path, default=None, help="coverage file (default <jobs>/../coverage.json)")
    ap.add_argument("--settle-audit", type=float, default=0.05,
                    help="share of fights whose flag settles are re-checked against the old six-frame rule")
    ap.add_argument("--old-settle", action="store_true",
                    help="six-stable-frame settle, awaited tweens and stock frame pacing (before SPEED_PLAN steps 3-4)")
    a = ap.parse_args()
    JOBS, GAME = a.jobs, a.game.resolve()
    JOBS.mkdir(parents=True, exist_ok=True)

    # Coverage: each fight's deck gets random cards from the pools an Ironclad can hold, so a
    # batch exercises hundreds of cards rather than the starter deck's three.
    pools = json.loads(subprocess.run(["dotnet", str(EMUDUMP), "pools"], capture_output=True, text=True, check=True).stdout)
    id_map = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))["cards"]
    sys.path.insert(0, str(ROOT))
    from tools.fidelity.fight_compare import snake  # noqa: PLC0415
    names = {v: snake(k) for k, v in id_map.items()}
    candidates = sorted({names[i] for i in pools["ironclad"] + pools["colorless"] if i in names})

    t_start = time.time()
    from tools.fidelity import coverage as cv  # noqa: PLC0415
    from tools.fidelity import fight_compare as fc  # noqa: PLC0415
    from tools.fidelity import reanchor_compare as rc  # noqa: PLC0415
    n_emu = max(1, min(16, os.cpu_count() or 1))
    nm = fc.names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cov_path = a.coverage or JOBS.parent / "coverage.json"
    cov = cv.load(cov_path)
    cov_lock = threading.Lock()  # the worker learns into cov while this thread draws decks from it
    rng = random.Random(a.start)

    def make_runs(n: int, offset: int) -> list:
        # Decks favour untested cards and leave out cards quarantined as suspect at this commit.
        with cov_lock:
            return [{"kind": "combat", "seed": "".join(rng.choice(ALPHABET) for _ in range(10)), "character": "IRONCLAD",
                     "ascension": 10, "neow": (offset + i) % 3, "chooser_seed": a.start * 1000 + offset + i,
                     "max_actions": 150, "deck_add": cv.pick(cov, candidates, a.deck_cards, rng, commit)}
                    for i in range(n)]

    job, out, emu = (JOBS / f"{a.name}{ext}" for ext in (".json", ".out.json", ".emu.json"))
    rea_path = JOBS / f"{a.name}.reanchor.json"
    run_dir = GAME.parent / "serve" / a.name
    if run_dir.exists():  # a pool needs a fresh run folder (old spec and result files would mix in)
        run_dir = run_dir.with_name(f"{a.name}.{time.strftime('%Y%m%d-%H%M%S')}")
    n_servers = max(1, min(a.parallel, a.fights))
    # Driver speed switches for every server (SPEED_PLAN.md steps 2-4), the same defaults as serve.py's
    # command line (serve.job_options): the flag settle reads the game's turn flags, awaited tweens finish
    # at once, frames are not padded to 6.9 ms, and a share of fights is settle-audited.
    jopts = serve.job_options(None, old_settle=a.old_settle, settle_audit=None if a.old_settle else a.settle_audit)
    pool = serve.Pool(GAME, run_dir, servers=n_servers, spares=1, jopts=jopts, echo=None,
                      max_restarts=a.max_restarts if a.max_restarts is not None else 3 * n_servers + 3 + a.fights // 50)
    serve.rr.install(GAME, ROOT / "referee" / "driver" / "bin" / "Release" / "net9.0")  # once, before any server starts
    chunk_dir = pool.run_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rea_meta: dict = {}

    def replay(tag: str, games: list) -> tuple[list, dict]:
        """EmuDump fight (the plain replay) and reanchor (each step from the game's own state) of one
        chunk, each split across processes (one emulator per process), all started at once; a fight's
        replay does not depend on the split. A part that produced fewer results is padded."""
        k = max(1, min(max(1, n_emu // 2), -(-len(games) // 12)))
        size = -(-len(games) // k)
        parts = [games[i:i + size] for i in range(0, len(games), size)]
        jobs = []
        for mode in ("fight", "reanchor"):
            for i, part in enumerate(parts):
                src, dst = chunk_dir / f"{tag}.{mode}.p{i}.in.json", chunk_dir / f"{tag}.{mode}.p{i}.json"
                src.write_text(json.dumps({"results": part}), encoding="utf-8")
                fh = open(dst, "w", encoding="utf-8")
                jobs.append((mode, len(part), src, dst, fh, subprocess.Popen(["dotnet", str(EMUDUMP), mode, str(src)], stdout=fh)))
        docs: dict = {"fight": {}, "reanchor": {}}
        for mode, n_in, src, dst, fh, p in jobs:
            p.wait()
            fh.close()
            part_doc = json.loads(dst.read_text(encoding="utf-8")) if dst.stat().st_size else {"results": []}
            doc = docs[mode]
            if not doc:
                doc.update({k_: v for k_, v in part_doc.items() if k_ != "results"}, results=[])
            got = part_doc.get("results", [])
            doc["results"] += got + [{"error": "emulator produced no result"}] * (n_in - len(got))
            src.unlink()
            dst.unlink()
        return docs["fight"]["results"], docs["reanchor"]

    def process(tag: str, ids: list, games: list) -> dict:
        """Replay one chunk, compare it fight by fight, fold it into coverage (worker thread)."""
        emus, rea = replay(tag, games)
        rea_meta.update({k: v for k, v in rea.items() if k != "results"})
        keys = rc.keys_of(rea)
        fights = {}
        for rid, g, e, r in zip(ids, games, emus, rea["results"]):
            # Per fight: identical, or the first differing field and the action before it.
            seed, ok = g.get("seed"), bool(g.get("steps")) and not g.get("error") and not e.get("error")
            f = {"emu": json.dumps(e), "rea": r, "analysis": rc.analyse(g, r, nm, "stop", keys), "ok": ok,
                 "first": None, "line": None, "group": None,
                 "errors": [f"{seed}: {side} error: {str(x['error'])[:200]}" for side, x in (("game", g), ("emulator", e))
                            if x.get("error")]}
            if ok:
                f["first"] = idx = cv.first_divergence(g, e, nm)
                if idx is None:
                    f["line"] = f"{seed}: {len(g['steps'])} steps identical"
                else:
                    act = g["steps"][idx].get("action")
                    d = fc.diff(fc.game_view(g["steps"][idx]["state"]), fc.emu_view(e["steps"][idx]["state"], nm))
                    field = rc.group(d[0][0]) if d else "(length)"
                    card = (act or {}).get("card") or ("end_turn" if act else "start")
                    head = f"{seed}: first difference at step {idx} after {act}"
                    f["group"] = (f"{field} after {card}", head[:200])
                    f["line"] = head + "".join(f"{NL}    {p}: game {str(x)[:150]} | emulator {str(y)[:150]}" for p, x, y in d[:3])
            fights[rid] = f
        # Coverage learns from every re-anchored step (SPEED_PLAN step 6), not only those before
        # the first divergence; update_reanchored replaces coverage.update.
        with cov_lock:
            cv.update_reanchored(cov, [f["analysis"] for f in fights.values()], commit)
            cv.save(cov_path, cov)
            cov_line = cv.summary(cov, candidates, commit)
        same = sum(1 for f in fights.values() if f["ok"] and f["first"] is None)
        return {"fights": fights, "identical": same, "coverage": cov_line}

    def on_chunk(tag: str, ids: list, res: dict) -> None:
        if tag == "canary":
            print(f"canary ok: {len(ids)} fights, no errors, {res['identical']} identical; {res['coverage']}", flush=True)
        else:
            print(f"chunk {tag[1:]} at {time.time() - t_start:.0f} s ({len(stream.finished)} of {a.fights} played): "
                  f"{len(ids)} fights, {res['identical']} identical; {res['coverage']}", flush=True)

    stream = Stream(pool, make_runs, a.fights, process, canary=a.canary * n_servers, chunk=a.chunk,
                    timeout=a.timeout, on_chunk=on_chunk)
    reason = stream.run()
    psum = pool.summary()
    pool.write_run_files(psum)
    if reason.startswith("canary"):
        print(f"{reason}; pool restarts {pool.restarts}, network {psum['network_connections']}")
        for r in stream.canary_bad[:5]:
            print("  ", r.get("seed"), str(r.get("error"))[:300])
        return 3

    t_end = time.time()  # every fight played and replayed; what follows is reporting
    # The batch's files, in spec order, from what the chunks computed (no second replay).
    ids = stream.ids
    merged = [stream.finished[i] for i in ids]
    data = stream.data
    job.write_text(json.dumps({"out": str(out).replace("\\", "/"), "runs": stream.specs}), encoding="utf-8")
    out.write_text(json.dumps({"results": merged}), encoding="utf-8")
    with open(emu, "w", encoding="utf-8") as f:
        f.write('{"results": [' + ",".join(data[i]["emu"] for i in ids) + "]}")
    rea_path.write_text(json.dumps({**rea_meta, "results": [data[i]["rea"] for i in ids]}), encoding="utf-8")
    pool_ok = not pool.failed and psum["results"] == psum["fights"] and not pool.seen_net and not pool.profile_touched
    report = {"exit": [0 if pool_ok else 1], "seconds": psum["wall_s"], "network_connections": psum["network_connections"],
              "real_profile_touched": psum["real_profile_touched"]}

    same, groups, errors = 0, collections.defaultdict(list), []
    for i in ids:
        d = data[i]
        errors += d["errors"]
        same += d["ok"] and d["first"] is None
        if d["group"]:
            groups[d["group"][0]].append(d["group"][1])
    cmp = NL.join(data[i]["line"] for i in ids if data[i]["line"])
    # Speed metrics. Headline: verified fights per minute = fights played in the game AND
    # replayed and compared in the emulator, over the batch's whole wall clock (start-up,
    # stalls and resumes included).
    played = [r for r in merged if r.get("steps")]
    actions = sum(max(0, len(r["steps"]) - 1) for r in played)
    # Verified transitions: steps where the emulator reproduced the game's next state, i.e.
    # every step before a fight's first divergence. The honest headline (SPEED_PLAN.md).
    vt = 0
    for i, g in zip(ids, merged):
        if data[i]["ok"]:
            idx = data[i]["first"]
            vt += (len(g["steps"]) if idx is None else idx) - 1
    # Re-anchored: every step whose card play or end of turn the emulator reproduced from the
    # game's state before it, divergences earlier in the fight notwithstanding.
    rtot = rc.summarise([data[i]["analysis"] for i in ids])["totals"]
    vt_first, vt = vt, rtot.get("verified", 0)
    minutes = max(1e-9, (t_end - t_start) / 60)
    per_fight = [psum["s_per_fight_per_server"]]
    stalls = sum(1 for r in merged if str(r.get("error", "")).startswith("stalled"))
    # Settle audit: steps where the flag settle and the old six-frame rule saw different states.
    audited = [r for r in merged if "settle_audit" in r]
    audit_bad = [(r.get("seed"), x["step"]) for r in audited for x in r["settle_audit"] if isinstance(x, dict)]
    metrics = {
        "batch": a.name, "copies": a.parallel, "fights": len(merged), "verified": len(played),
        "minutes": round(minutes, 2), "vt_per_min": round(vt / minutes), "vt_first_div_per_min": round(vt_first / minutes),
        "reanchored": rtot.get("reanchored", 0), "fight_equiv_per_min": round(vt / minutes / 18.9, 1),
        "verified_per_min": round(len(played) / minutes, 1),
        "actions_per_min": round(actions / minutes), "s_per_fight_per_copy": round(sum(per_fight) / max(1, len(per_fight)), 2),
        "identical": same, "identical_pct": round(100 * same / max(1, len(played)), 1),
        "stalls": stalls, "errors": len([r for r in merged if r.get("error")]),
    }
    mfile = JOBS / "metrics.v2.csv"  # v2: re-anchored headline (vt_per_min) plus the first-divergence one
    if not mfile.exists():
        mfile.write_text(",".join(metrics) + chr(10), encoding="utf-8")
    with open(mfile, "a", encoding="utf-8") as f:
        f.write(",".join(str(v) for v in metrics.values()) + chr(10))
    speed = (f"SPEED {metrics['vt_per_min']} verified transitions/min (= {metrics['fight_equiv_per_min']} fight-equivalents/min; "
             f"{metrics['verified_per_min']} fights compared/min, {metrics['actions_per_min']} actions/min), "
             f"{metrics['s_per_fight_per_copy']} s/fight per copy x {a.parallel} copies, {metrics['minutes']} min; "
             f"identical {metrics['identical_pct']}%, stalls {stalls}; "
             f"re-anchored {vt} of {metrics['reanchored']} steps match (first-divergence method: {metrics['vt_first_div_per_min']}/min)")
    lines = [speed, f"batch {a.name}: {a.fights} fights, session exit {report.get('exit')}, {report.get('seconds')} s, "
             f"network {report.get('network_connections')}, real profile touched {report.get('real_profile_touched')}",
             f"settle audit: {len(audited)} fights, {sum(len(r['settle_audit']) for r in audited)} steps, "
             f"{len(audit_bad)} disagreements{(' e.g. ' + str(audit_bad[:5])) if audit_bad else ''}",
             f"identical: {same}", f"diverging: {sum(len(v) for v in groups.values())}", f"errors: {len(errors)}"]
    if reason:
        lines.insert(1, f"STOPPED EARLY: {reason}; {len(played)} of {a.fights} fights played")
    for field, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {field}: {len(items)}  e.g. {items[0]}")
    lines += ["  " + e for e in errors[:10]]
    packs = JOBS / f"{a.name}.packs"  # one context pack per likely bug, plus an index
    pk = subprocess.run([sys.executable, "-m", "tools.fidelity.context_pack", str(out), str(emu), "--out", str(packs)],
                        capture_output=True, text=True, cwd=ROOT, check=False)
    lines.append(f"context packs: {packs / 'index.md'}" if pk.returncode == 0 else f"context packs failed: {pk.stderr[-300:]}")
    text = "\n".join(lines)
    (JOBS / f"{a.name}.summary.txt").write_text(text + "\n\n" + cmp, encoding="utf-8")
    print(text)
    return 1 if reason else 0


if __name__ == "__main__":
    sys.exit(main())
