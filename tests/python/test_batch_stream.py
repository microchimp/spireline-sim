"""referee/batch.py Stream: one pool for a whole batch, emulator replay in chunks, no barrier.

A fake pool stands in for the game servers (results arrive out of spec order) and a fake replay for
EmuDump, so these run without the game or dotnet. The canary is the first chunk and a broken one
stops the batch at once; every fight is replayed exactly once; decks drawn later see coverage that
earlier chunks updated while the pool was still playing; an early stop (network, pool failure, a
worker exception) closes the pool and still reports what was played. The last test runs a Stream
on the real serve.Pool against the fake game servers of test_serve_pool.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import batch  # noqa: E402
import serve  # noqa: E402


class FakePool:
    """submit / poll / close like serve.Pool; each poll finishes up to `per_poll` queued specs, last first."""

    def __init__(self, per_poll=3, error_for=lambda spec: None, net_after=None, fail_after=None):
        self.window, self.per_poll, self.error_for = 4, per_poll, error_for
        self.net_after, self.fail_after = net_after, fail_after
        self.specs, self.queue, self.results = {}, [], {}
        self.more, self.failed, self.seen_net = 0, "", set()
        self.started = self.closed = False
        self.abandoned = None
        self.polls = 0

    def submit(self, specs):
        ids = []
        for s in specs:
            rid = f"{len(self.specs):06d}"
            self.specs[rid] = s
            self.queue.append(rid)
            ids.append(rid)
        return ids

    @property
    def unclaimed(self):
        return len(self.queue)

    def start(self):
        self.started = True

    def poll(self):
        assert self.started and not self.closed
        self.polls += 1
        if self.net_after is not None and self.polls > self.net_after:
            self.seen_net.add("TCP 1.2.3.4:443")
        if self.fail_after is not None and self.polls > self.fail_after:
            self.failed = "giving up: 9 restarts"
        done, self.queue = self.queue[: self.per_poll], self.queue[self.per_poll:]
        out = []
        for rid in reversed(done):
            spec = self.specs[rid]
            r = {"seed": spec["seed"], "steps": [{}, {}], "timing_ms": {"play": 1, "reset": 1}, "_id": rid}
            if self.error_for(spec):
                r["error"] = self.error_for(spec)
            self.results[rid] = r
            out.append((rid, r))
        return out

    def close(self, abandon=False):
        self.closed, self.abandoned = True, abandon
        return []


class Recorder:
    """draw() and process() for a Stream; process() bumps a coverage version every chunk."""

    def __init__(self, pool, lag=0.0, fail_on=None):
        self.pool, self.lag, self.fail_on = pool, lag, fail_on
        self.version = 0
        self.lock = threading.Lock()
        self.drawn = []  # (offset, coverage version seen, pool closed?)
        self.chunks = []

    def draw(self, n, offset):
        with self.lock:
            self.drawn.append((offset, self.version, self.pool.closed))
            return [{"seed": f"S{offset + i:05d}", "v": self.version} for i in range(n)]

    def process(self, tag, ids, games):
        time.sleep(self.lag)
        if self.fail_on == tag:
            raise RuntimeError("EmuDump failed")
        self.chunks.append((tag, list(ids), [g["seed"] for g in games]))
        with self.lock:
            self.version += 1
        return {"fights": {i: {"tag": tag} for i in ids}, "identical": 0, "coverage": f"v{self.version}"}


def stream(pool, rec, fights, **kw):
    kw.setdefault("tick", 0.001)
    return batch.Stream(pool, rec.draw, fights, rec.process, **kw)


class StreamTest(unittest.TestCase):
    def test_canary_is_the_first_chunk_and_every_fight_is_replayed_once(self):
        pool = FakePool()
        rec = Recorder(pool)
        st = stream(pool, rec, 40, canary=6, chunk=8)
        self.assertEqual(st.run(), "")
        self.assertEqual(st.chunks[0], ("canary", st.ids[:6]))
        replayed = [i for _, ids in st.chunks for i in ids]
        self.assertEqual(sorted(replayed), st.ids)
        self.assertEqual(len(replayed), len(set(replayed)))
        self.assertEqual(st.ids, [f"{i:06d}" for i in range(40)])
        self.assertEqual([st.finished[i]["seed"] for i in st.ids], [f"S{i:05d}" for i in range(40)])
        self.assertFalse(any(set(st.finished[i]) & serve.TIMING_KEYS for i in st.ids))
        self.assertEqual(set(st.data), set(st.ids))
        self.assertTrue(pool.closed and not pool.abandoned)
        self.assertGreaterEqual(len(st.chunks[1][1]), 8)  # a full chunk; only the last few fights go in smaller ones

    def test_later_decks_see_coverage_from_chunks_finished_while_playing(self):
        pool = FakePool(per_poll=1)
        rec = Recorder(pool)
        st = stream(pool, rec, 60, canary=4, chunk=6)
        self.assertEqual(st.run(), "")
        seen = [(v, closed) for _, v, closed in rec.drawn]
        self.assertTrue(any(v >= 3 and not closed for v, closed in seen), seen)
        self.assertEqual([o for o, _, _ in rec.drawn], sorted(o for o, _, _ in rec.drawn))
        self.assertLessEqual(max(o for o, _, _ in rec.drawn), 59)

    def test_inbox_is_only_topped_up_to_the_window(self):
        pool = FakePool(per_poll=1)
        rec = Recorder(pool)
        orig = pool.poll
        peak = []

        def poll():
            peak.append(pool.unclaimed)
            return orig()

        pool.poll = poll
        stream(pool, rec, 30, chunk=5).run()
        self.assertLessEqual(max(peak), pool.window)

    def test_a_failed_canary_stops_at_once(self):
        pool = FakePool(error_for=lambda s: "idle_timeout" if s["seed"] == "S00001" else None)
        rec = Recorder(pool)
        st = stream(pool, rec, 40, canary=6, chunk=8)
        self.assertTrue(st.run().startswith("canary failed: 1 of 6"))
        self.assertEqual(rec.chunks, [])
        self.assertTrue(pool.closed and pool.abandoned)
        self.assertEqual([r["seed"] for r in st.canary_bad], ["S00001"])

    def test_a_stalled_canary_fight_is_tolerated(self):
        pool = FakePool(error_for=lambda s: serve.STALLED if s["seed"] == "S00002" else None)
        rec = Recorder(pool)
        st = stream(pool, rec, 20, canary=8, chunk=4)
        self.assertEqual(st.run(), "")
        self.assertEqual(st.chunks[0][0], "canary")

    def test_no_canary_when_the_batch_is_not_bigger_than_it(self):
        pool = FakePool()
        rec = Recorder(pool)
        st = stream(pool, rec, 6, canary=6, chunk=4)
        self.assertEqual(st.run(), "")
        self.assertNotIn("canary", [t for t, _ in st.chunks])

    def test_a_network_connection_stops_the_batch_and_what_was_played_is_still_replayed(self):
        pool = FakePool(per_poll=1, net_after=10)
        rec = Recorder(pool)
        st = stream(pool, rec, 50, canary=4, chunk=4)
        self.assertTrue(st.run().startswith("network connection"))
        self.assertTrue(pool.closed and pool.abandoned)
        self.assertEqual(set(st.data), set(st.ids))
        missing = [i for i in st.ids if st.finished[i].get("error") == "no result"]
        self.assertTrue(missing)
        self.assertLess(len(st.ids), 50)

    def test_a_pool_that_gives_up_stops_the_batch(self):
        pool = FakePool(per_poll=1, fail_after=5)
        rec = Recorder(pool)
        st = stream(pool, rec, 30, chunk=4)
        self.assertEqual(st.run(), "giving up: 9 restarts")
        self.assertEqual(set(st.data), set(st.ids))

    def test_a_worker_exception_closes_the_pool_and_propagates(self):
        pool = FakePool(per_poll=1)
        rec = Recorder(pool, fail_on="c1")
        st = stream(pool, rec, 30, canary=2, chunk=4)
        with self.assertRaises(RuntimeError):
            st.run()
        self.assertTrue(pool.closed and pool.abandoned)

    def test_a_slow_emulator_does_not_hold_the_servers_back(self):
        pool = FakePool(per_poll=1)
        rec = Recorder(pool, lag=0.05)
        st = stream(pool, rec, 60, chunk=4)
        self.assertEqual(st.run(), "")
        # chunks grow to take everything finished meanwhile, so there are far fewer than 60/4
        self.assertLess(len(st.chunks), 15)
        self.assertEqual(sorted(i for _, ids in st.chunks for i in ids), st.ids)

    def test_the_replay_left_after_the_last_fight_is_small(self):
        pool = FakePool(per_poll=1)
        rec = Recorder(pool, lag=0.02)
        st = stream(pool, rec, 60, chunk=20, tick=0.01)
        self.assertEqual(st.run(), "")
        self.assertLessEqual(len(st.chunks[-1][1]), 10, [len(ids) for _, ids in st.chunks])

    def test_on_the_real_pool_with_fake_servers(self):
        import test_serve_pool as tsp  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as d:
            pool = serve.Pool(Path(d) / "game", Path(d) / "run", servers=3, spares=1, launch=tsp.fake_launch,
                              connections=lambda pids: {int(p): set() for p in pids}, profile_state=lambda: [],
                              check_build=lambda g: "fake", kill=tsp.kill, echo=None)
            rec = Recorder(pool)
            st = batch.Stream(pool, rec.draw, 30, rec.process, canary=6, chunk=5, tick=0.02)
            try:
                self.assertEqual(st.run(), "")
            finally:
                if not pool.closed:
                    pool.close(abandon=True)
            self.assertEqual(st.chunks[0], ("canary", st.ids[:6]))
            self.assertEqual([st.finished[i]["seed"] for i in st.ids], [f"S{i:05d}" for i in range(30)])
            self.assertEqual(sorted(i for _, ids in st.chunks for i in ids), st.ids)
            self.assertTrue(all(s.rc == 0 for s in pool.servers if not s.killed))


if __name__ == "__main__":
    unittest.main()
