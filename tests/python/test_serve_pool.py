"""referee/serve.py Pool: the long-lived dispatcher, driven against fake servers (no game needed).

tests/python/fixtures/fake_fight_server.py speaks driver/Server.cs's inbox protocol (ready and
heartbeat files, hold, claim by rename, one appended line per fight, STOP), so these tests run the
real file protocol: results come back in spec order, a crash re-queues its spec once and a second
crash records it as stalled, a hung server is killed by the heartbeat rule, the spare is promoted,
specs can be submitted while the pool runs, an abandoned close does not play what nobody claimed,
every spec is played once by one server even with many claiming at once, and the safety checks
(build, network, real profile) are the pool's own.
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import serve  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fixtures" / "fake_fight_server.py"
# A venv python.exe on Windows is a launcher that starts the real interpreter as a child, whose pid
# would not be the one the pool sees; the base interpreter runs the fake in the process it starts.
PYTHON = getattr(sys, "_base_executable", sys.executable)


def fake_launch(game, scratch, job, below_normal=False):
    log = open(scratch / "game-stdout.log", "wb")
    proc = subprocess.Popen([PYTHON, str(FAKE), str(job)], stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def kill(s):
    s.proc.kill()


def spec(i: int, **fake) -> dict:
    return {"kind": "combat", "seed": f"SEED{i:06d}", "n_steps": 1 + i % 4, **({"fake": fake} if fake else {})}


class PoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.pools = []

    def tearDown(self):
        for p in self.pools:
            if not p.closed:
                p.close(abandon=True)
        self.tmp.cleanup()

    def pool(self, **kw) -> serve.Pool:
        opts = dict(servers=2, spares=0, launch=fake_launch, connections=lambda pids: {int(p): set() for p in pids},
                    profile_state=lambda: [], check_build=lambda game: "fake-digest", kill=kill, echo=None)
        opts.update(kw)
        p = serve.Pool(self.dir / "game", self.dir / f"run{len(self.pools)}", **opts)
        self.pools.append(p)
        return p

    def run_until_done(self, p: serve.Pool, timeout: float = 60) -> list:
        got = []
        t0 = time.time()
        while p.outstanding and time.time() - t0 < timeout and not p.failed:
            got += p.poll()
            time.sleep(0.02)
        got += p.close()
        return got

    def test_results_come_back_in_spec_order_without_timing_keys(self):
        p = self.pool(servers=3)
        specs = [spec(i, sleep=0.01 * ((7 * i) % 5)) for i in range(15)]
        ids = p.submit(specs)
        got = self.run_until_done(p)
        self.assertEqual(sorted(rid for rid, _ in got), ids)
        merged = p.merged(ids)
        self.assertEqual([r["seed"] for r in merged], [s["seed"] for s in specs])
        self.assertEqual([len(r["steps"]) for r in merged], [s["n_steps"] for s in specs])
        for r in merged:
            self.assertFalse(set(r) & (serve.TIMING_KEYS | serve.META_KEYS))
        self.assertEqual((p.restarts, p.requeued), (0, 0))
        self.assertTrue(all(s.rc == 0 for s in p.servers), [s.rc for s in p.servers])

    def test_submitting_while_running_keeps_ids_in_submission_order(self):
        p = self.pool(servers=2)
        p.more = 12
        first = p.submit([spec(i) for i in range(4)])
        p.start()
        later = []
        t0 = time.time()
        while (len(first) + len(later) < 12 or p.outstanding) and time.time() - t0 < 60:
            p.poll()
            if len(first) + len(later) < 12 and p.unclaimed < 2:
                n = len(first) + len(later)
                later += p.submit([spec(n)])
                p.more = 12 - n - 1
            time.sleep(0.02)
        p.close()
        ids = first + later
        self.assertEqual(ids, [f"{i:06d}" for i in range(12)])
        self.assertEqual([r["seed"] for r in p.merged()], [spec(i)["seed"] for i in range(12)])

    def test_a_crash_requeues_the_spec_once(self):
        p = self.pool(servers=2)
        ids = p.submit([spec(0), spec(1, crash="once"), spec(2)])
        self.run_until_done(p)
        self.assertEqual((p.restarts, p.requeued), (1, 1))
        self.assertFalse(any(r.get("error") for r in p.merged(ids)))

    def test_specs_a_dead_server_had_not_claimed_go_back_without_an_attempt(self):
        p = self.pool(servers=1, depth=3)
        ids = p.submit([spec(0, crash="once", sleep=0.3)] + [spec(i) for i in range(1, 6)])
        self.run_until_done(p)
        self.assertFalse(any(r.get("error") for r in p.merged(ids)))
        self.assertEqual((p.restarts, p.requeued), (1, 1))
        self.assertEqual(dict(p.attempts), {ids[0]: 1})

    def test_a_second_crash_records_the_spec_as_stalled(self):
        p = self.pool(servers=2)
        ids = p.submit([spec(0), spec(1, crash="always"), spec(2)])
        self.run_until_done(p)
        merged = p.merged(ids)
        self.assertEqual(merged[1]["error"], serve.STALLED)
        self.assertEqual(merged[1]["seed"], spec(1)["seed"])
        self.assertFalse(merged[0].get("error") or merged[2].get("error"))
        self.assertEqual((p.restarts, p.requeued), (2, 1))

    def test_a_run_start_exception_requeues_like_a_crash(self):
        p = self.pool(servers=1)
        ids = p.submit([spec(0, fatal="once"), spec(1)])
        self.run_until_done(p)
        self.assertFalse(any(r.get("error") for r in p.merged(ids)))
        self.assertTrue(any("server error on spec 000000" in e for e in p.events), p.events)

    def test_a_hung_server_is_killed_by_the_heartbeat_rule(self):
        p = self.pool(servers=1)
        p.hb_limit = 1.0
        ids = p.submit([spec(0, hang="once"), spec(1)])
        self.run_until_done(p)
        self.assertFalse(any(r.get("error") for r in p.merged(ids)))
        self.assertEqual(p.requeued, 1)
        self.assertTrue(any(s.killed.startswith("no heartbeat") for s in p.servers))

    def test_the_hot_spare_is_promoted_when_a_server_dies(self):
        p = self.pool(servers=1, spares=1)
        ids = p.submit([spec(i, sleep=0.05) for i in range(8)] + [spec(8, crash="once")] + [spec(9 + i) for i in range(4)])
        self.run_until_done(p)
        self.assertFalse(any(r.get("error") for r in p.merged(ids)))
        self.assertTrue(any("promoted from spare" in e for e in p.events), p.events)

    def test_too_many_restarts_set_failed(self):
        p = self.pool(servers=1, max_restarts=1)
        p.submit([spec(i, crash="always") for i in range(3)])
        self.run_until_done(p)
        self.assertTrue(p.failed.startswith("giving up"))

    def test_abandoned_close_withdraws_unclaimed_specs(self):
        p = self.pool(servers=1)
        ids = p.submit([spec(i, sleep=0.2) for i in range(10)])
        p.start()
        t0 = time.time()
        while not p.results and time.time() - t0 < 30:
            p.poll()
            time.sleep(0.02)
        p.close(abandon=True)
        self.assertLess(len(p.results), len(ids))
        self.assertTrue(list(p.inbox.rglob("*.json.withdrawn")))
        self.assertEqual(p.servers[0].rc, 0)
        self.assertEqual(p.merged(ids)[-1]["error"], "no result")

    def test_network_connections_and_profile_changes_are_reported(self):
        state = iter([["before"], ["after"]])
        p = self.pool(servers=1, connections=lambda pids: {int(q): {"TCP 1.2.3.4:5"} for q in pids},
                      profile_state=lambda: next(state))
        p.submit([spec(0, sleep=0.3)])
        self.run_until_done(p)
        s = p.summary()
        self.assertEqual(s["network_connections"], ["TCP 1.2.3.4:5"])
        self.assertTrue(s["real_profile_touched"])

    def test_the_build_check_runs_before_anything_starts(self):
        def refuse(game):
            raise SystemExit(2)

        with self.assertRaises(SystemExit):
            self.pool(check_build=refuse)
        self.assertFalse((self.dir / "run0").exists())

    def test_every_spec_is_played_once_by_one_server(self):
        # Many servers claiming at once: a claim is a rename, and on Windows a rename is not exclusive
        # (two servers can both "win" it; night04 lost 4 of 600 fights that way). No spec may be
        # answered twice or as a bad spec file.
        p = self.pool(servers=8)
        ids = p.submit([spec(i, sleep=0) for i in range(400)])
        self.run_until_done(p, timeout=120)
        self.assertEqual([r.get("error") for r in p.merged(ids) if r.get("error")], [])
        answered = collections.Counter(r.get("_id") for s in p.servers for r in s.lines)
        self.assertEqual([k for k, v in answered.items() if v > 1], [])
        self.assertEqual(sorted(answered), ids)

    def test_a_result_without_a_seed_gets_the_seed_of_its_spec(self):
        p = self.pool(servers=1)
        ids = p.submit([spec(0, bad=True), spec(1)])
        self.run_until_done(p)
        merged = p.merged(ids)
        self.assertEqual(merged[0]["seed"], spec(0)["seed"])
        self.assertTrue(merged[0]["error"].startswith("bad spec file"))

    def test_a_used_run_folder_is_refused(self):
        p = self.pool(servers=1)
        p.submit([spec(0)])
        self.run_until_done(p)
        with self.assertRaises(ValueError):
            serve.Pool(self.dir / "game", p.run_dir, servers=1, spares=0, launch=fake_launch, check_build=lambda g: "x",
                       profile_state=lambda: [], connections=lambda pids: {}, kill=kill, echo=None)

    def test_run_files_and_summary(self):
        p = self.pool(servers=2)
        p.submit([spec(i) for i in range(5)])
        self.run_until_done(p)
        s = p.summary()
        p.write_run_files(s)
        self.assertEqual((s["fights"], s["results"], s["servers"], s["sts2_dll_sha256"]), (5, 5, 2, "fake-digest"))
        self.assertEqual(len((p.run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()), 5)
        self.assertIn("events", json.loads((p.run_dir / "summary.json").read_text(encoding="utf-8")))
        self.assertEqual(len(p.table()), 1 + len(p.servers))


if __name__ == "__main__":
    unittest.main()
