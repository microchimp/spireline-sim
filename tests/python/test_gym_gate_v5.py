"""Rule v5 of the fight loop's promotion gate (rl/gym_gate.py, rl/gym_loop.py; D68 addendum 5 decision 4, 2026-11-21), on synthetic rows.

Pinned: the stratified, deaths-weighted reading of a part is the hand calculation; stage 1 rejects a part that is clearly worse (part h: 95%
upper bound under -1.0 point; part a3: under -2.0); a candidate is promoted by route F (part h gains +0.75 points with its 97.5% lower bound
above 0 and HP lost not worse by 0.5, part a3 not clearly worse) or by route A3 (part a3 gains +1.5 with the same bound and HP test, part h not
clearly worse); a part gaining its near bar goes on to the next stage, never past the last; the draws are numbered by era, so after a v5
promotion the new incumbent plays its bar on fresh draws (its selection games are never its bar); stored rows of another engine are used only
after an identity check per part; the ruler is built once from starts outside the fight exam and the loop's practice, at most one a run, its
runs added to the exam's exclusion list; the command line's default rule is v5 and v4 and v3 remain. Nothing plays a game: `play_tasks` is
replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gym_gate as gg  # noqa: E402
import gym_loop as gl  # noqa: E402

H_IDS = [f"h{i}" for i in range(20)]
A_IDS = [f"a{i}" for i in range(12)]
H_LITE = {s: {"id": s, "act": 1 if i < 10 else 2, "kind": "boss" if i < 10 else "elite", "cat": "act1-boss" if i < 10 else "act2-elite",
              "cell": "act1-boss-74" if i < 10 else "act2-elite", "kind_cell": "act1-boss" if i < 10 else "act2-elite", "stratum": "human",
              "human_survived": i % 2 == 0, "human_hp_lost": 30.0, "hp": 50} for i, s in enumerate(H_IDS)}
A_LITE = {s: {"id": s, "act": 3, "kind": "boss", "cat": "act3-boss", "cell": "act3-boss1-80" if i < 6 else "act3-boss1-73", "kind_cell": "act3-boss",
              "stratum": "human" if i % 2 == 0 else "own", "human_survived": i % 4 == 0, "hp": 50} for i, s in enumerate(A_IDS)}
WEIGHTS = {"h": {"act1-boss": 3.0, "act2-elite": 1.0}, "a3": {"act3-boss1-80": 1.0, "act3-boss1-73": 1.0}}


def rule(**over):
    r = json.loads(json.dumps(gl.PROMOTION_RULE_V5))  # a deep copy (the tuples become lists, which the code reads the same)
    for p in ("h", "a3"):
        r["parts"][p]["weights"] = WEIGHTS[p]
    for k, v in over.items():
        r[k] = v
    return r


def row(s, j, won, end_hp=None, eng="E1", capped=False, moves=None):
    return {"start": s, "j": j, "won": won, "capped": capped, "start_hp": 50, "end_hp": (30 if won else 0) if end_hp is None else end_hp,
            "engine": eng, "config": "a3-v1", "moves_sha": moves or f"{s}-{j}", "seconds": 2.0, "hp_lost": (20 if won else 50) if end_hp is None else 50 - end_hp}


def keyed(rows):
    return gg.indexed(rows)


def cells_of(lite):
    return {s: x["kind_cell"] if x["act"] < 3 else x["cell"] for s, x in lite.items()}


class Stats(unittest.TestCase):
    def test_the_stratified_reading_is_the_hand_calculation(self):
        # cell X (weight 3): starts x0 (draw diffs +1, 0), x1 (0, 0); cell Y (weight 1): y0 (+1, +1), y1 (-1, 0), y2 (0, 0)
        cells = {"x0": "X", "x1": "X", "y0": "Y", "y1": "Y", "y2": "Y"}
        c = [row("x0", 0, True), row("x0", 1, False), row("x1", 0, False), row("x1", 1, False), row("y0", 0, True), row("y0", 1, True),
             row("y1", 0, False), row("y1", 1, False), row("y2", 0, True), row("y2", 1, True)]
        i = [row("x0", 0, False), row("x0", 1, False), row("x1", 0, False), row("x1", 1, False), row("y0", 0, False), row("y0", 1, False),
             row("y1", 0, True), row("y1", 1, False), row("y2", 0, True), row("y2", 1, True)]
        r = gg.strat_read(keyed(c), keyed(i), cells, {"X": 3.0, "Y": 1.0})
        # start means: X (0.5, 0), Y (1, -0.5, 0); cell means 0.25 and 1/6; variances 0.125 and 0.5833...
        mx, my = 0.25, (1 - 0.5 + 0) / 3
        vx = ((0.5 - mx) ** 2 + (0 - mx) ** 2) / 1
        vy = ((1 - my) ** 2 + (-0.5 - my) ** 2 + (0 - my) ** 2) / 2
        self.assertAlmostEqual(r["win_diff"], 0.75 * mx + 0.25 * my)
        self.assertAlmostEqual(r["win_se"], math.sqrt(0.75 ** 2 * vx / 2 + 0.25 ** 2 * vy / 3))
        self.assertEqual((r["pairs"], r["starts"], r["cells"]), (10, 5, 2))
        self.assertAlmostEqual(r["by_cell"]["X"]["win_diff"], mx)
        # HP lost: a win loses 20, a loss the whole 50; the same weights
        self.assertAlmostEqual(r["by_cell"]["X"]["hp_diff"], ((-30 + 0) / 2 + 0) / 2)
        # a cell with no weight is reported, not counted
        r0 = gg.strat_read(keyed(c), keyed(i), cells, {"X": 1.0})
        self.assertAlmostEqual(r0["win_diff"], mx)
        self.assertIn("Y", r0["by_cell"])
        self.assertIsNone(gg.strat_read({}, {}, cells, {"X": 1.0})["win_diff"])


def read(m, se, hp=0.0):
    return {"win_diff": m / 100, "win_se": se / 100, "hp_diff": hp}


class Decide(unittest.TestCase):
    R = gl.PROMOTION_RULE_V5

    def v(self, h, a3, stage=2):
        return gg.v5_decide({"h": h, "a3": a3}, stage, self.R)

    def test_stage_1_rejects_a_clearly_worse_part_only(self):
        self.assertEqual(self.v(read(0, 0.5), read(0, 1.0), 1)["verdict"], "continue")
        v = self.v(read(-2.0, 0.5), read(0, 1.0), 1)  # upper bound -1.18: under -1.0
        self.assertEqual(v["verdict"], "reject")
        self.assertIn("part h is clearly worse", v["why"])
        v = self.v(read(0, 0.5), read(-4.0, 1.0), 1)  # upper bound -2.36: under -2.0
        self.assertEqual(v["verdict"], "reject")
        self.assertIn("part a3", v["why"])
        self.assertEqual(self.v(read(0, 0.5), read(-3.0, 1.0), 1)["verdict"], "continue")  # -1.36: not clearly worse

    def test_route_f(self):
        v = self.v(read(1.5, 0.6), read(0, 0.9))  # lower bound 1.5 - 1.96 x 0.6 = +0.32
        self.assertEqual((v["verdict"], v["path"]), ("promote", "F"))
        v = self.v(read(1.5, 0.6), read(-4.0, 0.9))  # part a3 clearly worse (upper bound -2.52): route F is blocked
        self.assertEqual(v["verdict"], "reject")  # and a clearly worse part never goes on
        self.assertEqual(v["routes"]["F"]["blocked_by"], ["a3"])
        v = self.v(read(1.0, 0.6), read(0, 0.9))  # +1.0 but the 97.5% lower bound -0.18: not yet; over the +0.5 near bar
        self.assertEqual(v["verdict"], "continue")
        self.assertIn("97.5% lower bound", v["why"])
        v = self.v(read(1.5, 0.6, hp=0.6), read(0, 0.9))  # HP lost worse by more than 0.5
        self.assertEqual(v["verdict"], "continue")
        self.assertIn("HP lost a fight is worse", v["why"])
        v = self.v(read(0.7, 0.2), read(0, 0.9))  # a tight +0.7: the lower bound clears 0 but the gain is under +0.75
        self.assertEqual(v["verdict"], "continue")
        self.assertIn("under the +0.75", v["why"])

    def test_route_a3(self):
        v = self.v(read(0, 0.6), read(2.5, 0.9))  # lower bound 2.5 - 1.76 = +0.74
        self.assertEqual((v["verdict"], v["path"]), ("promote", "A3"))
        v = self.v(read(-1.0, 0.3), read(2.5, 0.9))  # part h a point down but not clearly worse (upper bound -0.51, over -1.0)
        self.assertEqual((v["verdict"], v["path"]), ("promote", "A3"))
        v = self.v(read(-2.0, 0.5), read(2.5, 0.9))  # part h upper bound -1.18: clearly worse, route A3 blocked
        self.assertEqual(v["verdict"], "reject")
        self.assertEqual(v["routes"]["A3"]["blocked_by"], ["h"])
        v = self.v(read(0, 0.6), read(1.2, 0.9))  # under +1.5, over the +1.0 near bar
        self.assertEqual(v["verdict"], "continue")

    def test_near_bars_and_the_last_stage(self):
        self.assertEqual(self.v(read(0.3, 0.6), read(0.5, 0.9))["verdict"], "reject")  # neither part near
        self.assertEqual(self.v(read(0.6, 0.6), read(0, 0.9))["verdict"], "continue")
        self.assertEqual(self.v(read(0.6, 0.6), read(0, 0.9), stage=3)["verdict"], "reject")  # stage 3 is the last
        self.assertEqual(self.v(read(1.5, 0.45), read(0, 0.6), stage=3)["path"], "F")
        both = self.v(read(1.5, 0.6), read(2.5, 0.9))
        self.assertEqual(both["path"], "F")  # the first route that passes names the verdict
        self.assertTrue(both["routes"]["A3"]["ok"])

    def test_the_rule_constants(self):
        r = self.R
        self.assertEqual((r["parts"]["h"]["stage_draws"], r["parts"]["a3"]["stage_draws"]), ((2, 8, 16), (1, 4, 8)))
        self.assertEqual((r["routes"]["F"]["points"], r["routes"]["A3"]["points"]), (0.75, 1.5))
        self.assertAlmostEqual(r["routes"]["F"]["z"], 1.95996, places=4)
        self.assertEqual((r["parts"]["h"]["worse"], r["parts"]["a3"]["worse"]), (-1.0, -2.0))
        self.assertEqual(sum(r["set"]["h_quota"].values()), 500)
        self.assertEqual(sum(r["set"]["a3_human_quota"].values()) + sum(r["set"]["a3_own_quota"].values()), 414)
        self.assertEqual(sorted(r["parts"]["h"]["weights"]), sorted(gl.KINDS))
        self.assertEqual(gl.era_js(r, "h", 0, 2), tuple(range(8)))
        self.assertEqual(gl.era_js(r, "h", 1, 1), (16, 17))
        self.assertEqual(gl.era_js(r, "a3", 2, 3), tuple(range(16, 24)))


class Cli(unittest.TestCase):
    def test_v5_is_the_default_and_v4_and_v3_remain(self):
        p = gl.build_parser()
        self.assertIs(gl.active_rule(p.parse_args(["run"])), gl.PROMOTION_RULE_V5)
        self.assertIs(gl.active_rule(p.parse_args(["run", "--rule", "v4"])), gl.PROMOTION_RULE_V4)
        self.assertIs(gl.active_rule(p.parse_args(["gate", "--candidate", "c", "--rule", "v3"])), gl.PROMOTION_RULE)
        self.assertIs(gl.active_rule(argparse.Namespace()), gl.PROMOTION_RULE)  # an argument set from before the flag
        a = p.parse_args(["gate5-set", "--root", "R", "--seed", "7", "--rebuild"])
        self.assertEqual((a.cmd, a.seed, a.rebuild), ("gate5-set", 7, True))


class Gate(unittest.TestCase):
    """`run_gate_v5` end to end on a scratch folder, with a fake `play_tasks` that writes rows from the test's own outcome functions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "loop"
        self.loop = gl.Loop(self.root)
        paths = gl.gate5_paths(self.root)
        paths["dir"].mkdir()
        paths["h_lite"].write_text(json.dumps(list(H_LITE.values())), encoding="utf-8")
        paths["a3_lite"].write_text(json.dumps(list(A_LITE.values())), encoding="utf-8")
        paths["h"].write_text("", encoding="utf-8")
        paths["a3"].write_text("", encoding="utf-8")
        for v in ("v3", "cand"):
            (Path(self.tmp.name) / v).mkdir()
            (Path(self.tmp.name) / v / "model.json").write_text("{}", encoding="utf-8")
        self.loop.state["versions"]["v3"] = {"dir": str(Path(self.tmp.name) / "v3"), "n": 3}
        self.loop.state["incumbent"] = "v3"
        self.played: list[tuple] = []
        level = lambda s, j: s in H_IDS[:10] or s in A_IDS[:6]  # noqa: E731
        self.outcome = {"v3": level, "cand": level}
        self.replay_same = True
        self.faults: set = set()
        for p in (mock.patch.object(gl, "engine_tag", lambda: "E2"), mock.patch.object(gl, "play_tasks", self.fake_play)):
            p.start()
            self.addCleanup(p.stop)

    def fake_play(self, loop, tasks, target, player_dir, procs, k, prior, what, steps=None, daemon=None, extra_opts=None):
        self.played.append((what, len(tasks), Path(target).name, sorted({t[2] for t in tasks}), sorted({t[1][0] for t in tasks})))
        with open(target, "a", encoding="utf-8", newline="\n") as fh:
            for t in tasks:
                _a, s, j, _c, info = t
                who = info["player"]
                won = self.outcome[who if who in self.outcome else "cand"](s, j)
                moves = f"{s}-{j}" if self.replay_same or not what.startswith("identity") else "other"
                if (s, j) in self.faults:  # what fight_gym.capped_row writes for an engine fault: no engine, config or start HP
                    fh.write(json.dumps({"_task": ["a3-v1", s, j], "capped": True, "cap_reason": "error: EngineFault: x", "seconds": 1.0,
                                         "arm": "a3-v1", "start": s, "j": j, "cat": "act3-boss", "won": False, "end_hp": 0}) + "\n")
                    continue
                fh.write(json.dumps({**row(s, j, won, eng="E2", moves=moves), "player": who}) + "\n")
        return None

    def gate(self, cand="cand", **kw):
        return gl.run_gate_v5(self.loop, cand, Path(self.tmp.name) / cand, 4, 32, rule=rule(), **kw)

    def test_route_a3_promotes_and_the_next_era_plays_fresh_draws(self):
        self.outcome["cand"] = lambda s, j: s in H_IDS[:10] or s.startswith("a")  # level on part h, wins every Act 3 start
        v = self.gate()
        self.assertEqual((v["verdict"], v["stage"], v["path"], v["era"]), ("promote", 2, "A3", 0))
        self.assertEqual(v["draws"], {"h": [0, 7], "a3": [0, 3]})
        self.assertEqual(v["rule"], gl.PROMOTION_RULE_V5["name"])
        self.assertIn("human bar:", v["human"]["line"])
        self.assertEqual(v["human"]["a3_human"]["starts"], 6)
        self.assertAlmostEqual(v["parts"]["a3"]["win_diff"], 0.5)  # the 73 cell goes from 0 to all won; the weights are equal
        h_draws = set()
        for w, _n, _f, js, x in self.played:
            if w == "gate-v3" and "h" in x:
                h_draws.update(js)
        self.assertEqual(h_draws, set(range(8)))  # the incumbent's own era-0 draws of part h, played once
        ver = gl.promote(self.loop, "cand", Path(self.tmp.name) / "cand", v, "off")
        self.assertEqual((ver, gl.gate5_era(self.loop)), ("v4", 1))
        self.assertIn("era 1 begins", (self.root / "loop.log").read_text(encoding="utf-8"))
        # the next candidate meets v4 on era 1's draws: v4 plays them itself, nothing of its candidate games is its bar
        self.played.clear()
        (Path(self.tmp.name) / "c2").mkdir()
        self.outcome["c2"] = self.outcome["cand"]
        self.outcome["v4"] = self.outcome["cand"]
        v2 = self.gate("c2")
        self.assertEqual((v2["era"], v2["draws"]["h"][0], v2["draws"]["a3"][0]), (1, 16, 8))
        by_who = {}
        for w, n, f, js, _x in self.played:
            by_who.setdefault(w, set()).update(js)
        self.assertTrue(by_who["gate-v4"])
        self.assertTrue(min(by_who["gate-v4"]) >= 8 and min(by_who["gate-c2"]) >= 8)  # a3 era 1 begins at 8, h at 16
        self.assertEqual(v2["verdict"], "reject")  # the same player as the incumbent: level

    def test_route_f_and_a_resumed_gate_plays_nothing_more(self):
        self.outcome["cand"] = lambda s, j: s in H_IDS[:16] or s in A_IDS[:6]  # six more part-h starts won
        v = self.gate()
        self.assertEqual((v["verdict"], v["path"]), ("promote", "F"))
        n = len(self.played)
        self.assertEqual(self.gate()["verdict"], "promote")
        self.assertEqual(len(self.played), n)  # the games are on disk

    def test_stage_1_rejects_a_candidate_clearly_worse_at_act_3(self):
        self.outcome["v3"] = lambda s, j: s in H_IDS[:10] or s.startswith("a")
        self.outcome["cand"] = lambda s, j: s in H_IDS[:10]
        v = self.gate()
        self.assertEqual((v["verdict"], v["stage"]), ("reject", 1))
        self.assertIn("part a3", v["why"])
        self.assertNotIn("human", v)
        self.assertEqual({tuple(js) for w, _n, _f, js, x in self.played if w == "gate-cand" and "a" in x}, {(0,)})

    def test_stored_rows_of_another_engine_need_an_identity_check_per_part(self):
        d = gl.gate5_paths(self.root)["dir"]
        with open(d / "games-v3.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for s in H_IDS + A_IDS:
                for j in range(16):
                    fh.write(json.dumps(row(s, j, self.outcome["v3"](s, j), eng="E1")) + "\n")
        v = self.gate()
        ident = [json.loads(x) for x in (d / "identity.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted((x["player"], x["set"], x["ok"]) for x in ident), [("v3", "a3", True), ("v3", "h", True)])
        self.assertFalse(any(w == "gate-v3" for w, *_ in self.played))  # its stored games were used
        self.assertEqual(v["verdict"], "reject")
        # a failed check: those rows are not used and the incumbent plays again
        self.played.clear()
        (d / "identity.jsonl").unlink()
        for f in d.glob("identity-*"):
            f.unlink()
        self.replay_same = False
        self.gate()
        self.assertTrue(any(w == "gate-v3" for w, *_ in self.played))

    def test_a_capped_game_is_a_loss_of_this_engine_and_is_not_played_again(self):
        # the first live v5 gate (2026-11-21) met two engine faults at the Act 3 bosses: the bare capped rows were left out of the pairing and
        # the stage read incomplete; now they are stamped as losses of this engine
        self.faults = {("a7", 1), ("a7", 3)}
        self.outcome["cand"] = lambda s, j: s in H_IDS[:16] or s in A_IDS[:6]
        v = self.gate()
        self.assertEqual((v["verdict"], v["path"]), ("promote", "F"))
        self.assertEqual(v["parts"]["a3"]["pairs"], 12 * 4)
        rows = [json.loads(x) for x in (gl.gate5_paths(self.root)["dir"] / "games-cand.jsonl").read_text(encoding="utf-8").splitlines()]
        stamped = [r for r in rows if r.get("stamped")]
        self.assertEqual(sorted((r["start"], r["j"]) for r in stamped), [("a7", 1), ("a7", 3)])
        self.assertTrue(all(r["engine"] == "E2" and r["config"] == "a3-v1" and not r["won"] and r["start_hp"] == 50 and r["hp_lost"] == 50
                            for r in stamped))
        self.assertEqual(gg.hp_lost(stamped[0]), 50.0)  # the whole start HP, as a loss
        n = len(self.played)
        self.assertEqual(self.gate()["verdict"], "promote")
        self.assertEqual(len(self.played), n)  # nothing is played again
        self.assertIn("counted as lost", (self.root / "loop.log").read_text(encoding="utf-8"))

    def test_no_ruler_stops_the_gate(self):
        for f in gl.gate5_paths(self.root)["dir"].glob("*.lite.json"):
            f.unlink()
        v = self.gate()
        self.assertEqual(v["verdict"], "stopped")
        self.assertIn("gate5-set", v["why"])


class Builder(unittest.TestCase):
    """`build_gate5` on a synthetic loop folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "loop"
        (self.root / "starts").mkdir(parents=True)
        (self.root / "exam").mkdir()
        import human_starts as hs
        # human run ids outside the corpus's exam bucket by the fixed hash rule (one inside, for the guard test)
        good, bad = [], None
        i = 0
        while len(good) < 40 or bad is None:
            rid = f"r{i:04d}"
            if hs.exam_bucket(rid) == hs.EXAM_BUCKET:
                bad = bad or rid
            else:
                good.append(rid)
            i += 1
        self.bad = bad
        h = []
        for n, rid in enumerate(good):  # 4 cells of 10 starts, two starts of run r0000-like ids share a run in the boss cell
            kind, act = (("boss", 1) if n < 10 else ("elite", 1) if n < 20 else ("hallway", 2) if n < 30 else ("boss", 2))
            enc = 74 if act == 1 else 75
            h.append({"id": f"human:{rid}|f{n % 30 + 1:02d}", "run": rid, "seed": rid, "act": act, "kind": kind, "cat": f"act{act}-{kind}",
                      "encounter_id": enc, "floor": 17 if kind == "boss" and act == 1 else 33 if kind == "boss" else 5, "doc": {"x": n}})
        h.append({**h[0], "id": f"human:{good[0]}|f09"})  # a second start of the same run, in the same cell
        self.h = h
        self.write_set("human-starts", h)
        outcomes = {s["id"]: {"human_survived": True, "human_hp_lost": 10} for s in h}
        (self.root / "starts" / "human-outcomes.json").write_text(json.dumps({"outcomes": outcomes}), encoding="utf-8")
        h3 = [{"id": f"human3:q{n}|f48", "run": f"q{n}", "seed": f"q{n}", "act": 3, "kind": "boss", "cat": "act3-boss", "boss_no": 1,
               "encounter_id": 80 if n < 6 else 73, "loop_played": n == 1, "stratum": "human", "doc": {"y": n}} for n in range(12)]
        o3 = [{"id": f"own3:p{n}|f48", "run": f"p{n}", "seed": f"p{n}", "act": 3, "kind": "boss", "cat": "act3-boss", "boss_no": 1,
               "encounter_id": 80 if n < 6 else 73, "stratum": "own", "doc": {"z": n}} for n in range(12)]
        self.write_set("human-starts-act3", h3)
        self.write_set("own-starts-act3", o3)
        (self.root / "starts" / "human-outcomes-act3.json").write_text(json.dumps({"outcomes": {s["id"]: {"human_survived": False} for s in h3}}),
                                                                         encoding="utf-8")
        # the fight exam's exclusion list holds the second boss start's run and one Act 3 opening; the loop has played the second elite's run
        (self.root / "exam" / "exclude_runs.json").write_text(json.dumps({"run_seeds": [h[1]["run"], "q0"], "start_ids": []}), encoding="utf-8")
        (self.root / "state.json").write_text(json.dumps({"next_j": {h[11]["id"]: 3}}), encoding="utf-8")
        self.rule = json.loads(json.dumps(gl.PROMOTION_RULE_V5))
        self.rule["set"]["h_quota"] = {"act1-boss-74": 5, "act1-elite": 5, "act2-hallway": 20, "act2-boss-75": 3}
        self.rule["set"]["a3_human_quota"] = {"act3-boss1-80": 3, "act3-boss1-73": 2}
        self.rule["set"]["a3_own_quota"] = {"act3-boss1-80": 2, "act3-boss1-73": 2}

    def write_set(self, stem, starts):
        d = self.root / "starts"
        index, off = {}, 0
        with open(d / f"{stem}.jsonl", "wb") as fh:
            for s in starts:
                index[s["id"]] = fh.tell()
                fh.write((json.dumps(s) + "\n").encode("utf-8"))
        (d / f"{stem}.index.json").write_text(json.dumps(index), encoding="utf-8")
        (d / f"{stem}.lite.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "doc"} for s in starts]), encoding="utf-8")

    def test_the_ruler_is_built_outside_the_exam_and_the_practice(self):
        meta = gl.build_gate5(self.root, rule=self.rule)
        sets = gl.load_gate5(self.root)
        h = list(sets["h"].values())
        runs = [s["run"] for s in h]
        self.assertEqual(len(runs), len(set(runs)))  # one start a run
        self.assertNotIn(self.h[1]["run"], runs)  # on the exam's exclusion list
        self.assertNotIn(self.h[11]["run"], runs)  # its run was played by the loop
        self.assertEqual(meta["report_h"]["act2-hallway"]["chosen"], 10)  # a quota the pool cannot fill is short, and said so
        self.assertIn("act2-hallway", meta["short_cells"])
        self.assertEqual(meta["starts"]["a3"], 9)
        a3 = sets["a3"]
        self.assertNotIn("human3:q0|f48", a3)  # the exam's Act 3 opening
        self.assertTrue(all("human_survived" in s for s in h))
        self.assertEqual({s["stratum"] for s in a3.values()}, {"human", "own"})
        docs = [json.loads(x) for x in gl.gate5_paths(self.root)["h"].read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all("doc" in d for d in docs))
        ex = json.loads((self.root / "exam" / "exclude_runs.json").read_text(encoding="utf-8"))
        self.assertTrue(set(runs) <= set(ex["run_seeds"]) and set(sets["h"]) <= set(ex["start_ids"]))
        self.assertTrue((self.root / "exam" / "exclude_runs.json.pre-gate5.bak").is_file())
        with self.assertRaises(FileExistsError):
            gl.build_gate5(self.root, rule=self.rule)
        again = gl.build_gate5(self.root, rule=self.rule, rebuild=True)  # the ruler's own starts are now excluded: a rebuild draws others
        self.assertTrue(list(gl.gate5_paths(self.root)["dir"].glob("meta.json.*.bak")))
        self.assertTrue(set(gl.load_gate5(self.root)["h"]).isdisjoint(sets["h"]))
        self.assertEqual(again["rule"], gl.PROMOTION_RULE_V5["name"])

    def test_a_start_of_the_corpus_exam_fifth_stops_the_build(self):
        s = {**self.h[2], "id": f"human:{self.bad}|f03", "run": self.bad, "seed": self.bad, "act": 2, "kind": "elite", "cat": "act2-elite"}
        self.write_set("human-starts", self.h + [s])
        out = json.loads((self.root / "starts" / "human-outcomes.json").read_text(encoding="utf-8"))
        out["outcomes"][s["id"]] = {"human_survived": True}
        (self.root / "starts" / "human-outcomes.json").write_text(json.dumps(out), encoding="utf-8")
        self.rule["set"]["h_quota"] = {"act2-elite": 1}
        with self.assertRaises(AssertionError):
            gl.build_gate5(self.root, rule=self.rule)


if __name__ == "__main__":
    unittest.main()
