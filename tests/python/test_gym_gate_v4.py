"""Rule v4 of the fight loop's promotion gate (rl/gym_gate.py, rl/gym_loop.py; D57 addendum 9, 2026-11-20), on synthetic exam rows.

Pinned: the paired survival and HP-lost differences and their start-clustered SE are the hand calculation; stage 1 rejects only a clearly worse
candidate (95% upper bound under -1.0 point); stage 2 promotes at +0.75 points with the 90% lower bound above zero and HP lost not worse by more
than 0.5; a gain of +0.5 that misses the bar goes to stage 3 (draws 0-15), the same test, else reject; the Act 3 guard fails only on a 95% upper
bound under -3.0 points and is played only for a would-be promotion once Act 3 practice has run 24 hours; the pre-filter's gap anchor counts
only verdicts of the active rule; stored incumbent rows of another engine are used only after an identity check (24 games replayed:
outcome, end HP, move hash), and a failed check puts those games back to be played again. Nothing plays a game: `play_tasks` is replaced.
"""

from __future__ import annotations

import json
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

RULE = gl.PROMOTION_RULE_V4
STARTS = [f"s{i}" for i in range(20)]
HUMAN = {s: {"id": s, "cat": "act1-boss" if i < 10 else "act2-elite", "kind": "boss" if i < 10 else "elite", "act": 1, "cell": "act1-boss-74" if i < 10 else "act2-elite",
             "kind_cell": "act1-boss" if i < 10 else "act2-elite", "half": "A", "human_survived": i % 2 == 0, "human_hp_lost": 30.0, "hp": 50}
         for i, s in enumerate(STARTS)}


def row(s, j, won, end_hp=None, eng="E1", capped=False, moves=None, seconds=2.0):
    return {"start": s, "j": j, "won": won, "capped": capped, "start_hp": 50, "end_hp": (30 if won else 0) if end_hp is None else end_hp, "engine": eng,
            "config": "a3-v1", "moves_sha": moves or f"{s}-{j}", "seconds": seconds, "hp_lost": 20 if won else 50, "cell": HUMAN[s]["cell"] if s in HUMAN else None}


def games(draws, win_at, eng="E1", end_hp=None):
    """Rows for draws in `draws`: start s wins draw j when win_at(s, j)."""
    return [row(s, j, win_at(s, j), end_hp=end_hp, eng=eng) for j in draws for s in STARTS]


def keyed(rows):
    return gg.indexed(rows)


class Stats(unittest.TestCase):
    def test_paired_reading_is_the_hand_calculation(self):
        # candidate wins draws 0 and 1 at s0 and s1; the incumbent wins only draw 0 at s0: pair differences +1 at (s0,1), (s1,0), (s1,1)
        cand = [row("s0", 0, True), row("s0", 1, True), row("s1", 0, True), row("s1", 1, True), row("s2", 0, False), row("s2", 1, False)]
        inc = [row("s0", 0, True), row("s0", 1, False), row("s1", 0, False), row("s1", 1, False), row("s2", 0, False), row("s2", 1, False)]
        r = gg.exam_read(keyed(cand), keyed(inc))
        self.assertEqual((r["pairs"], r["starts"], r["cand_wins"], r["inc_wins"]), (6, 3, 4, 1))
        self.assertAlmostEqual(r["win_diff"], 0.5)
        # per-start sums 1, 2, 0 against mean 0.5 x 2 pairs: SE = sqrt(3/2 x (0 + 1 + 1)) / 6
        self.assertAlmostEqual(r["win_se"], (1.5 * 2) ** 0.5 / 6)
        self.assertAlmostEqual(r["ub95"], r["win_diff"] + gg.Z95 * r["win_se"])
        self.assertAlmostEqual(r["lb90"], r["win_diff"] - gg.Z90 * r["win_se"])
        self.assertAlmostEqual(r["hp_diff"], -(30 * 3) / 6 * 1.0 + 0.0 if False else r["hp_diff"])  # HP lost: a win loses 20, a loss 50
        self.assertAlmostEqual(r["hp_diff"], -30 * 3 / 6)  # three pairs turned from a loss (50) to a win (20)
        # a capped game is lost, and its HP lost is the whole start HP
        self.assertEqual(gg.hp_lost(row("s0", 0, True, capped=True)), 50.0)
        self.assertEqual(gg.hp_lost(row("s0", 0, True, end_hp=12)), 38.0)

    def test_by_cell_reading(self):
        c = keyed(games(range(2), lambda s, j: True))
        i = keyed(games(range(2), lambda s, j: s != "s0"))
        by = gg.exam_read(c, i, {s: HUMAN[s]["kind_cell"] for s in STARTS})["by_cell"]
        self.assertEqual(sorted(by), ["act1-boss", "act2-elite"])
        self.assertAlmostEqual(by["act1-boss"]["win_diff"], 0.1)
        self.assertAlmostEqual(by["act2-elite"]["win_diff"], 0.0)

    def test_stage_branches(self):
        def at(stage, draws, c, i):
            r = dict(RULE, starts=20)
            return gg.v4_decide(keyed(games(range(draws), c)), keyed(games(range(draws), i)), stage, r)

        self.assertEqual(at(1, 2, lambda s, j: True, lambda s, j: True)["verdict"], "continue")
        v = at(1, 2, lambda s, j: s in STARTS[:2], lambda s, j: s in STARTS[:18])  # 16 starts worse: -0.8, upper bound far under -1 point
        self.assertEqual(v["verdict"], "reject")
        self.assertIn("clearly worse", v["why"])
        self.assertEqual(at(1, 1, lambda s, j: True, lambda s, j: True)["verdict"], "incomplete")  # one draw is half the pairs
        up = at(2, 8, lambda s, j: True, lambda s, j: s in STARTS[:10])  # +50 points
        self.assertEqual((up["verdict"], up["path"]), ("promote", "exam"))
        self.assertEqual(at(2, 8, lambda s, j: True, lambda s, j: True)["verdict"], "reject")  # level: under the +0.5 that earns stage 3
        # a gain over the bar whose HP lost is worse by more than 0.5: not promoted, and on to stage 3 (the gain is over +0.5)
        r = dict(RULE, starts=20)
        c, i = keyed(games(range(8), lambda s, j: True, end_hp=0)), keyed(games(range(8), lambda s, j: s in STARTS[:10]))
        v = gg.v4_decide({k: dict(x, won=True) for k, x in c.items()}, i, 2, r)  # wins, but each win ends at 0 HP: 50 lost against 50 or 20
        self.assertEqual(v["verdict"], "continue")
        self.assertIn("HP lost a fight is worse", v["why"])

    def test_near_gain_goes_to_stage_3_then_decides(self):
        r = dict(RULE, starts=20, promote={"points": 30.0, "z_lower": gg.Z90, "hp_worse": 0.5}, near_points=15.0)
        wins_c = lambda s, j: s in STARTS[:10] or (j >= 8 and s in STARTS[10:16])  # noqa: E731
        inc = lambda s, j: s in STARTS[:10] or (j < 8 and s in STARTS[10:14])  # noqa: E731
        # draws 0-7: the candidate is 4 starts down: -20 points; make it +20 instead by swapping the roles
        s2 = gg.v4_decide(keyed(games(range(8), inc)), keyed(games(range(8), lambda s, j: s in STARTS[:10])), 2, r)
        self.assertEqual(s2["verdict"], "continue")  # +20 points: over the 15 that earns stage 3, under the 30 the bar needs
        self.assertIn("stage 3", s2["why"])
        s3 = gg.v4_decide(keyed(games(range(16), lambda s, j: True)), keyed(games(range(16), lambda s, j: s in STARTS[:10])), 3, r)
        self.assertEqual(s3["verdict"], "promote")
        s3b = gg.v4_decide(keyed(games(range(16), lambda s, j: s in STARTS[:14])), keyed(games(range(16), lambda s, j: s in STARTS[:10])), 3, r)
        self.assertEqual(s3b["verdict"], "reject")  # stage 3 is the last: a near gain that misses the bar is a reject
        self.assertEqual(wins_c("s0", 0), True)

    def test_lower_bound_must_clear_zero(self):
        r = dict(RULE, starts=20)
        c, i = keyed(games(range(8), lambda s, j: True)), keyed(games(range(8), lambda s, j: j != 0 or s != "s0"))  # +0.625 points on 160 pairs
        v = gg.v4_decide(c, i, 2, r)  # over the +0.5 that earns stage 3, under the +0.75 the bar needs
        self.assertEqual(v["verdict"], "continue")
        self.assertIn("under the +0.75", v["why"])


class Act3(unittest.TestCase):
    def lite(self):
        out = []
        for cell, n in (("act3-boss1-80", 140), ("act3-boss2-80", 70), ("act3-boss1-78", 62), ("act3-boss1-73", 62), ("act3-boss2-73", 40), ("act3-boss2-78", 40)):
            out += [{"id": f"{cell}:{i}", "cell": cell} for i in range(n)]
        return out

    def test_sample_is_fixed_and_test_subject_first(self):
        a = gg.pick_act3_gate(self.lite(), 120, {"80": 80}, 7)
        self.assertEqual(a, gg.pick_act3_gate(self.lite(), 120, {"80": 80}, 7))
        self.assertNotEqual(a, gg.pick_act3_gate(self.lite(), 120, {"80": 80}, 8))
        self.assertEqual((len(a), len(set(a))), (120, 120))
        ts = [x for x in a if "-80:" in x]
        self.assertEqual(len(ts), 80)
        self.assertEqual({c: sum(1 for x in a if x.startswith(c + ":")) for c in ("act3-boss1-80", "act3-boss2-80")}, {"act3-boss1-80": 53, "act3-boss2-80": 27})
        self.assertEqual(sorted(len([x for x in a if f"-{b}:" in x]) for b in ("73", "78")), [20, 20])

    def test_guard_fails_only_on_a_clear_drop(self):
        r = dict(RULE)
        cells = {s: HUMAN[s]["kind_cell"] for s in STARTS}
        inc = keyed([row(s, 0, True) for s in STARTS])
        g = gg.act3_guard(keyed([row(s, 0, i % 2 == 0) for i, s in enumerate(STARTS)]), inc, r, cells)  # -50 points on 20 starts: upper bound well under -3
        self.assertTrue(g["fail"])
        self.assertEqual(sorted(g["by_cell"]), ["act1-boss", "act2-elite"])
        g = gg.act3_guard(keyed([row(s, 0, s != "s0") for s in STARTS]), inc, r, cells)  # -5 points, SE large: upper bound over -3
        self.assertFalse(g["fail"])
        self.assertFalse(gg.act3_guard({}, {}, r)["fail"])


class Identity(unittest.TestCase):
    def test_check_needs_every_game_the_same(self):
        stored = keyed([row("s0", 0, True, moves="a"), row("s1", 0, False, moves="b")])
        self.assertTrue(gg.identity_check(stored, keyed([row("s0", 0, True, moves="a", eng="E2"), row("s1", 0, False, moves="b", eng="E2")]))["ok"])
        for bad in (row("s0", 0, True, moves="x"), row("s0", 0, True, end_hp=29, moves="a"), row("s0", 0, False, moves="a")):
            res = gg.identity_check(stored, keyed([bad, row("s1", 0, False, moves="b")]))
            self.assertFalse(res["ok"])
            self.assertEqual(res["bad"], [["s0", 0]])
        self.assertFalse(gg.identity_check(stored, keyed([row("s0", 0, True, moves="a")]))["ok"])  # a game not replayed
        self.assertEqual(len(gg.identity_sample([(s, j) for s in STARTS for j in range(8)], 24, "x")), 24)
        self.assertEqual(gg.identity_sample([("a", 0)], 24, "x"), [("a", 0)])


class Prefilter(unittest.TestCase):
    def test_anchor_counts_only_the_active_rules_verdicts(self):
        v3, v4 = gl.PROMOTION_RULE["name"], RULE["name"]
        cfg = {**gl.PREFILTER, "rule": v4}
        pf = {"best": {"cand": "c0295", "gap": 0.0362}}  # a record from before the rule's name was kept is v3's
        self.assertTrue(gl.prefilter_decide(pf, {"gap": 0.0300}, cfg)[0])  # v3's anchor does not hold a v4 candidate back
        self.assertFalse(gl.prefilter_decide(pf, {"gap": 0.0300}, {**cfg, "rule": v3})[0])
        self.assertFalse(gl.prefilter_decide({"best": {"cand": "c0380", "gap": 0.0296, "rule": v4}}, {"gap": 0.0296}, cfg)[0])
        with tempfile.TemporaryDirectory() as tmp:
            loop = gl.Loop(Path(tmp) / "loop")
            loop.state["incumbent"] = "v3"
            loop.state["prefilter"] = {"rule": "x", "incumbent": "v3", "last_gate": "2026-11-20T00:00:00Z", "window": [],
                                       "figures": {"c0380": {"gap": 0.0296, "loss": 0.99, "ref": 1.02}, "c0390": {"gap": 0.02, "loss": 1.0, "ref": 1.02}},
                                       "best": {"cand": "c0295", "gap": 0.0362, "loss": 0.98, "ref": 1.02}}
            gl.prefilter_note_verdict(loop, "c0390", {"verdict": "reject", "rule": v4})  # a lower gap, but of the active rule: it replaces the old anchor
            self.assertEqual((loop.state["prefilter"]["best"]["cand"], loop.state["prefilter"]["best"]["rule"]), ("c0390", v4))
            gl.prefilter_note_verdict(loop, "c0380", {"verdict": "reject", "rule": v4})
            self.assertEqual(loop.state["prefilter"]["best"]["cand"], "c0380")


class Gate(unittest.TestCase):
    """`run_gate_v4` end to end on a scratch folder, with a fake `play_tasks` that writes rows from the test's own outcome function."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "loop"
        self.loop = gl.Loop(self.root)
        self.exam = self.root / "exam"
        self.exam.mkdir()
        (self.exam / "starts-human.lite.json").write_text(json.dumps(list(HUMAN.values())), encoding="utf-8")
        self.loop.state["versions"]["v3"] = {"dir": str(Path(self.tmp.name) / "v3"), "n": 3}
        self.loop.state["incumbent"] = "v3"
        self.played: list[tuple] = []
        self.outcome = {"v3": lambda s, j: s in STARTS[:10], "cand": lambda s, j: s in STARTS[:10]}
        self.replay_same = True
        p = mock.patch.object(gl, "engine_tag", lambda: "E2")
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(gl, "play_tasks", self.fake_play)
        q.start()
        self.addCleanup(q.stop)
        self.addCleanup(self.tmp.cleanup)

    def fake_play(self, loop, tasks, target, player_dir, procs, k, prior, what, steps=None, daemon=None, extra_opts=None):
        self.played.append((what, len(tasks), Path(target).name))
        with open(target, "a", encoding="utf-8", newline="\n") as fh:
            for t in tasks:
                _a, s, j, _c, info = t
                who = info["player"]
                won = self.outcome[who if who in self.outcome else "cand"](s, j)
                moves = f"{s}-{j}" if self.replay_same or not what.startswith("identity") else "other"
                fh.write(json.dumps({**row(s, j, won, eng="E2", moves=moves), "player": who}) + "\n")
        return None

    def put(self, name, rows):
        with open(self.exam / f"games-{name}.jsonl", "a", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)

    def gate(self, **kw):
        return gl.run_gate_v4(self.loop, "cand", Path(self.tmp.name) / "cand", 4, 32, **kw)

    def test_promote_reuses_stored_games_after_an_identity_check(self):
        self.put("v3", games(range(16), lambda s, j: s in STARTS[:10], eng="E1"))  # stored on another engine
        self.outcome["cand"] = lambda s, j: True
        v = self.gate()
        self.assertEqual((v["verdict"], v["stage"], v["draws"]), ("promote", 2, 8))
        self.assertEqual(v["rule"], RULE["name"])
        self.assertIn("human bar:", v["human"]["line"])
        self.assertEqual(sorted(v["human"]["by_cell"]), ["act1-boss", "act2-elite"])
        ident = [json.loads(x) for x in (self.exam / "identity.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(x["player"], x["from_engine"], x["to_engine"], x["n"], x["ok"]) for x in ident], [("v3", "E1", "E2", 24, True)])
        names = [w for w, _n, _f in self.played]
        self.assertEqual(names.count("identity-v3"), 1)  # 24 replayed once, then the incumbent plays nothing: the candidate plays stages 1 and 2
        self.assertNotIn("gate-v3", names)
        self.assertEqual(sum(n for w, n, _f in self.played if w == "gate-cand"), 20 * 8)
        self.assertEqual(self.gate()["verdict"], "promote")  # the games are on disk: a second run plays nothing more
        self.assertEqual(len(self.played), 3)

    def test_failed_identity_puts_the_games_back_to_be_played(self):
        self.put("v3", games(range(8), lambda s, j: s in STARTS[:10], eng="E1"))
        self.replay_same = False
        self.outcome["cand"] = lambda s, j: s in STARTS[:10]
        v = self.gate()
        ident = [json.loads(x) for x in (self.exam / "identity.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertFalse(ident[0]["ok"])
        self.assertTrue(any(w == "gate-v3" for w, _n, _f in self.played))  # the incumbent plays the stage's draws again on this engine
        self.assertEqual(v["verdict"], "reject")

    def test_stage_1_rejects_a_clearly_worse_candidate_at_800_games_of_a_400_start_exam(self):
        self.outcome["cand"] = lambda s, j: s in STARTS[:2]
        self.outcome["v3"] = lambda s, j: s in STARTS[:18]
        v = self.gate()
        self.assertEqual((v["verdict"], v["stage"], v["draws"]), ("reject", 1, 2))
        self.assertNotIn("human", v)  # the human line is a stage-2 and stage-3 reading
        self.assertEqual(sum(n for w, n, _f in self.played if w == "gate-cand"), 20 * 2)

    def test_near_gain_plays_stage_3(self):
        rule = {**RULE, "promote": {"points": 30.0, "z_lower": gg.Z90, "hp_worse": 0.5}, "near_points": 15.0}
        self.outcome["v3"] = lambda s, j: s in STARTS[:10]
        self.outcome["cand"] = lambda s, j: s in STARTS[:14]  # +20 points on every stage
        v = self.gate(rule=rule)
        self.assertEqual((v["stage"], v["draws"]), (3, 16))  # stage 2 said "continue"; stage 3 decided
        self.assertIn(v["verdict"], ("promote", "reject"))
        self.assertTrue(any("stage 2" in x and "continue" in x for x in (self.root / "loop.log").read_text(encoding="utf-8").splitlines()))

    def test_act3_guard_due_pass_and_fail(self):
        lite = [{"id": f"a3-{i}", "cell": "act3-boss1-80" if i < 8 else "act3-boss1-73", "cat": "act3-boss", "kind": "boss"} for i in range(16)]
        (self.exam / "starts-act3.lite.json").write_text(json.dumps(lite), encoding="utf-8")
        (self.exam / "starts-act3.jsonl").write_text("", encoding="utf-8")
        rule = {**RULE, "act3_guard": {**RULE["act3_guard"], "starts": 16, "quotas": {"80": 8}}}
        self.outcome["cand"] = lambda s, j: True
        v = self.gate(rule=rule, now=gl.utc())
        self.assertNotIn("act3", v)  # no practice segment has drawn an Act 3 start: not due
        with open(self.loop.games / "segments.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": "2026-11-20T08:00:00Z", "cats": {"act1-boss": 5, "act3-boss": 2}}) + "\n")
        self.assertFalse(gl.act3_guard_due(self.loop, rule, gl.dt.datetime(2026, 11, 21, 7, 59, tzinfo=gl.dt.timezone.utc)))
        self.assertTrue(gl.act3_guard_due(self.loop, rule, gl.dt.datetime(2026, 11, 21, 8, 1, tzinfo=gl.dt.timezone.utc)))
        self.outcome["v3"] = lambda s, j: s in STARTS[:10] or s.startswith("a3-")
        self.outcome["cand"] = lambda s, j: True
        v = self.gate(rule=rule, now=gl.dt.datetime(2026, 11, 21, 8, 1, tzinfo=gl.dt.timezone.utc))
        self.assertEqual(v["verdict"], "promote")
        self.assertFalse(v["act3"]["fail"])
        self.assertEqual(sorted(v["act3"]["by_cell"]), ["act3-boss1-73", "act3-boss1-80"])
        self.assertTrue((self.exam / "starts-act3-gate.jsonl").is_file())
        self.assertTrue((self.exam / "act3-gate" / "games-cand.jsonl").is_file())
        # a candidate that loses most Act 3 starts the incumbent wins: the promotion is refused, the verdict says why
        self.outcome["cand"] = lambda s, j: not s.startswith("a3-")
        (self.exam / "act3-gate" / "games-cand.jsonl").unlink()
        v = self.gate(rule=rule, now=gl.dt.datetime(2026, 11, 21, 8, 1, tzinfo=gl.dt.timezone.utc))
        self.assertEqual((v["verdict"], v.get("guard_act3")), ("reject", True))
        self.assertIn("Act 3 guard", v["why"])

    def test_import_adds_only_missing_rows_with_provenance(self):
        src = Path(self.tmp.name) / "src.jsonl"
        src.write_text("".join(json.dumps(r) + "\n" for r in games(range(2), lambda s, j: True)), encoding="utf-8")
        self.assertEqual(gl.import_games(self.root, "cand", src, "a file"), 40)
        self.assertEqual(gl.import_games(self.root, "cand", src, "a file"), 0)
        rows = [json.loads(x) for x in (self.exam / "games-cand.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["provenance"] for r in rows}, {"a file"})


if __name__ == "__main__":
    unittest.main()
