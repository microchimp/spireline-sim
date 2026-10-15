"""referee/scenarios.py --mode targets: realistic specs aimed at the fix lab's weighted targets.

The small export of test_scenarios stands in for the game's tables and a hand-made profile for the Codex
aggregate counts. A spec must aim at a target the setup can reach, keep the item order the referee and
EmuDump rely on (act, deck, map items, the fight, combat items), stay in the character's pools and in the
act the profile says the target lives in, and hold each card, relic and potion as its own item.
"""

from __future__ import annotations

import collections
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scenarios  # noqa: E402
from test_scenarios import EXPORT, ID_MAP, INFO  # noqa: E402

PROFILE = {
    "runs": 1000,
    "meet": {"card": {"IC1": 300, "IC2": 200}, "relic": {"RED_SKULL": 100}, "encounter": {"HIVE_N": 50}},
    "acts": {
        "1": {"snapshots": 100, "rooms": {"monster": 90, "elite": 10}, "act_names": {"OVERGROWTH": 60, "UNDERDOCKS": 40},
              "encounters": {"monster": {"OVERGROWTH_N": 50, "UNDERDOCKS_N": 30, "OVERGROWTH_W": 10}, "elite": {"OVERGROWTH_E": 10}},
              "cards": {"IC1": 30, "IC2": 10, "IC3": 50}, "copies": {"IC1": 30, "IC2": 10, "IC3": 100}, "upgraded": {"IC3": 50},
              "relics": {"SR1": 20, "RED_SKULL": 5}, "potions": {"P1": 30},
              "deck_size": {"14": 100}, "relic_count": {"2": 100}, "potion_count": {"1": 100},
              "basics": {"STRIKE_IRONCLAD": {"4": 100}, "DEFEND_IRONCLAD": {"4": 100}, "BASH": {"1": 100}, "ASCENDERS_BANE": {"1": 100}},
              "pairs": {}},
        "2": {"snapshots": 50, "rooms": {"monster": 40, "boss": 10}, "act_names": {"HIVE": 50},
              "encounters": {"monster": {"HIVE_N": 30}, "boss": {"HIVE_B": 10}},
              "cards": {"IC2": 40, "IC4": 30}, "copies": {"IC2": 40, "IC4": 30}, "relics": {"SR2": 30}, "potions": {},
              "deck_size": {"20": 50}, "relic_count": {"3": 50}, "potion_count": {"0": 50},
              "basics": {"STRIKE_IRONCLAD": {"2": 50}, "DEFEND_IRONCLAD": {"3": 50}, "BASH": {"1": 50}, "ASCENDERS_BANE": {"1": 50}},
              "pairs": {"card:IC2|card:IC4": 30}},
    },
    "encounter_monsters": {"HIVE_N": ["HIVE_MONSTER"]},
}
TARGETS = {"profile": "cache/codex_profile.json", "targets": [
    {"kind": "card", "id": "IC2", "weight": 1.0, "reasons": []},
    {"kind": "relic", "id": "RED_SKULL", "weight": 0.5, "reasons": []},
    {"kind": "enemy", "id": "HIVE_MONSTER", "weight": 0.5, "reasons": []},
    {"kind": "relic", "id": "ORRERY", "weight": 9.0, "reasons": []},       # its pickup opens a screen the mirror cannot answer
    {"kind": "card", "id": "NEUTRALIZE", "weight": 9.0, "reasons": []},    # another character's
    {"kind": "event", "id": "NEOW", "weight": 9.0, "reasons": []},         # not a fight
]}


class AimTests(unittest.TestCase):
    def setUp(self):
        self.content = scenarios.Content(EXPORT, INFO, ID_MAP)
        self.aim = scenarios.Aim(TARGETS, PROFILE, self.content, {"KNOWNOVER1": "OVERGROWTH", "KNOWNUNDR1": "UNDERDOCKS"})

    def specs(self, n=120, seed=3):
        rng = random.Random(seed)
        return [scenarios.aimed_spec(self.aim, rng, i) for i in range(n)]

    def test_only_reachable_targets_are_drawn(self):
        self.assertEqual({"card:IC2", "relic:RED_SKULL", "enemy:HIVE_MONSTER"}, {f"{k}:{i}" for k, i, _ in self.aim.targets})
        self.assertEqual({"relic": 1, "card": 1, "event": 1}, dict(self.aim.skipped))
        got = collections.Counter(m["target"] for _, m in self.specs())
        self.assertGreater(got["card:IC2"], got["relic:RED_SKULL"])  # drawn by weight

    def test_item_order_and_pools(self):
        for spec, _ in self.specs():
            cmds = [it["cmd"] for it in spec["setup"]]
            self.assertEqual(1, cmds.count("fight"))
            f = cmds.index("fight")
            head = cmds[:f]
            if head[0] == "act":
                head = head[1:]
            self.assertEqual("deck_set", head[0])
            self.assertTrue(all(c in ("card", "relic", "potion") for c in head[1:]))
            self.assertTrue(all(c == "power" for c in cmds[f + 1:]))
            for it in spec["setup"]:
                if it["cmd"] == "card":
                    self.assertIn(it["id"], self.content.cards)
                    self.assertNotIn(it["id"], scenarios.BASICS)
                elif it["cmd"] == "relic":
                    self.assertIn(it["id"], self.content.relics)
                elif it["cmd"] == "potion":
                    self.assertIn(it["id"], self.content.potions)
            self.assertEqual([], scenarios.spec_flags(spec, self.content))  # no cross-character or duplicate item

    def test_the_target_is_in_the_spec_and_in_its_act(self):
        for spec, m in self.specs():
            setup = spec["setup"]
            enc = next(it["id"] for it in setup if it["cmd"] == "fight")
            act_name = self.aim.enc[enc][0]
            self.assertEqual(m["act"], scenarios.ACT_NUMBER[act_name])
            self.assertEqual(m["act"] > 1, setup[0] == {"cmd": "act", "index": m["act"]})
            kind, ident = m["target"].split(":")
            if kind == "card":
                self.assertIn(ident, [it["id"] for it in setup if it["cmd"] == "card"])
                self.assertIn(m["act"], (1, 2))  # IC2 is held in acts 1 and 2 only
            elif kind == "relic":
                self.assertIn(ident, [it["id"] for it in setup if it["cmd"] == "relic"])
                self.assertEqual(1, m["act"])
            elif kind == "enemy":
                self.assertEqual("HIVE_N", enc)  # the only encounter holding it

    def test_decks_follow_the_acts_shape(self):
        for spec, m in self.specs():
            deck = next(it["cards"] for it in spec["setup"] if it["cmd"] == "deck_set")
            basics = collections.Counter(c["id"] for c in deck)
            cards = [it for it in spec["setup"] if it["cmd"] == "card"]
            if m["act"] == 1:
                self.assertEqual({"STRIKE_IRONCLAD": 4, "DEFEND_IRONCLAD": 4, "BASH": 1, "ASCENDERS_BANE": 1}, dict(basics))
                self.assertIn(len(deck) + len(cards), (14, 15))  # the act's deck size; a second copy may add one
                self.assertTrue(all(it["id"] in ("IC1", "IC2", "IC3") for it in cards))  # held at act 1
            else:
                self.assertEqual(2, basics["STRIKE_IRONCLAD"])
                self.assertTrue(all(it["id"] in ("IC2", "IC4") for it in cards))

    def big_cache(self):
        cache = {f"KNOWNOVR{i:02d}": "OVERGROWTH" for i in range(80)} | {f"KNOWNUND{i:02d}": "UNDERDOCKS" for i in range(80)}
        return scenarios.Aim(TARGETS, PROFILE, self.content, cache), cache

    def test_act_one_seeds_come_from_the_cache_mostly(self):
        self.aim, cache = self.big_cache()
        specs = self.specs(200)
        known = sum(1 for spec, m in specs if m["act"] == 1 and spec["seed"] in cache)
        act1 = sum(1 for _, m in specs if m["act"] == 1)
        self.assertGreater(known, act1 // 2)
        for spec, m in specs:
            if spec["seed"] in cache:
                self.assertEqual(cache[spec["seed"]], self.aim.enc[m["encounter"]][0])
            self.assertEqual("cache" if spec["seed"] in cache else "fresh", m["seed_from"])

    def test_no_seed_twice_in_a_batch(self):
        # shrink's packs, traces and lab_groups key a batch's outputs by seed: a seed on two specs mixes their fights
        specs = self.specs(300)  # 2 cached seeds, many more act-1 specs
        seeds = [s["seed"] for s, _ in specs]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(2, sum(1 for _, m in specs if m["seed_from"] == "cache"))
        self.aim, cache = self.big_cache()
        seeds = [s["seed"] for s, _ in self.specs(300)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertGreater(len(set(seeds) & set(cache)), 60)
        clash = scenarios._seed(random.Random(1))  # a fresh seed that clashes with one already used is redrawn
        self.aim.used.add(clash)
        got = self.aim.seed(random.Random(1), 2, "HIVE")
        self.assertNotEqual(clash, got)
        self.assertIn(got, self.aim.used)

    def test_deterministic(self):
        a = [json.dumps(s, sort_keys=True) for s, _ in self.specs(30, 9)]
        self.setUp()  # one Aim is one batch (it holds the batch's used seeds)
        b = [json.dumps(s, sort_keys=True) for s, _ in self.specs(30, 9)]
        self.assertEqual(a, b)

    def test_act1_seed_cache(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "a.out.json"
            out.write_text(json.dumps({"results": [{"seed": "S1", "fight": {"current_act": "UNDERDOCKS"}},
                                                   {"seed": "S2", "fight": {"current_act": "HIVE"}}, None]}), encoding="utf-8")
            path = Path(td) / "cache" / "act1.json"
            self.assertEqual({"S1": "UNDERDOCKS"}, scenarios.act1_seed_cache(path, scan=[out]))
            got = scenarios.act1_seed_cache(path, [{"seed": "S3", "fight": {"current_act": "OVERGROWTH"}}])
            self.assertEqual({"S1": "UNDERDOCKS", "S3": "OVERGROWTH"}, got)



STARTER_TARGETS = {"profile": "cache/codex_profile.json", "targets": [
    {"kind": "relic", "id": "BURNING_BLOOD", "weight": 20.0, "reasons": []},
    {"kind": "card", "id": "IC2", "weight": 1.0, "reasons": [], "backlog_items": ["C1-F01"]},
    {"kind": "card", "id": "IC1", "weight": 1.0, "reasons": ["open backlog items name it: C1-F02"]},   # older targets.json: the reason only
    {"kind": "relic", "id": "RED_SKULL", "weight": 0.5, "reasons": ["no recorded or setup fight reaches it yet"]},
]}


class AimModeTests(unittest.TestCase):
    def setUp(self):
        self.content = scenarios.Content(EXPORT, INFO, ID_MAP)
        # BURNING_BLOOD is a starter relic, not in the setup's relic pool: it is reachable only through the content's relics
        self.content.relics.append("BURNING_BLOOD")

    def aim(self, mode):
        return scenarios.Aim(STARTER_TARGETS, PROFILE, self.content, None, mode)

    def test_all_keeps_every_reachable_target_at_its_weight(self):
        got = {f"{k}:{i}": w for k, i, w in self.aim("all").targets}
        self.assertEqual({"relic:BURNING_BLOOD": 20.0, "card:IC2": 1.0, "card:IC1": 1.0, "relic:RED_SKULL": 0.5}, got)

    def test_bugs_keeps_only_what_an_open_backlog_item_names(self):
        a = self.aim("bugs")
        self.assertEqual({"card:IC2", "card:IC1"}, {f"{k}:{i}" for k, i, _ in a.targets})
        self.assertEqual(2, a.skipped["not_in_backlog"])

    def test_gaps_weighs_the_starter_items_one_twentieth(self):
        got = {f"{k}:{i}": w for k, i, w in self.aim("gaps").targets}
        self.assertAlmostEqual(1.0, got["relic:BURNING_BLOOD"])
        self.assertEqual(1.0, got["card:IC2"])
        self.assertEqual(0.5, got["relic:RED_SKULL"])

    def test_specs_of_each_aim_aim_at_their_targets(self):
        for mode, allowed in (("bugs", {"card:IC2", "card:IC1"}), ("gaps", {"relic:BURNING_BLOOD", "card:IC2", "card:IC1", "relic:RED_SKULL"})):
            a = self.aim(mode)
            rng = random.Random(4)
            got = {scenarios.aimed_spec(a, rng, i)[1]["target"] for i in range(60)}
            self.assertTrue(got <= allowed, (mode, got))
        a = self.aim("gaps")
        rng = random.Random(5)
        counts = collections.Counter(scenarios.aimed_spec(a, rng, i)[1]["target"] for i in range(400))
        self.assertTrue(0.2 < counts["relic:BURNING_BLOOD"] / 400 < 0.36)  # 1 of 3.5 now; 20 of 22.5 without the down-weight
        rng = random.Random(5)
        allc = collections.Counter(scenarios.aimed_spec(self.aim("all"), rng, i)[1]["target"] for i in range(400))
        self.assertGreater(allc["relic:BURNING_BLOOD"], 300)


CHOICE_INFO = {**INFO, "relics": {**INFO["relics"], "SR1": "TransformSelect", "SR2": "TransformSelect", "SR3": "TransformSelect"}}
FEATURES = {"card:IC3": ["cmd:CardSelectCmd.FromHand"], "card:IC4": ["cmd:CardSelectCmd.FromCombatPile", "hook:AfterCardPlayed"],
            "card:IC5": ["cmd:DamageCmd.Attack"], "potion:P1": ["cmd:CardSelectCmd.FromSimpleGrid"]}


class ChoiceSpecTests(unittest.TestCase):
    def setUp(self):
        self.content = scenarios.Content(EXPORT, CHOICE_INFO, ID_MAP)
        self.pools = scenarios.ChoicePools(self.content, FEATURES)

    def specs(self, n=80, seed=2, edges="random"):
        rng = random.Random(seed)
        return [scenarios.choice_spec(self.content, self.pools, {"items": {}}, rng, i, edges) for i in range(n)]

    def test_pools_come_from_the_follow_up_and_the_models_commands(self):
        self.assertEqual(["SR1", "SR2", "SR3"], self.pools.relics)
        self.assertEqual(["IC3", "IC4"], self.pools.cards)
        self.assertEqual(["P1"], self.pools.potions)

    def test_without_features_the_cards_fall_back_to_the_known_ones(self):
        pools = scenarios.ChoicePools(self.content, None)
        self.assertEqual([], pools.cards)  # none of the fallback names is in this small export
        self.assertEqual(["SR1", "SR2", "SR3"], pools.relics)

    def test_item_order_pools_and_the_edge_key(self):
        for s in self.specs():
            cmds = [it["cmd"] for it in s["setup"]]
            f = cmds.index("fight")
            self.assertEqual("deck_set", cmds[0])
            self.assertTrue(all(c in ("card", "relic", "potion") for c in cmds[1:f]))
            self.assertTrue(all(c == "card" for c in cmds[f + 1:]))  # screen-opening cards start in the hand
            for it in s["setup"][f + 1:]:
                self.assertEqual("Hand", it["pile"])
                self.assertIn(it["id"], self.pools.cards)
            self.assertIn(s["choice_edges"], scenarios.EDGE_NAMES)
            self.assertTrue(set(s["setup"][0]["cards"][0]) == {"id", "upgraded"})
            self.assertLessEqual(len(s["setup"][0]["cards"]), 11)  # small decks
            self.assertGreaterEqual(len(s["setup"][0]["cards"]), 1)
            self.assertTrue(any(it["cmd"] == "relic" and it["id"] in self.pools.relics for it in s["setup"]))
            self.assertEqual([], scenarios.spec_flags(s, self.content))
            self.assertEqual(f"choices:", s["tag"][:8])

    def test_every_edge_is_drawn_and_a_fixed_edge_is_kept(self):
        self.assertEqual(set(scenarios.EDGE_NAMES), {s["choice_edges"] for s in self.specs(100)})
        self.assertEqual({"max"}, {s["choice_edges"] for s in self.specs(10, edges="max")})
        self.assertEqual({"cycle"}, {s["choice_edges"] for s in self.specs(10, edges="cycle")})

    def test_deterministic_and_a_shrunk_subset_keeps_its_edge(self):
        a = [json.dumps(s, sort_keys=True) for s in self.specs(20, 7)]
        self.assertEqual(a, [json.dumps(s, sort_keys=True) for s in self.specs(20, 7)])
        import shrink  # noqa: PLC0415
        spec = self.specs(1)[0]
        game = {"steps": [{"action": None}, {"action": {"kind": "end_turn"}}]}
        case = shrink.Case(spec, game, 1, ".hand")
        self.assertEqual(spec["choice_edges"], case.subset_spec(case.items[:1])["choice_edges"])



class AnalyzeAndTraceTests(unittest.TestCase):
    """--mode analyze on a finished batch.py batch, the order of its outputs, and trace calls cut into safe points."""

    def setUp(self):
        import contextlib
        import io
        import unittest.mock as mock
        self.mock, self.io, self.contextlib = mock, io, contextlib
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.addCleanup(self.td.cleanup)

    def write_batch(self, name="b0", n=2):
        specs = [{"kind": "combat", "seed": f"SEED{i:06d}", "deck_add": ["IC1"]} for i in range(n)]
        for ext, doc in ((".json", {"runs": specs}), (".out.json", {"results": [{"seed": s["seed"], "steps": [{}]} for s in specs]}),
                         (".emu.json", {"results": [{} for _ in specs]}), (".reanchor.json", {"results": [{} for _ in specs]})):
            (self.dir / f"{name}{ext}").write_text(json.dumps(doc), encoding="utf-8")
        (self.dir / f"{name}.summary.txt").write_text("SPEED 1\nbatch b0: 2 fights, network [], real profile touched False\n", encoding="utf-8")
        return specs

    def run_main(self, *extra, pack_error=True):
        content = scenarios.Content(EXPORT, INFO, ID_MAP)
        verdicts = [{"seed": "SEED000000", "tag": None, "flags": [], "identical": True},
                    {"seed": "SEED000001", "tag": None, "flags": [], "identical": False, "step": 3, "field": ".hand", "card": "IC1",
                     "diff": [], "mirror_gap": []}]
        argv = ["scenarios.py", "--name", "b0", "--mode", "analyze", "--jobs", str(self.dir), *extra]
        boom = self.mock.patch("tools.fidelity.context_pack.run", side_effect=RuntimeError("boom") if pack_error else None)
        with self.mock.patch.object(sys, "argv", argv), self.mock.patch.object(scenarios, "load_content", return_value=content), \
                self.mock.patch.object(scenarios, "judge", return_value=verdicts), boom, self.mock.patch.object(scenarios, "JOBS", self.dir), \
                self.contextlib.redirect_stdout(self.io.StringIO()) as out:
            rc = scenarios.main()
        return rc, out.getvalue()

    def test_verdicts_and_summary_are_written_before_the_packs_and_a_pack_failure_is_a_warning(self):
        self.write_batch()
        rc, out = self.run_main()
        self.assertEqual(0, rc)
        verdicts = json.loads((self.dir / "b0.verdicts.json").read_text(encoding="utf-8"))
        self.assertEqual(["SEED000000", "SEED000001"], [v["seed"] for v in verdicts])
        text = (self.dir / "b0.summary.txt").read_text(encoding="utf-8")
        self.assertIn("warning: context packs not written (RuntimeError: boom)", text)
        self.assertTrue(text.startswith("SPEED 1\nbatch b0: 2 fights, network [], real profile touched False\n\n# analysis\n"))  # batch.py's own text stays
        self.assertIn("2 fights: 1 identical, 1 diverging", text)

    def test_a_second_analysis_replaces_the_first_not_stacks(self):
        self.write_batch()
        self.run_main()
        self.run_main(pack_error=False)
        text = (self.dir / "b0.summary.txt").read_text(encoding="utf-8")
        self.assertEqual(1, text.count("# analysis"))
        self.assertNotIn("warning: context packs not written", text)
        self.assertIn("context packs:", text)

    def test_analysis_refuses_a_batch_that_is_not_finished(self):
        self.write_batch()
        (self.dir / "b0.emu.json").write_text(json.dumps({"results": [{}]}), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_main()

    def test_traces_go_in_calls_of_a_chunk_and_stop_at_the_deadline(self):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return self.mock.Mock(returncode=0)

        seeds = [f"S{i:09d}" for i in range(7)]
        with self.mock.patch.object(scenarios, "JOBS", self.dir), self.mock.patch.object(scenarios.subprocess, "run", fake_run):
            msg = scenarios.trace_batch("b0", "game", 8, 100, self.dir / "t.log", seeds=seeds, chunk=3, max_servers=8)
            self.assertEqual(3, len(calls))
            self.assertEqual(["--max", "3"], calls[0][calls[0].index("--max"):calls[0].index("--max") + 2])
            self.assertEqual(seeds[:3], [calls[0][i + 1] for i, x in enumerate(calls[0]) if x == "--seed"])
            self.assertEqual(seeds[6:], [calls[2][i + 1] for i, x in enumerate(calls[2]) if x == "--seed"])
            self.assertEqual("8", calls[0][calls[0].index("--servers") + 1])
            self.assertIn("traces: 0 fights (exit 0)", msg)
            calls.clear()
            msg = scenarios.trace_batch("b0", "game", 8, 100, self.dir / "t.log", seeds=seeds, chunk=3, deadline=0.0)
            self.assertEqual(1, len(calls))  # the first call always runs; the deadline stops the next
            self.assertIn("stopped at the time limit after 3 of 7", msg)
            calls.clear()
            scenarios.trace_batch("b0", "game", 8, 4, self.dir / "t.log", seeds=seeds, chunk=3)  # at most n fights
            self.assertEqual([3, 1], [sum(1 for x in c if x == "--seed") for c in calls])
            calls.clear()
            scenarios.trace_batch("b0", "game", 8, 40, self.dir / "t.log")  # no seeds: one call, as before
            self.assertEqual(1, len(calls))
            self.assertNotIn("--seed", calls[0])
            self.assertEqual("3", calls[0][calls[0].index("--servers") + 1])  # the default cap on servers stays 3

    def test_trace_order_puts_plausible_and_early_divergences_first(self):
        import trace_divergent  # noqa: PLC0415
        found = {"A": (0, 9, "x"), "B": (1, 2, "x"), "C": (2, 1, "x"), "D": (3, 5, "x")}
        verdicts = [{"seed": "A"}, {"seed": "B"}, {"seed": "C", "flags": ["cross_act"]}, {"seed": "D"}]
        with self.mock.patch.object(trace_divergent, "targets", return_value=found):
            self.assertEqual(["B", "D", "A", "C"], scenarios.trace_order([], [], {}, verdicts, {}))


if __name__ == "__main__":
    unittest.main()
