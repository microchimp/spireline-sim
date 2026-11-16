"""The ranking check of the V_run keep test (D47, D49): the fixed set of held-out decisions, the measures on the same states, the
rule, and the keep path that reads it (`run_train.ranking`, `gen_loop.vrun_keep_passed`, `fit_run_value_matched`)."""

import copy
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT, ROOT / "tests" / "python"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")

import gen_loop  # noqa: E402
from run_train import features, models, ranking, shards, vocab  # noqa: E402

STRIKE, DEFEND, BASH, INFLAME, SHRUG, ANGER = 472, 131, 30, 265, 433, 13
SMALL_MAP = (
    features.MapNode(3, 0, vocab.NODE_ANCIENT, ((2, 1), (4, 1))),
    features.MapNode(2, 1, vocab.NODE_MONSTER, ((2, 2), (3, 2))),
    features.MapNode(4, 1, vocab.NODE_UNKNOWN, ((3, 2),)),
    features.MapNode(2, 2, vocab.NODE_ELITE, ((3, 3),)),
    features.MapNode(3, 2, vocab.NODE_REST, ((3, 3),)),
    features.MapNode(3, 3, vocab.NODE_SHOP, ((3, 4),)),
    features.MapNode(3, 4, vocab.NODE_BOSS, ()),
)


def view(hp: int = 50, act: int = 0) -> features.RunView:
    return features.RunView(
        ascension=10, act_index=act, act_region=vocab.REGION_OVERGROWTH, floor=3, hp=hp, max_hp=80, gold=120,
        deck=(features.CardRef(STRIKE, False), features.CardRef(DEFEND, False), features.CardRef(BASH, False)),
        relics=(vocab.relic_id_by_name("BurningBlood"),), potions=(), max_potion_slots=2, map_nodes=SMALL_MAP, position=(3, 0), boss=83)


def entry(kind: str, v: features.RunView, options: list[features.Option]) -> dict:
    return {"i": 0, "type": kind, "floor": v.floor, "state": shards.view_to_json(v, 0),
            "options": [shards.option_to_json(o) for o in options], "chosen": 0, "values": [0.0] * len(options)}


def rest_entry(v: features.RunView) -> dict:
    deck = v.deck
    return entry("rest", v, [features.Option("rest.heal", 0)] + [features.Option("rest.smith", 1, deck_index=i) for i in range(len(deck))])


def card_entry(v: features.RunView) -> dict:
    cards = [features.Option("card.take", i, card=features.CardRef(c, False)) for i, c in enumerate((INFLAME, SHRUG, ANGER))]
    return entry("card_reward", v, cards + [features.Option("card.skip", 3)])


def map_entry(v: features.RunView, with_rest: bool) -> dict:
    nodes = [(2, 1, vocab.NODE_MONSTER), (3, 2, vocab.NODE_REST) if with_rest else (4, 1, vocab.NODE_UNKNOWN)]
    return entry("map", v, [features.Option("map.node", i, node=n) for i, n in enumerate(nodes)])


def forced_entry(v: features.RunView) -> dict:
    return entry("map", v, [features.Option("map.node", 0, node=(2, 1, vocab.NODE_MONSTER))])


def run_record(seed: str, hp: int = 50) -> dict:
    v = view(hp)
    decisions = [forced_entry(v), rest_entry(v), card_entry(v), map_entry(v, True), map_entry(v, False), rest_entry(view(hp + 10, 1)),
                 card_entry(v), map_entry(v, True), entry("shop", v, [features.Option("shop.leave", 0)])]
    for i, d in enumerate(decisions):
        d["i"] = i
    return {"schema": 3, "seed": seed, "generation": 9, "maps": [shards.map_to_json(v)], "decisions": decisions}


def write_job(root: Path, job: str, generation: int, shard_seeds: list[list[str]]) -> None:
    folder = root / job / "shards"
    folder.mkdir(parents=True)
    names = []
    for k, seeds in enumerate(shard_seeds):
        shards.write_shard(folder / f"r{k:04d}.jsonl.gz", [run_record(s, 30 + 5 * (n % 8)) for n, s in enumerate(seeds)])
        names.append(f"r{k:04d}.jsonl.gz")
    (folder / "shards.json").write_text(json.dumps({"generation": generation, "run_shards": names}), encoding="utf-8")


SMALL_QUOTAS = {"rest": 12, "card_reward": 12, "map_rest": 12, "map_other": 6}
SMALL_PER_RUN = {"rest": 2, "card_reward": 1, "map_rest": 1, "map_other": 1}


class Pool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"
        write_job(self.root, "j9a", 9, [[f"A{i:09d}" for i in range(0, 8)], [f"A{i:09d}" for i in range(8, 16)]])
        write_job(self.root, "j9b", 9, [[f"B{i:09d}" for i in range(8)]])
        write_job(self.root, "j11", 11, [[f"C{i:09d}" for i in range(8)]])
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **kw):
        kw.setdefault("quotas", SMALL_QUOTAS)
        kw.setdefault("per_run", SMALL_PER_RUN)
        return ranking.build_set(self.root, **kw)


class SetTests(Pool):
    def test_the_same_root_and_seed_give_the_same_set_and_another_seed_another_one(self):
        doc1, items1 = self.build()
        doc2, items2 = self.build()
        self.assertEqual(doc1, doc2)
        self.assertEqual([i.ref for i in items1], [i.ref for i in items2])
        self.assertEqual(doc1["digest"], ranking.set_digest(doc1))
        other, _ = self.build(seed=ranking.SET_SEED + 1)
        self.assertNotEqual(doc1["digest"], other["digest"])

    def test_quotas_per_run_limits_and_the_pool_generation_are_kept(self):
        doc, items = self.build()
        self.assertEqual(doc["counts"], SMALL_QUOTAS)
        self.assertTrue(all(not i.run_seed.startswith("C") for i in items))  # generation 11 is not in the pool
        per_run: dict = {}
        for i in items:
            per_run[(i.stratum, i.run_seed)] = per_run.get((i.stratum, i.run_seed), 0) + 1
        for (stratum, _), n in per_run.items():
            self.assertLessEqual(n, SMALL_PER_RUN[stratum])
        self.assertTrue(all(len(i.decision.options) >= 2 for i in items))  # a forced map step is never in the set
        self.assertEqual({i.stratum for i in items}, set(SMALL_QUOTAS))
        for i in items:
            kinds = {o.kind for o in i.decision.options}
            if i.stratum == "map_rest":
                self.assertIn(vocab.NODE_REST, {o.node[2] for o in i.decision.options})
            if i.stratum == "rest":
                self.assertEqual(kinds, {"rest.heal", "rest.smith"})

    def test_runs_of_the_fit_window_are_left_out(self):
        doc, items = self.build()
        leave = {i.run_seed for i in items[:6]}
        doc2, items2 = self.build(exclude_seeds=leave)
        self.assertTrue(leave.isdisjoint({i.run_seed for i in items2}))
        self.assertGreater(doc2["runs_excluded_in_fit_window"], 0)
        self.assertNotEqual(doc["digest"], doc2["digest"])

    def test_a_written_set_is_read_back_decision_for_decision_and_a_change_is_caught(self):
        doc, items = self.build()
        back = ranking.load_set(json.loads(json.dumps(doc)), self.root)
        self.assertEqual([(i.stratum, i.ref, i.run_seed, i.act) for i in items], [(i.stratum, i.ref, i.run_seed, i.act) for i in back])
        self.assertEqual([i.decision for i in items], [i.decision for i in back])
        self.assertEqual([i.view for i in items], [i.view for i in back])
        edited = copy.deepcopy(doc)
        edited["refs"][0][4] += 1
        with self.assertRaises(ValueError):
            ranking.load_set(edited, self.root)
        job, name = doc["refs"][0][1], doc["refs"][0][2]
        path = self.root / job / "shards" / name
        records = shards.read_shard(path)
        records[doc["refs"][0][3]]["seed"] = "ZZZZZZZZZZ"
        shards.write_shard(path, records)
        with self.assertRaises(ValueError):  # the shard changed under the set
            ranking.load_set(doc, self.root)


def values_for(items, heal_first: bool, rest_first: bool, skip_first: bool = False):
    """Synthetic head values: option 0 or 1 on top by decision kind (heal / rest node / skip as the 'first' choice)."""
    out = []
    for it in items:
        opts = it.decision.options
        v = [0.0] * len(opts)
        if it.stratum == "rest":
            v[0 if heal_first else 1] = 1.0
        elif it.stratum.startswith("map"):
            at = [k for k, o in enumerate(opts) if (o.node[2] == vocab.NODE_REST) == rest_first]
            v[at[0] if at else 0] = 1.0
        else:
            v[len(opts) - 1 if skip_first else 0] = 1.0
        out.append(v)
    return out


def human_leg(card=(0.30, 0.30, 0.004), rest=(0.60, 0.60, 0.005), n=5000):
    def one(t):
        return {"n": n, "candidate": t[0], "incumbent": t[1], "diff": t[0] - t[1], "se": t[2]}

    return {"measures": {"card_reward": one(card), "rest": one(rest)}}


class MeasureAndRuleTests(Pool):
    def setUp(self):
        super().setUp()
        _, self.items = ranking.build_set(self.root, quotas={"rest": 40, "card_reward": 20, "map_rest": 40, "map_other": 10},
                                          per_run=SMALL_PER_RUN | {"rest": 2})

    def judge(self, cand, inc, human="equal"):
        shard = ranking.measure(self.items, cand, inc)
        return shard, ranking.judge(shard, human_leg() if human == "equal" else human)

    def test_an_equal_head_passes(self):
        inc = values_for(self.items, True, True)
        shard, verdict = self.judge(inc, inc)
        self.assertTrue(verdict["pass"], verdict["checks"])
        self.assertEqual((shard["heal_rate"]["diff"], shard["rest_rate"]["diff"], shard["top_agreement_all"]), (0.0, 0.0, 1.0))

    def test_a_head_that_heals_less_fails_and_so_does_one_that_goes_to_rest_sites_less(self):
        inc = values_for(self.items, True, True)
        shard, verdict = self.judge(values_for(self.items, False, True), inc)
        self.assertFalse(verdict["pass"])
        self.assertEqual([c["name"] for c in verdict["checks"] if not c["pass"]], ["heal_rate"])
        self.assertLess(shard["heal_rate"]["diff"], -0.9)
        shard, verdict = self.judge(values_for(self.items, True, False), inc)
        self.assertFalse(verdict["pass"])
        self.assertEqual([c["name"] for c in verdict["checks"] if not c["pass"]], ["rest_rate"])
        self.assertLess(shard["rest_rate"]["diff"], -0.9)
        self.assertLess(shard["top_agreement_all"], 1.0)

    def test_a_head_that_rests_more_than_the_incumbent_passes(self):
        shard, verdict = self.judge(values_for(self.items, True, True), values_for(self.items, False, False))
        self.assertTrue(verdict["pass"], verdict["checks"])

    def test_the_margin_is_a_fixed_number_of_points_of_a_rate(self):
        base = {"n": 400, "incumbent": 0.50, "se": 0.01}
        for drop, ok in ((0.04, True), (0.05, True), (0.06, False)):
            shard = {"rest_rate": {**base, "candidate": 0.5 - drop, "diff": -drop}, "heal_rate": {**base, "candidate": 0.5, "diff": 0.0}}
            self.assertIs(ranking.judge(shard, human_leg())["pass"], ok, drop)
        shard = {"rest_rate": {**base, "candidate": 0.5, "diff": 0.0}, "heal_rate": {**base, "candidate": 0.43, "diff": -0.07}}
        self.assertFalse(ranking.judge(shard, human_leg(), margins={"rest_rate": 0.05, "heal_rate": 0.05})["pass"])
        self.assertTrue(ranking.judge(shard, human_leg(), margins={"rest_rate": 0.05, "heal_rate": 0.10})["pass"])

    def test_human_agreement_may_fall_two_standard_errors_and_no_more(self):
        shard = ranking.measure(self.items, values_for(self.items, True, True), values_for(self.items, True, True))
        self.assertTrue(ranking.judge(shard, human_leg(rest=(0.595, 0.60, 0.005)))["pass"])  # -1 SE
        self.assertTrue(ranking.judge(shard, human_leg(rest=(0.590, 0.60, 0.005)))["pass"])  # -2 SE
        bad = ranking.judge(shard, human_leg(card=(0.30, 0.31, 0.004)))  # -2.5 SE at the card rewards
        self.assertFalse(bad["pass"])
        self.assertEqual([c["name"] for c in bad["checks"] if not c["pass"]], ["human_agreement_card_reward"])
        self.assertTrue(ranking.judge(shard, human_leg(card=(0.34, 0.30, 0.004)))["pass"])  # agreeing more never fails
        zero_se = ranking.judge(shard, human_leg(card=(0.299, 0.30, 0.0)))  # identical heads' paired SE is 0
        self.assertFalse(zero_se["pass"])
        self.assertTrue(ranking.judge(shard, human_leg(card=(0.30, 0.30, 0.0)))["pass"])

    def test_a_missing_or_small_human_leg(self):
        shard = ranking.measure(self.items, values_for(self.items, True, True), values_for(self.items, True, True))
        self.assertFalse(ranking.judge(shard, None)["pass"])  # no human leg: not kept
        small = ranking.judge(shard, human_leg(card=(0.0, 0.5, 0.01), n=ranking.HUMAN_MIN_DECISIONS - 1))
        self.assertTrue(small["pass"])  # under the minimum it is reported, not judged
        self.assertTrue(all("reported only" in c["why"] for c in small["checks"] if c["name"].startswith("human_")))

    def test_paired_difference_and_se(self):
        p = ranking.paired([1, 1, 0, 0], [1, 0, 0, 0])
        self.assertEqual((p["n"], p["candidate"], p["incumbent"], p["diff"]), (4, 0.5, 0.25, 0.25))
        self.assertAlmostEqual(p["se"], 0.25, places=6)
        self.assertEqual(ranking.paired([], [])["n"], 0)
        self.assertEqual(ranking.paired([1, 0, 1], [1, 0, 1])["se"], 0.0)


def checkpoint(directory: Path, seed: int, name: str) -> Path:
    import torch

    torch.manual_seed(seed)
    fight, value = models.FightNet(), models.RunValueNet()
    with torch.no_grad():
        for net in (fight, value):
            for parameter in net.parameters():
                parameter.mul_(3.0)
    return models.save(directory / f"{name}.pt", fight, value, {"generation": seed})


class RunCheckTests(Pool):
    """The whole check with two real (random) heads on the fixture shards; the human leg is a stand-in."""

    @classmethod
    def setUpClass(cls):
        cls.heads = tempfile.TemporaryDirectory()
        cls.a = checkpoint(Path(cls.heads.name), 7, "a")
        cls.b = checkpoint(Path(cls.heads.name), 8, "b")

    @classmethod
    def tearDownClass(cls):
        cls.heads.cleanup()

    def run_check(self, cand, inc, human=lambda *a, **k: human_leg(), **kw):
        with mock.patch.object(ranking, "QUOTAS", SMALL_QUOTAS), mock.patch.object(ranking, "PER_RUN", SMALL_PER_RUN):
            return ranking.run_check(cand, inc, self.root, self.out, human=human, **kw)

    def test_the_same_head_passes_and_the_set_is_written_beside_the_record(self):
        rec = self.run_check(self.a, self.a)
        self.assertTrue(rec["pass"], rec["checks"])
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["shard"]["top_agreement_all"], 1.0)
        self.assertTrue((self.out / ranking.SET_FILE).is_file() and (self.out / ranking.RECORD_FILE).is_file())
        doc = json.loads((self.out / ranking.SET_FILE).read_text())
        self.assertEqual(rec["set"]["digest"], doc["digest"])
        self.assertEqual(rec["candidate"]["sha256"], rec["incumbent"]["sha256"])

    def test_the_second_check_reads_the_same_set_instead_of_drawing_it_again(self):
        first = self.run_check(self.a, self.b)
        with mock.patch.object(ranking, "build_set", side_effect=AssertionError("drew a new set")):
            second = self.run_check(self.b, self.a)
        self.assertEqual(first["set"]["digest"], second["set"]["digest"])
        self.assertEqual(first["shard"]["decisions"], second["shard"]["decisions"])
        self.assertEqual(first["shard"]["heal_rate"]["candidate"], second["shard"]["heal_rate"]["incumbent"])  # the same states, swapped

    def test_a_stored_set_of_another_seed_or_holding_window_runs_is_drawn_again(self):
        first = self.run_check(self.a, self.a)
        stored = json.loads((self.out / ranking.SET_FILE).read_text())
        window = {stored["refs"][0][-1]}
        again = self.run_check(self.a, self.a, exclude_seeds=window)
        self.assertNotEqual(first["set"]["digest"], again["set"]["digest"])
        self.assertTrue(window.isdisjoint({r[-1] for r in json.loads((self.out / ranking.SET_FILE).read_text())["refs"]}))
        with mock.patch.object(ranking, "QUOTAS", SMALL_QUOTAS), mock.patch.object(ranking, "PER_RUN", SMALL_PER_RUN):
            self.assertTrue(ranking.set_current(stored))
            self.assertFalse(ranking.set_current(stored, window))
            with mock.patch.object(ranking, "SET_SEED", ranking.SET_SEED + 1):
                self.assertFalse(ranking.set_current(stored))

    def test_without_a_human_leg_the_check_cannot_pass(self):
        rec = self.run_check(self.a, self.a, human=None)
        self.assertFalse(rec["pass"])
        self.assertEqual([c["name"] for c in rec["checks"] if not c["pass"]], ["human_agreement"])

    def test_different_heads_get_figures_for_every_measure(self):
        rec = self.run_check(self.a, self.b)
        for name in ("heal_rate", "rest_rate", "skip_rate", "elite_rate"):
            self.assertIn("diff", rec["shard"][name])
        self.assertLess(rec["shard"]["top_agreement_all"], 1.0)
        json.dumps(rec)

    def test_the_human_tool_is_called_with_both_heads(self):
        seen = {}

        def human(cand, inc, root, out, **kw):
            seen.update(cand=cand, inc=inc, root=root, out=out)
            return human_leg()

        self.run_check(self.a, self.b, human=human)
        self.assertEqual((Path(seen["cand"]).name, Path(seen["inc"]).name), ("a.npz", "b.npz"))
        self.assertEqual(Path(seen["out"]).parent, self.out)


class HumanLegTests(unittest.TestCase):
    """The human leg runs the Spireline exam tool as another process; only its command and its failure are tested here."""

    def test_the_command_names_both_heads_the_fifth_and_this_fork(self):
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "h.json"
            out.write_text(json.dumps(human_leg()))
            tool = Path(d) / "boss_entry_exam.py"
            tool.write_text("")
            seen = {}

            def fake_run(cmd, **kw):
                seen.update(cmd=cmd, kw=kw)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.object(ranking, "HUMAN_TOOL", tool), mock.patch.object(subprocess, "run", fake_run):
                doc = ranking.human_leg("cand.npz", "inc.npz", "runs", out, max_runs=321)
            cmd = seen["cmd"]
            self.assertEqual(doc, human_leg())
            self.assertEqual((cmd[1], cmd[3]), ("-I", "agreement"))
            for flag, value in (("--candidate", "cand.npz"), ("--incumbent", "inc.npz"), ("--runs-root", "runs"), ("--max-runs", "321"),
                                ("--fork", str(ranking.FORK_ROOT)), ("--exam-ids", str(ranking.HUMAN_FIFTH_IDS))):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
            self.assertTrue((ranking.FORK_ROOT / "rl" / "run_train" / "ranking.py").is_file())

    def test_a_failing_tool_raises_with_the_end_of_its_message(self):
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            tool = Path(d) / "boss_entry_exam.py"
            tool.write_text("")
            bad = subprocess.CompletedProcess([], 1, "", "Traceback\nFileNotFoundError: the corpus")
            with mock.patch.object(ranking, "HUMAN_TOOL", tool), mock.patch.object(subprocess, "run", return_value=bad):
                with self.assertRaises(RuntimeError) as ctx:
                    ranking.human_leg("c", "i", "r", Path(d) / "h.json")
            self.assertIn("the corpus", str(ctx.exception))
            with mock.patch.object(ranking, "HUMAN_TOOL", Path(d) / "absent.py"), self.assertRaises(FileNotFoundError):
                ranking.human_leg("c", "i", "r", Path(d) / "h.json")


class KeepPathTests(unittest.TestCase):
    """`ranking_ok` reads the ranking block for the record; since D64 `vrun_keep_passed` reads the export alone."""

    def test_the_keep_test_reads_the_ranking_block(self):
        with tempfile.TemporaryDirectory() as d:
            cand = Path(d)
            (cand / "run_value.npz").write_bytes(b"x")
            ok = {"all_computable_pass": True}
            self.assertTrue(gen_loop.ranking_ok(ok))  # a record from before D49 is not judged on it
            self.assertTrue(gen_loop.ranking_ok({**ok, "ranking": {"status": "ok", "pass": True}}))
            self.assertFalse(gen_loop.ranking_ok({**ok, "ranking": {"status": "ok", "pass": False}}))
            self.assertFalse(gen_loop.ranking_ok({**ok, "ranking": {"status": "not_run", "pass": False}}))
            self.assertFalse(gen_loop.ranking_ok({**ok, "ranking": {"status": "ok"}}))  # no verdict is not a pass
            self.assertTrue(gen_loop.ranking_ok({**ok, "ranking": ranking.no_incumbent()}))
            # D64: the block is recorded and the export decides, whatever the block says
            for keep in (ok, {**ok, "ranking": {"status": "ok", "pass": False}}, {**ok, "ranking": {"status": "not_run", "pass": False}},
                         {"all_computable_pass": False, "ranking": {"pass": True}}):
                self.assertTrue(gen_loop.vrun_keep_passed(keep, cand))
            self.assertEqual(gen_loop.macro_for({**ok, "ranking": {"pass": False}}, cand, enabled=True), "vrun")
            marked = gen_loop.keep_advisory({"all_computable_pass": True, "ranking": {"status": "ok", "pass": False}})
            self.assertEqual((marked["advisory"], marked["gated_on"], marked["old_rule_passed"]), (True, ["export"], False))
            self.assertIs(gen_loop.keep_advisory({**ok, "ranking": {"status": "ok", "pass": True}})["old_rule_passed"], True)
            (cand / "run_value.npz").unlink()
            self.assertFalse(gen_loop.vrun_keep_passed(ok, cand))  # nothing to keep without the export

    def test_a_check_that_cannot_run_is_a_failing_block_with_its_reason(self):
        with tempfile.TemporaryDirectory() as d:
            npz = Path(d) / "run_value.npz"
            npz.write_bytes(b"x")
            none = gen_loop.ranking_block(npz, None, d, Path(d))
            self.assertEqual((none["status"], none["pass"]), ("not_run", False))
            gone = gen_loop.ranking_block(npz, Path(d) / "absent.npz", d, Path(d))
            self.assertFalse(gone["pass"])

            def boom(*a, **k):
                raise OSError("no corpus")

            err = gen_loop.ranking_block(npz, npz, d, Path(d), check=boom)
            self.assertFalse(err["pass"])
            self.assertIn("no corpus", err["why"])

    def test_the_incumbent_can_be_named_three_ways(self):
        with tempfile.TemporaryDirectory() as d:
            gen = Path(d) / "gen3" / "candidate"
            gen.mkdir(parents=True)
            (gen / "run_value.npz").write_bytes(b"x")
            current = Path(d) / "current.json"
            current.write_text(json.dumps({"dir": str(gen)}))
            want = gen / "run_value.npz"
            self.assertEqual({gen_loop.incumbent_head(p) for p in (want, gen, current)}, {want})
            self.assertIsNone(gen_loop.incumbent_head(None))

    def test_the_summary_keeps_the_verdict_and_the_figures_only(self):
        block = {"status": "ok", "pass": False, "set": {"digest": "abc", "counts": {}}, "shard": {"big": 1},
                 "checks": [{"name": "heal_rate", "candidate": 0.2, "incumbent": 0.46, "diff": -0.26, "limit": -0.05, "pass": False, "n": 3000}]}
        s = gen_loop.ranking_summary({"ranking": block})
        self.assertEqual((s["pass"], s["set_digest"]), (False, "abc"))
        self.assertEqual(s["checks"][0], {"name": "heal_rate", "candidate": 0.2, "incumbent": 0.46, "diff": -0.26, "limit": -0.05, "pass": False})
        self.assertIsNone(gen_loop.ranking_summary({"all_computable_pass": True}))


class VrunFitCommandTests(unittest.TestCase):
    """`gen0.py vrun-fit` names the incumbent the ranking check needs, or refuses."""

    def args(self, **kw):
        import argparse

        base = dict(shards_root="runs", prior_sha="a" * 64, gen_dir=None, gens=None, beta=None, out="fit", incumbent=None,
                    ranking_only=False, lab=None, device=None, threads=2)
        return argparse.Namespace(**{**base, **kw})

    def doc(self, kept):
        block = {"status": "ok", "pass": kept, "checks": [{"name": "heal_rate", "pass": kept}]}
        return {"keep": {"ranking": block}, "window": {"runs_by_generation": {"11": 5}},
                "beta": 0.0, "kept": kept, "run_value": {"wall_min": 1.0}}

    def run_cmd(self, args, kept=True):
        import gen0

        with (mock.patch("sts2_gym.procutil.lower_priority"),
              mock.patch.object(gen_loop, "fit_run_value_matched", return_value=self.doc(kept)) as fit,
              mock.patch.object(gen_loop, "rerank_vrun_fit", return_value=self.doc(kept)) as rerank):
            rc = gen0.vrun_fit_cmd(args)
        return rc, fit, rerank

    def test_the_incumbent_is_named_or_the_command_refuses(self):
        with self.assertRaises(SystemExit):
            self.run_cmd(self.args())
        rc, fit, _ = self.run_cmd(self.args(incumbent="inc/run_value.npz"))
        self.assertEqual((rc, fit.call_args.kwargs["incumbent"]), (0, "inc/run_value.npz"))

    def test_the_generation_directory_is_the_default_incumbent_and_a_refused_check_exits_3(self):
        with mock.patch.object(gen_loop, "load_agent") as agent:
            agent.return_value.describe.return_value = {"prior_sha256": "b" * 64}
            rc, fit, _ = self.run_cmd(self.args(prior_sha=None, gen_dir="agent/gen11/candidate"), kept=False)
        self.assertEqual((rc, fit.call_args.kwargs["incumbent"]), (3, "agent/gen11/candidate"))

    def test_ranking_only_reruns_just_the_check(self):
        rc, fit, rerank = self.run_cmd(self.args(incumbent="inc", ranking_only=True))
        self.assertEqual(rc, 0)
        fit.assert_not_called()
        self.assertEqual(rerank.call_args.args[1], "inc")


if __name__ == "__main__":
    unittest.main()
