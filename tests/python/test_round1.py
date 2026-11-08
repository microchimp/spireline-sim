"""The round's code (rl/round1*.py, docs/rl/ROUND.md): the config, the label schema's helpers, the refit-prior loss pieces, the refit-judgement
pieces, the job chain's file handoffs and the fragment. Pure numpy / torch on synthetic data: nothing here plays the game or reads the corpus."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import round1  # noqa: E402
import round1_data as rd  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover
    torch = None


def rec(a: dict, k: int = 2, seed: str = "S1", g=None) -> dict:
    played = sorted(a)
    g = g or [[float(a[i])] * k for i in played]
    cont = [{"o": o, "k": kk, "G": g[j][kk], "end": "death", "floor": 10} for j, o in enumerate(played) for kk in range(k)]
    return {"id": f"{seed}|d001|card|f03", "kind": "card", "act": 0, "k": k, "played": played, "A": {str(i): v for i, v in a.items()},
            "continuations": cont, "run": {"seed": seed, "job": "j"}, "taken": played[0]}


class ConfigTests(unittest.TestCase):
    def test_defaults_file_matches_code(self):
        self.assertEqual(json.loads((ROOT / "rl" / "round1_config.json").read_text(encoding="utf-8")), rd.DEFAULTS)

    def test_memo_defaults(self):
        c = rd.load_config()
        self.assertEqual(c["label"]["k"], 2)  # memo 2.1: K 2
        self.assertEqual(sum(c["label"]["quota"].values()), 1200)  # the round's 1,200 labels
        self.assertEqual(c["judgement"]["human_weights"], [0.25, 1.0, 4.0])  # memo 3.4
        self.assertEqual(len(c["judgement"]["seeds"]), 3)
        self.assertEqual(c["prior"]["lambda"], 0.5)
        self.assertEqual(c["prior"]["label_gap_se"], 2.0)  # memo 3.5: clear winners, gap over 2 SE
        self.assertEqual(c["block"]["seeds"], 3000)
        self.assertEqual(c["block"]["keep_rule"], "v2")

    def test_unknown_key_and_bad_values_are_refused(self):
        with self.assertRaises(ValueError):
            rd.load_config(prior={"kl_betta": 1.0})
        with self.assertRaises(ValueError):
            rd.load_config(prior={"lambda_rule": "best"})
        with self.assertRaises(ValueError):
            rd.load_config(judgement={"fight_net": "retrain"})

    def test_file_overrides_one_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"prior": {"kl_beta": 3.0}, "_note": "x"}), encoding="utf-8")
            c = rd.load_config(p)
        self.assertEqual(c["prior"]["kl_beta"], 3.0)
        self.assertEqual(c["prior"]["lr"], rd.DEFAULTS["prior"]["lr"])

    def test_heldout_is_stable_and_about_the_fraction(self):
        xs = [rd.heldout(f"S{i}", 0.2) for i in range(4000)]
        self.assertEqual(xs, [rd.heldout(f"S{i}", 0.2) for i in range(4000)])
        self.assertAlmostEqual(sum(xs) / len(xs), 0.2, delta=0.03)


class LabelTests(unittest.TestCase):
    def test_pooled_sd_and_clear_winner(self):
        # option 0 beats option 1 by 10 on every draw, option 2 sits between with noisy draws
        r = rec({0: 5.0, 1: -5.0}, k=4, g=[[10.0, 12.0, 8.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
        sd = rd.pooled_pair_sd([r])
        self.assertAlmostEqual(sd, math.sqrt(sum((x - 10) ** 2 for x in (10, 12, 8, 10)) / 3), places=6)  # the paired differences' sd
        self.assertEqual(rd.clear_winner(r, sd, 2.0), 0)  # gap 10 > 2 * 2.31 / 2
        self.assertIsNone(rd.clear_winner(r, 30.0, 2.0))  # a noisy label names no winner
        self.assertIsNone(rd.clear_winner(r, None, 2.0))
        self.assertIsNone(rd.clear_winner(rec({0: 0.1, 1: -0.1}), 5.0, 2.0))

    def test_leg_picks_pairs_arms_and_hides_small_cells(self):
        items = []
        for i in range(30):
            items.append(rd.Item(rows=np.zeros((3, 4), np.float32), screen="card", act=0, seed=f"s{i}", chosen=0, played=[0, 1, 2],
                                 A={0: -1.0, 1: 2.0, 2: -1.0}, k=2, kinds=["card.take"] * 3, groups=[0, 1, 2]))
        vals = [[0.0, 1.0, 0.0]] * 30  # V prefers option 1, which the labels value
        good, bad = [[0.0, 0.0, 0.0]] * 30, [[0.0, -9.0, 0.0]] * 30
        leg = rd.leg_picks(items, {"judgement": (vals, good, 0.0), "wrong_prior": (vals, bad, math.inf)})
        self.assertEqual(leg["all"]["n"], 30)
        self.assertAlmostEqual(leg["all"]["judgement"][0], 2.0)
        self.assertAlmostEqual(leg["all"]["wrong_prior_minus_judgement"]["mean"], -3.0)  # the prior alone picks a -1 option
        self.assertNotIn("rest", leg)  # no rest decisions: no cell
        self.assertNotIn("act2", leg)
        self.assertAlmostEqual(leg["all"]["best_option"][0], 2.0)
        self.assertAlmostEqual(leg["all"]["players_own_pick"][0], -1.0)


@unittest.skipIf(torch is None, "torch")
class PriorTests(unittest.TestCase):
    def setUp(self):
        import human_prior

        human_prior.stack()  # the fork's feature code, with a stub native layer when no build is loaded
        import round1_prior as rp

        self.rp = rp
        self.rng = np.random.default_rng(1)

    def items(self, n=40, winner=True):
        out = []
        for i in range(n):
            rows = self.rng.normal(size=(3, 1863)).astype(np.float32)
            rows[:, 1500:] = 0.0
            it = rd.Item(rows=rows, screen="card", act=0, seed=f"s{i}", chosen=1, played=[0, 1, 2], A={0: 0.0, 1: 3.0, 2: 0.0}, k=2)
            it.winner = 1 if winner else None
            out.append(it)
        return out

    def test_sets_take_the_chosen_option_or_the_label_winner(self):
        rs = self.rp.sets_from_items(self.items())
        self.assertEqual((len(rs), rs.D.shape[1], rs.C.shape[1]), (40, 350, 1513))
        self.assertTrue((rs.Ch == 1).all())
        self.assertEqual(len(self.rp.sets_from_items(self.items(winner=False), label_winners=True)), 0)
        self.assertEqual(len(self.rp.sets_from_items(self.items(), label_winners=True)), 40)

    def test_kl_anchor_is_zero_at_the_old_prior_and_positive_away(self):
        t = torch
        z = t.tensor([[1.0, 2.0, -1.0], [0.5, 0.0, -1e9]])
        valid = t.tensor([[True, True, True], [True, True, False]])
        self.assertTrue(float(self.rp.kl_old_new(t, z, z, valid).abs().max()) < 1e-6)
        kl = self.rp.kl_old_new(t, z + t.tensor([[0.0, -1.0, 1.0], [2.0, 0.0, 0.0]]), z, valid)
        self.assertTrue(bool((kl > 0).all()))

    def test_set_loss_moves_toward_the_target_and_the_anchor_resists(self):
        rs = self.rp.sets_from_items(self.items())
        sg = self.rp.SetGpu(rs, torch.device("cpu"))

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(350 + 1513, 1)

            def forward(self, dense, counts):
                return self.lin(torch.cat([dense, counts], 1))[:, 0]

        def run(beta):
            torch.manual_seed(0)
            net, old = Net(), Net()
            old.load_state_dict(net.state_dict())
            opt = torch.optim.SGD(net.parameters(), lr=0.005)
            idx = torch.arange(len(rs))
            for _ in range(60):
                ce, z, valid = sg.ce(net, idx)
                with torch.no_grad():
                    z_old, _ = sg.logits(old, idx)
                loss = ce.mean() + beta * self.rp.kl_old_new(torch, z, z_old, valid).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                z, valid = sg.logits(net, idx)
                z_old, _ = sg.logits(old, idx)
                return float(sg.ce(net, idx)[0].mean()), float(self.rp.kl_old_new(torch, z, z_old, valid).mean())

        ce0, kl0 = run(0.0)
        ce1, kl1 = run(5.0)
        self.assertLess(ce0, math.log(3))  # it learned the target
        self.assertLess(kl1, kl0)  # the anchor kept it nearer the old prior
        self.assertGreater(ce1, ce0)

    def test_choose_lambda_fixed_and_labels(self):
        cfg = rd.load_config()
        leg = {"all": {f"new_lambda_{g:g}": [v, 0.1] for g, v in zip(cfg["prior"]["lambda_grid"], (0.5, 0.6, 0.7, 1.4), strict=True)}}
        leg["all"].update({"new_blend": [0.7, 0.1], "new_blend_minus_old_blend": {"mean": 0.0, "se": 0.1},
                           "new_lambda_1_minus_old_blend": {"mean": 0.7, "se": 0.1}})
        self.assertEqual(self.rp.choose_lambda(cfg, leg)[0], 0.5)
        cfg["prior"]["lambda_rule"] = "labels"
        self.assertEqual(self.rp.choose_lambda(cfg, leg)[0], 1.0)
        leg["all"]["new_lambda_1_minus_old_blend"]["mean"] = 0.05
        self.assertEqual(self.rp.choose_lambda(cfg, leg)[0], 0.5)  # not beyond one standard error


@unittest.skipIf(torch is None, "torch")
class JudgementTests(unittest.TestCase):
    def setUp(self):
        import human_prior

        human_prior.stack()

    def test_clears_logloss_reads_the_two_heads_from_the_floor_distribution(self):
        import round1_judgement as rj

        n = 60
        logits = np.zeros((n, 53), np.float32)
        logits[:, 40] = 20.0  # certain: the run ends past floor 32
        floor = np.where(np.arange(n) < 30, 10, 25)
        final = np.full(n, 41)
        ll = rj.clears_logloss(logits, final, floor)
        self.assertLess(ll["clears_act1"], 1e-3)
        self.assertLess(ll["clears_act2"], 1e-3)
        final2 = np.full(n, 12)  # a run that died at 12 although the net said past 32
        self.assertGreater(rj.clears_logloss(logits, final2, floor)["clears_act1"], 5.0)

    def test_source_input_is_inert_at_self_and_folds_nothing_into_play(self):
        import round1_judgement as rj
        from run_train import models

        net = rj.source_net_class()(256, 0.1)
        net.eval()
        plain = models.RunValueNet(256, 0.1)
        plain.load_state_dict({k: v for k, v in net.state_dict().items() if k != "src_w"})
        plain.eval()
        d, c = torch.randn(5, 350), torch.rand(5, 1513)
        with torch.no_grad():
            a, b = net(d, c, torch.zeros(5))[0], plain(d, c)[0]
            net.src_w.data += 1.0
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
            self.assertTrue(torch.allclose(net(d, c, torch.zeros(5))[0], b, atol=1e-6))  # self rows: input 0
            self.assertFalse(torch.allclose(net(d, c, torch.ones(5))[0], b, atol=1e-3))  # human rows: it acts

    def test_average_states_is_the_mean(self):
        import round1_judgement as rj

        s = rj.average_states([{"a": torch.zeros(2)}, {"a": torch.full((2,), 4.0)}])
        self.assertTrue(torch.allclose(s["a"], torch.full((2,), 2.0)))

    def test_train_one_fits_human_rows_only_through_the_source_column(self):
        import round1_judgement as rj
        from run_train import models

        rng = np.random.default_rng(0)
        n = 64
        self_d = {"dense": rng.normal(size=(n, 350)).astype(np.float16), "counts": np.zeros((n, 1513), np.float16),
                  "target": np.eye(53, dtype=np.float32)[rng.integers(0, 20, n)], "end_hp": np.full(n, 0.5, np.float32),
                  "val": np.arange(n) % 8 == 0, "final": rng.integers(1, 30, n), "floor": rng.integers(1, 32, n)}
        hum = {"dense": rng.normal(size=(32, 350)).astype(np.float16), "counts": np.zeros((32, 1513), np.float16),
               "cls": rng.integers(30, 52, 32).astype(np.int64), "n_decisions": np.int64(32)}
        cfg = rd.load_config(judgement={"epochs": 1, "batch": 32, "device": "cpu"})
        torch.manual_seed(0)
        base = {k: v.clone() for k, v in models.RunValueNet(256, 0.1).state_dict().items()}
        base["src_w"] = torch.zeros(256)
        out = rj.train_one(cfg, base, self_d, hum, 1.0, 1, torch.device("cpu"))
        self.assertEqual(set(out["state"]), set(base))
        self.assertGreater(float(out["state"]["src_w"].abs().sum()), 0.0)  # the human rows moved the source column
        score = rj.held_out_score(out["state"], self_d)
        self.assertIn("clears_act1", score)


class ChainTests(unittest.TestCase):
    def test_every_step_reads_only_what_an_earlier_step_leaves(self):
        left: set[str] = set()
        for s in round1.STEPS:
            needs, leaves = round1.IO[s]
            self.assertTrue(set(needs) <= left, (s, set(needs) - left))
            left |= set(leaves)

    def test_fragment_has_one_entry_per_step_and_names_its_files(self):
        cfg = rd.load_config()
        frag = round1.fragment(cfg, Path("D:/x/round"), "D:/x/cfg.json")
        self.assertEqual([k for k in frag if not k.startswith("_")], [f"round-{s}" for s in round1.STEPS])
        for s in round1.STEPS:
            e = frag[f"round-{s}"]
            self.assertIn(s, e["cmd"])
            self.assertIn("rl/round1.py", e["cmd"])
            self.assertFalse(e["enabled"])
            self.assertEqual(e["leaves"], [(Path("D:/x/round") / r).as_posix() for r in round1.IO[s][1]])
        self.assertIn("gpu", frag["round-refit-prior"]["cmd"][0].lower())
        json.dumps(frag)

    def test_promote_commands_parse_with_gen0s_own_parser(self):
        import gen0

        cfg = rd.load_config()
        ctx = round1.Ctx(cfg, Path("D:/x/round"))
        for which in ("smoke", "block"):
            cmd = round1.promote_cmd(ctx, which, Path("D:/x/agent/gen14/candidate-round1"), f"D:/x/{which}.json")
            a = gen0.build_parser().parse_args(cmd[2:])
            self.assertTrue(a.sequential)
            self.assertEqual(a.keep_rule, "v2")
        self.assertEqual(gen0.build_parser().parse_args(round1.promote_cmd(ctx, "block", Path("c"), "s.json")[2:]).batch_seeds, 300)

    def test_step_refuses_without_its_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = round1.Ctx(rd.load_config(), Path(d))
            with self.assertRaises(SystemExit):
                ctx.need("refit-prior")

    def test_done_records_input_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = round1.Ctx(rd.load_config(), Path(d))
            for rel in round1.IO["play"][1]:
                ctx.p(rel).parent.mkdir(parents=True, exist_ok=True)
                ctx.p(rel).write_text("{}", encoding="utf-8")
            got = ctx.done("play")
            self.assertEqual(list(got["outputs"]), round1.IO["play"][1])
            self.assertIn("play", json.loads(ctx.p("chain.json").read_text(encoding="utf-8"))["steps"])


class StepPlumbingTests(unittest.TestCase):
    def test_dry_run_config_is_valid_and_tiny(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = round1.Ctx(rd.load_config(), Path(d), dry=True)
            self.assertEqual(ctx.dir, Path(d) / "dry")
            self.assertEqual(ctx.cfg["label"]["k"], 1)
            self.assertEqual(ctx.cfg["play"]["runs"], 2)
            self.assertEqual(ctx.cfg["judgement"]["device"], "cpu")
            rd.validate(ctx.cfg)

    def test_exploring_runs_pick_one_player_by_schedule_macro_and_stakes(self):
        import hindsight

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            def job(name, config, macro, stakes, gen=13):
                j = root / name
                j.mkdir()
                (j / "job.json").write_text(json.dumps({"params": {"cmd": ["x", "--explore-tau", "0.1"]}, "rc": 0}), encoding="utf-8")
                (j / "summary.json").write_text(json.dumps({"agent": {"generation": gen, "prior_sha256": "p", "macro": macro, "stakes": stakes,
                                                                      "tau": 0.1, "eps": 0.02}}), encoding="utf-8")
                (j / "runs.jsonl").write_text(json.dumps({"seed": name, "ix": 0, "end": "death", "config": config}) + "\n", encoding="utf-8")

            job("a-selfplay", "a3-v1@h1e4b16", "m", "a3-v1@h1e4b16")
            job("b-selfplay", "a3-v1", "m", None)  # gen 11's plain player
            job("c-selfplay", "a3-v1@h1e4b16", "other", "a3-v1@h1e4b16")  # another macro
            got = hindsight.exploring_runs(root, 13, "p", config="a3-v1@h1e4b16", macro="m", stakes="a3-v1@h1e4b16")
            self.assertEqual([r["_job"] for r in got], ["a-selfplay"])
            self.assertEqual([r["_job"] for r in hindsight.exploring_runs(root, 13, "p", config="a3-v1")], ["b-selfplay"])  # the default is as it was

    def test_consolidate_labels_writes_the_documented_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "labels.d").mkdir()
            with open(out / "labels.d" / "S1.jsonl", "w", encoding="utf-8") as fh:
                for i in range(3):
                    r = rec({0: 1.0, 1: -1.0}, seed=f"S{i}")
                    r["id"] = f"S{i}|d001|card|f03"
                    fh.write(json.dumps(r) + "\n")
                fh.write(json.dumps({**rec({0: 1.0, 1: -1.0}, seed="X"), "id": "X|d002|card|f05", "refused": "no_return:timeout"}) + "\n")
            L = rd.load_config()["label"]
            round1.consolidate_labels(out, L, "a3-v1@h1e4b16")
            rows = [json.loads(x) for x in (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)  # the refused decision is dropped
            self.assertTrue(all(r["schema"] == rd.LABEL_SCHEMA and r["schedule"] == "a3-v1@h1e4b16" and "heldout" in r for r in rows))
            summ = json.loads((out / "label-summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summ["decisions"], summ["cells"]), (3, {"card/a1": 3}))


if __name__ == "__main__":
    unittest.main()
