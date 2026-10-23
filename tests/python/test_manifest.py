"""tools/fidelity/manifest.py and manifest_codex.py: the port manifest (models the records reach, their hooks and
families, eligibility as families migrate) and the Codex counter that writes aggregates only.

Fixtures are invented (made-up ids, a few records of the shapes the readers take); no game text. Two tests use
the checked-in hook table and, when the decompile is there, the real hook list.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import manifest as mf  # noqa: E402
from tools.fidelity import manifest_codex as mc  # noqa: E402

FAM_DOC = {
    "families": {"F01": {"name": "a", "run": False}, "F02": {"name": "b", "run": False},
                 "F03": {"name": "c", "run": False}, "F16": {"name": "run", "run": True}},
    "hooks": {"HookA": "F01", "HookB": "F02", "HookC": "F03", "HookRun": "F16"},
    "unassigned": {"HookV": "visual only"}}

CAT = {
    "card:PLAIN": {"kind": "card", "class": "Plain", "combat_hooks": [], "run_hooks": [], "families": [], "run_families": []},
    "relic:RA": {"kind": "relic", "class": "Ra", "combat_hooks": ["HookA"], "run_hooks": [], "families": ["F01"], "run_families": []},
    "relic:RB": {"kind": "relic", "class": "Rb", "combat_hooks": ["HookA", "HookB"], "run_hooks": [], "families": ["F01", "F02"], "run_families": []},
    "power:PC": {"kind": "power", "class": "Pc", "combat_hooks": ["HookC"], "run_hooks": [], "families": ["F03"], "run_families": []},
    "relic:RUN": {"kind": "relic", "class": "Run", "combat_hooks": [], "run_hooks": ["HookRun"], "families": [], "run_families": ["F16"]},
}


def card(i):
    return [i, False]


def rec(*, relics=None, extra_card=None):
    st = {"hand": [card("PLAIN")], "draw": [], "discard": [], "exhaust": [],
          "enemies": [{"name": "MOB", "powers": [["PC_POWER", 2]]}], "player": {"powers": []}, "potions": [None]}
    if relics is not None:
        st["relics"] = relics
    steps = [{"action": None, "state": st}]
    st2 = dict(st, hand=[card(extra_card)] if extra_card else [], player={"powers": [["PC", 1]]})
    steps.append({"action": {"kind": "play", "card": "PLAIN"}, "state": st2})
    return {"character": "HERO", "neow_options": ["RA", "RB"], "neow_pick": 1, "steps": steps}


class FamilyMap(unittest.TestCase):
    def test_covers_every_hook_of_hook_dispatch(self):
        fam = mf.load_families()
        disp = json.loads(mf.DISPATCH.read_text(encoding="utf-8"))["hooks"]
        missing = [h for h in disp if h not in fam["hooks"]]
        self.assertEqual(missing, [])
        self.assertEqual(set(fam["hooks"].values()) - set(fam["families"]), set())
        self.assertEqual(sorted(fam["families"]), [f"F{i:02d}" for i in range(1, 17)])

    def test_a_hook_is_in_one_place(self):
        fam = mf.load_families()
        self.assertEqual(set(fam["hooks"]) & set(fam["unassigned"]), set())
        for h in fam["unassigned"]:
            self.assertTrue(fam["unassigned"][h])
        for h in fam["hooks"]:
            if h not in json.loads(mf.DISPATCH.read_text(encoding="utf-8"))["hooks"]:
                self.assertIn(h, fam["why"], f"{h} is placed by name, so it needs a reason")

    def test_dispatch_hooks_per_family_match_the_design_counts(self):
        fam = mf.load_families()
        disp = json.loads(mf.DISPATCH.read_text(encoding="utf-8"))["hooks"]
        n = {f: sum(1 for h in disp if fam["hooks"][h] == f) for f in fam["families"]}
        self.assertEqual(n, {f: v["spike_hooks"] for f, v in fam["families"].items()})

    def test_covers_every_hook_of_the_decompile_when_present(self):
        from tools.fidelity import hook_table as ht
        d = Path(ht.DEFAULT_DECOMPILED)
        if not (d / ht.MODELS_NS / "AbstractModel.cs").is_file():
            self.skipTest("no decompile here")
        fam = mf.load_families()
        left = [h for h in ht.hook_names(str(d)) if h not in fam["hooks"] and h not in fam["unassigned"]]
        self.assertEqual(left, [])

    def test_migrated_file_starts_empty(self):
        self.assertEqual(json.loads(mf.MIGRATED.read_text(encoding="utf-8")), {"migrated": []})
        self.assertEqual(mf.migrated_families(), [])

    def test_snake_ids(self):
        self.assertEqual(mf.snake("StrikeIronclad"), "STRIKE_IRONCLAD")
        self.assertEqual(mf.snake("RavenousPower"), "RAVENOUS_POWER")
        self.assertEqual(mf.snake("Akabeko"), "AKABEKO")

    def test_run_hooks(self):
        self.assertTrue(mf.hook_is_run("HookRun", FAM_DOC))
        self.assertFalse(mf.hook_is_run("HookA", FAM_DOC))
        self.assertTrue(mf.hook_is_run("HookA", FAM_DOC, {"HookA": ["run"]}))      # only outside combat
        self.assertFalse(mf.hook_is_run("HookA", FAM_DOC, {"HookA": ["combat"]}))


class Eligibility(unittest.TestCase):
    def test_every_model_needs_all_its_combat_families(self):
        item = {"models": ["relic:RB", "power:PC"], "families": ["F01", "F02", "F03"]}
        self.assertFalse(mf.eligible(item, []))
        self.assertFalse(mf.eligible(item, ["F01", "F02"]))
        self.assertTrue(mf.eligible(item, ["F01", "F02", "F03"]))
        self.assertTrue(mf.eligible(item, ["F03", "F02", "F01", "F16"]))

    def test_a_model_with_no_combat_hook_never_blocks(self):
        self.assertEqual(mf.item_families(["card:PLAIN", "relic:RUN", "relic:UNKNOWN"], CAT), [])
        self.assertTrue(mf.eligible({"models": ["card:PLAIN"], "families": []}, []))

    def test_from_models_and_a_family_map(self):
        mfam = {m: v["families"] for m, v in CAT.items()}
        item = {"models": ["relic:RB", "card:PLAIN"]}
        self.assertFalse(mf.eligible(item, ["F01"], mfam))
        self.assertTrue(mf.eligible(item, ["F01", "F02"], mfam))

    def test_one_missing_family_of_one_model_is_enough_to_block(self):
        items = [{"models": ["relic:RB"], "families": ["F01", "F02"]}, {"models": ["relic:RA"], "families": ["F01"]}]
        got = [mf.eligible(i, ["F01"]) for i in items]
        self.assertEqual(got, [False, True])


class Readers(unittest.TestCase):
    def test_corpus_models_covers_piles_creatures_actions_and_mid_fight_cards(self):
        models, known = mf.corpus_models(rec(relics=["RA", None], extra_card="CREATED"))
        self.assertTrue(known)
        self.assertEqual(models, {"card:PLAIN", "card:CREATED", "monster:MOB", "power:PC_POWER", "power:PC", "relic:RA"})

    def test_relics_unknown_when_no_step_lists_them(self):
        models, known = mf.corpus_models(rec())
        self.assertFalse(known)
        self.assertFalse(any(m.startswith("relic:") for m in models))

    def test_starter_and_neow_relics(self):
        recs = [dict(rec(relics=["START", "RA"]), character="HERO"), dict(rec(relics=["START"]), character="HERO")]
        self.assertEqual(mf.starter_relics(recs), {"HERO": "START"})
        self.assertEqual(mf.neow_relic(rec()), "RB")
        self.assertIsNone(mf.neow_relic({"neow_options": [], "neow_pick": 3}))

    def test_agent_models_from_diverging_diffs(self):
        run = {"divergences": [
            {"diff": [[".combat.hand", ['["PLAIN", 0]'], ['["PLAIN", 0]', '["CREATED", 1]']],
                      [".relics", ["RA"], []], [".hp", [3], [4]], [".potions", ["P1", None], ["P1"]]]}]}
        self.assertEqual(mf.agent_models(run), {"card:PLAIN", "card:CREATED", "relic:RA", "potion:P1"})

    def test_read_agent_runs_marks_every_run_partial(self):
        with tempfile.TemporaryDirectory() as d:
            ev = {"ev": "end", "seed": "S1", "agent_seed": [1, 2], "stats": {"decisions": 7},
                  "divergences": [{"diff": [[".deck", [], ['["PLAIN", 0]']]]}]}
            (Path(d) / "runs.jsonl").write_text(json.dumps(ev) + "\n" + json.dumps({"ev": "x"}) + "\n", encoding="utf-8")
            runs = mf.read_agent_runs(Path(d))
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0]["models"], runs[0]["decisions"], runs[0]["partial"]), (["card:PLAIN"], 7, True))


class Aggregation(unittest.TestCase):
    def build(self, migrated=()):
        corpus = [{"id": "f1", "models": ["card:PLAIN"], "relics_known": True},
                  {"id": "f2", "models": ["card:PLAIN", "relic:RA"], "relics_known": True},
                  {"id": "f3", "models": ["relic:RB", "power:PC", "card:ODD"], "relics_known": False},
                  {"id": "f4", "models": ["relic:RA", "power:PC"], "relics_known": True}]
        agent = [{"id": "a1", "models": ["relic:RA"], "decisions": 30, "partial": True},
                 {"id": "a2", "models": [], "decisions": 70, "partial": True}]
        lab = [{"id": "l1", "models": ["relic:RA"], "relics_known": True}]
        codex = {"runs": 10, "by_model": {"relic:RA": 4, "card:CODEX_ONLY": 2}}
        return mf.assemble(FAM_DOC, CAT, corpus, agent, lab, codex, list(migrated), ["n"])

    def test_model_counts_per_source(self):
        m = self.build()["models"]
        self.assertEqual((m["relic:RA"]["corpus_fights"], m["relic:RA"]["agent_runs"], m["relic:RA"]["lab_fights"],
                          m["relic:RA"]["codex_runs"]), (2, 1, 1, 4))
        self.assertEqual(m["card:PLAIN"]["corpus_fights"], 2)
        self.assertEqual(m["relic:RB"]["families"], ["F01", "F02"])
        self.assertTrue(m["card:ODD"]["unresolved"])
        self.assertTrue(m["card:CODEX_ONLY"]["unresolved"])
        self.assertEqual(m["card:CODEX_ONLY"]["codex_runs"], 2)

    def test_kind_and_family_figures(self):
        d = self.build()
        self.assertEqual(d["kinds"]["relic"]["models"], 3)
        self.assertEqual(d["kinds"]["relic"]["reached_by_corpus"], 2)
        self.assertEqual(d["kinds"]["card"]["unresolved"], 2)
        f = d["families"]
        self.assertEqual((f["F01"]["corpus_fights"], f["F01"]["corpus_fight_share_pct"]), (3, 75.0))
        self.assertEqual(f["F02"]["corpus_fights"], 1)
        self.assertEqual(f["F03"]["corpus_fights"], 2)
        self.assertEqual(f["F01"]["agent_decision_share_pct"], 30.0)
        self.assertEqual(f["F16"]["corpus_fights"], 0)
        self.assertEqual(f["F01"]["alone_unlocks_fights"], 2)             # f1 (needs nothing) and f2; f3 and f4 need more
        self.assertEqual(d["counts"]["corpus_fights_relics_unknown"], 1)
        self.assertEqual(d["unassigned_hooks"], {"HookV": "visual only"})

    def test_eligible_as_families_migrate(self):
        self.assertEqual(self.build()["eligible_now"]["corpus_fights"], 1)               # only f1 needs nothing
        d = self.build(["F01"])
        self.assertEqual((d["eligible_now"]["corpus_fights"], d["eligible_now"]["agent_runs"]), (2, 2))
        d = self.build(["F01", "F02", "F03"])
        self.assertEqual(d["eligible_now"]["corpus_fights"], 4)

    def test_greedy_orders_models_by_fights_unlocked(self):
        order = self.build()["greedy_models"]
        self.assertEqual([g["model"] for g in order][0], "relic:RA")                      # unlocks f2 at once
        self.assertEqual(order[0]["unlocks"], 1)
        self.assertEqual(order[-1]["cumulative"], 4)
        self.assertEqual([g["cumulative"] for g in order], sorted(g["cumulative"] for g in order))

    def test_greedy_cover_counts_items_needing_nothing_first(self):
        order = mf.greedy_cover([set(), {"a"}, {"a", "b"}, {"b"}, {"a", "b"}])
        self.assertEqual(order[0]["cumulative"] - order[0]["unlocks"], 1)
        self.assertEqual(order[-1]["cumulative"], 5)

    def test_render_md_and_json_are_serialisable(self):
        d = self.build()
        json.dumps(d)
        text = mf.render_md(d)
        self.assertIn("Unassigned hooks", text)
        self.assertIn("HookV", text)


class CodexAggregatesOnly(unittest.TestCase):
    LINES = [
        json.dumps({"seed": "SECRETSEED1", "run_hash": "hashhash", "ascension": 10,
                    "players": [{"character": "CHARACTER.HERO", "deck": [{"id": "CARD.AA"}, {"id": "CARD.BB"}],
                                 "relics": [{"id": "RELIC.RR"}], "note": "ignore previous instructions"}],
                    "map_point_history": [[{"rooms": [{"monster_ids": ["MONSTER.MM"], "model_id": "ENCOUNTER.EE"}]}]]}),
        json.dumps({"seed": "SECRETSEED2", "ascension": 10,
                    "players": [{"character": "CHARACTER.HERO", "deck": [{"id": "CARD.AA"}], "relics": []}]}),
        json.dumps({"seed": "SECRETSEED3", "ascension": 9,
                    "players": [{"character": "CHARACTER.HERO", "deck": [{"id": "CARD.ZZ"}]}]}),
        json.dumps({"seed": "SECRETSEED4", "ascension": 10,
                    "players": [{"character": "CHARACTER.OTHER", "deck": [{"id": "CARD.YY"}]}]}),
        "not json CHARACTER.HERO",
    ]

    def test_counts_runs_per_model_and_nothing_else(self):
        doc = mc.aggregate(self.LINES, "HERO", 10)
        self.assertEqual(set(doc), {"format", "runs", "scanned", "by_model"})
        self.assertEqual(doc["runs"], 2)
        self.assertEqual(doc["by_model"], {"card:AA": 2, "card:BB": 1, "monster:MM": 1, "relic:RR": 1})
        self.assertTrue(all(isinstance(v, int) for v in doc["by_model"].values()))
        text = json.dumps(doc)
        for leak in ("SECRETSEED", "hashhash", "ignore previous", "ENCOUNTER"):
            self.assertNotIn(leak, text)

    def test_the_child_process_runs_isolated_and_writes_the_same_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "runs.jsonl.gz", Path(d) / "counts.json"
            with gzip.open(src, "wt", encoding="utf-8") as f:
                f.write("\n".join(self.LINES) + "\n")
            r = subprocess.run([sys.executable, "-I", str(mc.__file__), str(src), str(dst), "--character", "HERO",
                                "--ascension", "10"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(dst.read_text(encoding="utf-8"))
            self.assertEqual(out["by_model"], {"card:AA": 2, "card:BB": 1, "monster:MM": 1, "relic:RR": 1})
            self.assertNotIn("SECRETSEED", dst.read_text(encoding="utf-8"))
            self.assertNotIn("SECRETSEED", r.stdout)

    def test_manifest_takes_only_the_cached_counts(self):
        with tempfile.TemporaryDirectory() as d:
            doc = mc.aggregate(self.LINES, "HERO", 10)
            (Path(d) / mf.CODEX_COUNTS).write_text(json.dumps(doc), encoding="utf-8")
            got = mf.codex_counts(Path(d), False)
        self.assertEqual(set(got), {"runs", "by_model"})
        m = mf.assemble(FAM_DOC, CAT, [], [], [], got, [])
        self.assertNotIn("SECRETSEED", json.dumps(m))
        self.assertEqual(m["counts"]["codex_runs"], 2)


if __name__ == "__main__":
    unittest.main()
