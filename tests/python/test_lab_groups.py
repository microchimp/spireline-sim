"""tools/fidelity/lab_groups.py: symptoms merged into root mechanisms, host suspicion and backlog relations.

A fake lab (one finished aimed batch with shrink results, minimal packs, traces and the game's records) and
hand-made model features stand in for the referee's outputs and the decompile.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "referee"))

from tools.fidelity import lab_groups as lg  # noqa: E402
from tools.fidelity.lab_targets import Profile  # noqa: E402

FEATS = {"relic:EGG_A": ["hook:AfterCardChangedPiles", "cmd:CardCmd.Upgrade"],
         "relic:EGG_B": ["hook:AfterCardChangedPiles", "cmd:CardCmd.Upgrade"],
         "relic:LANTERN": ["hook:BeforeCombatStart", "cmd:PlayerCmd.GainEnergy"],
         **{f"relic:FILLER{i}": ["cmd:RelicCmd.Flash"] for i in range(40)},
         **{f"enemy:M{i}": ["cmd:DamageCmd.Attack"] for i in range(40)},
         "enemy:SLIME": ["cmd:DamageCmd.Attack"], "enemy:CRAB": ["cmd:DamageCmd.Attack"]}


def sym(seed, ingredients=None, phase="start", field=".hand", event=None, source="aimed", card="start", **kw):
    s = {"source": source, "ref": f"runs/x/b#{seed}", "seed": seed, "when": "2026-10-15T00:00:00Z", "step": 0, "field": field,
         "phase": phase, "card": card, "game": "g", "emu": "e", "first_event": event, "event_text": None, "hooks": [],
         "reproducible": True, "stale": False, "flags": [], "mirror_gap": False, "encounter": "SLIMES_WEAK",
         "shrink": "minimal" if ingredients is not None else None,
         "minimal_setup": [{"cmd": "relic", "id": i.split(":")[1]} for i in ingredients] if ingredients is not None else None,
         "ingredients": ingredients, "dropped": [], "implicated": None, "aim": None, "pool": "batch:b"}
    s.update(kw)
    if s["implicated"] is None:
        s["implicated"] = lg.implicated(card, s["encounter"], ingredients)
    return s


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.feats = lg.Features(FEATS, {"CardCmd.AutoPlay": {}}, {"SLIMES_WEAK": ["SLIME"], "CRABS_WEAK": ["CRAB"]})

    def groups(self, symptoms):
        return lg.group_symptoms(symptoms, self.feats)

    def test_ingredients_sharing_a_hook_merge(self):
        got = self.groups([sym("A1", ["relic:EGG_A"]), sym("A2", ["relic:EGG_B"]), sym("A3", ["relic:LANTERN"])])
        self.assertEqual(2, len(got))
        merged = next(g for g in got if len(g["members"]) == 2)
        self.assertIn("hook:AfterCardChangedPiles", merged["mechanism"])
        key = lg.group_key(merged)
        self.assertEqual([], key["ingredients"])  # no ingredient is common to both; the mechanism is
        self.assertEqual(sorted(["hook:AfterCardChangedPiles", "cmd:CardCmd.Upgrade"]), key["mechanism"])

    def test_common_features_do_not_merge(self):
        got = self.groups([sym("B1", ["relic:FILLER1"]), sym("B2", ["relic:FILLER2"])])
        self.assertEqual(2, len(got))  # RelicCmd.Flash is in nearly every relic

    def test_different_field_or_event_never_merge(self):
        got = self.groups([sym("C1", ["relic:EGG_A"]), sym("C2", ["relic:EGG_B"], field=".draw")])
        self.assertEqual(2, len(got))
        got = self.groups([sym("C3", ["relic:EGG_A"], event="x"), sym("C4", ["relic:EGG_B"], event="y")])
        self.assertEqual(2, len(got))

    def test_unshrunk_symptom_joins_the_group_holding_its_card(self):
        shrunk = sym("D1", ["card:HEADBUTT"], phase="play", card="HEADBUTT")
        loose = sym("D2", None, phase="play", card="HEADBUTT", source="corpus")
        other = sym("D3", None, phase="play", card="ANGER", source="corpus")
        got = self.groups([shrunk, loose, other])
        self.assertEqual(2, len(got))
        self.assertEqual({0, 1}, set(next(g for g in got if len(g["members"]) == 2)["members"]))

    def test_unshrunk_joins_only_with_a_rare_card_and_every_needed_ingredient(self):
        # a 3-item repro keeping a basic Strike once absorbed every unshrunk '.enemies after STRIKE_IRONCLAD' symptom
        shrunk = sym("E1", ["card:HEADBUTT", "card:STRIKE_IRONCLAD", "potion:BLOCK_POTION", "deck:set"], phase="play",
                     card="STRIKE_IRONCLAD", field=".enemies")
        strike = sym("E2", None, phase="play", card="STRIKE_IRONCLAD", source="corpus", field=".enemies")
        bash = sym("E3", None, phase="play", card="BASH", source="refset", field=".enemies")
        headbutt = sym("E4", None, phase="play", card="HEADBUTT", source="corpus", field=".enemies")  # potion unknown
        holds = sym("E5", None, phase="play", card="HEADBUTT", source="setup", field=".enemies",
                    held=["card:HEADBUTT", "card:ANGER", "potion:BLOCK_POTION", "encounter:SLIMES_WEAK"])
        lacks = sym("E6", None, phase="play", card="HEADBUTT", source="setup", field=".enemies",
                    held=["card:HEADBUTT", "encounter:SLIMES_WEAK"])
        got = self.groups([shrunk, strike, bash, headbutt, holds, lacks])
        self.assertEqual([[0, 4], [1], [2], [3, 5]], sorted(sorted(g["members"]) for g in got))
        key = lg.group_key(next(g for g in got if 1 in g["members"]))
        self.assertEqual(([], ["card:STRIKE_IRONCLAD"]), (key["ingredients"], key["implicated"]))  # none it never had

    def test_traced_symptoms_group_by_their_event(self):
        ev = "play: game nothing; emulator player BLOCK_NEXT_TURN"
        f = ".player.powers"
        a = sym("T1", None, phase="play", field=f, card="BLOODLETTING", source="aimed", first_event=ev)
        b = sym("T2", None, phase="play", field=f, card="BREAKTHROUGH", source="refset", first_event=ev)
        c = sym("T3", None, phase="play", field=f, card="BREAKTHROUGH", source="refset", first_event="play: game nothing; emulator draw")
        got = self.groups([a, b, c])
        self.assertEqual(2, len(got))  # the generic event names no model: it stays with its card
        key = lg.group_key(next(g for g in got if len(g["members"]) == 2))
        self.assertEqual(ev, key["event"])
        self.assertNotIn("implicated", key)
        shrunk = sym("T4", ["relic:LANTERN"], phase="play", field=".hand", first_event=ev)
        self.assertEqual(3, len(self.groups([a, b, c, shrunk])))  # another field: apart
        shrunk = sym("T5", ["relic:LANTERN"], phase="play", field=f, first_event=ev)
        self.assertEqual(2, len(self.groups([a, b, c, shrunk])))  # same event: the traced ones join the shrunk one

    def test_fight_alone_keeps_the_encounter_apart(self):
        a = sym("E1", ["encounter:SLIMES_WEAK"])
        b = sym("E2", ["encounter:CRABS_WEAK"], encounter="CRABS_WEAK")
        self.assertEqual(2, len(self.groups([a, b])))  # DamageCmd.Attack is every monster's
        self.assertEqual(["encounter:SLIMES_WEAK"], lg.implicated("start", "SLIMES_WEAK", ["encounter:SLIMES_WEAK"]))

    def test_stable_ids(self):
        a = lg.group_key(self.groups([sym("F1", ["relic:LANTERN"])])[0])
        b = lg.group_key(self.groups([sym("F9", ["relic:LANTERN"])])[0])
        self.assertEqual(lg.stable_id(a), lg.stable_id(b))
        self.assertTrue(lg.stable_id(a).startswith("G"))


class HostAndBacklogTests(unittest.TestCase):
    HOSTS = {"entries": [{"id": "boulder", "status": "open", "finding": {"date": "2026-10-13", "quote": "boulder stalls"},
                          "match": {"entities": ["card:ROLLING_BOULDER"], "phases": ["end_turn", "start"]}, "why": "stale"}]}

    def test_host_suspect_rules(self):
        h = lg.HostGaps(self.HOSTS, "the boulder stalls here")
        self.assertEqual([], h.stale_entries)
        sus, why = h.judge([sym("H1", ["card:ROLLING_BOULDER"])], {"card:ROLLING_BOULDER"})
        self.assertTrue(sus)
        self.assertIn("boulder", why)
        self.assertFalse(h.judge([sym("H2", ["card:ROLLING_BOULDER"], phase="play")], {"card:ROLLING_BOULDER"})[0])
        self.assertTrue(h.judge([sym("H3", ["relic:EGG_A"], flags=["cross_act"])], set())[0])
        self.assertFalse(h.judge([sym("H4", ["relic:EGG_A"], flags=["debug_room"])], set())[0])
        self.assertTrue(h.judge([sym("H5", ["relic:EGG_A"], stale=True)], set())[0])
        self.assertEqual(["boulder"], lg.HostGaps(self.HOSTS, "nothing").stale_entries)

    def test_stale_game_side(self):
        st = {"hand": [["A", False]], "draw": [], "discard": [], "player": {"hp": 5}}
        game = {"steps": [{"state": st}, {"action": {"kind": "play"}, "state": st}, {"action": {"kind": "play"}, "state": st},
                          {"action": {"kind": "play"}, "state": st}, {"action": {"kind": "play"}, "state": st}]}
        self.assertTrue(lg.stale_after(game, 1))
        game["steps"][3]["state"] = {**st, "player": {"hp": 4}}
        self.assertFalse(lg.stale_after(game, 1))

    def test_relations(self):
        items = [{"id": "I1", "lane": "fight", "entities": {"relic:EGG_A"}, "seeds": set(), "fields": set(), "text": ""},
                 {"id": "I2", "lane": "fight", "entities": {"relic:LANTERN"}, "seeds": set(), "fields": set(), "text": ""},
                 {"id": "I3", "lane": "fight", "entities": {"card:BASH", "card:ANGER"}, "seeds": set(), "fields": set(), "text": ""},
                 {"id": "I4", "lane": "fight", "entities": set(), "seeds": {"SEEDSEED01"}, "fields": set(), "text": ""}]
        groups = [{"key": {"family": "fight"}, "_entities": {"relic:EGG_A"}, "_seeds": set(), "_fields": set()},
                  {"key": {"family": "fight"}, "_entities": {"relic:LANTERN", "relic:EGG_A"}, "_seeds": set(), "_fields": set()},
                  {"key": {"family": "fight"}, "_entities": {"card:BASH"}, "_seeds": set(), "_fields": set()},
                  {"key": {"family": "fight"}, "_entities": {"card:ANGER"}, "_seeds": set(), "_fields": set()},
                  {"key": {"family": "fight"}, "_entities": {"potion:X"}, "_seeds": {"SEEDSEED01"}, "_fields": set()},
                  {"key": {"family": "fight"}, "_entities": {"relic:NONE"}, "_seeds": set(), "_fields": set()}]
        lg.match_backlog(groups, items)
        rel = [(g["relation"], g["backlog_ids"]) for g in groups]
        self.assertEqual("splits", rel[0][0])               # I1 also matches group 1
        self.assertEqual("merges", rel[1][0])               # group 1 matches I1 and I2
        self.assertEqual(("splits", ["I3"]), rel[2])        # I3 is matched by two groups
        self.assertEqual(("confirms", ["I4"]), rel[4])      # the item cites the group's seed
        self.assertEqual(("new", []), rel[5])

    def test_seed_citations_count_for_the_fight_the_seed_names(self):
        # targets mode borrowed SEEDSEED01 from cov60's recorded fight; aimed60s' fight with it is another fight
        def match(text):
            groups = [{"key": {"family": "fight"}, "_entities": {"potion:X"}, "_fields": set(),
                       "_seedrefs": {("SEEDSEED01", ("aimed60s", "20261015T0100-fights"), False)}},
                      {"key": {"family": "fight"}, "_entities": {"potion:Y"}, "_fields": set(),
                       "_seedrefs": {("SEEDSEED01", ("cov60",), True)}}]
            items = [{"id": "I1", "lane": "fight", "entities": set(), "seeds": {"SEEDSEED01"}, "fields": set(), "text": text}]
            lg.match_backlog(groups, items)
            return [g["backlog_ids"] for g in groups]
        self.assertEqual([[], ["I1"]], match("block lost after seedseed01"))         # no batch named: the seed's own fight
        self.assertEqual([[], ["I1"]], match("cov60 seedseed01 block lost"))
        self.assertEqual([["I1"], []], match("lab aimed60s seedseed01 block lost"))  # the borrowed copy, named
        self.assertEqual([["I1"], []], match("runs/20261015t0100-fights seedseed01"))

    def test_impact_from_the_profile(self):
        prof = Profile({"runs": 100, "meet": {"relic": {"EGG_A": 40, "LANTERN": 10}, "card": {"AGGRESSION": 30}},
                        "acts": {"1": {"relics": {"LANTERN": 10}, "pairs": {"relic:EGG_A|relic:LANTERN": 5}}}})
        self.assertEqual(40, lg.impact(prof, ["relic:EGG_A"]))
        self.assertEqual(5, lg.impact(prof, ["relic:EGG_A", "relic:LANTERN"]))  # 10 runs hold Lantern, half with the egg
        self.assertEqual(30, lg.impact(prof, ["power:AGGRESSION_POWER"]))       # through its power card


def write(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


class EndToEndTests(unittest.TestCase):
    def test_lab_run(self):
        with tempfile.TemporaryDirectory() as td:
            lab = Path(td) / "lab"
            job = lab / "runs" / "20261015T0100-fights"
            specs = [{"seed": f"SEED{i:06d}", "setup": [{"cmd": "relic", "id": r}, {"cmd": "card", "id": "ANGER"},
                                                        {"cmd": "fight", "id": "SLIMES_WEAK"}]}
                     for i, r in enumerate(["EGG_A", "EGG_B", "BRASS_LANTERN", "EGG_A"])]
            write(job / "b.json", {"runs": specs})
            write(job / "b.aims.json", {"aims": [{"seed": s["seed"], "target": "relic:EGG_A"} for s in specs]})
            verdicts = [{"seed": s["seed"], "identical": False, "step": 0, "field": ".hand", "card": "start", "flags": [],
                         "diff": [[".hand", "[A]", "[B]"]]} for s in specs[:3]] + [{"seed": specs[3]["seed"], "identical": True}]
            write(job / "b.verdicts.json", verdicts)
            write(job / "b.out.json", {"results": [{"seed": s["seed"], "steps": [{"state": {}}]} for s in specs]})
            cases = []
            for s in specs[:3]:
                cases.append({"seed": s["seed"], "status": "minimal", "of": 2, "fights": 6, "step": 0, "field": ".hand",
                              "items": [s["setup"][0]]})
                write(job / "b.packs" / "minimal" / f"{s['seed']}.json", {"runs": [{"seed": s["seed"], "setup": [s["setup"][0], s["setup"][2]]}]})
            write(job / "b.shrink.json", {"cases": cases})
            write(job / "b.packs" / "traces" / f"{specs[0]['seed']}.trace.json",
                  {"segment": {"from": "turn_start", "game": ["hand +A", "draw -1"], "emu": ["hand +B", "draw -1"],
                               "game_hooks": ["hook AfterCardChangedPiles x2"]}, "reproducible": True})
            (job / "done.rc").write_text("0", encoding="utf-8")
            write(lab / "cache" / "model_features.json", {"stamp": None, "features": {}})
            decomp = Path(td) / "no-decompile"
            hosts = Path(td) / "hosts.json"
            write(hosts, {"entries": []})
            backlog = Path(td) / "backlog-x.json"
            write(backlog, {"items": [{"id": "X-F01", "status": "open", "title": "Lantern energy", "evidence": "BRASS_LANTERN at start"}]})
            out, summ = lab / "groups.json", lab / "summary.md"
            rc = lg.main(["--lab", str(lab), "--out", str(out), "--summary", str(summ), "--decompiled", str(decomp),
                          "--host-gaps", str(hosts), "--backlog", str(backlog)])
            self.assertEqual(0, rc)
            doc = json.loads(out.read_text(encoding="utf-8"))
            text = out.read_text(encoding="utf-8") + summ.read_text(encoding="utf-8")
            self.assertNotIn(td.replace("\\", "/"), text.replace("\\", "/"))  # paths relative to the lab
            self.assertNotIn(td, text)
            self.assertEqual(3, sum(g["count"] for g in doc["groups"]))
            lantern = next(g for g in doc["groups"] if "relic:BRASS_LANTERN" in g["entities"])
            self.assertEqual(["X-F01"], lantern["backlog_ids"])
            self.assertEqual("confirms", lantern["relation"])
            self.assertEqual([{"item": "relic:BRASS_LANTERN", "removal_vanishes": True}, {"item": "card:ANGER", "removal_vanishes": False}],
                             lantern["ingredient_effect"])
            ex = lantern["examples"][0]
            self.assertEqual(("runs/20261015T0100-fights/b#SEED000002", "[A]", "[B]"), (ex["ref"], ex["game"], ex["emu"]))
            self.assertEqual("2026-10-15T01:00:00Z", lantern["first_seen_utc"])
            egg = next(g for g in doc["groups"] if "relic:EGG_A" in g["entities"])
            self.assertTrue(egg["examples"][0]["first_event"] or egg["examples"][-1]["first_event"])
            self.assertLessEqual(len(summ.read_text(encoding="utf-8")), 4000)
            # a second run keeps the ids and the first-seen times
            self.assertEqual(0, lg.main(["--lab", str(lab), "--out", str(out), "--summary", str(summ), "--decompiled", str(decomp),
                                         "--host-gaps", str(hosts), "--backlog", str(backlog)]))
            again = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual({g["id"]: g["first_seen_utc"] for g in doc["groups"]}, {g["id"]: g["first_seen_utc"] for g in again["groups"]})

    def test_a_seed_on_two_specs_of_a_batch(self):
        # an older targets batch drew cached seeds with replacement: spec 0 diverges, spec 2 (same seed) is identical
        with tempfile.TemporaryDirectory() as td:
            lab = Path(td) / "lab"
            job = lab / "runs" / "20261015T0100-fights"
            specs = [{"seed": s, "setup": [{"cmd": "card", "id": c}, {"cmd": "fight", "id": "SLIMES_WEAK"}]}
                     for s, c in (("SHAREDSD01", "HEADBUTT"), ("FRESHSEED1", "ANGER"), ("SHAREDSD01", "STAMPEDE"))]
            write(job / "b.json", {"runs": specs})
            write(job / "b.aims.json", {"aims": [{"seed": "SHAREDSD01", "target": "card:HEADBUTT", "seed_from": "cache"},
                                                 {"seed": "FRESHSEED1", "target": "card:ANGER", "seed_from": "fresh"},
                                                 {"seed": "SHAREDSD01", "target": "card:STAMPEDE", "seed_from": "cache"}]})
            write(job / "b.verdicts.json", [{"seed": "SHAREDSD01", "identical": False, "step": 1, "field": ".hand", "card": "HEADBUTT",
                                             "diff": [[".hand", "[A]", "[B]"]]},
                                            {"seed": "FRESHSEED1", "identical": False, "step": 0, "field": ".hand", "card": "start",
                                             "diff": [[".hand", "[A]", "[C]"]]},
                                            {"seed": "SHAREDSD01", "identical": True}])
            moving = {"seed": "SHAREDSD01", "steps": [{"state": {"hand": [i]}, "action": {"kind": "play"}} for i in range(6)]}
            still = {"seed": "SHAREDSD01", "steps": [{"state": {"hand": [0]}, "action": {"kind": "play"}} for _ in range(6)]}
            write(job / "b.out.json", {"results": [moving, {"seed": "FRESHSEED1", "steps": []}, still]})
            write(job / "b.shrink.json", {"cases": [{"seed": "SHAREDSD01", "status": "minimal", "items": [specs[0]["setup"][0]]}]})
            write(job / "b.packs" / "minimal" / "SHAREDSD01.json", {"runs": [specs[0]]})
            write(job / "b.packs" / "traces" / "SHAREDSD01.trace.json",
                  {"segment": {"from": "x", "game": ["hand +A"], "emu": ["hand +B"]}, "reproducible": True})
            got = {s["seed"]: s for s in lg.batch_symptoms(job / "b", lab, "w")}
            shared, fresh = got["SHAREDSD01"], got["FRESHSEED1"]
            self.assertEqual("card:HEADBUTT", shared["aim"])           # by index, not the last spec with the seed
            self.assertFalse(shared["stale"])                          # spec 0's record, not spec 2's
            self.assertEqual(["card:HEADBUTT"], shared["ingredients"])  # only spec 0 diverged: the shrink case is its
            self.assertEqual("x: game hand +A; emulator hand +B", shared["first_event"])  # past the start: its own
            verdicts = json.loads((job / "b.verdicts.json").read_text(encoding="utf-8"))
            write(job / "b.verdicts.json", [dict(verdicts[0], step=0)] + verdicts[1:])
            at_start = next(s for s in lg.batch_symptoms(job / "b", lab, "w") if s["seed"] == "SHAREDSD01")
            self.assertIsNone(at_start["first_event"])  # traced off spec 2's reanchor record (trace_divergent's by seed)
            write(job / "b.verdicts.json", verdicts)
            verdicts = json.loads((job / "b.verdicts.json").read_text(encoding="utf-8"))
            verdicts[2] = dict(verdicts[0], card="STAMPEDE")
            write(job / "b.verdicts.json", verdicts)
            both = [s for s in lg.batch_symptoms(job / "b", lab, "w") if s["seed"] == "SHAREDSD01"]
            self.assertEqual([None, None], [s["ingredients"] for s in both])  # two diverge: the case is either one's
            self.assertEqual(["card:HEADBUTT", "card:STAMPEDE"], [s["aim"] for s in both])
            self.assertEqual(["card:HEADBUTT", "encounter:SLIMES_WEAK"], shared["held"])
            self.assertEqual((False, True), (shared["seed_own"], fresh["seed_own"]))
            self.assertEqual(["b", "20261015T0100-fights"], shared["batch"])

    def test_source_kinds(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write(d / "c.json", {"format": 2, "label": "x", "totals": {"fights": 1}, "fights": {}})
            write(d / "r.json", {"set": "Ironclad A10 reference set v2", "headline": {}, "verdicts": []})
            write(d / "i.json", {"game": {"label": "game recordings", "kinds": {}}, "emulator": {}, "live": {}})
            write(d / "m.json", {"killed": 3, "kill_rate": 0.5, "survivors": []})
            write(d / "o.json", {"something": 1})
            self.assertEqual(["corpus", "refset", "invariants", "mutation", None],
                             [lg.source_kind(d / n) for n in ("c.json", "r.json", "i.json", "m.json", "o.json")])

    def test_job_json_names_outputs_kept_elsewhere(self):
        with tempfile.TemporaryDirectory() as td:
            lab, elsewhere = Path(td) / "lab", Path(td) / "corpus-run" / "results.json"
            write(elsewhere, {"format": 2, "totals": {"fights": 2}, "fights": {"f1": {"fd": 3, "field": ".hand", "card": "BASH"}}})
            job = lab / "runs" / "20261015T0200-presort"
            write(job / "job.json", {"outputs": {"corpus": str(elsewhere), "codex_walk": "missing-dir"}})
            (job / "done.rc").write_text("0", encoding="utf-8")
            unfinished = lab / "runs" / "20261015T0300-presort"
            write(unfinished / "job.json", {"outputs": {"corpus": str(elsewhere)}})
            got = lg.lab_sources(lab)
            self.assertEqual([("corpus", elsewhere, "2026-10-15T02:00:00Z")], got)
            syms = lg.corpus_symptoms(elsewhere, lab, "x")
            self.assertEqual([("play", ".hand", ["card:BASH"])], [(s["phase"], s["field"], s["implicated"]) for s in syms])

    def test_an_unfinished_job_is_read_only_when_named(self):
        with tempfile.TemporaryDirectory() as td:
            lab = Path(td) / "lab"
            job = lab / "runs" / "20261015T0200-fights"
            write(job / "corpus" / "results.json", {"format": 2, "totals": {"fights": 1}, "fights": {}})
            self.assertEqual([], lg.lab_sources(lab))
            got = lg.lab_sources(lab, (job,))
            self.assertEqual([("corpus", job / "corpus" / "results.json")], [(k, p) for k, p, _ in got])
            (job / "done.rc").write_text("0", encoding="utf-8")
            self.assertEqual(1, len(lg.lab_sources(lab, (job,))))  # finished and named: read once

    def test_a_walk_named_by_its_shard_files_is_read_as_its_folder_once(self):
        with tempfile.TemporaryDirectory() as td:
            job = Path(td) / "lab" / "runs" / "20261015T0200-presort"
            for i in range(2):
                (job / "codex").mkdir(parents=True, exist_ok=True)
                (job / "codex" / f"s{i}.jsonl").write_text("{}\n", encoding="utf-8")
            write(job / "job.json", {"outputs": {"codex_walk": ["codex/s0.jsonl", "codex/s1.jsonl"]}})
            got = lg.job_outputs(job, "w", set())
            self.assertEqual([("codex", job / "codex", "w")], got)
            self.assertEqual([], lg.job_outputs(job, "w", {str(job / "codex")}))

    def test_codex_walk_groups_by_component_field_class(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "walk"
            d.mkdir()
            stop = lambda c, e, m: {"status": "mismatch", "stop": {"component": c, "field": c, "class": "EMULATOR",  # noqa: E731
                                                                   "expected": e, "emulator": m, "floor": 3}}
            (d / "s0.jsonl").write_text("\n".join(json.dumps(x) for x in [
                stop("encounter", "NibbitsWeak", "SlimesWeak"), stop("encounter", "NibbitsWeak", "ShrinkerWeak"),
                stop("deck", "A, B", "A"), {"status": "complete"}]) + "\n", encoding="utf-8")
            syms = lg.walk_symptoms(d, Path(td), "2026-10-15T00:00:00Z")
            self.assertEqual(3, len(syms))
            got = lg.group_symptoms(syms, lg.Features({}))
            self.assertEqual([2, 1], sorted((len(g["members"]) for g in got), reverse=True))
            self.assertEqual(["SHRINKER_WEAK", "NIBBITS_WEAK"][1], syms[0]["walk_tokens"][0])



class StrategyInputsTests(unittest.TestCase):
    """What the strategy mix feeds the groups: card-screen gaps of the setup mirror, batch.py batches, whole runs, per-job counts."""

    def lab(self, td):
        lab = Path(td) / "lab"
        write(lab / "cache" / "model_features.json", {"stamp": None, "features": {}})
        hosts = Path(td) / "hosts.json"
        write(hosts, {"entries": []})
        return lab, hosts

    def groups(self, lab, td, hosts):
        out, summ = lab / "groups.json", lab / "summary.md"
        self.assertEqual(0, lg.main(["--lab", str(lab), "--out", str(out), "--summary", str(summ), "--decompiled", str(Path(td) / "none"),
                                     "--host-gaps", str(hosts), "--backlog", str(Path(td) / "no-backlog.json")]))
        return json.loads(out.read_text(encoding="utf-8"))

    def test_a_card_screen_the_mirror_could_not_answer_is_a_symptom_of_a_matching_fight(self):
        with tempfile.TemporaryDirectory() as td:
            lab, hosts = self.lab(td)
            job = lab / "runs" / "20261015T0100-fights"
            specs = [{"seed": f"CHOICE{i:04d}", "choice_edges": "none",
                      "setup": [{"cmd": "deck_set", "cards": []}, {"cmd": "relic", "id": "KIFUDA"}, {"cmd": "fight", "id": "SLIMES_WEAK"}]}
                     for i in range(3)]
            write(job / "fights" / "c.json", {"runs": specs})
            msg = "unsupported: the emulator's selection wants more cards than the game's screen required"
            verdicts = [{"seed": specs[0]["seed"], "identical": True, "mirror_gap": [1], "mirror_gap_why": {"1": msg}},
                        {"seed": specs[1]["seed"], "identical": True, "mirror_gap": [1], "mirror_gap_why": {"1": "unsupported: relic in a fight"}},
                        {"seed": specs[2]["seed"], "identical": True, "mirror_gap": []}]
            write(job / "fights" / "c.verdicts.json", verdicts)
            got = lg.batch_symptoms(job / "fights" / "c", lab, "w")
            self.assertEqual(["CHOICE0000"], [s["seed"] for s in got])  # other gaps (a relic in a fight) are limits of the mirror
            s = got[0]
            self.assertEqual(("setup", "setup.selection", ["relic:KIFUDA"], True), (s["phase"], s["field"], s["implicated"], s["mirror_gap"]))
            self.assertEqual(msg, s["first_event"])
            (job / "done.rc").write_text("0", encoding="utf-8")
            doc = self.groups(lab, td, hosts)
            self.assertEqual(1, len(doc["groups"]))
            g = doc["groups"][0]
            self.assertIn("relic:KIFUDA", g["entities"])
            self.assertEqual({"20261015T0100-fights": {"n": 1, "ready": 0}}, g["by_job"])  # a mirror gap is never fix-ready

    def test_a_diverging_fight_with_a_gap_is_one_symptom_not_two(self):
        with tempfile.TemporaryDirectory() as td:
            lab, _ = self.lab(td)
            job = lab / "runs" / "20261015T0100-fights"
            write(job / "fights" / "c.json", {"runs": [{"seed": "CHOICE0000", "setup": [{"cmd": "relic", "id": "KIFUDA"}, {"cmd": "fight", "id": "X"}]}]})
            write(job / "fights" / "c.verdicts.json", [{"seed": "CHOICE0000", "identical": False, "step": 2, "field": ".hand", "card": "BASH",
                                                        "diff": [[".hand", "a", "b"]], "mirror_gap": [0], "mirror_gap_why": {"0": "unsupported: selection"}}])
            got = lg.batch_symptoms(job / "fights" / "c", lab, "w")
            self.assertEqual([".hand"], [s["field"] for s in got])

    def test_a_batch_py_batch_without_verdicts_is_judged_from_its_records(self):
        import unittest.mock as mock
        import scenarios  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as td:
            lab, hosts = self.lab(td)
            job = lab / "runs" / "20261015T0200-fights"
            specs = [{"seed": f"RANDOM{i:04d}", "deck_add": ["ANGER"]} for i in range(2)]
            write(job / "fights" / "r.json", {"runs": specs})
            write(job / "fights" / "r.out.json", {"results": [{"seed": s["seed"], "steps": [{}]} for s in specs]})
            write(job / "fights" / "r.emu.json", {"results": [{}, {}]})
            write(job / "fights" / "r.reanchor.json", {"results": [{}, {}]})
            write(job / "fights" / "r.packs" / "traces" / "traced.reanchor.json", {"results": []})  # not a batch of its own
            (job / "done.rc").write_text("0", encoding="utf-8")
            verdicts = [{"seed": "RANDOM0000", "identical": True, "flags": []},
                        {"seed": "RANDOM0001", "identical": False, "step": 4, "field": ".draw", "card": "ANGER", "diff": [[".draw", "x", "y"]], "flags": []}]
            self.assertEqual([("batch", job / "fights" / "r", "2026-10-15T02:00:00Z")], lg.lab_sources(lab))
            with mock.patch.object(scenarios, "judge", return_value=verdicts) as judge:
                doc = self.groups(lab, td, hosts)
            self.assertEqual(1, judge.call_count)
            self.assertEqual(1, sum(g["count"] for g in doc["groups"]))
            self.assertEqual(2, lg.pool_size("batch", job / "fights" / "r", 1))  # the denominator is the batch's fights
            write(job / "fights" / "r.verdicts.json", verdicts)  # once verdicts exist they are what is read, once
            self.assertEqual([("batch", job / "fights" / "r", "2026-10-15T02:00:00Z")], lg.lab_sources(lab))
            self.assertIsNone(lg.derived_verdicts(job / "fights" / "missing"))
            write(job / "fights" / "r.emu.json", {"results": [{}]})  # sizes that disagree: not judged
            self.assertIsNone(lg.derived_verdicts(job / "fights" / "r"))

    def test_whole_runs_are_symptoms_at_their_first_divergence(self):
        with tempfile.TemporaryDirectory() as td:
            lab, hosts = self.lab(td)
            job = lab / "runs" / "20261015T0300-runs"
            runs = [{"seed": "RUNSEED001", "recorded": True, "decisions": 90, "fights": 12, "end": "death", "floor": 17,
                     "first": {"decision": 31, "component": "reward", "floor": 6, "choice": "[3]",
                               "diffs": [["offer.cards", "[\"ANGER\", \"CLASH\"]", "[\"ANGER\", \"HAVOC\"]"]]}, "outside": None},
                    {"seed": "RUNSEED002", "recorded": True, "decisions": 80, "fights": 11, "first": {"decision": 40, "component": "reward", "floor": 9,
                     "choice": "[1]", "diffs": [["offer.cards", "[\"BASH\", \"CLASH\"]", "[\"BASH\", \"HAVOC\"]"]]}},
                    {"seed": "RUNSEED003", "recorded": True, "decisions": 70, "fights": 9, "first": None, "outside": None},
                    {"seed": "RUNSEED004", "recorded": False, "error": "no result (session ended)"}]
            write(job / "runs" / "r.runs.json", {"runs": runs})
            write(job / "runs" / "r.out.json", {"results": []})
            (job / "done.rc").write_text("0", encoding="utf-8")
            self.assertEqual([("run", job / "runs" / "r.runs.json", "2026-10-15T03:00:00Z")], lg.lab_sources(lab))
            got = lg.run_symptoms(job / "runs" / "r.runs.json", lab, "w")
            self.assertEqual(["RUNSEED001", "RUNSEED002"], [s["seed"] for s in got])  # clean and unrecorded runs are not symptoms
            self.assertEqual(("run", "reward.offer.cards", 31), (got[0]["phase"], got[0]["field"], got[0]["step"]))
            self.assertIn("HAVOC", [t.upper() for t in got[0]["walk_tokens"]])
            doc = self.groups(lab, td, hosts)
            self.assertEqual([2], [g["count"] for g in doc["groups"]])  # the same place, one group
            self.assertEqual(2, doc["groups"][0]["impact_runs_est"])
            self.assertEqual({"20261015T0300-runs": {"n": 2, "ready": 2}}, doc["groups"][0]["by_job"])
            self.assertEqual(4, lg.pool_size("run", job / "runs" / "r.runs.json", 0))  # the runs played, clean ones included

    def test_by_job_counts_the_symptoms_a_fixer_can_start_from(self):
        mk = lambda ref, ev, gap: {"ref": ref, "first_event": ev, "mirror_gap": gap}  # noqa: E731
        got = lg.by_job([mk("runs/J1/fights/b#S1", "hand +A", False), mk("runs/J1/fights/b#S2", None, False),
                         mk("runs/J1/fights/b#S3", "hand +A", True), mk("runs/J2/fights/b#S1", "x", False), mk("corpus:F1", "x", False)])
        self.assertEqual({"J1": {"n": 3, "ready": 1}, "J2": {"n": 1, "ready": 1}}, got)


if __name__ == "__main__":
    unittest.main()
