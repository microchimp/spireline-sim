"""tools.fidelity.packet: one diagnosis packet per backlog item, built from recordings, with no game process.

Everything here is synthetic: a corpus of tiny fights written in the corpus's own format, a decompile tree in the
decompile's layout (no game code), a probe report, a ledger, a traced game record and a codex walk. The emulator is a
fake EmuDump that answers `fight` and `reanchor` with states in the shapes the real one writes, so the tests need
no .NET build and start no process but git.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import packet as pk  # noqa: E402
from tools.fidelity.corpus import Corpus  # noqa: E402

ID_MAP = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
CID, EID = ID_MAP["cards"], ID_MAP["enemies"]
SEED_CITED, SEED_NAMED, SEED_OTHER = "ABCDE12345", "NAMED00001", "OTHER00001"

DECOMPILE = {
    "MegaCrit.Sts2.Core.Commands/CreatureCmd.cs": """namespace MegaCrit.Sts2.Core.Commands;
public static class CreatureCmd
{
    public static async Task Damage(int amount)
    {
        await Task.CompletedTask;
    }
}
""",
    "MegaCrit.Sts2.Core.Hooks/Hook.cs": """namespace MegaCrit.Sts2.Core.Hooks;
public static class Hook
{
    public static decimal ModifyHpLost(decimal amount)
    {
        return amount;
    }
}
""",
    "MegaCrit.Sts2.Core.Models/AbstractModel.cs": """namespace MegaCrit.Sts2.Core.Models;
public abstract class AbstractModel
{
    public virtual decimal ModifyHpLostAfterOsty(decimal amount)
    {
        return amount;
    }
}
""",
    "MegaCrit.Sts2.Core.Models.Powers/HardenedShellPower.cs": """namespace MegaCrit.Sts2.Core.Models.Powers;
public sealed class HardenedShellPower : AbstractModel
{
    public override decimal ModifyHpLostAfterOsty(decimal amount)
    {
        return amount;
    }

    public override async Task BeforeSideTurnStart()
    {
        await Task.CompletedTask;
    }
}
""",
    "MegaCrit.Sts2.Core.Models.Monsters/SkulkingColony.cs": """namespace MegaCrit.Sts2.Core.Models.Monsters;
public sealed class SkulkingColony
{
}
""",
}

ENGINE = {
    "Core/Effects/CardEffects.cs": """namespace Sts2Emulator;
public static class CardEffects
{
    public static void LandDamageOnEnemy(int amount)
    {
        // HardenedShell is lowered by the HP lost here
    }
}
""",
}


def game_state(hp, shell, hand=1):
    return {"in_combat": True, "round": 1, "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 2, "powers": []},
            "hand": [["STRIKE_IRONCLAD", False]] * hand, "draw": [], "discard": [], "exhaust": [],
            "enemies": [{"name": "SKULKING_COLONY", "hp": hp, "max_hp": 80, "block": 0, "powers": [["HARDENED_SHELL_POWER", shell]]}],
            "rng": {"Shuffle": {"seed": 1, "counter": 1}, "CombatTargets": {"seed": 2, "counter": 0}}}


def emu_state(hp, shell, hand=1):
    card = {"def_id": CID["StrikeIronclad"], "upgraded": False}
    return {"in_combat": True, "draw_order": [], "rng": {"Shuffle": 1, "CombatTargets": 0},
            "snapshot": {"player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 2, "buffs": []},
                         "piles": {"hand": [card] * hand, "discard": [], "exhaust": []},
                         "enemies": [{"def_id": EID["SkulkingColony"], "hp": hp, "block": 0,
                                      "buffs": [{"name": "HardenedShell", "amount": shell}]}]}}


PLAY = {"kind": "play", "hand_index": 0, "card": "STRIKE_IRONCLAD", "target": 0, "target_slot": 0}


def fight(seed, n_steps=3, hp_after=70, shell_after=20, setup=None):
    """A game recording: the colony keeps its Hardened Shell at 20 after the first Strike (the game's rule)."""
    steps = [{"action": None, "state": game_state(80, 20, hand=n_steps)}]
    for k in range(1, n_steps):
        steps.append({"action": PLAY, "state": game_state(hp_after if k == 1 else hp_after - 6 * (k - 1), shell_after, hand=n_steps - k)})
    rec = {"seed": seed, "character": "IRONCLAD", "ascension": 10, "neow_pick": 0, "room": [0, 1], "deck_add": [], "steps": steps}
    if setup:
        rec["setup"] = setup
    return rec


def emu_doc_for(rec, shell_emulator=14, extra_draw=False, trace=False):
    """What EmuDump writes for the recording when the emulator lowers the shell after the first Strike."""
    steps = []
    for k, st in enumerate(rec["steps"]):
        g = st["state"]
        hp = g["enemies"][0]["hp"]
        shell = g["enemies"][0]["powers"][0][1] if k < 1 else shell_emulator
        s = {"action": st["action"], "state": emu_state(hp, shell, hand=len(g["hand"])), "rng_log": [], "choices": [], "moves": "same"}
        if extra_draw and k == 1:
            s["rng_log"] = [["Shuffle", 2, "CardEffects.X"]]
            s["state"]["rng"] = {"Shuffle": 2, "CombatTargets": 0}
        if trace and k >= 1:
            s["trace"] = [["M", "start", None, mini(hp, shell)], ["M", "play", "STRIKE_IRONCLAD", mini(hp, shell)],
                          ["M", "played", "STRIKE_IRONCLAD", mini(hp, shell)]]
        steps.append(s)
    return {"seed": rec["seed"], "steps": steps}


def mini(hp, shell):
    return {"player": [70, 0, 2], "powers": [], "enemies": [["SKULKING_COLONY", hp, 0, [["HardenedShell", shell]]]], "hand": [], "n": [0, 0, 0]}


class FakeEmu:
    """Answers EmuDump's modes from the recordings it is given; counts its calls."""

    available = True
    dll = Path("FakeEmuDump.dll")

    def __init__(self, shell_emulator=14, extra_draw=False):
        self.shell_emulator, self.extra_draw, self.calls = shell_emulator, extra_draw, []

    def run(self, mode, fights, *extra):
        self.calls.append((mode, extra))
        trace = "--trace-to" in extra
        results = [emu_doc_for(f, self.shell_emulator, self.extra_draw, trace) for f in fights]
        return {"results": results}


def filler(i):
    return {"seed": f"FILL{i:06d}", "character": "IRONCLAD", "ascension": 10, "neow_pick": 0, "room": [0, 1], "deck_add": [],
            "steps": [{"action": None, "state": dict(game_state(30, 0), enemies=[{"name": "CORPSE_SLUG", "hp": 30, "max_hp": 30, "block": 0, "powers": []}])}]}


class World:
    """A referee folder in a temp dir, with whatever evidence a test wants."""

    def __init__(self, tmp: Path, cited_diverges=True, n_filler=400):
        self.root = tmp / "referee"
        self.corpus = Corpus(self.root / "corpus")
        self.corpus.fights_dir.mkdir(parents=True)
        self.fights = {}  # name -> (id, record)
        metas, results = [], {}

        def add(name, rec, res, needs=()):
            fid = f"{len(metas):016x}"
            self.corpus.write(fid, rec)
            metas.append({"id": fid, "seed": rec["seed"], "steps": len(rec["steps"]), "needs": list(needs), "flags": [], "tags": [],
                          "sources": [f"jobs/{name}.out.json"]})
            results[fid] = res
            self.fights[name] = (fid, rec)

        div = {"n": 3, "fd": 1, "field": ".enemies", "path": ".enemies", "card": "STRIKE_IRONCLAD", "rv": "DM", "dp": {"1": [".enemies"]}}
        same = {"n": 3, "fd": None, "rv": "MM"}
        add("cited", fight(SEED_CITED, 3), div if cited_diverges else same)
        add("cited_long", fight(SEED_CITED, 6), dict(div, n=6, rv="DMMMM"))
        add("named", fight(SEED_NAMED, 2), dict(div, n=2, rv="D"))
        add("other", fight(SEED_OTHER, 2, setup=[{"cmd": "relic", "id": "VAJRA"}, {"cmd": "fight", "id": "SKULKING_COLONY_ELITE"}]),
            {"n": 2, "fd": 0, "field": ".enemies", "path": ".enemies", "card": "start", "rv": ""})
        for i in range(n_filler):
            add(f"f{i}", filler(i), {"n": 1, "fd": None, "rv": ""})
        self.corpus.save_index({"format": 2, "fights": metas, "scanned": {}, "excluded": {}})
        run_dir = self.corpus.path / "runs" / "r1"
        run_dir.mkdir(parents=True)
        (run_dir / "results.json").write_text(json.dumps({"format": 2, "label": "r1", "emulator": {"commit": "abc", "dirty": False},
                                                          "subset": False, "fights": results, "totals": {}}), encoding="utf-8")
        for rel_, text in DECOMPILE.items():
            f = self.root / "decompiled" / rel_
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text, encoding="utf-8")
        for rel_, text in ENGINE.items():
            f = self.root.parent / "engine" / rel_
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text, encoding="utf-8")
        (self.root / "backlog").mkdir(parents=True)
        (self.root / "backlog" / "ledger.md").write_text(
            "# Ledger\n\n## 2026-10-14, a branch\n\n"
            "- Hardened Shell | the colony loses its cap at 0 (CardEffects.LandDamageOnEnemy lowers the buff) | review of ABCDE12345\n"
            "- shuffle | something unrelated about the Shuffle stream | review\n", encoding="utf-8")
        with (self.root / "backlog" / "fix-attempts.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["attempt_id", "branch", "bug_class", "outcome", "rejection_causes", "notes", "lineage"])
            w.writerow(["a1-HARDENED_SHELL", "fix/shell", "power timing", "rejected", "wrong-rule", f"colony, checked on {SEED_CITED}", "hardened-shell"])
            w.writerow(["a2-OTHER", "fix/other", "pools", "accepted", "", "Hardened Shell is only an example here", "x"])
        probes = self.root / "probes"
        probes.mkdir()
        (probes / "r9.report.json").write_text(json.dumps({"name": "r9", "commit": "def", "verdicts": [
            {"id": "P91", "ledger": ["C1-F99"], "question": "Does Hardened Shell keep its Amount when HP is lost?", "status": "DIFFER",
             "game": 20, "emulator": 14, "game_rule": ["keeps"], "emulator_rule": ["lowers"], "answer": "the Amount stays; the cap is a tally",
             "game_trace": "DamageReceived x2", "note": "probed on the colony"},
            {"id": "P92", "ledger": ["C1-F98"], "question": "Does Inferno fire on Thorns?", "status": "AGREE", "game": 3, "emulator": 3}]}),
            encoding="utf-8")

    def add_refset(self) -> Path:
        """A reference set with one regression class: a minimal setup that fails the way the item says."""
        rs = self.root / "refset" / "v2"
        rec_dir = rs / "record" / "all"
        rec_dir.mkdir(parents=True)
        rec = fight("REFSET0001", 2, setup=[{"cmd": "relics_set", "ids": ["BURNING_BLOOD"]}, {"cmd": "fight", "id": "SKULKING_COLONY_ELITE"}])
        spec = {"id": "G056-REFSET0001", "part": "regression", "class": "power timing: Hardened Shell of the Skulking Colony after Strike", "source": "FINDINGS setup fights",
                "trigger": {"plays": ["STRIKE_IRONCLAD"], "relic_flashes": []}, "flags": [], "reason": "minimal setup", "spec": {}}
        other = {"id": "G057-OTHERSEED1", "part": "regression", "class": "pool order: Splash", "trigger": {"plays": ["SPLASH"]}, "flags": [], "spec": {}}
        (rs / "manifest.json").write_text(json.dumps({"specs": [spec, other]}), encoding="utf-8")
        (rs / "refs-all.json").write_text(json.dumps({"record": str(rec_dir)}), encoding="utf-8")
        (rec_dir / "orders.json").write_text(json.dumps({"a": ["G056-REFSET0001", "G057-OTHERSEED1"]}), encoding="utf-8")
        (rec_dir / "pass-a.out.json").write_text(json.dumps({"results": [rec, fight("OTHERSEED1", 2)]}), encoding="utf-8")
        return rs

    def sources(self, **kw) -> pk.Sources:
        d = dict(referee=self.root, corpus=self.corpus.path, ledger=self.root / "backlog" / "ledger.md", findings=None,
                 attempts=self.root / "backlog" / "fix-attempts.csv", probes=self.root / "probes", trace_dirs=(), minimal_dirs=(),
                 refset=None, decompiled=self.root / "decompiled", emudump=None, codex_walk=None, codex_data=None,
                 engine=self.root.parent / "engine", tests=self.root.parent / "no-tests")
        d.update(kw)
        return pk.Sources(**d)

    def context(self, tmp: Path, emu=None, do_sweep=False, **kw) -> pk.Context:
        cx = pk.Context.build(self.sources(**kw), tmp / "work", tmp / "sweeps", do_sweep=do_sweep)
        if emu is not None:
            cx.emu = emu
        return cx


ITEM = {"id": "T-01", "lane": "fight", "kind": "mechanism", "status": "open", "impact_runs": 1000,
        "title": "Hardened Shell is removed when its amount reaches 0",
        "evidence": f"Game: HardenedShellPower keeps Amount (ModifyHpLostAfterOsty caps it); CreatureCmd.Damage runs it. Main: "
                    f"CardEffects.LandDamageOnEnemy lowers the buff. Seen in {SEED_CITED}.",
        "impact_note": "Skulking Colony in 4898 runs.", "subsystem": "CardEffects.LandDamageOnEnemy"}


class TextTest(unittest.TestCase):
    def setUp(self):
        self.names = pk.Names(ID_MAP)

    def test_entities_read_english_ids_and_pascal(self):
        e = pk.entities_in("The Skulking Colony keeps HardenedShellPower; SELF_FORMING_CLAY and Tungsten Rod matter. The block is gone.", self.names)
        self.assertEqual({"SKULKING_COLONY", "HARDENED_SHELL", "SELF_FORMING_CLAY", "TUNGSTEN_ROD"} - set(e), set())
        self.assertNotIn("BLOCK", e)  # an everyday word that is also a model name

    def test_lone_lowercase_words_are_not_entities(self):
        self.assertEqual(pk.entities_in("the buffer fills and the rage subsides", self.names), {})

    def test_quoted_phrases_skip_apostrophes(self):
        text = "the game's rule; 'enemy move/intent not reproducible after the end of turn' stops, the emulator's AI"
        self.assertEqual(pk.quoted_phrases(text), [pk.norm("enemy move/intent not reproducible after the end of turn")])

    def test_the_opening_state_is_only_for_items_about_it(self):
        self.assertTrue(pk.about_opening({"lane": "harness", "title": "x"}))
        self.assertTrue(pk.about_opening({"lane": "fight", "title": "the enemies differ at start"}))
        self.assertTrue(pk.about_opening({"lane": "fight", "title": "the opening hand is wrong"}))
        self.assertFalse(pk.about_opening({"lane": "fight", "title": "Starting hand and restart of starter decks"}))

    def test_stop_step(self):
        self.assertEqual(pk.stop_step({"stop": "enemy move not reproducible after the end of turn at step 6"}), 6)
        self.assertIsNone(pk.stop_step({}))


class RefsTest(unittest.TestCase):
    def test_refs_find_seeds_calls_hooks_and_emulator_names(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t), n_filler=2)
            cx = w.context(Path(t))
            item = dict(ITEM, evidence=ITEM["evidence"] + " Rule probe P91; see C1-F99. ModifyHpLostAfterOsty and Hook.ModifyHpLost.")
            r = pk.refs_of(item, cx.names, {m["seed"] for m in cx.metas}, cx.decomp)
            self.assertEqual(r.seeds, [SEED_CITED])
            self.assertEqual(r.probe_ids, ["P91"])
            self.assertEqual(r.old_ids, ["C1-F99"])
            self.assertIn(("CreatureCmd", "Damage"), [(c, m) for c, m, _ in r.game_calls])
            self.assertIn(("Hook", "ModifyHpLost"), [(c, m) for c, m, _ in r.game_calls])
            self.assertIn(("ModifyHpLostAfterOsty", "AbstractModel"), r.hooks)
            self.assertIn("CardEffects.LandDamageOnEnemy", r.emu_names)  # not in the decompile: an emulator name
            self.assertIn("HARDENED_SHELL", r.entities)
            self.assertEqual(pk.sweep_targets(r)[:2], ["CreatureCmd.Damage", "Hook.ModifyHpLost"])


class RankTest(unittest.TestCase):
    def rank(self, w, item=ITEM, opening=False):
        cx = w.context(Path(w.root.parent))
        refs = pk.refs_of(item, cx.names, {m["seed"] for m in cx.metas}, cx.decomp)
        return pk.rank_fights(refs, cx.corpus, cx.metas, cx.run, cx.idf, opening), cx

    def test_cited_fights_come_first_and_the_smallest_wins(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t))
            cands, _ = self.rank(w)
            names = [c["id"] for c in cands]
            self.assertEqual(names[0], w.fights["cited"][0])  # cited, 3 steps, before the shorter uncited one
            self.assertIn(w.fights["cited_long"][0], names)
            self.assertTrue(all(c["cited"] for c in cands[:2]))
            self.assertFalse(cands[0]["quoted"])

    def test_an_uncited_fight_needs_names_the_item_also_names(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t))
            cands, _ = self.rank(w)
            named = next(c for c in cands if c["seed"] == SEED_NAMED)
            self.assertFalse(named["cited"])
            self.assertGreaterEqual(named["score"], pk.MIN_SCORE)
            self.assertEqual(named["hit"][:2], ["HARDENED_SHELL", "SKULKING_COLONY"][:2] if named["hit"][0] == "HARDENED_SHELL" else ["SKULKING_COLONY", "HARDENED_SHELL"])
            # the fight with the same enemy but a divergence at the opening state is left out
            self.assertNotIn(SEED_OTHER, [c["seed"] for c in cands])
            cands2, _ = self.rank(w, opening=True)
            self.assertIn(SEED_OTHER, [c["seed"] for c in cands2])  # an item about the opening state may take it

    def test_a_cited_fight_that_no_longer_diverges_is_never_chosen(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t), cited_diverges=False)
            cands, cx = self.rank(w)
            gone = next(c for c in cands if c["id"] == w.fights["cited"][0])
            self.assertFalse(gone["live"])
            first_live = next(c for c in cands if c["live"])
            self.assertNotEqual(first_live["id"], w.fights["cited"][0])
            self.assertEqual(first_live["id"], w.fights["cited_long"][0])  # still cited and live: tier 1 wins over the named fight

    def test_a_quoted_stop_reason_ties_a_fight(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t), n_filler=2)
            fid = w.fights["named"][0]
            path = w.corpus.path / "runs" / "r1" / "results.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["fights"][fid] = {"n": 2, "fd": None, "rv": "M-", "stop": "enemy move/intent not reproducible after the end of turn at step 1"}
            path.write_text(json.dumps(doc), encoding="utf-8")
            item = {"id": "T-02", "title": "x", "evidence": "17 fights stop at 'enemy move/intent not reproducible after the end of turn'"}
            cands, _ = self.rank(w, item)
            self.assertEqual([c["id"] for c in cands if c["quoted"]], [fid])
            self.assertEqual(cands[0]["step"], 1)


class PacketTest(unittest.TestCase):
    def build(self, t, emu, item=ITEM, **world_kw):
        w = World(Path(t), **world_kw)
        cx = w.context(Path(t), emu)
        out = Path(t) / "out"
        pkt = pk.build_packet(item, cx, out)
        return w, cx, out, pkt

    def test_the_packet_names_the_case_the_fields_and_every_section(self):
        with tempfile.TemporaryDirectory() as t:
            w, cx, out, pkt = self.build(t, FakeEmu())
            text = pkt["text"]
            self.assertEqual([h for h in text.splitlines() if h.startswith("## ")][:3], ["## Completeness", "## The item", "## 1. Smallest failing case [full]"])
            for n in range(1, 8):
                self.assertIn(f"## {n}. ", text)
            sec = pkt["sections"]
            self.assertEqual(sec["case"][0], "full")
            self.assertEqual(pkt["chosen"], w.fights["cited"][0])
            self.assertIn("the item cites its seed", text)
            self.assertIn("first divergence, plain replay: step 1 after play STRIKE_IRONCLAD", text)
            self.assertIn("HARDENED_SHELL:20  emulator HARDENED_SHELL:14", text)  # the differing power, game against emulator
            self.assertIn("re-anchored on the game's state before it: DIVERGES", text)
            self.assertTrue((out / "T-01" / f"{SEED_CITED}.case.out.json").is_file())  # the case, replayable
            # the item's cited fights beyond the chosen one are listed; the long recording of the same seed folds in
            self.assertIn("other fights tied to the item", text)

    def test_without_a_trace_the_event_section_is_partial_and_says_what_needs_the_game(self):
        with tempfile.TemporaryDirectory() as t:
            _, _, _, pkt = self.build(t, FakeEmu())
            text = pkt["text"]
            self.assertEqual(pkt["sections"]["event"][0], "partial")
            self.assertIn("NEEDS A GAME RUN (not run here)", text)
            self.assertIn("trace_divergent.py", text)
            self.assertEqual(pkt["sections"]["checksum"][0], "missing")
            self.assertIn("needs the game", pkt["sections"]["checksum"][1])

    def test_a_traced_record_gives_the_trace_diff_and_the_draw_difference(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t))
            fid, rec = w.fights["cited"]
            traced = json.loads(json.dumps(rec))
            traced["steps"][1]["trace"] = [["M", "play", "STRIKE_IRONCLAD", mini(70, 20)], ["R", "Shuffle", 2, "ListExtensions.UnstableShuffle < X.Y"],
                                           ["E", "DamageReceived", "SKULKING_COLONY#0", "player", 10, 0, "STRIKE_IRONCLAD"],
                                           ["M", "played", "STRIKE_IRONCLAD", mini(70, 20)]]
            tdir = Path(t) / "traces" / "n"
            tdir.mkdir(parents=True)
            (tdir / "traced.out.json").write_text(json.dumps({"results": [traced]}), encoding="utf-8")
            cx = w.context(Path(t), FakeEmu(shell_emulator=14), trace_dirs=(Path(t) / "traces",))
            pkt = pk.build_packet(ITEM, cx, Path(t) / "out")
            self.assertEqual(pkt["sections"]["event"][0], "full", pkt["sections"]["event"])
            self.assertIn("FIRST DIFFERENCE", pkt["text"])
            self.assertIn("traced game record", pkt["text"])
            self.assertEqual(pkt["sections"]["rng"][0], "full")  # the trace shows a draw the emulator never makes
            self.assertIn("FIRST DRAW DIFFERENCE on Shuffle", pkt["text"])
            self.assertTrue((Path(t) / "out" / "T-01" / f"{SEED_CITED}.trace.txt").is_file())

    def test_an_rng_stream_that_differs_without_a_trace_gets_counters_and_callers(self):
        with tempfile.TemporaryDirectory() as t:
            _, _, _, pkt = self.build(t, FakeEmu(extra_draw=True))
            self.assertEqual(pkt["sections"]["rng"][0], "partial")
            self.assertIn("Shuffle: game 0, emulator 1", pkt["text"])
            self.assertIn("by CardEffects.X", pkt["text"])

    def test_no_emudump_falls_back_to_the_stored_run(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t))
            cx = w.context(Path(t))
            self.assertFalse(cx.emu.available)
            pkt = pk.build_packet(ITEM, cx, Path(t) / "out")
            self.assertEqual(pkt["sections"]["case"][0], "partial")
            self.assertIn("no EmuDump", pkt["sections"]["case"][1])
            self.assertIn("step 1", pkt["text"])

    def test_probe_ledger_and_attempts_are_found(self):
        with tempfile.TemporaryDirectory() as t:
            item = dict(ITEM, evidence=ITEM["evidence"] + " Rule probe P91.")
            _, _, _, pkt = self.build(t, FakeEmu(), item)
            text = pkt["text"]
            self.assertEqual(pkt["sections"]["probe"][0], "full")
            self.assertIn("the Amount stays; the cap is a tally", text)
            self.assertNotIn("Does Inferno fire", text)  # P02 shares nothing with the item
            self.assertEqual(pkt["sections"]["ledger"][0], "full")
            self.assertIn("CardEffects.LandDamageOnEnemy lowers the buff", text)
            self.assertNotIn("unrelated about the Shuffle stream", text)
            self.assertIn("earlier fix attempt a1-HARDENED_SHELL", text)
            self.assertNotIn("a2-OTHER", text)  # one shared model name and no seed: not about this item

    def test_an_unprobed_rule_says_so(self):
        with tempfile.TemporaryDirectory() as t:
            item = dict(ITEM, id="T-03", title="Tungsten Rod", evidence="Tungsten Rod ignores nothing.", subsystem="", impact_note="")
            _, _, _, pkt = self.build(t, FakeEmu(), item)
            self.assertEqual(pkt["sections"]["probe"][0], "missing")
            self.assertIn("not probed", pkt["text"])

    def test_classes_section_names_game_files_hooks_and_engine_sites(self):
        with tempfile.TemporaryDirectory() as t:
            _, _, _, pkt = self.build(t, FakeEmu())
            text = pkt["text"]
            self.assertIn("Models.Powers/HardenedShellPower.cs (overrides ModifyHpLostAfterOsty, BeforeSideTurnStart)", text)
            self.assertIn("game CreatureCmd.Damage: Commands/CreatureCmd.cs", text)
            self.assertIn("emulator CardEffects.LandDamageOnEnemy (named by the item): engine/Core/Effects/CardEffects.cs:4", text.replace("src/Sts2Emulator", "engine"))
            self.assertIn("sweeps not run (--no-sweep)", text)
            self.assertEqual(pkt["sections"]["classes"][0], "partial")

    def test_a_tests_item_is_not_applicable_to_a_fight_step(self):
        with tempfile.TemporaryDirectory() as t:
            item = {"id": "T-04", "lane": "tests", "kind": "mechanism", "status": "open", "title": "mutation kill rate plateaus",
                    "evidence": "report only", "impact_note": "", "subsystem": "tools/fidelity/mutate.py"}
            _, _, _, pkt = self.build(t, FakeEmu(), item)
            s = pkt["sections"]
            self.assertEqual([s[k][0] for k in ("case", "event", "checksum", "rng", "probe")], ["n/a"] * 5)
            self.assertIn("tools/fidelity/mutate.py", pkt["text"])  # the code the item is about

    def test_the_packet_never_starts_a_game(self):
        calls = []
        real = subprocess.run

        def spy(cmd, *a, **k):
            calls.append(list(cmd) if isinstance(cmd, (list, tuple)) else [cmd])
            return real(cmd, *a, **k)

        with tempfile.TemporaryDirectory() as t, mock.patch.object(subprocess, "run", spy):
            self.build(t, FakeEmu())
        flat = " ".join(" ".join(map(str, c)) for c in calls).lower()
        for banned in ("serve.py", "run_referee", "trace_divergent", "sts2.exe", "scenarios.py", "batch.py"):
            self.assertNotIn(banned, flat)
        self.assertTrue(all(c[0] == "git" for c in calls), calls)


class NetstateTest(unittest.TestCase):
    """A traced game record that also holds the game's full state after each checksum, against EmuDump fight --netstate."""

    def test_the_game_subaction_where_the_emulator_stops_matching_is_named(self):
        from tests.python import (
            test_netstate_diff as tn,  # fixtures: the shapes ChecksumTap and --netstate write
        )

        g, e = tn.fight([(None, [("After player turn start", tn.net())], tn.net(game=False)),
                         (tn.PLAY, [(tn.PLAYED, tn.net(ehp=21, hand=()))], tn.net(ehp=24, hand=(), game=False))])
        for st in g["steps"]:
            st["trace"] = []
        with tempfile.TemporaryDirectory() as t:
            tdir = Path(t) / "traces"
            tdir.mkdir()
            (tdir / "traced.out.json").write_text(json.dumps({"results": [g]}), encoding="utf-8")

            class Emu:
                available = True

                def run(self, mode, fights, *extra):
                    if mode == "fight":
                        return {"results": [e]}
                    return {"results": [{"seed": "T", "steps": [{} for _ in g["steps"]]}]}  # no emulator trace: trace_diff reports it

            case = pk.Case(rec=g, step=1, live=True)
            tr = pk.trace_case(case, pk.TraceStore([tdir]), Emu(), pk.fc.names(ID_MAP))
            self.assertTrue(tr.has_net)
            self.assertEqual(tr.net["step"], 1)
            self.assertIn("finished action execution PlayCardAction", tr.net["subaction"])
            sec = pk.Section("checksum", "x")
            lines = pk.checksum_section(case, tr, None, None, sec)
            self.assertEqual(sec.status, "full")
            self.assertTrue(any("first divergent sub-action" in x for x in lines))
            self.assertTrue(any("enemies[0]" in x and "21" in x and "24" in x for x in lines))  # the game's 21 against the emulator's 24


class RefsetTest(unittest.TestCase):
    def test_a_regression_class_the_item_names_is_replayed_and_can_be_the_case(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t))
            rs = w.add_refset()
            cx = w.context(Path(t), FakeEmu(), refset=rs)
            item = dict(ITEM, evidence=ITEM["evidence"] + " Reference set G056 holds its minimal setup.")
            pkt = pk.build_packet(item, cx, Path(t) / "out")
            self.assertEqual(pkt["chosen"], "G056-REFSET0001")  # cited like the corpus fight, and shorter
            self.assertIn("reference set, regression class: power timing: Hardened Shell of the Skulking Colony after Strike", pkt["text"])
            self.assertIn("G056-REFSET0001", pkt["text"])
            self.assertNotIn("G057-OTHERSEED1", pkt["text"])  # another class: not tied to the item
            self.assertEqual(pkt["sections"]["case"][0], "full")

    def test_a_class_is_tied_by_two_rare_names_in_its_text_and_trigger(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t))
            cx = w.context(Path(t), FakeEmu(), refset=w.add_refset())
            refs = pk.refs_of(ITEM, cx.names, {m["seed"] for m in cx.metas}, cx.decomp)
            got = pk.refset_candidates(refs, cx)
            self.assertEqual([c["id"] for c in got], ["G056-REFSET0001"])  # HARDENED_SHELL and SKULKING_COLONY, both in the class text, both rare
        # without an emulator nothing can be judged: no candidates, never a guess
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t))
            cx = w.context(Path(t), None, refset=w.add_refset())
            refs = pk.refs_of(dict(ITEM, evidence="refset G056"), cx.names, set(), cx.decomp)
            self.assertEqual(pk.refset_candidates(refs, cx), [])

    def test_fight_mode_loads_a_reference_set_id(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t), n_filler=2)
            cx = w.context(Path(t), FakeEmu(), refset=w.add_refset())
            rec, meta = cx.load_fight("G056-REFSET0001")
            self.assertTrue(meta["ref"])
            self.assertEqual(rec["seed"], "REFSET0001")
            self.assertIsNone(cx.load_fight("G999-NOSUCHSEED"))


class RunLayerTest(unittest.TestCase):
    def walk_dir(self, t):
        d = Path(t) / "walk" / "main-abc"
        d.mkdir(parents=True)
        rows = [
            {"seed": "RUNAAAAAA1", "ascension": 10, "character": "IRONCLAD", "status": "mismatch", "index": 7, "floors_recorded": 12,
             "stop": {"act": 1, "floor": 6, "component": "ancient", "field": "options", "expected": "A,B,C", "emulator": "A,B,D",
                      "class": "PROFILE-DEPENDENT", "streams": ["UpFront"]}},
            {"seed": "RUNBBBBBB2", "ascension": 10, "character": "IRONCLAD", "status": "mismatch", "index": 9, "floors_recorded": 5,
             "stop": {"act": 1, "floor": 1, "component": "ancient", "field": "options", "expected": "A,B,C", "emulator": "A,B,E",
                      "class": "PROFILE-DEPENDENT", "streams": ["UpFront"]}},
            {"seed": "RUNCCCCCC3", "ascension": 10, "character": "IRONCLAD", "status": "mismatch", "index": 11, "floors_recorded": 5,
             "stop": {"act": 2, "floor": 20, "component": "gold_reward", "field": "gold gained", "expected": "10", "emulator": "12",
                      "class": "EMULATOR", "streams": ["Rewards", "Niche"]}},
            {"seed": "RUNDDDDDD4", "status": "complete", "index": 12, "floors_recorded": 3},
        ]
        (d / "s0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return d.parent

    def test_walk_stops_are_matched_by_class_component_and_the_earliest_run_is_the_case(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t), n_filler=2)
            cx = w.context(Path(t), codex_walk=self.walk_dir(t))
            self.assertEqual(cx.walk.label, "main-abc")
            self.assertEqual((cx.walk.total, len(cx.walk.rows)), (4, 3))
            item = {"id": "T-05", "lane": "run", "kind": "mechanism", "status": "open", "title": "Neow options and the ancient choice",
                    "evidence": "PROFILE-DEPENDENT stops: act-1 ancient options", "impact_note": "", "subsystem": ""}
            refs = pk.refs_of(item, cx.names, set(), cx.decomp)
            rc_ = pk.run_case(item, refs, cx)
            self.assertEqual(pk.signature(rc_["row"]), ("PROFILE-DEPENDENT", "ancient", "options"))
            self.assertEqual(rc_["row"]["seed"], "RUNBBBBBB2")  # act 1 floor 1 beats floor 6
            self.assertEqual(len(rc_["sigs"][0][2]), 2)
            cx.run_cases["T-05"] = rc_
            pkt = pk.build_packet(item, cx, Path(t) / "out")
            s = pkt["sections"]
            self.assertEqual((s["case"][0], s["event"][0], s["checksum"][0], s["rng"][0]), ("full", "full", "n/a", "partial"))
            self.assertIn("run `RUNBBBBBB2`", pkt["text"])
            self.assertIn("the game (history) had: A,B,C", pkt["text"])

    def test_a_cited_run_beats_the_signature_pick(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t), n_filler=2)
            cx = w.context(Path(t), codex_walk=self.walk_dir(t))
            item = {"id": "T-06", "lane": "run", "status": "open", "title": "ancient options", "evidence": "see RUNAAAAAA1", "impact_note": ""}
            rc_ = pk.run_case(item, pk.refs_of(item, cx.names, set(), cx.decomp), cx)
            self.assertTrue(rc_["cited"])
            self.assertEqual(rc_["row"]["seed"], "RUNAAAAAA1")

    def test_floor_lines_mark_the_stop_and_the_neow_choice(self):
        rec = {"map_point_history": [[
            {"map_point_type": "ancient", "rooms": [{"room_type": "ancient"}], "player_stats": [{"current_hp": 70, "max_hp": 80, "current_gold": 99,
             "ancient_choice": [{"TextKey": "A", "was_chosen": False}, {"TextKey": "B", "was_chosen": True}]}]},
            {"map_point_type": "monster", "rooms": [{"model_id": "ENCOUNTER.X_WEAK"}], "player_stats": [{"current_hp": 60, "max_hp": 80, "current_gold": 109,
             "card_choices": [{"card": {"id": "CARD.BASH"}, "was_picked": True}, {"card": {"id": "CARD.ANGER"}, "was_picked": False}]}]}]]}
        lines = pk.run_floor_lines(rec, {"floor": 2})
        self.assertIn("*B", lines[0])
        self.assertTrue(lines[1].endswith("<- the stop"))
        self.assertIn("*BASH, ANGER", lines[1])


class OutputTest(unittest.TestCase):
    def test_write_packets_writes_the_index_and_completeness(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t), n_filler=2)
            cx = w.context(Path(t), FakeEmu())
            out = Path(t) / "packets"
            items = [ITEM, dict(ITEM, id="T-09", status="open")]
            res = pk.write_packets(items, cx, out, jobs=1, quiet=True)
            self.assertEqual(len(res), 2)
            index = (out / "index.md").read_text(encoding="utf-8")
            self.assertIn("[T-01](T-01.md)", index)
            self.assertIn("each fixer is given the path of its packet", index)
            comp = json.loads((out / "completeness.json").read_text(encoding="utf-8"))
            self.assertEqual(comp["T-01"]["case"][0], "full")
            self.assertEqual(set(comp["T-01"]), {k for k, _ in pk.SECTIONS})

    def test_open_items_skips_other_states_and_keeps_the_backlog_order(self):
        doc = {"items": [{"id": "A", "status": "open"}, {"id": "B", "status": "in-progress-elsewhere"}, {"id": "C", "status": "open"},
                         {"id": "D", "status": "open"}]}
        self.assertEqual([i["id"] for i in pk.open_items(doc, 2)], ["A", "C"])

    def test_the_failing_fight_mode_builds_an_item_around_a_corpus_seed(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t), n_filler=2)
            cx = w.context(Path(t), FakeEmu())
            item = {"id": "fight-x", "lane": "fight", "kind": "failing fight", "status": "open", "title": "failing fight", "evidence": "",
                    "impact_note": "", "subsystem": ""}
            pkt = pk.build_packet(item, cx, Path(t) / "out", [SEED_NAMED])
            self.assertEqual(pkt["chosen"], w.fights["named"][0])
            self.assertEqual(pkt["sections"]["case"][0], "full")
            self.assertIn("HARDENED_SHELL", pkt["refs"].entities)  # read from the divergence itself


class SweepTest(unittest.TestCase):
    def test_a_sweep_table_round_trips_through_the_cache(self):
        with tempfile.TemporaryDirectory() as t:
            w = World(Path(t), n_filler=1)
            cx = w.context(Path(t))
            out = Path(t) / "sweeps" / "CreatureCmd.Damage.md"
            res = pk.make_sweep("CreatureCmd.Damage", cx.decomp.root, cx.engine, {}, 1, out)
            back = pk.parse_sweep(out, "CreatureCmd.Damage")
            self.assertEqual(res["n"], back["n"])
            lines = pk.sweep_summary(back, {"HARDENED_SHELL"}, cx.names, "sweeps/x.md")
            self.assertIn("full table: sweeps/x.md", lines[0])


if __name__ == "__main__":
    unittest.main()
