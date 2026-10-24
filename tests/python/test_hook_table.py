"""tools/fidelity/hook_table.py and hook_order.py: the game's hook table, its mapping onto the emulator, and the
listener-order check, each seen to pass on a faithful case and to FAIL on a deliberately broken one.

Fixtures are a few lines of made-up C# in the shapes the tools read (no game code). Two tests use the real
decompile and engine when they are there: the checked-in table has no unlisted override, and (opt-in,
SPIRELINE_HOOK_E2E=1, builds an instrumented engine) moving one dispatch call out of its phase fails the order
gate.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import hook_order as ho  # noqa: E402
from tools.fidelity import hook_table as ht  # noqa: E402

ABSTRACT = """
namespace MegaCrit.Sts2.Core.Models;
public abstract class AbstractModel
{
	public virtual Task AfterCardPlayed(PlayerChoiceContext choiceContext, CardPlay cardPlay)
	{
		return Task.CompletedTask;
	}
	public virtual Task AfterCardPlayedLate(PlayerChoiceContext choiceContext, CardPlay cardPlay)
	{
		return Task.CompletedTask;
	}
	public virtual Task BeforeHandDraw(Player player, PlayerChoiceContext choiceContext, ICombatState combatState)
	{
		return Task.CompletedTask;
	}
	public virtual Task AfterPlayerTurnStart(PlayerChoiceContext choiceContext, Player player)
	{
		return Task.CompletedTask;
	}
}
"""
HOOK = """
namespace MegaCrit.Sts2.Core.Hooks;
public static class Hook
{
	public static async Task AfterCardPlayed(ICombatState combatState, PlayerChoiceContext choiceContext, CardPlay cardPlay)
	{
		foreach (AbstractModel model2 in combatState.IterateHookListeners())
		{
			await model2.AfterCardPlayed(choiceContext, cardPlay);
		}
		foreach (AbstractModel model2 in combatState.IterateHookListeners())
		{
			await model2.AfterCardPlayedLate(choiceContext, cardPlay);
		}
	}
	public static async Task BeforeHandDraw(ICombatState combatState, Player player, PlayerChoiceContext ctx)
	{
		foreach (AbstractModel model in IterateCombatHookListeners(combatState))
		{
			await model.BeforeHandDraw(player, ctx, combatState);
		}
	}
	public static async Task AfterPlayerTurnStart(ICombatState combatState, PlayerChoiceContext ctx, Player player)
	{
		foreach (AbstractModel model in IterateCombatHookListeners(combatState))
		{
			await model.AfterPlayerTurnStart(ctx, player);
		}
	}
}
"""
MODELS = {
    "MegaCrit.Sts2.Core.Models/RelicModel.cs": "public abstract class RelicModel : AbstractModel\n{\n}\n",
    "MegaCrit.Sts2.Core.Models/CardModel.cs": "public abstract class CardModel : AbstractModel\n{\n}\n",
    "MegaCrit.Sts2.Core.Models.Relics/PenNib.cs": """
public sealed class PenNib : RelicModel
{
	public override Task AfterCardPlayed(PlayerChoiceContext choiceContext, CardPlay cardPlay)
	{
		Counter++;
		return Task.CompletedTask;
	}
}
""",
    "MegaCrit.Sts2.Core.Models.Relics/Lantern.cs": """
public sealed class Lantern : RelicModel
{
	public override Task AfterCardPlayedLate(PlayerChoiceContext choiceContext, CardPlay cardPlay)
	{
		Flash();
		return Task.CompletedTask;
	}
}
""",
    "MegaCrit.Sts2.Core.Models.Relics/Toolbox.cs": """
public abstract class ToolboxBase : RelicModel
{
	public override Task BeforeHandDraw(Player player, PlayerChoiceContext choiceContext, ICombatState combatState)
	{
		Uses = Uses + 1;
		return Task.CompletedTask;
	}
}
""",
    "MegaCrit.Sts2.Core.Models.Relics/Toolbox2.cs": """
public sealed class Toolbox : ToolboxBase
{
}
""",
    "MegaCrit.Sts2.Core.Models.Relics/Bellows.cs": """
public sealed class Bellows : RelicModel
{
	public override Task AfterPlayerTurnStart(PlayerChoiceContext choiceContext, Player player)
	{
		Draws++;
		return Task.CompletedTask;
	}
}
""",
}
RELICS_G = """
internal static class Relics
{
    private static readonly RelicDef[] _all =
    [
        new RelicDef(Id: 1, Name: "PenNib", Entry: "PEN_NIB"),
        new RelicDef(Id: 2, Name: "Lantern", Entry: "LANTERN"),
        new RelicDef(Id: 3, Name: "Toolbox", Entry: "TOOLBOX"),
        new RelicDef(Id: 4, Name: "Bellows", Entry: "BELLOWS"),
    ];
}
"""
RELIC_EFFECTS = """
namespace Sts2Emulator.Core.Effects;

public static class RelicEffects
{
    public const int PenNib = 1;
    public const int Lantern = 2;
    public const int Toolbox = 3;
    public const int Bellows = 4;

    public static void ApplyAfterCardPlayed(CombatState state)
    {
        if (HasRelic(state, PenNib))
        {
            state.Counter++;
        }
    }

    public static void ApplyBeforeHandDraw(CombatState state)
    {
        // `Toolbox.BeforeHandDraw`: one card from three.
        if (HasRelic(state, Toolbox))
        {
            state.Pick = 1;
        }
    }

    public static void ApplyAfterPlayerTurnStart(CombatState state)
    {
        if (HasRelic(state, Bellows))
        {
            state.Draws++;
        }
    }

    private static bool HasRelic(CombatState state, int id) => state.Relics.Contains(id);
}
"""
MANIFEST = {"hooks": {"AfterCardPlayed": {"methods": ["RelicEffects.ApplyAfterCardPlayed"]},
                      "AfterPlayerTurnStart": {"methods": ["RelicEffects.ApplyAfterPlayerTurnStart"]}}}
UNMODELLED = {"ui_only": "feedback only"}


# The turn start's last hook, for the order tests only (the table tests count the overrides of the fixture above).
ABSTRACT_SIDE = ABSTRACT.rstrip()[:-1] + """	public virtual Task AfterSideTurnStart(CombatSide side, ICombatState combatState)
	{
		return Task.CompletedTask;
	}
}
"""
HOOK_SIDE = HOOK.rstrip()[:-1] + """	public static async Task AfterSideTurnStart(ICombatState combatState, CombatSide side)
	{
		foreach (AbstractModel model in IterateCombatHookListeners(combatState))
		{
			await model.AfterSideTurnStart(side, combatState);
		}
	}
}
"""


class Fixture:
    def __init__(self, relic_effects: str = RELIC_EFFECTS, abstract: str = ABSTRACT, hook: str = HOOK):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.decompiled = base / "decompiled"
        for rel, text in {"MegaCrit.Sts2.Core.Models/AbstractModel.cs": abstract,
                          "MegaCrit.Sts2.Core.Hooks/Hook.cs": hook, **MODELS}.items():
            p = self.decompiled / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(text), encoding="utf-8")
        self.engine_root = base / "engine"
        (self.engine_root / "Generated").mkdir(parents=True)
        (self.engine_root / "Generated" / "Relics.g.cs").write_text(RELICS_G, encoding="utf-8")
        self.texts = {"Core/Effects/RelicEffects.cs": relic_effects}

    def engine(self) -> ht.Engine:
        return ht.Engine.load(self.engine_root, self.texts)

    def check(self, open_list=None) -> dict:
        _clear_caches()
        return ht.check(str(self.decompiled), self.engine(), MANIFEST, UNMODELLED, open_list or {})

    def close(self):
        self.tmp.cleanup()


def _clear_caches():
    for f in (ht.hook_names, ht.dispatchers, ht.model_classes, ht._mentions, ht.multiplayer_cards):
        f.cache_clear()


class GameTable(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()

    def tearDown(self):
        self.fx.close()

    def test_overrides_include_inherited_ones_and_skip_abstract_bases(self):
        _clear_caches()
        ov = {(o.model, o.hook, o.declared_in) for o in ht.overrides(str(self.fx.decompiled))}
        self.assertIn(("Toolbox", "BeforeHandDraw", "ToolboxBase"), ov)
        self.assertNotIn("ToolboxBase", {m for m, _, _ in ov})
        self.assertEqual(4, len(ov))

    def test_dispatcher_passes_in_order(self):
        _clear_caches()
        d = ht.dispatchers(str(self.fx.decompiled))
        self.assertEqual(["AfterCardPlayed", "AfterCardPlayedLate"], d["AfterCardPlayed"]["passes"])
        self.assertEqual(["combat-direct"], d["AfterCardPlayed"]["iterator"])
        self.assertEqual(["combat"], d["BeforeHandDraw"]["iterator"])

    def test_game_order_key_powers_before_relics_and_passes_first(self):
        power = {"kind": "power", "creature": 0, "index": 3}
        relic = {"kind": "relic", "creature": 0, "index": 0}
        enemy_power = {"kind": "power", "creature": 1, "index": 0}
        self.assertLess(ht.game_order_key(power), ht.game_order_key(relic))
        self.assertLess(ht.game_order_key(relic), ht.game_order_key(enemy_power))
        self.assertLess(ht.game_order_key(enemy_power, 0), ht.game_order_key(power, 1))


class StaticCheck(unittest.TestCase):
    def test_faithful_engine_passes(self):
        fx = Fixture()
        try:
            res = fx.check()
            status = {f"{r['model']}.{r['hook']}": r["status"] for r in res["rows"]}
            self.assertEqual("registered", status["PenNib.AfterCardPlayed"])          # named at its dispatch point
            self.assertEqual("registered", status["Toolbox.BeforeHandDraw"])         # cited in a comment there
            self.assertEqual("not_modelled", status["Lantern.AfterCardPlayedLate"])  # a flash only
            self.assertEqual([], res["problems"])
        finally:
            fx.close()

    def test_broken_dropped_registration_fails(self):
        fx = Fixture(RELIC_EFFECTS.replace("if (HasRelic(state, Bellows))", "if (state.Turn > 99)"))
        try:
            res = fx.check()
            self.assertEqual(1, res["counts"]["missing"])
            self.assertTrue(any(p.startswith("MISSING Bellows.AfterPlayerTurnStart") for p in res["problems"]))
            # Listed as open, it passes; the open entry goes stale once the registration is back.
            self.assertEqual([], fx.check({"missing Bellows.AfterPlayerTurnStart": ""})["problems"])
        finally:
            fx.close()
        fx = Fixture()
        try:
            res = fx.check({"missing Bellows.AfterPlayerTurnStart": ""})
            self.assertTrue(any(p.startswith("STALE open entry Bellows") for p in res["problems"]))
        finally:
            fx.close()

    def test_broken_dispatch_point_in_the_wrong_hook_fails(self):
        # Bellows named only in the after-card-played dispatch point: not registered at its turn-start hook.
        moved = RELIC_EFFECTS.replace("if (HasRelic(state, Bellows))", "if (state.Turn > 99)").replace(
            "if (HasRelic(state, PenNib))", "if (HasRelic(state, PenNib) || HasRelic(state, Bellows))")
        fx = Fixture(moved)
        try:
            res = fx.check()
            self.assertIn("RelicEffects.ApplyAfterCardPlayed",
                          next(p for p in res["problems"] if p.startswith("MISSING Bellows")))
        finally:
            fx.close()

    def test_stale_manifest_method_fails(self):
        fx = Fixture()
        try:
            _clear_caches()
            man = {"hooks": dict(MANIFEST["hooks"], AfterCardPlayed={"methods": ["RelicEffects.Gone"]})}
            res = ht.check(str(fx.decompiled), fx.engine(), man, UNMODELLED, {})
            self.assertTrue(any("STALE dispatch point AfterCardPlayed" in p for p in res["problems"]))
        finally:
            fx.close()

    def test_stale_generic_entry_fails(self):
        """A `generic` entry names the method that serves the listener; one that is renamed away is stale."""
        fx = Fixture()
        try:
            _clear_caches()
            hooks = dict(MANIFEST["hooks"], AfterCardPlayed={
                "methods": ["RelicEffects.ApplyAfterCardPlayed"],
                "generic": {"Lantern": "RelicEffects.Gone: the late pass is derived on read"}})
            res = ht.check(str(fx.decompiled), fx.engine(), {"hooks": hooks}, UNMODELLED, {})
            self.assertTrue(any("STALE generic entry AfterCardPlayed/Lantern" in p for p in res["problems"]), res["problems"])
            hooks["AfterCardPlayed"]["generic"]["Lantern"] = "RelicEffects.ApplyAfterCardPlayed: the late pass is derived on read"
            res = ht.check(str(fx.decompiled), fx.engine(), {"hooks": hooks}, UNMODELLED, {})
            self.assertFalse(any("STALE generic" in p for p in res["problems"]), res["problems"])
        finally:
            fx.close()

    def test_run_side_card_id_names_the_card(self):
        """A quest card the run side holds as `<Name>Card` (RunConstants.SpoilsMapCard) is named where it is read."""
        fx = Fixture(RELIC_EFFECTS + "\npublic static class RunMapGenerator\n{\n    static bool Holds(RunState s) => "
                     "s.Deck.Any(c => c.DefId == RunConstants.PenNibCard);\n}\n")
        try:
            _clear_caches()
            engine = fx.engine()
            self.assertTrue(ht.references(engine, "card", "PenNib"))
            self.assertFalse(ht.references(engine, "card", "Lantern"))
        finally:
            fx.close()


SNIPPET = """
public static class X
{
    public static void A(State s)
    {
        int ritual = Get(s, BuffId.Ritual);
        if (ritual > 0)
        {
            Gain(s, ritual);
        }
        if (s.Ok) Foo(BuffId.Tender);
        else if (Has(s, BuffId.Rage)) Bar();
        else Baz(BuffId.Rupture);
        switch (s.Id)
        {
            case BuffId.Vigor:
            case BuffId.Weak:
                Qux();
                break;
        }
        var list = new List<int> { 1, Get(s, BuffId.Frail) };
    }
}
"""


class Instrumenting(unittest.TestCase):
    def _points(self):
        engine = ht.Engine.load(Path(), {"Core/X.cs": SNIPPET})
        rel = "Core/X.cs"
        m = engine.masked[rel]
        rows = []
        for name in ("Ritual", "Tender", "Rage", "Rupture", "Vigor", "Frail"):
            off = m.index(f"BuffId.{name}")
            rows.append(ht.Row(f"{name}Power", "power", "AfterCardPlayed", "registered", sites=[[rel, off]]))
        points, skipped = ho.plan_probes(engine, rows)
        return engine, points, skipped

    def test_forms_and_lines_kept(self):
        engine, points, skipped = self._points()
        self.assertEqual([], skipped)
        forms = {p.pairs[0][0]: form for (_, _, form), p in points.items()}
        self.assertEqual("cond", forms["RitualPower"])        # moved from the read into its use, an if
        self.assertEqual("wrap", forms["TenderPower"])        # an embedded statement gets braces
        self.assertEqual("cond", forms["RagePower"])          # else-if: inside the condition
        self.assertEqual("wrap", forms["RupturePower"])       # else <statement>: braces, the else keeps it
        self.assertEqual("hit", forms["VigorPower"])          # after the section's last label
        self.assertEqual("hit", forms["FrailPower"])
        out = ho.apply_probes(engine.files, engine.masked, points, set())["Core/X.cs"]
        self.assertEqual(SNIPPET.count("\n"), out.count("\n"))
        self.assertIn("case BuffId.Weak:global::Sts2Emulator.HookProbe.Hit(", out)
        self.assertIn("else { global::Sts2Emulator.HookProbe.Hit(", out)
        self.assertIn("{ 1, Get(s, BuffId.Frail) }", out)          # nothing inside the initializer
        self.assertEqual(out.count("{"), out.count("}"))


def _probe(pid, model, kind, hook):
    return ho.Probe(pid, "Core/X.cs", pid, [(model, kind, hook)])


class OrderCheck(unittest.TestCase):
    """A swapped dispatch must fail the order check: listeners, then phases."""

    def setUp(self):
        self.fx = Fixture(abstract=ABSTRACT_SIDE, hook=HOOK_SIDE)
        _clear_caches()
        self.d = str(self.fx.decompiled)
        self.probes = {0: _probe(0, "TenderPower", "power", "AfterCardPlayed"),
                       1: _probe(1, "PenNib", "relic", "AfterCardPlayed"),
                       2: _probe(2, "Lantern", "relic", "AfterCardPlayedLate"),
                       3: _probe(3, "Toolbox", "relic", "BeforeHandDraw"),
                       4: _probe(4, "Bellows", "relic", "AfterPlayerTurnStart"),
                       5: _probe(5, "Candle", "relic", "AfterSideTurnStart")}
        self.listener_of = {("TenderPower", "power"): "p", ("PenNib", "relic"): "r1", ("Lantern", "relic"): "r2",
                            ("Toolbox", "relic"): "r3", ("Bellows", "relic"): "r4", ("Candle", "relic"): "r5"}
        self.positions = {"p": {"kind": "power", "creature": 0, "index": 0},
                          "r1": {"kind": "relic", "creature": 0, "index": 0},
                          "r2": {"kind": "relic", "creature": 0, "index": 1},
                          "r3": {"kind": "relic", "creature": 0, "index": 2},
                          "r4": {"kind": "relic", "creature": 0, "index": 3},
                          "r5": {"kind": "relic", "creature": 0, "index": 4}}

    def tearDown(self):
        self.fx.close()

    def _keys(self, log):
        inv, _ = ho.analyse(self.d, log, self.probes, self.positions, self.listener_of)
        return ho.order_keys(ho.summarise(inv)), inv

    def test_game_order_passes(self):
        # a play: (before) | play | effect | played | power, relic, then the Late relic
        keys, inv = self._keys([-101, -4, -5, 0, 1, 2, -102, -3, 3, 4])
        self.assertEqual(set(), keys)
        self.assertEqual([], inv)

    def test_relic_before_power_fails(self):
        keys, _ = self._keys([-101, -4, -5, 1, 0, 2])
        self.assertEqual({"order play_after AfterCardPlayed kind relic>power"}, keys)

    def test_late_pass_before_normal_fails(self):
        keys, _ = self._keys([-101, -4, -5, 2, 0, 1])
        self.assertTrue(any(" pass " in k for k in keys), keys)

    def test_phase_swap_fails(self):
        # turn start: AfterPlayerTurnStart's relic before the BeforeHandDraw one
        keys, _ = self._keys([-101, -3, 4, 3])
        self.assertEqual({"order turn_start BeforeHandDraw<AfterPlayerTurnStart phase relic>relic"}, keys)

    def test_reads_of_several_hooks_make_no_claim(self):
        self.probes[6] = ho.Probe(6, "Core/X.cs", 5, [("Bellows", "relic", "AfterPlayerTurnStart"),
                                                    ("Bellows", "relic", "BeforeHandDraw")])
        keys, _ = self._keys([-101, -3, 6, 3])
        self.assertEqual(set(), keys)

    # A screen in the turn start's setup pauses it; the game runs on to AfterSideTurnStart before the screen is asked
    # (the log shows the batch, then the harness answering the screen: -8), so the later setup listeners follow it.
    def test_after_side_turn_start_before_the_setup_is_out_of_order(self):
        keys, _ = self._keys([-101, -3, 5, 3, 4])
        self.assertEqual({"order turn_start BeforeHandDraw<AfterSideTurnStart phase relic>relic",
                          "order turn_start AfterPlayerTurnStart<AfterSideTurnStart phase relic>relic"}, keys)

    def test_screen_pause_runs_after_side_turn_start_first(self):
        # Toolbox asks (BeforeHandDraw): AfterSideTurnStart, the screen, then Bellows's AfterPlayerTurnStart
        keys, inv = self._keys([-101, -3, 3, 5, -8, 4])
        self.assertEqual(set(), keys)
        self.assertEqual([], inv)

    def test_pause_does_not_excuse_a_listener_that_belonged_before_it(self):
        # Bellows (AfterPlayerTurnStart) raised the screen; Toolbox's BeforeHandDraw came before it in the game
        keys, _ = self._keys([-101, -3, 4, 5, -8, 3])
        self.assertEqual({"order turn_start BeforeHandDraw<AfterSideTurnStart phase relic>relic",
                          "order turn_start BeforeHandDraw<AfterPlayerTurnStart phase relic>relic"}, keys)

    def test_listener_that_ran_before_the_screen_was_answered_is_not_after_it(self):
        # no screen between AfterSideTurnStart and the listener it ran ahead of: not a pause
        keys, _ = self._keys([-101, -3, 3, 5, 4, -8])
        self.assertEqual({"order turn_start AfterPlayerTurnStart<AfterSideTurnStart phase relic>relic"}, keys)

    def test_after_side_turn_start_ahead_of_the_listener_that_raised_the_screen_stays_out_of_order(self):
        # Candle ran before Toolbox (the screen's): out of order. Bellows came after the screen: in order.
        keys, _ = self._keys([-101, -3, 5, 3, -8, 4])
        self.assertEqual({"order turn_start BeforeHandDraw<AfterSideTurnStart phase relic>relic"}, keys)

    def test_screen_mark_cuts_no_segment(self):
        self.assertEqual([("turn_start", [(2, 3), (3, -8), (4, 4)])], ho.segments([-101, -3, 3, -8, 4])[-1:])


DECOMPILED = Path(os.environ.get("SPIRELINE_DECOMPILED", r"D:\spireline\referee\decompiled"))


@unittest.skipUnless((DECOMPILED / "MegaCrit.Sts2.Core.Hooks" / "Hook.cs").exists(), "no decompiled game here")
class RealTable(unittest.TestCase):
    def test_every_override_is_registered_or_listed(self):
        _clear_caches()
        engine = ht.Engine.load(ht.ENGINE)
        res = ht.check(str(DECOMPILED), engine, ht.load_json(ht.DISPATCH), ht.load_json(ht.UNMODELLED), ht.read_open())
        self.assertEqual([], res["problems"])
        self.assertGreater(res["counts"]["overrides"], 900)

    @unittest.skipUnless(os.environ.get("SPIRELINE_HOOK_E2E") == "1", "set SPIRELINE_HOOK_E2E=1 to build and run")
    def test_moved_dispatch_call_fails_the_order_gate(self):
        """The turn start's AfterPlayerTurnStart powers moved ahead of BeforeHandDraw: a new phase inversion."""
        _clear_caches()
        engine = ht.Engine.load(ht.ENGINE)
        rel = "Core/CombatEngine.cs"
        text = engine.files[rel]
        call = "            AfterPlayerTurnStartPowers(state, rng);\n"
        head = "            Effects.RelicEffects.ApplyBeforeHandDraw(state, state.Turn + 1, rng);\n"
        i = text.index(call)
        j = text.index(head)
        self.assertLess(j, i)
        moved = text[:j] + call + text[j:i] + text[i + len(call):]
        broken = ht.Engine.load(ht.ENGINE, dict(engine.files, **{rel: moved}))
        res = ht.check(str(DECOMPILED), broken, ht.load_json(ht.DISPATCH), ht.load_json(ht.UNMODELLED), ht.read_open())
        rows = [ht.Row(**r) for r in res["rows"]]
        with tempfile.TemporaryDirectory() as work:
            out = ht.order(str(DECOMPILED), broken, rows, Path(work), log=lambda *a, **k: None)
        keys = set(out["by_mutant"][-1]["keys"])
        open_order = {k for k in ht.read_open() if k.startswith("order ")}
        new = keys - open_order
        self.assertTrue(any("BeforeHandDraw<AfterPlayerTurnStart" in k and "turn_start" in k for k in new), new)




EXTRACTED = """
public static class X
{
    public static void A(State s)
    {
        if (Has(s, BuffId.Ritual)) { Ritual_AfterCardPlayed(s); }
        Helper(s);
        Plain(s);
    }

    internal static void Ritual_AfterCardPlayed(State s) { Gain(s, BuffId.Ritual); }
    private static void Helper(State s) { Gain(s, BuffId.Tender); }
    internal static void Plain(State s) { Gain(s, BuffId.Rage); }
}
"""


# The engine's extracted bodies, File.Function in ordinal order: the list tools/emudump.Tests/ExtractedBodyTests.cs pins too
# (there by reflection through ExtractedBody.Is), so a function the two rules would judge differently fails one of the suites.
EXTRACTED_FUNCTIONS = (
    "BuffSystem.DoubleDamage_ModifyDamageMultiplicative",
    "BuffSystem.Flutter_ModifyDamageMultiplicative",
    "BuffSystem.Shrink_ModifyDamageMultiplicative",
    "BuffSystem.Soar_ModifyDamageMultiplicative",
    "BuffSystem.Strength_ModifyDamageAdditive",
    "BuffSystem.Surrounded_ModifyDamageMultiplicative",
    "BuffSystem.Tainted_ModifyDamageAdditive",
    "BuffSystem.Tank_ModifyDamageMultiplicative",
    "BuffSystem.Tracking_ModifyDamageMultiplicative",
    "BuffSystem.Vigor_ModifyDamageAdditive",
    "BuffSystem.Vulnerable_ModifyDamageMultiplicative",
    "BuffSystem.Weak_ModifyDamageMultiplicative",
    "CardEffects.Accuracy_ModifyDamageAdditive",
    "CardEffects.BeatingRemnant_AfterDamageReceived",
    "CardEffects.Buffer_AfterModifyingHpLostAfterOsty",
    "CardEffects.Buffer_ModifyHpLostAfterOstyLate",
    "CardEffects.Calcify_ModifyDamageAdditive",
    "CardEffects.CurlUp_AfterDamageReceived",
    "CardEffects.Flutter_AfterDamageReceived",
    "CardEffects.Gigantification_ModifyDamageMultiplicative",
    "CardEffects.HardenedShell_AfterDamageReceived",
    "CardEffects.HardenedShell_ModifyHpLostBeforeOstyLate",
    "CardEffects.Lethality_ModifyDamageMultiplicative",
    "CardEffects.PersonalHive_AfterDamageReceived",
    "CardEffects.PhantomBlades_ModifyDamageAdditive",
    "CardEffects.Rupture_AfterDamageReceived",
    "CardEffects.SicEm_AfterDamageGiven",
    "CardEffects.Slippery_AfterModifyingHpLostAfterOsty",
    "CardEffects.Slippery_ModifyHpLostAfterOsty",
    "CardEffects.Slumber_AfterDamageReceived",
    "CombatEngine.Adaptable_ShouldStopCombatFromEnding",
    "CombatEngine.Constrict_AfterDeath",
    "CombatEngine.Dampen_AfterDeath",
    "CombatEngine.FairyInABottle_AfterPreventingDeath",
    "CombatEngine.GremlinHorn_AfterDeath",
    "CombatEngine.Heist_BeforeDeath",
    "CombatEngine.Hex_AfterDeath",
    "CombatEngine.Infested_AfterDeath",
    "CombatEngine.Infested_ShouldStopCombatFromEnding",
    "CombatEngine.PossessSpeed_AfterDeath",
    "CombatEngine.PossessStrength_AfterDeath",
    "CombatEngine.Queen_AfterDeath",
    "CombatEngine.Ravenous_AfterDeath",
    "CombatEngine.Ravenous_AfterDeathOf",
    "CombatEngine.SteamEruption_ShouldStopCombatFromEnding",
    "CombatEngine.Stock_ShouldStopCombatFromEnding",
    "CombatEngine.Surprise_AfterDeath",
    "CombatEngine.Surprise_ShouldStopCombatFromEnding",
    "CombatEngine.Swipe_BeforeDeath",
    "EnemyAI.Adaptable_ShouldCreatureBeRemovedFromCombatAfterDeath",
    "EnemyAI.Colossus_ModifyDamageMultiplicative",
    "EnemyAI.Illusion_ShouldCreatureBeRemovedFromCombatAfterDeath",
    "EnemyAI.Imbalanced_AfterDamageGiven",
    "EnemyAI.PainfulStabs_ShouldCreatureBeRemovedFromCombatAfterDeath",
    "EnemyAI.PaperCuts_AfterDamageGiven",
    "EnemyAI.Reattach_ShouldCreatureBeRemovedFromCombatAfterDeath",
    "EnemyAI.SteamEruption_ShouldCreatureBeRemovedFromCombatAfterDeath",
    "EnemyAI.TheGambit_AfterDamageReceived",
    "RelicEffects.BeatingRemnant_ModifyHpLostAfterOsty",
    "RelicEffects.CentennialPuzzle_AfterDamageReceived",
    "RelicEffects.DemonTongue_AfterDamageReceived",
    "RelicEffects.FakeStrikeDummy_ModifyDamageAdditive",
    "RelicEffects.LavaLamp_AfterDamageReceived",
    "RelicEffects.LizardTail_AfterPreventingDeath",
    "RelicEffects.MiniatureCannon_ModifyDamageAdditive",
    "RelicEffects.PenNib_ModifyDamageMultiplicative",
    "RelicEffects.RedSkull_AfterCurrentHpChanged",
    "RelicEffects.SelfFormingClay_AfterDamageReceived",
    "RelicEffects.StrikeDummy_ModifyDamageAdditive",
    "RelicEffects.TungstenRod_ModifyHpLostAfterOsty",
    "RelicEffects.VitruvianMinion_ModifyDamageMultiplicative",
)


class ExtractedBodies(unittest.TestCase):
    def test_extracted_internal_functions_are_callees_of_their_dispatch_method(self):
        """R5 (MIGRATION_F08.md): an inline body extracted into an internal <Model>_<Hook> function still belongs to the
        dispatch method that calls it, like a private helper; an internal helper with no such name does not."""
        engine = ht.Engine.load(Path(), {"Core/X.cs": EXTRACTED})
        self.assertEqual(ht.callees(engine, "X.A"), ("X.Ritual_AfterCardPlayed", "X.Helper"))

    def test_only_internal_static_functions_are_extracted_bodies(self):
        engine = ht.Engine.load(Path(), {"Core/X.cs": EXTRACTED})
        self.assertEqual(ht.extracted_functions(engine), ("X.Ritual_AfterCardPlayed",))

    def test_the_name_rule_is_an_upper_case_initial_then_the_first_underscore_then_an_upper_case_letter(self):
        """ExtractedBody.cs's rule (Is), decided at the first underscore: Foo_Bar_baz is an extracted name, Foo_bar_Baz and Foo__Bar are not."""
        names = ["Strength_ModifyDamageAdditive", "A_B", "Foo1_Bar", "Foo_Bar_baz",
                 "Foo_bar_Baz", "Foo__Bar", "_Foo_Bar", "foo_Bar", "Foo_", "Foo_1Bar", "FooBar"]
        text = "public static class X\n{\n" + "".join(f"    internal static void {n}(State s) {{ }}\n" for n in names) + "}\n"
        engine = ht.Engine.load(Path(), {"Core/X.cs": text})
        self.assertEqual({m.name for m in engine.methods["Core/X.cs"]}, set(names))
        self.assertEqual(ht.extracted_functions(engine),
                         ("X.A_B", "X.Foo1_Bar", "X.Foo_Bar_baz", "X.Strength_ModifyDamageAdditive"))

    def test_the_engines_extracted_functions_are_the_pinned_list(self):
        """The same list as tools/emudump.Tests/ExtractedBodyTests.cs pins by reflection."""
        self.assertEqual(ht.extracted_functions(ht.Engine.load(ht.ENGINE)), EXTRACTED_FUNCTIONS)


if __name__ == "__main__":
    unittest.main()
