"""tools/kernel: the hook descriptor's extractor, generator and `construct shape`, each seen to catch a broken case.

The descriptor cases break a copy of the checked-in tools/kernel/hooks.json one field at a time and diff it against
the intact one, both ways. The extractor cases use a few lines of made-up C# in the shapes it reads (no game
code). One test runs `construct shape` on the real decompile when it is there.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fidelity import hook_table as ht  # noqa: E402
from tools.kernel import construct, gen_hooks, hooks_extract as hx  # noqa: E402

ABSTRACT = """
namespace MegaCrit.Sts2.Core.Models;
public abstract class AbstractModel
{
	public virtual Task AfterThing(PlayerChoiceContext choiceContext, int n)
	{
		return Task.CompletedTask;
	}
	public virtual decimal ModifyThing(decimal amount)
	{
		return 0m;
	}
	public virtual bool ShouldThing()
	{
		return true;
	}
}
"""

HOOK = """
namespace MegaCrit.Sts2.Core.Hooks;
public static class Hook
{
	public static async Task AfterThing(ICombatState combatState, PlayerChoiceContext choiceContext, int n)
	{
		foreach (AbstractModel model in IterateCombatHookListeners(combatState))
		{
			choiceContext.PushModel(model);
			await model.AfterThing(choiceContext, n);
			choiceContext.PopModel(model);
		}
	}

	public static decimal ModifyThing(ICombatState combatState, decimal amount, out IEnumerable<AbstractModel> modifiers)
	{
		decimal num = amount;
		List<AbstractModel> list = new List<AbstractModel>();
		foreach (AbstractModel item in IterateCombatHookListeners(combatState))
		{
			decimal num2 = item.ModifyThing(num);
			num += num2;
			if (num2 != 0m)
			{
				list.Add(item);
			}
		}
		modifiers = list;
		return num;
	}

	public static bool ShouldThing(IRunState runState)
	{
		foreach (AbstractModel item in runState.IterateHookListeners(null))
		{
			if (!item.ShouldThing())
			{
				return false;
			}
		}
		return true;
	}

	private static IEnumerable<AbstractModel> IterateCombatHookListeners(ICombatState combatState)
	{
		return combatState.IterateHookListeners();
	}
}
"""


def fixture(hook_text: str) -> str:
    d = Path(tempfile.mkdtemp(prefix="kernel-shape-"))
    (d / ht.MODELS_NS).mkdir(parents=True)
    (d / ht.MODELS_NS / "AbstractModel.cs").write_text(textwrap.dedent(ABSTRACT), encoding="utf-8")
    (d / hx.HOOK_FILE).parent.mkdir(parents=True)
    (d / hx.HOOK_FILE).write_text(textwrap.dedent(hook_text), encoding="utf-8")
    return str(d)


class Extractor(unittest.TestCase):
    def test_classifies_the_fixture(self):
        got = hx.extract(fixture(HOOK))["dispatchers"]
        self.assertEqual(sorted(got), ["AfterThing", "ModifyThing", "ShouldThing"])
        self.assertEqual((got["AfterThing"]["iterator"], got["AfterThing"]["context"]), ("combat", "passed"))
        self.assertEqual(got["AfterThing"]["passes"], [{"hook": "AfterThing", "op": "event", "changed": "none"}])
        self.assertEqual(got["ModifyThing"]["passes"],
                         [{"hook": "ModifyThing", "op": "fold-add", "changed": "nonzero"}])
        self.assertEqual((got["ShouldThing"]["iterator"], got["ShouldThing"]["passes"][0]["op"]),
                         ("run", "predicate-all"))

    def test_unclassifiable_op_is_an_error_naming_the_dispatcher(self):
        bad = HOOK.replace("num += num2;", "num -= num2;")
        with self.assertRaises(hx.ExtractError) as cm:
            hx.extract(fixture(bad))
        self.assertIn("ModifyThing", str(cm.exception))

    def test_unexplained_early_return_is_an_error(self):
        bad = HOOK.replace("\t\tforeach (AbstractModel item in runState",
                           "\t\tif (runState == null)\n\t\t{\n\t\t\treturn true;\n\t\t}\n"
                           "\t\tforeach (AbstractModel item in runState")
        with self.assertRaises(hx.ExtractError) as cm:
            hx.extract(fixture(bad))
        self.assertIn("ShouldThing", str(cm.exception))

    def test_unknown_iterator_is_an_error(self):
        bad = HOOK.replace("runState.IterateHookListeners(null)", "runState.SomeOtherListeners()")
        with self.assertRaises(hx.ExtractError) as cm:
            hx.extract(fixture(bad))
        self.assertIn("ShouldThing", str(cm.exception))


def _desc() -> dict:
    return hx.load()


def _step(d, name, field, value):
    d["dispatchers"][name][field] = value


BREAKS = {
    "dispatcher removed": (lambda d: d["dispatchers"].pop("AfterActEntered"), "AfterActEntered", "only in the decompile"),
    "dispatcher added": (lambda d: d["dispatchers"].__setitem__("AfterMadeUp", copy.deepcopy(
        d["dispatchers"]["AfterActEntered"])), "AfterMadeUp", "only in the descriptor"),
    "iterator": (lambda d: _step(d, "AfterDeath", "iterator", "combat"), "AfterDeath", "iterator"),
    "pass removed": (lambda d: d["dispatchers"]["ModifyBlock"]["passes"].pop(), "ModifyBlock", "pass 1"),
    "op": (lambda d: d["dispatchers"]["ModifyBlock"]["passes"][0].__setitem__("op", "fold-set"), "ModifyBlock",
           "pass 0 op"),
    "changed": (lambda d: d["dispatchers"]["ModifyHandDraw"]["passes"][0].__setitem__("changed", "ne"),
                "ModifyHandDraw", "pass 0 changed"),
    "when": (lambda d: d["dispatchers"]["ModifyDamage"]["passes"][2].pop("when"), "ModifyDamage", "pass 2 when"),
    "context": (lambda d: _step(d, "AfterCardDrawn", "context", "passed-unpushed"), "AfterCardDrawn", "context"),
    "wait point": (lambda d: _step(d, "AfterTurnEnd", "wait_all", [1]), "AfterTurnEnd", "wait_all"),
    "pre step": (lambda d: _step(d, "ModifyBlock", "pre", [{"step": "enchant-add"}]), "ModifyBlock", "pre"),
    "post step": (lambda d: _step(d, "ModifyDamage", "post", []), "ModifyDamage", "post"),
    "partner": (lambda d: _step(d, "ShouldDie", "partner", []), "ShouldDie", "partner"),
}


class Descriptor(unittest.TestCase):
    def test_intact_descriptor_has_no_difference(self):
        self.assertEqual(construct.diff(_desc(), _desc()), [])

    def test_each_kind_of_difference_is_caught_both_ways(self):
        for case, (breaker, name, field) in BREAKS.items():
            with self.subTest(case):
                good, bad = _desc(), _desc()
                breaker(bad)
                forward = construct.diff(bad, good)
                self.assertTrue(any(p.startswith(f"{name}: ") and field in p for p in forward), (case, forward))
                backward = construct.diff(good, bad)
                self.assertTrue(any(p.startswith(f"{name}: ") for p in backward), (case, backward))

    def test_args_are_not_part_of_the_shape(self):
        bad = _desc()
        bad["dispatchers"]["AfterAttack"]["args"] = []
        self.assertEqual(construct.diff(bad, _desc()), [])

    def test_stable_formatting(self):
        text = (hx.DESCRIPTOR).read_text(encoding="utf-8")
        self.assertEqual(hx.dumps(json.loads(text)), text)


class Generator(unittest.TestCase):
    def test_deterministic(self):
        d = _desc()
        shuffled = copy.deepcopy(d)
        names = list(shuffled["dispatchers"])
        random.Random(7).shuffle(names)
        shuffled["dispatchers"] = {n: shuffled["dispatchers"][n] for n in names}
        self.assertEqual(gen_hooks.render(d), gen_hooks.render(d))
        self.assertEqual(gen_hooks.render(d), gen_hooks.render(shuffled))

    def test_checked_in_files_are_fresh(self):
        self.assertEqual(construct.generated(_desc()), [])

    def test_stale_or_missing_generated_file_is_caught(self):
        d = _desc()
        root = Path(tempfile.mkdtemp(prefix="kernel-gen-"))
        (root / "Core2").mkdir()
        gen_hooks.write(d, root)
        self.assertEqual(construct.generated(d, root), [])
        p = root / gen_hooks.HOOKS
        p.write_text(p.read_text(encoding="utf-8").replace("ch ??= new()", "ch = new()", 1), encoding="utf-8")
        self.assertEqual(construct.generated(d, root), [f"{gen_hooks.HOOKS}: stale (differs from a fresh generation "
                                                        "from hooks.json)"])
        (root / gen_hooks.HOOK_ID).unlink()
        self.assertIn(f"{gen_hooks.HOOK_ID}: missing", construct.generated(d, root))

    def test_descriptor_change_makes_the_generated_files_stale(self):
        d = _desc()
        d["dispatchers"]["ModifyHandDraw"]["passes"][0]["changed"] = "ne"
        self.assertIn(f"{gen_hooks.HOOKS}: stale (differs from a fresh generation from hooks.json)",
                      construct.generated(d))

    def test_unknown_kernel_type_is_an_error(self):
        d = _desc()
        d["dispatchers"]["AfterAttack"]["args"][0]["type"] = "AttackCommand"
        self.assertTrue(construct.generated(d)[0].startswith("generator: AfterAttack"))


@unittest.skipUnless(os.path.isfile(os.path.join(ht.DEFAULT_DECOMPILED, str(hx.HOOK_FILE))), "no local decompile")
class RealDecompile(unittest.TestCase):
    def test_construct_shape_passes(self):
        self.assertEqual(construct.main(["shape"]), 0)


if __name__ == "__main__":
    unittest.main()
