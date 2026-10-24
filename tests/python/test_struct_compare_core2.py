"""struct_compare for the core2 engine (docs/rebuild/COMPARER.md): core2 runs over the old engine's state, so it is judged with
old's rows; the kernel's own fields are a surplus of the census and never compared against the record."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from tools.fidelity import struct_compare as sc  # noqa: E402
from test_struct_compare import ROWS, fight, net, paths, snap  # noqa: E402


def with_private(g: dict, e: dict, kernel: dict) -> tuple[dict, dict]:
    for s in e["steps"]:
        s["game_view"]["private"] = {"kernel": kernel}
    return g, e


KERNEL = {"executor_paused": False, "queue": {"actions": [{"id": 0, "kind": "Hook", "state": "GatheringPlayerChoice"}]},
          "hook_actions": [{"id": 1, "screen": {"type": "Screen"}}]}


class Core2Scope(unittest.TestCase):
    def test_core2_with_no_kernel_fields_is_judged_as_old(self):
        e0 = snap()["enemies"][0]
        moved = snap(enemies=[{**e0, "move": "GOOP_MOVE"}])  # old has no move id: unmodelled for old and for core2 (its state is old's)
        for engine in ("old", "core2"):
            r = sc.compare_fight(*fight([moved], [snap()]), ROWS, engine=engine)
            self.assertIsNone(r["first"], engine)
            self.assertEqual(r["surplus"], [])
        self.assertEqual(paths(sc.compare_fight(*fight([moved], [snap()]), ROWS, engine="core3")), ["snap.enemies[0].move"])

    def test_the_kernels_private_block_is_a_surplus_never_compared(self):
        g, e = with_private(*fight([snap()], [snap()]), KERNEL)
        r = sc.compare_fight(g, e, ROWS, engine="core2")
        self.assertIsNone(r["first"])
        self.assertIn("private.kernel.queue.actions[*].state", r["surplus"])
        self.assertIn("private.kernel.hook_actions[*].screen.type", r["surplus"])
        self.assertFalse([p for p in r["census"] if p.startswith("private")])
        self.assertEqual(sc.judge(g, e, ROWS, "core2")["surplus"], r["surplus"])
        self.assertNotIn("surplus", sc.judge(*fight([snap()], [snap()]), ROWS, "core2"))

    def test_the_games_sequencing_fields_the_kernel_writes_are_surplus_not_compared(self):
        rec = net(last_action=7, last_hook=2)  # the live game's counters
        eng = net(last_action=1, last_hook=0)  # the kernel's own
        rs = snap()
        g, e = fight([rs], [rs], rec_nets=[rec], eng_nets=[eng])
        r = sc.compare_fight(g, e, ROWS, engine="core2")
        self.assertIsNone(r["first"])
        self.assertEqual(r["surplus"], ["net.last_action", "net.last_hook"])
        self.assertFalse({"net.last_action", "net.last_hook"} & set(r["census"]))
        self.assertEqual(r["allowed"].get("kernel-sequencing"), None)  # not a difference an allowed row excused: never compared
        # An engine that writes none leaves the record's side to the unmodelled row, as for old.
        g, e = fight([rs], [rs], rec_nets=[rec], eng_nets=[net()])
        r = sc.compare_fight(g, e, ROWS, engine="core2")
        self.assertIsNone(r["first"])
        self.assertEqual(r["surplus"], [])
        self.assertIn("net.last_action", r["unmodelled"])
        # The same paths on another engine are compared (the row is core2's).
        g, e = fight([rs], [rs], rec_nets=[rec], eng_nets=[eng])
        self.assertEqual(paths(sc.compare_fight(g, e, ROWS, engine="core3")), ["net.last_action", "net.last_hook"])

    def test_a_surplus_never_hides_a_real_difference_beside_it(self):
        rs = snap()
        bad = net()
        bad["players"][0]["energy"] = 1
        g, e = fight([rs], [rs], rec_nets=[net(last_action=7)], eng_nets=[{**bad, "last_action": 1}])
        r = sc.compare_fight(g, e, ROWS, engine="core2")
        self.assertEqual(paths(r), ["net.players[0].energy"])
        self.assertEqual(r["surplus"], ["net.last_action"])


class Migrated(unittest.TestCase):
    """core2 inherits an old row only while none of the row's families is migrated; a row with no families is always inherited."""

    def ids(self, migrated):
        return {r["id"] for r in sc.for_engine(ROWS, "core2", migrated)}

    def test_rows_that_could_hide_a_rule_list_their_families(self):
        by = {r["id"]: r.get("families") for r in ROWS}
        self.assertEqual({k: v for k, v in by.items() if v}, {"fight-lost": ["F10"], "after-fight-turn-state": ["F10"], "cost-outside-hand": ["F07"]})

    def test_a_migrated_family_drops_its_rows_and_only_those(self):
        every = self.ids([])
        self.assertEqual(self.ids(["F06", "F11"]), every)  # a family no row lists changes nothing
        self.assertEqual(every - self.ids(["F10"]), {"fight-lost", "after-fight-turn-state"})
        self.assertEqual(every - self.ids(["F07"]), {"cost-outside-hand"})
        self.assertEqual(every - self.ids(["F10", "F07", "F12"]), {"fight-lost", "after-fight-turn-state", "cost-outside-hand"})
        # The representation limits stay however many families are migrated; old itself never loses a row.
        limits = {r["id"] for r in ROWS if not r.get("families") and "old" in r["engines"]}
        self.assertTrue({"intent-vocabulary", "power-folds", "upgrade-count", "affliction-amount", "enemy-move-id", "saved-props-floor",
                         "relic-counters"} <= limits)
        every_family = [f"F{i:02d}" for i in range(1, 18)]
        self.assertLessEqual(limits, self.ids(every_family))
        self.assertEqual({r["id"] for r in sc.for_engine(ROWS, "old", every_family)}, {r["id"] for r in sc.for_engine(ROWS, "old", [])})
        self.assertEqual({r["id"] for r in sc.for_engine(ROWS, "core3", every_family)}, {"dead-enemies", "after-last-checksum", "late-steps"})

    def test_a_lost_fight_is_compared_in_full_once_its_family_is_migrated(self):
        rec = snap(in_combat=False, player={**snap()["player"], "hp": 0}, hand=[])
        eng = snap(in_combat=False, player={**snap()["player"], "hp": 0, "block": 9})
        self.assertIsNone(sc.compare_fight(*fight([rec], [eng]), ROWS, engine="core2", migrated=[])["first"])
        self.assertIsNone(sc.compare_fight(*fight([rec], [eng]), ROWS, engine="core2", migrated=["F06"])["first"])
        r = sc.compare_fight(*fight([rec], [eng]), ROWS, engine="core2", migrated=["F10"])
        self.assertIn("snap.player.block", paths(r))
        self.assertIn("snap.hand#len", paths(r))
        self.assertIsNone(sc.compare_fight(*fight([rec], [eng]), ROWS, engine="old", migrated=["F10"])["first"])  # old keeps its rows

    def test_a_pile_cost_is_compared_once_the_cost_family_is_migrated(self):
        eng = net()
        eng["players"][0]["piles"]["Draw"][0]["cost"] = 3
        rs = snap()
        g, e = fight([rs], [rs], rec_nets=[net()], eng_nets=[eng])
        self.assertIsNone(sc.compare_fight(g, e, ROWS, engine="core2", migrated=["F10"])["first"])  # F10 is not the costs
        r = sc.compare_fight(g, e, ROWS, engine="core2", migrated=["F07"])
        self.assertEqual(paths(r), ["net.players[0].piles.Draw[0].cost"])
        self.assertIn("net.players[*].piles.Draw[*].cost", r["census"])

    def test_the_default_is_the_migrated_families_file(self):
        self.assertEqual(sc.for_engine(ROWS, "core2"), sc.for_engine(ROWS, "core2", sc.load_migrated()))

    def test_a_bad_families_list_is_refused(self):
        for bad in ("F10", ["F1"], [10], [""]):
            row = {"id": "x", "kind": "ignore", "engines": ["old"], "families": bad, "patterns": ["a"], "reason": "r", "legacy": "y"}
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "a.json"
                f.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    sc.load_rows(f)


if __name__ == "__main__":
    unittest.main()
