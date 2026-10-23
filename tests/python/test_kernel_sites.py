"""tools/kernel/sites.py: the site inventory and its switch groups, each rule seen to catch a broken case.

Small made-up inventories (identifiers only) plus the checked-in one: two families sharing a site switch together, a
split site lets them part, a dispatcher's passes and a flow join their families, and check_groups flags a group list
where a site, a dispatcher or a flow straddles two groups or a family is in none.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.kernel import sites  # noqa: E402

FAMILIES = {"families": {f: {"name": f} for f in ("F01", "F02", "F03", "F04")},
            "hooks": {"A": "F01", "B": "F02", "C": "F03", "D": "F04", "D2": "F04"}}
HOOKS = {"dispatchers": {
    "A": {"passes": [{"hook": "A"}]}, "B": {"passes": [{"hook": "B"}]}, "C": {"passes": [{"hook": "C"}]},
    "D": {"passes": [{"hook": "D"}, {"hook": "D2"}]}}}


def inv(dispatch_hooks: dict, flows=(), splits=None) -> dict:
    return sites.build({"hooks": dispatch_hooks}, FAMILIES, HOOKS, None, flows=list(flows), splits=splits or {})


def names(i: dict) -> list[str]:
    return [g["name"] for g in i["groups"]]


class SitesTest(unittest.TestCase):
    def test_families_that_share_a_site_switch_together(self):
        i = inv({"A": {"methods": ["X.M"]}, "B": {"methods": ["X.M"]}, "C": {"methods": ["Y.N"]}})
        self.assertEqual(["F01+F02", "F03", "F04"], names(i))
        self.assertEqual({"X.M": ["F01", "F02"]}, i["bridges"])
        self.assertEqual([], sites.check_groups(i))

    def test_a_split_site_lets_its_families_part(self):
        i = inv({"A": {"methods": ["X.M"]}, "B": {"methods": ["X.M"]}},
                splits={"X.M": {"F01": "anchor a", "F02": "anchor b"}})
        self.assertEqual(["F01", "F02", "F03", "F04"], names(i))
        self.assertEqual([], sites.check_groups(i))

    def test_a_split_that_misses_a_family_is_a_problem(self):
        i = inv({"A": {"methods": ["X.M"]}, "B": {"methods": ["X.M"]}}, splits={"X.M": {"F01": "a"}})
        self.assertTrue(any("names no part" in p for p in i["problems"]))

    def test_a_flow_is_one_switch(self):
        i = inv({"A": {"methods": ["X.M"]}, "C": {"methods": ["Y.N"]}},
                flows=[{"families": ["F01", "F03"], "why": "w"}])
        self.assertEqual(["F01+F03", "F02", "F04"], names(i))

    def test_generic_entries_name_their_sites(self):
        self.assertEqual(["A.M", "A.N[r]"], sites.generic_sites("A.M + N[r]: why"))
        self.assertEqual([], sites.generic_sites("fold P/Q (power_aliases.json): why"))
        i = inv({"A": {"methods": [], "generic": {"m": "X.M: why"}}, "B": {"methods": ["X.M"]}})
        self.assertEqual(["F01+F02", "F03", "F04"], names(i))

    def test_broken_groups_are_caught(self):
        i = inv({"A": {"methods": ["X.M"]}, "B": {"methods": ["X.M"]}, "D": {"methods": ["Z.O"]}},
                flows=[{"families": ["F03", "F04"], "why": "w"}])
        self.assertEqual([], sites.check_groups(i))
        split = copy.deepcopy(i)
        split["groups"] = [{"families": ["F01"]}, {"families": ["F02"]}, {"families": ["F03", "F04"]}]
        self.assertTrue(any("site X.M" in p and "straddles" in p for p in sites.check_groups(split)))
        none = copy.deepcopy(i)
        none["groups"] = [{"families": ["F01", "F02"]}, {"families": ["F03"]}]
        self.assertTrue(any("dispatcher D" in p and "0 groups" in p for p in sites.check_groups(none)))
        self.assertTrue(any("flow" in p for p in sites.check_groups(none)))
        twice = copy.deepcopy(i)
        twice["groups"].append({"families": ["F01"]})
        self.assertTrue(any("family F01 is in 2 groups" in p for p in sites.check_groups(twice)))

    def test_checked_in_inventory_is_fresh_and_holds(self):
        fresh, cs = sites.fresh()
        self.assertEqual([], fresh["problems"])
        self.assertEqual(sites.dump(fresh), sites.OUT.read_text(encoding="utf-8"))
        self.assertEqual(cs, sites.CS.read_text(encoding="utf-8"))
        checked = json.loads(sites.OUT.read_text(encoding="utf-8"))
        self.assertEqual([], sites.check_groups(checked))
        # The turn-start flow is one switch whatever the bridges are.
        flow = {"F02", "F03", "F13"}
        self.assertEqual(1, sum(1 for g in checked["groups"] if flow & set(g["families"])))


if __name__ == "__main__":
    unittest.main()
