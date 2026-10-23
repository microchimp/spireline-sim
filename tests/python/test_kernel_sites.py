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

    SRC = "class X {\n  void M() {\n    // a\n    DoA();\n    // b\n    DoB();\n  }\n}\n"
    DISPATCH = {"A": {"methods": ["X.M"]}, "B": {"methods": ["X.M"]}}

    @staticmethod
    def reader(files: dict):
        return lambda rel: files.get(rel)

    def split(self, rec=None, src=SRC, files=None, dispatch=None):
        """The inventory of two families sharing X.M with one SPLITS record, read against `files` (default: X.cs = src)."""
        rec = rec if rec is not None else self.split_rec(src)
        return sites.build({"hooks": dispatch or self.DISPATCH}, FAMILIES, HOOKS, None, flows=[],
                           splits={"X.M": rec}, read=self.reader(files if files is not None else {"X.cs": src}))

    @staticmethod
    def split_rec(src, parts=(("F01", "DoA();"), ("F02", "DoB();")), **kw):
        rec = {"kind": "split", "file": "X.cs", "sha256": sites.source_hash(src), "reason": "separate statements",
               "parts": [{"family": f, "anchor": a} for f, a in parts]}
        rec.update(kw)
        return rec

    def test_a_split_site_lets_its_families_part(self):
        i = inv({"A": {"methods": ["X.M"]}, "B": {"methods": ["X.M"]}},
                splits={"X.M": {"kind": "split", "reason": "r",
                                "parts": [{"family": "F01", "anchor": "a"}, {"family": "F02", "anchor": "b"}]}})
        self.assertEqual(["F01", "F02", "F03", "F04"], names(i))
        self.assertEqual([], sites.check_groups(i))
        self.assertEqual([], i["problems"])

    def test_a_split_that_misses_a_family_is_a_problem(self):
        i = inv({"A": {"methods": ["X.M"]}, "B": {"methods": ["X.M"]}},
                splits={"X.M": {"kind": "split", "reason": "r", "parts": [{"family": "F01", "anchor": "a"}]}})
        self.assertTrue(any("names no part" in p for p in i["problems"]))

    def test_a_split_is_read_against_the_source(self):
        i = self.split()
        self.assertEqual([], i["problems"])
        self.assertEqual(["F01", "F02", "F03", "F04"], names(i))
        self.assertEqual([], sites.check_groups(i))
        crlf = self.split(files={"X.cs": self.SRC.replace("\n", "\r\n")})
        self.assertEqual([], crlf["problems"], "line ends do not move the hash or the anchors")

    def test_a_split_with_a_broken_anchor_is_a_problem(self):
        gone = self.split(self.split_rec(self.SRC, parts=(("F01", "DoA();"), ("F02", "DoZ();"))))
        self.assertTrue(any("anchor of F02 is not in X.cs" in p for p in gone["problems"]))
        twice_src = self.SRC.replace("DoB();", "DoA();\n    DoB();")
        twice = self.split(self.split_rec(twice_src), src=twice_src)
        self.assertTrue(any("anchor of F01 is not unique" in p and "2 places" in p for p in twice["problems"]))
        order = self.split(self.split_rec(self.SRC, parts=(("F02", "DoB();"), ("F01", "DoA();"))))
        self.assertTrue(any("anchor of F01 comes before" in p for p in order["problems"]))
        blank = self.split(self.split_rec(self.SRC, parts=(("F01", "DoA();"), ("F02", ""))))
        self.assertTrue(any("no anchor for F02" in p for p in blank["problems"]))

    def test_a_split_goes_stale_when_its_file_changes(self):
        rec = self.split_rec(self.SRC)
        edited = self.SRC.replace("DoB();", "DoB(); // later edit")
        stale = self.split(rec, files={"X.cs": edited})
        self.assertTrue(any("changed since it was read" in p for p in stale["problems"]))
        self.assertFalse(any("anchor" in p for p in stale["problems"]), "the anchors alone still hold")
        missing = self.split(rec, files={})
        self.assertTrue(any("cannot be read" in p for p in missing["problems"]))

    def test_a_split_record_must_be_whole(self):
        for rec, text in ((self.split_rec(self.SRC, reason=""), "has no reason"),
                          (self.split_rec(self.SRC, parts=(("F01", "DoA();"),)), "at least two parts"),
                          (self.split_rec(self.SRC, parts=(("F01", "DoA();"), ("F01", "DoB();"))), "named twice"),
                          (self.split_rec(self.SRC, file="Y.cs"), "cannot be read"),
                          (self.split_rec(self.SRC, kind="halve"), "unknown record kind")):
            self.assertTrue(any(text in p for p in self.split(rec)["problems"]), text)

    def test_a_false_listing_takes_the_hook_off_the_site(self):
        rec = {"kind": "false_listing", "hooks": ["B"], "reason": "X.M only reads the hook's result, it never runs it"}
        i = self.split(rec)
        self.assertEqual([], i["problems"])
        self.assertEqual(["F01"], i["sites"]["X.M"]["families"])
        self.assertEqual({}, i["bridges"])
        self.assertEqual(["F01", "F02", "F03", "F04"], names(i))
        self.assertEqual([], sites.check_groups(i))
        only = self.split(rec, dispatch={"B": {"methods": ["X.M"]}})
        self.assertNotIn("X.M", only["sites"], "a site with no hook left leaves the inventory")
        self.assertEqual([], only["problems"])

    def test_a_false_listing_that_is_not_listed_is_a_problem(self):
        unlisted = self.split({"kind": "false_listing", "hooks": ["C"], "reason": "r"})
        self.assertTrue(any("does not list X.M for C" in p for p in unlisted["problems"]))
        reasonless = self.split({"kind": "false_listing", "hooks": ["B"], "reason": " "})
        self.assertTrue(any("has no reason" in p for p in reasonless["problems"]))
        nohook = self.split({"kind": "false_listing", "hooks": [], "reason": "r"})
        self.assertTrue(any("names no hook" in p for p in nohook["problems"]))
        nosite = sites.build({"hooks": {"A": {"methods": ["X.M"]}}}, FAMILIES, HOOKS, None, flows=[],
                             splits={"Y.N": {"kind": "false_listing", "hooks": ["A"], "reason": "r"}})
        self.assertTrue(any("not a site of the inventory" in p for p in nosite["problems"]))

    def by_caller(self, rec=None, src=SRC, known=None):
        callers = [{"family": "F01", "caller": "X.CallA", "anchor": "DoA();"},
                   {"family": "F02", "caller": "X.CallB", "anchor": "DoB();"}]
        rec = rec if rec is not None else {"kind": "by_caller", "file": "X.cs", "sha256": sites.source_hash(src),
                                           "reason": "one helper, one family per caller", "callers": callers}
        i = sites.build({"hooks": self.DISPATCH}, FAMILIES, HOOKS, None, flows=[], splits={"X.M": rec},
                        read=self.reader({"X.cs": src}))
        return i, rec

    def test_a_site_split_by_caller_parts_its_families(self):
        i, rec = self.by_caller()
        self.assertEqual([], i["problems"])
        self.assertEqual(["F01", "F02", "F03", "F04"], names(i))
        self.assertEqual([], sites.check_groups(i))
        self.assertEqual([], sites.check_record("X.M", rec, {"A", "B"}, None, None, {"X.CallA", "X.CallB"}))

    def test_a_split_by_caller_that_is_broken_is_a_problem(self):
        i, rec = self.by_caller()
        gone = copy.deepcopy(rec)
        gone["callers"][1]["anchor"] = "DoZ();"
        self.assertTrue(any("anchor of F02 is not in X.cs" in p for p in self.by_caller(gone)[0]["problems"]))
        short = copy.deepcopy(rec)
        short["callers"].pop()
        self.assertTrue(any("names no part for ['F02']" in p for p in self.by_caller(short)[0]["problems"]))
        problems = sites.check_record("X.M", rec, {"A", "B"}, None, None, {"X.CallA"})
        self.assertEqual(["STALE caller X.CallB of X.M: not a method of the engine"], problems)
        stale = self.by_caller(src=self.SRC + "// edit\n", rec=rec)[0]
        self.assertTrue(any("changed since it was read" in p for p in stale["problems"]))
        unnamed = copy.deepcopy(rec)
        unnamed["callers"][0]["caller"] = ""
        self.assertTrue(any("is not named" in p for p in self.by_caller(unnamed)[0]["problems"]))

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
