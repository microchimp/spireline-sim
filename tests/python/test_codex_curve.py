"""tools/reference/codex_curve.py: the human reference curve from the Codex corpus, aggregates only (D13).

Constructed data only: a tiny gz corpus of five fake runs made here. No corpus file, no game, no native library.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.reference import codex_curve as cc  # noqa: E402

SEEDS = ("FAKESEED01", "FAKESEED02", "FAKESEED03", "FAKESEED04", "FAKESEED05")
HASHES = tuple(f"fakehash{i:08d}" for i in range(1, 6))


def point(kind: str, room: str, hp: int = 60, choices: bool = False, picked: bool = False, rest: str | None = None,
          removed: list | None = None, model: str = "ENCOUNTER.FAKE_FIGHT") -> dict:
    ps: dict = {"current_hp": hp, "max_hp": 80, "damage_taken": 0, "player_id": 1}
    if choices:
        ps["card_choices"] = [{"card": {"id": "CARD.A"}, "was_picked": picked}, {"card": {"id": "CARD.B"}, "was_picked": False}]
    if rest:
        ps["rest_site_choices"] = [rest]
    if removed:
        ps["cards_removed"] = removed
    return {"map_point_type": kind, "player_stats": [ps], "rooms": [{"room_type": room, "model_id": model, "turns_taken": 2}]}


def fake_run(n: int, *, character: str = "IRONCLAD", ascension: int = 10, win: bool = False, abandoned: bool = False, i: int = 0,
             killer: str = "ENCOUNTER.FAKE_KILLER", deck_extra: list | None = None, first_removed: list | None = None) -> dict:
    """A run of `n` floors: Act 1 holds floors 1-17, Act 2 18-33, Act 3 34-49; every act starts with its ancient and ends with its boss."""
    acts, f = [], 0
    for size in (17, 16, 16):
        cnt = min(size, n - f)
        if cnt <= 0:
            break
        pts = []
        for j in range(cnt):
            f += 1
            if j == 0:
                pts.append(point("ancient", "event", model="EVENT.FAKE_NEOW"))
            elif j == size - 1:
                pts.append(point("boss", "boss", hp=70, model="ENCOUNTER.FAKE_BOSS"))
            elif j % 5 == 4:
                pts.append(point("rest_site", "rest_site", rest="SMITH" if f % 2 else "HEAL"))
            else:
                pts.append(point("monster", "monster", choices=True, picked=(f % 2 == 0)))
        acts.append(pts)
    if first_removed and n >= 4:
        acts[0][3]["player_stats"][0]["cards_removed"] = first_removed
    if not win and acts:
        acts[-1][-1]["rooms"][0]["model_id"] = killer
    deck = [{"id": "CARD.STRIKE_IRONCLAD", "floor_added_to_deck": 1} for _ in range(10)] + (deck_extra or [])
    return {"acts": ["ACT.OVERGROWTH", "ACT.HIVE", "ACT.GLORY"], "ascension": ascension, "build_id": "v0.0.0-test", "game_mode": "standard",
            "killed_by_encounter": "NONE.NONE" if win else killer, "killed_by_event": "NONE.NONE", "map_point_history": acts,
            "modifiers": [], "platform_type": "steam", "run_time": 1000 + i,
            "players": [{"badges": [], "character": f"CHARACTER.{character}", "deck": deck, "id": 1, "relics": [], "potions": []}],
            "schema_version": 9, "seed": SEEDS[i], "start_time": 1700000000 + i, "was_abandoned": abandoned, "win": win,
            "run_hash": HASHES[i], "player_token": None}


def five_runs() -> list[dict]:
    return [fake_run(5, i=0, killer="ENCOUNTER.SLIMES"),                       # dies on floor 5, Act 1
            fake_run(18, i=1, killer="ENCOUNTER.BIG_BOSS",                     # beat the Act 1 boss, died on floor 18
                     deck_extra=[{"id": "CARD.X", "floor_added_to_deck": 2}],
                     first_removed=[{"id": "CARD.STRIKE_IRONCLAD", "floor_added_to_deck": 1}]),
            fake_run(49, i=2, win=True),                                       # a win
            fake_run(3, i=3, abandoned=True),                                  # abandoned: not analysed
            fake_run(10, i=4, character="SILENT")]                             # another character


def write_corpus(d: Path, runs: list[dict]) -> Path:
    p = d / "runs-v0.0.0-test.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    return p


class CodexCurve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = write_corpus(Path(cls.tmp.name), five_runs())
        cls.doc = cc.build(cls.path, "IRONCLAD", 10, min_cell=1, n_boot=200, generated="2026-01-01T00:00Z")
        cls.ref = cls.doc["reference"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_selection(self):
        ctx = cc.scan(cc.open_lines(self.path))[0]
        self.assertEqual(ctx["lines"], 5)
        self.assertEqual(ctx["by_character"], {"IRONCLAD": 4, "SILENT": 1})
        self.assertEqual(ctx["by_ascension"], {"10": 5})
        f = ctx["funnel"]
        self.assertEqual((f["character_ascension"], f["abandoned"], f["analysed"]), (4, 1, 3))
        self.assertEqual(self.ref["runs"], 3)
        self.assertEqual(self.doc["header"]["corpus"], "v0.0.0")
        self.assertEqual(cc.corpus_name(Path("runs-v0.107.1.jsonl.gz")), "v0.107.1")
        _, sums, aband = cc.scan(cc.open_lines(self.path), "IRONCLAD", 9)
        self.assertEqual((len(sums), len(aband)), (0, 0))  # no Ascension 9 run

    def test_survival_curve(self):
        reach = {int(k): v["share"] for k, v in self.ref["survival"]["reach"].items()}
        self.assertEqual(max(reach), 49)
        self.assertEqual((reach[1], reach[5]), (1.0, 1.0))
        self.assertEqual((reach[6], reach[18]), (round(2 / 3, 4), round(2 / 3, 4)))
        self.assertEqual((reach[19], reach[49]), (round(1 / 3, 4), round(1 / 3, 4)))
        ended = self.ref["survival"]["ended"]
        self.assertEqual(ended["by_act1_boss"]["k"], 1)       # only the floor 5 run ended by the Act 1 boss
        self.assertEqual(ended["by_act2_boss"]["k"], 2)
        bk = self.ref["boss_kill"]
        self.assertEqual((bk["act1"]["killed_of_all"]["k"], bk["act2"]["killed_of_all"]["k"], bk["win"]["k"]), (2, 1, 1))
        self.assertEqual(bk["act1"]["killed_of_reached"]["share"], 1.0)  # two reached the Act 1 boss floor, both killed it
        self.assertEqual(bk["floor_rule_vs_act_lists_disagree_runs"], 0)
        self.assertEqual(self.ref["floor"]["mean"], round((5 + 18 + 49) / 3, 3))
        ends = {t["cause"]: t["n"] for t in self.ref["end_causes"]["top10"]}
        self.assertEqual(ends, {"SLIMES": 1, "BIG_BOSS": 1})

    def test_deck_cards_and_rests(self):
        s = cc.summarize(five_runs()[1])  # 10 kept + 1 gained on floor 2; 1 more, held from the start, removed on floor 4
        self.assertEqual(s["deck"][1:5], [11, 12, 12, 11])
        self.assertEqual(s["n"], 18)
        self.assertGreater(sum(s["screens"]), 0)
        rests = self.ref["rests"]["all"]
        self.assertEqual(rests["smith"]["k"] + rests["heal"]["k"], rests["choices"])
        self.assertEqual(self.ref["boss_hp"]["1"]["hp"]["n"], 2)

    def test_suppression(self):
        self.assertEqual(cc.share(25, 100, 20), {"n": 100, "k": 25, "share": 0.25})
        for k, n, m in ((19, 100, 20), (5, 10, 20), (95, 100, 20), (1, 3, 2)):
            self.assertEqual(cc.share(k, n, m), {"n": None, "k": None, "share": None}, (k, n, m))
        self.assertEqual(cc.share(0, 100, 20)["share"], 0.0)
        # the default floor of 20 runs hides every cell of a five run corpus
        doc = cc.build(self.path, "IRONCLAD", 10, n_boot=50, generated="x")
        cells = list(doc["reference"]["survival"]["reach"].values())
        self.assertTrue(all(c["share"] is None for c in cells))
        self.assertIsNone(doc["reference"]["deck"]["by_floor"]["1"]["mean"])
        self.assertIsNone(doc["reference"]["boss_kill"]["win"]["share"])
        self.assertEqual(doc["reference"]["end_causes"]["top10"], [])
        self.assertEqual(doc["context"]["by_character"], {"cells_under_floor": 2})  # IRONCLAD and SILENT, each under 20, are not listed
        self.assertEqual(doc["context"]["funnel"]["analysed"], "<20")

    def test_no_seed_or_player_field(self):
        blob, md = json.dumps(self.doc), cc.render_md(self.doc)
        for needle in SEEDS + HASHES:
            self.assertNotIn(needle, blob + md)
        for key in ("seed", "run_hash", "player_token", "start_time", "run_time", "player_id", "players"):
            self.assertNotIn(f'"{key}"', blob)
        cc.check_clean(self.doc)
        with self.assertRaises(ValueError):
            cc.check_clean({"a": [{"seed": "X"}]})
        self.assertIn("Aggregates only", cc.render_md(self.doc))

    def test_output_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sub" / "ref"
            self.assertEqual(cc.main([str(self.path), "--out", str(out), "--min-cell", "1", "--boot", "50"]), 0)
            doc = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(doc["format"], cc.FORMAT)
            self.assertTrue(out.with_suffix(".md").read_text(encoding="utf-8").startswith("# Human reference"))


if __name__ == "__main__":
    unittest.main()
