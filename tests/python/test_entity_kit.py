"""Entity audit kit: coverage scan and packet helpers on synthetic inputs (no game files)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fidelity import entity_coverage as cov
from tools.fidelity import entity_packets as pk


class EntityKit(unittest.TestCase):
    def test_snake(self):
        assert cov.snake("PaperPhrog") == "PAPER_PHROG"
        assert pk.snake("FeelNoPainPower") == "FEEL_NO_PAIN_POWER"


    def test_find_method_and_block_end(self):
        src = ["class A", "{", "  public void Foo()", "  {", "    if (x) { y(); }", "  }", "  public int Bar() => 1;", "}"]
        assert pk.find_method(src, "Foo") == (2, 5)
        assert pk.find_method(src, "Bar") == (6, 6)


    def test_template_detection(self):
        plain = ["public sealed class S : CardModel", "{", "public override async Task OnPlay(X x)", "{",
                 "await DamageCmd.Attack(6m).Execute(x);", "}", "}"]
        assert pk.is_template(plain, "Cards")
        hooked = plain[:-1] + ["public override async Task AfterCardExhausted(X x)", "{", "}", "}"]
        assert not pk.is_template(hooked, "Cards")


    def test_scan_file_presence_and_attribution(self):
        tmp_path = Path(tempfile.mkdtemp())
        steps = [{"action": None, "state": {"relics": ["BURNING_BLOOD"], "player": {"powers": [["BARRICADE_POWER", 1]]},
                                             "enemies": [{"name": "VANTOM", "powers": []}]}},
                 {"action": {"kind": "play", "card": "BASH", "played": True}, "state": {"relics": [], "player": {"powers": []}, "enemies": []}}]
        out = {"results": [{"seed": "S1", "fight": {"kind": "boss"}, "steps": steps}]}
        ver = [{"seed": "S1", "identical": False, "step": 1, "card": "BASH", "diff": [[".enemies", "VANTOM", "x"], [".hand", "BASH", "y"]]}]
        (tmp_path / "b.out.json").write_text(json.dumps(out))
        (tmp_path / "b.verdicts.json").write_text(json.dumps(ver))
        (kind, step, identical, text, card, first), = cov.scan_file(str(tmp_path / "b.out.json"))
        assert (kind, step, identical, card) == ("boss", 1, False, "BASH")
        assert first["relic:BURNING_BLOOD"] == 0 and first["card:BASH"] == 1 and first["enemy:VANTOM"] == 0
        assert "VANTOM" in text and "BASH" not in text  # pile fields are not attribution


if __name__ == "__main__":
    unittest.main()
