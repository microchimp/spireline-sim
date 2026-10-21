"""The codex walk treats the number of creatures created in a fight as a hidden fact.

Each creature a fight creates rolls its HP on Rng.Niche, and the run history lists only the joiners
it saw. The walk takes that count as a fact settled by the furthest-walk rule, and settles a fact it
left open again once the facts after it are settled (Gremlin gang: joiners against the Merc's theft).

Needs the user's codex data (data/spire-codex) and a build of EmuDump; skipped without them.
"""
import gzip
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DLL = Path(os.environ.get("EMUDUMP_DLL", ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"))
CODEX = Path(os.environ.get("CODEX_RUNS", "D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz"))
DECOMPILED = Path(os.environ.get("DECOMPILED", "D:/spireline/referee/decompiled"))
# input line -> what the walk must do: runs that stopped on a Niche-tainted deck before the count
# was a fact (3568, 19196: four creatures created; 637642, 613376: Gremlin gang with the Merc's theft)
LINES = {3568: "complete", 19196: "complete", 637642: "complete", 613376: "complete"}


@unittest.skipUnless(DLL.exists() and CODEX.exists() and DECOMPILED.exists(), "needs EmuDump build and codex data")
class JoinedFact(unittest.TestCase):
    def test_runs_that_need_the_count_complete_and_none_gets_worse(self) -> None:
        with gzip.open(CODEX, "rt", encoding="utf-8") as fh:
            recs = {i: l for i, l in enumerate(fh) if i in LINES}
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td) / "in.jsonl", Path(td) / "out.jsonl"
            src.write_text("".join(recs[i] for i in sorted(recs)), encoding="utf-8")
            subprocess.run(["dotnet", str(DLL), "codex", str(src), str(out), "--decompiled", str(DECOMPILED),
                            "--character", "IRONCLAD", "--ascension", "10"], check=True, capture_output=True)
            res = {json.loads(l)["seed"]: json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()}
        for i, want in LINES.items():
            seed = json.loads(recs[i])["seed"]
            self.assertEqual(res[seed]["status"], want, f"line {i}: {res[seed].get('stop')}")
        facts = {h["fact"] for r in res.values() for h in r.get("hidden_facts", [])}
        self.assertIn("joined", facts)


if __name__ == "__main__":
    unittest.main()
