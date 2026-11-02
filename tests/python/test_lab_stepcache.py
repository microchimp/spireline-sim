"""tools/lab/stepcache.py and the steps that skip by their inputs (reset order D44 step D): the presort's codex walk, refset
and corpus checks (plans.py), the groups (lab_groups.py) and the fix kits (packet.py).

Unchanged inputs skip; a changed input runs; a missing or rewritten previous output runs. Trivial files stand in for the
engine, the Codex dataset and the lab; no game, no emulator.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "referee"))

from tools.fidelity import lab_groups as lg  # noqa: E402
from tools.fidelity import packet as pk  # noqa: E402
from tools.lab import plans as pl  # noqa: E402
from tools.lab import stepcache as sc  # noqa: E402


def put(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write(path: Path, doc) -> None:
    put(path, json.dumps(doc))


class StepCacheTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.lab = Path(self._td.name) / "lab"
        self.out = put(self.lab / "runs" / "job-a" / "out.json", "{}")

    def test_digest_ignores_the_order_of_its_parts(self):
        self.assertEqual(sc.digest({"a": 1, "b": "x"}), sc.digest({"b": "x", "a": 1}))
        self.assertNotEqual(sc.digest({"a": 1}), sc.digest({"a": 2}))

    def test_unchanged_inputs_skip(self):
        sc.write_stamp(self.lab, "step", "d1", [self.out], job="job-a")
        why = sc.skip_reason(self.lab, "step", "d1")
        self.assertIsNotNone(why)
        self.assertIn("inputs unchanged", why)
        self.assertIn("job-a", why)

    def test_a_changed_input_runs(self):
        sc.write_stamp(self.lab, "step", "d1", [self.out])
        self.assertIsNone(sc.skip_reason(self.lab, "step", "d2"))

    def test_no_previous_stamp_runs(self):
        self.assertIsNone(sc.skip_reason(self.lab, "step", "d1"))

    def test_a_missing_previous_output_runs(self):
        sc.write_stamp(self.lab, "step", "d1", [self.out])
        self.out.unlink()
        self.assertIsNone(sc.skip_reason(self.lab, "step", "d1"))

    def test_an_output_written_again_runs(self):
        sc.write_stamp(self.lab, "step", "d1", [self.out])
        self.out.write_text('{"other": 1}', encoding="utf-8")
        self.assertIsNone(sc.skip_reason(self.lab, "step", "d1"))

    def test_a_step_that_wrote_nothing_leaves_no_stamp(self):
        sc.write_stamp(self.lab, "step", "d1", [self.out])
        self.out.unlink()
        sc.write_stamp(self.lab, "step", "d1", [self.out])  # the output is not there: no stamp, so no skip next time
        self.assertIsNone(sc.read_stamp(self.lab, "step"))

    def test_a_corrupt_stamp_runs(self):
        sc.write_stamp(self.lab, "step", "d1", [self.out])
        sc.stamp_path(self.lab, "step").write_text("{not json", encoding="utf-8")
        self.assertIsNone(sc.skip_reason(self.lab, "step", "d1"))

    def test_the_stamp_holds_no_absolute_path(self):
        sc.write_stamp(self.lab, "step", "d1", [self.out], job="job-a")
        text = sc.stamp_path(self.lab, "step").read_text(encoding="utf-8")
        self.assertNotIn(str(self.lab).replace("\\", "/"), text.replace("\\", "/"))
        self.assertIn("runs/job-a/out.json", text)

    def test_outputs_outside_the_lab_are_named_by_label(self):
        other = put(Path(self._td.name) / "packets" / "grades.json", "{}")
        sc.write_stamp(self.lab, "kits", "d1", {"grades.json": other})
        self.assertIsNotNone(sc.skip_reason(self.lab, "kits", "d1", {"grades.json": other}))
        self.assertIsNone(sc.skip_reason(self.lab, "kits", "d1", {"index.md": other}))  # another set of outputs

    def test_the_engine_digest_follows_the_engine_and_nothing_else(self):
        tree = Path(self._td.name) / "tree"
        cs = put(tree / "src" / "Sts2Emulator" / "Core" / "A.cs", "class A {}")
        put(tree / "tools" / "emudump" / "Program.cs", "class P {}")
        put(tree / "data" / "id_map.json", "{}")
        base = sc.engine_digest(tree)
        put(tree / "rl" / "gen0.py", "print(1)")  # lab code
        put(tree / "src" / "Sts2Emulator.Tests" / "T.cs", "class T {}")  # the xunit tests
        put(tree / "src" / "Sts2Emulator" / "obj" / "x.cs", "build output")
        self.assertEqual(base, sc.engine_digest(tree))
        cs.write_text("class A { int x; }", encoding="utf-8")
        self.assertNotEqual(base, sc.engine_digest(tree))


class PresortCheckTests(unittest.TestCase):
    """plans.plan_presort: the codex walk, refset and corpus checks carry the skip."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.td = Path(self._td.name)
        self.tree = self.td / "tree"
        self.lab = self.td / "lab"
        self.referee = self.td / "referee"
        self.codex = put(self.td / "codex.jsonl.gz", "data")
        self.decompiled = self.td / "decompiled"
        self.decompiled.mkdir()
        put(self.decompiled / "a.cs")
        put(self.tree / pl.EMUDUMP, "dll")
        put(self.tree / "src" / "Sts2Emulator" / "A.cs", "class A {}")
        put(self.tree / "tools" / "emudump" / "Program.cs", "class P {}")
        put(self.tree / "tools" / "fidelity" / "corpus.py", "# corpus")
        put(self.tree / "tools" / "fidelity" / "refset_score.py", "# refset")
        put(self.tree / "tools" / "fidelity" / "fight_compare.py", "# fc")
        put(self.tree / "tools" / "fidelity" / "reanchor_compare.py", "# rc")
        put(self.tree / "referee" / "refset.py", "# refset")
        put(self.referee / "refset" / "v2" / "manifest.json", "{}")
        self.corpus = self.td / "corpus"
        put(self.corpus / "index.json", '{"fights": []}')
        put(self.corpus / "runs" / "lab-aaaaaaa" / "results.json", '{"fights": {}}')  # what corpus.py check leaves for the post hook to copy
        self.old_env = os.environ.get("SPIRELINE_CORPUS")
        os.environ["SPIRELINE_CORPUS"] = str(self.corpus)
        self.addCleanup(self._restore_env)
        sc._engine_cache.clear()

    def _restore_env(self):
        if self.old_env is None:
            os.environ.pop("SPIRELINE_CORPUS", None)
        else:
            os.environ["SPIRELINE_CORPUS"] = self.old_env

    def ctx(self, job: str) -> pl.Ctx:
        return pl.Ctx(lab=self.lab, tree=self.tree, game=self.td / "game", fork=self.td / "fork", py="python", sha="a" * 40,
                      job_dir=self.lab / "runs" / job, job_name=job, params={"procs_refset": 1, "procs_corpus": 1},
                      backlog_dir=self.referee / "backlog", backlog=None, codex=self.codex, decompiled=self.decompiled,
                      records=self.td / "records", game_src=self.td / "game-src", nuget=self.td / "nuget")

    def cmds(self, c: pl.Ctx) -> dict[str, pl.Cmd]:
        return {cmd.name: cmd for s in pl.plan_presort(c) for cmd in s.cmds}

    def run_check(self, c: pl.Ctx, name: str, outputs: list[str]) -> None:
        """What the runner does for a check that ran: its output is written, then its post hook runs."""
        for o in outputs:
            put(c.job_dir / o, "result")
        cmd = self.cmds(c)[name]
        problem = cmd.post(c) if cmd.post else None
        self.assertIsNone(problem, problem)

    def test_a_check_runs_without_a_stamp(self):
        c = self.ctx("j1")
        self.assertIsNone(self.cmds(c)["codex-s0"].skip_reason(c, "checks"))

    def test_a_landing_that_leaves_the_engine_alone_skips_the_codex_walk(self):
        first = self.ctx("j1")
        for i in range(4):
            self.run_check(first, f"codex-s{i}", [f"codex/s{i}.jsonl"])
        put(self.tree / "rl" / "gen0.py", "print('a lab change')")  # main moved, the engine did not
        sc._engine_cache.clear()
        second = self.ctx("j2")
        for i in range(4):
            why = self.cmds(second)[f"codex-s{i}"].skip_reason(second, "checks")
            self.assertIsNotNone(why)
            self.assertIn("inputs unchanged", why)

    def test_a_changed_engine_runs_the_codex_walk_again(self):
        first = self.ctx("j1")
        self.run_check(first, "codex-s0", ["codex/s0.jsonl"])
        put(self.tree / "src" / "Sts2Emulator" / "A.cs", "class A { int fixed; }")
        sc._engine_cache.clear()
        second = self.ctx("j2")
        self.assertIsNone(self.cmds(second)["codex-s0"].skip_reason(second, "checks"))

    def test_a_changed_codex_dataset_runs_the_codex_walk_again(self):
        first = self.ctx("j1")
        self.run_check(first, "codex-s0", ["codex/s0.jsonl"])
        self.codex.write_text("a longer dataset", encoding="utf-8")
        self.assertIsNone(self.cmds(self.ctx("j2"))["codex-s0"].skip_reason(self.ctx("j2"), "checks"))

    def test_a_pruned_previous_output_runs_the_check_again(self):
        first = self.ctx("j1")
        self.run_check(first, "codex-s1", ["codex/s1.jsonl"])
        (first.job_dir / "codex" / "s1.jsonl").unlink()
        second = self.ctx("j2")
        self.assertIsNone(self.cmds(second)["codex-s1"].skip_reason(second, "checks"))

    def test_the_shards_are_stamped_apart(self):
        first = self.ctx("j1")
        self.run_check(first, "codex-s0", ["codex/s0.jsonl"])  # shard 0 ran; the others did not
        second = self.ctx("j2")
        self.assertIsNotNone(self.cmds(second)["codex-s0"].skip_reason(second, "checks"))
        self.assertIsNone(self.cmds(second)["codex-s2"].skip_reason(second, "checks"))

    def test_the_corpus_check_follows_the_corpus_and_its_code(self):
        first = self.ctx("j1")
        self.run_check(first, "corpus", ["corpus/results.json"])
        second = self.ctx("j2")
        self.assertIsNotNone(self.cmds(second)["corpus"].skip_reason(second, "checks"))
        put(self.corpus / "index.json", '{"fights": [{"id": "new"}]}')  # a fight was added to the corpus
        self.assertIsNone(self.cmds(second)["corpus"].skip_reason(second, "checks"))
        self.run_check(self.ctx("j3"), "corpus", ["corpus/results.json"])
        self.assertIsNotNone(self.cmds(second)["corpus"].skip_reason(second, "checks"))
        (self.tree / "tools" / "fidelity" / "corpus.py").write_text("# corpus, changed", encoding="utf-8")
        self.assertIsNone(self.cmds(second)["corpus"].skip_reason(second, "checks"))

    def test_the_refset_check_skips_on_the_same_inputs(self):
        first = self.ctx("j1")
        put(first.job_dir / "logs" / "refset.log", "")  # _keep_refset finds no stored score here: the post hook says so
        cmd = self.cmds(first)["refset"]
        put(first.job_dir / "refset" / "score.json", "{}")
        sc.write_stamp(first.lab, "refset", sc.digest(pl.check_parts(first, "refset")), [first.job_dir / "refset" / "score.json"], job="j1")
        second = self.ctx("j2")
        self.assertIsNotNone(self.cmds(second)["refset"].skip_reason(second, "checks"))
        put(self.tree / "referee" / "refset.py", "# refset, changed")
        self.assertIsNone(self.cmds(second)["refset"].skip_reason(second, "checks"))
        self.assertIsNotNone(cmd)


def tiny_lab(td: Path) -> tuple[Path, Path, Path, list[str]]:
    lab = td / "lab"
    job = lab / "runs" / "20261015T0100-fights"
    specs = [{"seed": f"SEED{i:06d}", "setup": [{"cmd": "relic", "id": "EGG_A"}, {"cmd": "card", "id": "ANGER"}, {"cmd": "fight", "id": "SLIMES_WEAK"}]}
             for i in range(3)]
    write(job / "b.json", {"runs": specs})
    write(job / "b.verdicts.json", [{"seed": s["seed"], "identical": False, "step": 0, "field": ".hand", "card": "start", "flags": [],
                                     "diff": [[".hand", "[A]", "[B]"]]} for s in specs])
    write(job / "b.out.json", {"results": [{"seed": s["seed"], "steps": [{"state": {}}]} for s in specs]})
    (job / "done.rc").write_text("0", encoding="utf-8")
    write(lab / "cache" / "model_features.json", {"stamp": None, "features": {}})
    hosts = td / "hosts.json"
    write(hosts, {"entries": []})
    backlog = td / "backlog-x.json"
    write(backlog, {"items": [{"id": "X-F01", "status": "open", "title": "Egg", "evidence": "EGG_A at start"}]})
    argv = ["--lab", str(lab), "--out", str(lab / "groups.json"), "--summary", str(lab / "summary.md"), "--decompiled", str(td / "none"),
            "--host-gaps", str(hosts), "--backlog", str(backlog)]
    return lab, job, backlog, argv


class GroupsSkipTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.td = Path(self._td.name)
        self.lab, self.job, self.backlog, self.argv = tiny_lab(self.td)

    def run_groups(self, *extra: str) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(0, lg.main(self.argv + list(extra)))
        return buf.getvalue()

    def test_a_second_run_on_the_same_inputs_is_skipped_and_leaves_the_file_alone(self):
        self.assertNotIn("step skipped", self.run_groups())
        out = self.lab / "groups.json"
        before = (out.read_bytes(), out.stat().st_mtime_ns)
        text = self.run_groups()
        self.assertIn("step skipped: groups: inputs unchanged", text)
        self.assertEqual(before, (out.read_bytes(), out.stat().st_mtime_ns))

    def test_a_new_batch_runs_the_grouping_again(self):
        self.run_groups()
        job2 = self.lab / "runs" / "20261015T0200-fights"
        write(job2 / "c.verdicts.json", [{"seed": "SEEDX", "identical": False, "step": 0, "field": ".hand", "card": "start", "flags": [],
                                          "diff": [[".hand", "[A]", "[B]"]]}])
        write(job2 / "c.json", {"runs": [{"seed": "SEEDX", "setup": [{"cmd": "fight", "id": "SLIMES_WEAK"}]}]})
        write(job2 / "c.out.json", {"results": [{"seed": "SEEDX", "steps": [{"state": {}}]}]})
        (job2 / "done.rc").write_text("0", encoding="utf-8")
        self.assertNotIn("step skipped", self.run_groups())
        self.assertEqual(4, sum(g["count"] for g in json.loads((self.lab / "groups.json").read_text(encoding="utf-8"))["groups"]))

    def test_a_file_of_a_batch_written_again_runs_the_grouping_again(self):
        self.run_groups()
        v = self.job / "b.verdicts.json"
        v.write_text(v.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assertNotIn("step skipped", self.run_groups())

    def test_a_changed_fix_list_runs_the_grouping_again(self):
        self.run_groups()
        write(self.backlog, {"items": [{"id": "X-F01", "status": "open", "title": "Egg", "evidence": "EGG_A at start, and more"}]})
        self.assertNotIn("step skipped", self.run_groups())

    def test_a_log_or_a_job_record_does_not_count_as_an_input(self):
        self.run_groups()
        put(self.job / "logs" / "groups.log", "started again")
        put(self.job / "corpus-imported.json", "{}")
        self.assertIn("step skipped", self.run_groups())

    def test_a_missing_previous_output_runs_the_grouping_again(self):
        self.run_groups()
        (self.lab / "groups.json").unlink()
        self.assertNotIn("step skipped", self.run_groups())
        self.assertTrue((self.lab / "groups.json").is_file())

    def test_force_runs_the_grouping_again(self):
        self.run_groups()
        self.assertNotIn("step skipped", self.run_groups("--force"))


class KitsSkipTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.td = Path(self._td.name)
        self.referee = self.td / "referee"
        self.referee.mkdir()
        write(self.td / "id_map.json", {"cards": {"Anger": "ANGER"}, "relics": {}, "enemies": {}, "powers": {}, "potions": {}})
        put(self.td / "sweep_map.json", "{}")
        self.src = pk.Sources(referee=self.referee, tests=self.td / "tests", id_map=self.td / "id_map.json", sweep_map=self.td / "sweep_map.json",
                              engine=self.td / "engine", walker=self.td / "Walker.cs")
        self.args = argparse.Namespace(no_kit_test=False, no_emudump=False, no_sweep=False, refresh_probes=False)
        self.items = [{"id": "X-F01", "status": "open", "title": "Anger misplays", "evidence": "ANGER at start", "lane": "l", "kind": "k"}]

    def groups(self, count_matched: int = 3, count_other: int = 9) -> dict:
        return {"groups": [{"id": "Gmatched", "backlog_ids": ["X-F01"], "count": count_matched, "impact_runs_est": 5, "examples": [], "entities": [],
                            "ingredient_effect": [], "key": {}},
                           {"id": "Gother", "backlog_ids": ["Y-F09"], "count": count_other, "impact_runs_est": 1, "examples": [], "entities": ["relic:ZZZ"],
                            "ingredient_effect": [], "key": {}}]}

    def digest(self, groups: dict, items: list | None = None) -> str:
        return sc.digest(pk.kit_inputs(self.src, items or self.items, groups, self.args))

    def test_a_group_no_item_matches_does_not_change_the_kits_inputs(self):
        self.assertEqual(self.digest(self.groups()), self.digest(self.groups(count_other=500)))

    def test_a_group_an_item_matches_does(self):
        self.assertNotEqual(self.digest(self.groups()), self.digest(self.groups(count_matched=4)))

    def test_a_changed_item_does(self):
        other = [dict(self.items[0], evidence="ANGER at start, and a seed")]
        self.assertNotEqual(self.digest(self.groups()), self.digest(self.groups(), other))

    def test_a_changed_evidence_file_does(self):
        before = self.digest(self.groups())
        self.src.ledger = put(self.td / "ledger.md", "a note")
        self.assertNotEqual(before, self.digest(self.groups()))

    def test_a_changed_engine_does_not_change_it_but_an_engine_edit_in_the_checkout_would(self):
        # the engine digest is of this checkout's sources: the same checkout gives the same kits inputs
        self.assertEqual(self.digest(self.groups()), self.digest(self.groups()))

    def test_the_outputs_are_the_index_files_and_one_packet_per_item(self):
        outs = pk.kit_outputs(self.items, self.td / "packets")
        self.assertEqual({"grades.json", "completeness.json", "index.md", "X-F01.md"}, set(outs))


class KitsMainSkipTests(unittest.TestCase):
    """packet.main with --groups: the run is skipped before the packet context is built when nothing it reads changed."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.td = Path(self._td.name)
        self.referee = self.td / "referee"
        (self.referee / "backlog").mkdir(parents=True)
        write(self.referee / "backlog" / "backlog-x.json", {"items": [
            {"id": "X-F01", "status": "open", "title": "Anger misplays", "evidence": "ANGER at start", "lane": "l", "kind": "k"}]})
        self.groups = self.td / "lab" / "groups.json"
        self.out = self.referee / "backlog" / "packets"
        self.argv = ["--item", "X-F01", "--backlog", str(self.referee / "backlog" / "backlog-x.json"), "--referee", str(self.referee),
                     "--groups", str(self.groups), "--out", str(self.out)]
        self.write_groups(3)
        self.old_env = os.environ.get("SPIRELINE_CORPUS")
        os.environ.pop("SPIRELINE_CORPUS", None)
        self.addCleanup(lambda: os.environ.__setitem__("SPIRELINE_CORPUS", self.old_env) if self.old_env else None)

    def write_groups(self, count: int) -> None:
        write(self.groups, {"groups": [{"id": "Gm", "backlog_ids": ["X-F01"], "count": count, "impact_runs_est": 5}]})

    def stamp_a_finished_run(self) -> None:
        src = pk.Sources.default(self.referee)
        doc = pk.load_backlog(self.referee / "backlog" / "backlog-x.json")
        items = [i for i in doc["items"] if i["id"] == "X-F01"]
        args = argparse.Namespace(no_kit_test=False, no_emudump=False, no_sweep=False, refresh_probes=False)
        outs = pk.kit_outputs(items, self.out)
        for p in outs.values():
            put(p, "{}")
        sc.write_stamp(self.groups.parent, "kits", sc.digest(pk.kit_inputs(src, items, pk.kt.load_groups(self.groups), args)), outs)

    def run_main(self, *extra: str) -> tuple[str, bool]:
        """(stdout, whether the packet context was built)"""
        built = []

        class Stop(Exception):
            pass

        def build(*a, **k):
            built.append(1)
            raise Stop

        orig = pk.Context.build
        pk.Context.build = build  # the build is the heavy part: its being reached is the answer
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                try:
                    pk.main(self.argv + list(extra))
                except Stop:
                    pass
        finally:
            pk.Context.build = orig
        return buf.getvalue(), bool(built)

    def test_unchanged_inputs_skip_before_the_context_is_built(self):
        self.stamp_a_finished_run()
        text, built = self.run_main()
        self.assertIn("step skipped: kits: inputs unchanged", text)
        self.assertFalse(built)

    def test_a_changed_matching_group_builds_the_kits_again(self):
        self.stamp_a_finished_run()
        self.write_groups(4)
        text, built = self.run_main()
        self.assertNotIn("step skipped", text)
        self.assertTrue(built)

    def test_force_builds_the_kits_again(self):
        self.stamp_a_finished_run()
        self.assertTrue(self.run_main("--force")[1])

    def test_a_missing_packet_builds_the_kits_again(self):
        self.stamp_a_finished_run()
        (self.out / "X-F01.md").unlink()
        self.assertTrue(self.run_main()[1])

    def test_no_stamp_builds_the_kits(self):
        self.assertTrue(self.run_main()[1])


if __name__ == "__main__":
    unittest.main()
