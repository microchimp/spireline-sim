"""tools/lab/groupcache.py and the grouping step that uses it (tools/fidelity/lab_groups.py): what the step learns from a finished
job and from each source is kept under LAB/cache/groups, so a source is read once in its life.

A fake lab of four finished jobs (a fights job with two batches, a presort with a codex walk, a corpus result and a reference-set
score, an invariants job with a mutation report, a runs job) stands in for the real one. Its files are aged by an hour: the cache
trusts nothing newer than two seconds. No game, no emulator.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "referee"))

from tools.fidelity import lab_groups as lg  # noqa: E402
from tools.lab import groupcache as gc  # noqa: E402
from tools.lab import stepcache as sc  # noqa: E402

T0 = time.time_ns() - 3600 * 10**9  # the time every fixture file carries: an hour ago


def put(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write(path: Path, doc) -> Path:
    return put(path, json.dumps(doc))


def age(path: Path, ns: int = T0) -> None:
    os.utime(path, ns=(ns, ns))


def age_all(root: Path, ns: int = T0) -> None:
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            age(Path(dp) / f, ns)


def verdicts(seeds, field=".hand"):
    return [{"seed": s, "identical": False, "step": 0, "field": field, "card": "start", "flags": [], "diff": [[field, "[A]", "[B]"]]} for s in seeds]


def batch(job: Path, name: str, seeds, field=".hand") -> None:
    specs = [{"seed": s, "setup": [{"cmd": "relic", "id": "EGG_A"}, {"cmd": "card", "id": "ANGER"}, {"cmd": "fight", "id": "SLIMES_WEAK"}]} for s in seeds]
    write(job / f"{name}.json", {"runs": specs})
    write(job / f"{name}.verdicts.json", verdicts(seeds, field))
    write(job / f"{name}.out.json", {"results": [{"seed": s, "steps": [{"state": {}}]} for s in seeds]})
    write(job / f"{name}.shrink.json", {"cases": [{"seed": seeds[0], "status": "minimal", "items": [{"cmd": "relic", "id": "EGG_A"}], "of": 3, "fights": 4}]})
    write(job / f"{name}.packs" / "minimal" / f"{seeds[0]}.json", {"runs": [{"setup": [{"cmd": "relic", "id": "EGG_A"}, {"cmd": "fight", "id": "SLIMES_WEAK"}]}]})
    write(job / f"{name}.packs" / "traces" / f"{seeds[0]}.trace.json", {"error": "no trace"})


def make_lab(td: Path) -> tuple[Path, list[str]]:
    lab = td / "lab"
    runs = lab / "runs"
    fights = runs / "20261015T0100-fights"
    batch(fights, "b1", ["SEEDA00001", "SEEDA00002", "SEEDA00003"])
    batch(fights, "b2", ["SEEDB00001", "SEEDB00002"], ".draw")
    (fights / "done.rc").write_text("0", encoding="utf-8")
    pre = runs / "20261015T0200-presort"
    walk = {"checks": [], "status": "ok"}
    stop = {"floor": 3, "component": "relics", "field": "gold", "class": "K", "expected": "ANGER", "emulator": "BASH", "note": "n", "act": 1}
    put(pre / "codex" / "s0.jsonl", "\n".join(json.dumps(x) for x in (walk, {"status": "mismatch", "stop": stop, "checks": []},
                                                                      {"status": "mismatch", "stop": {**stop, "floor": 4}, "checks": []})) + "\n")
    write(pre / "corpus" / "results.json", {"format": 1, "totals": {"fights": 5}, "fights": {"F1": {"fd": 2, "card": "BASH", "field": ".enemies", "stop": "x"}}})
    write(pre / "refset" / "score.json", {"set": "Ironclad A10 reference set v2", "headline": {}, "record_dir": str(td / "referee" / "refset" / "v2" / "record" / "all"),
                                          "verdicts": [{"id": "R1", "seed": "RSEED00001", "identical": False, "step": 1, "field": ".hand", "card": "BASH",
                                                        "diff": [[".hand", "a", "b"]]}]})
    write(pre / "job.json", {"outputs": {"codex_walk": ["codex/s0.jsonl"], "refset_score": ["refset/score.json"], "corpus": ["corpus/results.json"]}})
    (pre / "done.rc").write_text("0", encoding="utf-8")
    inv = runs / "20261015T0300-invariants"
    write(inv / "invariants.json", {"label": "game recordings", "game": {"kinds": {"energy": [["energy 3 > 5 for CARD_X", 2]]}}})
    write(inv / "mutation.json", {"kill": 5, "survivors": [{"id": "Foo.cs:12:op", "old": "a", "new": "b", "op": "swap"}]})
    (inv / "done.rc").write_text("0", encoding="utf-8")
    rn = runs / "20261015T0400-runs"
    write(rn / "runs" / "x.runs.json", {"runs": [{"seed": "RUNSEED001", "recorded": True,
                                                   "first": {"decision": 3, "component": "shop", "floor": 4, "choice": 1, "diffs": [[".gold: 10 vs 20", "ANGER", "BASH"]]}}]})
    (rn / "done.rc").write_text("0", encoding="utf-8")
    write(td / "referee" / "refset" / "v2" / "manifest.json", {"specs": [{"id": "R1", "spec": {"setup": [{"cmd": "fight", "id": "SLIMES_WEAK"}]}}]})
    write(lab / "cache" / "model_features.json", {"stamp": None, "features": {}})
    write(td / "hosts.json", {"entries": []})
    write(td / "backlog-x.json", {"items": [{"id": "X-F01", "status": "open", "title": "Egg", "evidence": "EGG_A at start"}]})
    age_all(td)
    return lab, ["--lab", str(lab), "--out", str(lab / "groups.json"), "--summary", str(lab / "summary.md"), "--decompiled", str(td / "none"),
                 "--host-gaps", str(td / "hosts.json"), "--backlog", str(td / "backlog-x.json"), "--referee", str(td / "referee")]


class CacheCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.td = Path(self._td.name)
        self.lab, self.argv = make_lab(self.td)
        self.cache = self.lab / "cache" / "groups"
        self.now_ns: int | None = None  # the listing's 'now' for the next run: None = the real clock
        self.fake_root: Path | None = None

    def run_step(self, *extra: str) -> tuple[str, dict]:
        """The grouping step again from nothing but the cache: its previous output and stamp are removed first."""
        for p in (self.lab / "groups.json", self.lab / "summary.md", self.lab / "stamps" / "groups.json"):
            p.unlink(missing_ok=True)
        real = gc.GroupCache

        def make(lab, root, **kw):
            if self.now_ns is not None:
                kw["now_ns"] = self.now_ns
            if self.fake_root is not None:
                return real(lab, self.fake_root, code_files=("code.py",), **kw)
            return real(lab, root, **kw)

        buf = io.StringIO()
        with mock.patch.object(lg, "GroupCache", make), contextlib.redirect_stdout(buf):
            self.assertEqual(0, lg.main(self.argv + list(extra)))
        doc = json.loads((self.lab / "groups.json").read_text(encoding="utf-8"))
        doc.pop("generated_utc")
        return buf.getvalue(), doc

    @staticmethod
    def counts(text: str) -> dict:
        m = re.search(r"jobs (\d+) \(source list from cache (\d+), found (\d+), uncached (\d+)\); sources (\d+) \(from cache (\d+), read (\d+), uncached (\d+)\); "
                      r"symptoms (\d+) from cache, (\d+) read", text)
        assert m, text
        keys = ("jobs", "jobs_hit", "jobs_found", "jobs_uncached", "sources", "hit", "read", "uncached", "symptoms_hit", "symptoms_read")
        return dict(zip(keys, map(int, m.groups())))

    def cache_files(self) -> list[Path]:
        return sorted(p for p in self.cache.rglob("*") if p.is_file()) if self.cache.is_dir() else []

    def baseline(self) -> dict:
        return self.run_step("--no-cache")[1]


class ListingTests(CacheCase):
    def test_the_listing_gives_the_stat_signature_and_the_same_digest(self):
        ls = gc.Listing()
        files = [p for p in (self.lab / "runs").rglob("*") if p.is_file()]
        folders = [p for p in (self.lab / "runs").rglob("*") if p.is_dir()]
        for d in folders + [self.lab / "runs"]:  # every folder listed first: the signatures below come from the listing
            ls.names(d)
        for p in files + folders:  # a folder's size is 0 in a listing and 4096 to os.stat: the listing asks for folders
            self.assertEqual(sc.stat_sig(p), ls.sig(p))
        self.assertEqual([(str(p).replace("\\", "/"), sc.stat_sig(p), os.stat(p).st_mtime_ns) for p in files + folders], ls.stat_files(files + folders))
        for d in (self.lab / "runs" / "20261015T0100-fights", self.lab / "runs" / "20261015T0100-fights" / "b1.packs"):
            self.assertEqual(sorted(os.listdir(d)), ls.names(d))
        self.assertEqual(sc.stats_hash(files + folders, self.lab), ls.stats_hash(files + folders, self.lab))
        self.assertEqual(sc.stats_hash(files + folders, self.lab), ls.rows_hash(ls.stat_files(files + folders), self.lab))
        self.assertEqual(sc.MISSING, ls.sig(self.lab / "runs" / "nothing.json"))

    def test_walk_files_is_stepcache_walk_files_without_the_skipped_folders(self):
        top = self.lab / "runs" / "20261015T0100-fights" / "b1.packs"
        put(top / "obj" / "x.json", "{}")
        put(top / "deep" / "er" / "y.json", "{}")
        self.assertEqual(sorted(sc.walk_files(top)), sorted(gc.Listing().walk_files(top)))

    def test_the_skip_digest_is_the_one_stepcache_made(self):
        """inputs_parts counts the same files and gives the same digest whether the folders are listed or each file is asked for."""
        sources = lg.lab_sources(self.lab)
        listing: dict = {}
        files = [f for kind, path, _w in sources for f in lg.source_files(kind, path, listing, self.td / "referee")]
        parts = lg.inputs_parts(self.lab, sources, self.td / "backlog-x.json", self.td / "hosts.json", self.td / "none", self.td / "referee")
        self.assertEqual(sc.stats_hash(files, self.lab), parts["sources"])


class HitTests(CacheCase):
    def test_a_cache_hit_gives_the_groups_of_a_run_without_it_and_reads_nothing(self):
        base = self.baseline()
        self.assertFalse(self.cache.exists())  # --no-cache leaves the folder alone
        text, miss = self.run_step()
        c = self.counts(text)
        self.assertEqual(base, miss)
        self.assertEqual((c["sources"], c["hit"]), (c["read"], 0))
        self.assertEqual(4, c["jobs"])
        self.assertGreaterEqual(c["sources"], 7)
        self.assertTrue(self.cache_files())
        with mock.patch.object(lg, "source_kind", side_effect=AssertionError("sniffed")), \
                mock.patch.object(lg, "pool_size", side_effect=AssertionError("pool")), \
                mock.patch.object(lg, "batch_symptoms", side_effect=AssertionError("read")), \
                mock.patch.object(lg, "walk_symptoms", side_effect=AssertionError("read")), \
                mock.patch.object(lg, "run_symptoms", side_effect=AssertionError("read")), \
                mock.patch.object(lg, "corpus_symptoms", side_effect=AssertionError("read")), \
                mock.patch.object(lg, "refset_symptoms", side_effect=AssertionError("read")), \
                mock.patch.object(lg, "invariant_symptoms", side_effect=AssertionError("read")), \
                mock.patch.object(lg, "mutation_symptoms", side_effect=AssertionError("read")):
            text, hit = self.run_step()
        h = self.counts(text)
        self.assertEqual(base, hit)
        self.assertEqual((h["jobs_hit"], h["jobs_found"], h["hit"], h["read"]), (4, 0, c["sources"], 0))
        self.assertEqual(c["symptoms_read"], h["symptoms_hit"])

    def test_the_sources_come_in_the_same_order_with_and_without_the_cache(self):
        plain = lg.lab_sources(self.lab)
        cache = gc.GroupCache(self.lab, ROOT)
        cache.files_of = lambda kind, path: lg.source_files(kind, path, cache.listing, self.td / "referee")
        first = lg.lab_sources(self.lab, (), cache)
        for kind, path, when in first:  # fill it
            tok = cache.lookup(kind, path, when)
            cache.store(tok, [], 0, f"{kind}:x")
        cache.finish()
        again = gc.GroupCache(self.lab, ROOT)
        self.assertEqual(plain, first)
        self.assertEqual(plain, lg.lab_sources(self.lab, (), again))
        self.assertEqual(4, again.stats.jobs_hit)

    def test_a_cached_source_keeps_every_symptom_field_as_the_reader_made_it(self):
        for kind, path, when in lg.lab_sources(self.lab):
            reader = {"batch": lg.batch_symptoms, "run": lg.run_symptoms, "corpus": lg.corpus_symptoms, "refset": lg.refset_symptoms,
                      "codex": lg.walk_symptoms, "invariants": lg.invariant_symptoms, "mutation": lg.mutation_symptoms}[kind]
            got = reader(path, self.lab, when)
            self.assertEqual(got, json.loads(gzip.decompress(gzip.compress(json.dumps(got, separators=(",", ":")).encode()))), kind)

    def test_the_cache_holds_no_absolute_path(self):
        self.run_step()
        self.assertTrue(self.cache_files())
        for p in self.cache_files():
            text = gzip.decompress(p.read_bytes()).decode() if p.name.endswith(".gz") else p.read_text(encoding="utf-8")
            for bad in (str(self.td), str(self.td).replace("\\", "/"), str(self.td).replace("\\", "\\\\")):
                self.assertNotIn(bad, text, p.name)

    def test_force_clears_the_cache_and_reads_again(self):
        self.run_step()
        stray = put(self.cache / "jobs" / "stray.json", "{}")
        text, _ = self.run_step("--force")
        c = self.counts(text)
        self.assertEqual(0, c["hit"])
        self.assertFalse(stray.exists())
        self.assertTrue(self.cache_files())

    def test_no_cache_neither_reads_nor_writes(self):
        self.run_step()
        before = {p: p.stat().st_mtime_ns for p in self.cache_files()}
        text, _ = self.run_step("--no-cache")
        self.assertNotIn("cache:", text)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in self.cache_files()})


class MissTests(CacheCase):
    def test_a_changed_file_misses_that_source_alone(self):
        self.run_step()
        v = self.lab / "runs" / "20261015T0100-fights" / "b1.verdicts.json"
        write(v, verdicts(["SEEDA00001", "SEEDA00002", "SEEDA00003", "SEEDA00001"]))  # another size
        age(v, T0 - 7 * 10**9)  # and another time, an old one
        text, doc = self.run_step()
        c = self.counts(text)
        self.assertEqual((c["read"], c["jobs_found"], c["jobs_hit"]), (1, 1, 3))
        self.assertEqual(c["sources"] - 1, c["hit"])
        self.assertEqual(self.baseline(), doc)
        text, _ = self.run_step()
        self.assertEqual(0, self.counts(text)["read"])  # and the new reading is kept

    def test_a_file_of_the_packs_folder_changing_misses_its_batch(self):
        self.run_step()
        m = self.lab / "runs" / "20261015T0100-fights" / "b1.packs" / "minimal" / "SEEDA00001.json"
        write(m, {"runs": [{"setup": [{"cmd": "relic", "id": "EGG_B"}, {"cmd": "fight", "id": "SLIMES_WEAK"}]}]})
        age(m, T0 - 5 * 10**9)
        text, doc = self.run_step()
        c = self.counts(text)
        self.assertEqual(1, c["read"])
        self.assertEqual(self.baseline(), doc)

    def test_a_new_job_is_the_only_one_read(self):
        self.run_step()
        job = self.lab / "runs" / "20261016T0100-fights"
        batch(job, "c1", ["SEEDC00001", "SEEDC00002"])
        (job / "done.rc").write_text("0", encoding="utf-8")
        age_all(job)
        text, doc = self.run_step()
        c = self.counts(text)
        self.assertEqual((c["jobs"], c["jobs_hit"], c["jobs_found"], c["read"]), (5, 4, 1, 1))
        self.assertEqual(self.baseline(), doc)

    def test_changed_code_misses_everything(self):
        self.fake_root = self.td / "root"
        code = put(self.fake_root / "code.py", "x = 1\n")
        text, _ = self.run_step()
        self.assertEqual(0, self.counts(text)["hit"])
        text, _ = self.run_step()
        c = self.counts(text)
        self.assertEqual((c["read"], c["jobs_hit"]), (0, 4))
        code.write_text("x = 2\n", encoding="utf-8")
        text, doc = self.run_step()
        c = self.counts(text)
        self.assertEqual((c["hit"], c["jobs_hit"], c["jobs_found"]), (0, 0, 4))
        self.assertEqual(c["sources"], c["read"])
        self.assertEqual(self.baseline(), doc)
        text, _ = self.run_step()  # the new code's entries stand
        self.assertEqual(0, self.counts(text)["read"])

    def test_the_old_code_files_are_removed_by_the_clean_up(self):
        self.fake_root = self.td / "root"
        code = put(self.fake_root / "code.py", "x = 1\n")
        self.run_step()
        old = set(self.cache_files())
        code.write_text("x = 2\n", encoding="utf-8")
        self.run_step()
        for p in old:
            age(p, time.time_ns() - 2 * 3600 * 10**9)  # past the hour another run might still be writing one
        self.run_step()
        self.assertTrue(all(not p.name.endswith(".gz") or p not in old for p in self.cache_files()))

    def test_a_job_that_is_gone_loses_its_cache_file(self):
        self.run_step()
        gone = self.cache / "jobs" / "20261015T0400-runs.json"
        self.assertTrue(gone.is_file())
        for dp, dn, fn in os.walk(self.lab / "runs" / "20261015T0400-runs", topdown=False):
            for f in fn:
                os.unlink(os.path.join(dp, f))
            for d in dn:
                os.rmdir(os.path.join(dp, d))
        os.rmdir(self.lab / "runs" / "20261015T0400-runs")
        for p in self.cache_files():
            age(p, time.time_ns() - 2 * 3600 * 10**9)
        text, _ = self.run_step()
        self.assertEqual(3, self.counts(text)["jobs"])
        self.assertFalse(gone.exists())
        self.assertTrue((self.cache / "jobs" / "20261015T0100-fights.json").exists())


class TwoSecondRuleTests(CacheCase):
    def test_nothing_newer_than_two_seconds_is_stored(self):
        self.now_ns = T0 + 1 * 10**9  # every file is one second old
        text, doc = self.run_step()
        c = self.counts(text)
        self.assertEqual((c["jobs_uncached"], c["hit"]), (4, 0))
        self.assertEqual(c["sources"], c["uncached"])
        self.assertEqual([], self.cache_files())
        self.assertEqual(self.baseline(), doc)

    def test_a_size_and_time_match_is_not_trusted_for_a_file_newer_than_two_seconds(self):
        self.now_ns = T0 + 3 * 10**9
        text, _ = self.run_step()
        self.assertEqual(0, self.counts(text)["uncached"])
        self.assertTrue(self.cache_files())
        text, _ = self.run_step()
        self.assertEqual(self.counts(text)["sources"], self.counts(text)["hit"])  # three seconds: trusted
        self.now_ns = T0 + 1 * 10**9  # the same files and the same entries, but the files are one second old
        text, doc = self.run_step()
        c = self.counts(text)
        self.assertEqual((c["hit"], c["jobs_hit"]), (0, 0))
        self.assertEqual(c["sources"], c["read"] + c["uncached"])
        self.assertEqual(self.baseline(), doc)

    def test_one_warm_file_keeps_only_its_job_out(self):
        self.now_ns = T0 + 3 * 10**9
        self.run_step()
        age(self.lab / "runs" / "20261015T0400-runs" / "runs" / "x.runs.json", T0 + 2 * 10**9)  # one second old at 'now'
        text, doc = self.run_step()
        c = self.counts(text)
        self.assertEqual(3, c["jobs_hit"])
        self.assertEqual(1, c["jobs_uncached"])
        self.assertEqual(self.baseline(), doc)

    def test_the_clean_up_keeps_the_files_of_a_job_a_warm_file_left_out_of_this_run(self):
        self.now_ns = T0 + 3 * 10**9
        self.run_step()
        before = set(self.cache_files())
        age(self.lab / "runs" / "20261015T0400-runs" / "runs" / "x.runs.json", T0 + 2 * 10**9)  # one second old at 'now'
        for p in before:
            age(p, time.time_ns() - 2 * 3600 * 10**9)  # old enough for the clean-up to take
        self.run_step()
        self.assertEqual(before, set(self.cache_files()))

    def test_a_file_from_the_future_is_not_trusted_either(self):
        age(self.lab / "runs" / "20261015T0400-runs" / "runs" / "x.runs.json", time.time_ns() + 3600 * 10**9)
        text, _ = self.run_step()
        self.assertEqual(1, self.counts(text)["jobs_uncached"])


class DamagedCacheTests(CacheCase):
    def test_every_kind_of_damage_is_deleted_and_rebuilt_without_a_failure(self):
        base = self.baseline()
        self.run_step()
        jobs = sorted((self.cache / "jobs").glob("*.json"))
        blobs = sorted((self.cache / "sym").rglob("*.json.gz"))
        self.assertGreaterEqual(len(jobs), 4)
        self.assertGreaterEqual(len(blobs), 3)
        jobs[0].write_text("{not json", encoding="utf-8")  # garbage
        jobs[1].write_text(json.dumps({"format": 1, "code": 5}), encoding="utf-8")  # valid json, wrong shape
        blobs[0].write_bytes(blobs[0].read_bytes()[:20])  # cut short
        blobs[1].write_bytes(gzip.compress(json.dumps([{"a": 1}]).encode()))  # valid json in valid gzip, but not what the entry wrote
        blobs[2].write_bytes(b"\x00\x01 not gzip at all")
        text, doc = self.run_step()
        self.assertEqual(base, doc)
        self.assertRegex(text, r"corrupt files deleted [1-9]")
        text, doc = self.run_step()
        c = self.counts(text)
        self.assertEqual(base, doc)
        self.assertNotIn("corrupt", text)
        self.assertEqual((c["read"], c["hit"]), (0, c["sources"]))  # whole again

    def test_a_missing_symptom_file_is_a_miss(self):
        base = self.baseline()
        self.run_step()
        blob = sorted((self.cache / "sym").rglob("*.json.gz"))[0]
        blob.unlink()
        text, doc = self.run_step()
        self.assertEqual(base, doc)
        self.assertEqual(1, self.counts(text)["read"])

    def test_a_cache_that_cannot_be_written_costs_only_the_speed(self):
        base = self.baseline()
        put(self.lab / "cache" / "groups", "a file where the folder should be")
        text, doc = self.run_step()
        self.assertEqual(base, doc)
        self.assertIn("going on without it", text)
        self.assertEqual(0, self.counts(text)["hit"])


class ProgressTests(CacheCase):
    def test_a_run_that_stops_keeps_the_jobs_it_finished(self):
        base = self.baseline()
        with mock.patch.object(lg, "run_symptoms", side_effect=RuntimeError("stopped")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                self.run_step()
        names = sorted(p.name for p in (self.cache / "jobs").glob("*.json"))
        self.assertEqual(["20261015T0100-fights.json", "20261015T0200-presort.json", "20261015T0300-invariants.json"], names)
        text, doc = self.run_step()
        c = self.counts(text)
        self.assertEqual(base, doc)
        self.assertEqual(1, c["read"])  # only the runs job, the one that was not reached
        self.assertEqual(c["sources"] - 1, c["hit"])


class UnfinishedJobTests(CacheCase):
    def test_the_runners_own_unfinished_job_is_read_and_not_cached(self):
        own = self.lab / "runs" / "20261017T0100-fights"
        batch(own, "o1", ["SEEDO00001"])
        age_all(own)
        text, doc = self.run_step("--job", str(own))
        c = self.counts(text)
        self.assertEqual(1, c["uncached"])
        self.assertEqual(c["sources"] - 1, c["read"])
        self.assertFalse((self.cache / "jobs" / "20261017T0100-fights.json").exists())
        self.assertIn("batch:runs/20261017T0100-fights/o1", json.dumps(doc))
        text, doc2 = self.run_step("--job", str(own))
        self.assertEqual(doc, doc2)
        self.assertEqual(1, self.counts(text)["uncached"])


if __name__ == "__main__":
    unittest.main()
