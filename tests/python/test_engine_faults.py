"""Engine faults (docs/ENGINE_FAULTS.md): the step's fault code raised as `EngineFault`, and the drivers' rows and counts.

The pure parts (the fault's fields, the counts, each driver's row) need no build; the last test steps engine fault class 2's
state on the real library, where the engine stops a hang of the game and the binding raises the fault.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sts2_gym import faults  # noqa: E402

TEXT = "GameHangs\tCardEffects.EndlessDraw.After\tPillage draws forever: for 64 draws ..."


class FaultTests(unittest.TestCase):
    def test_the_text_gives_class_frame_and_message(self):
        f = faults.EngineFault(TEXT)
        self.assertEqual((f.cls, f.frame), ("GameHangs", "CardEffects.EndlessDraw.After"))
        self.assertTrue(f.detail.startswith("Pillage draws forever"))
        self.assertEqual(f.reason, "engine_fault:GameHangs")
        self.assertEqual(f.row(), {"reason": "engine_fault:GameHangs", "engine_fault": "GameHangs", "fault_frame": "CardEffects.EndlessDraw.After"})
        self.assertIsInstance(f, RuntimeError)
        self.assertIs(faults.fault_of(f), f)
        self.assertIsNone(faults.fault_of(RuntimeError("other")))

    def test_a_short_text_still_names_a_class(self):
        f = faults.EngineFault("")
        self.assertEqual((f.cls, f.frame, f.detail), ("unknown", "unknown", ""))

    def test_counts_read_every_drivers_row(self):
        rows = [
            {"end": "engine_fault", "engine_fault": "GameHangs"},  # an arm's or the seed exam's run
            {"capped": True, "cap_reason": "engine_fault:InvalidOperationException at ActionQueueSet.GetReadyAction"},  # the gym pool
            {"error": "...", "reason": "engine_fault:GameHangs"},  # a corpus fight
            {"capped": True, "cap_reason": "worker died"},
            {"end": "win"},
        ]
        self.assertEqual(faults.engine_fault_counts(rows), {"GameHangs": 2, "InvalidOperationException": 1})
        self.assertEqual(faults.engine_fault_counts([]), {})


class BindingTests(unittest.TestCase):
    """native.step and native.run_step raise the fault on the code, against a stand-in library."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tests" / "python"))
        import test_solver_binding  # noqa: PLC0415

        cls.native, cls.lib = test_solver_binding.load_native_without_a_library()

    def test_the_combat_step_raises_the_fault(self):
        native = self.native
        self.lib.Sts2_Step.return_value = native.ENGINE_FAULT_CODE
        with mock.patch.object(native, "_get_last_error", return_value=TEXT):
            with self.assertRaises(native.EngineFault) as caught:
                native.step(1, 0, None, None)
        self.assertEqual(caught.exception.cls, "GameHangs")
        self.assertNotIsInstance(caught.exception, native.StepRefused)
        self.lib.Sts2_Step.return_value = -1
        with mock.patch.object(native, "_get_last_error", return_value="refused"), self.assertRaises(native.StepRefused):
            native.step(1, 0, None, None)

    def test_the_run_step_raises_the_fault(self):
        native = self.native
        self.lib.Sts2Run_Step.return_value = native.ENGINE_FAULT_CODE
        with mock.patch.object(native, "_get_last_error", return_value=TEXT), self.assertRaises(native.EngineFault):
            native.run_step(1, 0, -1, None, None, None, None)
        self.lib.Sts2Run_Step.return_value = -1
        self.assertEqual(native.run_step(1, 0, -1, None, None, None, None), -1)


class DriverTests(unittest.TestCase):
    """Each driver writes a fault as a row with its class and counts it; none runs it again."""

    def test_a_judgement_arms_run_is_a_fault_row_and_the_arm_counts_it(self):
        import gen0  # noqa: PLC0415

        with mock.patch.object(gen0, "play_emu_run", side_effect=faults.EngineFault(TEXT)):
            row = gen0._run_safe("SEEDXXXXXX", 3, 16, 10)  # noqa: SLF001
        self.assertEqual((row["end"], row["reason"], row["engine_fault"], row["fault_frame"]),
                         ("engine_fault", "engine_fault:GameHangs", "GameHangs", "CardEffects.EndlessDraw.After"))
        self.assertTrue(row["crash"])
        ok = {"seed": "S2", "ix": 4, "win": False, "floor": 20, "end": "death", "decisions": 9, "wall": 1.0, "crash": False}
        block = gen0.engine_block([row, ok])
        self.assertEqual(block["engine_faults"], {"GameHangs": 1})
        self.assertNotIn("engine_faults", gen0.engine_block([ok]))

    def test_a_seed_exam_run_is_a_fault_row(self):
        import gen0  # noqa: PLC0415
        import run_seed_exam  # noqa: PLC0415

        with mock.patch.object(gen0, "_setup"), mock.patch.object(gen0, "play_emu_run", side_effect=faults.EngineFault(TEXT)):
            row = run_seed_exam.play_task("SEEDXXXXXX", "k" * 16, 0, 5, "a3-v1", 10, None, True)
        self.assertEqual((row["end"], row["engine_fault"]), ("engine_fault", "GameHangs"))
        self.assertNotIn("SEEDXXXXXX", json.dumps(row))

    def test_the_gym_pools_worker_sends_the_fault_as_its_reason(self):
        import fight_gym  # noqa: PLC0415

        mine, theirs = mp.Pipe()

        def fn(_task):
            raise faults.EngineFault(TEXT)

        mine.send(("arm", "s", 0))
        mine.send(None)
        fight_gym._worker_main(theirs, None, (), fn)  # noqa: SLF001
        kind, msg = mine.recv()
        self.assertEqual((kind, msg), ("error", "engine_fault:GameHangs at CardEffects.EndlessDraw.After"))
        self.assertEqual(faults.engine_fault_counts([{"capped": True, "cap_reason": msg}]), {"GameHangs": 1})

    def test_the_teacher_does_not_ask_for_a_faulted_run_again(self):
        import run_teacher  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as d:
            rows = [{"seed": "A", "capped": True, "cap_reason": "engine_fault:GameHangs at CardEffects.EndlessDraw.After"},
                    {"seed": "B", "capped": True, "cap_reason": "worker died"},
                    {"seed": "C", "floor": 12}]
            (Path(d) / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual(run_teacher.fault_seeds(d), {"A"})
            self.assertEqual({r["seed"] for r in run_teacher.done_rows(d)}, {"C"})


def _hang(queue) -> None:
    """Engine fault class 2's state stepped on the real library (a child process: a library without the fix never returns)."""
    try:
        for p in (ROOT / "rl", ROOT / "src", ROOT):
            sys.path.insert(0, str(p))
        os.environ["STS2_ENGINE"] = "core2"
        from sts2_gym import names, native  # noqa: PLC0415

        buffs = {name: buff_id for buff_id, name in names.buffs().items()}
        cards = {info.name: card_id for card_id, info in names.cards().items()}
        h = native.create(5)
        doc = native.get_state_json(h)
        native.destroy(h)
        template = doc["piles"]["hand"][0]
        card = lambda name: {**template, "def_id": cards[name]}  # noqa: E731
        doc["pending_selection"] = None
        doc["piles"]["hand"] = [card("Pillage"), card("DefendIronclad")]
        doc["piles"]["draw"] = {"known_top": [], "unordered": [card("StrikeIronclad"), card("StrikeIronclad")], "known_bottom": []}
        doc["piles"]["discard"] = [card("PommelStrike")]
        doc["player"]["energy"] = 3
        doc["player"]["buffs"] = [{"id": buffs["Hellraiser"], "amount": 1}, {"id": buffs["Ringing"], "amount": 1}]
        h = native.load_combat_snapshot(doc, 1)
        import ctypes  # noqa: PLC0415

        obs = (ctypes.c_int * native.OBS_SIZE)()
        reward = (ctypes.c_float * 1)()
        before = native.engine_faults_seen()
        try:
            native.step(h, 0, obs, reward, 0)
            queue.put(("stepped", None))
        except native.EngineFault as fault:
            again = None
            try:
                native.step(h, 1, obs, reward)
            except native.EngineFault as second:
                again = second.cls
            queue.put(("fault", [fault.cls, fault.frame, again, native.engine_faults_seen() - before]))
    except Exception as error:  # noqa: BLE001 -- the parent skips when the native library is not there
        queue.put(("error", f"{type(error).__name__}: {error}"))


class LibraryTests(unittest.TestCase):
    def test_a_hang_of_the_game_is_a_fault_the_binding_raises(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        p = ctx.Process(target=_hang, args=(queue,), daemon=True)
        p.start()
        p.join(60)
        hung = p.is_alive()
        if hung:
            p.kill()  # this test's own child
            p.join()
        said = []
        while not queue.empty():
            said.append(queue.get())
        errors = [msg for kind, msg in said if kind == "error"]
        if errors and os.environ.get("STS2_REQUIRE_AOT") != "1":
            self.skipTest(f"native library unavailable: {errors[0]}")
        self.assertFalse(hung, "the step did not return")
        self.assertEqual([kind for kind, _ in said], ["fault"], said)
        cls, frame, again, counted = said[0][1]
        self.assertEqual((cls, again, counted), ("GameHangs", "GameHangs", 1))
        self.assertIn("EndlessDraw", frame)


if __name__ == "__main__":
    unittest.main()
