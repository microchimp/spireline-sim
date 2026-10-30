"""Task functions for the spawn-pool tests: module-level so a spawned worker can import them (tests/python/test_spawn_pool.py)."""

from __future__ import annotations

import os
import time
from pathlib import Path


def ok(tag: str = "") -> tuple[str, int]:
    return tag, os.getpid()


def die(code: int = 3) -> None:
    os._exit(code)


def hang(seconds: float = 120.0) -> None:
    time.sleep(seconds)


def boom() -> None:
    raise ValueError("a task that raises")


class FakeAgent:
    """What run_block asks of an agent when there is no prior server: a generation and a description."""
    generation = 7
    prior_server = None
    prior_authkey = None

    def describe(self) -> dict:
        return {"generation": 7, "label": "fake"}


def fake_task(seed, ix, sims, ascension, agent, record, character) -> dict:
    """run_block's task. The seed's name picks the behaviour; POOL_FAKE_DIR remembers which seeds were tried before (so
    DIEONCE and HANGONCE misbehave on the first attempt only); POOL_FAKE_SLEEP is the normal run's length; POOL_FAKE_TOKEN
    is copied into the row to tell the rows of one pass from another's."""
    d = os.environ.get("POOL_FAKE_DIR", "")
    first = True
    if d:
        mark = Path(d) / seed
        first = not mark.exists()
        mark.write_text("1")
    if seed.startswith("DIEALWAYS") or seed.startswith("ALLDIE") or (seed.startswith("DIEONCE") and first):
        os._exit(3)
    if seed.startswith("HANGALWAYS") or (seed.startswith("HANGONCE") and first):
        time.sleep(120)
    time.sleep(float(os.environ.get("POOL_FAKE_SLEEP", "0")))
    return {"seed": seed, "ix": ix, "win": False, "floor": 5, "end": "death", "decisions": 10, "wall": 0.5, "ai_state_assumed": 0,
            "pid": os.getpid(), "token": os.environ.get("POOL_FAKE_TOKEN", "")}


def prio_task(seed, ix, sims, ascension, agent, record, character) -> dict:
    """fake_task's row plus the worker's Windows priority class under `prio` (None elsewhere): the streamed arm's tests."""
    row = fake_task(seed, ix, sims, ascension, agent, record, character)
    prio = None
    if os.name == "nt":
        import ctypes  # noqa: PLC0415

        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        prio = int(k32.GetPriorityClass(k32.GetCurrentProcess()))
    return {**row, "prio": prio}
