"""Workers for `deckfights.play_in_processes`, for test_run_train_deckfights.py.

A spawned process imports its target by module name before it runs a line of it. These live
apart from the test module, which pulls in numpy, torch and the whole `rl` package, so a
worker is up in a fraction of a second and a test's stall budget measures the stall rather
than the host's process start-up (about 3 s on Windows through the test module, against a
budget of 1 s: the stalled-worker test could never pass there).
"""

from __future__ import annotations

import time


def ok_worker(conn, jobs, *args: object):
    for index, spec in jobs:
        conn.send([(index, {"index": index, "seed": spec["seed"]})])
    conn.send(None)
    conn.close()


def hang_worker(conn, jobs, *args: object):
    index, spec = jobs[0]
    conn.send([(index, {"index": index, "seed": spec["seed"]})])
    time.sleep(60)  # stalls: never sends again


def crash_worker(conn, jobs, *args: object):
    if jobs[0][0] == 0:
        raise RuntimeError("a native abort stand-in")
    ok_worker(conn, jobs, *args)
