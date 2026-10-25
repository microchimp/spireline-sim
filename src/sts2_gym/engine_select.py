"""Which emulator engine a process runs (docs/rebuild/NATIVE_KERNEL.md section 3).

`STS2_ENGINE` is read once by `sts2_gym.native`, when the library loads, and inherited by every process a command spawns.
Importing this module loads no library, so an entry point can pin the engine before the import that would load it.
"""

from __future__ import annotations

import os

ENGINE_ENV = "STS2_ENGINE"


def select_engine(engine: str) -> None:
    """Set the engine of this process and every process it starts.

    "old" clears the variable, whatever the environment said, so a command that did not ask for core2 runs old; anything
    else (`core2`, `core2:F08,F10`) is passed on for `native` to check.
    """
    if engine == "old":
        os.environ.pop(ENGINE_ENV, None)
    else:
        os.environ[ENGINE_ENV] = engine
