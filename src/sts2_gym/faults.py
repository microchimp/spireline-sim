"""Engine faults (docs/ENGINE_FAULTS.md): the exception a step raises when the engine's own code threw, and their counts.

Pure: no native code is loaded here, so a driver or a test can name a fault without the library. `sts2_gym.native` raises
`EngineFault` when `Sts2_Step`, `Sts2_StepTargeted` or `Sts2Run_Step` returns `ENGINE_FAULT_CODE`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# What the step exports return when an exception escaped the engine's step. A refused answer to a stopped step's screen is -1.
ENGINE_FAULT_CODE = -2


class EngineFault(RuntimeError):
    """Raised when a step's engine code threw: the step export returned the fault code.

    `cls` is the exception's class ("ArgumentOutOfRangeException", "GameHangs" for a state where the game itself never
    finishes the step), `frame` the top frame in the engine ("CardEffects.DrawCards"), `detail` its message. The combat or
    run is left where the exception found it, part way through the step: its handle refuses every later step with the same
    fault, and the caller ends the fight or run as `engine_fault:<cls>` (`reason`), counted, never retried. A `RuntimeError`,
    so code that handles a failed native call handles this too; not a `native.StepRefused`, whose combat is back on its
    screen and can go on.
    """

    def __init__(self, text: str) -> None:
        cls, frame, detail = [*text.split("\t", 2), "", "", ""][:3]
        self.cls = cls or "unknown"
        self.frame = frame or "unknown"
        self.detail = detail
        super().__init__(f"engine fault {self.cls} at {self.frame}: {detail}")

    @property
    def reason(self) -> str:
        """The reason a driver writes for the fight or run: `engine_fault:<class>`."""
        return f"engine_fault:{self.cls}"

    def row(self) -> dict[str, str]:
        """The fields a driver's row carries for this fault."""
        return {"reason": self.reason, "engine_fault": self.cls, "fault_frame": self.frame}


def fault_of(error: BaseException) -> EngineFault | None:
    """`error` when it is an engine fault, else None."""
    return error if isinstance(error, EngineFault) else None


def engine_fault_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """How many of `rows` ended on an engine fault, by class: the `engine_fault` field, or a `reason` or `cap_reason` of
    `engine_fault:<class>`. A job's summary carries this as `engine_faults`."""
    out: dict[str, int] = {}
    for r in rows:
        cls = r.get("engine_fault")
        if not cls:
            why = str(r.get("reason") or r.get("cap_reason") or "")
            cls = why[len("engine_fault:"):].split()[0] if why.startswith("engine_fault:") and len(why) > len("engine_fault:") else None
        if cls:
            out[cls] = out.get(cls, 0) + 1
    return dict(sorted(out.items()))
