"""The names of the run chooser's recipes (strings only; no numpy, no native code).

A candidate's agent.json `macro` is `rules`, `vrun` or a blended recipe:

    vrun+pi_h:lambda=<inf|number>:pi_h=<8 hex of the human prior's weights sha256>

(`imitation_chooser`: the run judgement V plus lambda times the log of the human prior pi_h on the deck decisions). The string is the
recipe's identity wherever the lab compares agents: an emulator arm's summary names it (`gen0.emu_arm_match`), and the ledger counts
promotion attempts per recipe (`gen_loop.attempts_before`): the plain macros (`rules`, `vrun`) are one recipe, written "" in the
ledger's `macro` column and in every row from before the column; each blended string is a recipe of its own.
"""

from __future__ import annotations

import math
import re

PREFIX = "vrun+pi_h"
_MACRO = re.compile(r"vrun\+pi_h:lambda=(inf|[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?):pi_h=([0-9a-f]{8})")


def format_lambda(lam: float) -> str:
    return "inf" if math.isinf(lam) else f"{lam:g}"


def macro_name(lam: float, weights_sha256: str) -> str:
    """The recipe string a candidate's agent.json carries as `macro`."""
    if not lam >= 0:
        raise ValueError(f"lambda is 0 or more (inf is the prior alone), not {lam}")
    if not re.fullmatch(r"[0-9a-f]{8,64}", weights_sha256):
        raise ValueError("the prior's sha256 is lowercase hex")
    return f"{PREFIX}:lambda={format_lambda(lam)}:pi_h={weights_sha256[:8]}"


def is_blended(name: str | None) -> bool:
    return bool(name) and str(name).startswith(PREFIX)


def parse_macro(name: str) -> tuple[float, str]:
    """(lambda, the 8 hex of the prior's sha256) of a blended recipe string."""
    m = _MACRO.fullmatch(name)
    if not m:
        raise ValueError(f"{name!r} is not a {PREFIX}:lambda=<inf|number>:pi_h=<8 hex> recipe")
    return (math.inf if m[1] == "inf" else float(m[1])), m[2]


def recipe_of(macro: str | None) -> str:
    """The ledger's recipe of a macro: "" for the plain macros (rules, vrun, none named), the string itself for a blended one."""
    return str(macro) if is_blended(macro) else ""
