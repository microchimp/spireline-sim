"""The names of the run chooser's recipes (strings only; no numpy, no native code).

A candidate's agent.json `macro` is `rules`, `vrun` or a blended recipe:

    vrun+pi_h:lambda=<inf|number>:pi_h=<8 hex of the human prior's weights sha256>[:pick=sample|:pick=skipcal<bias>]

(`imitation_chooser`: the run judgement V plus lambda times the log of the human prior pi_h on the deck decisions). The optional last
part says how the deck screen's option is taken: absent, the argmax of V + lambda log pi_h; `pick=sample`, a draw from
softmax((V + lambda log pi_h) / 1) seeded by the run seed and the floor; `pick=skipcal<bias>`, the argmax with `bias` floors added to
the skip option's score (the bias a calibration run chose, `rl/skip_calibrate.py`). The string is the recipe's identity wherever the lab
compares agents: an emulator arm's summary names it (`gen0.emu_arm_match`), and the ledger counts promotion attempts per recipe
(`gen_loop.attempts_before`): the plain macros (`rules`, `vrun`) are one recipe, written "" in the ledger's `macro` column and in every
row from before the column; each blended string is a recipe of its own.

Two more identity parts live beside the macro (agent.json `search_schedule` and `potion_hold`), because they change how the fights are
played and not what the chooser picks (2026-11-06, `docs/rl/STAKES_AND_HOLD.md`):

    stakes   a stakes schedule: the base search config's simulations times a multiplier per fight kind (hallway, elite, boss), named
             `<base config>@h<hall>e<elite>b<boss>`, e.g. `a3-v1@h1e4b16`
    hold     the hallway potion hold: a potion move is dropped from a hallway fight's root unless the kill check fails and the
             player would fall under `hold` per cent of max HP after the enemies' intended damage, named `hold<per cent>`

A search config's name is `<base>[@h<H>e<E>b<B>][+hold<N>]` (`search_configs.get` reads it, so every consumer of a config name plays
a stakes agent's schedule); the ledger's recipe is the macro, then `+stakes:<base>@h<H>e<E>b<B>`, then `+hold:<N>`, so an attempt
counts under the schedule and the hold it was played with. An agent with neither is the recipe `recipe_of(macro)` always was.
"""

from __future__ import annotations

import math
import re

PREFIX = "vrun+pi_h"
_MACRO = re.compile(
    r"vrun\+pi_h:lambda=(inf|[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?):pi_h=([0-9a-f]{8})(?::pick=(sample|skipcal-?[0-9]*\.?[0-9]+))?")
PICK_ARGMAX, PICK_SAMPLE, PICK_SKIPCAL = "argmax", "sample", "skipcal"
PICKS = (PICK_ARGMAX, PICK_SAMPLE, PICK_SKIPCAL)
FIGHT_KINDS = ("hall", "elite", "boss")
MAX_MULT = 64  # a stakes multiplier is a whole number from 1 to this
_BASE = r"[a-z0-9][a-z0-9-]*"
_STAKES = re.compile(r"(" + _BASE + r")@h(\d+)e(\d+)b(\d+)")
_CONFIG = re.compile(r"(" + _BASE + r")(?:@h(\d+)e(\d+)b(\d+))?(?:\+hold(\d+))?")
_RECIPE_STAKES = re.compile(r"\+stakes:([^+]+)")
_RECIPE_HOLD = re.compile(r"\+hold:(\d+)")


def format_lambda(lam: float) -> str:
    return "inf" if math.isinf(lam) else f"{lam:g}"


def format_bias(bias: float) -> str:
    """The skip bias as the recipe writes it: at most three decimals, never `-0` or an exponent."""
    b = round(float(bias), 3) + 0.0
    return f"{b:.3f}".rstrip("0").rstrip(".") if b != 0 else "0"


def pick_name(pick: str, skip_bias: float = 0.0) -> str:
    """The recipe's `pick=` part for a mode ("" for the argmax, which has none)."""
    if pick == PICK_ARGMAX:
        return ""
    if pick == PICK_SAMPLE:
        return f":pick={PICK_SAMPLE}"
    if pick == PICK_SKIPCAL:
        return f":pick={PICK_SKIPCAL}{format_bias(skip_bias)}"
    raise ValueError(f"unknown pick mode {pick!r}; one of {', '.join(PICKS)}")


def macro_name(lam: float, weights_sha256: str, pick: str = PICK_ARGMAX, skip_bias: float = 0.0) -> str:
    """The recipe string a candidate's agent.json carries as `macro`."""
    if not lam >= 0:
        raise ValueError(f"lambda is 0 or more (inf is the prior alone), not {lam}")
    if not re.fullmatch(r"[0-9a-f]{8,64}", weights_sha256):
        raise ValueError("the prior's sha256 is lowercase hex")
    if pick != PICK_SKIPCAL and skip_bias:
        raise ValueError("a skip bias belongs to the skipcal pick mode")
    return f"{PREFIX}:lambda={format_lambda(lam)}:pi_h={weights_sha256[:8]}{pick_name(pick, skip_bias)}"


def is_blended(name: str | None) -> bool:
    return bool(name) and str(name).startswith(PREFIX)


def parse_macro(name: str) -> tuple[float, str]:
    """(lambda, the 8 hex of the prior's sha256) of a blended recipe string (the pick mode, if any, is `parse_pick`'s)."""
    m = _MACRO.fullmatch(name)
    if not m:
        raise ValueError(f"{name!r} is not a {PREFIX}:lambda=<inf|number>:pi_h=<8 hex>[:pick=<mode>] recipe")
    return (math.inf if m[1] == "inf" else float(m[1])), m[2]


def parse_pick(name: str) -> tuple[str, float]:
    """(mode, skip bias) of a blended recipe string: ("argmax", 0.0) without a `pick=` part."""
    m = _MACRO.fullmatch(name)
    if not m:
        raise ValueError(f"{name!r} is not a {PREFIX}:lambda=<inf|number>:pi_h=<8 hex>[:pick=<mode>] recipe")
    p = m[3]
    if p is None:
        return PICK_ARGMAX, 0.0
    if p == PICK_SAMPLE:
        return PICK_SAMPLE, 0.0
    return PICK_SKIPCAL, float(p[len(PICK_SKIPCAL):])


# ------------------------------------------------------------------------------------------------------------------ search identity


def _mult(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_MULT:
        raise ValueError(f"the stakes multiplier of {what} is a whole number from 1 to {MAX_MULT}, not {value!r}")
    return value


def stakes_mult(mult) -> dict[str, int]:
    """A validated {"hall", "elite", "boss"} multiplier dict from a dict or a (hall, elite, boss) sequence."""
    if isinstance(mult, dict):
        extra = set(mult) - set(FIGHT_KINDS)
        if extra:
            raise ValueError(f"a stakes schedule names the fight kinds {', '.join(FIGHT_KINDS)}, not {sorted(extra)}")
        return {k: _mult(mult.get(k, 1), k) for k in FIGHT_KINDS}
    seq = list(mult)
    if len(seq) != len(FIGHT_KINDS):
        raise ValueError("a stakes schedule is three multipliers: hall, elite, boss")
    return {k: _mult(v, k) for k, v in zip(FIGHT_KINDS, seq, strict=True)}


def stakes_tag(base: str, mult) -> str:
    """`<base>@h<hall>e<elite>b<boss>`: the base search config and the multiplier of each fight kind."""
    m = stakes_mult(mult)
    if not re.fullmatch(_BASE, base):
        raise ValueError(f"{base!r} is not a search config name")
    return f"{base}@h{m['hall']}e{m['elite']}b{m['boss']}"


def parse_stakes_tag(tag: str) -> tuple[str, dict[str, int]]:
    m = _STAKES.fullmatch(tag)
    if not m:
        raise ValueError(f"{tag!r} is not a <base>@h<hall>e<elite>b<boss> stakes schedule")
    return m[1], stakes_mult([int(m[2]), int(m[3]), int(m[4])])


def hold_pct(value) -> int:
    """The hold's threshold as a whole number of per cent of max HP (1 to 99)."""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 99:
        raise ValueError(f"the potion hold's threshold is a whole number of per cent of max HP from 1 to 99, not {value!r}")
    return value


def config_name(base: str, mult=None, hold: int | None = None) -> str:
    """The name of a search config with a stakes schedule and/or a potion hold: `a3-v1@h1e4b16+hold30`. With neither it is `base`."""
    if mult is None and not re.fullmatch(_BASE, base):
        raise ValueError(f"{base!r} is not a search config name")
    name = stakes_tag(base, mult) if mult is not None else base
    return name + (f"+hold{hold_pct(hold)}" if hold is not None else "")


def parse_config_name(name: str) -> tuple[str, dict[str, int] | None, int | None]:
    """(base, multipliers or None, hold per cent or None) of a config name; a plain name is (name, None, None). Not a name of this
    form (a `flat-N`, say, parses as base `flat-N`): the caller decides whether the base exists."""
    m = _CONFIG.fullmatch(name)
    if not m:
        raise ValueError(f"{name!r} is not a search config name")
    mult = stakes_mult([int(m[2]), int(m[3]), int(m[4])]) if m[2] is not None else None
    return m[1], mult, (hold_pct(int(m[5])) if m[5] is not None else None)


def recipe_of(macro: str | None) -> str:
    """The ledger's recipe of a macro: "" for the plain macros (rules, vrun, none named), the string itself for a blended one."""
    return str(macro) if is_blended(macro) else ""


def full_recipe(macro: str | None, stakes: str | None = None, hold: int | None = None) -> str:
    """The ledger's recipe of an agent: its macro's, then `+stakes:<tag>` and `+hold:<per cent>` where it has them. An agent with
    neither is `recipe_of(macro)` (so every agent that existed before the two parts keeps its recipe). A plain macro with a part
    names itself first (`vrun+stakes:a3-v1@h1e4b16`)."""
    if stakes is None and hold is None:
        return recipe_of(macro)
    base = str(macro) if macro else "rules"
    if stakes is not None:
        parse_stakes_tag(stakes)
        base += f"+stakes:{stakes}"
    if hold is not None:
        base += f"+hold:{hold_pct(hold)}"
    return base


def split_recipe(recipe: str) -> tuple[str, str | None, int | None]:
    """The inverse of `full_recipe`: (macro, stakes tag or None, hold per cent or None). A "" recipe is ("", None, None)."""
    stakes = _RECIPE_STAKES.search(recipe)
    hold = _RECIPE_HOLD.search(recipe)
    cut = min((m.start() for m in (stakes, hold) if m), default=len(recipe))
    return recipe[:cut], (stakes[1] if stakes else None), (int(hold[1]) if hold else None)
