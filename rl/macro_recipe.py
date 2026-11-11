"""The names of the run chooser's recipes (strings only; no numpy, no native code).

A candidate's agent.json `macro` is `rules`, `vrun` or a blended recipe:

    vrun+pi_h:lambda=<inf|number>:pi_h=<8 hex of the human prior's weights sha256>[:act<N>=<inf|number>...][:pick=sample|:pick=skipcal<bias>]

(`imitation_chooser`: the run judgement V plus lambda times the log of the human prior pi_h on the deck decisions). The optional last
part says how the deck screen's option is taken: absent, the argmax of V + lambda log pi_h; `pick=sample`, a draw from
softmax((V + lambda log pi_h) / 1) seeded by the run seed and the floor; `pick=skipcal<bias>`, the argmax with `bias` floors added to
the skip option's score (the bias a calibration run chose, `rl/skip_calibrate.py`). The optional `act<N>=<lambda>` parts (N 1 to 3, in
that order, at most one a act, none equal to the base lambda) give an act its own lambda in place of `lambda=`: `lambda=0.5:...:act1=0`
decides Act 1's deck screens by V alone (lambda 0 is `vrun`) and every later act by V plus 0.5 log pi_h (2026-11-08, D55 addendum 12; the
act of a decision is its floor's, `act_of_floor`). A recipe without the parts is the recipe it always was, byte for byte. The string is the recipe's identity wherever the lab
compares agents: an emulator arm's summary names it (`gen0.emu_arm_match`), and the ledger counts promotion attempts per recipe
(`gen_loop.attempts_before`): the plain macros (`rules`, `vrun`) are one recipe, written "" in the ledger's `macro` column and in every
row from before the column; each blended string is a recipe of its own.

The run student `pi_s` (`run_student`, D56/D57 part 3; 2026-11-11) adds a recipe of its own beside the blended one:

    vrun+pi_s:lambda=<inf|number>:pi_h=<8 hex>[:act<N>=<lambda>...]:pi_s=<8 hex of the student's weights' sha256>:<kind>=<l1>/<l2>/<l3>...[:pick=...]

It is the blended recipe above (the human prior's blend on every deck screen the student does not take) with the student named after
it, and one `<kind>=<l1>/<l2>/<l3>` part for each decision kind the student has taken over (`STUDENT_KINDS`, in that order, once each):
the lambda of `V + lambda log pi_s` in Acts 1, 2 and 3 (lambda 0 is `vrun` for that kind and act, `inf` the student alone). A recipe
without the parts is the recipe it always was, byte for byte; `parse_macro`, `parse_act_lambdas` and `parse_pick` read a student
recipe's blended part exactly as they read the plain one's.

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
from collections.abc import Mapping, Sequence

PREFIX = "vrun+pi_h"
STUDENT_PREFIX = "vrun+pi_s"
# the decision kinds a student can take over (`rl/run_student.py`), in the order a recipe writes them
STUDENT_KINDS = ("card", "ancient", "rest", "route", "shop", "event", "reward", "select")
_LAMBDA = r"(?:inf|[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)"
_MACRO = re.compile(
    r"vrun\+pi_h:lambda=(" + _LAMBDA + r"):pi_h=([0-9a-f]{8})((?::act[1-3]=" + _LAMBDA + r")*)"
    r"(?::pick=(sample|skipcal-?[0-9]*\.?[0-9]+))?")
_ACT_PART = re.compile(r":act([1-3])=(" + _LAMBDA + r")")
_KIND_ALT = "|".join(STUDENT_KINDS)
_STUDENT = re.compile(
    r"vrun\+pi_s(:lambda=" + _LAMBDA + r":pi_h=[0-9a-f]{8}(?::act[1-3]=" + _LAMBDA + r")*)"
    r":pi_s=([0-9a-f]{8})((?::(?:" + _KIND_ALT + r")=" + _LAMBDA + "/" + _LAMBDA + "/" + _LAMBDA + r")+)"
    r"((?::pick=(?:sample|skipcal-?[0-9]*\.?[0-9]+))?)")
_STUDENT_KIND = re.compile(r":(" + _KIND_ALT + r")=(" + _LAMBDA + ")/(" + _LAMBDA + ")/(" + _LAMBDA + ")")
ACT_LAST_FLOORS = (17, 33)  # the boss floors of Acts 1 and 2 (run_fields.ACT_BOSS_FLOORS): a floor past them is in the next act
ACTS = 3
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


def act_part(acts: Mapping[int, float] | None, lam: float) -> str:
    """The recipe's `:act<N>=<lambda>` parts for a per-act lambda map (acts numbered from 1): in act order, and an act whose lambda is the
    base `lam` is not written (the recipe would name the same chooser twice). "" for none."""
    out = []
    for act in sorted(acts or {}):
        if isinstance(act, bool) or act not in range(1, ACTS + 1):
            raise ValueError(f"a per-act lambda names an act from 1 to {ACTS}, not {act!r}")
        value = float(acts[act])
        if not value >= 0:
            raise ValueError(f"lambda is 0 or more (inf is the prior alone), not {value} for act {act}")
        if value != lam:
            out.append(f":act{act}={format_lambda(value)}")
    return "".join(out)


def macro_name(lam: float, weights_sha256: str, pick: str = PICK_ARGMAX, skip_bias: float = 0.0,
               acts: Mapping[int, float] | None = None) -> str:
    """The recipe string a candidate's agent.json carries as `macro`. `acts` ({act number from 1: lambda}) gives an act its own lambda."""
    if not lam >= 0:
        raise ValueError(f"lambda is 0 or more (inf is the prior alone), not {lam}")
    if not re.fullmatch(r"[0-9a-f]{8,64}", weights_sha256):
        raise ValueError("the prior's sha256 is lowercase hex")
    if pick != PICK_SKIPCAL and skip_bias:
        raise ValueError("a skip bias belongs to the skipcal pick mode")
    return f"{PREFIX}:lambda={format_lambda(lam)}:pi_h={weights_sha256[:8]}{act_part(acts, lam)}{pick_name(pick, skip_bias)}"


def is_blended(name: str | None) -> bool:
    """A recipe of the blended chooser: the human prior's (`vrun+pi_h`) or the student's (`vrun+pi_s`, which keeps that blend on the
    screens it has not taken over)."""
    return bool(name) and str(name).startswith((PREFIX, STUDENT_PREFIX))


def is_student(name: str | None) -> bool:
    return bool(name) and str(name).startswith(STUDENT_PREFIX)


def _parse_lambda(text: str) -> float:
    return math.inf if text == "inf" else float(text)


def blended_of(name: str) -> str:
    """The human-prior recipe a recipe is built on: a `vrun+pi_h` recipe is itself; a student recipe's is the same blend with the student's
    parts left out (and its `pick=` part kept). ValueError for a string of neither form."""
    if not is_student(name):
        return name
    m = _STUDENT.fullmatch(name)
    if not m:
        raise ValueError(f"{name!r} is not a {STUDENT_PREFIX}:lambda=<..>:pi_h=<8 hex>[:act<N>=<..>]:pi_s=<8 hex>:<kind>=<l1>/<l2>/<l3>... recipe")
    return PREFIX + m[1] + m[4]


def _match(name: str) -> re.Match:
    m = _MACRO.fullmatch(blended_of(name))
    if not m:
        raise ValueError(f"{name!r} is not a {PREFIX}:lambda=<inf|number>:pi_h=<8 hex>[:act<N>=<lambda>][:pick=<mode>] recipe")
    return m


def parse_macro(name: str) -> tuple[float, str]:
    """(lambda, the 8 hex of the prior's sha256) of a blended recipe string (the per-act lambdas are `parse_act_lambdas`'s, the pick
    mode, if any, `parse_pick`'s)."""
    m = _match(name)
    parse_act_lambdas(name)  # a recipe that is not in its one canonical form is refused here too
    return _parse_lambda(m[1]), m[2]


def parse_act_lambdas(name: str) -> dict[int, float]:
    """{act number from 1: lambda} of the `:act<N>=<lambda>` parts of a blended recipe string ({} without any). One canonical form per
    chooser: the acts in order, none twice, none equal to the base lambda (`act_part` writes no such part)."""
    m = _match(name)
    base, found = _parse_lambda(m[1]), _ACT_PART.findall(m[3] or "")
    acts = [int(a) for a, _ in found]
    if acts != sorted(set(acts)):
        raise ValueError(f"{name!r}: the act<N> parts come once each, in act order")
    out = {int(a): _parse_lambda(v) for a, v in found}
    if any(v == base for v in out.values()):
        raise ValueError(f"{name!r}: an act<N> part equal to the base lambda names the same chooser twice; leave it out")
    return out


def student_part(sha8_or_sha: str, kinds: Mapping[str, Sequence[float]]) -> str:
    """The student's recipe parts: `:pi_s=<8 hex>` and one `:<kind>=<l1>/<l2>/<l3>` for each kind in `kinds` (the lambda of the student in
    Acts 1, 2 and 3: 0 or more, `inf` allowed), in `STUDENT_KINDS` order. At least one kind."""
    if not re.fullmatch(r"[0-9a-f]{8,64}", sha8_or_sha):
        raise ValueError("the student's sha256 is lowercase hex")
    if not kinds:
        raise ValueError("a student recipe names at least one kind")
    bad = [k for k in kinds if k not in STUDENT_KINDS]
    if bad:
        raise ValueError(f"a student takes over the kinds {', '.join(STUDENT_KINDS)}, not {bad}")
    out = [f":pi_s={sha8_or_sha[:8]}"]
    for kind in STUDENT_KINDS:
        if kind not in kinds:
            continue
        lams = [float(x) for x in kinds[kind]]
        if len(lams) != ACTS or not all(x >= 0 for x in lams):
            raise ValueError(f"the student's lambda for {kind} is three numbers (Acts 1 to 3), each 0 or more, not {kinds[kind]!r}")
        out.append(f":{kind}=" + "/".join(format_lambda(x) for x in lams))
    return "".join(out)


def student_name(lam: float, pi_h_sha256: str, pi_s_sha256: str, kinds: Mapping[str, Sequence[float]], pick: str = PICK_ARGMAX,
                 skip_bias: float = 0.0, acts: Mapping[int, float] | None = None) -> str:
    """The recipe string of a student candidate: the human prior's blend (`macro_name`, its `acts`) and the student's parts, then the pick."""
    base = macro_name(lam, pi_h_sha256, PICK_ARGMAX, 0.0, acts)
    return STUDENT_PREFIX + base[len(PREFIX):] + student_part(pi_s_sha256, kinds) + pick_name(pick, skip_bias)


def parse_student(name: str) -> tuple[str, dict[str, tuple[float, float, float]]] | None:
    """(the 8 hex of the student's sha256, {kind: (lambda in Act 1, Act 2, Act 3)}) of a student recipe; None for any other recipe. One
    canonical form: the kinds in `STUDENT_KINDS` order, each once, and the per-act lambdas of the human prior's part as `parse_act_lambdas`
    reads them."""
    if not is_student(name):
        return None
    m = _STUDENT.fullmatch(name)
    if not m:
        blended_of(name)  # the refusal's message
    parse_act_lambdas(name)
    found = _STUDENT_KIND.findall(m[3])
    kinds = [k for k, *_ in found]
    if kinds != [k for k in STUDENT_KINDS if k in set(kinds)] or len(set(kinds)) != len(kinds):
        raise ValueError(f"{name!r}: the student's kinds come once each, in the order {', '.join(STUDENT_KINDS)}")
    return m[2], {k: (_parse_lambda(a), _parse_lambda(b), _parse_lambda(c)) for k, a, b, c in found}


def student_lambda_for_floor(lams: Sequence[float], floor: int) -> float:
    """The student's lambda for a kind on `floor`: the entry of the floor's act (`act_of_floor`)."""
    return float(lams[min(act_of_floor(floor), ACTS - 1)])


def parse_pick(name: str) -> tuple[str, float]:
    """(mode, skip bias) of a blended recipe string: ("argmax", 0.0) without a `pick=` part."""
    m = _match(name)
    p = m[4]
    if p is None:
        return PICK_ARGMAX, 0.0
    if p == PICK_SAMPLE:
        return PICK_SAMPLE, 0.0
    return PICK_SKIPCAL, float(p[len(PICK_SKIPCAL):])


def act_of_floor(floor: int) -> int:
    """The 0-based act of a floor (the floors count across acts, `run_fields.ACT_BOSS_FLOORS`): floors up to the Act 1 boss's (17) are
    Act 1, up to the Act 2 boss's (33) Act 2, later ones Act 3. The Act 2 ancient, the first decision of its act, is floor 18."""
    return sum(int(floor) > last for last in ACT_LAST_FLOORS)


def lambda_for_floor(lam: float, acts: Mapping[int, float] | None, floor: int) -> float:
    """The lambda a deck screen on `floor` is decided with: the act's own when the recipe names one, else the base."""
    return float((acts or {}).get(act_of_floor(floor) + 1, lam))


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
