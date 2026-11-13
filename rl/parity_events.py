"""Block P3 of the information-parity side blocks: what an event option pays (D59 lane D, `docs/rl/HUMAN_PRIOR.md` "v4").

A human reads an event's options with their effects ("Lose 6 HP. Gain 55 gold", "Obtain a random relic", "Remove a card"). The run encoder holds an event option as a slot
(event id x option index x step in block B) and leaves what it pays to be learned from the few times each is chosen. This module measures it on the engine: at every event
decision of many random runs (fights skipped with the engine's resolve hook, so the sample reaches all three acts quickly; each event is walked into at a random 45 to 100 in 100 of its max HP and 30 to 300 gold) a **clone of the run takes each option once** and the
one-step change is read off: HP, max HP, gold, deck size, relics, curses, potions, and where the option leads (a fight, a card reward, a relic reward, a deck selection to
remove, upgrade or transform a card, the event's next page). The mean over the sample is the table `rl/parity_data/event_payoffs.json`, keyed by the option the run encoder
already names, `(event id, option index, step)`.

The table, not a clone at play, is the encoding: the teacher's stored runs and the human records carry no live engine to clone, and a table is the same number at train and at
play. A key the sample never reached has `known` 0 and zeros. A side block: `BASE_SIZE` and the feature version are untouched.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DELTAS = ("hp", "max_hp", "gold", "deck", "relics", "curses", "potions")
# where the option leads (the share of the sample's clones that end there): a fight, a card reward, a relic reward, a deck selection (remove / upgrade or enchant /
# transform or duplicate), a bundle or grid, the event's next page, and the option being refused (locked: the sample could not take it)
LEADS = ("fight", "card_reward", "relic_reward", "select_remove", "select_upgrade", "select_transform", "other_screen", "continues", "locked")
SCALES = {"hp": 30.0, "max_hp": 20.0, "gold": 250.0, "deck": 3.0, "relics": 2.0, "curses": 2.0, "potions": 2.0}
EVENT_WIDTH = 1 + len(DELTAS) + len(LEADS)  # known, the deltas, the leads
TABLE_FILE = Path(__file__).resolve().parent / "parity_data" / "event_payoffs.json"
# RunPhase numbers (`sts2_gym.run_constants`) and DeckSelection numbers (`Core/Run/RunState.cs`) the leads are read from
_SELECT_KIND = {1: "select_upgrade", 4: "select_upgrade", 7: "select_remove", 8: "select_remove", 9: "select_remove", 2: "select_transform", 3: "select_transform",
                5: "select_transform", 6: "select_transform", 10: "select_transform"}


def _snap(session) -> dict[str, int]:  # noqa: ANN001
    from run_train import vocab  # noqa: PLC0415

    v = session.view()
    return {"hp": v.hp, "max_hp": v.max_hp, "gold": v.gold, "deck": len(v.deck), "relics": len(v.relics),
            "curses": sum(1 for c in v.deck if vocab.card(c.card_id).card_type == "Curse"), "potions": len(v.potions)}


def _lead_of(session) -> str:  # noqa: ANN001
    """Where a clone stands after taking an option: one of LEADS (the phase, and for a deck selection what it is for)."""
    from sts2_gym import native  # noqa: PLC0415
    from sts2_gym import run_constants as rc  # noqa: PLC0415

    phase = session.phase
    if phase == rc.PHASE_COMBAT:
        return "fight"
    if phase == rc.PHASE_CARD_REWARD:
        return "card_reward"
    if phase == rc.PHASE_RELIC_REWARD:
        return "relic_reward"
    if phase == rc.PHASE_EVENT:
        # the page that shows the outcome (the engine's `EVENT_RESULT_PENDING`, event id -1) is the end of the event, as the engine's own session steps past it by itself
        return "" if int(native.run_info(session.handle)[9]) == -1 else "continues"
    if phase == rc.PHASE_TRANSFORM_SELECT:
        kind = native.run_state_list(session.handle, 20, 3)
        return _SELECT_KIND.get(int(kind[0]) if kind else 0, "other_screen")
    if phase in (rc.PHASE_MAP, rc.PHASE_COMPLETE):
        return ""  # the event is over: the deltas say all
    return "other_screen"


HP_FRACTION = (0.45, 1.0)  # a run walks into its events with this share of its max HP (the sample's runs resolve fights without damage, so every run would be at full HP)
GOLD_RANGE = (30, 300)


def _typical_state(session, rng: random.Random) -> None:  # noqa: ANN001
    """Give the run, at the first page of an event, the HP and gold a played run walks in with (uniform over the ranges above), so that an option that heals or costs gold shows what it does at a
    typical state and a gold-locked option is refused at a typical rate. The run goes on from the changed state."""
    from sts2_gym import native  # noqa: PLC0415

    info = session.info
    max_hp = int(info["max_hp"])
    hp = max(1, int(max_hp * (HP_FRACTION[0] + (HP_FRACTION[1] - HP_FRACTION[0]) * rng.random())))
    obs = session.env._run_obs_buf  # noqa: SLF001
    native.run_debug_set_hp(session.handle, hp, max_hp, obs)
    native.run_debug_set_player(session.handle, obs, gold=rng.randint(*GOLD_RANGE))


def sample_run(seed: str, rng: random.Random, out: list, max_steps: int = 4000) -> None:
    """Play one random run (fights resolved by the engine's hook, "?" rooms preferred) and, at each event decision, take every option on a clone. Appends
    `(event, deltas | None, lead)` rows to `out` (`deltas` None: the clone refused the option)."""
    import hindsight  # noqa: PLC0415
    from run_train import vocab  # noqa: PLC0415
    from run_train.engine import RunSession  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415
    from sts2_gym import run_constants as rc  # noqa: PLC0415

    s = RunSession(seed, 10)
    try:
        for _ in range(max_steps):
            phase = s.phase
            if phase == rc.PHASE_COMPLETE:
                break
            if phase == rc.PHASE_COMBAT:
                native.run_debug_resolve_combat(s.handle, s.env._run_obs_buf)  # noqa: SLF001
                continue
            d = s.decision()
            if d is None:
                if s.phase == rc.PHASE_COMPLETE:
                    break
                continue
            if d.type == "event":
                if any(o.event is not None and o.event[2] == 0 for o in d.options):
                    _typical_state(s, rng)
                    d = s.decision() or d  # the page's options as the changed state leaves them (a gold price may now lock one)
                before = _snap(s)
                for opt in d.options:
                    if opt.kind != "event.option" or opt.event is None:
                        continue
                    c = hindsight.clone_session(s, rng.randrange(1 << 30))
                    try:
                        try:
                            c.apply(opt)
                        except Exception:  # noqa: BLE001 -- a refused option (a lock the mask did not carry)
                            out.append((tuple(opt.event), None, "locked"))
                            continue
                        if c.phase == rc.PHASE_COMPLETE:
                            after = dict(before, hp=0)
                        else:
                            after = _snap(c)
                        out.append((tuple(opt.event), {k: after[k] - before[k] for k in DELTAS}, _lead_of(c)))
                    finally:
                        c.close()
            opts = list(d.options)
            if d.type == "map":
                unknown = [o for o in opts if o.node and o.node[2] == vocab.NODE_UNKNOWN]
                pick = rng.choice(unknown or opts)
            else:
                pick = rng.choice(opts)
            s.apply(pick)
    finally:
        s.close()


def sample(first: int, count: int, namespace: str = "PEV") -> list:
    """The records of runs `first` .. `first + count - 1` of the sample's seed namespace (deterministic)."""
    out: list = []
    for i in range(first, first + count):
        sample_run(f"{namespace}{i:07d}", random.Random(i), out)
    return out


def build_table(records) -> dict[str, dict]:  # noqa: ANN001
    """Key `"event:option:step"` -> {n, mean deltas, std of the deltas, lead shares}, from the sample's records."""
    by: dict[tuple, list] = defaultdict(list)
    for key, deltas, lead in records:
        by[tuple(key)].append((deltas, lead))
    table: dict[str, dict] = {}
    for key, rows in sorted(by.items()):
        n = len(rows)
        taken = [d for d, _ in rows if d is not None]
        row: dict = {"n": n, "n_taken": len(taken)}
        for name in DELTAS:
            xs = np.array([d[name] for d in taken], dtype=np.float64) if taken else np.zeros(1)
            row[name] = round(float(xs.mean()), 3)
            row[name + "_sd"] = round(float(xs.std()), 3)
        for lead in LEADS:
            row["lead_" + lead] = round(sum(1 for _, ld in rows if ld == lead) / n, 4)
        table[f"{key[0]}:{key[1]}:{key[2]}"] = row
    return table


@cache
def payoff_table() -> dict[tuple[int, int, int], dict]:
    raw = json.loads(TABLE_FILE.read_text(encoding="utf-8"))
    return {tuple(int(x) for x in k.split(":")): v for k, v in raw["options"].items()}


def event_payoff(event) -> np.ndarray:  # noqa: ANN001
    """The payoff vector of one event option `(event id, option index, step)`: known (1 when the sample reached it), the mean deltas over their scales (clipped to [-1, 1]) and
    the lead shares. A key the table lacks is all zeros with known 0. float32 (EVENT_WIDTH,)."""
    out = np.zeros(EVENT_WIDTH, dtype=np.float32)
    if event is None:
        return out
    row = payoff_table().get((int(event[0]), int(event[1]), int(event[2])))
    if row is None:
        return out
    out[0] = 1.0
    for i, name in enumerate(DELTAS):
        out[1 + i] = max(-1.0, min(1.0, row[name] / SCALES[name]))
    for i, lead in enumerate(LEADS):
        out[1 + len(DELTAS) + i] = row["lead_" + lead]
    return out


def main() -> None:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "show"])
    ap.add_argument("--runs", type=int, default=3000)
    ap.add_argument("--first", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "build":
        recs = sample(a.first, a.runs)
        table = build_table(recs)
        doc = {"runs": a.runs, "first_seed": a.first, "records": len(recs), "scales": SCALES, "deltas": list(DELTAS), "leads": list(LEADS), "options": table}
        TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        TABLE_FILE.write_text(json.dumps(doc, indent=0, sort_keys=True), encoding="utf-8")
        events = {k.split(":")[0] for k in table}
        print(f"wrote {TABLE_FILE.name}: {len(recs)} option clones over {a.runs} runs, {len(table)} options of {len(events)} events")
    else:
        for k, v in sorted(payoff_table().items()):
            print(k, v["n"], {n: v[n] for n in DELTAS if v[n]}, {n[5:]: v[n] for n in v if n.startswith("lead_") and v[n]})


if __name__ == "__main__":
    main()
