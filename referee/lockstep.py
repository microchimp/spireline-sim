"""Lockstep: the emulator replays the real game's decisions, and the PLAY agent searches from the emulator.

Round A1 (PLAN.md section 5 "The bridge, decided"). One `Lockstep` per run:

  * the emulator starts from the same seed as the referee and replays every decision the referee takes;
  * at each referee decision the agent picks from the EMULATOR's lockstep state: combat by IS-MCTS
    (rl/live_search.search_live; every simulation forks the position with resampling, so the search never
    reads the lockstep state's RNG counters or draw order), everything else by the rules policy;
  * the referee's observation and the emulator's are projected onto the same agent-visible fields and
    hashed; a difference is logged as an agent-weighted divergence record and, where a re-anchor exists
    (fight start, room entry), the emulator is re-anchored from the referee's record; otherwise the run goes
    on on the referee alone and every later decision is scored but not ranked (`ranked: false`).

The pure parts (projection, divergence log, re-anchor plan, the loop over a message stream) import no native
code, so they run against a fake referee stream; `EmuLock` is the real emulator side and imports the fork's
rl/ and sts2_gym lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]

COMBAT_KINDS = ("combat", "select_cards_combat", "card_reward_combat")
# Referee decision kinds the emulator side mirrors. Any other kind suspends the lockstep (reason unmapped:<kind>) until
# the next re-anchor point.
MAPPED_KINDS = ("neow", "ancient", "map", "card_reward", "rest", "reward", "select_cards", "treasure", "shop",
                "bundle", "event") + COMBAT_KINDS
# Reasons that are a gap in this adapter's mapping, not a difference of the emulator's: reported apart.
GAP_PREFIXES = ("unmapped", "apply:", "choose_error")


def _paths() -> None:
    for p in (ROOT / "rl", ROOT / "src", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


# Referee actions beyond a screen's own options (driver/Env.cs, RunPlay.cs), and what each maps to on the emulator side:
#   "none"                   a card selection that allows 0: the emulator's select.offer_done where it has one
#   "done"                   the end of a fight card screen taken with full_select (driver/Env.cs): the emulator's decline,
#                            the action one past its candidates; the screen's picks are its candidates
#   "skip"                   a card reward's Skip: card.skip
#   "alt:<id>"               any other card-reward alternative (reroll, a relic's): no emulator option
#   ["use_potion", s, id]    a potion used outside a fight: no emulator option
#   ["discard_potion", s, id] a potion discarded outside a fight: no emulator option
# An action with no emulator option is applied as `no_emulator_option:...`, which suspends the lockstep like any other loss.
NEW_ACTION_KINDS = ("none", "done", "card_alt", "potion_use", "potion_discard")
NO_EMULATOR_KINDS = ("card_alt", "potion_use", "potion_discard")


def extra_kind(label: Any) -> str | None:
    """Which of the extra referee actions a label is (see above), else None."""
    if label == "none":
        return "none"
    if label == "done":
        return "done"
    if isinstance(label, str) and label.startswith("alt:"):
        return "card_alt"
    if isinstance(label, list) and label and label[0] == "use_potion":
        return "potion_use"
    if isinstance(label, list) and label and label[0] == "discard_potion":
        return "potion_discard"
    return None


def character_supported(character: str, native=None) -> str | None:
    """None when the emulator can start `character`; else why not. The native run API starts Ironclad unless the library
    binding has run_reset_character (Sts2Run_Reset takes no character)."""
    if character == "IRONCLAD":
        return None
    if native is None:
        _paths()
        import leakprobes  # noqa: PLC0415

        leakprobes.native_env()
        from sts2_gym import native  # noqa: PLC0415
    if hasattr(native, "run_reset_character"):
        return None
    return (f"--character {character}: the native emulator library has no run_reset_character (Sts2Run_Reset takes no character), "
            f"so the emulator side cannot start {character}. Use IRONCLAD, or add the native export first.")


def require_character(character: str, native=None) -> None:
    """Exit at once (before any worker starts) when the emulator cannot start `character`."""
    why = character_supported(character, native)
    if why:
        raise SystemExit(why)


# ------------------------------------------------------------------ projection and hashes


def emu_enemy_hp(obs, offset: int, slot: int, slots: int) -> list:
    """The HP of every enemy the emulator's roster holds, from its observation buffer, in roster order. A slot holds an
    enemy when its HP or max HP is above 0: a dead enemy the roster still holds (the game keeps one that dies in its
    own move, or one a power keeps standing) lists at 0, like the referee's `combat.enemies`, and a slot past the end of
    the roster (all zeros) lists nothing. Both sides list what they hold, so a corpse only one side holds is a diff."""
    return [int(obs[offset + k * slot]) for k in range(slots)
            if obs[offset + k * slot] > 0 or obs[offset + k * slot + 1] > 0]


def project(view: dict) -> dict:
    """The agent-visible fields both sides share, canonical (multisets sorted). Run level, plus the hand and
    the enemies' HP in a fight (every enemy the side holds, dead ones at 0). Nothing here is a seed, a counter or the
    draw order."""
    out = {k: view.get(k) for k in ("act", "floor", "hp", "max_hp", "gold")}
    for k in ("deck", "relics", "potions"):
        out[k] = sorted(json.dumps(x) for x in view.get(k, []) if x is not None)
    c = view.get("combat")
    if c:
        out["combat"] = {"hand": sorted(json.dumps(x) for x in c.get("hand", [])),
                         "enemy_hp": [h for h in c.get("enemy_hp", [])]}
    return out


def agent_rng(entropy):
    """The agent's own RNG: every search's resample seeds come from it. `entropy` is the agent seed (a list of ints,
    e.g. [base agent seed, run index]); nothing of the referee's game seed may enter here (DESIGN 3.2: the hidden
    parts never come from the game's seed)."""
    import numpy  # noqa: PLC0415
    return numpy.random.default_rng(list(entropy))


def fresh_agent_seed() -> int:
    """A base agent seed unrelated to any game seed; the run record keeps it so a run can be replayed."""
    return int.from_bytes(os.urandom(6), "big")


def run_outcome(m: dict, done: bool) -> dict:
    """The run's outcome from the referee's end message: class win/death/crash/truncated, the end string, final floor."""
    end = m.get("end")
    cls = "truncated" if not done else "win" if str(end).lower() in ("win", "victory", "won") else \
        "crash" if str(end).lower() == "crash" else "death"
    return {"class": cls, "end": end, "floor": m.get("floor")}


def view_hash(view: dict) -> str:
    return hashlib.sha256(json.dumps(project(view), sort_keys=True).encode()).hexdigest()[:16]


def structural_diff(ref: dict, emu: dict) -> list:
    """(path, referee value, emulator value) for every field of the projections that differs. The lab's
    comparer (tools.fidelity.fight_compare.diff) when it imports, else the same walk here."""
    a, b = project(ref), project(emu)
    try:
        _paths()
        from tools.fidelity.fight_compare import diff  # noqa: PLC0415
        return [list(d) for d in diff(a, b)]
    except Exception:  # noqa: BLE001 -- the comparer is a nicety; the fallback is the same walk
        return [[f".{k}", a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def ref_view(obs: dict) -> dict:
    """The referee's `obs` (driver/Env.cs Obs) as a view. Card entries are [entry, upgraded, ...]."""
    v = {"act": obs.get("act"), "floor": obs.get("floor"), "hp": obs.get("hp"), "max_hp": obs.get("max_hp"),
         "gold": obs.get("gold"), "deck": [[c[0], int(c[1])] for c in obs.get("deck", [])],
         "relics": list(obs.get("relics", [])), "potions": [p for p in obs.get("potions", []) if p]}
    c = obs.get("combat")
    if c:
        hp = []
        for e in c.get("enemies", []) or []:
            if isinstance(e, dict):
                hp.append(e.get("hp", e.get("current_hp")))
        v["combat"] = {"hand": [[h[0], int(h[1])] for h in c.get("hand", [])], "enemy_hp": hp}
    return v


# ------------------------------------------------------------------ divergence log and re-anchor plan


def mid_screen_pick(kind: str, msg: dict) -> bool:
    """A further pick of a fight card screen the referee has already taken a card from (driver ctx pick > 1). The referee
    moves the picked cards only when the screen closes, the emulator after each pick, so the two hands are not comparable
    until then: EmuLock.sync compares the screen's candidates instead, and the views meet again at the next decision."""
    ctx = msg.get("ctx") if isinstance(msg, dict) else None
    return kind == "select_cards_combat" and isinstance(ctx, dict) and int(ctx.get("pick", 1)) > 1


def act_type(kind: str, label: Any) -> str:
    """What a chosen action was: a card play, an end turn, a potion (in a fight or out of one), else a choice on a
    screen (a map node, a reward, a card selection, an event option)."""
    if kind == "combat" and isinstance(label, list) and label:
        return {"play": "play", "end": "end_turn", "potion": "potion"}.get(label[0], "choice")
    return "potion" if extra_kind(label) in ("potion_use", "potion_discard") else "choice"


def reanchor_plan(kind: str, prev_kind: str | None) -> str | None:
    """Where a re-anchor can happen: the first decision of a fight (fight start) or a map choice (room entry).
    Mid-fight and mid-screen divergences wait for the next boundary."""
    if kind in COMBAT_KINDS and prev_kind not in COMBAT_KINDS:
        return "fight_start"
    if kind == "map":
        return "room_entry"
    return None


@dataclass
class DivergenceLog:
    records: list = field(default_factory=list)

    def add(self, **rec: Any) -> dict:
        self.records.append(rec)
        return rec


class EmuSide(Protocol):
    """What the loop needs of the emulator side (EmuLock, or a fake in the tests)."""

    def sync(self, kind: str, labels: list, msg: dict) -> str | None: ...  # a reason when the structure disagrees
    def view(self) -> dict: ...
    def choose(self, kind: str, labels: list, msg: dict) -> int | None: ...
    def apply(self, kind: str, labels: list, idx: int) -> str | None: ...
    def reanchor(self, plan: str, msg: dict) -> bool: ...


class Lockstep:
    """The per-run state machine. `decide(msg)` answers one referee decision with an action index.

    A difference (hash, structure, or a decision the adapter cannot mirror) suspends the lockstep: the decisions after
    it are scored but not ranked, and at every later re-anchor point (fight start, room entry) the emulator is
    re-anchored from the referee's record and, once its view equals the referee's again, the lockstep resumes there.
    """

    def __init__(self, emu: EmuSide | None, fallback, seed: str = "", sims: int = 60,
                 agent_seed=None):
        self.emu, self.fallback, self.seed, self.sims = emu, fallback, seed, sims
        self.agent_seed = agent_seed
        self.open: dict | None = None  # the record of the loss now in force; its not-ranked span grows each decision
        self.log = DivergenceLog()
        self.lost: str | None = None if emu is not None else "no_emulator"
        self.n = 0
        self.assumed: set[int] = set()  # decisions whose search saw enemy moves it could not re-roll
        self.prev_kind: str | None = None
        # Where the run is, for every record: the step before (what was acted), the room type of the last map node
        # taken, and the fight's roster. The referee's message carries no encounter id, so a fight is named by the
        # enemies it began with.
        self.prev_act: dict | None = None
        self.room_type: str | None = None
        self.roster: str | None = None
        self.round: int | None = None
        self.ctx: dict = {}
        self.stats = {"decisions": 0, "ranked": 0, "not_ranked": 0, "diverged": 0, "gaps": 0, "reanchors": 0,
                      "reanchor_failed": 0, "resumed": 0, "first_loss": None, "losses": {}}

    def decide(self, msg: dict) -> int:
        kind, labels = msg["kind"], msg["actions"]
        i, self.n = self.n, self.n + 1
        self.stats["decisions"] += 1
        self._track(kind, msg)
        ranked = False
        if self.emu is not None:
            ranked = self._check(i, kind, labels, msg) if self.lost is None else self._resume(i, kind, labels, msg)
        choice = None
        if ranked:
            try:
                choice = self.emu.choose(kind, labels, msg)
            except Exception as e:  # noqa: BLE001 -- a search failure suspends the lockstep, not the run
                self._lose(i, f"choose_error:{type(e).__name__}:{e}"[:200], kind=kind, msg=msg)
        if ranked and getattr(self.emu, "ai_state_assumed", False):
            self.assumed.add(i)
        if choice is None:
            choice = self.fallback(msg)
        if self.lost is None and self.emu is not None:
            try:
                err = self.emu.apply(kind, labels, choice)
            except Exception as e:  # noqa: BLE001 -- the emulator refused the referee's step: suspend
                err = f"{type(e).__name__}:{e}"[:200]
            if err:
                self._lose(i, f"apply:{err}", kind=kind, msg=msg)
        scored = ranked and self.lost is None
        self.stats["ranked" if scored else "not_ranked"] += 1
        if not scored and self.open is not None:
            sp = self.open["not_ranked_span"]
            sp["first"] = i if sp["first"] is None else sp["first"]
            sp["last"], sp["count"] = i, sp["count"] + 1
        self.prev_kind = kind
        if isinstance(choice, int) and 0 <= choice < len(labels):
            self.prev_act = {"type": act_type(kind, labels[choice]), "kind": kind, "label": labels[choice]}
            if kind == "map" and isinstance(labels[choice], list) and len(labels[choice]) > 2:
                self.room_type = labels[choice][2]
        return choice

    def _track(self, kind: str, msg: dict) -> None:
        """The context every record of this decision carries (set before the decision's own step is taken)."""
        c = (msg.get("obs") or {}).get("combat")
        if not c or kind not in COMBAT_KINDS:
            self.roster, self.round = None, None
        else:
            rnd = c.get("round") if isinstance(c.get("round"), int) else None
            if self.prev_kind not in COMBAT_KINDS or (rnd is not None and self.round is not None and rnd < self.round):
                names = sorted(str(e.get("name")) for e in c.get("enemies") or [] if isinstance(e, dict))
                self.roster = "+".join(names) or None  # a new fight: the enemies it began with
            self.round = rnd
        self.ctx = {"acted": self.prev_act, "turn": self.round, "encounter": self.roster, "room_type": self.room_type}

    def finish(self, outcome: dict) -> None:
        """Run end: every divergence record carries the run id, the agent seed and the run's outcome."""
        for r in self.log.records:
            r["run"], r["agent_seed"], r["outcome"] = self.seed, self.agent_seed, outcome
            r["ai_state_assumed"] = r["decision"] in self.assumed

    def _lose(self, i: int, reason: str, kind: str | None = None, msg: dict | None = None,
              rec: dict | None = None) -> None:
        self.lost = reason
        key = reason.split(":")[0] + (":" + reason.split(":")[1] if reason.startswith(("unmapped", "apply")) else "")
        self.stats["losses"][key] = self.stats["losses"].get(key, 0) + 1
        self.stats["first_loss"] = self.stats["first_loss"] or {"decision": i, "reason": reason}
        if kind is not None and reason.startswith("apply:"):  # an unmirrorable step is logged like a structure gap
            self.stats["gaps"] += 1
            obs = (msg or {}).get("obs", {})
            rec = self.log.add(decision=i, floor=obs.get("floor"), room=_room(obs), type=kind, reason=reason, gap=True,
                               diff=[], n_diff=0, agent_weighted=True, reanchor_plan=None, reanchored=False, **self.ctx)
        if rec is None:  # a loss with no record of its own (a search failure): one, so its span has a home
            obs = (msg or {}).get("obs", {})
            rec = self.log.add(decision=i, floor=obs.get("floor"), room=_room(obs), type=kind, reason=reason, gap=False,
                               diff=[], n_diff=0, agent_weighted=True, reanchor_plan=None, reanchored=False, **self.ctx)
        rec["not_ranked_span"] = {"first": None, "last": None, "count": 0}  # decisions scored, not ranked, after it
        self.open = rec

    def _equal(self, kind: str, labels: list, msg: dict) -> tuple[str | None, list, dict, dict]:
        reason = None if kind in MAPPED_KINDS else f"unmapped:{kind}"
        reason = reason or self.emu.sync(kind, labels, msg)
        r, e = ref_view(msg.get("obs", {})), self.emu.view()
        if mid_screen_pick(kind, msg):
            return reason, [], r, e  # inside a screen: its candidates were compared in sync
        if kind == "treasure":
            # The emulator opens the chest and holds its relic when the referee is first asked whether to take it (the
            # 6 of 12 treasure divergences of the events round); the question's own offer is the referee's side of that.
            r["relics"] = r["relics"] + [x[1] for x in labels if isinstance(x, list) and len(x) > 1 and x[0] == "take"]
        return reason, ([] if reason else structural_diff(r, e)), r, e

    def _reanchor(self, plan: str, kind: str, labels: list, msg: dict) -> bool:
        """Re-anchor, then the same comparison again: a re-anchor counts only when the views agree after it."""
        try:
            ok = self.emu.reanchor(plan, msg)
        except Exception:  # noqa: BLE001 -- a failed re-anchor waits for the next point
            ok = False
        reason, d = None, []
        if ok:
            reason, d, _, _ = self._equal(kind, labels, msg)
            ok = not reason and not d
        self.stats["reanchors" if ok else "reanchor_failed"] += 1
        if not ok:  # why: the emulator side's own reason, else the views that still differ after it
            why = getattr(self.emu, "fail", None) or (reason or "diff:" + ",".join(f"{x[0]}{x[1]}/{x[2]}" if x[0] in (".floor", ".act") else str(x[0]) for x in d[:4]))
            fails = self.stats.setdefault("reanchor_fail_reasons", {})
            fails[f"{plan}:{why}"[:80]] = fails.get(f"{plan}:{why}"[:80], 0) + 1
        by = self.stats.setdefault("reanchor_by_plan", {})
        by[f"{plan}:{'ok' if ok else 'failed'}"] = by.get(f"{plan}:{'ok' if ok else 'failed'}", 0) + 1
        return ok

    def _resume(self, i: int, kind: str, labels: list, msg: dict) -> bool:
        plan = reanchor_plan(kind, self.prev_kind)
        if plan is None or kind not in MAPPED_KINDS or not self._reanchor(plan, kind, labels, msg):
            return False
        self.lost, self.open = None, None
        self.stats["resumed"] += 1
        return True

    def _check(self, i: int, kind: str, labels: list, msg: dict) -> bool:
        obs = msg.get("obs", {})
        reason, d, r, e = self._equal(kind, labels, msg)
        if not reason and not d:
            return True
        gap = bool(reason) and reason.startswith(GAP_PREFIXES)
        self.stats["gaps" if gap else "diverged"] += 1
        rec = self.log.add(decision=i, floor=obs.get("floor"), room=_room(obs), type=kind,
                           ref_hash=view_hash(r), emu_hash=view_hash(e), referee_state_hash=msg.get("h"),
                           reason=reason, gap=gap, diff=d[:12], n_diff=len(d), agent_weighted=True, **self.ctx)
        plan = reanchor_plan(kind, self.prev_kind)
        rec["reanchor_plan"] = plan
        rec["reanchored"] = bool(plan) and not gap and self._reanchor(plan, kind, labels, msg)
        if rec["reanchored"]:
            return True
        self._lose(i, reason or "state_diverged", rec=rec)
        return False


def _room(obs: dict) -> str:
    return "combat" if obs.get("combat") else "non_combat"


# ------------------------------------------------------------------ the emulator side


def _event_names() -> dict[int, str]:
    """The emulator's event ids as the game's ModelId.Entry (the constant's own name: EventSunkenTreasury ->
    SUNKEN_TREASURY), read off RunConstants. Empty when the source is not there."""
    import re  # noqa: PLC0415

    path = ROOT / "src" / "Sts2Emulator" / "Core" / "Run" / "RunConstants.cs"
    if not path.is_file():
        return {}
    out: dict[int, str] = {}
    for name, value in re.findall(r"public const int Event(\w+)\s*=\s*(-?\d+);", path.read_text(encoding="utf-8")):
        out.setdefault(int(value), re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper())
    return out


class EmuLock:
    """The real emulator side: a RunSession from the referee's seed, the rules policy, IS-MCTS in fights.

    Re-anchoring uses the run replay's own hooks (scripts/replay_real_run.py's path into the native library):
    `Sts2Run_DebugResolveCombat` ends a fight the emulator is still in, the emulator is walked to its map screen, and
    `Sts2Run_DebugSetPlayer` sets the player's holdings (HP, max HP, gold, deck, relics, potion belt) to the referee's
    record. No new state format and no native rebuild; what stays the emulator's own is its RNG and map position.
    """

    REST = {"HEAL": "heal", "SMITH": "smith", "CLONE": "clone", "LIFT": "lift", "DIG": "dig", "HATCH": "hatch",
            "COOK": "cook", "KINDLE": "kindle"}
    SCREEN = {"neow": "neow", "ancient": "neow", "map": "map", "card_reward": "card_reward", "rest": "rest",
              "select_cards": "card_select", "shop": "shop", "bundle": "bundle", "event": "event"}

    def __init__(self, seed: str, ascension: int = 10, sims: int = 60, agent_seed=None, character: str = "IRONCLAD"):
        import leakprobes  # noqa: PLC0415 -- the native library's location

        leakprobes.native_env()
        from run_train import engine, features, selfplay, vocab  # noqa: PLC0415
        import live_search  # noqa: PLC0415
        import search_configs  # noqa: PLC0415
        from sts2_gym import native  # noqa: PLC0415

        # `sims`: a flat budget, or a search config name (rl/search_configs.py)
        self.cfg = search_configs.get(sims) if isinstance(sims, str) else search_configs.flat(int(sims))
        self.position_of = search_configs.position_of
        self.E, self.F, self.V, self.ls, self.native = engine, features, vocab, live_search, native
        self.seed, self.sims = seed, sims
        self.s = engine.RunSession(seed, ascension, engine.CombatBudget(self.cfg.sims, self.cfg.sims),
                                   **({} if character == "IRONCLAD" else {"character": character}))
        self.rules = selfplay.RulesPlayer()
        # the agent's RNG is seeded from the agent seed alone, never from the game seed (DESIGN 3.2)
        self.rng = agent_rng(agent_seed if agent_seed is not None else [fresh_agent_seed()])
        self.slug = {k: {v: self.V.game_slug(n) for n, v in m.items()} for k, m in self._id_map().items()}
        self.ids = {k: {v: i for i, v in m.items()} for k, m in self.slug.items()}
        self.dec = None
        self.rq: list = []  # the referee's reward answers not yet mirrored: [type, take]
        self._msg: dict | None = None
        self.fail: str | None = None  # why the last reanchor() returned False
        self.event_names = _event_names()

    @staticmethod
    def _id_map() -> dict:
        return {k: v for k, v in json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")).items()
                if k in ("cards", "relics", "potions")}

    def close(self) -> None:
        self.s.close()

    # -- state
    def _card(self, c) -> list:
        return [self.slug["cards"].get(c.card_id, f"?{c.card_id}"), int(c.upgraded)]

    def _obs(self):
        return self.s.env._run_obs_buf

    def _hand(self) -> list:
        obs, ls = self._obs(), self.ls
        return [[self.slug["cards"].get(obs[ls._HAND + k * ls._CARD], "?"), int(obs[ls._HAND + k * ls._CARD + 1])]
                for k in range(ls._hand_size(obs))]

    def view(self) -> dict:
        v = self.s.view()
        out = {"act": self.s.act_index, "floor": v.floor, "hp": v.hp, "max_hp": v.max_hp, "gold": v.gold,
               "deck": [self._card(c) for c in v.deck],
               "relics": [self.slug["relics"].get(r, f"?{r}") for r in v.relics],
               "potions": [self.slug["potions"].get(p, f"?{p}") for p in v.potions]}
        if self.s.phase == self.E.rc.PHASE_COMBAT:
            obs, ls = self._obs(), self.ls
            hp = emu_enemy_hp(obs, ls._ENEMIES, ls._ENEMY, self.native.MAX_ENEMIES)
            out["combat"] = {"hand": self._hand(), "enemy_hp": hp}
        return out

    def _decision(self):
        self.dec = None if self.s.phase in (self.E.rc.PHASE_COMBAT, self.E.rc.PHASE_COMPLETE) else self.s.decision()
        return self.dec

    def _take(self, kind: str) -> bool:
        o = next((o for o in self.dec.options if o.kind == kind), None) if self.dec else None
        if o is not None:
            self.s.apply(o)
        return o is not None

    # -- reaching the referee's screen: what the referee does without asking, the emulator does here
    def _flush_rewards(self) -> None:
        """Mirror the referee's reward answers in its order, as it gives them. RunSession.decision() opens a card reward
        before it offers the potion and relic items, so on the items screen the answer is stepped on the engine's own
        item index (engine._rewards' layout: gold, potion, relic, cards) without going through decision()."""
        rc = self.E.rc
        for _ in range(20):
            ph = self.s.phase
            if ph == rc.PHASE_RELIC_REWARD:
                gold, potion, relic, cards = self.native.run_state_list(self.s.handle, 6, 4)
                if gold:
                    self.s._step(0)  # gold is taken before any question on both sides
                    continue
                if not self.rq:
                    return
                t, take = self.rq.pop(0)
                present = [k for k, n in (("potion", potion), ("relic", relic), ("card", cards)) if n]
                if take and t in present:
                    self.s._step(present.index(t))
                continue
            if ph != rc.PHASE_CARD_REWARD:
                return
            k = next((k for k, q in enumerate(self.rq) if q[0] == "card"), None)
            if k is None:
                return
            _, take = self.rq.pop(k)
            if take:
                return  # the referee's card_reward decision follows
            self._decision()
            self._take("card.skip")

    def _forced(self, want: str | None) -> None:
        """Steps the referee took without a decision: leaving a rewards screen it has answered (gold, skipped items),
        and any screen with a single option (Env.Decide asks only with two or more)."""
        self._flush_rewards()
        for _ in range(30):
            d = self._decision()
            if d is None or d.type == want:
                return
            if d.type == "rewards" and want != "rewards":
                self.rq.clear()
                self._take("reward.proceed")
            elif d.type == "card_reward" and want not in ("rewards", "card_reward"):
                self.rq.clear()
                self._take("card.skip")
            elif len(d.options) == 1:
                self.s.apply(d.options[0])
            else:
                return

    def sync(self, kind: str, labels: list, msg: dict | None = None) -> str | None:
        self._msg = msg
        if kind not in COMBAT_KINDS and self.s.phase == self.E.rc.PHASE_COMBAT:
            # The referee left its fight with no question: a lone End Turn that ends the fight (a power's end-of-turn
            # damage, a retaliation on the enemy's turn) is taken for it, so the emulator stands where the referee does.
            self._auto_combat()
        if kind in COMBAT_KINDS:
            self._forced("combat")
            self._auto_combat()
            if self.s.phase != self.E.rc.PHASE_COMBAT:
                return f"phase:{self.s.phase}"
            return self._screen_state(kind, labels, msg)
        if kind == "reward":
            self._flush_rewards()
            ph = self.s.phase
            return None if ph in (self.E.rc.PHASE_RELIC_REWARD, self.E.rc.PHASE_CARD_REWARD) else f"phase:{ph}!=rewards"
        if kind == "treasure":  # the emulator takes the chest's relic itself (engine._treasure)
            self._forced("map")
            want = ((msg or {}).get("obs") or {}).get("floor")
            if want is not None and self.s.floor < want and self.dec is not None and len(self.dec.options) == 1:
                self.s.apply(self.dec.options[0])  # the referee walked a lone path into the chest room
                self._forced("map")
            return None
        self._forced(self.SCREEN[kind])
        if kind == "event":
            return self._sync_event(msg)
        got = None if self.dec is None else self.dec.type
        return None if got == self.SCREEN[kind] else f"screen:{got}!={self.SCREEN[kind]}"

    def _sync_event(self, msg: dict | None) -> str | None:
        """Reach the event page the referee is asking about. The referee asks only with two or more enabled options, so
        a page where the emulator has a single option is one the referee passed without asking."""
        for _ in range(12):
            self._flush_rewards()
            d = self._decision()
            if d is None or d.type != "event" or len(d.options) != 1:
                break
            self.s.apply(d.options[0])
            self._forced("event")
        got = None if self.dec is None else self.dec.type
        if got != "event":
            return f"screen:{got}!=event"
        ctx = (msg or {}).get("ctx")
        ref = ctx if isinstance(ctx, str) else None
        ev = self.dec.options[0].event
        emu = self.event_names.get(int(ev[0])) if ev is not None and ev[0] is not None else None
        if ref and emu and ref != emu:
            return f"event:{ref}!={emu}"  # a different event: the emulator's room is not the referee's
        return None

    def _auto_combat(self) -> None:
        """The referee asks only when two or more actions are legal (Env.Decide), so a lone End Turn is taken for it."""
        env = self.s.env
        for _ in range(40):
            if self.s.phase != self.E.rc.PHASE_COMBAT:
                return
            legal = [i for i in range(32) if self.native.run_action_mask(self.s.handle, 32)[i]]
            if len(legal) != 1:
                return
            self.native.run_step(self.s.handle, legal[0], -1, env._run_obs_buf, env._run_rew_buf, env._run_terminal_buf,
                                 env._run_truncated_buf)

    def _screen_state(self, kind: str, labels: list, msg: dict | None) -> str | None:
        """Both sides must stand on the same side of a fight card screen, so that they leave it together. The referee asks `combat`
        only once its screen has closed, and a further pick only while the screen is open with the cards left: an emulator
        screen still open at a `combat` decision, closed at a further pick, or showing other cards is a difference."""
        obs, ls = self._obs(), self.ls
        is_open = bool(obs[ls._SELECTION_KIND])
        if mid_screen_pick(kind, msg or {}):
            if not is_open:
                return "selection:closed_early"
            count = int(obs[ls._SELECTION_COUNT])
            ref = sorted(json.dumps([x[0], int(x[1])]) for x in labels if isinstance(x, list))
            if count != len(ref):
                return "selection:candidates_differ"
            if count <= getattr(ls, "_MAX_CANDIDATES", 10):
                emu = sorted(json.dumps(self._card_of(ls._card_at(obs, ls._SELECTION + a * ls._CARD))) for a in range(count))
                if emu != ref:
                    return "selection:candidates_differ"
        elif kind == "combat" and is_open:
            return "selection:left_open"
        return None

    def _card_of(self, c) -> list:
        return [self.slug["cards"].get(c[0], "?"), int(c[1])]

    def _select_action(self, label: list) -> int | None:
        """The emulator's candidate index for a referee selection label [entry, upgraded]; for "done" its decline (one past
        the candidates; apply checks it is legal); None for "none"."""
        if label == "done":
            obs, ls = self._obs(), self.ls
            n = int(obs[ls._SELECTION_COUNT]) if obs[ls._SELECTION_KIND] else -1
            return n if 0 <= n < 32 else None
        if not (isinstance(label, list) and len(label) > 1):
            return None  # the "none" action of a screen that allows 0 cards: the emulator's fight selection has no such action
        obs, ls = self._obs(), self.ls
        for a in range(min(obs[ls._SELECTION_COUNT], 32)):
            c = ls._card_at(obs, ls._SELECTION + a * ls._CARD)
            if [self.slug["cards"].get(c[0], "?"), int(c[1])] == [label[0], int(label[1])]:
                return a
        return None

    # -- re-anchoring (the run replay's hooks)
    def _to_map(self, floor: int | None = None) -> bool:
        """Walk the emulator to its map screen: a fight it is still in is resolved, every other screen is left the
        cheapest way (proceed, skip, leave) or by the rules policy. With `floor`, on past maps below that floor too: a
        suspended emulator stands still while the referee goes on, and the floor is not a holding that can be set (13 of
        the 52 failed re-anchors of the events-round smoke were the emulator 1-3 floors behind)."""
        rc, env = self.E.rc, self.s.env
        self.rq.clear()
        for _ in range(600):
            ph = self.s.phase
            if ph == rc.PHASE_COMPLETE:
                return False
            if ph == rc.PHASE_COMBAT:
                if not self.native.run_debug_resolve_combat(self.s.handle, env._run_obs_buf):
                    return False
                continue
            d = self._decision()
            if d is None:
                continue
            if d.type == "map":
                if floor is None or self.s.view().floor >= floor:
                    return True
                self.s.apply(d.options[self.rules.pick(self.s.view(), d, self.rng).index])
                continue
            o = next((o for o in d.options if o.kind in ("reward.proceed", "card.skip", "shop.leave", "rest.leave")),
                     None) or d.options[self.rules.pick(self.s.view(), d, self.rng).index]
            self.s.apply(o)
        return False

    def _set_player(self, obs: dict) -> bool:
        cards = [(self.ids["cards"].get(c[0]), int(c[1]), 0, 0) for c in obs.get("deck", [])]
        relics = [self.ids["relics"].get(r) for r in obs.get("relics", [])]
        belt = obs.get("potions", [])
        potions = [self.ids["potions"].get(p) if p else 0 for p in belt]
        if any(c[0] is None for c in cards) or None in relics or None in potions:
            return False  # an id the emulator does not have: not re-anchorable
        self.native.run_debug_set_player(self.s.handle, self._obs(), hp=int(obs["hp"]), max_hp=int(obs["max_hp"]),
                                         gold=int(obs["gold"]), cards=cards, relics=relics, potions=potions,
                                         max_potion_slots=len(belt))
        return True

    def _anchor_fight(self, obs: dict) -> bool:
        """Put the fight the emulator stands in on the referee's record of it (obs["combat"], as the agent is shown it:
        the draw pile as a sorted list, no streams). The native call writes it over a copy of the fight and keeps the
        emulator's own streams and, where its cards are the record's, its draw order, so a search that follows still
        resamples the hidden state from the emulator's clone (resample_hidden=True)."""
        record = obs.get("combat")
        if not record:
            self.fail = "no_record"
            return False
        status, why = self.native.run_reanchor_fight(self.s.handle, record, self.s.env._run_obs_buf)
        if status < 0:
            self.fail = f"anchor:{why}"[:60]
            return False
        return True

    def reanchor(self, plan: str, msg: dict) -> bool:
        self.fail = None
        obs = msg.get("obs", {})
        if plan == "room_entry":
            if not self._to_map(obs.get("floor")):
                self.fail = "to_map"
            elif not self._set_player(obs):
                self.fail = "set_player:unknown_id"
            elif self._decision() is None:
                self.fail = "no_decision"
            return self.fail is None
        if plan == "fight_start":
            # The referee entered this fight without a map decision (a lone path), or the emulator's own fight differs
            # from it. A fight the emulator stands in at the referee's floor is put on the referee's record of it
            # (Sts2Run_ReanchorFight); one at an earlier floor is a fight the suspended emulator never left, which is
            # resolved and walked past as at a room entry, and the fight it then enters is put on the record too.
            floor = obs.get("floor")
            if self.s.phase == self.E.rc.PHASE_COMBAT and (floor is None or self.s.view().floor >= floor):
                return self._anchor_fight(obs)
            if not self._to_map(None if floor is None else floor - 1):
                self.fail = "to_map"
            elif not self._set_player(obs):
                self.fail = "set_player:unknown_id"
            else:
                d = self._decision()
                if d is None or len(d.options) != 1:
                    self.fail = f"map_options:{None if d is None else len(d.options)}"
                else:
                    self.s.apply(d.options[0])
                    self._auto_combat()
                    if self.s.phase != self.E.rc.PHASE_COMBAT:
                        self.fail = "not_in_fight"
                    else:
                        return self._anchor_fight(obs)
            return self.fail is None
        self.fail = "no_plan"
        return False

    # -- non-combat pairing: referee label index -> emulator option index
    def _pair(self, kind: str, labels: list) -> list[int | None]:
        if any(extra_kind(lab) in NO_EMULATOR_KINDS for lab in labels):
            # Actions the emulator has no option for are paired as None; the screen's own labels pair as before.
            core = [i for i, lab in enumerate(labels) if extra_kind(lab) not in NO_EMULATOR_KINDS]
            sub = self._pair(kind, [labels[i] for i in core])
            out: list[int | None] = [None] * len(labels)
            for i, j in zip(core, sub):
                out[i] = j
            return out
        opts = self.dec.options
        if kind == "event":
            # The labels are the enabled options' TextKeys (or relic ids) in page order, the emulator's options its legal
            # actions in action order: paired by position, and only when the counts agree (a locked option the two sides
            # count differently is a gap, not a guess).
            order = sorted(range(len(opts)), key=lambda j: opts[j].action)
            return list(order) if len(order) == len(labels) else [None] * len(labels)
        out: list[int | None] = []
        for lab in labels:
            hit = None
            for j, o in enumerate(opts):
                if hit is not None or j in out:
                    continue
                if kind == "map" and o.kind == "map.node" and list(o.node[:2]) == list(lab[:2]):
                    hit = j
                elif kind in ("neow", "ancient") and o.kind == "neow.relic" and lab == self.slug["relics"].get(o.relic):
                    hit = j
                elif kind == "card_reward":
                    if o.kind == "card.take" and isinstance(lab, list) and self._card(o.card) == [lab[0], int(lab[1])]:
                        hit = j
                    elif o.kind == "card.skip" and lab in ("skip", "Skip"):
                        hit = j
                elif kind == "rest" and o.kind == "rest." + self.REST.get(lab, "?"):
                    hit = j
                elif kind == "select_cards" and lab == "none":
                    if o.kind == "select.offer_done":
                        hit = j
                elif kind == "select_cards":
                    deck = (self.dec.view or self.s.view()).deck
                    card = o.card if o.card is not None else (deck[o.deck_index] if o.deck_index is not None else None)
                    if card is not None and self._card(card) == [lab[0], int(lab[1])]:
                        hit = j
                elif kind == "shop":
                    # [kind, slot, cost] or ["leave"]: by kind and price (a removal opens a card choice: not paired)
                    if lab[0] == "leave" and o.kind == "shop.leave":
                        hit = j
                    elif lab[0] in ("card", "relic", "potion") and o.kind == "shop." + lab[0] and o.price == lab[2]:
                        hit = j
                elif kind == "bundle" and o.kind == "bundle.pick":
                    if sorted(self.slug["cards"].get(c.card_id, "?") for c in o.cards) == sorted(lab):
                        hit = j
            out.append(hit)
        return out

    def choose(self, kind: str, labels: list, msg: dict) -> int | None:
        if kind in COMBAT_KINDS:
            return self._choose_combat(kind, labels, msg)
        if kind in ("reward", "treasure"):
            return 0  # take (a card reward's pick follows as its own decision)
        pair = self._pair(kind, labels)
        pick = self.rules.pick(self.s.view(), self.dec, self.rng).index
        for ref_i, emu_j in enumerate(pair):
            if emu_j == pick:
                return ref_i
        return None

    def apply(self, kind: str, labels: list, idx: int) -> str | None:
        if kind in COMBAT_KINDS:
            return self._apply_combat(kind, labels, idx)
        if extra_kind(labels[idx]) in NO_EMULATOR_KINDS:
            return f"no_emulator_option:{kind}:{json.dumps(labels[idx])[:60]}"
        if kind == "reward":
            ctx = (self._msg or {}).get("ctx") or {}
            self.rq.append([ctx.get("type", "card"), labels[idx] == "take"])
            self._flush_rewards()
            return None
        if kind == "treasure":
            return None if labels[idx] != "skip" else "treasure_skip"  # the emulator has already taken the relic
        j = self._pair(kind, labels)[idx]
        if j is None:
            return f"no_emulator_option:{kind}:{json.dumps(labels[idx])[:60]}"
        o = self.dec.options[j]
        if o.kind == "rest.smith" and o.deck_index is None:
            self.s._step(o.action)  # the referee's select_cards follows
            return None
        self.s.apply(o)
        return None

    # -- combat: a referee action is matched by what it plays, not by its hand position
    def _key(self, action: int, target: int):
        obs, ls = self._obs(), self.ls
        hand = self._hand()
        if action < len(hand):
            return ("card", *hand[action], target)
        if action == len(hand):
            return ("end",)
        return ("potion", self.slug["potions"].get(obs[ls.POTION_OFFSET + 2 * (action - len(hand) - 1)], "?"), target)

    def _action_of(self, label: list, msg: dict | None = None) -> tuple[int, int] | None:
        obs, ls = self._obs(), self.ls
        hand = self._hand()
        if label[0] == "end":
            return len(hand), -1
        living = ls._living_targets(obs)
        t = label[3] if len(label) > 3 else -1
        slot = living[t] if 0 <= t < len(living) else -1
        if label[0] == "play":
            ref_hand = (((msg or {}).get("obs") or {}).get("combat") or {}).get("hand") or []
            h = int(label[1])
            want = [label[2], int(ref_hand[h][1])] if h < len(ref_hand) else None
            same = [k for k, c in enumerate(hand) if (c == want if want else c[0] == label[2])]
            if not same:
                return None
            return (h if h in same else same[0]), slot
        if label[0] == "potion":
            for s in [k for k in (int(label[1]), *range(ls.MAX_POTIONS)) if 0 <= k < ls.MAX_POTIONS]:
                if self.slug["potions"].get(obs[ls.POTION_OFFSET + 2 * s], "?") == label[2]:
                    return len(hand) + 1 + s, slot
            return None
        return None

    def _choose_combat(self, kind: str, labels: list, msg: dict) -> int | None:
        self.ai_state_assumed = False
        root = self.s.combat(fork_combat=True).combat_fork(None)
        try:
            pos = self.position_of(root, self.s.info) if self.cfg.schedule else None
            res = self.ls.search_live(root.handle, seed=int(self.rng.integers(1 << 30)), **self.cfg.search_kwargs(pos))
        finally:
            root.close() if hasattr(root, "close") else self.native.destroy(root.handle)
        self.ai_state_assumed = bool(res.ai_state_assumed)
        if kind != "combat":
            return next((i for i, lab in enumerate(labels) if self._select_action(lab) == res.best.action), None)
        want = self._key(res.best.action, res.best.target)
        for i, lab in enumerate(labels):
            a = self._action_of(lab, msg)
            if a is not None and self._key(*a) == want:
                return i
        return None

    def _apply_combat(self, kind: str, labels: list, idx: int) -> str | None:
        a = self._action_of(labels[idx], self._msg) if kind == "combat" else (self._select_action(labels[idx]), -1)
        if a is None or a[0] is None:
            return f"unmapped_label:{json.dumps(labels[idx])[:60]}"
        if not self.native.run_action_mask(self.s.handle, 32)[a[0]]:
            return f"illegal:{json.dumps(labels[idx])[:60]}"
        env = self.s.env
        st = self.native.run_step(self.s.handle, a[0], a[1], env._run_obs_buf, env._run_rew_buf,
                                  env._run_terminal_buf, env._run_truncated_buf)
        return None if st == 0 else f"refused:{a}"


# ------------------------------------------------------------------ the referee-only fallback


def referee_only_policy(msg: dict) -> int:
    """Scored-not-ranked play: the first card play, else the first non-end action, else 0. Deterministic."""
    acts = msg["actions"]
    if "done" in acts and mid_screen_pick(msg.get("kind"), msg):
        return acts.index("done")  # a fight card screen: one card, as it was always taken, then the screen's end
    for i, a in enumerate(acts):
        if isinstance(a, list) and a and a[0] == "play":
            return i
    for i, a in enumerate(acts):
        if not (isinstance(a, list) and a and a[0] == "end"):
            return i
    return 0


def play_run(env, seed: str, *, ascension: int = 10, sims: int = 60, emu_factory=None, max_decisions: int = 4000,
             agent_seed=None, character: str = "IRONCLAD"):
    """One run on a GameEnv under the PLAY agent. Returns (end message, Lockstep)."""
    agent_seed = list(agent_seed) if agent_seed is not None else [fresh_agent_seed(), 0]
    emu = (emu_factory or (lambda s: EmuLock(s, ascension, sims, agent_seed, character)))(seed)
    ls = Lockstep(emu, referee_only_policy, seed, sims, agent_seed)
    try:
        m, done = env.reset(seed, hash=True, **({} if character == "IRONCLAD" else {"character": character}))
        n = 0
        while not done and n < max_decisions:
            m, done = env.step(ls.decide(m))
            n += 1
        ls.finish(run_outcome(m, done))
        return m, ls
    except BaseException as e:
        ls.finish(run_outcome({"end": "crash", "floor": None}, True))
        e.lockstep = ls  # the worker's crash record keeps the divergences so far
        raise
    finally:
        if emu is not None and hasattr(emu, "close"):
            emu.close()
