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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]

COMBAT_KINDS = ("combat", "select_cards_combat", "card_reward_combat")
# Referee decision kinds the emulator side can replay and answer. Others end the lockstep (reason unmapped:<kind>).
MAPPED_KINDS = ("neow", "ancient", "map", "card_reward", "rest", "reward", "select_cards") + COMBAT_KINDS


def _paths() -> None:
    for p in (ROOT / "rl", ROOT / "src", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


# ------------------------------------------------------------------ projection and hashes


def project(view: dict) -> dict:
    """The agent-visible fields both sides share, canonical (multisets sorted). Run level, plus the hand and
    the enemies' HP in a fight. Nothing here is a seed, a counter or the draw order."""
    out = {k: view.get(k) for k in ("act", "floor", "hp", "max_hp", "gold")}
    for k in ("deck", "relics", "potions"):
        out[k] = sorted(json.dumps(x) for x in view.get(k, []) if x is not None)
    c = view.get("combat")
    if c:
        out["combat"] = {"hand": sorted(json.dumps(x) for x in c.get("hand", [])),
                         "enemy_hp": [h for h in c.get("enemy_hp", [])]}
    return out


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

    def sync(self, kind: str, labels: list) -> str | None: ...        # a reason when the structure disagrees
    def view(self) -> dict: ...
    def choose(self, kind: str, labels: list, msg: dict) -> int | None: ...
    def apply(self, kind: str, labels: list, idx: int) -> str | None: ...
    def reanchor(self, plan: str, msg: dict) -> bool: ...


class Lockstep:
    """The per-run state machine. `decide(msg)` answers one referee decision with an action index."""

    def __init__(self, emu: EmuSide | None, fallback, seed: str = "", sims: int = 60):
        self.emu, self.fallback, self.seed, self.sims = emu, fallback, seed, sims
        self.log = DivergenceLog()
        self.lost: str | None = None if emu is not None else "no_emulator"
        self.n = 0
        self.prev_kind: str | None = None
        self.stats = {"decisions": 0, "ranked": 0, "not_ranked": 0, "diverged": 0, "reanchors": 0,
                      "reanchor_failed": 0, "first_loss": None}

    def decide(self, msg: dict) -> int:
        kind, labels, obs = msg["kind"], msg["actions"], msg.get("obs", {})
        i, self.n = self.n, self.n + 1
        self.stats["decisions"] += 1
        ranked = self.lost is None
        if ranked:
            ranked = self._check(i, kind, labels, msg)
        choice = None
        if self.lost is None:
            try:
                choice = self.emu.choose(kind, labels, msg)
            except Exception as e:  # noqa: BLE001 -- a search failure ends the lockstep, not the run
                self._lose(i, f"choose_error:{type(e).__name__}:{e}")
        if choice is None:
            choice = self.fallback(msg)
        if self.lost is None:
            err = self.emu.apply(kind, labels, choice)
            if err:
                self._lose(i, f"apply:{err}")
        self.stats["ranked" if ranked and self.lost is None else "not_ranked"] += 1
        self.prev_kind = kind
        return choice

    def _lose(self, i: int, reason: str) -> None:
        self.lost = reason
        self.stats["first_loss"] = self.stats["first_loss"] or {"decision": i, "reason": reason}

    def _check(self, i: int, kind: str, labels: list, msg: dict) -> bool:
        obs = msg.get("obs", {})
        reason = None if kind in MAPPED_KINDS else f"unmapped:{kind}"
        reason = reason or self.emu.sync(kind, labels)
        r, e = ref_view(obs), self.emu.view()
        d = [] if reason else structural_diff(r, e)
        if not reason and not d:
            return True
        self.stats["diverged"] += 1
        rec = self.log.add(decision=i, floor=obs.get("floor"), room=_room(obs), type=kind,
                           ref_hash=view_hash(r), emu_hash=view_hash(e), referee_state_hash=msg.get("h"),
                           reason=reason, diff=d[:12], n_diff=len(d), agent_weighted=True)
        plan = reanchor_plan(kind, self.prev_kind)
        rec["reanchor_plan"] = plan
        if plan and not reason and self.emu.reanchor(plan, msg):
            self.stats["reanchors"] += 1
            rec["reanchored"] = True
            return True
        if plan:
            self.stats["reanchor_failed"] += 1
        rec["reanchored"] = False
        self._lose(i, reason or "state_diverged")
        return False


def _room(obs: dict) -> str:
    return "combat" if obs.get("combat") else "non_combat"


# ------------------------------------------------------------------ the emulator side


class EmuLock:
    """The real emulator side: a RunSession from the referee's seed, the rules policy, IS-MCTS in fights."""

    def __init__(self, seed: str, ascension: int = 10, sims: int = 60):
        import leakprobes  # noqa: PLC0415 -- the native library's location

        leakprobes.native_env()
        from run_train import engine, features, selfplay, vocab  # noqa: PLC0415
        import live_search  # noqa: PLC0415
        from sts2_gym import native  # noqa: PLC0415

        self.E, self.F, self.V, self.ls, self.native = engine, features, vocab, live_search, native
        self.seed, self.sims = seed, sims
        self.s = engine.RunSession(seed, ascension, engine.CombatBudget(sims, sims))
        self.rules = selfplay.RulesPlayer()
        self.rng = __import__("numpy").random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
        self.slug = {k: {v: self.V.game_slug(n) for n, v in m.items()} for k, m in self._id_map().items()}
        self.dec = None
        self.moves = 0

    @staticmethod
    def _id_map() -> dict:
        return {k: v for k, v in json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")).items()
                if k in ("cards", "relics", "potions")}

    def close(self) -> None:
        self.s.close()

    # -- state
    def _card(self, c) -> list:
        return [self.slug["cards"].get(c.card_id, f"?{c.card_id}"), int(c.upgraded)]

    def view(self) -> dict:
        v = self.s.view()
        out = {"act": self.s.act_index, "floor": v.floor, "hp": v.hp, "max_hp": v.max_hp, "gold": v.gold,
               "deck": [self._card(c) for c in v.deck],
               "relics": [self.slug["relics"].get(r, f"?{r}") for r in v.relics],
               "potions": [self.slug["potions"].get(p, f"?{p}") for p in v.potions]}
        if self.s.phase == self.E.rc.PHASE_COMBAT:
            obs = self.s.env._run_obs_buf
            ls = self.ls
            hand = [[self.slug["cards"].get(obs[ls._HAND + k * ls._CARD], "?"), int(obs[ls._HAND + k * ls._CARD + 1])]
                    for k in range(ls._hand_size(obs))]
            hp = [int(obs[ls._ENEMIES + k * ls._ENEMY]) for k in range(self.native.MAX_ENEMIES)
                  if obs[ls._ENEMIES + k * ls._ENEMY] > 0]
            out["combat"] = {"hand": hand, "enemy_hp": hp}
        return out

    def sync(self, kind: str, labels: list) -> str | None:
        if kind in COMBAT_KINDS:
            self._auto_combat()
            return None if self.s.phase == self.E.rc.PHASE_COMBAT else f"phase:{self.s.phase}"
        self.dec = self.s.decision()
        want = {"neow": "neow", "ancient": "neow", "map": "map", "card_reward": "card_reward", "rest": "rest",
                "reward": "rewards", "select_cards": "card_select"}[kind]
        got = None if self.dec is None else self.dec.type
        # The emulator opens a card reward without asking; the referee asks take/skip first, then the cards.
        ok = got == want or (kind == "reward" and got == "card_reward")
        return None if ok else f"screen:{got}!={want}"

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

    def _select_action(self, label: list) -> int | None:
        """The emulator's candidate index for a referee selection label [entry, upgraded]."""
        obs, ls = self.s.env._run_obs_buf, self.ls
        for a in range(min(obs[ls._SELECTION_COUNT], 32)):
            c = ls._card_at(obs, ls._SELECTION + a * ls._CARD)
            if [self.slug["cards"].get(c[0], "?"), int(c[1])] == [label[0], int(label[1])]:
                return a
        return None

    def reanchor(self, plan: str, msg: dict) -> bool:
        return False  # no Python entry to the C# Reanchor (tools/emudump) yet: reported, not faked

    # -- non-combat pairing: referee label index -> emulator option index
    def _pair(self, kind: str, labels: list) -> list[int | None]:
        opts = self.dec.options
        out: list[int | None] = []
        for lab in labels:
            hit = None
            for j, o in enumerate(opts):
                if kind == "map" and o.kind == "map.node" and list(o.node[:2]) == list(lab[:2]):
                    hit = j
                elif kind in ("neow", "ancient") and o.kind == "neow.relic" and lab == self.slug["relics"].get(o.relic):
                    hit = j
                elif kind == "card_reward":
                    if o.kind == "card.take" and isinstance(lab, list) and self._card(o.card) == [lab[0], int(lab[1])]:
                        hit = j
                    elif o.kind == "card.skip" and lab in ("skip", "Skip"):
                        hit = j
                elif kind == "rest" and o.kind == "rest." + {"HEAL": "heal", "SMITH": "smith", "CLONE": "clone",
                                                              "LIFT": "lift", "DIG": "dig", "HATCH": "hatch",
                                                              "COOK": "cook", "KINDLE": "kindle"}.get(lab, "?"):
                    hit = j
                elif kind == "select_cards":
                    deck = (self.dec.view or self.s.view()).deck
                    card = o.card if o.card is not None else (deck[o.deck_index] if o.deck_index is not None else None)
                    if card is not None and self._card(card) == [lab[0], int(lab[1])] and j not in out and hit is None:
                        hit = j
                elif kind == "reward":
                    if lab == "take" and o.kind in ("reward.potion", "reward.relic"):
                        hit = j
                    elif lab == "skip" and o.kind == "reward.proceed":
                        hit = j
            out.append(hit)
        return out

    def choose(self, kind: str, labels: list, msg: dict) -> int | None:
        if kind in COMBAT_KINDS:
            return self._choose_combat(kind, labels, msg)
        if kind == "reward" and self.dec.type == "card_reward":
            return next((i for i, lab in enumerate(labels) if lab == "take"), None)
        pair = self._pair(kind, labels)
        pick = self.rules.pick(self.s.view(), self.dec, self.rng).index
        for ref_i, emu_j in enumerate(pair):
            if emu_j == pick:
                return ref_i
        return None

    def apply(self, kind: str, labels: list, idx: int) -> str | None:
        if kind in COMBAT_KINDS:
            return self._apply_combat(kind, labels, idx)
        if kind == "reward" and self.dec.type == "card_reward":
            if labels[idx] == "take":
                return None  # the card pick follows as the referee's card_reward decision
            j = next((k for k, o in enumerate(self.dec.options) if o.kind == "card.skip"), None)
        else:
            j = self._pair(kind, labels)[idx]
        if j is None:
            return f"no_emulator_option:{kind}:{labels[idx]}"
        o = self.dec.options[j]
        if o.kind == "rest.smith" and o.deck_index is None:
            self.s._step(o.action)  # the referee's select_cards follows; unmapped, so the lockstep ends there
            return None
        self.s.apply(o)
        return None

    # -- combat
    def _living(self) -> list[int]:
        return self.ls._living_targets(self.s.env._run_obs_buf)

    def _action_of(self, label: list) -> tuple[int, int] | None:
        obs = self.s.env._run_obs_buf
        hand = self.ls._hand_size(obs)
        if label[0] == "end":
            return hand, -1
        living = self._living()
        t = label[3] if len(label) > 3 else -1
        slot = living[t] if 0 <= t < len(living) else -1
        if label[0] == "play":
            return int(label[1]), slot
        if label[0] == "potion":
            return hand + 1 + int(label[1]), slot
        return None

    def _choose_combat(self, kind: str, labels: list, msg: dict) -> int | None:
        if kind == "card_reward_combat":
            return None
        root = self.s.combat(fork_combat=True).combat_fork(None)
        try:
            res = self.ls.search_live(root.handle, max_sims=self.sims, min_child_sims=2, confirm_pairs=0,
                                      time_budget_s=120.0, seed=int(self.rng.integers(1 << 30)))
        finally:
            root.close() if hasattr(root, "close") else self.native.destroy(root.handle)
        want = (res.best.action, res.best.target)
        for i, lab in enumerate(labels):
            if (kind == "combat" and self._action_of(lab) == want) or (kind != "combat" and self._select_action(lab) == want[0]):
                return i
        return None

    def _apply_combat(self, kind: str, labels: list, idx: int) -> str | None:
        if kind == "card_reward_combat":
            return f"unmapped_apply:{kind}"
        a = self._action_of(labels[idx]) if kind == "combat" else (self._select_action(labels[idx]), -1)
        if a is None or a[0] is None:
            return f"unmapped_label:{labels[idx][:1]}"
        env = self.s.env
        st = self.native.run_step(self.s.handle, a[0], a[1], env._run_obs_buf, env._run_rew_buf,
                                  env._run_terminal_buf, env._run_truncated_buf)
        return None if st == 0 else f"refused:{a}"


# ------------------------------------------------------------------ the referee-only fallback


def referee_only_policy(msg: dict) -> int:
    """Scored-not-ranked play: the first card play, else the first non-end action, else 0. Deterministic."""
    acts = msg["actions"]
    for i, a in enumerate(acts):
        if isinstance(a, list) and a and a[0] == "play":
            return i
    for i, a in enumerate(acts):
        if not (isinstance(a, list) and a and a[0] == "end"):
            return i
    return 0


def play_run(env, seed: str, *, ascension: int = 10, sims: int = 60, emu_factory=None, max_decisions: int = 4000):
    """One run on a GameEnv under the PLAY agent. Returns (end message, Lockstep)."""
    emu = (emu_factory or (lambda s: EmuLock(s, ascension, sims)))(seed)
    ls = Lockstep(emu, referee_only_policy, seed, sims)
    try:
        m, done = env.reset(seed, hash=True)
        n = 0
        while not done and n < max_decisions:
            m, done = env.step(ls.decide(m))
            n += 1
        return m, ls
    finally:
        if emu is not None and hasattr(emu, "close"):
            emu.close()
