"""Oracle-free invariants over recorded fights: what must hold at every step whatever the game does.

The same checks run on the game's recording (referee out.json) and on the emulator's replay of it
(``EmuDump fight`` output), through one neutral frame. A violation in the game's recording is a
recording or driver issue; one in the emulator's replay is an emulator bug or a tool artefact.

  hp_range        0 <= hp <= max_hp for the player and every enemy; max_hp > 0
  energy          0 <= energy <= ENERGY_CAP
  block           0 <= block <= BLOCK_CAP
  power_amount    no power or buff held at amount 0; negative only for the stats that can go negative
  power_dup       one entry per power on a creature (the instanced powers may repeat)
  hand_limit      hand <= 10 cards; pile sizes sane; every card id known
  enemy_count     1 <= enemies <= MAX_ENEMIES while in combat
  rng_monotone    each RNG stream's call count never falls and never skips from present to absent
  turn_monotone   the round counter never falls and moves by at most 1 per step
  dead_stay_dead  an enemy at 0 HP stays there (the few revivers are listed)
  conservation    cards across hand/draw/discard/exhaust change only by a known event: a Power leaves
                  when played, a card is turned into another (upgrade, transform), or a game model
                  that can create that card is present (data/card_flow.json, from the decompile)
  refused_play    a play the game refused (played=false) changes nothing but RNG
  action_card     the card an action names is the card that was in the hand at that index
  draw_order      with no shuffle in between, the draw pile keeps its order (cards leave or join, none move)
  hand_order      a play leaves the rest of the hand in order
  exhaust_grows   the exhaust pile only grows (the cards that pull from it are listed)
  (emulator only) counters_nonneg, combat_counter_monotone, orbs
  (emulator live state, C#) EmuDump invariants: what no snapshot shows, after every replayed step

    python -m tools.fidelity.invariants run --jobs D:/spireline/referee/jobs \
        --emudump tools/emudump/bin/Release/net9.0/EmuDump.dll --work <scratch dir> [--json report.json]
    python -m tools.fidelity.invariants game <out.json>...      # the game's recordings only
    python -m tools.fidelity.invariants emu <out.json> <emu.json>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from .card_flow import card_types
from .fight_compare import NOT_POWERS, names, power

ROOT = Path(__file__).resolve().parents[2]
HAND_LIMIT = 10
MAX_PILE = 150          # cards in one pile; the biggest recorded deck plus generated cards is far below
MAX_ENEMIES = 8
ENERGY_CAP = 30
BLOCK_CAP = 999
# Which powers may be negative, and which a creature may hold several of, come from the game's
# classes (data/card_flow.json). The temporary-stat powers are the game's subclasses of one emulator buff.
EXTRA_SIGNED = {"TEMPORARY_STRENGTH", "TEMPORARY_DEXTERITY", "TEMPORARY_FOCUS", "SLOW_COUNT", "NEXT_TURN_ENERGY",
                "CRIMSON_MANTLE_SELF_DAMAGE"}
# Powers that play a card for the player on their own.
AUTO_PLAYERS = {"MAYHEM", "HELLRAISER", "STAMPEDE", "CASCADE", "AUTOMATION"}
# Enemies that return from 0 HP on their own.
REVIVERS = {"DECIMILLIPEDE", "DECIMILLIPEDE_SEGMENT", "KNOWLEDGE_DEMON", "BYGONE_EFFIGY", "FOGMOG", "TERROR_EEL"}
# Streams the emulator reports as -1 when the fight has not created them.
ABSENT = -1


class Frame:
    """One step's state in the terms both sides share."""

    __slots__ = (
        "action",
        "block",
        "enemies",
        "energy",
        "hp",
        "in_combat",
        "max_energy",
        "max_hp",
        "piles",
        "powers",
        "raw",
        "relics",
        "rng",
        "selecting",
        "side",
        "turn",
    )

    def __init__(self, **kw: object) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def cards(self) -> collections.Counter:
        c: collections.Counter = collections.Counter()
        for p in self.piles.values():
            c.update(p)
        return c


def _powers(raw, side: str) -> list:
    out = []
    for p in raw:
        if side == "game":
            n, a = p
        else:
            n, a = p["name"], p["amount"]
        out.append((power(n), a))
    return out


def game_frame(step: dict) -> Frame:
    st = step["state"]
    a = step.get("action")
    if not st.get("in_combat"):
        return Frame(in_combat=False, side="game", action=a)
    p = st["player"]
    return Frame(
        in_combat=True, side="game", action=a, turn=st.get("round"),
        hp=p["hp"], max_hp=p["max_hp"], block=p["block"], energy=p["energy"], max_energy=None,
        powers=_powers(p["powers"], "game"),
        enemies=[{"name": e["name"], "hp": e["hp"], "max_hp": e["max_hp"], "block": e["block"],
                  "powers": _powers(e["powers"], "game")} for e in st["enemies"]],
        piles={k: [(c[0], bool(c[1])) for c in st[k]] for k in ("hand", "draw", "discard", "exhaust")},
        rng={k: v["counter"] for k, v in st["rng"].items()}, relics=None, raw=st)


def emu_frame(step: dict, nm: dict) -> Frame:
    st = step["state"]
    a = step.get("action")
    if "snapshot" not in st or not st.get("in_combat"):
        return Frame(in_combat=False, side="emu", action=a)
    s = st["snapshot"]
    p = s["player"]
    card = lambda c: (nm["cards"].get(c["def_id"], f"?{c['def_id']}"), bool(c["upgraded"]))  # noqa: E731
    return Frame(
        in_combat=True, side="emu", action=a, turn=s["turn"],
        hp=p["hp"], max_hp=p["max_hp"], block=p["block"], energy=p["energy"], max_energy=p.get("max_energy"),
        powers=_powers(p["buffs"], "emu"),
        enemies=[{"name": nm["enemies"].get(e["def_id"], f"?{e['def_id']}"), "hp": e["hp"], "max_hp": e["max_hp"],
                  "block": e["block"], "powers": _powers(e["buffs"], "emu")} for e in s["enemies"]],
        piles={"hand": [card(c) for c in s["piles"]["hand"]],
               "draw": [(nm["cards"].get(d, f"?{d}"), bool(u)) for d, u in st["draw_order"]],
               "discard": [card(c) for c in s["piles"]["discard"]],
               "exhaust": [card(c) for c in s["piles"]["exhaust"]]},
        rng=st["rng"], relics=[nm["relics"].get(r["id"], f"?{r['id']}") for r in s["relics"]], raw=st,
        selecting=s["pending_selection"] is not None)


class Judge:
    """Holds the tables the checks need; ``fight`` returns the violations of one fight's frames."""

    def __init__(self, id_map: dict | None = None, flow: dict | None = None, types: dict | None = None) -> None:
        self.nm = names(id_map or json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
        self.flow = flow if flow is not None else json.loads((ROOT / "data" / "card_flow.json").read_text(encoding="utf-8"))
        self.types = types if types is not None else card_types(ROOT / "src" / "Sts2Emulator" / "Generated" / "Cards.g.cs")
        self.cards_known = set(self.types)
        self.creators = self.flow["creators"]
        self.removers = {k: set(v) for k, v in self.flow["removers"].items()}
        self.signed = set(self.flow["powers"]["allow_negative"]) | EXTRA_SIGNED
        self.instanced = set(self.flow["powers"]["instanced"]) | {"TEMPORARY_STRENGTH"}

    # ---- one frame -------------------------------------------------------------------------------
    def frame(self, f: Frame) -> list:
        v = []
        if not f.in_combat:
            return v

        def bad(check, detail):
            v.append((check, detail))

        if f.max_hp <= 0 or not 0 <= f.hp <= f.max_hp:
            bad("hp_range", f"player hp {f.hp}/{f.max_hp}")
        if not 0 <= f.energy <= ENERGY_CAP:
            bad("energy", f"energy {f.energy}")
        if f.max_energy is not None and f.energy > max(f.max_energy, 3) + 12:
            bad("energy", f"energy {f.energy} far above max {f.max_energy}")
        if not 0 <= f.block <= BLOCK_CAP:
            bad("block", f"player block {f.block}")
        self._powers("player", f.powers, bad)
        if not f.enemies:
            bad("no_enemies", "in combat with no enemy left")
        elif len(f.enemies) > MAX_ENEMIES:
            bad("enemy_count", f"{len(f.enemies)} enemies")
        for i, e in enumerate(f.enemies):
            if e["max_hp"] <= 0 or not 0 <= e["hp"] <= e["max_hp"]:
                bad("hp_range", f"enemy {i} {e['name']} hp {e['hp']}/{e['max_hp']}")
            if not 0 <= e["block"] <= BLOCK_CAP:
                bad("block", f"enemy {i} {e['name']} block {e['block']}")
            self._powers(f"enemy {i} {e['name']}", e["powers"], bad)
        if len(f.piles["hand"]) > HAND_LIMIT:
            bad("hand_limit", f"hand {len(f.piles['hand'])}")
        for k, p in f.piles.items():
            if len(p) > MAX_PILE:
                bad("hand_limit", f"{k} pile {len(p)}")
            for c in p:
                if c[0] not in self.cards_known:
                    bad("hand_limit", f"unknown card {c[0]} in {k}")
        if f.side == "emu":
            self._emu_frame(f, bad)
        return v

    def _powers(self, who, powers, bad):
        seen = collections.Counter()
        for name, amount in powers:
            seen[name] += 1
            if name in NOT_POWERS and who == "player":
                continue  # emulator-internal counters, see fight_compare.NOT_POWERS
            if amount == 0:
                bad("power_amount", f"{who} holds {name} at 0")
            elif amount < 0 and name not in self.signed:
                bad("power_amount", f"{who} holds {name} at {amount}")
        for name, n in seen.items():
            if n > 1 and name not in self.instanced:
                bad("power_dup", f"{who} holds {name} {n} times")

    def _emu_frame(self, f: Frame, bad) -> None:
        s = f.raw["snapshot"]
        for k, x in s["counters"].items():
            if isinstance(x, int) and not isinstance(x, bool) and x < 0:
                bad("counters_nonneg", f"counter {k} = {x}")
        pl = s["player"]
        if len(pl["orbs"]) > max(pl["orb_capacity"], 0):
            bad("orbs", f"{len(pl['orbs'])} orbs in {pl['orb_capacity']} slots")

    # ---- one step to the next --------------------------------------------------------------------
    def transition(self, a: Frame, b: Frame, act: dict | None) -> list:
        v = []
        if not (a.in_combat and b.in_combat):
            return v
        for k in sorted(set(a.rng) | set(b.rng)):
            x, y = a.rng.get(k, ABSENT), b.rng.get(k, ABSENT)
            if (x != ABSENT and y == ABSENT) or (x != ABSENT and y < x):
                v.append(("rng_monotone", f"{k} {x} -> {y}"))
        if a.turn is not None and b.turn is not None and not 0 <= b.turn - a.turn <= 1:
            v.append(("turn_monotone", f"turn {a.turn} -> {b.turn}"))
        for i, (ea, eb) in enumerate(zip(a.enemies, b.enemies)):
            if ea["hp"] <= 0 < eb["hp"] and ea["name"] not in REVIVERS:
                v.append(("dead_stay_dead", f"enemy {i} {ea['name']} back from 0 to {eb['hp']}"))
        if act and act.get("played") is False:
            for what in ("hp", "block", "energy"):
                if getattr(a, what) != getattr(b, what):
                    v.append(("refused_play", f"{what} {getattr(a, what)} -> {getattr(b, what)}"))
            if a.piles != b.piles:
                v.append(("refused_play", "piles changed"))
        elif act and act.get("kind") == "play":
            hand = a.piles["hand"]
            i = act.get("hand_index")
            if a.side == "game" and (i is None or i >= len(hand) or hand[i][0] != act.get("card")):
                # the emulator's own hand may differ from the game's once they diverge; only the recording is judged
                v.append(("action_card", f"{act.get('card')} not at hand[{i}] ({[c[0] for c in hand]})"))
            v.extend(self._hand_order(a, b, act))
        v.extend(self._draw_order(a, b))
        v.extend(self._exhaust(a, b, act))
        if a.side == "emu":
            v.extend(self._combat_counters(a, b))
        v.extend(self.conservation(a, b, act))
        return v

    @staticmethod
    def _subsequence(small: list, big: list) -> bool:
        it = iter(big)
        return all(x in it for x in small)

    @staticmethod
    def _swapped(a: Frame, b: Frame, pile: str) -> set:
        """Cards (name, upgrade) that left ``pile`` for another pile while a twin came the other way.

        Such a card cannot be followed through the step, so the order checks leave it out.
        """
        gain, loss = collections.Counter(), collections.Counter()
        for k in a.piles:
            if k != pile:
                before, after = collections.Counter(a.piles[k]), collections.Counter(b.piles[k])
                gain += after - before
                loss += before - after
        return set(gain) & set(loss)

    @classmethod
    def _order_kept(cls, before: list, after: list, skip: set) -> bool:
        """Whether a pile that only lost cards, or only gained them, kept the order of the rest.

        A step that both removes and adds cards (draw two, put one back) cannot be judged: a twin could
        be the one that left or the one that came, and the order of the rest depends on which.
        """
        before, after = [c for c in before if c not in skip], [c for c in after if c not in skip]
        cb, ca = collections.Counter(before), collections.Counter(after)
        if not cb - ca:
            return cls._subsequence(before, after)
        if not ca - cb:
            return cls._subsequence(after, before)
        return True

    def _hand_order(self, a: Frame, b: Frame, act: dict) -> list:
        # A play that draws and makes nothing takes its card out and leaves the rest of the hand as it was.
        # (With draws or moves, a twin of the card that left may be the one that came: undecidable.)
        hand = a.piles["hand"]
        i = act.get("hand_index")
        if i is None or i >= len(hand) or hand[i][0] != act.get("card"):
            return []  # the emulator's hand is not the one the action was recorded against
        rest = hand[:i] + hand[i + 1:]
        after = b.piles["hand"]
        if collections.Counter(rest) != collections.Counter(after) or rest == after:
            return []
        return [("hand_order", f"hand reordered by a play of {act.get('card')}: {[c[0] for c in hand]} -> "
                               f"{[c[0] for c in after]}")]

    def _draw_order(self, a: Frame, b: Frame) -> list:
        # With no shuffle in between, the draw pile only loses cards from the top or gains new ones; it
        # never reorders what stays.
        if a.rng.get("Shuffle", ABSENT) != b.rng.get("Shuffle", ABSENT):
            return []
        if self._order_kept(a.piles["draw"], b.piles["draw"], self._swapped(a, b, "draw")):
            return []
        return [("draw_order", f"draw pile reordered without a shuffle: {[c[0] for c in a.piles['draw']][:8]} -> "
                               f"{[c[0] for c in b.piles['draw']][:8]}")]

    PULLS_FROM_EXHAUST = {"EXHUME"}

    def _exhaust(self, a: Frame, b: Frame, act: dict | None) -> list:
        gone = collections.Counter(c[0] for c in a.piles["exhaust"]) - collections.Counter(c[0] for c in b.piles["exhaust"])
        if not gone or self._played(a, act) in self.PULLS_FROM_EXHAUST:
            return []
        return [("exhaust_grows", f"left the exhaust pile: {sorted(gone.elements())} (played {self._played(a, act)})")]

    @staticmethod
    def _combat_counters(a: Frame, b: Frame) -> list:
        ca, cb = a.raw["snapshot"]["counters"], b.raw["snapshot"]["counters"]
        return [("combat_counter_monotone", f"{k} {ca[k]} -> {cb[k]}") for k in sorted(ca)
                if k.endswith("_this_combat") and isinstance(ca[k], int) and k in cb and cb[k] < ca[k]]

    def _type(self, name: str) -> str:
        return self.types.get(name, {}).get("type", "?")

    @staticmethod
    def _played(a: Frame, act: dict | None) -> str | None:
        """Name the card an action plays: the one in this side's hand at that index.

        The emulator's hand may differ from the one the game recorded the action against.
        """
        if not act or act.get("kind") != "play":
            return None
        hand, i = a.piles["hand"], act.get("hand_index")
        if a.side == "emu":  # one past the hand is the end of the turn, whatever card the game named
            return hand[i][0] if i is not None and i < len(hand) else None
        return hand[i][0] if i is not None and i < len(hand) else act.get("card")

    def _causes(self, a: Frame, b: Frame, act: dict | None, removed_powers: list) -> list:
        """Collect the game models whose presence could explain a card appearing in this step."""
        c = []
        if a.selecting:
            # A card selection was open, so this action answers it: a generated option (Splash, Discovery)
            # arrives in the hand. The replay leaves a screen open when the recorded choice cannot be applied.
            c.append({"named": [], "open": True})
        played = self._played(a, act)
        if played:
            c.append(self.creators.get("card", {}).get(played))
        for name, _ in a.powers + b.powers:
            c.append(self.creators.get("power", {}).get(name))
        for r in (a.relics or []) + (b.relics or []):
            c.append(self.creators.get("relic", {}).get(r))
        if a.relics is None and not (act and act.get("kind") == "play"):
            # The game's recording does not list relics. Relics that make cards do it as a turn starts, so
            # across an end of turn any of them may have, while a single play is explained by what it names.
            c.extend(self.creators.get("relic", {}).values())
        for e in a.enemies + b.enemies:
            c.append(self.creators.get("monster", {}).get(e["name"]))
        for name in removed_powers:  # a Power auto-played this step (Mayhem, Cascade...) may create too
            c.append(self.creators.get("card", {}).get(name))
        return [x for x in c if x]

    def conservation(self, a: Frame, b: Frame, act: dict | None) -> list:
        before, after = a.cards(), b.cards()
        rem, add = before - after, after - before
        if not rem and not add:
            return []
        v = []
        # Upgrade and transform first: a removed card paired with an added one keeps the count (the same
        # name for an upgrade; any for a transform, whose cause is not checked).
        left_rem, left_add = list(rem.elements()), list(add.elements())
        for c in list(left_rem):
            same = next((x for x in left_add if x[0] == c[0]), None)
            if same:
                left_rem.remove(c)
                left_add.remove(same)
        while left_rem and left_add:
            left_rem.pop()
            left_add.pop()
        # A Power that is played leaves the piles.
        removed_powers = [c[0] for c in left_rem if self._type(c[0]) == "Power"]
        left_rem = [c for c in left_rem if self._type(c[0]) != "Power"]
        removers = {e["name"] for e in a.enemies if e["name"] in self.removers.get("monster", ())}
        auto = sorted({n for n, _ in a.powers + b.powers} & AUTO_PLAYERS)
        played = self._played(a, act)
        for c in left_rem:
            if removers:
                continue
            if auto:
                # A card the player's own power plays for them (Mayhem at the start of the turn, Hellraiser on a
                # draw) is in flight: out of the piles, its effects not yet in the state. A snapshot taken then
                # is a recording issue, not a lost card.
                v.append(("conservation_inflight", f"{c[0]} vanished with {'/'.join(auto)} in play (played {played})"))
            else:
                v.append(("conservation", f"card vanished: {c[0]} (played {played})"))
        if left_add:
            causes = self._causes(a, b, act, removed_powers)
            for c in left_add:
                if any(c[0] in x["named"] or x["open"] for x in causes):
                    continue
                if auto:  # the in-flight card landing, or what it made
                    v.append(("conservation_inflight", f"{c[0]} appeared with {'/'.join(auto)} in play (played {played})"))
                else:
                    v.append(("conservation", f"card appeared with no known cause: {c[0]} (played {played})"))
        return v

    # ---- a whole fight ---------------------------------------------------------------------------
    def fight(self, frames: list) -> list:
        """Violations as (step, check, detail) for one fight."""
        out = []
        for i, f in enumerate(frames):
            for check, detail in self.frame(f):
                out.append((i, check, detail))
            if i:
                for check, detail in self.transition(frames[i - 1], f, f.action):
                    out.append((i, check, detail))
        return out


# ---- corpus --------------------------------------------------------------------------------------
def _fight_key(r: dict) -> str:
    acts = [s.get("action") for s in r["steps"]]
    ch = r.get("choices")
    return hashlib.md5(json.dumps([r["seed"], r.get("ascension"), r.get("deck_add"), r.get("neow_pick"),  # noqa: S324
                                   r.get("room"), acts, ch], sort_keys=True).encode()).hexdigest()


def clean_fights(paths: list, dedupe: bool = True):
    """Yield (file name, fight) for every clean recorded fight, each distinct fight once."""
    seen = set()
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        for r in doc["results"]:
            if r.get("error") or not r.get("steps") or r.get("ascension") is None:
                continue
            k = _fight_key(r)
            if dedupe and k in seen:
                continue
            seen.add(k)
            yield Path(p).name, r


def check_game(judge: Judge, fight: dict) -> list:
    return judge.fight([game_frame(s) for s in fight["steps"]])


def check_emu(judge: Judge, emu: dict) -> list:
    return judge.fight([emu_frame(s, judge.nm) for s in emu.get("steps", [])])


class Report:
    def __init__(self, label: str) -> None:
        self.label = label
        self.fights = 0
        self.steps = 0
        self.by_check: collections.Counter = collections.Counter()
        self.fights_with: collections.Counter = collections.Counter()
        self.examples: dict = collections.defaultdict(list)
        self.kinds: dict = collections.defaultdict(collections.Counter)

    def add(self, seed: str, source: str, steps: int, violations: list) -> None:
        self.fights += 1
        self.steps += steps
        for check in {c for _, c, _ in violations}:
            self.fights_with[check] += 1
        for step, check, detail in violations:
            self.by_check[check] += 1
            self.kinds[check][re.sub(r"[-\d]+", "#", detail)[:90]] += 1
            if len(self.examples[check]) < 6:
                self.examples[check].append({"seed": seed, "file": source, "step": step, "detail": detail})

    def as_dict(self) -> dict:
        return {"label": self.label, "fights": self.fights, "steps": self.steps,
                "violations": dict(self.by_check), "fights_with": dict(self.fights_with),
                "kinds": {k: v.most_common(8) for k, v in self.kinds.items()}, "examples": dict(self.examples)}

    def print(self) -> None:
        print(f"== {self.label}: {self.fights} fights, {self.steps} steps")
        if not self.by_check:
            print("   no violations")
        for check, n in self.by_check.most_common():
            print(f"   {check:16} {n:6} in {self.fights_with[check]} fights")
            for k, c in self.kinds[check].most_common(4):
                print(f"       {c:5}  {k}")
            for ex in self.examples[check][:2]:
                print(f"       e.g. {ex['seed']} step {ex['step']} ({ex['file']}): {ex['detail'][:110]}")


def run(args) -> int:
    jobs = sorted(str(p) for p in Path(args.jobs).glob("*.out.json"))
    judge = Judge()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    game, emu = Report("game recordings"), Report("emulator replay")
    notes: collections.Counter = collections.Counter()
    shard, idx = [], 0

    def flush() -> None:
        nonlocal shard, idx
        if not shard:
            return
        src = work / f"shard{idx:03d}.json"
        dst = work / f"shard{idx:03d}.emu.json"
        src.write_text(json.dumps({"results": [r for _, r in shard]}), encoding="utf-8")
        with Path(dst).open("w", encoding="utf-8") as fh:
            subprocess.run(["dotnet", args.emudump, "fight", str(src)], stdout=fh, check=True)
        emus = json.loads(dst.read_text(encoding="utf-8"))["results"]
        for (name, g), e in zip(shard, emus):
            game.add(g["seed"], name, len(g["steps"]), check_game(judge, g))
            if e.get("error"):
                notes["emulator replay error"] += 1
                continue
            for st in e["steps"]:
                for c in st.get("choices") or []:
                    if "unexpected_selection" in c:
                        notes["selection left open after a step"] += 1
                    if "error" in c:
                        notes["recorded choice not applicable"] += 1
            emu.add(g["seed"], name, len(e["steps"]), check_emu(judge, e))
        if not args.keep:
            src.unlink()
            dst.unlink()
        shard, idx = [], idx + 1

    for name, r in clean_fights(jobs):
        shard.append((name, r))
        if len(shard) >= args.shard:
            flush()
    flush()
    for rep in (game, emu):
        rep.print()
    print("notes:", dict(notes))
    live = live_pass(args.emudump, args.jobs)
    print(f"== emulator live state (EmuDump invariants): {live['fights']} fights, {live['steps']} steps in combat, "
          f"{live['replay_errors']} replay errors")
    for check, v in live["checks"].items():
        print(f"   {check:20} {v['violations']:6} in {v['fights']} fights")
    for ex in live["examples"][:6]:
        print(f"       e.g. {ex['seed']} step {ex['step']}: {ex['check']}: {ex['detail'][:110]}")
    if not live["checks"]:
        print("   no violations")
    if args.json:
        Path(args.json).write_text(json.dumps({"game": game.as_dict(), "emulator": emu.as_dict(), "live": live,
                                               "notes": dict(notes)}, indent=1), encoding="utf-8")
    return 0


def live_pass(emudump: str, jobs: str) -> dict:
    """Judge the engine's own state after every replayed step of every distinct recorded fight (C#, StateInvariants)."""
    out = subprocess.run(["dotnet", emudump, "invariants", "--dir", jobs], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="every distinct recorded fight, game and emulator replay")
    r.add_argument("--jobs", required=True)
    r.add_argument("--emudump", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--shard", type=int, default=120)
    r.add_argument("--keep", action="store_true")
    r.add_argument("--json")
    m = sub.add_parser("meta", help="metamorphic relations on the emulator alone, over every distinct recorded start")
    m.add_argument("--jobs", required=True)
    m.add_argument("--emudump", required=True)
    m.add_argument("--work", required=True)
    m.add_argument("--seeds", type=int, default=4, help="chooser seeds per fight")
    m.add_argument("--shards", type=int, default=8)
    m.add_argument("--parallel", type=int, default=4)
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--relations")
    m.add_argument("--json")
    g = sub.add_parser("game", help="the game's recordings only")
    g.add_argument("files", nargs="+")
    e = sub.add_parser("emu", help="one recording and its emulator replay")
    e.add_argument("game")
    e.add_argument("emulator")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "meta":
        return meta(a)
    judge = Judge()
    if a.cmd == "game":
        rep = Report("game recordings")
        for name, r_ in clean_fights(a.files, dedupe=False):
            rep.add(r_["seed"], name, len(r_["steps"]), check_game(judge, r_))
        rep.print()
        return 0
    games = json.loads(Path(a.game).read_text(encoding="utf-8"))["results"]
    emus = json.loads(Path(a.emulator).read_text(encoding="utf-8"))["results"]
    rep = Report("emulator replay")
    for g_, e_ in zip(games, emus):
        if not e_.get("error"):
            rep.add(g_["seed"], Path(a.emulator).name, len(e_["steps"]), check_emu(judge, e_))
    rep.print()
    return 0


# ---- metamorphic relations on the emulator alone (EmuDump meta) ------------------------------------
# What a failure of each relation means once it is understood. A rule matches on the relation (prefix),
# the card the failing step played and the first differing path; the first rule that matches decides.
# Anything unmatched is reported "unclassified" and needs a look.
BY_DESIGN = "by design (not a bug)"
ARTEFACT = "tool artefact"
TRIAGE = [
    # (relation prefix, card regex, detail regex, verdict, why). Each reason was checked against the game's class.
    ("pad", r"^MIND_BLAST$", r"Enemies", BY_DESIGN,
     "MindBlast: its damage is the size of the draw pile (CalculatedDamageVar, multiplier = pile count)"),
    ("pad", r"^(SEEKER_STRIKE|HIDDEN_GEM|SECRET_WEAPON|SECRET_TECHNIQUE)$", r".", BY_DESIGN,
     "the card picks from the draw pile (StableShuffle / NextItem on CombatCardSelection, or a screen over it): the pile is its input"),
    ("pad", r"^END_TURN$", r"CardSelectionRng", BY_DESIGN,
     "MayhemPower auto-plays the top card of the draw pile; a pile reader there (Seeker Strike) rolls over the pad too"),
    ("discardorder", r"^SELECT$", r".", ARTEFACT,
     "an open screen over the discard pile names its cards by position (the relation now stops at a screen)"),
    ("handorder", r"^HAVOC$", r".", ARTEFACT,
     "Havoc plays the top card and an auto-played card answers its own screens with the first option (the relation now stops there)"),
    ("discardorder", r"^HAVOC$", r".", ARTEFACT, "same: the auto-played card chooses from the discard pile by position"),
    ("upgrade", r"^SECRET_WEAPON$", r"blocks", BY_DESIGN,
     "SecretWeapon.OnUpgrade removes Exhaust, so Feel No Pain pays its block only for the plain card"),
    ("linear", r"^BASH$", r"OneTwoPunch", BY_DESIGN,
     "the first play's Vulnerable lands before the replay hits, and Vulnerable multiplies and rounds: not linear"),
    ("monotone", r"^BLOCK$", r"hit points by", ARTEFACT,
     "extra block put on at the start of the turn is read by Body Slam and kin (the relation now adds it just before the enemies' turn)"),
]


def triage(f: dict) -> tuple:
    for rel, card, detail, verdict, why in TRIAGE:
        if f["relation"].startswith(rel) and re.search(card, f["card"]) and re.search(detail, f["detail"]):
            return verdict, why
    return "unclassified", ""


def _start_key(r: dict) -> str:
    return "|".join(map(str, [r["seed"], r.get("ascension"), json.dumps(r.get("deck_add")), r.get("neow_pick"),
                              json.dumps(r.get("room"))]))


def meta(args) -> int:
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    jobs = sorted(str(p) for p in Path(args.jobs).glob("*.out.json"))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    starts, seen = [], set()
    for p in jobs:
        for r in json.loads(Path(p).read_text(encoding="utf-8"))["results"]:
            if r.get("error") or not r.get("steps") or r.get("ascension") is None:
                continue
            k = _start_key(r)
            if k not in seen:
                seen.add(k)
                starts.append({k_: r[k_] for k_ in ("seed", "ascension", "deck_add", "neow_pick", "room", "choices", "steps")
                               if k_ in r} | {"steps": []})
    starts = starts[: args.limit] if args.limit else starts
    shards = [starts[i :: args.shards] for i in range(args.shards)]
    files = []
    for i, sh in enumerate(shards):
        f = work / f"meta{i:02d}.json"
        f.write_text(json.dumps({"results": sh}), encoding="utf-8")
        files.append(f)

    def one(f: Path) -> dict:
        cmd = ["dotnet", args.emudump, "meta", str(f), "--chooser-seeds", str(args.seeds)]
        if args.relations:
            cmd += ["--relations", args.relations]
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if out.returncode != 0:  # one retry: a process killed from outside (memory, an updating host) is not a finding
            out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if out.returncode != 0:
            return {"crash": f"exit {out.returncode} {out.stderr[-300:]!r} {out.stdout[-100:]!r}", "failures": [], "instances": {}, "fights": 0, "opened": 0, "unopenable": 0,
                    "open_errors": {}, "ms": {}}
        return json.loads(out.stdout)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(one, files))
    total = collections.Counter()
    inst, ms, errors = collections.Counter(), collections.Counter(), collections.Counter()
    failures = []
    for r in results:
        for k in ("fights", "opened", "unopenable"):
            total[k] += r[k]
        inst.update(r["instances"])
        ms.update(r["ms"])
        errors.update(r["open_errors"])
        failures += r["failures"]
        if "crash" in r:
            errors["EmuDump crashed: " + r["crash"][-120:].replace("\n", " ")] += 1
    for f in failures:
        f["verdict"], f["why"] = triage(f)
    print(f"== metamorphic: {total['fights']} distinct starts, {total['opened']} opened, {total['unopenable']} not")
    print("   instances:", dict(inst))
    for k, n in errors.items():
        print(f"   open error x{n}: {k[:140]}")
    rel_fail = collections.Counter(f["relation"] for f in failures)
    for rel, n in rel_fail.most_common():
        cards = collections.Counter(f["card"] for f in failures if f["relation"] == rel)
        verdicts = collections.Counter(f["verdict"] for f in failures if f["relation"] == rel)
        print(f"   {rel:34} {n:5} failures of {inst.get(rel.split(':')[0], '?')} checks; verdicts {dict(verdicts)}")
        print(f"       cards: {', '.join(f'{c} x{k}' for c, k in cards.most_common(8))}")
        paths = collections.Counter(re.sub(r"\[\d+\]", "[#]", f["detail"].split(":")[0])[:70] for f in failures
                                    if f["relation"] == rel)
        print(f"       first path: {', '.join(f'{p} x{k}' for p, k in paths.most_common(4))}")
    if args.json:
        Path(args.json).write_text(json.dumps({"fights": dict(total), "instances": dict(inst), "ms": dict(ms),
                                               "open_errors": dict(errors), "failures": failures}, indent=1),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
