"""The rule probes: small scenarios whose outcome in the real game settles a disputed rule.

Each probe is a setup fight (referee/driver/Setup.cs: the developers' console commands, so every state
is one the game's own code builds) plus a scripted list of actions. One step of it is measured. Each
hypothesis predicts that measurement from the game's state before the step, so the random opening
hand does not matter; the game's value names the rule the game follows, and the emulator's value (the
same step re-anchored on the game's state, tools/emudump/Reanchor.cs) names the one it follows.

Rules here are only stated, never copied: the class and method the question is about are named so a
reader can check them in their own decompile.

A probe's functions get V views (referee/probe.py): pre is always the GAME's state before the step.
"""

# ruff: noqa: ARG005  (every hypothesis and measure takes the same (pre, ctx) / (pre, post, ctx) arguments)
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SEED = "P9PM918FBG"  # Neow option 1 is GOLDEN_PEARL (gold only), so nothing from Neow acts in a fight
NEOW = 1


def card(cid: str, pile: str = "Hand", pos: str | None = None, upgraded: bool = False) -> dict:
    """A card made in the fight's scope and added to a pile (CardConsoleCmd); "top" of the draw pile is drawn next."""
    it = {"cmd": "card", "id": cid, "pile": pile}
    if pos:
        it["pos"] = pos
    if upgraded:
        it["upgraded"] = True
    return it


def deck_set(cards: list[dict]) -> dict:
    """The deck exactly as listed, before the fight (RefsetSetup.cs deck_set: CreateCard, CardCmd.Upgrade, CardCmd.Enchant)."""
    return {"cmd": "deck_set", "cards": cards}


def dcard(cid: str, enchantment: str = "", amount: int = 0, n: int = 1, upgraded: bool = False) -> list[dict]:
    return [{"id": cid, "upgraded": upgraded, "enchantment": enchantment, "amount": amount} for _ in range(n)]


def power(pid: str, amount: int, target: str = "player") -> dict:
    return {"cmd": "power", "id": pid, "amount": amount, "target": target}


def relic(rid: str) -> dict:
    return {"cmd": "relic", "id": rid}


def potion(pid: str) -> dict:
    return {"cmd": "potion", "id": pid}


def fight(eid: str) -> dict:
    return {"cmd": "fight", "id": eid}


def energy(n: int) -> dict:
    return {"cmd": "energy", "amount": n}


def play(cid: str, target: int = -1) -> dict:
    """A play by card id: the driver plays the first playable copy in hand (Setup.cs Scripted)."""
    return {"kind": "play", "card": cid, "hand_index": -1, "target_slot": target}


def end_turn() -> dict:
    return {"kind": "end_turn"}


def drink(pid: str, slot: int = 0, target: int = -1) -> dict:
    return {"kind": "potion", "potion": pid, "slot": slot, "target_slot": target}


@dataclass
class Hyp:
    name: str
    answer: str  # the rule this outcome means, in words
    predict: Callable[..., Any]  # (pre, ctx) -> the measurement this rule gives


@dataclass
class Probe:
    id: str
    ledger: list[str]
    question: str
    setup: list[dict]
    script: list[dict]
    measure: Callable[..., Any]  # (pre, post, ctx) -> value; post is None when that side has no state
    hyps: list[Hyp] = field(default_factory=list)
    at: int | Callable[..., int | None] | None = None  # step measured (1 = after the first action); default the last
    need: Callable[..., str | None] | None = None  # (pre, ctx) -> why the scenario is not reached, or None
    evidence: Callable[..., str | None] | None = None  # (game trace of the step) -> what the game did, in order
    game_class: str = ""  # where the rule lives in the game (class.method), for the reader's own check
    ends: bool = False  # the measured step may end the fight (the measurement is then how it ended)
    note: str = ""  # what the probe cannot see, or a representation difference it works around

    def spec(self) -> dict:
        return {"kind": "combat", "seed": SEED, "character": "IRONCLAD", "ascension": 10, "neow": NEOW,
                "chooser_seed": 1, "max_actions": len(self.script), "setup": self.setup, "script": self.script,
                "use_potions": True, "trace": True, "tag": "probe:" + self.id}

    def step(self, game_steps: list) -> int | None:
        if callable(self.at):
            return self.at(game_steps)
        return self.at if self.at is not None else len(self.script)


# ------------------------------------------------------------------ measurement helpers


def dealt(i: int) -> Callable:
    """HP plus block enemy i lost (by its place among the living before the step)."""
    def m(pre, post, ctx):
        if post is None or i >= len(pre.enemies):
            return None
        a = pre.enemies[i]
        b = post.enemy_like(a, i, pre)
        return None if b is None else (a["hp"] + a["block"]) - (b["hp"] + b["block"])
    return m


def attackers(pre) -> list[int]:
    return [i for i, e in enumerate(pre.enemies) if (e.get("intent") or {}).get("kind") == "ATTACK"]


def events(trace: list, kind: str, name: str | None = None) -> list[int]:
    return [i for i, ev in enumerate(trace or []) if ev and ev[0] == kind and (name is None or (len(ev) > 1 and ev[1] == name))]


def damage_lines(trace: list) -> list[str]:
    """The step's DamageReceived entries, in the game's order: receiver <- dealer unblocked+blocked."""
    out = []
    for ev in trace or []:
        if ev and ev[0] == "E" and len(ev) > 1:
            if ev[1] == "DamageReceived":
                out.append(f"{ev[2]}<-{ev[3]} {ev[4]}+{ev[5]}")
            elif ev[1] in ("CreatureDied", "Died", "Death", "CreatureKilled"):
                out.append(f"died {ev[2]}")
    return out


def trace_digest(trace: list, keep=("DamageReceived", "BlockGained", "PowerReceived", "CardDrawn", "CardExhausted",
                                    "CardGenerated", "EnergySpent"), hooks=(), limit: int = 14) -> str | None:
    """A compact, ordered list of what the game did in the step (combat log entries, chosen hooks, marks)."""
    out = []
    for ev in trace or []:
        if not ev:
            continue
        if ev[0] == "E" and len(ev) > 1 and (ev[1] in keep or ev[1].endswith("Died") or "Death" in ev[1]):
            out.append(":".join(str(x) for x in ev[1:] if x is not None and x != ""))
        elif ev[0] == "H" and len(ev) > 1 and ev[1] in hooks:
            out.append("hook " + ev[1])
        elif ev[0] == "M" and len(ev) > 1 and ev[1] in ("play", "played", "moved"):
            out.append(f"{ev[1]} {ev[2]}")
    if not out:
        return None
    return " | ".join(out[:limit]) + (" | ..." if len(out) > limit else "")


def first_shuffle_step(steps: list) -> int | None:
    """The first step whose Shuffle counter rose (the draw found the pile empty)."""
    for j in range(1, len(steps)):
        a = steps[j - 1]["state"].get("rng", {}).get("Shuffle", {}).get("counter")
        b = steps[j]["state"].get("rng", {}).get("Shuffle", {}).get("counter")
        if a is not None and b is not None and b > a:
            return j
    return None


def hug_order(trace: list) -> str | None:
    """In what order the game shuffles, runs Hook.AfterShuffle, makes the Soot and places it, by RNG caller."""
    tr = trace or []
    shuffle = [i for i, ev in enumerate(tr) if ev and ev[0] == "R" and len(ev) > 3 and ev[1] == "Shuffle" and "Shuffle" in str(ev[3])]
    place = [i for i, ev in enumerate(tr) if ev and ev[0] == "R" and len(ev) > 3 and ev[1] == "Shuffle" and "Shuffle" not in str(ev[3])]
    soot = [i for i, ev in enumerate(tr) if ev and ev[0] == "E" and len(ev) > 2 and ev[1] == "CardGenerated" and ev[2] == "SOOT"]
    hook = events(tr, "H", "AfterShuffle")
    drawn = [i for i, ev in enumerate(tr) if ev and ev[0] == "E" and len(ev) > 2 and ev[1] == "CardDrawn" and ev[2] == "SOOT"]
    if not shuffle or not soot or not hook:
        return f"shuffle draws {len(shuffle)}, AfterShuffle {len(hook)}, Soot made {len(soot)}"
    order = sorted([(shuffle[0], f"shuffle ({len(shuffle)} draws)"), (hook[0], "Hook.AfterShuffle"), (soot[0], "Soot made")]
                   + [(i, f"Soot placed ({tr[i][3].split(' <')[0]} draws Shuffle)") for i in place[:1]]
                   + [(i, "Soot drawn") for i in drawn if i > soot[0]][:1])
    return " -> ".join(x for _, x in order)


def blow_order(trace: list) -> str | None:
    """Which landed first in the step: the player's retaliation on the attacker, or the attacker's blow."""
    lines = damage_lines(trace)
    hits = [k for k, x in enumerate(lines) if x.startswith("player<-")]
    back = [k for k, x in enumerate(lines) if not x.startswith("player<-") and "<-player" in x]
    if not hits or not back:
        return " | ".join(lines[:8]) or None
    first = "the retaliation (Thorns) lands before the blow" if back[0] < hits[0] else "the blow lands before the retaliation"
    return first + ": " + " | ".join(lines[:6])


# ------------------------------------------------------------------ the probes

PROBES: list[Probe] = [
    Probe(
        "P01", ["C1-F10"], "Does Pen Nib double the attack after Strength is added (not before)?",
        [relic("PEN_NIB"), fight("DEVOTED_SCULPTOR_WEAK"), power("STRENGTH_POWER", 3), energy(7)]
        + [card("POMMEL_STRIKE") for _ in range(5)] + [card("POMMEL_STRIKE", "Draw", "top") for _ in range(5)],
        [play("POMMEL_STRIKE", 0) for _ in range(10)],
        dealt(0),
        [Hyp("add-then-double", "Strength is added, then Pen Nib doubles: (9+S)x2",
             lambda pre, ctx: (9 + pre.power("STRENGTH")) * 2),
         Hyp("double-then-add", "Pen Nib doubles the base, then Strength is added: 9x2+S",
             lambda pre, ctx: 18 + pre.power("STRENGTH")),
         Hyp("not-doubled", "the 10th attack is not doubled", lambda pre, ctx: 9 + pre.power("STRENGTH"))],
        at=10,
        need=lambda pre, ctx: None if "POMMEL_STRIKE" in pre.hand and pre.enemies and pre.power("STRENGTH") > 0
        else "no Pommel Strike in hand, no enemy or no Strength before the 10th attack",
        evidence=lambda tr: trace_digest(tr, keep=("DamageReceived",)),
        game_class="Hook.ModifyDamage (additive hooks, then multiplicative); PenNib.ModifyDamageMultiplicative",
    ),
    Probe(
        "P02", ["C1-F10"], "Does Gigantification triple the attack after Strength is added?",
        [fight("NIBBITS_WEAK"), power("STRENGTH_POWER", 3), power("GIGANTIFICATION_POWER", 1), card("STRIKE_IRONCLAD")],
        [play("STRIKE_IRONCLAD", 0)],
        dealt(0),
        [Hyp("add-then-triple", "Strength is added, then x3: (6+S)x3", lambda pre, ctx: (6 + pre.power("STRENGTH")) * 3),
         Hyp("triple-then-add", "x3 on the base, then Strength: 18+S", lambda pre, ctx: 18 + pre.power("STRENGTH")),
         Hyp("not-tripled", "no x3", lambda pre, ctx: 6 + pre.power("STRENGTH"))],
        evidence=lambda tr: trace_digest(tr, keep=("DamageReceived",)),
        game_class="Hook.ModifyDamage; GigantificationPower.ModifyDamageMultiplicative",
    ),
    Probe(
        "P03", ["C1-F08"], "Does the player's Thorns hit the attacker before its blow, and does the dead attacker's blow still land?",
        [fight("TOADPOLES_WEAK"), power("THORNS_POWER", 100)],
        [end_turn()],
        lambda pre, post, ctx: (pre.hp - post.hp > 0, len(post.enemies) < len(pre.enemies)),
        [Hyp("blow-lands", "the blow lands although the retaliation killed its attacker", lambda pre, ctx: (True, True)),
         Hyp("blow-cancelled", "a blow whose attacker died to the retaliation does not land", lambda pre, ctx: (False, True))],
        need=lambda pre, ctx: None if attackers(pre) else "no enemy intends to attack on turn 1",
        evidence=blow_order,
        game_class="ThornsPower.BeforeDamageReceived; CreatureCmd.Damage",
        note="the order inside the step is in the game's trace only, not in either side's state; P18 makes the order decide the fight",
    ),
    Probe(
        "P04", ["C1-F11"], "Does Gremlin Horn pay once when a reviving boss (Test Subject, Adaptable) is killed by one Strike?",
        [relic("GREMLIN_HORN"), fight("TEST_SUBJECT_BOSS"), power("STRENGTH_POWER", 200), card("STRIKE_IRONCLAD")],
        [play("STRIKE_IRONCLAD", 0)],
        lambda pre, post, ctx: None if post is None else (post.energy - pre.energy, len(post.hand) - len(pre.hand)),
        [Hyp("once", "one death answered: +1 energy, +1 card (net 0, 0 after the Strike)", lambda pre, ctx: (0, 0)),
         Hyp("twice", "the death hooks run twice (net +1, +1)", lambda pre, ctx: (1, 1)),
         Hyp("never", "a reviving death pays nothing (net -1, -1)", lambda pre, ctx: (-1, -1))],
        need=lambda pre, ctx: None if "STRIKE_IRONCLAD" in pre.hand and pre.draw else "no Strike in hand or empty draw pile",
        evidence=lambda tr: trace_digest(tr, keep=("DamageReceived", "CardDrawn", "PowerReceived")),
        game_class="GremlinHorn.AfterDeath; AdaptablePower.AfterDeath / ShouldCreatureBeRemovedFromCombatAfterDeath",
    ),
    Probe(
        "P05", ["C1-F13"], "With Fairy in a Bottle and Lizard Tail, which one saves the player from a lethal blow?",
        [relic("LIZARD_TAIL"), potion("FAIRY_IN_A_BOTTLE"), fight("NIBBITS_WEAK"), power("STRENGTH_POWER", 200, "enemy:0")],
        [end_turn()],
        lambda pre, post, ctx: None if post is None else (
            "dead" if post.hp <= 0 else ("fairy" if "FAIRY_IN_A_BOTTLE" not in (post.potions or []) else "lizard")),
        [Hyp("fairy-first", "ShouldDie (the potion) is asked before ShouldDieLate (Lizard Tail)", lambda pre, ctx: "fairy"),
         Hyp("lizard-first", "Lizard Tail is spent first, the potion is kept", lambda pre, ctx: "lizard")],
        need=lambda pre, ctx: None if attackers(pre) and "FAIRY_IN_A_BOTTLE" in (pre.potions or [])
        else "no attack intended, or the potion is not in the belt",
        evidence=lambda tr: trace_digest(tr, keep=("DamageReceived",), hooks=("AfterPreventingDeath", "AfterPotionUsed")),
        game_class="Hook.ShouldDie / ShouldDieLate; FairyInABottle.ShouldDie; LizardTail.ShouldDieLate",
    ),
    Probe(
        "P06", ["C1-F14"], "Does Unmovable double every block gain of the first block card of the turn (Second Wind's gains)?",
        [fight("NIBBITS_WEAK"), power("UNMOVABLE_POWER", 1), card("DEFEND_IRONCLAD"), card("DEFEND_IRONCLAD"), card("SECOND_WIND")],
        [play("SECOND_WIND")],
        lambda pre, post, ctx: None if post is None else post.block - pre.block,
        [Hyp("whole-card", "gains of the current play are not counted: every gain doubled (10 per card)",
             lambda pre, ctx: 10 * ctx.non_attacks(pre.hand, "SECOND_WIND")),
         Hyp("first-gain", "only the first gain doubled", lambda pre, ctx: 5 * ctx.non_attacks(pre.hand, "SECOND_WIND") + 5),
         Hyp("none", "nothing doubled", lambda pre, ctx: 5 * ctx.non_attacks(pre.hand, "SECOND_WIND"))],
        need=lambda pre, ctx: None if "SECOND_WIND" in pre.hand and pre.block == 0 and ctx.non_attacks(pre.hand, "SECOND_WIND") >= 2
        else "Second Wind missing, block already gained, or fewer than 2 non-attacks",
        evidence=lambda tr: trace_digest(tr, keep=("BlockGained", "CardExhausted")),
        game_class="UnmovablePower.ModifyBlockMultiplicative (BlockGainedEntry count skips the current CardPlay)",
    ),
    Probe(
        "P07", ["C1-F14"], "Does Unmovable leave Plating's end-of-turn block undoubled (not a card or monster move)?",
        [fight("NIBBITS_WEAK"), power("UNMOVABLE_POWER", 1), power("PLATING_POWER", 5)],
        [end_turn()],
        lambda pre, post, ctx: None if post is None else pre.hp - post.hp,
        [Hyp("not-doubled", "Plating's Unpowered block is not doubled", lambda pre, ctx: ctx.hit_after_block(pre, pre.power("PLATING"))),
         Hyp("doubled", "Unmovable doubles Plating's block", lambda pre, ctx: ctx.hit_after_block(pre, 2 * pre.power("PLATING")))],
        need=lambda pre, ctx: None if len(attackers(pre)) == 1 and pre.block == 0 else "not exactly one attacker on turn 1",
        evidence=lambda tr: trace_digest(tr, keep=("BlockGained", "DamageReceived")),
        game_class="UnmovablePower.ModifyBlockMultiplicative (props.IsCardOrMonsterMove); PlatingPower.BeforeSideTurnEndEarly",
    ),
    Probe(
        "P08", ["C1-F29"], "Does Artifact block the Strength a temporary Strength gain (Flex Potion) takes back at end of turn?",
        [potion("FLEX_POTION"), fight("NIBBITS_WEAK"), power("ARTIFACT_POWER", 1)],
        [drink("FLEX_POTION"), end_turn()],
        lambda pre, post, ctx: None if post is None else (post.power("STRENGTH"), post.power("ARTIFACT")),
        [Hyp("artifact-blocks", "the take-back is a debuff: Artifact blocks it and is spent",
             lambda pre, ctx: (pre.power("STRENGTH"), pre.power("ARTIFACT") - 1)),
         Hyp("artifact-ignored", "the take-back bypasses Artifact",
             lambda pre, ctx: (pre.power("STRENGTH") - pre.power("TEMPORARY_STRENGTH"), pre.power("ARTIFACT")))],
        at=2,
        need=lambda pre, ctx: None if pre.power("TEMPORARY_STRENGTH") > 0 and pre.power("ARTIFACT") > 0
        else "the potion's temporary Strength or the Artifact is missing after the drink",
        evidence=lambda tr: trace_digest(tr, keep=("PowerReceived",)),
        game_class="TemporaryStrengthPower.AfterSideTurnEnd; ArtifactPower.TryModifyPowerAmountReceived",
    ),
    Probe(
        "P09", ["C1-F18"], "Does Biiig Hug's Soot join the draw pile after the shuffle (Hook.AfterShuffle runs after CardPileCmd.Shuffle)?",
        [relic("BIIIG_HUG"), fight("FUZZY_WURM_CRAWLER_WEAK")],
        [end_turn(), end_turn(), end_turn()],
        lambda pre, post, ctx: {"Shuffle": post.rng.get("Shuffle"), "soot_in_hand": post.hand.count("SOOT"),
                                "soot_in_draw": post.draw.count("SOOT"), "draw": len(post.draw)},
        [],
        at=first_shuffle_step,
        evidence=hug_order,
        note="no hypothesis predicts the state; the game's answer is its trace (shuffle draws, the hook, the Soot's own placement draw)",
        game_class="CardPileCmd.Shuffle then Hook.AfterShuffle; BiiigHug.AfterShuffle (CardPilePosition.Random)",
    ),
    Probe(
        "P10", ["C1-F27"], "Does Unsettling Lamp double a temporary Strength-down (Dark Shackles)?",
        [relic("UNSETTLING_LAMP"), fight("NIBBITS_WEAK"), card("DARK_SHACKLES")],
        [play("DARK_SHACKLES", 0)],
        lambda pre, post, ctx: None if post is None or not post.enemies else post.enemies[0]["powers"].get("STRENGTH", 0)
        - pre.enemies[0]["powers"].get("STRENGTH", 0),
        [Hyp("doubled", "the lamp doubles the first debuff card's powers: -18", lambda pre, ctx: -18),
         Hyp("plain", "the temporary Strength-down bypasses the lamp: -9", lambda pre, ctx: -9)],
        evidence=lambda tr: trace_digest(tr, keep=("PowerReceived",)),
        game_class="UnsettlingLamp.ModifyPowerAmountGivenMultiplicative; TemporaryStrengthPower.BeforeApplied",
    ),
    Probe(
        "P11", ["C1-F26"], "Does Belt Buckle take its Dexterity away when Alchemize fills the empty belt (Hook.AfterPotionProcured)?",
        [relic("BELT_BUCKLE"), fight("NIBBITS_WEAK"), card("ALCHEMIZE")],
        [play("ALCHEMIZE")],
        lambda pre, post, ctx: (post.power("DEXTERITY"), sum(1 for p in (post.potions or []) if p)),
        [Hyp("procure-hook", "AfterPotionProcured runs: the Dexterity goes with the first potion", lambda pre, ctx: (0, 1)),
         Hyp("dex-kept", "the Dexterity stays though the belt holds a potion", lambda pre, ctx: (pre.power("DEXTERITY"), 1))],
        need=lambda pre, ctx: None if pre.power("DEXTERITY") > 0 and not any(pre.potions or []) and "ALCHEMIZE" in pre.hand
        else "no Dexterity from an empty belt at the start, or no Alchemize",
        evidence=lambda tr: trace_digest(tr, keep=("PowerReceived",), hooks=("AfterPotionProcured",)),
        game_class="BeltBuckle.BeforeCombatStart / AfterPotionProcured; Alchemize.OnPlay (PotionCmd.TryToProcure)",
    ),
    Probe(
        "P12", ["C1-F21"], "When Mangle kills its target, does its Strength loss go to the dead target (not to a survivor)?",
        [fight("TOADPOLES_WEAK"), power("STRENGTH_POWER", 30), card("MANGLE")],
        [play("MANGLE", 0)],
        lambda pre, post, ctx: None if post is None or len(post.enemies) != len(pre.enemies) - 1 else
        sum(e["powers"].get("STRENGTH", 0) for e in post.enemies) - sum(e["powers"].get("STRENGTH", 0) for e in pre.enemies[1:]),
        [Hyp("dead-target", "PowerCmd.Apply goes to the play's own (dead) target: survivors unchanged", lambda pre, ctx: 0),
         Hyp("retarget", "the Strength loss lands on a surviving enemy", lambda pre, ctx: -10)],
        need=lambda pre, ctx: None if len(pre.enemies) >= 2 and "MANGLE" in pre.hand else "fewer than two enemies or no Mangle",
        evidence=lambda tr: trace_digest(tr, keep=("DamageReceived", "PowerReceived")),
        game_class="Mangle.OnPlay (cardPlay.Target for both the hit and PowerCmd.Apply<ManglePower>)",
    ),
    Probe(
        "P13", ["C1-F02"], "Does Inferno fire on the HP the player loses to an enemy's Thorns (unpowered damage runs AfterDamageReceived)?",
        [fight("TOADPOLES_WEAK"), power("INFERNO_POWER", 3), power("THORNS_POWER", 3, "enemy:0"), card("STRIKE_IRONCLAD")],
        [play("STRIKE_IRONCLAD", 0)],
        dealt(1),
        [Hyp("inferno-fires", "the Thorns damage runs the after-damage hooks: Inferno hits every enemy",
             lambda pre, ctx: pre.power("INFERNO")),
         Hyp("silent", "unpowered damage skips Inferno", lambda pre, ctx: 0)],
        need=lambda pre, ctx: None if len(pre.enemies) >= 2 and pre.enemies[0]["powers"].get("THORNS", 0) > 0
        else "no second enemy or no Thorns on the target",
        evidence=lambda tr: " | ".join(damage_lines(tr)[:8]) or None,
        game_class="InfernoPower.AfterDamageReceived; ThornsPower.BeforeDamageReceived (CreatureCmd.Damage, Unpowered)",
    ),
    Probe(
        "P14", ["C1-F15"], "Does Fisticuffs gain block from the whole hit, blocked part included?",
        [fight("SEWER_CLAM_NORMAL"), card("FISTICUFFS")],
        [play("FISTICUFFS", 0)],
        lambda pre, post, ctx: None if post is None else post.block - pre.block,
        [Hyp("total-damage", "TotalDamage (blocked + unblocked) + overkill", lambda pre, ctx: 7 + pre.power("STRENGTH")),
         Hyp("hp-lost", "only the HP the target lost", lambda pre, ctx: max(0, 7 + pre.power("STRENGTH") - pre.enemies[0]["block"]))],
        need=lambda pre, ctx: None if pre.enemies and pre.enemies[0]["block"] > 0 else "the target has no block",
        evidence=lambda tr: trace_digest(tr, keep=("DamageReceived", "BlockGained")),
        game_class="Fisticuffs.OnPlay (DamageResult.TotalDamage + OverkillDamage)",
    ),
    Probe(
        "P15", ["C1-F16"], "Does Bloodletting's HP loss give Self-Forming Clay its next-turn block (and cost exactly 3 HP)?",
        [relic("SELF_FORMING_CLAY"), fight("NIBBITS_WEAK"), card("BLOODLETTING")],
        [play("BLOODLETTING")],
        lambda pre, post, ctx: (pre.hp - post.hp, post.energy - pre.energy, post.power("SELF_FORMING_CLAY") + post.power("BLOCK_NEXT_TURN")),
        [Hyp("clay-triggers", "the self-damage is HP lost: 3 HP, +2 energy, Clay's power 3", lambda pre, ctx: (3, 2, 3)),
         Hyp("clay-silent", "3 HP, +2 energy, no Clay power", lambda pre, ctx: (3, 2, 0))],
        evidence=lambda tr: trace_digest(tr, keep=("DamageReceived", "PowerReceived", "EnergySpent")),
        game_class="Bloodletting.OnPlay (CreatureCmd.Damage, Unblockable|Unpowered|Move); SelfFormingClay.AfterDamageReceived",
        note="next-turn block counted from either power: the game holds it as SELF_FORMING_CLAY_POWER, the emulator as its "
             "BLOCK_NEXT_TURN buff (the game's BlockNextTurnPower is another power)",
    ),
    Probe(
        "P16", ["C1-F05"], "Does Havoc exhaust an ally-targeted card (Demonic Shield) unplayed when there is no ally?",
        [fight("NIBBITS_WEAK"), card("DEMONIC_SHIELD", "Draw", "top"), card("HAVOC")],
        [play("HAVOC")],
        lambda pre, post, ctx: None if post is None else (post.hp - pre.hp, post.block - pre.block, "DEMONIC_SHIELD" in post.exhaust),
        [Hyp("unplayed", "no legal target: exhausted without its effect", lambda pre, ctx: (0, 0, True)),
         Hyp("played", "played on the player: -1 HP", lambda pre, ctx: (-1, 0, True))],
        need=lambda pre, ctx: None if pre.draw[:1] == ["DEMONIC_SHIELD"] and "HAVOC" in pre.hand else "Demonic Shield not on top or no Havoc",
        evidence=lambda tr: trace_digest(tr, keep=("DamageReceived", "BlockGained", "CardExhausted")),
        game_class="CardPileCmd.AutoPlayFromDrawPile (forceExhaust); CardCmd.AutoPlay target resolution",
    ),
    Probe(
        "P17", ["C1-F05"], "Does Havoc's auto-played Power card take effect and leave combat, not the exhaust pile?",
        [fight("NIBBITS_WEAK"), card("INFLAME", "Draw", "top"), card("HAVOC")],
        [play("HAVOC")],
        lambda pre, post, ctx: None if post is None else (post.power("STRENGTH") > pre.power("STRENGTH"), "INFLAME" in post.exhaust),
        [Hyp("power-played", "the Power is played and gone (not exhausted)", lambda pre, ctx: (True, False)),
         Hyp("played-exhausted", "played, then exhausted", lambda pre, ctx: (True, True)),
         Hyp("exhausted-unplayed", "exhausted without its effect", lambda pre, ctx: (False, True))],
        need=lambda pre, ctx: None if pre.draw[:1] == ["INFLAME"] and "HAVOC" in pre.hand else "Inflame not on top or no Havoc",
        evidence=lambda tr: trace_digest(tr, keep=("PowerReceived", "CardExhausted")),
        game_class="CardPileCmd.AutoPlayFromDrawPile (forceExhaust); CardModel play of a Power card",
    ),
]

PROBES.append(Probe(
    "P18", ["C1-F08"], "When Thorns kills the last enemy whose blow is lethal, is the fight won or lost?",
    [fight("NIBBITS_WEAK"), power("THORNS_POWER", 100), power("STRENGTH_POWER", 200, "enemy:0")],
    [end_turn()],
    lambda pre, post, ctx: post.outcome,
    [Hyp("lost", "the lethal blow still lands after the retaliation killed the attacker: the player dies", lambda pre, ctx: "lost"),
     Hyp("won", "the fight ends with the last enemy's death, before its blow", lambda pre, ctx: "won")],
    need=lambda pre, ctx: None if len(pre.enemies) == 1 and attackers(pre) else "not one attacking enemy",
    evidence=blow_order,
    game_class="ThornsPower.BeforeDamageReceived; CreatureCmd.Damage; CombatManager's end-of-combat check",
    ends=True,
    note="a fight that ends leaves the emulator only its run phase: the re-anchored dump cannot say who won, the plain replay can",
))

# The turn start's screens (owner decision 2026-10-14, Selector.cs): with the UI, CombatManager.StartTurn runs on
# while SetupPlayerTurn waits on a screen (Hook.AfterSideTurnStart, the orbs, the win check) and the player picks
# after it. A screen in BeforeHandDraw (Foregone Conclusion) then has the side's turn start run BEFORE the hand
# draw: Draw Cards Next Turn is removed and Clarity ticks down before ModifyHandDraw reads them. Answered at once
# (the old driver), the pick came first and the draw kept the bonus.
PROBES.append(Probe(
    "P19", ["C1-F01"], "With a turn-start screen open (Foregone Conclusion), does the side's turn start run before the "
    "pick, so Draw Cards Next Turn is gone before the hand draw?",
    [fight("NIBBITS_WEAK"), power("FOREGONE_CONCLUSION_POWER", 1), power("DRAW_CARDS_NEXT_TURN_POWER", 2)],
    [end_turn()],
    lambda pre, post, ctx: None if post is None else len(post.hand),
    [Hyp("side-first", "AfterSideTurnStart runs while the screen waits: the bonus is removed before the draw: 1 + 5",
         lambda pre, ctx: 6),
     Hyp("pick-first", "the pick, the hand draw with the bonus, then AfterSideTurnStart: 1 + 5 + 2", lambda pre, ctx: 8)],
    need=lambda pre, ctx: None if pre.power("FOREGONE_CONCLUSION") and pre.power("DRAW_CARDS_NEXT_TURN") else
    "Foregone Conclusion or Draw Cards Next Turn not on the player",
    evidence=lambda tr: trace_digest(tr, keep=("CardDrawn", "PowerReceived", "PowerRemoved"),
                                     hooks=("BeforeHandDraw", "AfterSideTurnStart", "AfterPlayerTurnStart"), limit=20),
    game_class="CombatManager.StartTurn / SetupPlayerTurn; HookPlayerChoiceContext.WaitForPauseOrCompletionWithoutAssigningTask; "
               "DrawCardsNextTurnPower.AfterSideTurnStart",
    note="the game's answer depends on the driver: pick_timing \"ui\" answers on the screen as a player does",
))

PROBES.append(Probe(
    "P20", ["C1-F01"], "With Foregone Conclusion's screen open, does Clarity tick down (and go) before the hand draw?",
    [fight("NIBBITS_WEAK"), power("FOREGONE_CONCLUSION_POWER", 1), power("CLARITY_POWER", 1)],
    [end_turn()],
    lambda pre, post, ctx: None if post is None else len(post.hand),
    [Hyp("side-first", "Clarity's last turn ticks off while the screen waits: no extra card: 1 + 5", lambda pre, ctx: 6),
     Hyp("pick-first", "the hand draw reads Clarity first: 1 + 6", lambda pre, ctx: 7)],
    need=lambda pre, ctx: None if pre.power("FOREGONE_CONCLUSION") and pre.power("CLARITY") else
    "Foregone Conclusion or Clarity not on the player",
    evidence=lambda tr: trace_digest(tr, keep=("CardDrawn", "PowerReceived", "PowerRemoved"),
                                     hooks=("BeforeHandDraw", "AfterSideTurnStart"), limit=20),
    game_class="CombatManager.StartTurn; ClarityPower.ModifyHandDraw / AfterSideTurnStart",
))

# ------------------------------------------------------------------ fix-kit probes (round C5)
# Each answers the first question of a fix kit that had no probe. The ledger holds the kit's own id, which
# tools/fidelity/packet.py probe_requests matches. Decks are set whole (deck_set) so the opening hand is
# known; the Ascender's Bane of the A10 deck goes with it unless a probe lists it.

PROBES.append(Probe(
    "P21", ["C5-F01"], "When Echo Form plays a card twice, does Rage's after-card-played block come once or once per repetition?",
    [fight("NIBBITS_WEAK"), power("ECHO_FORM_POWER", 1), power("RAGE_POWER", 3), card("STRIKE_IRONCLAD")],
    [play("STRIKE_IRONCLAD", 0)],
    lambda pre, post, ctx: None if post is None else post.block - pre.block,
    [Hyp("per-repetition", "BeforeCardPlayed..AfterCardPlayed run for each repetition: Rage pays 3 twice", lambda pre, ctx: 6),
     Hyp("once", "a play count above one re-runs only the effect: Rage pays 3 once", lambda pre, ctx: 3)],
    need=lambda pre, ctx: None if pre.power("ECHO_FORM") and pre.power("RAGE") and "STRIKE_IRONCLAD" in pre.hand else
    "Echo Form, Rage or the Strike is missing",
    evidence=lambda tr: trace_digest(tr, keep=("DamageReceived", "BlockGained", "PowerReceived"),
                                     hooks=("BeforeCardPlayed", "AfterCardPlayed"), limit=20),
    game_class="CardModel.OnPlayWrapper; Hook.ModifyCardPlayCount; EchoFormPower.ModifyCardPlayCount; RagePower.AfterCardPlayed",
))

PROBES.append(Probe(
    "P22", ["C5-F03"], "Does Tezcatara's Ember add its own damage value (3) to a Strike, or its enchant amount (1)?",
    [deck_set(dcard("STRIKE_IRONCLAD", "TEZCATARAS_EMBER", 1, n=5)), fight("NIBBITS_WEAK")],
    [play("STRIKE_IRONCLAD", 0)],
    dealt(0),
    [Hyp("own-damage-var", "EnchantDamageAdditive returns the enchantment's DamageVar (3): 6 + 3 + Strength",
         lambda pre, ctx: 9 + pre.power("STRENGTH")),
     Hyp("enchant-amount", "the additive is the enchant amount (1): 6 + 1 + Strength", lambda pre, ctx: 7 + pre.power("STRENGTH"))],
    need=lambda pre, ctx: None if "STRIKE_IRONCLAD" in pre.hand and pre.enemies else "no Strike in hand or no enemy",
    evidence=lambda tr: trace_digest(tr, keep=("DamageReceived",)),
    game_class="TezcatarasEmber.EnchantDamageAdditive; CardModel.EnchantDamageAdditive; Hook.ModifyDamage",
    note="the enchantment is set on the deck's cards before the fight (CardCmd.Enchant); the emulator side reads the same deck",
))

PROBES.append(Probe(
    "P23", ["C5-F03"], "Does Sharp's damage bonus apply to Hand of Greed (a hit made from the card's DamageVar)?",
    [deck_set(dcard("HAND_OF_GREED", "SHARP", 3, n=5)), fight("NIBBITS_WEAK")],
    [play("HAND_OF_GREED", 0)],
    dealt(0),
    [Hyp("sharp-applies", "EnchantDamageAdditive adds to the attack: 20 + 3 + Strength", lambda pre, ctx: 23 + pre.power("STRENGTH")),
     Hyp("sharp-skipped", "the card's own number is used as is: 20 + Strength", lambda pre, ctx: 20 + pre.power("STRENGTH"))],
    need=lambda pre, ctx: None if "HAND_OF_GREED" in pre.hand and pre.enemies else "no Hand of Greed in hand or no enemy",
    evidence=lambda tr: trace_digest(tr, keep=("DamageReceived",)),
    game_class="HandOfGreed.OnPlay; DamageCmd.Attack; Sharp.EnchantDamageAdditive; Hook.ModifyDamage",
))

PROBES.append(Probe(
    "P24", ["C5-F03"], "Does Nimble's block bonus apply to each block Second Wind gains per exhausted card?",
    [deck_set(dcard("SECOND_WIND", "NIMBLE", 2) + dcard("DEFEND_IRONCLAD", n=4)), fight("NIBBITS_WEAK")],
    [play("SECOND_WIND")],
    lambda pre, post, ctx: None if post is None else post.block - pre.block,
    [Hyp("nimble-each", "every gain is 5 + 2 (+ Dexterity): the bonus applies per card",
         lambda pre, ctx: (len(pre.hand) - 1) * (7 + pre.power("DEXTERITY"))),
     Hyp("nimble-skipped", "every gain is the card's 5 (+ Dexterity)",
         lambda pre, ctx: (len(pre.hand) - 1) * (5 + pre.power("DEXTERITY")))],
    need=lambda pre, ctx: None if "SECOND_WIND" in pre.hand and len(pre.hand) > 1
    and all(c in ("SECOND_WIND", "DEFEND_IRONCLAD") for c in pre.hand) else "Second Wind not in a hand of only Defends",
    evidence=lambda tr: trace_digest(tr, keep=("BlockGained", "CardExhausted")),
    game_class="SecondWind.OnPlay; CreatureCmd.GainBlock; Nimble.EnchantBlockAdditive; Hook.ModifyBlock",
    note="Second Wind exhausts every other non-Attack card in hand; the hand is the deck (an Ascender's Bane left in it would count)",
))

PROBES.append(Probe(
    "P25", ["C5-F07"], "In a Glory elite (Soul Nexus), is the room an elite room: does Booming Conch add its two cards and one energy on turn 1?",
    [relic("BOOMING_CONCH"), fight("SOUL_NEXUS_ELITE"), card("STRIKE_IRONCLAD")],
    [play("STRIKE_IRONCLAD", 0)],
    lambda pre, post, ctx: None if post is None else (post.energy, len(post.hand)),
    [Hyp("elite", "RoomType is Elite: 3 + 1 energy, 5 + 2 cards and the Strike made here; one energy and one card spent",
         lambda pre, ctx: (3, 7)),
     Hyp("not-elite", "the room is not an elite room: no Conch bonus", lambda pre, ctx: (2, 5))],
    need=lambda pre, ctx: None if "STRIKE_IRONCLAD" in pre.hand and pre.enemies else "no Strike in hand or no enemy",
    evidence=lambda tr: trace_digest(tr, keep=("CardDrawn", "EnergySpent", "PowerReceived"),
                                     hooks=("ModifyHandDraw", "AfterSideTurnStart"), limit=20),
    game_class="BoomingConch.ModifyHandDraw; BoomingConch.AfterSideTurnStart; AbstractRoom.RoomType; EncounterModel.RoomType",
    note="the turn-1 draw happens before the step, so the bonus is read from the hand and energy left after the first play",
))

PROBES.append(Probe(
    "P26", ["C5-F09"], "Does Bellows upgrade the cards that cannot be upgraded (Ascender's Bane, a curse)?",
    [deck_set(dcard("ASCENDERS_BANE") + dcard("INJURY") + dcard("DEFEND_IRONCLAD", n=3)), relic("BELLOWS"), fight("NIBBITS_WEAK")],
    [play("DEFEND_IRONCLAD")],
    lambda pre, post, ctx: None if post is None else sorted(
        (c, u) for c, u in zip(post.hand, post.hand_up, strict=False) if c in ("ASCENDERS_BANE", "INJURY")),
    [Hyp("skipped", "CardCmd.Upgrade leaves cards that cannot be upgraded as they are",
         lambda pre, ctx: [("ASCENDERS_BANE", False), ("INJURY", False)]),
     Hyp("upgraded-flag", "every card in the opening hand shows as upgraded",
         lambda pre, ctx: [("ASCENDERS_BANE", True), ("INJURY", True)])],
    need=lambda pre, ctx: None if "ASCENDERS_BANE" in pre.hand and "INJURY" in pre.hand and "DEFEND_IRONCLAD" in pre.hand else
    "Bane, Injury or a Defend not in the opening hand",
    evidence=lambda tr: trace_digest(tr, keep=("CardGenerated",), hooks=("AfterPlayerTurnStart",)),
    game_class="Bellows.AfterPlayerTurnStart; CardCmd.Upgrade; CardModel.IsUpgradable",
))


def _strike_places(pre, post, ctx):
    """Where the Strike the setup put on top of the draw pile is after the step: discard, hand, exhaust, draw."""
    if post is None:
        return None
    return [post.discard.count("STRIKE_IRONCLAD"), post.hand.count("STRIKE_IRONCLAD"), post.exhaust.count("STRIKE_IRONCLAD"),
            post.draw.count("STRIKE_IRONCLAD")]


PROBES.append(Probe(
    "P27", ["C5-H03"], "With Hellraiser, where does a Strike drawn at the next turn's start end up once the game has auto-played it?",
    [deck_set(dcard("DEFEND_IRONCLAD", n=10)), fight("NIBBITS_WEAK"), power("HELLRAISER_POWER", 1), card("STRIKE_IRONCLAD", "Draw", "top")],
    [end_turn()],
    _strike_places,
    [Hyp("played-to-discard", "the drawn Strike is auto-played, then discarded", lambda pre, ctx: [1, 0, 0, 0]),
     Hyp("played-exhausted", "the drawn Strike is auto-played, then exhausted", lambda pre, ctx: [0, 0, 1, 0]),
     Hyp("kept-in-hand", "the drawn Strike stays in the hand", lambda pre, ctx: [0, 1, 0, 0])],
    need=lambda pre, ctx: None if pre.draw[:1] == ["STRIKE_IRONCLAD"] and pre.power("HELLRAISER") else "no Strike on top or no Hellraiser",
    evidence=lambda tr: trace_digest(tr, keep=("CardDrawn", "DamageReceived", "CardExhausted"), hooks=("AfterCardDrawnEarly",), limit=20),
    game_class="HellraiserPower.AfterCardDrawnEarly; CardCmd.AutoPlay",
    note="the deck is ten Defends, so the only Strike is the one the setup puts on top of the draw pile",
))

PROBES.append(Probe(
    "P28", ["C5-H03"], "With Mayhem, where does the card auto-played from the draw pile at turn start end up?",
    [deck_set(dcard("DEFEND_IRONCLAD", n=10)), fight("NIBBITS_WEAK"), power("MAYHEM_POWER", 1), card("STRIKE_IRONCLAD", "Draw", "top")],
    [end_turn()],
    _strike_places,
    [Hyp("played-to-discard", "the top card is auto-played before the draw, then discarded", lambda pre, ctx: [1, 0, 0, 0]),
     Hyp("played-exhausted", "it is auto-played, then exhausted", lambda pre, ctx: [0, 0, 1, 0]),
     Hyp("drawn-instead", "it is not auto-played and is drawn", lambda pre, ctx: [0, 1, 0, 0])],
    need=lambda pre, ctx: None if pre.draw[:1] == ["STRIKE_IRONCLAD"] and pre.power("MAYHEM") else "no Strike on top or no Mayhem",
    evidence=lambda tr: trace_digest(tr, keep=("CardDrawn", "DamageReceived", "CardExhausted"),
                                     hooks=("AfterAutoPrePlayPhaseEntered",), limit=20),
    game_class="MayhemPower.AfterAutoPrePlayPhaseEntered; CardPileCmd.AutoPlayFromDrawPile",
))

PROBES.append(Probe(
    "P29", ["C5-F06"], "When a Doomed Fabricator dies at the end of the enemy turn with bots it fabricated that turn, is the fight won at once (the bots are secondary enemies) or does it go on?",
    [fight("FABRICATOR_NORMAL"), power("DOOM_POWER", 999, "enemy:0")],
    [end_turn()],
    lambda pre, post, ctx: None if post is None else post.outcome,
    [Hyp("won", "the last primary enemy's death ends the fight; the bots (MinionPower) go with it", lambda pre, ctx: "won"),
     Hyp("going", "the bots are living enemies: the fight goes on with them", lambda pre, ctx: "going")],
    need=lambda pre, ctx: None if len(pre.enemies) == 1 else "not the Fabricator alone",
    evidence=lambda tr: trace_digest(tr, keep=("DamageReceived", "CreatureDied", "PowerReceived"),
                                     hooks=("BeforeSideTurnEnd", "AfterDiedToDoom"), limit=24),
    game_class="DoomPower.BeforeSideTurnEnd; MinionPower; Fabricator.SpawnBot; CombatManager's end-of-combat check",
    ends=True,
    note="the item's post-fight HP gap is not tied to a rule yet; this asks how a fight with fabricated bots ends. The Battleworn Dummy half is "
         "not a combat probe: the setup fight cannot set the dummy's setting and its Battle Friend only waits (its time limit escapes it)",
))

BY_ID = {p.id: p for p in PROBES}
