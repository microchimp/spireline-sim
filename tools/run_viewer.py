#!/usr/bin/env python3
"""Render one run record (the driver's `run.json`) as a single self-contained HTML page, floor by floor.

    python tools/run_viewer.py --record run.json --out page.html --title "..." [--generation 5] [--note "..."]
        [--annotations annotations.json]

Input: the run record the driver writes (see referee/lockstep_trace.py, phase replay): `seed`, `character`, `ascension`, `acts`,
`end`, `floor`, and `decisions[]`, each with `i`, `screen` (start, neow, map, combat, rewards, rest, shop, event, treasure,
crystal_sphere, act_change, ancient), `floor`, `offer`, `choice`, `room`, `snap` (the game's run state after the decision; only
`players[0]` is read: hp, max hp, gold, potions, relics, deck) and, for `combat`, `encounter`, `result` and `fight.steps[]`
(`action` = play / potion / end_turn; `state` = round, player, hand, enemies). Every key is optional: a section whose data is
missing is left out, so older or partial records still render.

Output: standard library only, inline CSS and SVG, no external assets, no absolute paths. Names are the record's ids made readable
(STRIKE_IRONCLAD -> Strike); nothing else of the game's text is added.

Annotations (the solver's hook): a JSON object keyed by decision index (the record's `i`, as a string or a number) whose value is a
string, a list of strings, or {"title": ..., "text": ...}; each is shown under its decision. Empty today.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

CSS = """
:root{--bg:#f6f4ef;--fg:#222;--mut:#6b6b6b;--card:#fff;--line:#ddd8cc;--hp:#c0392b;--gold:#b8860b;--ok:#2e7d32;--bad:#b71c1c;--note:#e8f0fe}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:16px}h1{margin:.2em 0}h2{margin:1.4em 0 .4em;border-bottom:2px solid var(--line);padding-bottom:.2em}
h3{margin:.2em 0}.mut{color:var(--mut)}.facts{display:flex;flex-wrap:wrap;gap:6px 18px;margin:.4em 0}.facts b{font-weight:600}
.floor{background:var(--card);border:1px solid var(--line);border-radius:6px;margin:10px 0;padding:8px 12px}
.floor>h3{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}.act{margin:22px 0 4px;font-size:1.25em;font-weight:700;color:#444}
.dec{margin:6px 0;padding:4px 0 4px 8px;border-left:3px solid var(--line)}.dec .no{float:right;font-size:11px;color:var(--mut)}
.chip{display:inline-block;padding:0 6px;margin:1px 2px;border:1px solid var(--line);border-radius:10px;background:#faf8f2;white-space:nowrap}
.chip.pick{background:#e3f2e1;border-color:#9ccc95;font-weight:600}.chip.skip{color:var(--mut);text-decoration:line-through}
.chip.pot{background:#e8eefc;border-color:#9fb4e8}.tag{font-size:11px;padding:0 6px;border-radius:3px;color:#fff;background:#777}
.tag.Start,.tag.Ancient{background:#444}.tag.Monster{background:#6d7b8d}.tag.Elite{background:#a15c00}.tag.Boss{background:#8e1b1b}.tag.Unknown{background:#5b6abf}
.tag.RestSite{background:#2e7d32}.tag.Shop{background:#8a6d00}.tag.Treasure{background:#a0522d}.tag.Event{background:#5b6abf}
table{border-collapse:collapse;width:100%;margin:4px 0}th,td{border:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top}
th{background:#efebe0;font-weight:600}td.n{text-align:right;white-space:nowrap}.loss{color:var(--bad)}.gain{color:var(--ok)}
.bar{display:inline-block;height:8px;background:#ddd;width:90px;vertical-align:middle;border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--hp)}details>summary{cursor:pointer;color:#345}
.note{margin:4px 0;padding:6px 10px;background:var(--note);border-left:4px solid #4a78d0;border-radius:3px;white-space:pre-wrap}
.note b{display:block}svg text{font:10px system-ui,sans-serif;fill:#555}.end{border-color:#b71c1c}nav{margin:6px 0;line-height:1.9}
nav a{text-decoration:none;color:inherit}.pw{color:var(--mut);font-size:12px}
"""

_PREFIX = re.compile(r"^(CARD|RELIC|POTION|ENCOUNTER|EVENT|POWER|CHARACTER)\.")
_CHAR_SUFFIX = re.compile(r"_(IRONCLAD|SILENT|DEFECT|REGENT|NECROBINDER)$")
ROOM_LABEL = {"Monster": "Fight", "Elite": "Elite fight", "Boss": "Boss fight", "RestSite": "Rest site", "Shop": "Shop",
              "Treasure": "Treasure", "Event": "Event", "Unknown": "Unknown room", "Ancient": "Ancient"}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def pretty(x, card: bool = False) -> str:
    """STRIKE_IRONCLAD -> 'Strike'; 'RELIC.BURNING_BLOOD' -> 'Burning Blood'; 'VULNERABLE_POWER' -> 'Vulnerable'."""
    if x is None:
        return "?"
    s = _PREFIX.sub("", str(x))
    s = re.sub(r"_POWER$", "", s)
    if card:
        s = _CHAR_SUFFIX.sub("", s)
    return " ".join(w.capitalize() for w in s.replace(".", " ").split("_") if w)


def enc_name(x) -> str:
    """ENCOUNTER.TOADPOLES_WEAK -> 'Toadpoles' (the pool suffix is the room kind, shown as a tag)."""
    return pretty(re.sub(r"_(WEAK|NORMAL|ELITE|BOSS)$", "", _PREFIX.sub("", str(x))))


def card_name(c) -> str:
    """A card as the record shows it: an id, or [id, upgrade level / upgraded flag]."""
    if isinstance(c, dict):
        c = c.get("id")
    if isinstance(c, (list, tuple)):
        base = pretty(c[0], True)
        up = c[1] if len(c) > 1 else 0
        if isinstance(up, bool):
            return base + ("+" if up else "")
        if isinstance(up, int) and up > 0:
            return base + ("+" if up == 1 else f"+{up}")
        return base
    return pretty(c, True)


def chip(text: str, cls: str = "") -> str:
    return f'<span class="chip {cls}">{esc(text)}</span>'


def option_label(opt) -> str:
    """Event / Neow / Ancient option: 'TABLET_OF_TRUTH.pages.INITIAL.options.SMASH' -> 'Smash'; relic ids -> names."""
    s = str(opt[0] if isinstance(opt, (list, tuple)) and opt else opt)
    return pretty(s.split(".options.")[-1] if ".options." in s else s)


def powers_text(powers) -> str:
    out = []
    for p in powers or []:
        if isinstance(p, (list, tuple)) and p:
            out.append(pretty(p[0]) + (f" {p[1]}" if len(p) > 1 and p[1] is not None else ""))
        else:
            out.append(pretty(p))
    return ", ".join(out)


def intent_text(intents) -> str:
    out = []
    for it in intents or []:
        if isinstance(it, (list, tuple)) and it:
            t = str(it[0])
            if len(it) > 1 and it[1]:
                t += f" {it[1]}" + (f"x{it[2]}" if len(it) > 2 and isinstance(it[2], int) and it[2] > 1 else "")
            out.append(t)
        else:
            out.append(str(it))
    return ", ".join(out)


def player_of(d: dict) -> dict:
    ps = ((d.get("snap") or {}).get("players")) or []
    return ps[0] if ps else {}


def potion_ids(d: dict) -> list:
    return [p.get("id") if isinstance(p, dict) else p for p in (player_of(d).get("potions") or [])]


def hp_bar(hp, mx) -> str:
    try:
        w = max(0, min(100, round(100 * hp / mx)))
    except (TypeError, ZeroDivisionError):
        return ""
    return f'<span class="bar"><i style="width:{w}%"></i></span>'


def delta(a, b, unit: str = "") -> str:
    if a is None or b is None or a == b:
        return ""
    d = b - a
    return f' <span class="{"gain" if d > 0 else "loss"}">({d:+d}{unit})</span>'


# ---------------------------------------------------------------- fights

def enemy_names(enemies: list) -> list:
    names = [pretty(e.get("name")) for e in enemies]
    return [f"{n} {i + 1}" if names.count(n) > 1 else n for i, n in enumerate(names)]


def fight_turns(steps: list) -> list:
    """Group `fight.steps` into turns. A step's `state` is the state after its `action`, so an end_turn step carries the next
    turn's opening state (after the enemies acted)."""
    turns, cur, prev = [], None, None
    for s in steps or []:
        a, st = s.get("action"), s.get("state") or {}
        if a is None or cur is None:
            if st.get("in_combat", True) or cur is None:
                cur = {"round": st.get("round"), "state": st, "plays": [], "hp_end": None, "enemies_end": None}
                turns.append(cur)
            prev = st
            continue
        kind = a.get("kind")
        before = prev or {}
        names = enemy_names(before.get("enemies") or [])
        tslot = a.get("target_slot", a.get("target"))
        target = names[tslot] if isinstance(tslot, int) and 0 <= tslot < len(names) else None
        if kind == "play":
            if a.get("played") is not False:
                hi = a.get("hand_index")
                hand = before.get("hand") or []
                up = hand[hi][1] if isinstance(hi, int) and 0 <= hi < len(hand) and isinstance(hand[hi], (list, tuple)) and len(hand[hi]) > 1 else 0
                cur["plays"].append({"what": card_name([a.get("card"), up]), "target": target, "potion": False})
        elif kind == "potion":
            cur["plays"].append({"what": pretty(a.get("potion")), "target": target, "potion": True})
        elif kind == "end_turn":
            cur["block_end"] = (before.get("player") or {}).get("block")
            cur["hp_end"] = (st.get("player") or {}).get("hp")
            cur["enemies_end"] = st.get("enemies")
            if st.get("in_combat", True):
                cur = {"round": st.get("round"), "state": st, "plays": [], "hp_end": None, "enemies_end": None}
                turns.append(cur)
        if not st.get("in_combat", True):
            cur["hp_end"] = (st.get("player") or {}).get("hp")
            cur["enemies_end"] = st.get("enemies")
            cur["finished"] = True
        prev = st
    if turns and turns[-1]["hp_end"] is None:
        turns[-1]["hp_end"] = ((prev or {}).get("player") or {}).get("hp")
        turns[-1]["enemies_end"] = (prev or {}).get("enemies")
    return turns


def render_turns(turns: list) -> str:
    rows = []
    for t in turns:
        st = t["state"]
        pl = st.get("player") or {}
        enemies = st.get("enemies") or []
        names = enemy_names(enemies)
        en = "<br>".join(
            esc(f"{n} {e.get('hp')}/{e.get('max_hp')}" + (f", block {e['block']}" if e.get("block") else "")
                + (f" - {intent_text(e.get('intents'))}" if e.get("intents") else ""))
            + (f' <span class="pw">{esc(powers_text(e.get("powers")))}</span>' if e.get("powers") else "")
            for n, e in zip(names, enemies))
        hand = " ".join(chip(card_name(c)) for c in st.get("hand") or [])
        plays = " ".join(chip(p["what"] + (f" -> {p['target']}" if p["target"] else ""), "pot" if p["potion"] else "pick") for p in t["plays"]) \
            or '<span class="mut">no card played</span>'
        hp0, hp1 = pl.get("hp"), t.get("hp_end")
        left = [e for e in (t.get("enemies_end") or []) if e.get("hp", 0) > 0]
        end_note = ""
        if t.get("finished") and not left:
            end_note = " - fight over"
        pw = f'<div class="pw">{esc(powers_text(pl.get("powers")))}</div>' if pl.get("powers") else ""
        rows.append(
            f'<tr><td class="n">{esc(t["round"])}</td>'
            f'<td class="n">{esc(hp0)}/{esc(pl.get("max_hp"))} {hp_bar(hp0, pl.get("max_hp"))}<br>to {esc(hp1)}{delta(hp0, hp1)}{esc(end_note)}'
            f'<br>energy {esc(pl.get("energy"))}</td>'
            f'<td>{hand}{pw}</td><td>{plays}</td><td>{en}</td></tr>')
    return ("<table><tr><th>Turn</th><th>Player HP</th><th>Hand</th><th>Played</th><th>Enemies at the start of the turn</th></tr>"
            + "".join(rows) + "</table>")


def render_choices(fight: dict) -> str:
    out = []
    for c in fight.get("choices") or []:
        opts, picked = c.get("options") or [], c.get("picked") or []
        if c.get("kind") == "bundle":
            shown = [", ".join(card_name(x) for x in o) if isinstance(o, list) else card_name(o) for o in opts]
        else:
            shown = [card_name(o) for o in opts]
        took = ", ".join(shown[i] for i in picked if isinstance(i, int) and 0 <= i < len(shown)) or "nothing"
        out.append(f'<div class="pw">Mid-fight choice at step {esc(c.get("step"))}: picked {esc(took)} from '
                   + esc("; ".join(shown)) + "</div>")
    return "".join(out)


# ---------------------------------------------------------------- decisions

def render_decision(d: dict, prev: dict | None, ctx: dict) -> str:
    scr, ch, off = d.get("screen"), d.get("choice"), d.get("offer")
    p, pp = player_of(d), player_of(prev or {})
    hp0, hp1 = pp.get("current_hp"), p.get("current_hp")
    gold0, gold1 = pp.get("gold"), p.get("gold")
    body = ""
    if scr == "start":
        body = f'<b>Start</b> - {esc(pretty(p.get("character_id")))} with {len(p.get("deck") or [])} cards and {len(p.get("relics") or [])} relic(s)'
    elif scr in ("neow", "ancient"):
        opts = [option_label(o) for o in off or []]
        idx = d.get("choice_index")
        label = "Neow's offer" if scr == "neow" else f"{pretty(d.get('event'))} offers"
        body = f"<b>{esc(label)}</b> " + " ".join(chip(o, "pick" if i == idx or (idx is None and pretty(ch) == o) else "") for i, o in enumerate(opts)) \
            + f' - took <b>{esc(pretty(ch))}</b>{delta(hp0, hp1, " HP")}'
    elif scr == "map":
        offer = off or []
        chosen = ch if isinstance(ch, list) else None
        alts = " ".join(
            chip(f"{ROOM_LABEL.get(o[2], o[2])} (col {o[0]})", "pick" if chosen and o[0] == chosen[0] and o[1] == chosen[1] else "")
            for o in offer if isinstance(o, list) and len(o) >= 3)
        node = next((o[2] for o in offer if isinstance(o, list) and chosen and o[:2] == chosen[:2]), None)
        room = d.get("room") or []
        rk, rid = (room[0] if room else None), (room[1] if len(room) > 1 else None)
        what = enc_name(rid) if rk in ("Monster", "Elite", "Boss") and rid else (pretty(rid) if rid else "")
        became = (", which was " + ("an " if ROOM_LABEL.get(rk, rk)[:1] in "AEIOU" else "a ") + ROOM_LABEL.get(rk, rk).lower()) if node == "Unknown" and rk else ""
        body = (f"<b>Map</b> - paths offered {alts}" + (f" - chose the {ROOM_LABEL.get(node, node).lower()} in column {esc(chosen[0])}" if node else "")
                + esc(became) + (f' - <b>{esc(what)}</b>' if what else "") + delta(hp0, hp1, " HP"))
    elif scr == "combat":
        fight = d.get("fight") or {}
        steps = fight.get("steps") or []
        turns = fight_turns(steps)
        first = (steps[0].get("state") if steps else {}) or {}
        pl0 = first.get("player") or {}
        enemies = ", ".join(f"{n} ({e.get('hp')} HP)" for n, e in zip(enemy_names(first.get("enemies") or []), first.get("enemies") or []))
        potions = [pl["what"] for t in turns for pl in t["plays"] if pl["potion"]]
        res = d.get("result")
        hp_before = pl0.get("hp", hp0)
        body = (f'<b>{esc(enc_name(d.get("encounter")))}</b> - {esc(enemies)}<br>'
                f'HP {esc(hp_before)} to {esc(hp1)}{delta(hp_before, hp1)} of {esc(p.get("max_hp"))}; {len(turns)} turn(s); '
                f'result <b class="{"gain" if res == "win" else "loss"}">{esc(res)}</b>'
                + (f'; potions used: {esc(", ".join(potions))}' if potions else ""))
        if turns:
            open_ = ctx["is_boss"] or ctx["is_elite"] or res != "win"
            body += f'<details{" open" if open_ else ""}><summary>Turn by turn ({len(turns)} turns, {sum(len(t["plays"]) for t in turns)} plays)</summary>' \
                + render_turns(turns) + render_choices(fight) + "</details>"
    elif scr == "rewards":
        body = render_reward(d)
    elif scr == "rest":
        body = render_rest(d, hp0, hp1)
    elif scr == "shop":
        body = render_shop(d, gold0, gold1)
    elif scr == "event":
        opts = [option_label(o) for o in off or []]
        idx = d.get("choice_index")
        body = (f'<b>Event: {esc(pretty(d.get("event")))}</b> ' + " ".join(chip(o, "pick" if i == idx else "") for i, o in enumerate(opts))
                + f' - took <b>{esc(option_label(ch))}</b>' + sel_text(d) + f'{delta(hp0, hp1, " HP")}{delta(gold0, gold1, " gold")}')
    elif scr == "treasure":
        got = ", ".join(pretty(r) for r in off or [])
        body = f'<b>Treasure chest</b> - relic {esc(got)}' + (" taken" if ch == "take" else f" ({esc(ch)})")
    elif scr == "crystal_sphere":
        c = ch if isinstance(ch, dict) else {}
        body = f'<b>Crystal sphere</b> - used the {esc(c.get("tool", "?"))} tool at ({esc(c.get("x"))}, {esc(c.get("y"))})'
    elif scr == "act_change":
        body = f'<b>Next act</b> - moved on after the boss'
    elif scr != "start":
        body = f"<b>{esc(pretty(scr))}</b> - {esc(json.dumps(ch)[:200])}"
    note = ctx["annotations"].get(str(d.get("i")))
    if render_note(note):
        ctx["placed"] = ctx.get("placed", 0) + 1
    return (f'<div class="dec" id="d{esc(d.get("i"))}"><span class="no">decision {esc(d.get("i"))}</span>{body}'
            f'{render_note(note)}</div>')


def sel_names(d: dict) -> str:
    out = []
    for s in d.get("selections") or []:
        opts, picked = s.get("options") or [], s.get("picked") or []
        names = [card_name(opts[i]) for i in picked if isinstance(i, int) and 0 <= i < len(opts)]
        if names:
            out.append(", ".join(names))
    return "; ".join(out)


def sel_text(d: dict) -> str:
    n = sel_names(d)
    return f' - card picked: <b>{esc(n)}</b>' if n else ""


def render_reward(d: dict) -> str:
    ch, off = d.get("choice"), d.get("offer") or []

    def item(r) -> str:
        if not isinstance(r, dict):
            return pretty(r)
        t = r.get("type")
        if t == "gold":
            return f'{r.get("amount")} gold'
        if t == "potion":
            return f'potion {pretty(r.get("id"))}'
        if t == "relic":
            return f'relic {pretty(r.get("id"))}'
        if t == "card":
            return "card reward (" + " / ".join(card_name(c) for c in r.get("cards") or []) + ")"
        if t == "special_card":
            return f'card {card_name(r.get("card"))}'
        return pretty(t) + (f' {pretty(r.get("id"))}' if r.get("id") else "")

    offered = " ".join(chip(item(r), "pick" if r == ch else "") for r in off)
    if ch == "proceed":
        left = ", ".join(item(r) for r in off)
        return f'<b>Rewards</b> - moved on' + (f", skipping: {esc(left)}" if left else "")
    if isinstance(ch, dict) and ch.get("type") == "card":
        if "card" in d:
            others = ", ".join(card_name(c) for c in ch.get("cards") or [] if c != d["card"])
            return (f'<b>Rewards</b> - card reward: took <b>{esc(card_name(d["card"]))}</b>, passed on {esc(others)}'
                    if others else f'<b>Rewards</b> - took <b>{esc(card_name(d["card"]))}</b>')
        return f'<b>Rewards</b> - skipped the card reward ({esc(item(ch))})'
    return f'<b>Rewards</b> - offered {offered} - took <b>{esc(item(ch))}</b>' + ("" if d.get("taken", True) else " (not taken)")


def render_rest(d: dict, hp0, hp1) -> str:
    ch, off = d.get("choice"), d.get("offer") or []
    opts = " ".join(chip(pretty(o[0] if isinstance(o, (list, tuple)) else o), "pick" if (o[0] if isinstance(o, (list, tuple)) else o) == ch else "")
                    for o in off)
    if ch == "proceed":
        return "<b>Rest site</b> - left"
    what = {"HEAL": "rested and healed", "SMITH": "upgraded a card"}.get(ch, f"chose {pretty(ch)}")
    extra = ""
    if sel_names(d):
        what = {"SMITH": "upgraded", "TOKE": "removed"}.get(ch, what + ", card picked")
        extra = f" <b>{esc(sel_names(d))}</b>"
    heal = f" HP {esc(hp0)} to {esc(hp1)}{delta(hp0, hp1)}" if ch == "HEAL" else ""
    return f'<b>Rest site</b> - options {opts} - {esc(what)}{extra}{heal}'


def render_shop(d: dict, gold0, gold1) -> str:
    ch, off = d.get("choice"), d.get("offer") or {}
    if ch == "leave":
        return f'<b>Shop</b> - left with {esc(gold1)} gold'
    if isinstance(ch, dict) and "buy" in ch:
        kind, slot = ch["buy"], ch.get("slot")
        items = off.get(kind + "s") or []
        it = items[slot] if isinstance(slot, int) and 0 <= slot < len(items) else None
        if it:
            nm = card_name(it) if kind == "card" else pretty(it[0])
            price = it[2] if kind == "card" and len(it) > 2 else (it[1] if len(it) > 1 else "?")
            return f'<b>Shop</b> - bought {esc(kind)} <b>{esc(nm)}</b> for {esc(price)} gold ({esc(gold1)} left)'
        return f'<b>Shop</b> - bought a {esc(kind)} (slot {esc(slot)})'
    if isinstance(ch, dict) and ("remove" in ch or "removal" in ch):
        return f'<b>Shop</b> - paid {esc(off.get("removal", "?"))} gold to remove a card{sel_text(d)}'
    return f'<b>Shop</b> - {esc(json.dumps(ch)[:120])}'


def render_note(note) -> str:
    if note is None or note == "" or note == []:
        return ""
    if isinstance(note, dict):
        title, text = note.get("title"), note.get("text", "")
    else:
        title, text = None, note
    if isinstance(text, list):
        text = "\n".join(str(x) for x in text)
    return f'<div class="note">' + (f"<b>{esc(title)}</b>" if title else "") + esc(text) + "</div>"


# ---------------------------------------------------------------- page

def sections(decisions: list) -> list:
    """Split the decisions into floors. A map decision opens the next floor (named by the room it led to); an act_change goes first
    into the section after it; neow and start make the start section."""
    secs, cur, pending = [], None, []
    for d in decisions:
        scr = d.get("screen")
        if scr == "act_change":
            pending.append(d)
            continue
        if scr == "map":
            room = d.get("room") or []
            offer, ch = d.get("offer") or [], d.get("choice")
            node = next((o[2] for o in offer if isinstance(o, list) and isinstance(ch, list) and o[:2] == ch[:2]), None)
            known = ("Monster", "Elite", "Boss", "RestSite", "Shop", "Treasure", "Event", "Ancient")
            kind = node if node in known else (room[0] if room and room[0] in known else (node or "Unknown"))
            cur = {"kind": "map", "floor": (d.get("floor") or 0) + 1, "room": kind, "node": kind, "decs": pending + [d]}
            pending = []
            secs.append(cur)
        elif cur is None:
            cur = {"kind": "start", "floor": d.get("floor"), "room": "Start", "node": "Start", "decs": pending + [d]}
            pending = []
            secs.append(cur)
        else:
            cur["decs"].append(d)
    if pending and cur is not None:
        cur["decs"].extend(pending)
    return secs


def strip_svg(rows: list) -> str:
    """HP and max HP per floor above, gold below, potion count as numbers under the axis. One point per floor section."""
    n = len(rows)
    if n < 2:
        return ""
    W, H, L, R = 1040, 250, 40, 10
    x = lambda i: L + (W - L - R) * i / (n - 1)  # noqa: E731
    top = max([r["max_hp"] or 0 for r in rows] + [1])
    gmax = max([r["gold"] or 0 for r in rows] + [1])
    yh = lambda v: 10 + 110 * (1 - (v or 0) / top)  # noqa: E731
    yg = lambda v: 215 - 75 * (v or 0) / gmax  # noqa: E731
    pts = lambda key, fy: " ".join(f"{x(i):.1f},{fy(r[key]):.1f}" for i, r in enumerate(rows))  # noqa: E731
    mx = " ".join(f"{x(i):.1f},{yh(r['max_hp']):.1f}" for i, r in enumerate(rows))
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="HP, max HP, gold and potions per floor">',
         f'<line x1="{L}" y1="120" x2="{W - R}" y2="120" stroke="#bbb"/><line x1="{L}" y1="215" x2="{W - R}" y2="215" stroke="#bbb"/>',
         f'<text x="2" y="14">HP {top}</text><text x="2" y="122">0</text><text x="2" y="146">gold {gmax}</text><text x="2" y="228">0</text>',
         f'<polyline points="{mx}" fill="none" stroke="#999" stroke-dasharray="4 3"/>',
         f'<polyline points="{pts("hp", yh)}" fill="none" stroke="#c0392b" stroke-width="2"/>',
         f'<polyline points="{pts("gold", yg)}" fill="none" stroke="#b8860b" stroke-width="2"/>']
    for i, r in enumerate(rows):
        s.append(f'<circle cx="{x(i):.1f}" cy="{yh(r["hp"]):.1f}" r="3" fill="#c0392b"><title>floor {esc(r["floor"])}: HP {esc(r["hp"])}/'
                 f'{esc(r["max_hp"])}, gold {esc(r["gold"])}, potions {esc(r["potions"])}</title></circle>')
        if i % max(1, n // 24) == 0:
            s.append(f'<text x="{x(i) - 6:.1f}" y="236">{esc(r["floor"])}</text>')
        if r["potions"]:
            s.append(f'<text x="{x(i) - 3:.1f}" y="248" style="fill:#345">{esc(r["potions"])}</text>')
    s.append(f'<text x="{L}" y="248">potions held</text><text x="{W - 80}" y="14">red = HP, grey dashed = max HP, amber = gold</text></svg>')
    return "".join(s)


def render(record: dict, title: str = "A run", generation=None, note: str = "", annotations: dict | None = None) -> str:
    decs = record.get("decisions") or []
    ann = {str(k): v for k, v in (annotations or {}).items()}
    acts = record.get("acts") or []
    last = decs[-1] if decs else {}
    secs = sections(decs)
    # run strip rows: the state at the end of each floor
    rows = []
    for s in secs:
        p = player_of(s["decs"][-1])
        rows.append({"floor": s["floor"], "hp": p.get("current_hp"), "max_hp": p.get("max_hp"), "gold": p.get("gold"),
                     "potions": len(p.get("potions") or [])})
    fights = [d for d in decs if d.get("screen") == "combat"]
    won = sum(1 for d in fights if d.get("result") == "win")
    end = record.get("end")
    fin = player_of(last)
    how = f'{end}' if end else "unknown"
    if last.get("screen") == "combat" and last.get("result") not in (None, "win"):
        steps = (last.get("fight") or {}).get("steps") or []
        rd = (steps[-1].get("state") or {}).get("round") if steps else None
        how = f'{end or last.get("result")} in the fight against {enc_name(last.get("encounter"))}' + (f" on turn {rd}" if rd else "")
    out = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
           f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>", f"<h1>{esc(title)}</h1>"]
    facts = [("Generation", generation), ("Seed", record.get("seed")), ("Character", pretty(record.get("character")) if record.get("character") else None),
             ("Ascension", record.get("ascension")), ("Acts", ", ".join(pretty(a) for a in acts) if acts else None),
             ("Floor reached", record.get("floor")), ("How it ended", how), ("Fights won", f"{won} of {len(fights)}"),
             ("Decisions recorded", len(decs)), ("Final deck", f"{len(fin.get('deck') or [])} cards"),
             ("Final HP", f"{fin.get('current_hp')}/{fin.get('max_hp')}" if fin else None), ("Gold at the end", fin.get("gold"))]
    out.append('<div class="facts">' + "".join(f"<span>{esc(k)}: <b>{esc(v)}</b></span>" for k, v in facts if v not in (None, "")) + "</div>")
    if note:
        out.append(f'<p class="mut">{esc(note)}</p>')
    out.append("<h2>The run at a glance</h2>")
    out.append(strip_svg(rows))
    out.append("<details><summary>Numbers per floor</summary><table><tr><th>Floor</th><th>Room</th><th>HP</th><th>Max HP</th><th>Gold</th>"
               "<th>Potions</th></tr>" + "".join(
                   f'<tr><td class="n"><a href="#f{i}">{esc(r["floor"])}</a></td><td>{esc(ROOM_LABEL.get(s["node"], s["node"]))}</td>'
                   f'<td class="n">{esc(r["hp"])}</td><td class="n">{esc(r["max_hp"])}</td><td class="n">{esc(r["gold"])}</td>'
                   f'<td class="n">{esc(r["potions"])}</td></tr>' for i, (r, s) in enumerate(zip(rows, secs))) + "</table></details>")
    out.append("<h2>Floor by floor</h2>")
    prev, act = None, None
    ctx = {"annotations": ann, "placed": 0}
    for i, s in enumerate(secs):
        p0 = player_of(s["decs"][0])
        a = (s["decs"][0].get("snap") or {}).get("current_act_index")
        if a is not None and a != act:
            act = a
            nm = pretty(acts[a]) if isinstance(a, int) and a < len(acts) else ""
            out.append(f'<div class="act">Act {a + 1}{" - " + esc(nm) if nm else ""}</div>')
        endp = player_of(s["decs"][-1])
        node = s["node"]
        is_boss, is_elite = s["room"] == "Boss" or node == "Boss", s["room"] == "Elite" or node == "Elite"
        head = (f'<h3><span>Floor {esc(s["floor"])}</span><span class="tag {esc(node)}">{esc(ROOM_LABEL.get(node, node))}</span>'
                f'<span class="mut">HP {esc(endp.get("current_hp"))}/{esc(endp.get("max_hp"))} - gold {esc(endp.get("gold"))} - '
                f'{len(endp.get("potions") or [])} potion(s)</span></h3>')
        is_end = i == len(secs) - 1 and last.get("screen") == "combat" and last.get("result") not in (None, "win")
        out.append(f'<section class="floor{" end" if is_end else ""}" id="f{i}">{head}')
        for d in s["decs"]:
            ctx.update(is_boss=is_boss, is_elite=is_elite)
            out.append(render_decision(d, prev, ctx))
            prev = d
        out.append("</section>")
    out.append("<h2>The end</h2>")
    deck = ", ".join(card_name([c.get("id"), c.get("current_upgrade_level", 0)]) for c in fin.get("deck") or [] if isinstance(c, dict))
    relics = ", ".join(pretty(r.get("id")) for r in fin.get("relics") or [] if isinstance(r, dict))
    out.append(f"<p>The run ended: <b>{esc(how)}</b> (floor {esc(record.get('floor'))}).</p>")
    if deck:
        out.append(f"<p><b>Final deck ({len(fin.get('deck') or [])}):</b> {esc(deck)}</p>")
    if relics:
        out.append(f"<p><b>Relics:</b> {esc(relics)}</p>")
    out.append(f'<p class="mut">Rendered by tools/run_viewer.py from the run record ({len(decs)} decisions; {ctx["placed"]} annotation(s)).</p>')
    out.append("</main></body></html>")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", type=Path, required=True, help="the driver's run.json")
    ap.add_argument("--out", type=Path, required=True, help="the HTML page to write")
    ap.add_argument("--title", default="A run")
    ap.add_argument("--generation", default=None, help="agent generation shown in the header")
    ap.add_argument("--note", default="", help="one line under the header (a replay note, the block, what the original reached)")
    ap.add_argument("--annotations", type=Path, default=None, help="JSON keyed by decision index; rendered under the decision")
    a = ap.parse_args(argv)
    record = json.loads(a.record.read_text(encoding="utf-8", errors="replace"))
    ann = json.loads(a.annotations.read_text(encoding="utf-8")) if a.annotations else {}
    a.out.write_text(render(record, a.title, a.generation, a.note, ann), encoding="utf-8")
    print(f"wrote {a.out.name}: {len(record.get('decisions') or [])} decisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
