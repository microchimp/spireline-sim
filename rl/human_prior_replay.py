#!/usr/bin/env python3
"""The copy's half of the human prior v1: the true map of every human run, and the shop shelves and prices a replay vouches for.

    python rl/human_prior_replay.py run [--procs 2] [--limit-runs N] [--chunk-runs 40] [--resume]

A recorded run's seed fixes its three act maps (the stream `act_<k>_map`, Ascension, and a Spoils Map card in the deck for Act 2:
`docs/diagnostics/2026-11-05-human-seed-replay.md`), so the copy can draw them again. For every usable run outside the exam fifth
this writes one `truth-NNNNN.json.gz` per chunk of runs, with, per run:

  * the maps the record's route is a path through, the act by act identity check (`status`: "ok", "no_path" = the route is not a path
    of the regenerated map, "golden" = a Golden Compass had been picked up by the end of the act, so the act's map is one column), the
    reading that matched (the seed alone, or with the Spoils Map card the record gained in Act 1), Winged Boots' jumps, and the node
    column of the route at every step (-1 where several paths of the same room types fit and the record cannot tell them apart);
  * the shop visits a history replay of the run layer stays in step at (`rl/human_prior.py` v1 takes the shelf and the prices from
    them): per visit whether the replay was clean on entry, whether the copy's shelf equals the record's, the copy's price of every
    item and of the removal service, and whether the prices of what the human bought add up to the gold the record says was spent.

Fights are resolved, never played (a history replay): this is the probe of the seed replay (`scripts/replay_real_run.py`'s
`HistoryReplayer` with the card-offer set comparison, the shop shelf and price test and the event-option search that probe added).
The exam fifth is dropped before a worker sees a line (`rl/human_prior.iter_run_lines`) and `assert_no_exam` runs on every chunk.
Aggregates only leave the referee folder; the truth files stay in it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT / "scripts", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_LIB = "D:/spireline/fork/wt-lockstep-hidden/out"  # the copy's library built from the engine source of fork main (native API matches sts2_gym)
OUT_DIR = Path("D:/spireline/referee/human-prior")
NODE_BOSS = 6
SPOILS_MAP = "CARD.SPOILS_MAP"
GOLDEN_COMPASS = "RELIC.GOLDEN_COMPASS"
WINGED_BOOTS = "RELIC.WINGED_BOOTS"
BOOTS_JUMPS = 3
PRICE_CHANGING = {"MEMBERSHIP_CARD", "THE_COURIER", "SMILING_MASK"}  # relics that change the prices of the rest of a visit
PROCEED_TOKENS = ("PROCEED", "CONTINUE", "LEAVE", "DONE")


def use_copy_library() -> None:
    """Point `sts2_gym` at the copy's library (the lockstep worktree's build unless `HP_COPY_LIB` names another)."""
    os.environ.setdefault("STS2_LIB_PATH", os.environ.get("HP_COPY_LIB", DEFAULT_LIB))
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")


# ---------------------------------------------------------------------------------------------------------------- the routes


def kids_of(graph: dict) -> tuple[dict, dict, int]:
    """(children by node, nodes by row, boss row) of a map graph `{"nodes": {(col, row): type}, "edges": [(a, b)], "current": start}`."""
    kids: dict[tuple, list] = defaultdict(list)
    for a, b in graph["edges"]:
        kids[tuple(a)].append(tuple(b))
    rows: dict[int, list] = defaultdict(list)
    for (c, rw), _t in graph["nodes"].items():
        rows[rw].append((c, rw))
    boss_row = max((rw for (_c, rw), t in graph["nodes"].items() if t == NODE_BOSS), default=max(rows))
    return kids, rows, boss_row


def match_route(graph: dict, seq: list[int], boots_floor: int | None = None, base: int = 0, max_jumps: int = BOOTS_JUMPS) -> dict:
    """Which paths of `graph` from its start have the room types `seq` in order (a route that ends early is a prefix)?

    Winged Boots (picked up on floor `boots_floor`) lets the player step to any node of the next row, up to `max_jumps` times, on the
    floors after it; `base` is the number of points before this act, so the point `k + 1` of the act is floor `base + k + 2`.
    Returns `{"ok": bool, "jumps": the fewest jumps any matching path uses, "cols": the node column at every step 0..len(seq) (-1 where
    the matching paths disagree), "unique": whether exactly one path matches, "best": the longest prefix that matches}`. Only the
    matching paths with the fewest jumps count (the human is taken to have jumped no more than the record needs)."""
    nodes = graph["nodes"]
    kids, rows, boss_row = kids_of(graph)
    start = (tuple(graph["current"]), 0)
    limit = max_jumps if boots_floor is not None else 0
    n = len(seq)

    def successors(state: tuple, k: int):
        node, used = state
        opts = set(kids.get(node, ()))
        for c in opts:
            if nodes.get(c) == seq[k]:
                yield (c, used)
        floor = base + k + 2
        if limit and used < limit and boots_floor is not None and floor > boots_floor and node[1] + 1 < boss_row:
            for c in rows.get(node[1] + 1, ()):
                if c not in opts and nodes.get(c) == seq[k]:
                    yield (c, used + 1)

    layers: list[set] = [{start}]
    for k in range(n):
        nxt = {t for s in layers[k] for t in successors(s, k)}
        if not nxt:
            return {"ok": False, "jumps": 0, "cols": [], "unique": False, "best": k}
        layers.append(nxt)
    fewest = min(u for _node, u in layers[n])
    good: list[set] = [set() for _ in range(n + 1)]
    good[n] = {s for s in layers[n] if s[1] == fewest}
    for k in range(n - 1, -1, -1):
        good[k] = {s for s in layers[k] if any(t in good[k + 1] for t in successors(s, k))}
    cols = []
    unique = True
    for k in range(n + 1):
        at = {node for node, _u in good[k]}
        unique &= len(at) == 1
        cols.append(next(iter(at))[0] if len(at) == 1 else -1)
    return {"ok": True, "jumps": fewest, "cols": cols, "unique": unique, "best": n}


# ---------------------------------------------------------------------------------------------------------------- the maps


def read_act_maps(seed: str, n_acts: int, spoils: bool, rr, Sts2RunEnv, native) -> list[dict]:
    """The act maps the copy draws for `seed` (Ironclad, Ascension 10): `[{"graph", "region", "boss"}]` for Act 1 to `n_acts`. The acts
    after the first are entered by the copy's own act transition, with nothing played; `spoils` puts a Spoils Map card in the deck
    before Act 2 is drawn."""
    env = Sts2RunEnv(seed=seed, ascension=10)
    try:
        env.reset()
        out = []
        for act in range(1, n_acts + 1):
            if act > 1:
                if act == 2 and spoils:
                    info = env._info()
                    deck = [(c["card_id"], int(c["upgraded"]), c["enchantment"], c["enchant_amount"]) for c in info["deck"]]
                    deck.append((rr.CARD_IDS["SPOILS_MAP"], 0, 0, 0))
                    env.debug_set_player(cards=deck)
                env.debug_enter_next_act()
            info = env._info()
            boss = rr.ENCOUNTER_NAMES.get(native.run_state_list(env._run_handle, 14, 4)[1])
            out.append({"graph": env.map_graph(), "region": info["act"], "boss": boss})
        return out
    finally:
        env.close()


ACT_NAME = {"ACT.OVERGROWTH": "overgrowth", "ACT.UNDERDOCKS": "underdocks", "ACT.HIVE": "hive", "ACT.GLORY": "glory"}


def check_maps(r: dict, rr, Sts2RunEnv, native) -> dict:
    """The identity check of one run: for each act the record reached, whether its route is a path through the regenerated map.

    The reading order for an act: the seed alone (Act 2 first with the Spoils Map card the record gained in Act 1, when it did, since
    the engine then draws a different map), Winged Boots' jumps when the run holds them. Acts after a Golden Compass pick-up are not
    tested (the compass makes the act's map one column)."""
    mph = r["map_point_history"]
    pl = r["players"][0]
    relics = {x.get("id"): x.get("floor_added_to_deck") for x in (pl.get("relics") or [])}
    compass = relics.get(GOLDEN_COMPASS)
    boots = relics.get(WINGED_BOOTS)
    spoils_act1 = any(c.get("id") == SPOILS_MAP for p in mph[0] for c in (p["player_stats"][0].get("cards_gained") or []))
    out: dict = {"compass": compass is not None, "boots": boots, "spoils_act1": spoils_act1, "acts": []}
    plain = read_act_maps(r["seed"], len(mph), False, rr, Sts2RunEnv, native)
    withcard = read_act_maps(r["seed"], len(mph), True, rr, Sts2RunEnv, native) if spoils_act1 and len(mph) >= 2 else None
    base = 0
    for a, pts in enumerate(mph):
        ent: dict = {"act": a + 1, "points": len(pts)}
        golden = compass is not None and compass <= base + len(pts)
        seq = [rr.HISTORY_NODES[p["map_point_type"]] for p in pts[1:]]
        ent["complete"] = bool(pts and pts[-1]["map_point_type"] == "boss")
        readings = [("plain", plain[a])]
        if withcard is not None and a >= 1:
            readings.insert(0, ("spoils", withcard[a]))
        got = None
        for name, rd in readings:
            m = match_route(rd["graph"], seq, boots, base)
            if m["ok"]:
                got = (name, rd, m)
                break
        if golden:
            ent["status"] = "golden"
        elif got is None:
            ent["status"] = "no_path"
            ent["best_prefix"] = match_route(plain[a]["graph"], seq, boots, base)["best"]
        else:
            name, rd, m = got
            ent.update(status="ok", reading=name, jumps=m["jumps"], unique=m["unique"], cols=m["cols"])
            ent.update(_map_doc(rd["graph"]))
            ent["region_ok"] = ACT_NAME.get((r.get("acts") or [None] * 3)[a]) == rd["region"]
            boss_pts = [p for p in pts if p["map_point_type"] == "boss"]
            if boss_pts:
                ent["boss_ok"] = rr.encounter_model_to_emulator(boss_pts[0]["rooms"][0]["model_id"]) == rd["boss"]
        out["acts"].append(ent)
        base += len(pts)
    return out


def _map_doc(graph: dict) -> dict:
    """The map as plain lists: nodes `[col, row, type]`, edges `[col, row, col, row]`, the start `[col, row]`."""
    return {"nodes": [[c, rw, int(t)] for (c, rw), t in sorted(graph["nodes"].items(), key=lambda kv: (kv[0][1], kv[0][0]))],
            "edges": [[int(a[0]), int(a[1]), int(b[0]), int(b[1])] for a, b in graph["edges"]], "start": [int(graph["current"][0]), int(graph["current"][1])]}


# ----------------------------------------------------------------------------------------------------------- the shop replay


def make_replayer(rr, rc, native, valid_actions):
    """`ShopReplayer`, built once the copy's library is loaded: the fork's `HistoryReplayer` with the changes the probe made for this
    corpus (a card offer compared as a set and answered group by group, the shop shelf and its prices read off the copy, the event
    option found by trying the copy's options on exact clones) and a log of every shop visit."""
    event_key = rr._EVENT_CHOICE

    class ShopReplayer(rr.HistoryReplayer):
        def __init__(self, run):
            super().__init__(run, verbose=False)
            self.clean = True  # nothing the copy could not answer the human's way, no event option unresolved, no gold or HP difference so far
            self.clean2 = True  # the same, but an event option with several equal-effect answers is let through
            self.shops: list[dict] = []
            self.event_stats = Counter()
            self.floor_now = 0
            self.groups_floor = None
            self.groups_left: list = []
            self.plan = None
            self.plan_floor = None
            self.plan_i = 0
            self._shelf_ok = None

        # ---- bookkeeping ------------------------------------------------------------------------------------------
        def flow(self, floor, what, expected):
            super().flow(floor, what, expected)
            if not str(what).startswith("catching up"):
                self.clean = False
                self.clean2 = False

        def unclean(self, benign=False):
            self.clean = False
            if not benign:
                self.clean2 = False

        def outcome(self, floor, kind, room, stats):
            before = len(self.report.divergences)
            super().outcome(floor, kind, room, stats)
            if any(d.category == "outcome" for d in self.report.divergences[before:]):
                self.clean = False
                self.clean2 = False

        def enter(self, floor, kind, room, stats):
            self._shelf_ok = None
            super().enter(floor, kind, room, stats)

        def settle(self, floor, kind, stats):
            self.floor_now = floor
            super().settle(floor, kind, stats)

        # ---- card rewards -----------------------------------------------------------------------------------------
        def card_reward(self, floor, stats):
            if self.groups_floor != floor:
                self.groups_floor = floor
                allc = stats.get("card_choices") or []
                self.groups_left = [allc[i: i + 3] for i in range(0, len(allc), 3)]
            emu = self.emu_card_reward()
            emu_set = sorted(emu)
            pick_i = next((i for i, g in enumerate(self.groups_left) if sorted(rr.history_card(c.get("card") or {}) for c in g) == emu_set), None)
            if pick_i is None:
                pick_i = 0 if self.groups_left else None
            offered = self.groups_left.pop(pick_i) if pick_i is not None else []
            real = [rr.history_card(c.get("card") or {}) for c in offered]
            self.report.record("card_reward", floor, "cards offered (as a set)", sorted(real), emu_set)
            picked = next((rr.history_card(c.get("card") or {}) for c in offered if c.get("was_picked")), None)
            self.step(emu.index(picked) if picked in emu else rc.REWARD_SKIP_ACTION)

        # ---- shops ------------------------------------------------------------------------------------------------
        def compare_shelf(self, floor, stats):
            cards, relics, potions = self.emu_shelf()
            left = Counter(rr.history_card(c.get("card") or {})[0] for c in stats.get("card_choices") or [])
            gained = Counter(rr.slug(c.get("id")) for c in stats.get("cards_gained") or [])
            have = Counter(e for e, _ in cards)
            rest = have - left
            cards_ok = not (left - have) and not (rest - gained)
            a = self.report.record("shop", floor, "cards", "shelf = cards left + cards bought" if cards_ok else sorted(left.items()),
                                   "shelf = cards left + cards bought" if cards_ok else sorted(have.items()))
            b = self.report.record("shop", floor, "relics", sorted(rr.slug(c.get("choice")) for c in stats.get("relic_choices") or []), sorted(relics))
            c = self.report.record("shop", floor, "potions", sorted(rr.slug(c.get("choice")) for c in stats.get("potion_choices") or []), sorted(potions))
            self._shelf_ok = bool(a and b and c)

        def shop(self, stats):
            if self.cursor["shop done"]:
                self.step(rc.SHOP_SKIP_ACTION)
                return
            self.cursor["shop done"] += 1
            slots = self.info["shop_slots"]
            try:
                upgraded = list(native.run_state_list(self.env._run_handle, 22, 7))
            except RuntimeError:
                upgraded = []
            visit: dict = {"floor": self.floor_now, "clean": bool(self.clean), "clean2": bool(self.clean2), "shelf_ok": bool(self._shelf_ok),
                           "cards": [[int(slots[s]["item_id"]), int(slots[s]["cost"]), int(upgraded[i]) if i < len(upgraded) else 0]
                                     for i, s in enumerate(rr.SHOP_CARD_SLOTS) if slots[s]["item_id"]],
                           "relics": [[int(slots[s]["item_id"]), int(slots[s]["cost"])] for s in rr.SHOP_RELIC_SLOTS if slots[s]["item_id"]],
                           "potions": [[int(slots[s]["item_id"]), int(slots[s]["cost"])] for s in rr.SHOP_POTION_SLOTS if slots[s]["item_id"]],
                           "removal": int(slots[rc.SHOP_REMOVE_ACTION]["cost"]) if rc.SHOP_REMOVE_ACTION < len(slots) else None}
            wanted = []
            for g in stats.get("cards_gained") or []:
                wanted.append((rr.SHOP_CARD_SLOTS, rr.CARD_IDS.get(rr.slug(g.get("id")) or "")))
            for choice in stats.get("relic_choices") or []:
                if choice.get("was_picked"):
                    wanted.append((rr.SHOP_RELIC_SLOTS, rr.RELIC_IDS.get(rr.slug(choice.get("choice")) or "")))
            for choice in stats.get("potion_choices") or []:
                if choice.get("was_picked"):
                    wanted.append((rr.SHOP_POTION_SLOTS, rr.POTION_IDS.get(rr.slug(choice.get("choice")) or "")))
            spent = int(stats.get("gold_spent") or 0)
            removal = bool(stats.get("cards_removed"))
            predicted, found_all, plan, taken = 0, True, [], set()
            for pool, item in wanted:
                slot = next((s for s in pool if item and slots[s]["item_id"] == item and s not in taken), None)
                if slot is None:
                    found_all = False
                    continue
                taken.add(slot)
                predicted += int(slots[slot]["cost"])
                plan.append(slot)
            if removal:
                predicted += visit["removal"] or 0
            bought_relics = {rr.slug(c.get("choice")) for c in stats.get("relic_choices") or [] if c.get("was_picked")}
            visit.update(purchase=bool(wanted or removal), spent=spent, predicted=predicted, found_all=found_all,
                         exact=(predicted == spent) if (wanted or removal) else None,
                         moving=bool(bought_relics & PRICE_CHANGING))
            self.shops.append(visit)
            for slot in plan:
                if slot in valid_actions(self.env):
                    self.step(slot)
                    if self.phase != rc.PHASE_SHOP:
                        return
            if removal and rc.SHOP_REMOVE_ACTION in valid_actions(self.env):
                self.step(rc.SHOP_REMOVE_ACTION)
                return
            self.step(rc.SHOP_SKIP_ACTION)

        # ---- events -----------------------------------------------------------------------------------------------
        def event_option(self, stats):
            keys = [str((c.get("title") or {}).get("key") or "") for c in stats.get("event_choices") or []]
            keys = [k for k in keys if event_key.match(k)]
            floor = self.floor_now
            if self.plan_floor != floor:
                self.plan_floor = floor
                self.plan = None
                self.plan_i = 0
                if keys and self.phase == rc.PHASE_EVENT:
                    self.plan = self.find_plan(stats, keys)
            if self.plan is not None:
                if self.plan_i < len(self.plan):
                    a = self.plan[self.plan_i]
                    self.plan_i += 1
                    if a in valid_actions(self.env):
                        self.step(a)
                        return
                if self.proceed():
                    return
                raise rr.UnresolvableChoice("event: nothing left to answer")
            self.event_stats["unresolved"] += 1
            return super().event_option(stats)

        def find_plan(self, stats, keys):
            depth = max(1, sum(1 for k in keys if event_key.match(k).group(3) not in PROCEED_TOKENS))
            want = {"gold": int(stats.get("current_gold") or 0), "max_hp": int(stats.get("max_hp") or 0), "hp": int(stats.get("current_hp") or 0)}
            deck0 = Counter(int(c["card_id"]) for c in self.info["deck"])
            relics0 = len([x for x in self.info["relics"] if int(x)])
            want_deck = Counter(deck0)
            for c in stats.get("cards_gained") or []:
                want_deck[rr.CARD_IDS.get(rr.slug(c.get("id")) or "", -1)] += 1
            for c in stats.get("cards_removed") or []:
                want_deck[rr.CARD_IDS.get(rr.slug(c.get("id")) or "", -1)] -= 1
            for t in stats.get("cards_transformed") or []:
                want_deck[rr.CARD_IDS.get(rr.slug((t.get("original_card") or {}).get("id")) or "", -1)] -= 1
                want_deck[rr.CARD_IDS.get(rr.slug((t.get("final_card") or {}).get("id")) or "", -1)] += 1
            want_deck = +want_deck
            want_relics = relics0 + sum(1 for c in stats.get("relic_choices") or [] if c.get("was_picked"))
            want_up = sum(1 for c in self.info["deck"] if c["upgraded"]) + len(stats.get("upgraded_cards") or [])
            results = []

            def evaluate(info):
                """(deficit, achievable): the points the leaf could have earned and did not, and how many it could have."""
                got = top = 0
                ph = int(info["phase"])
                if ph != rc.PHASE_COMBAT:
                    top += 2
                    got += 2 * (int(info["gold"]) == want["gold"])
                top += 1
                got += int(info["player_max_hp"]) == want["max_hp"]
                if ph != rc.PHASE_COMBAT:
                    top += 1
                    got += int(info["player_hp"]) == want["hp"]
                if ph in (rc.PHASE_MAP, rc.PHASE_EVENT, rc.PHASE_RELIC_REWARD, rc.PHASE_CARD_REWARD):
                    have = Counter(int(c["card_id"]) for c in info["deck"])
                    top += 4
                    got += 2 * (have == want_deck)
                    got += len([x for x in info["relics"] if int(x)]) == want_relics
                    got += sum(1 for c in info["deck"] if c["upgraded"]) == want_up
                return top - got, top

            def explore(env, path, d):
                info = env._info()
                ph = int(info["phase"])
                if d >= depth or ph != rc.PHASE_EVENT:
                    steps = 0
                    while int(env._info()["phase"]) == rc.PHASE_EVENT and steps < 4:
                        env.step(rc.EVENT_SKIP_ACTION)
                        steps += 1
                    results.append((evaluate(env._info()), tuple(path)))
                    return
                acts = valid_actions(env)
                if len(acts) > 1 and rc.EVENT_SKIP_ACTION in acts and int(info["event_id"]) != 41:
                    acts = [x for x in acts if x != rc.EVENT_SKIP_ACTION]  # "leave" is the way out of a page, not a choice, unless it is all there is
                for a in acts:
                    c = env.clone(resample_hidden=False)
                    try:
                        c.step(a)
                        explore(c, path + [a], d + 1)
                    finally:
                        c.close()

            first = self.env.clone(resample_hidden=False)
            try:
                explore(first, [], 0)
            finally:
                first.close()
            if not results:
                self.event_stats["no_candidates"] += 1
                return None
            best = min((d, -t) for (d, t), _ in results)
            top = sorted({p for (d, t), p in results if (d, -t) == best})
            firsts = {p[0] for p in top}
            self.event_stats["resolved"] += 1
            if len(firsts) > 1:
                self.event_stats["ambiguous"] += 1
                self.unclean(benign=best[0] == 0)
            elif best[0] > 0:
                self.event_stats["partial_match"] += 1
                self.unclean()
            return list(top[0])

    return ShopReplayer


# ---------------------------------------------------------------------------------------------------------------- one run

_LIB: dict = {}


def _load() -> dict:
    """The copy's library and the fork's replayer, once per process."""
    if not _LIB:
        use_copy_library()
        import replay_real_run as rr  # noqa: PLC0415
        from sts2_gym import Sts2RunEnv, native  # noqa: PLC0415
        from sts2_gym import run_constants as rc  # noqa: PLC0415
        from sts2_gym.commands import valid_actions  # noqa: PLC0415

        _LIB.update(rr=rr, rc=rc, native=native, Sts2RunEnv=Sts2RunEnv, valid_actions=valid_actions)
        _LIB["ShopReplayer"] = make_replayer(rr, rc, native, valid_actions)
    return _LIB


def truth_of_run(r: dict, replay: bool = True) -> dict:
    """The truth record of one run: the map identity check and (when the maps all hold) the shop replay."""
    import human_starts as hs  # noqa: PLC0415

    lib = _load()
    rr = lib["rr"]
    doc: dict = {"id": hs.run_id(r)}
    t0 = time.time()
    try:
        doc.update(check_maps(r, rr, lib["Sts2RunEnv"], lib["native"]))
    except Exception as exc:  # a run the copy cannot draw at all
        doc["maps_crash"] = f"{type(exc).__name__}: {exc}"[:160]
        doc["acts"] = []
        return doc
    doc["maps_seconds"] = round(time.time() - t0, 2)
    usable = not doc["compass"] and all(a["status"] == "ok" for a in doc["acts"])
    if replay and usable:
        t1 = time.time()
        try:
            run = rr.RealRun(seed=r["seed"], ascension=10, states=[], label="run", history=r)
            rp = lib["ShopReplayer"](run)
            rep = rp.replay()
            doc["shops"] = rp.shops
            doc["replay"] = {"floors": rep.floors, "clean_end": bool(rp.clean), "clean2_end": bool(rp.clean2), "stopped": bool(rep.stopped),
                             "events": dict(rp.event_stats)}
        except Exception as exc:
            doc["replay_crash"] = f"{type(exc).__name__}: {exc}"[:160]
            doc["shops"] = []
        doc["replay_seconds"] = round(time.time() - t1, 2)
    return doc


def run_chunk(job: dict) -> dict:
    """A chunk of raw run lines -> one `truth-NNNNN.json.gz`. Runs in a pool worker."""
    import human_starts as hs  # noqa: PLC0415

    exam = hs.read_exam_ids(job["exam_ids"])
    docs, ids = [], []
    for line in job["lines"]:
        r = json.loads(line)
        rid = hs.run_id(r)
        if hs.is_exam(rid, exam):
            raise AssertionError(f"human_prior_replay: run {rid} belongs to the exam fifth and may not be read")
        ids.append(rid)
        docs.append(truth_of_run(r, replay=not job.get("maps_only")))
    hs.assert_no_exam(ids, exam)
    path = Path(job["out"])
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump({"run_ids": ids, "runs": docs}, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return {"path": str(path), "runs": len(ids)}


def below_normal() -> None:
    import human_starts as hs  # noqa: PLC0415

    hs.below_normal()


def cmd_run(a: argparse.Namespace) -> int:
    import multiprocessing as mp  # noqa: PLC0415
    from collections import Counter as C  # noqa: PLC0415

    import human_prior as hp  # noqa: PLC0415
    import human_starts as hs  # noqa: PLC0415
    from spawn_pool import SupervisedPool  # noqa: PLC0415

    if a.procs > 2:
        raise SystemExit("human_prior_replay: at most 2 processes on the copy")
    if a.lib:
        os.environ["HP_COPY_LIB"] = a.lib
    use_copy_library()
    below_normal()
    out_dir = Path(a.out_dir) / "truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    exam = hs.read_exam_ids(a.exam_ids)
    pool = SupervisedPool(mp.get_context("spawn"), max(1, a.procs), initializer=below_normal, run_timeout_s=3600.0)
    t0 = time.time()
    parent: Counter = C()
    pending: list[tuple[int, dict]] = []
    jobs: dict[int, dict] = {}
    done = 0
    retried: set[int] = set()
    lost = 0

    def submit_ready() -> None:
        while pending and pool.free():
            k, job = pending.pop(0)
            pool.submit(k, run_chunk, (job,))

    def drain() -> None:
        nonlocal done, lost
        for d in pool.collect():
            if d.status == "ok":
                done += 1
                if done % 10 == 0:
                    print(f"human_prior_replay: {done} chunks done ({time.time() - t0:.0f} s)", flush=True)
            elif d.key not in retried:
                retried.add(d.key)
                pending.append((d.key, jobs[d.key]))
            else:
                lost += 1
                print(f"human_prior_replay: chunk {d.key} lost: {d.status} {d.value}", file=sys.stderr)

    chunk: list[str] = []
    ix = 0

    def add(lines: list[str]) -> None:
        nonlocal ix
        path = out_dir / f"truth-{ix:05d}.json.gz"
        if a.resume and path.is_file():
            ix += 1
            return
        job = {"lines": lines, "out": str(path), "exam_ids": str(a.exam_ids), "maps_only": a.maps_only}
        jobs[ix] = job
        pending.append((ix, job))
        ix += 1

    try:
        for line in hp.iter_run_lines(Path(a.corpus), exam, parent, a.limit_runs):
            chunk.append(line)
            if len(chunk) >= a.chunk_runs:
                add(chunk)
                chunk = []
            submit_ready()
            drain()
            while len(pending) > 2 * a.procs:
                pool.wait(0.2)
                submit_ready()
                drain()
        if chunk:
            add(chunk)
        while pending or pool.inflight():
            submit_ready()
            pool.wait(0.3)
            drain()
    finally:
        pool.close()
    info = {"chunks": ix, "chunks_done": done, "chunks_lost": lost, "seconds": round(time.time() - t0, 1), "parent": dict(parent)}
    (out_dir / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))
    return 1 if lost else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--corpus", default=str(Path("D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz")))
    r.add_argument("--exam-ids", default=str(Path("D:/spireline/data/spire-codex/human-exam-fifth-ids.txt")))
    r.add_argument("--out-dir", default=str(OUT_DIR))
    r.add_argument("--procs", type=int, default=2)
    r.add_argument("--chunk-runs", type=int, default=40)
    r.add_argument("--limit-runs", type=int, default=0)
    r.add_argument("--lib", default="", help="the copy's library folder (default: the lockstep worktree's build)")
    r.add_argument("--resume", action="store_true", help="keep the chunk files already written (same chunking)")
    r.add_argument("--maps-only", action="store_true", help="the identity check only, no shop replay")
    a = ap.parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
