"""M1 of the F11 diagnosis: the lockstep's re-anchor with the game's hidden state (stream counters, the draw pile in order).

The firewall (PLAN.md section 5, "The bridge, decided"): the hidden state reaches the EMULATOR side's re-anchor only. It is
never in the agent's observation, the divergence records, the fallback, the pairing or the search input, and a search
forks the position with resample_hidden=True, which redraws the order whatever the lockstep state was anchored to.
Pure parts run against fakes; the clone and search checks need the native library and skip without it."""

import json
import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))

import leakprobes  # noqa: E402
import lockstep  # noqa: E402

try:
    leakprobes.native_env()
    from sts2_gym import native  # noqa: E402

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

SENT = "SENTINEL_HIDDEN_9D3"
HIDDEN = {"rng": {"Shuffle": {"seed": 1, "counter": 99999}, SENT: {"seed": 0, "counter": 0}}, "draw": [[SENT, False]]}


def obs(hp=80, floor=1, combat=False):
    o = {"act": 0, "floor": floor, "hp": hp, "max_hp": 80, "gold": 99, "deck": [["STRIKE_IRONCLAD", 0]],
         "relics": ["BURNING_BLOOD"], "potions": []}
    if combat:
        o["combat"] = {"hand": [["STRIKE_IRONCLAD", 0, 1, True]], "enemies": [{"hp": 15}]}
    return o


def msg(kind, o, hidden=None):
    m = {"t": "decide", "kind": kind, "obs": o, "actions": [["x"]] * 2, "h": "abc"}
    if hidden is not None:
        m[lockstep.ANCHOR_KEY] = hidden
    return m


class SpyEmu:
    """An emulator side that remembers every message it is handed and the hidden state it is told."""

    wants_anchor_state = True

    def __init__(self, reanchor_ok=True):
        self.seen, self.hidden, self.anchored, self.reanchor_ok = [], [], None, reanchor_ok
        self.view_now = lockstep.ref_view(obs())

    def take_anchor_state(self, hidden):
        self.hidden.append(hidden)

    def sync(self, kind, labels, msg=None):
        self.seen.append(msg)

    def view(self):
        return self.anchored or self.view_now

    def choose(self, kind, labels, msg):
        self.seen.append(msg)
        return 0

    def apply(self, kind, labels, idx):
        return None

    def reanchor(self, plan, msg):
        self.seen.append(msg)
        self.anchored = lockstep.ref_view(msg["obs"])
        return self.reanchor_ok


class FirewallTests(unittest.TestCase):
    def test_the_hidden_state_reaches_the_emulator_sides_hook_and_no_message(self):
        emu, fallbacks = SpyEmu(), []
        ls = lockstep.Lockstep(emu, lambda m: (fallbacks.append(m), 0)[1])
        stream = [msg("map", obs(floor=1), HIDDEN),                      # in sync
                  msg("map", obs(hp=50, floor=2), HIDDEN),               # a difference at a room entry: re-anchored
                  msg("combat", obs(floor=2, combat=True), HIDDEN),
                  msg("combat", obs(floor=2, combat=True))]              # no hidden state on this one
        for m in stream:
            ls.decide(m)
        self.assertEqual(emu.hidden, [HIDDEN, HIDDEN, HIDDEN, None])  # told at every decision, None when absent
        self.assertTrue(emu.seen, "the emulator side was handed messages")
        for m in emu.seen + fallbacks:
            self.assertNotIn(lockstep.ANCHOR_KEY, m)
            self.assertNotIn(SENT, json.dumps(m))
        self.assertTrue(all(lockstep.ANCHOR_KEY in m for m in stream[:3]), "the caller's messages are not changed")
        self.assertNotIn(SENT, json.dumps(ls.log.records))
        self.assertNotIn(SENT, json.dumps(ls.stats))

    def test_an_emulator_side_that_does_not_take_it_is_never_handed_it(self):
        class Plain(SpyEmu):
            take_anchor_state = None

        emu = Plain()
        ls = lockstep.Lockstep(emu, lambda m: 0)
        ls.decide(msg("map", obs(), HIDDEN))
        self.assertTrue(all(lockstep.ANCHOR_KEY not in m for m in emu.seen))

    def test_the_agent_observation_the_driver_builds_has_no_anchor_state(self):
        src = (ROOT / "referee" / "driver" / "Env.cs").read_text(encoding="utf-8")
        self.assertEqual(src.count('["anchor_state"]'), 1, "one site writes the field: Env.Decide's message, beside obs")
        obs_body = src[src.index("internal static JsonObject Obs()"):src.index("private static bool CombatKind")]
        self.assertNotIn("anchor", obs_body.lower())
        self.assertNotIn("Rng", obs_body.replace("c.Remove(\"rng\")", ""))  # the public copy drops the counters

    def test_the_driver_is_asked_for_it_only_by_an_emulator_side_that_takes_it(self):
        class Env:
            def __init__(self):
                self.resets = []

            def reset(self, seed, **k):
                self.resets.append(k)
                return {"t": "end", "end": "death", "floor": 1}, True

        env = Env()
        lockstep.play_run(env, "S1", emu_factory=lambda s: SpyEmu())
        self.assertEqual(env.resets[-1]["spec"], {"anchor_state": True})
        lockstep.play_run(env, "S1", emu_factory=lambda s: SpyEmu(), anchor_state=False)
        self.assertNotIn("spec", env.resets[-1])
        plain = SpyEmu()
        plain.wants_anchor_state = False
        lockstep.play_run(env, "S1", emu_factory=lambda s: plain)
        self.assertNotIn("spec", env.resets[-1])  # off by default
        with mock.patch.dict(os.environ, {"SPIRELINE_ANCHOR_STATE": "0"}):
            lockstep.play_run(env, "S1", emu_factory=lambda s: SpyEmu())
        self.assertNotIn("spec", env.resets[-1])


def bare_emu(hidden):
    """An EmuLock with the native calls stubbed (the constructor needs the native library)."""
    e = lockstep.EmuLock.__new__(lockstep.EmuLock)
    e.fail = None
    e.s = SimpleNamespace(handle=3, env=SimpleNamespace(_run_obs_buf=object()))
    e.native = SimpleNamespace()
    e.take_anchor_state(hidden)
    return e


class EmuLockAnchorTests(unittest.TestCase):
    RECORD = {"hand": [["BASH", False]], "draw": [["DEFEND", False], ["STRIKE", False]], "enemies": []}
    GAME = {"rng": {"Shuffle": {"seed": 5, "counter": 12}}, "draw": [["STRIKE", False], ["DEFEND", False]]}

    def test_a_fight_start_with_the_hidden_state_anchors_streams_and_draw_order(self):
        e, calls = bare_emu(self.GAME), []
        e.native.run_reanchor_fight_hidden = lambda h, rec, buf: (calls.append(rec), (4, "moves=same;patched=draw"))[1]
        e.native.run_reanchor_fight = lambda *a: self.fail("the observation-only call was used")
        self.assertTrue(e._anchor_fight({"floor": 3, "combat": self.RECORD}))
        self.assertEqual(calls[0]["draw"], self.GAME["draw"])  # the game's order, not the sorted list
        self.assertEqual(calls[0]["rng"], self.GAME["rng"])
        self.assertEqual(calls[0]["hand"], self.RECORD["hand"])  # the public part of the record is the observation's
        self.assertEqual(e.anchored_with, "hidden")

    def test_a_fight_start_without_it_anchors_as_before(self):
        e, calls = bare_emu(None), []
        e.native.run_reanchor_fight = lambda h, rec, buf: (calls.append(rec), (1, "moves=same;patched=hand"))[1]
        e.native.run_reanchor_fight_hidden = lambda *a: self.fail("the hidden call was used with no hidden state")
        self.assertTrue(e._anchor_fight({"floor": 3, "combat": self.RECORD}))
        self.assertEqual(calls, [self.RECORD])
        self.assertEqual(e.anchored_with, "observation")

    def test_a_room_entry_puts_the_runs_streams_on_the_games(self):
        e, calls = bare_emu(self.GAME), []
        e._to_map, e._set_player, e._decision = (lambda floor=None: True), (lambda o: True), (lambda: object())
        e.native.run_debug_set_streams = lambda h, rng: (calls.append((h, rng)), 3)[1]
        self.assertTrue(e.reanchor("room_entry", {"obs": {"floor": 4}}))
        self.assertEqual(calls, [(3, self.GAME["rng"])])
        self.assertEqual(e.anchored_with, "streams")
        self.assertIsNone(e._hidden)  # consumed by the re-anchor
        e.take_anchor_state(self.GAME)
        e.native.run_debug_set_streams = lambda h, rng: -2
        self.assertFalse(e.reanchor("room_entry", {"obs": {"floor": 4}}))
        self.assertEqual(e.fail, "set_streams")
        e.take_anchor_state(None)  # no hidden state: the emulator keeps its streams, the re-anchor is as before
        e.native.run_debug_set_streams = lambda *a: self.fail("streams set with no hidden state")
        self.assertTrue(e.reanchor("room_entry", {"obs": {"floor": 4}}))


def to_fight(seed="QS2GYXRKWN"):
    e = lockstep.EmuLock(seed, sims=2)
    for _ in range(60):
        if e.s.phase == e.E.rc.PHASE_COMBAT:
            return e
        d = e.s.decision()
        e.s.apply(d.options[e.rules.pick(e.s.view(), d, e.rng).index])
    raise RuntimeError("no fight reached")


@unittest.skipUnless(HAVE_NATIVE, "needs the native emulator library")
class NativeFirewallTests(unittest.TestCase):
    def setUp(self):
        self.e = to_fight()
        self.addCleanup(self.e.close)

    def piles(self, handle, resample=False, seed=0):
        buf = (native.ctypes.c_int * native.RUN_OBS_SIZE)()
        h = native.run_fork_combat(handle, resample, seed, buf)
        try:
            return native.get_pile(h, "draw")
        finally:
            native.destroy(h)

    def anchor(self, e):
        """Anchor the fight on the game's hidden state: the draw pile in an order of its own, and streams at counters of its own."""
        h = native.run_fork_combat(e.s.handle, False, 0, (native.ctypes.c_int * native.RUN_OBS_SIZE)())
        try:
            piles = {p: native.get_pile(h, p) for p in ("hand", "draw", "discard", "exhaust")}
            roster = native.get_enemies(h)
        finally:
            native.destroy(h)
        id_map = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
        enemy = {v: re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", k).upper() for k, v in id_map["enemies"].items()}

        def pile(cards):
            return [[e.slug["cards"][c], bool(up)] for c, up in cards]

        # a draw pile of 12 distinct cards in an order of its own, so that two shuffles of it are never the same by chance
        want = ("ANGER", "CLEAVE", "IRON_WAVE", "POMMEL_STRIKE", "SHRUG_IT_OFF", "TWIN_STRIKE", "BODY_SLAM", "CLASH",
                "HEADBUTT", "WILD_STRIKE", "TRUE_GRIT", "ARMAMENTS", "BASH", "FLEX", "HAVOC", "WARCRY")
        ids = sorted(i for i, slug in e.slug["cards"].items() if slug in want)
        self.assertGreaterEqual(len(ids), 12, "the emulator knows the cards the test draws")
        draw = [(i, False) for i in ids[:12]][::-1]
        self.assertNotEqual(draw, list(piles["draw"]), "the game's order must differ from the emulator's own")
        view = e.s.view()
        public = {"in_combat": True, "round": 1,
                  "player": {"hp": view.hp, "max_hp": view.max_hp, "block": 0, "energy": 3, "powers": []},
                  "hand": pile(piles["hand"]), "draw": sorted(pile(draw)), "discard": pile(piles["discard"]), "exhaust": [],
                  "enemies": [{"name": enemy[d], "hp": hp, "max_hp": hp, "block": 0, "powers": []} for d, _s, hp in roster]}
        names = ("Shuffle", "CombatTargets", "CombatCardSelection", "CombatCardGeneration", "CombatEnergyCosts",
                 "CombatPotionGeneration", "Niche", "MonsterAi")
        hidden = {"draw": pile(draw), "rng": {n: {"seed": 700 + i, "counter": 11 + i} for i, n in enumerate(names)}}
        e.take_anchor_state(hidden)
        self.assertTrue(e.reanchor("fight_start", {"obs": {"floor": view.floor, "combat": public}}), e.fail)
        self.assertEqual(e.anchored_with, "hidden")
        self.assertIsNone(e._hidden, "the hidden state is cleared once the re-anchor has used it")
        return draw

    @staticmethod
    def spread(orders):
        """The share of clone pairs whose draw orders differ."""
        n = len(orders)
        differ = sum(1 for i in range(n) for j in range(i + 1, n) if orders[i] != orders[j])
        return differ / (n * (n - 1) // 2)

    def test_the_anchored_state_holds_the_games_order_and_the_searchs_own_clones_redraw_it(self):
        e = self.e
        game_order = self.anchor(e)
        base = tuple(game_order)
        self.assertEqual(self.piles(e.s.handle), list(game_order))  # the lockstep state is the game's
        buf = (native.ctypes.c_int * native.RUN_OBS_SIZE)()
        # (The streams' seeds are not in a combat snapshot document; RunReanchorFightTests checks them on the combat state.)
        # The search's own path (rl/live_search.py): its root is a faithful fork of the lockstep state and every
        # simulation is native.clone(root, resample_hidden=True, seed) of that root.
        root = native.run_fork_combat(e.s.handle, False, 0, buf)
        try:
            self.assertEqual(tuple(native.get_pile(root, "draw")), base)
            orders = []
            for seed in range(1, 51):
                c = native.clone(root, True, seed, buf)
                try:
                    orders.append(tuple(native.get_pile(c, "draw")))
                finally:
                    native.destroy(c)
        finally:
            native.destroy(root)
        self.assertGreaterEqual(self.spread(orders), 0.9, "at least 90% of clone pairs have different draw orders")
        self.assertLessEqual(sum(o == base for o in orders), 5, "the game's order is reproduced by a few clones at most")
        # the run-level clone (resample_hidden=True) redraws the order too
        run_orders = []
        for seed in range(1, 51):
            clone = native.run_clone(e.s.handle, True, seed, buf)
            try:
                run_orders.append(tuple(self.piles(clone)))
            finally:
                native.run_destroy(clone)
        self.assertGreaterEqual(self.spread(run_orders), 0.9)
        self.assertLessEqual(sum(o == base for o in run_orders), 5)
        self.assertEqual(self.piles(e.s.handle), list(game_order), "cloning does not move the lockstep state")

    def test_the_search_is_handed_no_hidden_state(self):
        e = self.e
        self.anchor(e)
        e.take_anchor_state({"draw": [[SENT, False]], "rng": {SENT: {"seed": 1, "counter": 2}}})
        seen = []

        def search_live(*a, **k):
            seen.append(repr((a, sorted(k.items(), key=lambda kv: kv[0]))))
            return SimpleNamespace(best=SimpleNamespace(action=0, target=-1), ai_state_assumed=False)

        with mock.patch.object(e.ls, "search_live", search_live):
            e.choose("combat", [["end"]], {"obs": obs(combat=True), "kind": "combat", "actions": [["end"]]})
        self.assertEqual(len(seen), 1)
        self.assertNotIn(SENT, seen[0])
        self.assertNotIn("draw", repr(sorted(e.cfg.search_kwargs(None).keys())))


if __name__ == "__main__":
    unittest.main()
