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
from collections import namedtuple
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


class DeckOrderTests(unittest.TestCase):
    """The hidden fight-start re-anchor also puts the emulator's run deck in the game's order (obs["deck"], `Deck.Cards`
    order): the next fight's opening shuffle is position based and the comparer holds the deck as a multiset (F11 mechanism A)."""

    Card = namedtuple("Card", "card_id upgraded enchantment enchant_amount")
    GAME = EmuLockAnchorTests.GAME
    RECORD = EmuLockAnchorTests.RECORD
    IDS = {"A": 1, "B": 2, "C": 3}

    def c(self, name, up=False, ench=0, amount=0):
        return self.Card(self.IDS[name], up, ench, amount)

    def emu(self, deck, hidden=GAME):
        """A fake session whose State.Deck is `deck`; Sts2Run_DebugSetPlayer replaces it with the cards it is handed."""
        e = bare_emu(hidden)
        e.ids = {"cards": dict(self.IDS)}
        state, calls = {"deck": list(deck)}, []
        e.F = SimpleNamespace(engine_deck=lambda env: tuple(state["deck"]))
        e._obs = lambda: "obs"

        def set_player(handle, buf, *, cards=None, **kw):
            calls.append((handle, buf, cards, kw))
            state["deck"] = [self.Card(*x) for x in cards]

        e.native.run_debug_set_player = set_player
        e.native.run_reanchor_fight_hidden = lambda h, rec, buf: (4, "moves=same;patched=draw")
        e.native.run_reanchor_fight = lambda h, rec, buf: (1, "moves=same;patched=hand")
        return e, state, calls

    def anchor(self, e, deck):
        obs = {"floor": 3, "combat": self.RECORD}
        if deck is not None:
            obs["deck"] = deck
        return e._anchor_fight(obs)

    def test_a_deck_in_another_order_is_put_in_the_games_order(self):
        e, state, calls = self.emu([self.c("A"), self.c("B"), self.c("C")])
        self.assertTrue(self.anchor(e, [["A", 0], ["C", 0], ["B", 0]]))
        self.assertEqual(state["deck"], [self.c("A"), self.c("C"), self.c("B")])
        self.assertEqual(e.deck_order, "set")
        self.assertEqual(len(calls), 1)
        handle, buf, cards, kw = calls[0]
        self.assertEqual((handle, buf), (3, "obs"))
        self.assertEqual(kw, {})  # the cards alone: hp, gold, relics and potions are left as they are

    def test_each_card_keeps_its_own_upgrade_and_enchantment(self):
        e, state, _ = self.emu([self.c("A"), self.c("A", up=True), self.c("B", ench=5, amount=2), self.c("A", ench=7, amount=1)])
        self.assertTrue(self.anchor(e, [["B", 0], ["A", 0], ["A", 1], ["A", 0]]))
        self.assertEqual(state["deck"], [self.c("B", ench=5, amount=2), self.c("A"), self.c("A", up=True), self.c("A", ench=7, amount=1)])

    def test_a_deck_already_in_that_order_is_left_alone(self):
        e, state, calls = self.emu([self.c("A"), self.c("B"), self.c("C")])
        self.assertTrue(self.anchor(e, [["A", 0], ["B", 0], ["C", 0]]))
        self.assertEqual((e.deck_order, calls), ("same", []))

    def test_another_deck_is_not_the_re_anchors_to_set(self):
        # A card more, fewer, or another upgrade state is a content difference: the comparer reports it as it always did.
        for obs_deck in ([["A", 0], ["C", 0], ["C", 0]], [["A", 0], ["B", 0]], [["A", 0], ["B", 1], ["C", 0]]):
            e, state, calls = self.emu([self.c("A"), self.c("B"), self.c("C")])
            self.assertTrue(self.anchor(e, obs_deck))
            self.assertEqual((e.deck_order, calls, state["deck"][1]), ("other_deck", [], self.c("B")), obs_deck)

    def test_no_deck_in_the_observation_or_an_unknown_card_sets_nothing(self):
        e, state, calls = self.emu([self.c("A"), self.c("B")])
        self.assertTrue(self.anchor(e, None))
        self.assertEqual((e.deck_order, calls), (None, []))
        self.assertTrue(self.anchor(e, [["B", 0], ["ZZZ", 0]]))
        self.assertEqual((e.deck_order, calls), ("unknown_id", []))

    def test_without_the_hidden_state_the_deck_is_left_alone(self):
        e, state, calls = self.emu([self.c("A"), self.c("B"), self.c("C")], hidden=None)
        self.assertTrue(self.anchor(e, [["A", 0], ["C", 0], ["B", 0]]))
        self.assertEqual((e.anchored_with, e.deck_order, calls), ("observation", None, []))

    def test_a_failed_anchor_sets_nothing(self):
        e, state, calls = self.emu([self.c("A"), self.c("B"), self.c("C")])
        e.native.run_reanchor_fight_hidden = lambda h, rec, buf: (-4, "unknown card")
        self.assertFalse(self.anchor(e, [["A", 0], ["C", 0], ["B", 0]]))
        self.assertEqual(calls, [])

    def test_the_fight_start_plan_reaches_it_and_the_room_entry_plan_does_not(self):
        e, state, calls = self.emu([self.c("A"), self.c("B"), self.c("C")])
        e.E = SimpleNamespace(rc=SimpleNamespace(PHASE_COMBAT=7))
        e.s.phase, e.s.view = 7, lambda: SimpleNamespace(floor=3)
        obs = {"floor": 3, "combat": self.RECORD, "deck": [["A", 0], ["C", 0], ["B", 0]]}
        self.assertTrue(e.reanchor("fight_start", {"obs": obs}))
        self.assertEqual(state["deck"], [self.c("A"), self.c("C"), self.c("B")])
        e._order_deck = lambda obs: self.fail("the room-entry path reorders the deck")
        e._to_map, e._set_player, e._set_streams, e._decision = (lambda floor=None: True), (lambda o: True), (lambda: True), (lambda: object())
        self.assertTrue(e.reanchor("room_entry", {"obs": obs}))


class SetPlayerEnchantmentTests(unittest.TestCase):
    """The room-entry re-anchor (`EmuLock._set_player`) writes the game's deck to the emulator. The game's record gives a card's id
    and upgrade only, so it used to write `(id, upgraded, 0, 0)` and strip every card's enchantment (a Power's Swift, a
    play-twice enchantment) at each room entry. A card the emulator's own deck already holds, matched by (id, upgraded) in
    order as `_order_deck` does, keeps its enchantment and the enchantment's amount (the tuple's third and fourth numbers);
    a card new to the emulator is written plain."""

    Card = DeckOrderTests.Card
    IDS = {"A": 1, "B": 2, "C": 3}

    def c(self, name, up=False, ench=0, amount=0):
        return self.Card(self.IDS[name], up, ench, amount)

    def emu(self, deck):
        e = bare_emu(None)
        e.ids = {"cards": dict(self.IDS), "relics": {"R": 9}, "potions": {"P": 8}}
        e.F = SimpleNamespace(engine_deck=lambda env: tuple(deck))
        e._obs = lambda: "obs"
        calls = []
        e.native.run_debug_set_player = lambda handle, buf, **kw: calls.append((handle, buf, kw))
        return e, calls

    def room_entry(self, e, game_deck):
        o = {"floor": 4, "hp": 50, "max_hp": 80, "gold": 120, "deck": game_deck, "relics": ["R"], "potions": ["P", 0]}
        e._to_map, e._set_streams, e._decision = (lambda floor=None: True), (lambda: True), (lambda: object())
        self.assertTrue(e.reanchor("room_entry", {"obs": o}))

    def test_a_room_entry_keeps_the_enchantment_the_emulators_deck_holds(self):
        e, calls = self.emu([self.c("A", ench=5, amount=2), self.c("B", up=True, ench=3, amount=1), self.c("C")])
        self.room_entry(e, [["A", 0], ["B", 1], ["C", 0]])
        (handle, buf, kw), = calls
        self.assertEqual(kw["cards"], [(1, 0, 5, 2), (2, 1, 3, 1), (3, 0, 0, 0)])
        # the rest of the write is as before
        self.assertEqual((handle, buf), (3, "obs"))
        self.assertEqual((kw["hp"], kw["max_hp"], kw["gold"], kw["relics"], kw["potions"], kw["max_potion_slots"]),
                         (50, 80, 120, [9], [8, 0], 2))

    def test_a_reorder_or_an_added_card_does_not_strip_what_the_emulator_knows(self):
        e, calls = self.emu([self.c("A", ench=5, amount=2), self.c("B")])
        self.room_entry(e, [["C", 0], ["B", 0], ["A", 0]])  # another order, and a card more
        self.assertEqual(calls[0][2]["cards"], [(3, 0, 0, 0), (2, 0, 0, 0), (1, 0, 5, 2)])

    def test_a_card_new_to_the_emulator_is_written_plain(self):
        # a card the emulator does not hold, and one it holds in the other upgrade state, are both new to it
        e, calls = self.emu([self.c("A", ench=5, amount=2), self.c("B", ench=4, amount=1)])
        self.room_entry(e, [["A", 0], ["A", 1], ["C", 0], ["B", 1]])
        self.assertEqual(calls[0][2]["cards"], [(1, 0, 5, 2), (1, 1, 0, 0), (3, 0, 0, 0), (2, 1, 0, 0)])

    def test_duplicates_keep_the_enchantment_on_the_matched_position(self):
        for emu_deck, want in (([self.c("A"), self.c("A", ench=7, amount=1)], [(1, 0, 0, 0), (1, 0, 7, 1)]),
                               ([self.c("A", ench=7, amount=1), self.c("A")], [(1, 0, 7, 1), (1, 0, 0, 0)])):
            e, calls = self.emu(emu_deck)
            self.room_entry(e, [["A", 0], ["A", 0]])
            self.assertEqual(calls[0][2]["cards"], want, emu_deck)
        # fewer copies in the game's deck than the emulator's: the first ones match, in order
        e, calls = self.emu([self.c("A", ench=7, amount=1), self.c("A"), self.c("A", ench=2, amount=2)])
        self.room_entry(e, [["A", 0], ["A", 0]])
        self.assertEqual(calls[0][2]["cards"], [(1, 0, 7, 1), (1, 0, 0, 0)])

    def test_the_fight_start_walk_keeps_them_too(self):
        # `_set_player` also runs on the fight-start plan when the emulator stands at an earlier floor
        e, calls = self.emu([self.c("A", ench=5, amount=2)])
        e.E = SimpleNamespace(rc=SimpleNamespace(PHASE_COMBAT=7))
        e.s.phase, e.s.view = 7, lambda: SimpleNamespace(floor=1)
        e._to_map = lambda floor=None: True
        e._decision = lambda: SimpleNamespace(options=[object()])
        e.s.apply, e._auto_combat = (lambda o: None), (lambda: None)
        e._anchor_fight = lambda obs: True
        o = {"floor": 4, "hp": 50, "max_hp": 80, "gold": 120, "deck": [["A", 0]], "relics": ["R"], "potions": []}
        self.assertTrue(e.reanchor("fight_start", {"obs": o}))
        self.assertEqual(calls[0][2]["cards"], [(1, 0, 5, 2)])

    def test_an_unknown_card_writes_nothing(self):
        e, calls = self.emu([self.c("A", ench=5, amount=2)])
        self.assertFalse(e._set_player({"hp": 5, "max_hp": 80, "gold": 1, "deck": [["A", 0], ["ZZZ", 0]], "relics": [], "potions": []}))
        self.assertEqual(calls, [])


class StreamsMovedTests(unittest.TestCase):
    """Review 1.3: a room-entry stream anchor says how many streams the game named and how many the native call moved; a call
    that moved none is not reported as a stream anchor."""

    def rng(self):
        r = {n: {"seed": 7, "counter": 5} for n in sorted(lockstep.EMU_STREAM_NAMES)}
        r["Nope"] = {"seed": 7, "counter": 5}  # a name the emulator has no stream for
        return r

    def room_entry(self, moved):
        e = bare_emu({"rng": self.rng()})
        e._to_map, e._set_player, e._decision = (lambda floor=None: True), (lambda o: True), (lambda: object())
        e.native.run_debug_set_streams = lambda h, rng: moved
        self.assertTrue(e.reanchor("room_entry", {"obs": {"floor": 4}}))
        return e

    def test_seventeen_names_of_which_one_is_unknown_report_one_unmoved(self):
        self.assertEqual(len(lockstep.EMU_STREAM_NAMES), 16)
        e = self.room_entry(16)
        self.assertEqual(e.streams, {"named": 17, "moved": 16, "unmoved": 1, "unmoved_names": ["Nope"]})
        self.assertEqual(e.anchored_with, "streams")

    def test_zero_moved_is_not_a_stream_anchor(self):
        e = self.room_entry(0)
        self.assertEqual((e.streams["named"], e.streams["moved"], e.streams["unmoved"]), (17, 0, 17))
        self.assertIsNone(e.anchored_with)

    def test_a_failed_call_moved_none_and_the_next_reanchor_starts_clean(self):
        e = bare_emu({"rng": self.rng()})
        e._to_map, e._set_player, e._decision = (lambda floor=None: True), (lambda o: True), (lambda: object())
        e.native.run_debug_set_streams = lambda h, rng: -2
        self.assertFalse(e.reanchor("room_entry", {"obs": {"floor": 4}}))
        self.assertEqual((e.streams["moved"], e.anchored_with), (0, None))
        e.take_anchor_state(None)  # no hidden state at all: no stream report is left from the earlier re-anchor
        self.assertTrue(e.reanchor("room_entry", {"obs": {"floor": 4}}))
        self.assertIsNone(e.streams)

    def test_the_emulator_name_set_is_the_native_mapping(self):
        src = (ROOT / "src" / "Sts2Emulator" / "Interop" / "RunNativeExports.cs").read_text(encoding="utf-8")
        body = src.split("StreamNamed(RunEngine run, string name)", 1)[1].split("_ => null", 1)[0]
        names = {n for line in body.splitlines() if "=>" in line for n in re.findall(r'"(\w+)"', line.split("=>")[0])}
        self.assertEqual(names, set(lockstep.EMU_STREAM_NAMES))


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

    def anchor(self, e, deck=None):
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
        obs = {"floor": view.floor, "combat": public, **({"deck": deck} if deck is not None else {})}
        self.assertTrue(e.reanchor("fight_start", {"obs": obs}), e.fail)
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

    def test_a_fight_start_puts_the_run_deck_in_the_games_order(self):
        e = self.e
        before = list(e.F.engine_deck(e.s.env))
        game = before[::-1]  # the game's deck: the same cards the other way round
        self.assertNotEqual(before, game)
        draw = self.anchor(e, deck=[[e.slug["cards"][c.card_id], int(c.upgraded)] for c in game])
        self.assertEqual(e.deck_order, "set")
        self.assertEqual(list(e.F.engine_deck(e.s.env)), game, "State.Deck is in the game's order")
        self.assertEqual(self.piles(e.s.handle), list(draw), "the fight in progress is left as the re-anchor put it")
        self.assertEqual(sorted(map(str, e.s.view().deck)), sorted(map(str, before)), "the same cards")

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


import re as _re  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_REPO = _Path(__file__).resolve().parents[2]


class GameCardAndRelicStateTests(unittest.TestCase):
    """D11F2: the driver's anchor_state carries each deck card's enchantment and Mad Science's tinker fields and each relic's saved
    counters (referee/driver/AnchorHoldings.cs). The room-entry and fight-start re-anchors write them over D11F1's emulator-side
    guess, which stays the fallback where the message has none (`SetPlayerEnchantmentTests`)."""

    Card = DeckOrderTests.Card
    IDS = {"A": 1, "INFERNO": 2, "MAD_SCIENCE": 3}
    RELICS = {"R": 9, "PENDULUM": 10, "SILVER_CRUCIBLE": 11, "JOSS_PAPER": 12, "POLLINOUS_CORE": 13}

    def c(self, name, up=False, ench=0, amount=0):
        return self.Card(self.IDS[name], up, ench, amount)

    def emu(self, deck, hidden):
        e = bare_emu(None)
        e.ids = {"cards": dict(self.IDS), "relics": dict(self.RELICS), "potions": {"P": 8}}
        e.F = SimpleNamespace(engine_deck=lambda env: tuple(deck))
        e._obs = lambda: "obs"
        calls = []
        e.native.run_debug_set_player = lambda handle, buf, **kw: calls.append((handle, buf, kw))
        e._hidden = hidden
        return e, calls

    def room_entry(self, e, game_deck, relics=("R",)):
        o = {"floor": 4, "hp": 50, "max_hp": 80, "gold": 120, "deck": game_deck, "relics": list(relics), "potions": ["P"]}
        e._to_map, e._set_streams, e._decision = (lambda floor=None: True), (lambda: True), (lambda: object())
        hidden = e._hidden
        self.assertTrue(e.reanchor("room_entry", {"obs": o}))
        e._hidden = hidden

    def test_an_enchantment_the_emulator_never_held_is_set(self):
        e, calls = self.emu([self.c("INFERNO"), self.c("A")], {"deck": [{"id": "INFERNO", "up": 0, "ench": "SWIFT", "amount": 2}, {"id": "A", "up": 0}]})
        self.room_entry(e, [["INFERNO", 0], ["A", 0]])
        self.assertEqual(calls[0][2]["cards"], [(2, 0, 3, 2), (1, 0, 0, 0)])  # Swift is ordinal 3
        self.assertNotIn("card_tinker", calls[0][2])  # no Mad Science: the original call shape
        self.assertNotIn("relic_counters", calls[0][2])

    def test_the_game_names_the_right_copy_among_duplicates_and_a_plain_one_stays_plain(self):
        # D11F1 would pair the emulator's enchanted first copy; the game says the second copy is the enchanted one
        e, calls = self.emu([self.c("A", ench=7, amount=1), self.c("A"), self.c("INFERNO", ench=3, amount=1)],
                            {"deck": [{"id": "A", "up": 0}, {"id": "A", "up": 0, "ench": "VIGOROUS", "amount": 1}, {"id": "INFERNO", "up": 0}]})
        self.room_entry(e, [["A", 0], ["A", 0], ["INFERNO", 0]])
        self.assertEqual(calls[0][2]["cards"], [(1, 0, 0, 0), (1, 0, 9, 1), (2, 0, 0, 0)])  # Vigorous is 9; the emulator's Swift is gone

    def test_mad_sciences_tinker_fields_go_with_the_cards_in_their_order(self):
        e, calls = self.emu([self.c("A"), self.c("MAD_SCIENCE")],
                            {"deck": [{"id": "A", "up": 0}, {"id": "MAD_SCIENCE", "up": 1, "tinker_type": 2, "tinker_rider": 5}]})
        self.room_entry(e, [["A", 0], ["MAD_SCIENCE", 1]])
        self.assertEqual(calls[0][2]["card_tinker"], [(-1, 0), (2, 5)])

    def test_an_unknown_enchantment_or_a_deck_that_does_not_line_up_keeps_d11f1s_guess(self):
        for deck in ([{"id": "INFERNO", "up": 0, "ench": "SLUMBERING_ESSENCE", "amount": 1}],  # no ordinal for it
                     [{"id": "A", "up": 0, "ench": "SWIFT", "amount": 2}],  # another card at the position
                     [{"id": "INFERNO", "up": 1, "ench": "SWIFT", "amount": 2}],  # another upgrade
                     [{"id": "INFERNO", "up": 0}, {"id": "A", "up": 0}]):  # another length
            e, calls = self.emu([self.c("INFERNO", ench=5, amount=2)], {"deck": deck})
            self.room_entry(e, [["INFERNO", 0]])
            self.assertEqual(calls[0][2]["cards"], [(2, 0, 5, 2)], deck)

    def test_without_the_drivers_data_it_is_d11f1s_write_unchanged(self):
        for hidden in (None, {"rng": {}}, {"deck": None, "relics": None}):
            e, calls = self.emu([self.c("INFERNO", ench=5, amount=2)], hidden)
            self.room_entry(e, [["INFERNO", 0]])
            self.assertEqual(calls[0][2]["cards"], [(2, 0, 5, 2)])
            self.assertNotIn("relic_counters", calls[0][2])
            self.assertNotIn("card_tinker", calls[0][2])

    def test_the_pendulum_counter_is_set_at_a_room_entry(self):
        relics = [{"id": "R"}, {"id": "PENDULUM", "ints": {"TurnsSeen": 2}}, {"id": "SILVER_CRUCIBLE", "ints": {"TimesUsed": 1}},
                  {"id": "JOSS_PAPER"}, {"id": "POLLINOUS_CORE", "ints": {"TurnsSeen": 0}}]
        e, calls = self.emu([self.c("A")], {"relics": relics})
        self.room_entry(e, [["A", 0]], relics=["R", "PENDULUM", "SILVER_CRUCIBLE", "JOSS_PAPER", "POLLINOUS_CORE"])
        # Pendulum 2; the crucible counts down from three uses (3 - 1); Joss Paper's absent property is its default 0; the core 0
        self.assertEqual(calls[0][2]["relic_counters"], [(10, 2), (11, 2), (12, 0), (13, 0)])
        self.assertEqual(e.relic_counters, 4)

    def test_the_fight_start_anchor_sets_the_counters_on_the_open_fight(self):
        hidden = {**EmuLockAnchorTests.GAME, "relics": [{"id": "PENDULUM", "ints": {"TurnsSeen": 1}}]}
        e, calls = self.emu([self.c("A")], hidden)
        e.native.run_reanchor_fight_hidden = lambda h, rec, buf: (4, "moves=same;patched=draw")
        self.assertTrue(e._anchor_fight({"floor": 3, "combat": EmuLockAnchorTests.RECORD}))
        self.assertEqual([kw for _, _, kw in calls], [{"relic_counters": [(10, 1)]}])  # counters alone: nothing else is touched
        # the observation-only anchor (no hidden state) and a hidden state without relics write none
        for h in (None, dict(EmuLockAnchorTests.GAME)):
            e, calls = self.emu([self.c("A")], h)
            e.native.run_reanchor_fight_hidden = lambda hd, rec, buf: (4, "x")
            e.native.run_reanchor_fight = lambda hd, rec, buf: (1, "x")
            self.assertTrue(e._anchor_fight({"floor": 3, "combat": EmuLockAnchorTests.RECORD}))
            self.assertEqual(calls, [])

    def test_the_fight_start_reorder_carries_the_games_enchantments_too(self):
        hidden = {**EmuLockAnchorTests.GAME, "deck": [{"id": "A", "up": 0, "ench": "SWIFT", "amount": 1}, {"id": "INFERNO", "up": 0}]}
        e, calls = self.emu([self.c("A"), self.c("INFERNO")], hidden)  # the same order, but the game's Swift is new to the emulator
        e.native.run_reanchor_fight_hidden = lambda h, rec, buf: (4, "x")
        self.assertTrue(e._anchor_fight({"floor": 3, "combat": EmuLockAnchorTests.RECORD, "deck": [["A", 0], ["INFERNO", 0]]}))
        self.assertEqual((e.deck_order, calls[0][2]["cards"]), ("set", [(1, 0, 3, 1), (2, 0, 0, 0)]))

    def walk(self, hidden):
        """The fight_start plan on the walk path (the emulator is not in the fight yet): `_set_player`, the map step that enters
        the fight, then the anchor. Entering the fight ticks the emulator's Pendulum, as the game's own turn-1 tick already
        did in the counter the driver carries (2)."""
        e, calls = self.emu([self.c("A")], hidden)
        state = {"counter": None, "gold": None}

        def set_player(handle, buf, **kw):
            calls.append(kw)
            for _relic, value in kw.get("relic_counters", []):
                state["counter"] = value
            if kw.get("gold", -1) >= 0:
                state["gold"] = kw["gold"]

        def enter_fight(option):
            state["counter"] = (state["counter"] or 0) + 1  # the emulator's own fight-start tick
            state["gold"] += 12  # the emulator's own room-entry gold (an open Maw Bank), already in the game's 120
            e.s.phase = 7

        e.native.run_debug_set_player = set_player
        e.native.run_reanchor_fight = lambda h, rec, buf: (1, "x")
        e.native.run_reanchor_fight_hidden = lambda h, rec, buf: (4, "x")
        e.E = SimpleNamespace(rc=SimpleNamespace(PHASE_COMBAT=7))
        e.s.phase, e.s.view, e.s.apply = 1, (lambda: SimpleNamespace(floor=3)), enter_fight
        e._to_map, e._decision, e._auto_combat = (lambda floor=None: True), (lambda: SimpleNamespace(options=[object()])), (lambda: None)
        obs = {"floor": 4, "hp": 50, "max_hp": 80, "gold": 120, "deck": [["A", 0]], "relics": ["PENDULUM"], "potions": ["P"],
               "combat": EmuLockAnchorTests.RECORD}
        self.assertTrue(e.reanchor("fight_start", {"obs": obs}))
        return state, calls, e

    def test_the_walk_path_ends_with_the_games_counter_not_one_more(self):
        relics = [{"id": "PENDULUM", "ints": {"TurnsSeen": 2}}]
        for name, hidden in (("observation", {"relics": relics}), ("hidden", {**EmuLockAnchorTests.GAME, "relics": relics})):
            state, calls, e = self.walk(hidden)
            self.assertEqual(state["counter"], 2, name)  # written once, after the fight was entered (not 3)
            self.assertEqual([("relic_counters" in kw) for kw in calls], [False, True], name)  # the player write carries none
            self.assertEqual(calls[-1], {"relic_counters": [(10, 2)], "gold": 120}, name)
            self.assertEqual(e.relic_counters, 1, name)

    def test_the_walk_path_ends_with_the_games_gold_not_the_room_paid_twice(self):
        # The observation is taken in the fight, after the game's room-entry gold; the walk enters the room on the emulator,
        # which pays its own on top. The anchor writes the game's gold once the emulator is in the fight (2026-10-29: 49
        # fight-start re-anchors failed on `.gold`, every one a Maw Bank run).
        for name, hidden in (("none", None), ("observation", {"relics": []}), ("hidden", dict(EmuLockAnchorTests.GAME))):
            state, calls, e = self.walk(hidden)
            self.assertEqual(state["gold"], 120, name)
            self.assertEqual(calls[-1], {"gold": 120}, name)  # no counters to write: the gold alone
            self.assertIsNone(e.relic_counters, name)

    def test_the_maw_banks_used_flag_is_a_counter_the_reanchors_carry(self):
        refset = _import_refset()
        self.assertEqual(refset.relic_counter_writes([{"id": "MAW_BANK", "bools": {"HasItemBeenBought": True}}]), [("MAW_BANK", 1)])
        self.assertEqual(refset.relic_counter_writes([{"id": "MAW_BANK", "bools": {"HasItemBeenBought": False}}]), [("MAW_BANK", 0)])
        self.assertEqual(refset.relic_counter_writes([{"id": "MAW_BANK"}]), [("MAW_BANK", 0)])  # the default: open

    def test_the_mapping_of_the_games_relic_properties(self):
        refset = _import_refset()
        relics = [{"id": "VENERABLE_TEA_SET", "bools": {"GainEnergyInNextCombat": True}}, {"id": "SILKEN_TRESS", "bools": {"IsUsed": True}},
                  {"id": "BONE_TEA", "ints": {"CombatsLeft": 2}}, {"id": "BURNING_BLOOD"}, {"id": "SILVER_CRUCIBLE", "ints": {"TimesUsed": 3}}]
        self.assertEqual(refset.relic_counter_writes(relics),
                         [("VENERABLE_TEA_SET", 1), ("BONE_TEA", 2), ("SILVER_CRUCIBLE", 0)])  # Silken Tress used: left alone
        self.assertEqual(refset.relic_counter_writes([{"id": "SILKEN_TRESS"}]), [("SILKEN_TRESS", 0)])
        self.assertEqual(refset.relic_counter_writes(None), [])

    def test_the_tables_agree_with_the_emulators_source(self):
        # the enchantment ordinals are Card.cs's enum order
        src = (_REPO / "src/Sts2Emulator/Core/Card.cs").read_text(encoding="utf-8")
        body = _re.search(r"public enum Enchantment\s*\{(.*?)\n\}", src, _re.S).group(1)
        body = _re.sub(r"//.*", "", body)
        names = [_re.sub(r"(?<!^)(?=[A-Z])", "_", n).upper() for n in _re.findall(r"\b([A-Z][A-Za-z]*)\s*(?:=\s*\d+)?\s*,", body)]
        mod = __import__(type(bare_emu(None)).__module__, fromlist=["ENCHANTMENTS"])
        self.assertEqual(tuple(names), mod.ENCHANTMENTS)
        # every relic property the emulator's snapshot compares (RunSnap.RelicProps) is mapped, or is state outside Counter
        refset = _import_refset()
        snap = (_REPO / "tools/emudump/RunSnap.cs").read_text(encoding="utf-8")
        props = set(_re.findall(r'new\("([A-Z_]+)", "(?:ints|bools)", "(\w+)"', snap))
        outside = {("SILVER_CRUCIBLE", "TreasureRoomsEntered"), ("LAVA_LAMP", "TookDamageThisCombat"), ("LAVA_ROCK", "HasTriggered"),
                   ("LIZARD_TAIL", "WasUsed")}
        mapped = {(relic, name) for relic, (name, _) in refset.COUNTER_MAP.items()}
        self.assertEqual(props, mapped | outside)


def _import_refset():
    try:
        from referee import refset  # noqa: PLC0415
    except ImportError:
        import refset  # noqa: PLC0415
    return refset


if __name__ == "__main__":
    unittest.main()
