# Kit HeldPlayedCard (no lockstep rank or weight; rebuild-lane finding; family: combat state hygiene / snapshot round trip)

## Header
- Base: fork `main` at `f1a869e0` or later (branch `cloud/combat-core-1` from the tip you clone; one commit per kit).
- Cloud half: the scenario test (fails before, passes after), the fix, both .NET suites and the Python suite green, the PR to `main` describing each commit. Local half (the landing session on the PC): the corpus check against the stored baseline, the byte proof, EmuDump, the lockstep replay of the seeds the kit names, landing commit by commit.
- Allowed-file footprint: `src/Sts2Emulator/Core/CombatEngine.cs` (`ReleaseHeldPlayedCard` made idempotent), `src/Sts2Emulator/Support/Invariants/StateInvariants.cs` (held means in no pile); new `src/Sts2Emulator.Tests/HeldPlayedCardTests.cs`. Do not change `Interop/CombatSnapshot.cs` or anything under `Core2/`.
- Expected corpus footprint: 0 fights change expected (the state the fix acts on is not reached by replayed recordings); byte proof identical. The route that leaves the slot set is found locally with the debug probe the kit describes.
- Run-first counts (record them on your base before any change): PC at `f1a869e0`: Sts2Emulator.Tests 6,774 passed / 0 failed / 7 skipped (6,781), EmuDump.Tests 246, Python no new failing name; Linux cloud at `4a4b15f` (`docs/CLOUD_ENV.md`): 6,864 passed of 6,871 (7 static skips), EmuDump.Tests 281, the four Python modules 88 tests (2 data skips). Your counts may differ by the tests the base added since; the rule is 0 failed and no test changed.
- Stop rules: stop and report if test 1 (the deterministic broken-state test) does not fail before the change, or if the only fix you can find restores the slot from a snapshot. A stop is a report, not a failure.

## Symptom
Found by the rebuild lane (HANDOFF section 0, "For Lane B, second finding"), not by a lockstep record, so there is no screen, run seed or
diff to quote. In a search that loads a document taken in the middle of a selection screen (Hidden Daggers is the card that showed it),
answers the screen(s) and steps once more, the played card ends up in the discard pile twice: one extra Hidden Daggers. The extra card
appears on the step AFTER the answers, not on the answering step. Not reproduced yet by the rebuild lane or by this writer. Root, not a
consequence: the emulator's own state holds a card in two places (a pile and the `HeldPlayedCard` slot) and the next step files the slot
again. Rare: it needs the slot to be set while its card is also filed.

## Expected behaviour (game)
There is no such slot in the game. `PlayCardAction.ExecuteAction` runs `CardModel.OnPlay` while the card sits in the Play pile; the card
moves from the Play pile to its result pile (Discard, or Exhaust) once, after `OnPlay` has finished, through `CardPileCmd.Add`. A card is
always in exactly one pile, and `CombatState.IterateHookListeners` visits the Play pile as a pile of its own. So one Hidden Daggers played
(discard 2 chosen cards, then 2 Shivs into the hand) leaves exactly one Hidden Daggers in the discard pile, however many screens it asked
and however the state was copied or saved in between.

## Emulator mechanism and its other uses
The emulator has no Play pile. `CombatState.HeldPlayedCard` (+ `HeldPlayedCardExhausts`, `src/Sts2Emulator/Core/CombatState.cs`) stands in for it:
- Set: in the card-placement method of `Core/CombatEngine.cs` that `ResolvePlay` calls (it ends with `ResetPlayedCardGrowth(state); return restsOn;`,
  about lines 1170-1300). The card is first put in its result pile as usual; then, if `PendingSelection` is open and `AutoPlaying` is false and
  exactly one card was added to the exhaust pile (or the discard pile), the card is taken back out of that pile into the slot. This happens ONLY
  when the screen is left standing after `OnPlay`, i.e. a step that cannot be replayed: a plain `System.Random` (not `CountingRandom`) or a
  combat start. A replayable step (`CountingRandom`, what `Support/Fight.cs` uses) stops inside `OnPlay` (`HookChoicePending`) and never reaches
  this code with a screen open, so the slot stays null there.
- Cleared: only by `ReleaseHeldPlayedCard` (`CombatEngine.cs`, about line 495): a no-op while `PendingSelection` is open; otherwise it clears
  the slot AND adds the card to the exhaust or discard pile, with no check that the card is already filed. Called from the tail of `StepCore`
  (about line 422, once per step, after `AutoAnswerLoneSelections`) and from `AnswerSelectionsAsVakuu` (about line 6518).
- Copied: `Core/StateCloning.cs` (Clone/CopyTo, about line 233) copies the reference and the flag; `Core/CardUid.cs` (`LiveCards`, `Rewrite`,
  lines ~99 and ~110) lists and renumbers it.
- Written: `Interop/CombatSnapshot.cs` `Write` (about lines 1645-1662) appends the held card to the discard or exhaust pile of the DOCUMENT ("a
  document has no play pile"). The reader (`Read`, which ends with `ReadPendingSelection`, `ReadStoppedStep`, `CardUids.NumberAll`) builds a fresh
  `CombatState`, so a loaded state has `HeldPlayedCard == null` and the card in its pile under a NEW uid. By design: the field is never restored.
- Read defensively by the rebuild lane's `Core2/OldState.cs` (`PlayCard`, comment "the field can outlive its play") and `Core2/NotModelled.cs`
  (`HeldCardFiled`): they treat a held card whose uid is in a pile as stale. Do not edit `Core2/`; keep its test
  (`Core2/ListenerOrderTests.cs` `AStaleHeldPlayedCardIsNotASecondPlayPileVisit`) passing.
- Invariant today: `Support/Invariants/StateInvariants.cs` line ~111 and `tools/emudump.Tests/LifecycleTests.cs` line ~1265 flag "held with no
  selection open" at rest. Nothing flags "held AND filed".

So the lifetime rule is "slot set <=> screen open", enforced only by the single `Release` at the end of a step. Any route that closes the
screen, or files the card (a replayed play, a document round trip that refiles it), without passing through `Release`, or that carries the slot
into a state where the card is already filed, leaves a stale slot; the next step's `Release` then files the card a second time.

Which route does it is NOT shown by the code reading. Candidates, in order of suspicion: (1) a state that was written to a document and read back
while a copy of the pre-write state (slot set) keeps being stepped, so the two meet through `CopyTo`/`Clone` (the loaded copy has the card filed
under a new uid, the other copy still holds the slot); (2) a stopped-step replay (`CombatEngine.Step` with `Replay`) whose `Before` was taken while
a slot was set; (3) `ScreenHiddenState.Whole` put back onto a state. The probe below settles it.

## Scenario test
Project `src/Sts2Emulator.Tests`, new file `HeldPlayedCardTests.cs`. Copy the setup of `PlayedCardPlacementTests.cs` (plain `new Random(0)`
steps leave the screen standing, which is the only way to get the slot set) and the card/hand from `Core2/CardUidTests.cs` (`HiddenDaggersAsked`:
hand Hidden Daggers, Strike, Defend, Neutralize, energy 1, one enemy; its `Numbered` helper gives every card a uid).
1. `AFiledCardIsNotFiledTwice` (must fail before the fix; this is the deterministic form). Build the combat, `CombatEngine.Step(state, 0, new Random(0))`
   (play Hidden Daggers): assert `PendingSelection` is open and `HeldPlayedCard` is the Daggers. Now put the state in the broken shape the rebuild
   lane saw: `state.DiscardPile.Add(state.HeldPlayedCard!.Value)` and `state.PendingSelection = null` (the screen closed and the card filed, the
   slot left set). Step once (end turn: action `state.Hand.Count`, plain `Random`). Assert exactly one Hidden Daggers across Hand, DrawPile,
   DiscardPile, ExhaustPile, and `HeldPlayedCard == null`. Before the fix the step's tail `Release` files it again: two. Add a second test with the
   discard copy `WithUid(state.NewCardUid())` (what a snapshot read gives) expecting the same, which a uid-only staleness check cannot pass.
2. `SnapshotLoadedSelectionFilesTheCardOnce` (the rebuild lane's suggested form; it may already pass, it is the regression guard). Play Hidden Daggers
   with the plain Random (screen standing), `var read = CombatSnapshot.Read(CombatSnapshot.Write(state), seed: 1, out _)`; assert `read.HeldPlayedCard`
   is null and the Daggers is in `read.DiscardPile`; answer both screens (`Step(read, 0, rng)` twice), step once more; assert one Hidden Daggers in
   total, two Shivs in the hand, `HeldPlayedCard == null`. Repeat with the play done on a `CountingRandom` (a stopped step: the document carries
   `stopped_step`, as `CardUidTests.ASelectionTailReadFromASnapshotGetsUidsOnArrivalToo` does through `Fight.Play()`).
3. Assert the new invariant (below) after every step in 1 and 2.

## Fix sketch
Probe first (landing session, which can run the game and the lab): in a debug build, after every `CombatEngine.Step` return, throw when
`HeldPlayedCard != null && (PendingSelection == null || the card's uid is in any pile)` and print the stack and the step's action; run the three
document-loaded lab searches that showed the extra Daggers. The throw names the route (candidates 1-3 above) and decides where the root fix goes.
Fix that holds whatever the route:
- `ReleaseHeldPlayedCard`: when the screen is closed, clear the slot, and file the card only if it is in no pile (compare by `Uid` when non-zero;
  for the uid-renumbered case compare the card with the last card of its result pile and treat an equal-valued last card as filed, as
  `OldState.PlayCard` does). Never add a card that a pile already has.
- Where the route is found: clear the slot at that site (the one place that closes a screen or copies a state across a document).
- Add to `StateInvariants.cs` the second half of the rule: a held card is in no pile, and `PendingSelection` is open.
No change to `CombatSnapshot` (the writer filing the card is right; do not start restoring the slot). No change under `Core2/`.

## Gate counts
C# `Sts2Emulator.Tests` 6,755 passed / 0 failed (plus the new tests); `tools/emudump.Tests` 246; Python: no new failing name. Corpus (4,298 fights,
0 regressed) and the byte proof are run locally by the landing session; the byte proof must stay identical (the fix only acts on a state the corpus
never reaches unless the route is a corpus route).

## Confidence
Mechanism of the stale slot and of the double filing: high (read from `Release` and the setters). That test 1 fails before and passes after: high.
That the idempotent `Release` is the whole root fix: low, because the route that leaves the slot set is not found; the probe raises it to high.
