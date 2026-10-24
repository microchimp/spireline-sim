# Kit Bolas (no lockstep rank or weight; rebuild-lane finding from the per-combat card uid audit; family: card lifecycle / return-to-hand queue)

## Header
- Base: fork `main` at `f1a869e0` or later (branch `cloud/combat-core-1` from the tip you clone; one commit per kit).
- Cloud half: the scenario test (fails before, passes after), the fix, both .NET suites and the Python suite green, the PR to `main` describing each commit. Local half (the landing session on the PC): the corpus check against the stored baseline, the byte proof, EmuDump, the lockstep replay of the seeds the kit names, landing commit by commit.
- Allowed-file footprint: `src/Sts2Emulator/Core/CombatEngine.cs` (`ReturnQueuedCardsToHandBeforeDraw`, `QueueAttackPlayLifecycleEffects`), `src/Sts2Emulator/Core/Effects/CardEffects.cs` (the Thrumming Hatchet arm only), `src/Sts2Emulator/Core/CombatObservation.cs` (the next-hand display); tests in `src/Sts2Emulator.Tests/Cards/Colorless/BolasTests.cs`. Nothing under `Core2/`, no change to the snapshot format.
- Expected corpus footprint: the three uid-audit fights change (expected, an improvement or neutral), every other fight byte-identical; 0 regressions. The landing session counts them with `EmuDump fight --card-uid-check`.
- Run-first counts (record them on your base before any change): PC at `f1a869e0`: Sts2Emulator.Tests 6,774 passed / 0 failed / 7 skipped (6,781), EmuDump.Tests 246, Python no new failing name; Linux cloud at `4a4b15f` (`docs/CLOUD_ENV.md`): 6,864 passed of 6,871 (7 static skips), EmuDump.Tests 281, the four Python modules 88 tests (2 data skips). Your counts may differ by the tests the base added since; the rule is 0 failed and no test changed.
- Stop rules: stop and report if more than a return-queue change is needed, if a pile-order rule (F11) is involved, or if any existing `BolasTests` case needs its expectation changed. A stop is a report, not a failure.

## Symptom
Found by the per-combat card uid audit (`docs/rebuild/CARD_UID.md`, rounds 2 and 3): on 3 corpus fights the old engine's turn-start return of a
played Bolas puts a SECOND Bolas into the hand, a card no pile gave up. The audit calls it "Bolas returning while already drawn back to hand".
There is no screen or diff to quote (the audit only saw duplicate or invented uids); the lockstep view of such a fight would show a hand with one
more card than the game's from the start of the turn after the Bolas play, then every draw and shuffle off by one. Root. Thrumming Hatchet shares
the queue and the same rule, so it has the same defect.

## Expected behaviour (game)
Bolas (`Models.Cards.Bolas`) has no return logic in `OnPlay` (just the attack). The return is `Bolas.BeforeHandDraw`, a hook every card in a
combat pile answers: if the owner is the hook's player AND `CombatManager.History.CardPlaysFinished` holds an entry for THIS card instance
(identity, `e.CardPlay.Card == this`) with `HappenedLastPlayerTurn(owner)`, then, only if the card's `Pile` is null or not the Hand, it is moved with
`CardPileCmd.Add(this, PileType.Hand)`. Consequences: (a) it is one move of one specific card, from whichever pile it is in (draw, discard, exhaust);
(b) if it is already in the hand (drawn back, retained, put back by a card) nothing happens; (c) a card played twice last turn is moved once (the test
is "any play finished last turn", not per play); (d) the card that comes back is the same object, with whatever state it has now; nothing is ever
created. `Models.Cards.ThrummingHatchet.BeforeHandDraw` has the identical body.

## Emulator mechanism and its other uses
- Queue, on play: `src/Sts2Emulator/Core/CombatEngine.cs` `QueueAttackPlayLifecycleEffects` (called from the play body when `isLast`, about lines
  929-935; it adds `card with { FreeThisTurn = false }` to `state.ReturnToHandBeforeDraw` when `card.DefId == CL.Bolas`) and
  `Core/Effects/CardEffects.cs`, the `CL.ThrummingHatchet` arm (about line 240, adds the same note). One note per PLAY of the card, not per card.
- Consume, at turn start: `CombatEngine.ReturnQueuedCardsToHandBeforeDraw` (about line 5611). For each note: `RemoveFirstMatchingCard(DiscardPile, note)`
  (first card with the note's DefId and Upgraded flag, by value), then also removes the first matching card from the DRAW pile, then from the EXHAUST
  pile (so up to three removals for one note), then adds `note.WithUid(uid)` to the hand, where `uid` is the first removed card's uid and, when no pile
  held a match, `state.NewCardUid()`: a card is invented. The list is cleared.
- So it adds a second Bolas when: (1) the card was played twice in a turn (two notes: the first moves it to the hand, the second finds no pile and
  invents one); (2) the card is already in the hand at turn start (retained, or moved back mid-turn and kept) and nothing is removed from any pile;
  (3) the pile copy no longer equals the note by value (upgraded in between). It can also remove a different Bolas (a copy in the draw pile) than the
  played one, and with a copy in the discard pile AND one in the draw pile removes both for one return.
- Other users of `ReturnToHandBeforeDraw`: `Core/CombatObservation.cs` (about lines 84 and 197-202: the hand shown for next turn adds the notes, so the
  observation also shows the invented card), `Core/StateCloning.cs` (line ~223, copies the list), `Core/CombatFactory.cs` (reset),
  `Interop/CombatSnapshot.cs` (`incoming.return_to_hand` written and read back; notes are NOT live cards, the `Core/CardUid.cs` doc says so: a note read
  from a document has uid 0 after `NumberAll`, so a fix that matches by uid must fall back to value matching for a note with `Uid == 0`).
  `CopiesToHandBeforeDraw` (Nightmare) is a different queue of new cards and stays as it is. Do not touch `Core2/`.

## Scenario test
Project `src/Sts2Emulator.Tests`, `Cards/Colorless/BolasTests.cs` (same file as `DamagesTargetAndReturnsBeforeNextDraw`; copy its setup:
`CombatFactory.NewCombat(seed: 0)`, set `Hand`, `DrawPile`, `DiscardPile`, `Energy = 0`, one enemy with `DefId = 16` and a Defend intent,
`CombatEngine.Step(state, 0, new Random(0))`). Number the cards first (`CardUids.NumberAll(state)` in `Core/CardUid.cs`, as `Core2/CardUidTests.cs`
`Numbered` does) and keep `var bolas = state.Hand[0]`.
1. `PlayedTwiceInATurnReturnsOnce` (fails before): play Bolas (it goes to the discard pile); move the same instance back (`DiscardPile.Remove`,
   `Hand.Add`) as a card that returns it mid-turn would; play it again; end the turn (`Step(state, state.Hand.Count, rng)`). Assert exactly one Bolas
   across Hand, DrawPile, DiscardPile, ExhaustPile and that it has the original uid. Before: two (one with a fresh uid).
2. `ARetainedBolasIsNotReturnedAgain` (fails before): play it, put the instance back in the hand with `RetainThisTurn = true`, end the turn; the hand
   at the new turn has one Bolas, same uid, none invented.
3. `ReturnsThePlayedCopyNotAnotherOne` (fails before): a second Bolas (not upgraded, its own uid) in the draw pile; play the first, end the turn: the
   one in the hand has the first's uid and the other is still in the draw pile (before: both removed, one added).
4. The same for Thrumming Hatchet (copy test 1), and keep all of the existing `BolasTests` (a note with `Uid == 0` still returns the card by value).

## Fix sketch
In `ReturnQueuedCardsToHandBeforeDraw`: process notes by card, not by play. Skip a note whose uid (when non-zero) was already handled in this call;
skip it when a card with that uid is in the `Hand`; otherwise find that card by uid in Discard, then Draw (`RemoveFromDrawPileAt`, keep the draw-pile
bookkeeping), then Exhaust, remove ONLY that one card, and add it to the hand (the card taken from the pile, with `FreeThisTurn = false`, rather than
the note copy). Never call `NewCardUid` here: a note whose card is in no pile is dropped. For a note with `Uid == 0` (snapshot-read, hand-built test
states) keep today's value match but still do not invent a card and do not remove from more than one pile. In `QueueAttackPlayLifecycleEffects` and the
Thrumming Hatchet arm, do not queue a second note for a card whose uid already has one. Make `Core/CombatObservation.cs` (the "next hand" display)
apply the same skips, or it will still show the invented card.

## Gate counts
C# `Sts2Emulator.Tests` 6,755 passed / 0 failed (plus the new tests); `tools/emudump.Tests` 246 (its Reanchor and uid tests read this queue); Python:
no new failing name. Corpus (4,298 fights, 0 regressed) and the byte proof are run locally by the landing session: expect the 3 audit fights to change
(`EmuDump fight <in> --card-uid-check` should then report 0 fights needing a fix and no "return a new card" row) and everything else byte-identical.

## Confidence
Mechanism of the second card (invented uid when no pile holds a match; one note per play): high, read from the code against the game's hook. Which of
the three situations the 3 corpus fights are (played twice, retained, already drawn back): medium; the CARD_UID.md wording says "already drawn back to
hand". One `EmuDump fight --card-uid-check` run on those 3 fights before the fix would name it; the fix covers all three.
