# Card uid (K2b)

2026-10-23, branch `rebuild/k2b-card-uid`. `TIER1_KERNEL.md` section 3, "Card identity first": the game's cards are
objects, and a listener snapshot asks at each card's turn whether that card is still in the combat. The old engine's
`CardInstance` is a value. K2b gives it a per-combat uid that no existing output can see; the kernel's `CardHandle`
will be that uid.

## What the uid is

- `CardInstance` carries a private `CardUidSlot` field, read and set through the internal `Uid` property
  (`CardUids.UidOf`, `CardUids.WithUid`, `CombatState.Fresh`).
- Equality and hashing: a `CardUidSlot` equals every other slot and hashes to 0, so the record's own synthesized
  `Equals`/`GetHashCode` ignore it. Cards with the same values and different uids stay equal everywhere the old engine
  compares them (`IndexOf`, `Remove`, `Contains`, sets).
- Printing and serializers: the property is internal, so `ToString`/`PrintMembers` and every serializer of public
  members leave it out. The snapshot writer (`Interop/CombatSnapshot.cs`) and the EmuDump dumps write named fields and
  do not name it.
- State walk: `[IdentityOnly]` on the slot and on `CombatState.LastCardUid`; `tools/emudump/StateWalk.cs` skips such
  fields, so the full-state hash and the lifecycle field list are unchanged.
- Counter: `CombatState.LastCardUid`, the last uid handed out. `CopyTo`/`Clone` copy it, so a clone's cards keep their
  uids and its new cards continue the sequence.

## When a card gets one

- Combat start: `CombatFactory.Reset` clones the deck into the draw pile and calls `CardUids.NumberAll`: uids 1..N in
  walk order, counter N.
- Created during the combat: `state.Fresh(card)` (or `Uid = state.NewCardUid()` in an initializer) where it is created:
  generated, copied (the game's `CreateClone`), or transformed into a new card.
- A `with` change keeps the uid: moves between piles, upgrades, cost and keyword changes.
- Read from a document: `CombatSnapshot.ReadCombat` numbers the whole combat (`NumberAll`); EmuDump's reanchor
  write-back numbers only the cards the game had and the emulator did not (`NumberNew`).

Live slots, in walk order (`CardUids.LivePiles`/`LiveCards`): Hand, Draw, Discard, Exhaust, AutoPlayQueue, then
HeldPlayedCard. Not live: `LastAttackOrSkill*` (memory of a card), `Stolen*` (out of the combat),
`ReturnToHandBeforeDraw` (a note of a card that stays in its pile until it returns), `CopiesToHandBeforeDraw` and a
selection's `AfterSelectionToHand` (templates of cards not yet created; their uid is handed out on arrival).

## Creation-site audit

| File | Sites | What they do |
| --- | --- | --- |
| `Core/CombatFactory.cs` | 1 | `Reset`: deck clone into the draw pile, `NumberAll`. Every `Reset`/`NewCombat` overload ends there. |
| `Core/Effects/CardEffects.cs` | 37 | 35 `new CardInstance` sites (generated cards into hand, discard or draw; Shivs; status cards; cards transformed in place in the hand), fresh each, two Hidden Daggers Shivs fresh one by one (was one value repeated); 2 copies: Anger's clone into the discard, the repeated copies of the first hand card. |
| `Core/CombatEngine.cs` | 9 | 7 `new CardInstance` sites (Juggling's copies into the hand, a selection screen's generated pick, Soul, the cards a screen transforms, Wither); Nightmare's queued clones fresh as they are handed over; Bolas's return keeps the uid of the card taken out of the piles (`RemoveFirstMatchingCard` now returns it), and a return no pile gave up a card for is a new card. |
| `Core/EnemyAI.cs` | 2 | Status cards an enemy puts into the discard or the hand. |
| `Core/Effects/RelicEffects.cs` | 2 | Copies: Music Box's Ethereal copy, Burning Sticks' copy. (Jeweled Mask moves a card: keeps.) |
| `Core/Effects/PotionEffects.cs` | 0 | Only moves (the hand into the draw pile). |
| `Core/Run/*` | 0 | Builds deck cards only; no run code writes a combat pile. Deck cards are renumbered at every combat start. |
| `Core/StateCloning.cs` | 1 | `CopyTo` copies the counter; pile values carry their uids. |
| `Interop/CombatSnapshot.cs` | 1 | `ReadCombat` numbers the read combat. The writer is unchanged. |
| `Interop/NativeExports.cs` | 1 | The add-card-to-hand export: fresh. Deck arrays go through `Reset`; `Sts2_CanEnchant`'s temporary card needs none. |
| `tools/emudump` | 4 | Setup items into a combat pile: fresh; reanchor write-back: `NumberNew`; HookProbe's built piles and inserted card: fresh. (`Fight`, `RefsetSetup` and the Codex walker build the run deck only.) |

## Safety net and invariant

`Core2/CardUidInvariant.cs`: `Check` reports the first live card with no uid, a uid above the counter, or a uid an
earlier card holds; `Normalise` gives each such card a fresh uid (the earlier of two keeps its own), returns how many,
and counts them process-wide (`NormalisedCards`, `ActingCalls`). The kernel calls it when it snapshots a pass.

`EmuDump fight <in> --card-uid-check` normalises after every step and writes `card_uid_fixed: ["<breach>:<card>"]` on a
step where it acted. Over the corpus (4043 replayable fights):

| Round | Fights needing it | Cards renumbered | Cause |
| --- | --- | --- | --- |
| after the `new CardInstance` audit | 133 | 187 | 181 Anger (clone kept the uid), 6 Bolas (return) |
| Anger fresh, Bolas keeps the removed card's uid | 3 | 6 | Bolas returning while already drawn back to hand |
| that return a new card | 0 | 0 | |

The last row also marks an old-engine fidelity question, left as it is: a Bolas drawn back into the hand before its
turn-start return still returns, so the hand gains a second Bolas no pile gave up.
