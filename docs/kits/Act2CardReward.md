# Kit Act2CardReward: the Act-2 first-reward `.deck` cluster is a Thieving Hopper's card the emulator never claimed

Written fresh on `w5/h2-hopper` (fork main `a9c7da95`). It replaces the text on the retired branch `w4/b14-act2rewards`, whose reading was wrong: its kit said the game keeps the stolen card in the deck through the fight (commit cdfb5667), and its fix put the card back in the deck at the kill (commit 6605a58b). Neither is the game's rule. That branch is never merged.

## Symptom
Lockstep records of type `reward` or `card_reward`, floor 19 and later, with one `.deck` entry: the referee deck holds a card (for example FIGHT_ME) that the emulator deck lacks. Reading the diff as [referee, emulator]: the lists are referee first, emulator second (`structural_diff(ref, emu)` in `referee/lockstep.py`; the hash check on the VK63UAFP46 record confirms it: the first hash is the 20-card deck, the second the 19-card one). The cluster was about 180 records, weight about 1,900. Run VK63UAFP46, decision 100, floor 19 is the sample.

## What it really is
Not a game rule and not a C# error. The emulator's run engine books a Thieving Hopper fight right: the steal removes the card from `RunState.Deck` when the combat ends, and the kill adds a `special_card` row to the reward screen. The reward-state list (list 6) reports it as the fifth count (gold, potion, relic, card items, special rows). Two Python layers read only the first four, so the row was never claimed:
- the lockstep (`_flush_rewards`): the emulator stands at the referee's card-reward question with the gold taken and the row not, deck 19 against the game's 20; the screen is then left with `reward.proceed` and the card is lost; the next `room_entry` re-anchor copies the game's deck back and hides the loss (no record is written there);
- the run layer (`rl/run_train/engine.py`, `_rewards`): training and the nightly's emulator arm lose a Hopper's card for good (the screen is left by the auto skip).

## The game's rule (by class and method)
- `SwipePower.Steal` removes the card's deck version from the deck (`CardPileCmd.RemoveFromDeck`): the deck is one card short from the steal on.
- `SwipePower.BeforeDeath` registers the card with the run (`IRunState.AddCard`; the deck is untouched) and hangs a new `SpecialCardReward` on the room (`CombatRoom.AddExtraReward`).
- `SpecialCardReward.OnSelect` appends the card to the deck (`CardPileCmd.Add`, at the bottom of the pile); `OnSkipped` only writes history.
- The set is ordered by each reward's `RewardsSetIndex`: gold, potion, relic, special card, card.
- The referee's driver (`TakeRewards` in `referee/driver/RunPlay.cs`) takes gold and every `SpecialCardReward` without asking and asks only about card, potion and relic rows.
The record of VK63UAFP46 (decisions 58-62) agrees: deck 20 before the fight, 19 after the kill, 19 on the screen with gold, `special_card` and card, 20 once the special row is claimed (the returned card last), 21 after the pick. The fixture `tests/python/fixtures/hopper_returned_card.json` is that cut.

## The fix (three commits on this branch)
- H2a (C#): `RunEngine.DebugResolveCombat` (behind `Sts2Run_DebugResolveCombat`, which passes no stolen lists) kept nothing: it replaced the suspended fight's own `StolenCards` and `StolenBackCards` with empty lists. A list not given now leaves the fight's list standing. Test `StolenLootRewardTests.AResolvedFightKeepsTheStealItHolds`. No mask change.
- H2b (run layer): `_rewards` reads five counts, claims gold, then the special rows, then opens the card item, all unasked. Test `HopperReturnedCardTests` in `tests/python/test_run_train_engine.py`.
- H2c (lockstep): `_flush_rewards` reads five counts and steps the special rows at the referee's card question, in the game's set order, so a potion or relic question still sees the deck without the card. Tests in `MappingTests` (`test_lockstep.py`). The comparer (`Lockstep._equal`, `project`) is untouched.

## What the live replay must show (VK63UAFP46, under QUIET.lock and PAUSE)
Target floor 19, not a decision number (the replay drifts):
- the game side unchanged: deck 19, 19, 20, 21 at decisions 59-62;
- at the card-reward question the emulator's deck is 20 with the returned card last, `ref_hash` equals `emu_hash`, and no divergence record is written;
- after the pick the emulator's deck order equals the game's (the returned card, then the pick);
- no `room_entry` re-anchor at floor 20, and `deck_order` reads "same" at the next fight start;
- the nightly: the roughly 180 records (weight about 1,900) of this cluster drop.

## Not in this kit (follow-ups)
- The Lantern Key row is the same kind of `SpecialCardReward`; `_flush_rewards` and `_rewards` now claim it too, which the game's driver does, but no run has been replayed for it.
- The driver silently skips every other reward type (`RunPlay.cs`, the branch after card, potion and relic): check what the emulator side does for each.
- `_set_player` in `referee/lockstep.py` zeroes every deck card's enchantment at each `room_entry` re-anchor.
- The 20 in-fight Hopper `.deck` records (which card is stolen) are a separate cause.
