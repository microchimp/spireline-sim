# Auto-play: one play pipeline (owner design, backlog C1-F05)

Mechanism owner's design for every card the game plays for the player. Method names are
the game's (decompiled, local only); no game code is quoted.

## The game's pipeline

Two entry points, one body.

- **Chosen play**: `PlayCardAction.ExecuteAction` checks `CanPlay` and the target, calls
  `SpendResources` (energy, then stars: `AfterEnergySpent`, `AfterStarsSpent`), and hands
  `ResourceInfo {EnergySpent = EnergyValue = spent, StarsSpent = StarValue = spent}` to
  `OnPlayWrapper(isAutoPlay: false)`.
- **Auto-play**: `CardCmd.AutoPlay(card, target, type, skipXCapture)`:
  1. returns at once when the combat is over or ending, or the owner is dead (no roll);
  2. an Unplayable card, or a `Hook.ShouldPlay` refusal, is filed away unplayed
     (`MoveToResultPileWithoutPlaying`: removed from combat for a dupe, exhaust for an
     Exhaust card or `ExhaustOnNextPlay`, else discard; a Power goes to the discard);
  3. target: a given target is used; an AnyEnemy card given none rolls
     `Rng.CombatTargets.NextItem(HittableEnemies)` (also over one enemy); AnyAlly rolls over
     the OTHER living players (none solo); no target means filed away unplayed; other
     target types never roll;
  4. X is captured from the current energy, which is not spent (`skipXCapture` when the
     caller spent it, Whispering Earring);
  5. `ResourceInfo {EnergySpent 0, EnergyValue = GetAmountToSpend(), StarsSpent 0}` and
     `OnPlayWrapper(isAutoPlay: true)`.
- **`CardModel.OnPlayWrapper`, the one body**: into the Play pile; the result pile is
  decided BEFORE the play: `GetResultPileTypeForCardPlay` (dupe or Power none, Exhaust or
  `ExhaustOnNextPlay` exhaust, else discard), then `ModifyCardPlayResultPileTypeAndPosition`
  in listener order: Corruption (Skill to exhaust) and Feral (free Attack to hand) ignore
  the pile they are given, Rebound and Nostalgia only redirect the discard. Then the play
  count (`GeneratePlayCount`), and per play `BeforeCardPlayed`, `CardPlayStarted`, `OnPlay`,
  enchantment and affliction `OnPlay`, `CardPlayFinished`, `AfterCardPlayed`; then the card
  goes to the result pile if still in Play.
- **Hooks that tell the two apart** (decompile grep `IsAutoPlay`, `Resources.`): Void Form
  counts only chosen plays; Pael's Eye and Brilliant Scarf ignore auto-plays; Feral, Right
  Hand Hand, Intimidating Helmet and Ivory Tile read `EnergyValue` (the card's cost for an
  auto-play); Black Hole reads `StarsSpent` (0). Every other hook fires for both.
- **Timing**: every caller awaits `CardCmd.AutoPlay` where it stands, so the auto-play is
  finished before the caller's next line:
  - inside the causing card's `OnPlay`: Knife Trap, each Catastrophe pick, Beat Down,
    Uproar, Decisions Decisions, `AutoPlayFromDrawPile` (Havoc, Cascade);
  - inside the draw: Hellraiser's `AfterCardDrawnEarly`, before every `AfterCardDrawn` hook
    for that card (Chains of Binding, Confused, Speedster, ...) and before the next card;
    `CardPileCmd.Draw` stops once the combat is over;
  - inside the discard command: `CardCmd.DiscardAndDraw` (every `CardCmd.Discard`) plays
    the Sly cards after all the discards and after its draw;
  - inside a phase hook, in listener order (powers, relics, potions, orbs, then cards in
    hand, draw, discard, exhaust): AutoPrePlay early Bombardment; AutoPrePlay Mayhem,
    History Course, Imbued; AutoPrePlay late Whispering Earring. AutoPostPlay Stampede,
    I Am Invincible, Howl From Beyond, before `BeforeTurnEnd` and the hand flush.
- **RNG**: CombatTargets for the target roll (Beat Down rolls its own for AnyEnemy cards and
  passes it); Shuffle (the run's stream) for the callers' picks: Catastrophe and Uproar
  `StableShuffle` per pick, Beat Down one `StableShuffle`, Stampede `NextItem` per stack,
  and `ShuffleIfNecessary`. A caller's own pick rolls even after the fight is over
  (Catastrophe, Stampede); the auto-play itself then returns without a roll and the card
  stays where it is.

## Target structure (implemented)

```
PlayCard (chosen)                      AutoPlay(card, AutoPlaySource)  = CardCmd.AutoPlay
  CanPlayFromHand                        combat over -> return, card stays in its pile
  SpendResources                         gate (BeginAutoPlay): Unplayable, ShouldPlay,
                                           target (given, or CombatTargets for AnyEnemy)
                                           -> filed away unplayed
            \                            /
             ResolvePlay(card, PlayRequest)   = OnPlayWrapper
```

- `AutoPlaySource`: the caller's target, Havoc's forced exhaust, `IsDupe`, and for the
  earring the stars it spent (`skipXCapture`) and its own selector (`VakuuCardSelector`).
- `PlayRequest`: `IsAutoPlay`, the resources (`EnergyValue`, `LastStarsSpent`),
  `ForceExhaust`, `IsDupe`. `ResolvePlay` is the only body: chosen plays and every auto-play
  run the same play-count rules, Before/After hooks, counters and result-pile rules, each
  play settling its own count and pile when it is set up.
- `DecideResultPile`: the result pile is decided as the play is set up, before the effect
  (the card's own pile, then Corruption and Feral over it, then Rebound and Nostalgia over a
  discard, each spent there), so an auto-play inside the card cannot move it. Shining
  Strike's own pile is the discard (Rebound redirects it and is spent); it then tops itself
  in its `OnPlay`.
- `SettlePlayCount` (`GeneratePlayCount`): after the result pile and before the first
  effect, the card's replays (Spiral, Glam, Hidden Gem's Replay), then Burst, Echo Form,
  Throwing Axe, Signal Boost, Duplication and One-Two Punch, each spent there; Echo Form
  counts the turn's plays started before this one. A play nested inside the card (Havoc's
  Strike, a Sly card its discard plays, a Strike Hellraiser plays out of its draw) is set up
  later and finds them already spent, so it never takes the outer card's.
- `AutoPlayFromPile(pile, index)`: the card leaves its pile only if the auto-play starts.
- A play nested inside another (a Strike Hellraiser plays out of a Pommel Strike's draw, a
  Shiv out of Knife Trap) saves and restores the outer play's per-play fields (`PlayFrame`).
- `AutoPlaying` (a screen raised inside the play is answered at once, as the emulator cannot
  suspend a play) is set for every auto-play except the earring's, whose selector answers it.
- `AutoPlayQueue` is only the turn-one pre-play list (Imbued), played at the AutoPrePlay
  point; `AutoPlayTargetIndex` is gone (the target travels in `AutoPlaySource`).

## Sweep: every game caller and its emulator site

31 rows: 1 chosen play, 13 model callers of `CardCmd.AutoPlay`, 5 sources of
`CardPileCmd.AutoPlayFromDrawPile`, 12 sources of `CardCmd.Discard`/`DiscardAndDraw` (Sly).
Status: **converted** (goes through `AutoPlay` at the game's moment) and **tested**,
**blocked** (another owner's lock), **unreachable**. 22 converted (all tested), 8 blocked,
1 unreachable.

| Game caller | Emulator site | Before (main) | Now | Test |
|---|---|---|---|---|
| `PlayCardAction` (chosen) | `PlayCard` | own core | `ResolvePlay` | whole suite |
| `HellraiserPower.AfterCardDrawnEarly` | `CardEffects.DrawCards` | queued, after the action | inline, before the AfterCardDrawn hooks; the draw stops when the fight ends | HellraiserPlaysInsideTheDraw..., ...BeforeChainsOfBindingBindsIt |
| `KnifeTrap.OnPlay` | Knife Trap arm | queued + AutoPlayTargetIndex | inline, given target | KnifeTrapPlayedSixth... |
| `Catastrophe.OnPlay` | `AutoPlayRandomFromDrawPile` | all picks, then queued | pick, play, pick | CatastrophePicksAgain..., AfterAKill... |
| `BeatDown.OnPlay` | `AutoPlayAttacksFromDiscard` | queued, AutoPlay's roll | own roll, inline, from where the card is | BeatDownRollsOnce... |
| `Uproar.OnPlay` | `AutoPlayRandomDrawPileAttack` | inline | inline, from the pile | UproarsAttackPlaysInside..., AutoPlayUproarTests |
| `DecisionsDecisions.OnPlay` | `AutoPlayCardRepeatedly` | AutoPlayCore | shared core | existing |
| `Bombardment` (AutoPrePlay early) | `AutoPlayBombardmentsFromExhaust` | AutoPlayCore | shared core | existing |
| `HowlFromBeyond` (AutoPostPlay) | `AutoPlayHowlsFromExhaust` | AutoPlayCore | shared core | existing |
| `StampedePower` (AutoPostPlay) | `AutoPlayStampedeAttacks` | own hand core, stops rolling at a kill | shared core, a roll per stack | StampedesAttackIsDoubled..., StampedeRolls... |
| `HistoryCourse` (AutoPrePlay) | `AutoPlayHistoryCourseDupe` | queued, dupe filed to discard | dupe, result pile none (Corruption/Feral still apply), last finished play incl. auto-plays | HistoryCourses..., CorruptionExhaustsAHistoryCourseDupe... |
| `Imbued` (AutoPrePlay, turn 1) | `QueueCombatStartAutoPlays` + drain | pre-play list | unchanged moment, shared core | EnchantmentCombatTests |
| `WhisperingEarring` (AutoPrePlay late) | `RunWhisperingEarringAutoPlay` | chosen play | `SpendResources`, then `AutoPlay` with its selector | ItsPlaysAreAutoPlaysThatPaid, ItsSelectorAnswers... |
| `AutoPlayFromDrawPile`: Havoc, Cascade, Mayhem, Distilled Chaos | `AutoPlayFromDrawPile` | AutoPlayCore | shared core | AnAutoPlayedAttackPaysRage, Duplication..., Corruption..., VoidForm... |
| `AutoPlayFromDrawPile`: I Am Invincible | `AutoPlayIAmInvincible` | after the hand flush | AutoPostPlay, after Stampede | IAmInvincibleBlocksBeforeOrichalcum |
| `CardCmd.DiscardAndDraw`: Calculated Gamble | `DiscardMovedCards(drawAfter)` | queued | Sly after the draw, inside the card | ASlyCardPlaysAfterTheDraw... |
| `CardCmd.Discard`: Storm of Steel, Shadow Step | `DiscardMovedCards` | queued | inside the card | ASlyCardPlaysInsideTheCard... |
| `CardCmd.Discard`: Scrape | Scrape arm | straight to the pile, no hooks, reversed | the discard command, in draw order | ItDiscardsThroughTheDiscardCommand |
| Acrobatics, Dagger Throw, Hidden Daggers, Prepared, Survivor, Tools of the Trade, Gambler's Brew, Gambling Chip | selection answers | per pick | **blocked**: the screen is answered after the card's own play ended, one pick at a time (selection lock, fix/turn-start-choices-2) | - |
| `AutoSlay CombatRoomHandler` | none | game's test bot | **unreachable** | - |

## Open gaps (owned here unless named)

- Replays: a play count above one runs the effect again, not the whole
  `BeforeCardPlayed`..`AfterCardPlayed` per play (Rage, Juggling, Velvet Choker and the
  counters see one play). Same body for chosen and auto-plays; next unit of this owner.
- Selection lock: Sly discards made by a screen (8 rows above); a Decisions Decisions that
  is itself auto-played answers its screen at once and plays no Skill; screens inside
  Havoc/Mayhem-played cards take the first card.
- Draw at a fight already over: the game's `Draw` returns, the emulator draws (only a fight
  that ends inside the draw stops it). Asking at entry too improves 5 corpus fights (Pommel
  Strike, Pillage, Flash of Steel killing then drawing) but loses 30, because Inferno's burn
  runs before the hand draw here and after it in the game (`AfterPlayerTurnStart`). Turn-start
  owner.
- Deaths inside a card (Uproar's own kill and Gremlin Horn's draw before Uproar picks): death
  pipeline owner.
- A given target that has died (Knife Trap's later Shivs, Beat Down's roll): the card-target
  pin (ledger, card target).
- Hellraiser's cap of 9 auto-plays a turn against enemies with infinite HP: not modelled.

## Edge list (rule check)

- Combat ends mid-way: the auto-play returns before the gate and the card stays put; the
  caller's own picks still roll (Catastrophe, Stampede); a draw stops.
- Empty piles: Catastrophe and Uproar pick nothing; `AutoPlayFromDrawPile` shuffles first.
- Several instances: Hellraiser is per drawn Strike; Stampede per stack; Knife Trap per Shiv.
- Nested play: frame saved and restored; deaths answered per play, own corpses only.
- Reset/Clone/snapshot: no new state; `AutoPlayTargetIndex` removed from state, Clone,
  snapshot and the at-rest invariant.
- RNG: only the streams listed above.
