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
     Exhaust card or `ExhaustOnNextPlay`, else discard);
  3. target: a given target is used; an AnyEnemy card given none rolls
     `Rng.CombatTargets.NextItem(HittableEnemies)` (also over one enemy); AnyAlly rolls over
     the OTHER living players (none solo); no target means filed away unplayed; other
     target types never roll;
  4. X is captured from the current energy, which is not spent (`skipXCapture` when the
     caller spent it, Whispering Earring);
  5. `ResourceInfo {EnergySpent 0, EnergyValue = the cost it would spend, StarsSpent 0}`
     and `OnPlayWrapper(isAutoPlay: true)`.
- **`CardModel.OnPlayWrapper`, the one body**: into the Play pile; the result pile is
  decided BEFORE the play (`GetResultPileTypeForCardPlay`: dupe or Power none, Exhaust or
  `ExhaustOnNextPlay` exhaust, else discard; then `ModifyCardPlayResultPileTypeAndPosition`:
  Corruption, Feral, Nostalgia, Rebound, and its `AfterModifying...`); the play count
  (`GeneratePlayCount`: replay and enchantment, `ModifyCardPlayCount`: Burst, Echo Form,
  Duplication, One-Two Punch, Signal Boost, Throwing Axe); per play: `BeforeCardPlayed`,
  `CardPlayStarted`, `OnPlay`, enchantment and affliction `OnPlay`, `CardPlayFinished`,
  `AfterCardPlayed`; then the card goes to the result pile if still in Play.
- **Hooks that tell the two apart** (decompile grep `IsAutoPlay`, `Resources.`): Void Form
  counts only chosen plays; Pael's Eye and Brilliant Scarf ignore auto-plays; Feral,
  Right Hand Hand, Intimidating Helmet and Ivory Tile read `EnergyValue` (the card's cost
  for an auto-play); Black Hole reads `StarsSpent` (0). Every other hook fires for both.
- **Timing**: every caller awaits `CardCmd.AutoPlay` where it stands, so the auto-play is
  finished before the caller's next line: inside the causing card's `OnPlay` (Knife Trap,
  each Catastrophe pick, Beat Down, Uproar, Decisions Decisions, `AutoPlayFromDrawPile`),
  inside the draw (Hellraiser's `AfterCardDrawnEarly`, before every `AfterCardDrawn` hook
  and before the next card is drawn), inside the discard command (Sly, after every discard
  and the draw), inside a phase hook (AutoPrePlay: Bombardment early; Imbued, Mayhem,
  History Course; Whispering Earring late. AutoPostPlay: Stampede, I Am Invincible, Howl
  From Beyond, in listener order: powers, relics, then cards hand, draw, discard, exhaust).
- **RNG**: CombatTargets for the target roll (Beat Down rolls its own for AnyEnemy cards and
  passes it); Shuffle for the callers' picks (Catastrophe and Uproar `StableShuffle` per
  pick, Beat Down one `StableShuffle`, Stampede `NextItem` per stack) and for
  `ShuffleIfNecessary`. A caller's own pick rolls even after the fight is over (Catastrophe,
  Stampede); the auto-play itself then returns without a roll.

## The emulator before this branch (main 646e83d)

Three play cores: `PlayCard` (chosen, and Whispering Earring), `AutoPlayCore` (queue drain,
draw-pile command, Uproar, Decisions, Bombardment, Howl) and `AutoPlayCardFromHandCore`
(Stampede). The auto cores skip most of the body: the play count, Rage/Juggling/Calamity
(AutoPlayCore), Free Attack/Skill/Power, Danse Macabre, Spirit of Ash, Sealed Throne,
Veilpiercer, Serpent Form, Oblivion, Haunt, Devour Life, Right Hand Hand, Make It So, Music
Box, Corruption, Feral, Particle Wall, Razor Tooth, enchantment `OnPlay`, the after-each-play
relics. `AutoPlayQueue` defers six callers to after the whole action.

## Sweep: every game caller and its emulator site

| Game caller | Emulator site before | Before | Target |
|---|---|---|---|
| `PlayCardAction` (chosen) | `CombatEngine.PlayCard` | own core | front + shared core |
| `HellraiserPower.AfterCardDrawnEarly` | `CardEffects.DrawCards` -> AutoPlayQueue | deferred past the draw and the action | inline in the draw, before the AfterCardDrawn hooks |
| `KnifeTrap.OnPlay` | AutoPlayQueue + AutoPlayTargetIndex | deferred | inline, target given |
| `Catastrophe.OnPlay` | `AutoPlayRandomFromDrawPile` -> queue | all picks first, deferred | pick, play, pick, play |
| `BeatDown.OnPlay` | `AutoPlayAttacksFromDiscard` -> queue | deferred, AutoPlay's own roll | own CombatTargets roll, then play, inline |
| `CardCmd.DiscardAndDraw` (Sly) | `DiscardMovedCards` -> queue | deferred | inline after the discards |
| `HistoryCourse.AfterAutoPrePlayPhaseEntered` | turn start -> queue | dupe filed to discard, deferred | dupe, leaves combat, at the hook |
| `Imbued.AfterAutoPrePlayPhaseEntered` | `CombatFactory` -> queue, drained pre-play | right moment | unchanged moment, shared core |
| `Uproar.OnPlay` | `AutoPlayCard` | inline, AutoPlayCore | shared core |
| `DecisionsDecisions.OnPlay` | `AutoPlayCardRepeatedly` | inline, AutoPlayCore | shared core |
| `Bombardment` (exhaust, AutoPrePlay early) | `AutoPlayBombardmentsFromExhaust` | AutoPlayCore | shared core |
| `HowlFromBeyond` (exhaust, AutoPostPlay) | `AutoPlayHowlsFromExhaust` | AutoPlayCore | shared core |
| `StampedePower.AfterAutoPostPlayPhaseEntered` | `AutoPlayStampedeAttacks` | own hand core | shared core |
| `WhisperingEarring` (AutoPrePlay late) | `RunWhisperingEarringAutoPlay` -> `PlayCard` | chosen play | spend, then auto-play (isAutoPlay) |
| `CardPileCmd.AutoPlayFromDrawPile`: Havoc, Cascade, Mayhem, Distilled Chaos | `AutoPlayFromDrawPile` | AutoPlayCore | shared core |
| `IAmInvincible.AfterAutoPostPlayPhaseEntered` | after the hand flush | wrong phase | AutoPostPlay, after Stampede, before Howl |
| `AutoSlay CombatRoomHandler` | none | game's test bot | unreachable |

## Target structure

```
PlayCard (chosen)                       AutoPlay(card, target?, source)
  CanPlayFromHand                         fight over -> return
  SpendResources                          gate: Unplayable, ShouldPlay, target -> file away
  out of hand                             out of its pile
            \                             /
             ResolvePlay(card, PlayRequest)   = OnPlayWrapper
```

`PlayRequest` carries the target, `IsAutoPlay`, `EnergyValue`, `ForceExhaust` and `IsDupe`.
Every caller calls `AutoPlay` where the game awaits it; `AutoPlayQueue` is kept only as the
turn-1 pre-play list (Imbued, and states built by hand), drained at the AutoPrePlay point.
A play nested inside another (a Strike Hellraiser plays out of a Pommel Strike's draw) saves
and restores the outer play's per-play fields (`PlayFrame`).

## Edge list (rule check)

- Combat ends mid-way: the auto-play returns before the gate; the caller's own picks still
  roll (Catastrophe, Stampede); the draw stops (`IsOverOrEnding`).
- Empty piles: Catastrophe and Uproar pick nothing; `AutoPlayFromDrawPile` shuffles first.
- Several instances: Hellraiser is per drawn Strike; Stampede per stack; Knife Trap per Shiv.
- Nested play: frame saved and restored; deaths answered per play, own corpses only.
- Selections inside an auto-play: still answered immediately (`AutoPlaying`); the screen the
  game asks is the selection owner's (fix/turn-start-choices-2).
- RNG: only the streams listed above; counts checked per stream on the corpus.
</content>
</invoke>
