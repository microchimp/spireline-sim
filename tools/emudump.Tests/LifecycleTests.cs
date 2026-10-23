// The state lifecycle, generated from the state itself. Every field of CombatState and RunState, and of
// the creatures, cards, buffs, orbs, relics and screens they hold, is found by reflection
// (tools/emudump/StateWalk.cs) and checked at each stage that copies, clears, prints or overwrites it:
//
//   clone      StateCloning.Clone / CopyTo copies it and does not share it: the field is set to a new
//              value, the state cloned, the field changed again, and the clone must still hold the first value.
//   reset      CombatFactory.Reset on a reused combat (the native combat env's episodes: NativeCombat
//              keeps one CombatState for every reset) leaves it as a fresh combat has it; RunEngine.Reset
//              likewise for the run (and the engine, and no static field a run changes is left changed).
//   dump       the state EmuDump fight prints per step changes when it changes (CombatSnapshot plus the
//              draw order and stream counters); for the run, RunSnap.Export (EmuDump run).
//   load       CombatSnapshot.Read gives back what CombatSnapshot.Write was given.
//   reanchor   OldAnchor.Anchor, writing the game's recorded state over the emulator's, writes it, or it
//              is hidden state carried from the emulator's previous step on purpose.
//
// A field a stage does not handle is a finding unless the stage's allow-list below names it with the
// reason. Findings already known on main are listed in lifecycle-open.txt (open defects, each waiting
// for an owner): a stage fails on a finding not listed there and on a listed one that is fixed. The allow-lists hold only reasons that are true by design; everything else is reported. An
// allow-list entry for a field that no longer exists, or that the stage now handles, fails too, so
// the lists stay exact. The full ledger (every field x stage) is written to lifecycle-ledger.md next
// to the test assembly. Run: dotnet test tools/emudump.Tests --filter Lifecycle
using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Nodes;
using EmuDump.Engines.Old;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Xunit;
using Xunit.Abstractions;

[Trait("Category", "Lifecycle")]
public class LifecycleTests(ITestOutputHelper output)
{
    // ============================================================================ allow-lists

    const string Idle = "idle whenever a step has returned (CombatSnapshotTests.NotCarried; Transient_fields_are_idle_after_every_step checks it over decks that auto-play cards, open selections and end fights)";
    const string IdleQueue = "the turn-one pre-play list (Imbued cards, queued when the combat is built, drained by ResolvePrePlayPhase): empty whenever a step returns, a fight-ending step included (Transient_fields_are_idle_after_every_step; its decks hold no Imbued card and no turn-one screen, so a combat built and not yet observed, or held at a turn-one screen, is not covered); CombatFactory.Reset does not clear it (open: reset CombatState.AutoPlayQueue in lifecycle-open.txt)";
    const string StepArgument = "the step's own argument, overwritten on entry to every Step before anything reads it (CombatSnapshotTests.NotCarried)";
    const string Captured = "captured afresh at the start of every play, before anything reads it (CombatSnapshotTests.NotCarried)";
    const string Unread = "differential-testing introspection; nothing in play reads it (CombatSnapshotTests.NotCarried)";
    const string NoRng = "the snapshot carries no stream: a load seeds its own (CombatSnapshotTests.Streams); EmuDump's dump prints each stream's counter";
    const string RetainUnread = "CardInstance.Retain is written by DrawRareCards and read nowhere; the snapshot schema has no field for it (CombatSnapshotTests)";
    const string Folded = "a document has no play pile: Write folds the held card into the pile it is bound for (CombatSnapshotTests.NotCarried)";
    const string WithSelection = "written inside the open selection (CombatSnapshot's pending_selection); with no selection open it means nothing";
    const string Derived = "derived from the fields beside it by EnemyAI.UpdateSecondaryIntents when loaded";
    const string LoaderSets = "always true while the player is to act; the loader sets it (CombatSnapshotTests.NotCarried)";
    const string HolderRetain = "the holder as a whole carries Retain (see *.Retain); each of its other fields is checked on its own";
    const string SeedDerived = "derived from RunState.StringSeed: RunState.Clone rebuilds every stream from it at the original's positions (StateCloning.CloneAt)";
    const string RunSeed = "the run's seed, fixed for the run; the replay starts from the game's own seed, and the stream counters are in the snapshot";
    const string Profile = "the run's profile (UnlockState): immutable input, set by RunEngine.Reset and handed to each combat the run builds; a bare, loaded or re-anchored combat has the mature one (every character), and so do the game's recorded fights";

    // Reanchor (Reanchor.cs header): the game's snapshot shows HP, block, energy, powers, the four piles
    // (card id, upgrade, order), the living enemies and every stream; everything else is the emulator's
    // own state after its previous step, carried on purpose. MustWrite below is what it must write.
    const string Setup = "the fight's fixed setup (character, ascension, encounter, room): the same on both sides all fight";
    const string Identity = "the creature's identity: the write-back pairs the game's creatures with the emulator's by it";
    const string AiState = "enemy AI state, carried; the next move is put on the game's by OldAnchor.AnchorMoves where the record has intents";
    const string EnemyHidden = "an enemy's hidden state the game's snapshot does not show, carried";
    const string CardHidden = "a card's hidden modifiers (cost changes, enchantment, bonuses, keywords for the turn), carried with the card: PatchPiles keeps each card's identity";
    const string BuffData = "a power's hidden per-instance Data, carried: PatchBuffs keeps each instance's Data";
    const string HiddenCards = "cards the game's snapshot does not list (queued, held, remembered, pending), carried";
    const string HiddenPlayer = "the player's state the game's snapshot does not show (relics, orbs, belt, gold, stars, Osty, pile knowledge), carried";
    const string Counter = "a per-turn or per-combat counter or flag the game's snapshot does not show, carried (Reanchor.cs header)";
    const string TurnStartHeld = "a turn start stopped at a relic's screen: where it goes on, carried with the open selection it waits on (PendingSelection*)";
    const string StepLocal = "in-step bookkeeping, idle when a step has returned with the fight going on (Transient_fields_are_idle_after_every_step), carried";
    const string IntentRoll = "the last intent pass (IntentParticles), what a resampled fork rolls the current moves again from: never changed once made, so Clone and CopyTo share its contents; no game state or document has it, so a dump, a load and a re-anchor leave it, and a fork whose moves no longer match the pass keeps them (ResampleLeakTests)";
    const string Assumed = "ai_state_assumed (DESIGN.md 1.1): set only on a fork whose resample kept the true hidden AI state (IntentParticles.Resample), carried by Clone and CopyTo; no game state or document has it, so a dump, a load and a re-anchor leave it, and a reset clears it (ResampleLeakTests)";
    const string SharedStop = "a stopped step's record (HookChoiceReplay) is never changed once made, so Clone and CopyTo share it by design: a replay copies Before and RngBefore into the combat, PutBack copies the streams and the whole copy out, and CardInstance is a value type (StoppedStepLifecycleTests.CloneAndCopyToShareTheStoppedStep)";
    const string CheckedStop = "a stopped step loads only when its replay from the document's public start reaches its screen the way the step went (CombatEngine.CheckStoppedStep, run by Read and so by Write), and a stamped record is no real stop, so Write refuses it; real stops round-trip whole in StoppedStepDocumentTests (ALoadedStopWritesBackTheSameDocument, TheRecordComesBackFromTheDocument). The agent's document writes the public part only: no stream, no answering copy's hidden state, the start's unseen cards a multiset, the cards taken and the screens reached as the player saw them; the save document adds the true world (WriteSave; StoppedStepLifecycleTests)";

    /// <summary>
    /// Per stage, the fields it leaves alone on purpose, with the reason. A key ending in * matches every
    /// field under that prefix; a key starting with * every field ending so. A wildcard never excuses a
    /// field the reanchor stage must write (MustWrite).
    /// </summary>
    static readonly Dictionary<string, Dictionary<string, string>> Allowed = new()
    {
        ["clone"] = new()
        {
            ["CombatState.LastIntentRoll*"] = IntentRoll,
            ["RunState.Rng"] = SeedDerived,
            ["RunState.Rng.Seed"] = SeedDerived,
            ["RunState.Rng.StringSeed"] = SeedDerived,
            ["RunState.PlayerRng"] = SeedDerived,
            ["RunState.PlayerRng.Seed"] = SeedDerived,
            ["CombatState.PendingSelection.Replay*"] = SharedStop,
        },
        ["copyto"] = new()
        {
            ["CombatState.LastIntentRoll*"] = IntentRoll,
            ["CombatState.PendingSelection.Replay*"] = SharedStop,
        },
        ["reset"] = new()
        {
            ["CombatState.Unlocks"] = Profile,
            ["CombatState.TargetEnemyIndex"] = StepArgument,
            ["CombatState.StrangleBeforePlay"] = Captured,
            ["CombatState.ExecutedThisEnemyTurn"] = Unread,
        },
        ["dump"] = new()
        {
            ["CombatState.LastIntentRoll*"] = IntentRoll,
            ["CombatState.AiStateAssumed"] = Assumed,
            ["CombatState.Unlocks"] = Profile,
            ["CombatState.TargetEnemyIndex"] = StepArgument,
            ["CombatState.AutoPlaying"] = Idle,
            ["CombatState.ResolvingDefendCard"] = Idle,
            ["CombatState.AutoPlayQueue*"] = IdleQueue,
            ["CombatState.HeldPlayedCardExhausts"] = Folded,
            ["CombatState.StrangleBeforePlay"] = Captured,
            ["CombatState.ExecutedThisEnemyTurn"] = Unread,
            ["CombatState.PlayerTurn"] = LoaderSets,
            ["CombatState.EndTurnAwaitingSelection"] = WithSelection,
            ["*.Retain"] = RetainUnread,
        },
        ["load"] = new()
        {
            ["CombatState.LastIntentRoll*"] = IntentRoll,
            ["CombatState.AiStateAssumed"] = Assumed,
            ["CombatState.Unlocks"] = Profile,
            ["CombatState.ShuffleRng"] = NoRng,
            ["CombatState.TargetRng"] = NoRng,
            ["CombatState.AiRng"] = NoRng,
            ["CombatState.CardSelectionRng"] = NoRng,
            ["CombatState.CardGenerationRng"] = NoRng,
            ["CombatState.EnergyCostRng"] = NoRng,
            ["CombatState.PotionGenerationRng"] = NoRng,
            ["CombatState.OrbGenerationRng"] = NoRng,
            ["CombatState.NicheHpRng"] = NoRng,
            ["CombatState.TargetEnemyIndex"] = StepArgument,
            ["CombatState.AutoPlaying"] = Idle,
            ["CombatState.ResolvingDefendCard"] = Idle,
            ["CombatState.AutoPlayQueue*"] = IdleQueue,
            ["CombatState.HeldPlayedCard*"] = Folded,
            ["CombatState.StrangleBeforePlay"] = Captured,
            ["CombatState.ExecutedThisEnemyTurn"] = Unread,
            ["CombatState.PlayerTurn"] = LoaderSets,
            ["CombatState.EndTurnAwaitingSelection"] = WithSelection,
            ["CombatState.Enemies[].SecondaryIntent*"] = Derived,
            ["CombatState.LastAttackOrSkillThisTurn"] = HolderRetain,
            ["CombatState.LastAttackOrSkillLastTurn"] = HolderRetain,
            ["CombatState.Enemies[].StolenCard"] = HolderRetain,
            ["CombatState.PendingSelection.Replay*"] = CheckedStop,
            ["*.Retain"] = RetainUnread,
        },
        ["reanchor"] = new()
        {
            ["CombatState.LastIntentRoll*"] = IntentRoll,
            ["CombatState.AiStateAssumed"] = Assumed,
            ["CombatState.AscensionLevel"] = Setup,
            ["CombatState.Character"] = Setup,
            ["CombatState.Unlocks"] = Profile,
            ["CombatState.EncounterId"] = Setup,
            ["CombatState.IsEliteCombat"] = Setup,
            ["CombatState.IsEliteRoom"] = Setup,
            ["CombatState.Enemies[].DefId"] = Identity,
            ["CombatState.Enemies[].Slot"] = Identity,
            ["CombatState.Enemies[].CurrentIntent*"] = AiState,
            ["CombatState.Enemies[].SecondaryIntent*"] = AiState,
            ["CombatState.Enemies[].MoveIndex"] = AiState,
            ["CombatState.Enemies[].LastMove"] = AiState,
            ["CombatState.Enemies[].LastMoveRepeats"] = AiState,
            ["CombatState.Enemies[].OnceOnlyMoveUsed"] = AiState,
            ["CombatState.Enemies[].MoveHistory"] = AiState,
            ["CombatState.Enemies[].StartsOnBranch"] = AiState,
            ["CombatState.Enemies[].RollsNextMove"] = AiState,
            ["CombatState.Enemies[].StarterMove"] = AiState,
            ["CombatState.Enemies[].LastBranch"] = AiState,
            ["CombatState.Enemies[].RepeatStreak"] = AiState,
            ["CombatState.Enemies[].BranchCooldown"] = AiState,
            ["CombatState.Enemies[].PoweredHitsThisTurn"] = EnemyHidden,
            ["CombatState.Enemies[].HardenedShellDamageThisTurn"] = EnemyHidden,
            ["CombatState.Enemies[].CurlUpArmed"] = EnemyHidden,
            ["CombatState.Enemies[].NemesisIntangibleOn"] = EnemyHidden,
            ["CombatState.Enemies[].StolenGold"] = EnemyHidden,
            ["CombatState.Enemies[].HeistGold"] = EnemyHidden,
            ["CombatState.Enemies[].StolenCard*"] = EnemyHidden,
            ["CombatState.Enemies[].Escaped"] = EnemyHidden,
            ["CombatState.Enemies[].OffBalance"] = EnemyHidden,
            ["CombatState.Enemies[].LastSpawnedDefId"] = EnemyHidden,
            ["CombatState.Enemies[].Buffs[].Data"] = BuffData,
            ["CombatState.PlayerBuffs[].Data"] = BuffData,
            ["CombatState.Hand[].*"] = CardHidden,
            ["CombatState.DrawPile[].*"] = CardHidden,
            ["CombatState.DiscardPile[].*"] = CardHidden,
            ["CombatState.ExhaustPile[].*"] = CardHidden,
            ["CombatState.ReturnToHandBeforeDraw*"] = HiddenCards,
            ["CombatState.CopiesToHandBeforeDraw*"] = HiddenCards,
            ["CombatState.StolenCards*"] = HiddenCards,
            ["CombatState.StolenBackCards*"] = HiddenCards,
            ["CombatState.AutoPlayQueue*"] = HiddenCards,
            ["CombatState.HeldPlayedCard*"] = HiddenCards,
            ["CombatState.LastAttackOrSkillThisTurn*"] = HiddenCards,
            ["CombatState.LastAttackOrSkillLastTurn*"] = HiddenCards,
            ["CombatState.PendingSelection*"] = HiddenCards,
            ["CombatState.TurnStartHeldPhase"] = TurnStartHeld,
            ["CombatState.TurnStartHeldRelic"] = TurnStartHeld,
            ["CombatState.StrangleBeforePlay"] = HiddenCards,
            ["CombatState.Relics*"] = HiddenPlayer,
            ["CombatState.Orbs*"] = HiddenPlayer,
            ["CombatState.OrbCapacity"] = HiddenPlayer,
            ["CombatState.BaseOrbSlots"] = HiddenPlayer,
            ["CombatState.Stars"] = HiddenPlayer,
            ["CombatState.OstyHp"] = HiddenPlayer,
            ["CombatState.OstyMaxHp"] = HiddenPlayer,
            ["CombatState.MaxEnergy"] = HiddenPlayer,
            ["CombatState.PotionSlots"] = HiddenPlayer + " (the belt is written only when the record has it: setup fights)",
            ["CombatState.MaxPotionSlots"] = HiddenPlayer + " (the belt is written only when the record has it: setup fights)",
            ["CombatState.PlayerGold"] = HiddenPlayer,
            ["CombatState.PlayerDebuffsAtRoundStart*"] = HiddenPlayer,
            ["CombatState.KnownTopCount"] = HiddenPlayer + " (only clamped to the written pile)",
            ["CombatState.KnownBottomCount"] = HiddenPlayer + " (only clamped to the written pile)",
            ["CombatState.PlayerTurn"] = StepLocal,
            ["CombatState.TargetEnemyIndex"] = StepLocal,
            ["CombatState.AutoPlaying"] = StepLocal,
            ["CombatState.ResolvingDefendCard"] = StepLocal,
            ["CombatState.EndTurnAwaitingSelection"] = StepLocal,
            ["CombatState.ExecutedThisEnemyTurn"] = StepLocal,
            ["CombatState.AfterimageBeforePlay"] = StepLocal,
            ["CombatState.StormBeforePlay"] = StepLocal,
            ["CombatState.SubroutineBeforePlay"] = StepLocal,
            ["CombatState.EndTurnAfterPlay"] = StepLocal,
            ["CombatState.LethalAttackPlay"] = StepLocal,
            ["CombatState.PlayedCardBonusBlock"] = StepLocal,
            ["CombatState.PlayedCardBonusDamage"] = StepLocal,
            ["CombatState.PlayedCardCostBump"] = StepLocal,
            ["CombatState.PlayedCardCostForCombat"] = StepLocal,
            ["CombatState.PlayedCardEnchantGrew"] = StepLocal,
            ["CombatState.PlayedCardEnchantSpent"] = StepLocal,
            ["CombatState.PlayedCardStarsSpent"] = StepLocal,
            ["CombatState.AttackCardsPlayedThisTurn"] = Counter,
            ["CombatState.AttackOrSkillCardsPlayedThisTurn"] = Counter,
            ["CombatState.BattlewornDummyRanOutOfTime"] = Counter,
            ["CombatState.BeltBuckleApplied"] = Counter,
            ["CombatState.BlockGainsThisTurn"] = Counter,
            ["CombatState.BoundCardPlayedThisTurn"] = Counter,
            ["CombatState.BoundCardsThisTurn"] = Counter,
            ["CombatState.BurningSticksUsed"] = Counter,
            ["CombatState.CardPlaysLastTurn"] = Counter,
            ["CombatState.CardPlaysStartedThisTurn"] = Counter,
            ["CombatState.CardPlaySeriesStartedThisTurn"] = Counter,
            ["CombatState.CardPlaysThisTurn"] = Counter,
            ["CombatState.CardsDrawnThisCombat"] = Counter,
            ["CombatState.CardsExhaustedThisTurn"] = Counter,
            ["CombatState.CardsGeneratedThisCombat"] = Counter,
            ["CombatState.CardsPlayedThisCombat"] = Counter,
            ["CombatState.CardsPlayedThisTurn"] = Counter,
            ["CombatState.DoomAppliedThisTurn"] = Counter,
            ["CombatState.EnergySpentThisTurn"] = Counter,
            ["CombatState.EtherealCardPlaysThisCombat"] = Counter,
            ["CombatState.EtherealExhaustCount"] = Counter,
            ["CombatState.EtherealExhaustsThisTurn"] = Counter,
            ["CombatState.ExtraCardRewards"] = Counter,
            ["CombatState.ExtraCardsDrawnThisTurn"] = Counter,
            ["CombatState.EscapedEnemies"] = Counter,
            ["CombatState.DepartedEnemyDefIds"] = Counter,
            ["CombatState.FatGremlinEscaped"] = Counter,
            ["CombatState.FurCoatBites"] = Counter,
            ["CombatState.GamblingChipDiscarded"] = Counter,
            ["CombatState.LightningOrbsChanneledThisCombat"] = Counter,
            ["CombatState.MercGoldWasStolen"] = Counter,
            ["CombatState.MusicBoxUsedThisTurn"] = Counter,
            ["CombatState.OstyAttacksThisTurn"] = Counter,
            ["CombatState.PaelsLegionCard"] = Counter,
            ["CombatState.PaelsLegionCooldown"] = Counter,
            ["CombatState.PenNibArmed"] = Counter,
            ["CombatState.PlayerHitTurn"] = Counter,
            ["CombatState.PlayerHitTurnBefore"] = Counter,
            ["CombatState.PlayerHpLostThisTurn"] = Counter,
            ["CombatState.RainbowRingAttacks"] = Counter,
            ["CombatState.RainbowRingPaidThisTurn"] = Counter,
            ["CombatState.RainbowRingPowers"] = Counter,
            ["CombatState.RainbowRingSkills"] = Counter,
            ["CombatState.RoyaltiesGold"] = Counter,
            ["CombatState.RuptureOwed"] = Counter,
            ["CombatState.ShivsPlayedThisTurn"] = Counter,
            ["CombatState.SkillCardsPlayedThisTurn"] = Counter,
            ["CombatState.SkillPlayedWhileSmoggy"] = Counter,
            ["CombatState.StarsGainedThisTurn"] = Counter,
            ["CombatState.StatusCardsDrawnThisTurn"] = Counter,
            ["CombatState.StolenBackGold"] = Counter,
            ["CombatState.TenderCardsPlayedThisTurn"] = Counter,
            ["CombatState.TookUnblockedDamage"] = Counter,
            ["CombatState.UnblockedDamageHitCount"] = Counter,
            ["CombatState.UnblockedDamageThisTurn"] = Counter,
            ["CombatState.UnsettlingLampCard"] = Counter,
            ["CombatState.UnsettlingLampSpent"] = Counter,
            ["CombatState.VambraceCard"] = Counter,
            ["CombatState.VambraceSpent"] = Counter,
            ["CombatState.VoidFormCardsPlayedThisTurn"] = Counter,
            ["CombatState.WhisperingEarringSpent"] = Counter,
        },
        ["run-reset"] = new(),
        ["run-dump"] = new()
        {
            ["RunState.Unlocks"] = Profile,
            ["RunState.StringSeed"] = RunSeed,
            ["RunState.Rng.Seed"] = RunSeed,
            ["RunState.Rng.StringSeed"] = RunSeed,
            ["RunState.PlayerRng.Seed"] = RunSeed,
            ["RunState.ActiveCombat"] = "the fight: compared step by step as the decision's fight (run_compare2.fight_diff), and its outcome after it",
            ["RunState.ActiveCombatRng"] = "the fight's own stream, alive only during the fight; the run's counters are in the snapshot",
            ["RunState.MapNodes*"] = "the act's map: compared at each act change (run_compare2.compare_map), and the room entered at each map decision",
            ["RunState.MapNodeTypes"] = "the map screen's offer, compared per decision (run_compare2: the offer and the room entered)",
            ["RunState.MapChoices"] = "the map screen's offer, compared per decision (run_compare2: the offer and the room entered)",
            ["RunState.MapOptionCoords"] = "the map screen's offer, compared per decision (run_compare2: the offer and the room entered)",
            ["RunState.Pending*"] = "a screen's pending state, which the save has no field for: what it does shows in the next decisions' offers and snapshots, compared there",
            ["RunState.Reward*"] = "the rewards screen: compared per decision as its offer, and after a fight as outcome_offer",
            ["RunState.Shop*"] = "the shop screen: compared per decision as its offer (ShopRemovalsUsed is in the snapshot)",
            ["RunState.Event*"] = "the event screen: compared per decision as its enabled options and option count (run_compare2.event_options)",
            ["RunState.ResumeEvent*"] = "the event screen: compared per decision as its enabled options and option count (run_compare2.event_options)",
            ["RunState.Neow*"] = "Neow's screen: compared per decision as its offer",
            ["RunState.BundleOffer"] = "a bundle screen: compared per decision as its offer",
            ["RunState.SelectedBundle"] = "a bundle screen: compared per decision as its offer",
            ["RunState.CrystalSphere*"] = "the Crystal Sphere screen, a minigame inside one decision",
            ["RunState.TinkerCardType"] = "a screen's pending state: compared per decision as the screen's offer",
            ["RunState.TransformSelectedDeckIndex"] = "a screen's pending state: compared per decision as the screen's offer",
            ["RunState.RestResultPending"] = "the rest screen: compared per decision as its enabled options",
            ["RunState.RestOptionsTaken"] = "the rest screen: compared per decision as its enabled options",
            ["RunState.AwaitingActStartNode"] = "screen flow: the decision's screen is compared",
            ["RunState.ReturnToRewardScreenAfterCardReward"] = "screen flow: the decision's screen is compared",
            ["RunState.LastPlayerWon"] = "the fight's result, compared as the fight's outcome (run_compare2.fight_outcome)",
            ["RunState.RelicBag._shuffleOrder"] = "the bag's rarity order, one entry per rarity (a stamped second entry is no state a run reaches; the export refuses it)",
            ["RunState.SharedRelicBag._shuffleOrder"] = "the bag's rarity order, one entry per rarity (a stamped second entry is no state a run reaches; the export refuses it)",
            ["RunState.RelicBag"] = "the bag as a whole (its rarity order, above); each of its fields is checked on its own",
            ["RunState.SharedRelicBag"] = "the bag as a whole (its rarity order, above); each of its fields is checked on its own",
            ["RunState.Relics[].Counter"] = "relic counters are exported in the game's units for the relics in RunSnap.RelicProps (RUNBOOK: add a counter there); the starting relic has none",
        },
    };

    /// <summary>What the game's snapshot shows and the write-back must therefore write, whatever the allow-list says.</summary>
    static readonly HashSet<string> MustWrite =
    [
        "CombatState.PlayerHp", "CombatState.PlayerMaxHp", "CombatState.PlayerBlock", "CombatState.Energy", "CombatState.Turn",
        "CombatState.PlayerBuffs", "CombatState.PlayerBuffs[].Id", "CombatState.PlayerBuffs[].Magnitude",
        "CombatState.Hand", "CombatState.Hand[].DefId", "CombatState.Hand[].Upgraded",
        "CombatState.DrawPile", "CombatState.DrawPile[].DefId", "CombatState.DrawPile[].Upgraded",
        "CombatState.DiscardPile", "CombatState.DiscardPile[].DefId", "CombatState.DiscardPile[].Upgraded",
        "CombatState.ExhaustPile", "CombatState.ExhaustPile[].DefId", "CombatState.ExhaustPile[].Upgraded",
        "CombatState.Enemies", "CombatState.Enemies[].Hp", "CombatState.Enemies[].MaxHp", "CombatState.Enemies[].Block",
        "CombatState.Enemies[].Buffs", "CombatState.Enemies[].Buffs[].Id", "CombatState.Enemies[].Buffs[].Magnitude",
        "CombatState.ShuffleRng", "CombatState.TargetRng", "CombatState.AiRng", "CombatState.CardSelectionRng",
        "CombatState.CardGenerationRng", "CombatState.EnergyCostRng", "CombatState.PotionGenerationRng",
        "CombatState.OrbGenerationRng", "CombatState.NicheHpRng",
    ];

    static bool Matches(string key, string path) =>
        key.StartsWith('*') ? path.EndsWith(key[1..], StringComparison.Ordinal)
        : key.EndsWith('*') ? path.StartsWith(key[..^1], StringComparison.Ordinal)
        : key == path;

    static string? AllowedWhy(string stage, string path)
    {
        if (stage == "reanchor" && MustWrite.Contains(path)) return null;
        if (!Allowed.TryGetValue(stage, out var a)) return null;
        if (a.TryGetValue(path, out var exact)) return exact;
        foreach (var (k, why) in a)
        {
            if (k.Contains('*') && Matches(k, path)) return why;
        }

        return null;
    }

    // ============================================================================ inventory

    sealed record Member(string Path, string[] Segs, Type Type);

    /// <summary>Every field reachable from <paramref name="root"/>, through the engine's own holder types (first element of a collection).</summary>
    static List<Member> Inventory(Type root, params Type[] stopAt)
    {
        var list = new List<Member>();
        void Walk(Type t, string prefix, List<string> segs, HashSet<Type> stack)
        {
            foreach (var f in StateWalk.Fields(t))
            {
                string name = StateWalk.Name(f);
                var ft = f.FieldType;
                var path = prefix + "." + name;
                list.Add(new Member(path, [.. segs, name], ft));
                var (holder, coll) = Holder(ft);
                if (holder == null || stack.Contains(holder) || stopAt.Contains(holder)) continue;
                string seg = coll ? name + "[]" : name;
                Walk(holder, prefix + "." + seg, [.. segs, seg], [.. stack, holder]);
            }
        }

        Walk(root, root.Name, [], [root]);
        return list;
    }

    static (Type? Holder, bool Collection) Holder(Type t)
    {
        var u = Nullable.GetUnderlyingType(t) ?? t;
        if (StateWalk.IsEngineType(u)) return (u, false);
        var el = StateWalk.ElementType(u);
        if (el != null)
        {
            var eu = Nullable.GetUnderlyingType(el) ?? el;
            if (StateWalk.IsEngineType(eu)) return (eu, true);
        }

        return (null, false);
    }

    // ============================================================================ stamping

    const BindingFlags Inst = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    static FieldInfo Field(Type t, string name) =>
        StateWalk.Fields(t).First(f => StateWalk.Name(f) == name);

    /// <summary>
    /// A new holder or element of <paramref name="t"/>: a copy of one the built run already has (so ids
    /// are real: a card, a buff, an enemy the engine knows), else a default one with its collections empty.
    /// </summary>
    static object Create(Type t)
    {
        t = Nullable.GetUnderlyingType(t) ?? t;
        if (t == typeof(string)) return "";
        if (t.IsArray) return Array.CreateInstance(t.GetElementType()!, 0);
        if (t == typeof(PendingCardSelection)) return OpenSelection();
        if (t == typeof(HookChoiceReplay)) return StoppedStep();
        if (StateWalk.IsEngineType(t) && Prototype(t) is { } proto) return DeepCopy(proto)!;
        if (t.IsInterface && StateWalk.ElementType(t) is { } el) return Activator.CreateInstance(typeof(List<>).MakeGenericType(el))!;
        if (t.IsValueType && !StateWalk.IsEngineType(t)) return Activator.CreateInstance(t)!;
        var o = t.IsValueType ? Activator.CreateInstance(t)!  // boxed: a record struct's collections are filled in the box
            : t.GetConstructor(Inst, Type.EmptyTypes) != null ? Activator.CreateInstance(t, nonPublic: true)! : RuntimeHelpers.GetUninitializedObject(t);
        foreach (var f in StateWalk.Fields(t))
        {
            if (f.GetValue(o) == null && f.FieldType != typeof(string) && !StateWalk.IsStream(f.FieldType)
                && (f.FieldType.IsArray || StateWalk.ElementType(f.FieldType) != null))
            {
                f.SetValue(o, Create(f.FieldType));
            }
        }

        return o;
    }

    static readonly Dictionary<Type, object?> Prototypes = new();

    /// <summary>The first instance of <paramref name="t"/> in the built run (its combat included), or null.</summary>
    static object? Prototype(Type t)
    {
        lock (Prototypes)
        {
            if (Prototypes.TryGetValue(t, out var hit)) return hit;
            object? found = null;
            var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
            void Look(object? v)
            {
                if (found != null || v == null || v is string || StateWalk.IsStream(v.GetType())) return;
                if (!v.GetType().IsValueType && !seen.Add(v)) return;
                if (v.GetType() == t)
                {
                    found = v;
                    return;
                }

                if (v is IDictionary d)
                {
                    foreach (DictionaryEntry e in d) Look(e.Value);
                }
                else if (v is IEnumerable seq)
                {
                    foreach (var x in seq) Look(x);
                }
                else if (StateWalk.IsEngineType(v.GetType()))
                {
                    foreach (var f in StateWalk.Fields(v.GetType())) Look(f.GetValue(v));
                }
            }

            Look(BuildRunInCombat());
            Prototypes[t] = found;
            return found;
        }
    }

    /// <summary>A copy that shares no collection or holder with the original.</summary>
    static object? DeepCopy(object? v)
    {
        if (v == null || v is string || v.GetType().IsPrimitive || v.GetType().IsEnum) return v;
        var t = v.GetType();
        if (v is CountingRandom cr) return cr.Copy();
        if (v is GameRng gr) return gr.Copy();
        if (v is Array a)
        {
            var copy = Array.CreateInstance(t.GetElementType()!, a.Length);
            for (int i = 0; i < a.Length; i++) copy.SetValue(DeepCopy(a.GetValue(i)), i);
            return copy;
        }

        if (v is IDictionary d)
        {
            var copy = (IDictionary)Activator.CreateInstance(t)!;
            foreach (DictionaryEntry e in d) copy[DeepCopy(e.Key)!] = DeepCopy(e.Value);
            return copy;
        }

        if (v is IList l && t.IsGenericType)
        {
            var copy = (IList)Activator.CreateInstance(t)!;
            foreach (var x in l) copy.Add(DeepCopy(x));
            return copy;
        }

        if (StateWalk.ElementType(t) != null && t.GetMethod("Add") is { } add)
        {
            var copy = Activator.CreateInstance(t)!;
            foreach (var x in (IEnumerable)v) add.Invoke(copy, [DeepCopy(x)]);
            return copy;
        }

        if (!StateWalk.IsEngineType(t)) return v;
        var o = MemberwiseCloneOf(v);
        foreach (var f in StateWalk.Fields(t)) f.SetValue(o, DeepCopy(f.GetValue(o)));
        return o;
    }

    static readonly MethodInfo Memberwise = typeof(object).GetMethod("MemberwiseClone", BindingFlags.Instance | BindingFlags.NonPublic)!;

    static object MemberwiseCloneOf(object v) => Memberwise.Invoke(v, null)!;

    static object? FirstOf(object coll)
    {
        if (coll is IDictionary d)
        {
            foreach (DictionaryEntry e in d) return e.Value;
            return null;
        }

        foreach (var x in (IEnumerable)coll) return x;
        return null;
    }

    /// <summary>The collection with its first element replaced (or added when it has none); a new array when it grows.</summary>
    static object SetFirst(object coll, object? value, Type elementType)
    {
        switch (coll)
        {
            case Array a when a.Length > 0:
                a.SetValue(value, 0);
                return a;
            case Array:
                var grown = Array.CreateInstance(elementType, 1);
                grown.SetValue(value, 0);
                return grown;
            case IList l when l.Count > 0:
                l[0] = value;
                return l;
            case IList l:
                l.Add(value);
                return l;
            case IDictionary d:
                object? key = null;
                foreach (DictionaryEntry e in d)
                {
                    key = e.Key;
                    break;
                }

                key ??= Create(d.GetType().GetGenericArguments()[0]);
                d[key] = value;
                return d;
        }

        throw new NotSupportedException("collection " + coll.GetType());
    }

    /// <summary>A value of <paramref name="t"/> other than <paramref name="cur"/>; collections and holders change in place.</summary>
    static object? NewValue(Type t, object? cur, int salt)
    {
        var u = Nullable.GetUnderlyingType(t);
        if (u != null) return NewValue(u, cur ?? Create(u), salt);
        if (t == typeof(int)) return (int)(cur ?? 0) + salt;
        if (t == typeof(long)) return (long)(cur ?? 0L) + salt;
        if (t == typeof(uint)) return (uint)(cur ?? 0u) + (uint)salt;
        if (t == typeof(short)) return (short)((short)(cur ?? (short)0) + salt);
        if (t == typeof(byte)) return (byte)((byte)(cur ?? (byte)0) + salt);
        if (t == typeof(bool)) return !(bool)(cur ?? false);
        if (t == typeof(UnlockState))
        {
            // Another profile: one more (or one fewer) modelled epoch locked.
            var profile = (UnlockState?)cur ?? UnlockState.All;
            string epoch = UnlockState.Epochs.Modelled.Order(StringComparer.Ordinal).ElementAt(Math.Abs(salt) % UnlockState.Epochs.Modelled.Count);
            return profile.IsEpochRevealed(epoch) ? profile.WithLockedEpochs(epoch) : UnlockState.All;
        }
        if (t == typeof(float)) return (float)(cur ?? 0f) + 0.5f * salt;
        if (t == typeof(double)) return (double)(cur ?? 0d) + 0.5 * salt;
        if (t == typeof(string)) return ((string?)cur ?? "") + "~" + salt;
        if (t.IsEnum)
        {
            var values = Enum.GetValues(t).Cast<object>().Distinct().ToList();
            int at = cur == null ? -1 : values.FindIndex(v => v.Equals(cur));
            var next = values[((at < 0 ? 0 : at) + salt) % values.Count];
            return next.Equals(cur) ? Enum.ToObject(t, Convert.ToInt64(cur) + salt) : next;
        }

        if (t == typeof(CountingRandom)) return cur is CountingRandom c ? new CountingRandom(c.Seed ^ (0x51 * salt), c.CallCount + salt) : new CountingRandom(977 * salt);
        if (t == typeof(GameRng))
        {
            if (cur is not GameRng g) return new GameRng(31 * salt);
            for (int i = 0; i < salt; i++) g.NextInt();  // in place: a shared stream moves for both
            return g;
        }

        if (t == typeof(Random)) return cur is CountingRandom held ? NewValue(typeof(CountingRandom), held, salt) : new Random(salt);
        if (t.IsArray)
        {
            var et = t.GetElementType()!;
            var a = (Array?)cur ?? Array.CreateInstance(et, 0);
            if (a.Length == 0) return SetFirst(a, NewValue(et, null, salt), et);
            a.SetValue(NewValue(et, a.GetValue(0), salt), 0);  // in place
            return a;
        }

        if (StateWalk.ElementType(t) is { } el && cur is not string)
        {
            var coll = cur ?? Create(t);
            bool holderElement = StateWalk.IsEngineType(Nullable.GetUnderlyingType(el) ?? el);
            if (coll is IDictionary { Count: > 0 } full)
            {
                // A dictionary's keys are usually a fixed set (rarities, coordinates): its first value changes.
                object? key = null;
                foreach (DictionaryEntry e in full)
                {
                    key = e.Key;
                    break;
                }

                full[key!] = holderElement ? Create(el) : NewValue(el, full[key!], salt);
                return coll;
            }

            // A list grows by a copy of its first element (an id it holds is one the engine knows), else a new one.
            object? first = coll is IDictionary ? null : FirstOf(coll);
            object? item = holderElement ? Create(el) : first != null ? DeepCopy(first)
                : el == typeof(int) ? AnyCardId  // an empty id list: most hold card ids, and a made-up id is refused
                : NewValue(el, null, salt + 1000);
            if (item is BuffState buff && coll is IEnumerable<BuffState> held)
            {
                // A holder keeps one buff per id: the new one takes an id the list does not have.
                var taken = held.Select(b => b.Id).ToHashSet();
                item = buff with { Id = Enum.GetValues<BuffId>().First(id => !taken.Contains(id) && id != BuffId.Stunned), Magnitude = Math.Max(1, buff.Magnitude) };
            }
            switch (coll)
            {
                case IDictionary d:
                    var kt = t.GetGenericArguments()[0];
                    object? key = NewValue(kt, null, salt + 7000);
                    while (d.Contains(key!)) key = NewValue(kt, key, 1);
                    d[key!] = item;
                    break;
                case IList l:
                    l.Add(item);
                    break;
                default:
                    coll.GetType().GetMethod("Add")!.Invoke(coll, [item]);
                    break;
            }

            return coll;
        }

        if (t.IsValueType && typeof(ITuple).IsAssignableFrom(t))
        {
            var f = t.GetField("Item1")!;
            object boxed = cur ?? Activator.CreateInstance(t)!;
            f.SetValue(boxed, NewValue(f.FieldType, f.GetValue(boxed), salt));
            return boxed;
        }

        if (StateWalk.IsEngineType(t))
        {
            // A holder as a whole (a screen, a bag): a fresh one, every leaf of it set.
            var o = cur ?? Create(t);
            foreach (var f in StateWalk.Fields(t))
            {
                if (!StateWalk.IsEngineType(f.FieldType)) f.SetValue(o, NewValue(f.FieldType, f.GetValue(o), salt));
            }

            return o;
        }

        throw new NotSupportedException("no stamp for " + t);
    }

    /// <summary>Sets the member at <paramref name="segs"/> to a new value (creating holders on the way), in place where it can.</summary>
    static object Stamp(object container, string[] segs, int salt, int i = 0)
    {
        string seg = segs[i];
        bool into = seg.EndsWith("[]");
        var f = Field(container.GetType(), into ? seg[..^2] : seg);
        var cur = f.GetValue(container);
        if (i == segs.Length - 1)
        {
            f.SetValue(container, NewValue(f.FieldType, cur, salt));
            return container;
        }

        if (into)
        {
            var coll = cur ?? Create(f.FieldType);
            var et = StateWalk.ElementType(coll.GetType()) ?? StateWalk.ElementType(f.FieldType)!;
            var item = FirstOf(coll) ?? Create(et);
            coll = SetFirst(coll, Stamp(item, segs, salt, i + 1), et);
            f.SetValue(container, coll);
        }
        else
        {
            var holder = cur ?? Create(f.FieldType);
            f.SetValue(container, Stamp(holder, segs, salt, i + 1));
        }

        return container;
    }

    const string Absent = "<absent>";

    /// <summary>The canonical text of the member at <paramref name="segs"/> (the first element of a collection on the way).</summary>
    static string Read(object? container, string[] segs, int i = 0)
    {
        if (container == null) return Absent;
        string seg = segs[i];
        bool into = seg.EndsWith("[]");
        var f = StateWalk.Fields(container.GetType()).FirstOrDefault(x => StateWalk.Name(x) == (into ? seg[..^2] : seg));
        if (f == null) return Absent;
        var v = f.GetValue(container);
        if (i == segs.Length - 1) return StateWalk.Canon(v);
        if (into) v = v == null ? null : FirstOf(v);
        return Read(v, segs, i + 1);
    }

    // ============================================================================ fixtures

    static RunEngine BuildRunInCombat()
    {
        var e = new RunEngine();
        e.Reset("LIFECYCLE1");
        e.StartCombat(RunConstants.StarterDeckIds, RunConstants.SlitheringStranglerEncounterId,
            [RunConstants.StartingRelicFor(PlayerCharacter.Ironclad)], playerHp: 64, playerMaxHp: 80, potionIds: [13], playerGold: 99);
        // Something in every holder, so a stamp copies a real one (ids the engine and the snapshot know).
        var c = e.State.ActiveCombat!;
        BuffSystem.Apply(c.PlayerBuffs, BuffId.Strength, 2);
        foreach (var en in c.Enemies) BuffSystem.Apply(en.Buffs, BuffId.Weak, 1);
        c.NoteDeparted(c.Enemies[0]);  // an enemy id, not the card id an empty int list is stamped with
        c.Orbs.Add(new OrbState(OrbType.Frost));
        c.DiscardPile.Add(c.DrawPile[^1]);
        c.ExhaustPile.Add(c.DrawPile[^2]);
        c.DrawPile.RemoveRange(c.DrawPile.Count - 2, 2);
        c.KnownBottomCount = 0;
        c.OrbCapacity = Math.Max(c.OrbCapacity, 1);
        return e;
    }

    /// <summary>A screen that is open (the snapshot refuses a selection of kind None), on a card the hand holds.</summary>
    static object AnyCardId => Template.Value.State.ActiveCombat!.Hand[0].DefId;

    static PendingCardSelection OpenSelection() => new()
    {
        Kind = Enum.GetValues<CardSelectionKind>().First(k => k != CardSelectionKind.None),
        Candidates = [0],
        SourceCardDefId = Template.Value.State.ActiveCombat!.Hand[0].DefId,
    };

    /// <summary>
    /// A stopped step's record as a step that stops leaves one (the engine never makes one without its start
    /// and its stream): the template's combat as the start, a play of its first card, no answer yet.
    /// </summary>
    static HookChoiceReplay StoppedStep() => new()
    {
        Before = Template.Value.State.ActiveCombat!.Clone(),
        RngBefore = new CountingRandom(977),
        Action = 0,
        TargetEnemyIndex = -1,
        Answers = new List<StepAnswer>(),
        Hidden = new List<ScreenHiddenState>(),
        Taken = new List<CardInstance>(),
        Screens = new List<StepScreen>(),
        Peeks = new List<bool>(),
    };

    static readonly Lazy<RunEngine> Template = new(BuildRunInCombat);

    // Fixtures are copies of one built run (a fresh build per field costs a quarter second). The copy
    // need not be exact: every check sets the field it tests before it looks at it.
    static RunEngine RunInCombat() => Template.Value.Clone();

    static CombatState Combat() => Template.Value.State.ActiveCombat!.Clone();

    /// <summary>One episode of the native combat env on a combat it keeps (NativeCombat.Reset: UseDefaultLevel, SeedStreams, CombatFactory.Reset).</summary>
    static void Episode(CombatState s, int seed)
    {
        s.AscensionLevel = Ascension.DefaultLevel;
        s.MaxPotionSlots = RunConstants.StartingPotionSlots(Ascension.DefaultLevel);
        CountingRandom Stream(string name) => new(new GameRng((uint)seed, name).RawSeed);
        s.NicheHpRng = Stream("niche");
        s.ShuffleRng = Stream("shuffle");
        s.AiRng = Stream("monster_ai");
        s.TargetRng = Stream("combat_targets");
        s.CardSelectionRng = Stream("combat_card_selection");
        s.CardGenerationRng = Stream("combat_card_generation");
        s.PotionGenerationRng = Stream("combat_potion_generation");
        var rng = new CountingRandom(seed);
        CombatFactory.Reset(s, rng);
        PrePlay.Invoke(null, [s, rng]);  // CombatEngine.ResolvePrePlayPhase (internal to the engine)
    }

    static readonly MethodInfo PrePlay = typeof(CombatEngine).GetMethod("ResolvePrePlayPhase", BindingFlags.Static | BindingFlags.NonPublic)!;

    // ============================================================================ the stages

    sealed class Ledger
    {
        public readonly Dictionary<(string Stage, string Path), string> Verdict = new();  // ok | allowed: why | FINDING: what
        public readonly List<string> Findings = [];

        public void Ok(string stage, Member m) => Verdict[(stage, m.Path)] = "ok";

        public void Miss(string stage, Member m, string what)
        {
            if (AllowedWhy(stage, m.Path) is { } why)
            {
                Verdict[(stage, m.Path)] = "allowed: " + why;
                return;
            }

            Verdict[(stage, m.Path)] = "FINDING: " + what;
            Findings.Add($"{stage,-9} {m.Path}: {what}");
        }
    }

    static string Short(string s) => s.Length <= 60 ? s : s[..57] + "...";

    static List<string> Differences(object a, object b)
    {
        var out_ = new List<string>();
        foreach (var f in StateWalk.Fields(a.GetType()))
        {
            if (StateWalk.Canon(f.GetValue(a)) != StateWalk.Canon(f.GetValue(b))) out_.Add(StateWalk.Name(f));
        }

        return out_;
    }

    static readonly List<Member> CombatMembers = Inventory(typeof(CombatState));
    static readonly List<Member> RunMembers = Inventory(typeof(RunState), typeof(CombatState));

    /// <summary>
    /// One stage over its fields. Each field's check runs under a watchdog: a stamped value can send the
    /// engine into a loop, and a field whose check hangs is a finding, not a hung suite.
    /// </summary>
    static void Stage(Ledger l, string stage, List<Member> members, Func<Member, string?> check, List<string> timing)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        foreach (var m in members)
        {
            var task = Task.Run(() =>
            {
                try
                {
                    return check(m);
                }
                catch (Exception ex)
                {
                    return "threw " + ex.GetType().Name + ": " + ex.Message;
                }
            });
            if (!task.Wait(TimeSpan.FromSeconds(20)))
            {
                l.Miss(stage, m, "its check hung (20 s)");
                continue;
            }

            if (task.Result is { } what) l.Miss(stage, m, what);
            else l.Ok(stage, m);
            Progress($"{stage} {m.Path} {sw.ElapsedMilliseconds} ms");
        }

        timing.Add($"{stage}: {members.Count} fields in {sw.Elapsed.TotalSeconds:F1} s");
    }

    /// <summary>Where a long run of the stages is (lifecycle-progress.txt next to the test assembly, emptied when the
    /// stages start). Diagnostics only, so best effort: the file is opened once and shared, and if another process
    /// holds it (a scanner or indexer, or a second run in the same worktree) progress is skipped. It used to be
    /// reopened for every line, and one sharing violation inside the stages' Lazy failed all nine tests that read it.</summary>
    static void Progress(string line)
    {
        lock (FieldLock)
        {
            if (_progressOff) return;
            try
            {
                _progress ??= new StreamWriter(new FileStream(ProgressFile, FileMode.Create, FileAccess.Write,
                    FileShare.ReadWrite | FileShare.Delete)) { AutoFlush = true };
                _progress.WriteLine(line);
            }
            catch (IOException)
            {
                _progressOff = true;
            }
        }
    }

    static StreamWriter? _progress;
    static bool _progressOff;

    static readonly string ProgressFile = Path.Combine(AppContext.BaseDirectory, "lifecycle-progress.txt");

    static readonly object FieldLock = new();

    static string? CloneCheck(Member m, Func<object> fixture, Func<object, object> clone)
    {
        var s = fixture();
        Stamp(s, m.Segs, 1);
        string want = Read(s, m.Segs);
        var c = clone(s);
        string copied = Read(c, m.Segs);
        Stamp(s, m.Segs, 2);  // the original changes again: a shared member changes in the clone too
        string after = Read(c, m.Segs);
        if (copied != want) return $"not copied ({Short(want)} -> {Short(copied)})";
        return after != want ? "shared with the original" : null;
    }

    static string? ResetCheck(Member m)
    {
        // A reused combat after one more episode against one that only had the episodes.
        var fresh = new CombatState();
        Episode(fresh, 5);
        Episode(fresh, 5);
        var used = new CombatState();
        Episode(used, 5);
        Stamp(used, m.Segs, 1);
        Episode(used, 5);
        var d = Differences(used, fresh);
        return d.Count > 0 ? "survives CombatFactory.Reset (then differs: " + string.Join(",", d.Take(4)) + ")" : null;
    }

    static readonly Lazy<string> FreshRun = new(() => StateWalk.Canon(FreshEngine()));

    static string? RunResetCheck(Member m)
    {
        var e = RunInCombat();
        Stamp(e.State, m.Segs, 1);
        e.Reset("LIFECYCLE2");
        return StateWalk.Canon(e) != FreshRun.Value ? "survives RunEngine.Reset" : null;
    }

    static string? DumpCheck(Member m)
    {
        var s = Combat();
        string before = OldDump.DumpCombat(s, true, "Combat").ToJsonString();
        Stamp(s, m.Segs, 1);
        string after;
        try
        {
            after = OldDump.DumpCombat(s, true, "Combat").ToJsonString();
        }
        catch (Exception ex)
        {
            return "the dump threw on it: " + ex.GetType().Name + ": " + ex.Message;
        }

        return after == before ? "not in EmuDump's per-step state" : null;
    }

    static string? LoadCheck(Member m)
    {
        var s = Combat();
        Stamp(s, m.Segs, 1);
        var back = CombatSnapshot.Read(CombatSnapshot.Write(s), 0, out _);
        bool same = m.Path.StartsWith("CombatState.DrawPile")  // Read shuffles the unknown middle of the pile by its seed
            ? Multiset(back.DrawPile) == Multiset(s.DrawPile)
            : Read(back, m.Segs) == Read(s, m.Segs);
        return same ? null : $"CombatSnapshot.Read does not give it back ({Short(Read(s, m.Segs))} -> {Short(Read(back, m.Segs))})";
    }

    static readonly Lazy<RunEngine> RunTemplate = new(FreshEngine);

    static string? RunDumpCheck(Member m)
    {
        // Out of a fight, where the run snapshot is taken at a decision (EmuDump run, RunReplay.cs).
        var e = RunTemplate.Value.Clone();
        string before = RunSnap.Export(e).ToJsonString();
        Stamp(e.State, m.Segs, 1);
        return RunSnap.Export(e).ToJsonString() == before ? "not in RunSnap.Export (EmuDump run's snapshot)" : null;
    }

    static string? ReanchorCheck(Member m)
    {
        var s = Combat();
        var game = GameView(s);
        string clean = Read(s, m.Segs);
        Stamp(s, m.Segs, 1);
        string? problem = OldAnchor.Anchor(OldReanchor.Tables, s, game, new JsonArray());
        if (problem != null) return "the write-back refused the state: " + problem;
        return Read(s, m.Segs) != clean ? "carried from the emulator's previous step, not written from the game's state" : null;
    }

    static readonly Lazy<(Ledger Ledger, List<string> Timing)> All = new(() =>
    {
        Progress("stages start");  // opens the file afresh: this run's progress only
        var l = new Ledger();
        var t = new List<string>();
        Stage(l, "clone", CombatMembers, m => CloneCheck(m, Combat, s => ((CombatState)s).Clone()), t);
        Stage(l, "copyto", CombatMembers, m => CloneCheck(m, Combat, s => { var into = new CombatState(); ((CombatState)s).CopyTo(into); return into; }), t);
        Stage(l, "clone", RunMembers, m => CloneCheck(m, () => RunInCombat().State, s => ((RunState)s).Clone()), t);
        Stage(l, "reset", CombatMembers, ResetCheck, t);
        Stage(l, "run-reset", RunMembers, RunResetCheck, t);
        if (StateWalk.Canon(FreshEngine()) != FreshRun.Value)
        {
            l.Findings.Add("run-reset (a field initializer shares an object between runs: a fresh engine changed while other runs' states were changed)");
        }

        Stage(l, "dump", CombatMembers, DumpCheck, t);
        Stage(l, "load", CombatMembers, LoadCheck, t);
        Stage(l, "run-dump", RunMembers, RunDumpCheck, t);
        Stage(l, "reanchor", CombatMembers, ReanchorCheck, t);
        WriteLedger(l, [.. CombatMembers, .. RunMembers]);
        return (l, t);
    });

    // ============================================================================ open findings

    /// <summary>
    /// lifecycle-open.txt: the findings already known on main, each a defect waiting for an owner (not a
    /// reason it is fine). A stage passes while each of its findings is listed there, and fails on one that
    /// is not (a field added without handling, or a regression) and on a line that matches no finding
    /// (fixed: delete it). LIFECYCLE_STRICT=1 fails on the listed ones too.
    /// </summary>
    static readonly Lazy<List<(string Stage, string Key, string Why)>> Open = new(() =>
        File.ReadAllLines(Path.Combine(AppContext.BaseDirectory, "lifecycle-open.txt"))
            .Where(x => x.Length > 0 && !x.StartsWith('#'))
            .Select(x => x.Split('\t'))
            .Select(x => (x[0], x[1], x.Length > 2 ? x[2] : ""))
            .ToList());

    static bool Strict => Environment.GetEnvironmentVariable("LIFECYCLE_STRICT") == "1";

    /// <summary>A finding line's stage and field ("reset     CombatState.Stars: survives ...").</summary>
    static (string Stage, string Path) Parse(string finding)
    {
        int sp = finding.IndexOf(' ');
        if (sp < 0) return (finding, "");
        string rest = finding[sp..].TrimStart();
        int colon = rest.IndexOf(": ", StringComparison.Ordinal);
        return (finding[..sp], colon < 0 ? rest : rest[..colon]);
    }

    static bool IsOpen(string stage, string path) => IsOpen(Open.Value, stage, path);

    static bool IsOpen(List<(string Stage, string Key, string Why)> open, string stage, string path) =>
        open.Any(o => o.Stage == stage && Matches(o.Key, path));

    /// <summary>One stage's verdict on its findings against the open list: what is new, what is listed, and the
    /// listed lines that match no finding any more. It fails on new and gone ones (and listed ones when strict).</summary>
    static (List<string> Fresh, List<string> Known, List<string> Gone, List<string> Fail) Judge(
        string stage, List<string> findings, List<(string Stage, string Key, string Why)> open, bool strict)
    {
        var mine = findings.Where(f => Parse(f).Stage == stage || (stage == "run-reset" && f.StartsWith("run-reset"))).ToList();
        var known = mine.Where(f => IsOpen(open, stage, Parse(f).Path)).ToList();
        var fresh = mine.Except(known).ToList();
        var gone = open.Where(o => o.Stage == stage && !mine.Any(f => Matches(o.Key, Parse(f).Path)))
            .Select(o => $"{o.Stage} {o.Key}: listed in lifecycle-open.txt but handled now; delete the line").ToList();
        return (fresh, known, gone, [.. fresh, .. gone, .. (strict ? known : [])]);
    }

    void Report(string stage)
    {
        var (l, timing) = All.Value;
        foreach (var line in timing) output.WriteLine(line);
        var (fresh, known, gone, fail) = Judge(stage, l.Findings, Open.Value, Strict);
        output.WriteLine($"{stage}: {fresh.Count} new findings, {known.Count} open (lifecycle-open.txt), {gone.Count} fixed lines");
        foreach (var f in fresh) output.WriteLine("NEW  " + f);
        foreach (var f in known) output.WriteLine("open " + f);
        foreach (var f in gone) output.WriteLine("gone " + f);
        Assert.True(fail.Count == 0, $"{fresh.Count} fields not handled by {stage} and not in lifecycle-open.txt, {gone.Count} lines there now fixed"
            + (Strict ? $", {known.Count} open (LIFECYCLE_STRICT)" : "") + " (lifecycle-ledger.md lists every field):\n" + string.Join("\n", fail));
    }

    [Fact]
    public void The_checks_fail_on_deliberately_broken_cases()
    {
        // A stage that loses a field, one that shares a list with the original, and the open list's
        // verdict on a new finding, a listed one, a fixed line and strict mode.
        var pen = CombatMembers.Single(m => m.Path == "CombatState.PenNibArmed");
        var hand = CombatMembers.Single(m => m.Path == "CombatState.Hand");
        Assert.Null(CloneCheck(pen, Combat, s => ((CombatState)s).Clone()));
        Assert.StartsWith("not copied", CloneCheck(pen, Combat, s => { var c = ((CombatState)s).Clone(); c.PenNibArmed = default; return c; }));
        Assert.Null(CloneCheck(hand, Combat, s => ((CombatState)s).Clone()));
        Assert.Equal("shared with the original", CloneCheck(hand, Combat, s => Memberwise.Invoke(s, null)!));

        List<string> findings = ["reset     CombatState.Stars: survives CombatFactory.Reset", "reset     CombatState.Orbs[].Type: survives CombatFactory.Reset"];
        List<(string, string, string)> open = [("reset", "CombatState.Orbs*", "open"), ("reset", "CombatState.Gone", "fixed since")];
        var (fresh, known, gone, fail) = Judge("reset", findings, open, strict: false);
        Assert.Equal(["reset     CombatState.Stars: survives CombatFactory.Reset"], fresh);
        Assert.Single(known);
        Assert.Single(gone);
        Assert.Equal(2, fail.Count);
        Assert.Empty(Judge("reset", findings[1..], open[..1], strict: false).Fail);
        Assert.Single(Judge("reset", findings[1..], open[..1], strict: true).Fail);

        // The transient check: a state at rest has nothing; each field left busy is named, by whether the fight goes on.
        var rest = Combat();
        Assert.Empty(IdleFindings(rest, false, "t"));
        var busy = Combat();
        busy.AutoPlayQueue.Add(busy.Hand[0]);
        busy.AutoPlaying = true;
        busy.ResolvingDefendCard = true;
        busy.HeldPlayedCard = busy.Hand[0];
        busy.PendingSelection = null;
        Assert.Equal(["AutoPlaying", "ResolvingDefendCard", "AutoPlayQueue", "HeldPlayedCard"],
            IdleFindings(busy, false, "t").Select(f => Parse(f).Path.Replace("CombatState.", "")).ToList());
        Assert.All(IdleFindings(busy, true, "t"), f => Assert.StartsWith("transient-end ", f));
    }

    [Fact]
    public void Clone_copies_every_field_and_shares_none() => Report("clone");

    [Fact]
    public void CopyTo_copies_every_field_and_shares_none() => Report("copyto");

    [Fact]
    public void A_reused_combat_reset_is_a_fresh_combat() => Report("reset");

    [Fact]
    public void A_run_reset_is_a_fresh_run() => Report("run-reset");

    [Fact]
    public void EmuDump_dump_shows_every_combat_field() => Report("dump");

    [Fact]
    public void CombatSnapshot_loads_back_every_field_it_writes() => Report("load");

    [Fact]
    public void The_run_snapshot_shows_every_run_field() => Report("run-dump");

    [Fact]
    public void Reanchor_writes_every_field_or_carries_it_on_purpose() => Report("reanchor");

    [Fact]
    public void The_allow_lists_name_only_fields_that_exist_and_are_not_handled()
    {
        var (l, _) = All.Value;
        var paths = CombatMembers.Concat(RunMembers).Select(m => m.Path).ToHashSet();
        var stale = new List<string>();
        foreach (var (stage, entries) in Allowed)
        {
            foreach (var (key, _) in entries)
            {
                if (key.Contains('*'))
                {
                    bool used = paths.Any(p => Matches(key, p) && l.Verdict.GetValueOrDefault((stage, p), "").StartsWith("allowed"));
                    if (!used) stale.Add($"{stage}: {key} excuses no field; remove the entry");
                }
                else if (!paths.Contains(key)) stale.Add($"{stage}: {key} is no field (renamed or removed?)");
                else if (l.Verdict.TryGetValue((stage, key), out var v) && v == "ok") stale.Add($"{stage}: {key} is handled now; remove the entry");
            }
        }

        Assert.True(stale.Count == 0, string.Join("\n", stale));
    }

    [Fact]
    public void Transient_fields_are_idle_after_every_step()
    {
        // The allowances marked Idle, IdleQueue and StepLocal rest on this: whenever a step has returned, these
        // fields are at rest (CombatSnapshotTests.NotCarried), so no reset, snapshot or write-back finds anything
        // in them. Seeded random legal play through whole fights, checked after every step, with decks that
        // auto-play cards (Hellraiser, Havoc, Knife Trap, Sly discards), open selections and hold the played
        // card, against weak and strong encounters so fights also end in the middle of a card. A finding is
        // "transient" while the fight goes on and "transient-end" after a step that ended it; both are judged
        // against lifecycle-open.txt like the stages.
        var findings = new List<string>();
        int[][] decks =
        [
            RunConstants.StarterDeckIds.ToArray(),
            [IC.Hellraiser, IC.Hellraiser, IC.BattleTrance, IC.BattleTrance, IC.Havoc, IC.Havoc, IC.PommelStrike,
             IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Armaments],
            [SI.CalculatedGamble, SI.DaggerThrow, SI.Acrobatics, SI.Survivor, SI.Prepared, SI.Reflex, SI.Reflex, SI.Tactician,
             SI.KnifeTrap, IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Havoc],
        ];
        int[] encounters = [RunConstants.SlitheringStranglerEncounterId, RunConstants.SlimesWeakEncounterId];
        int steps = 0, ends = 0, queued = 0;
        for (int d = 0; d < decks.Length; d++)
        {
            foreach (int encounter in encounters)
            {
                for (int seed = 1; seed <= 12; seed++)
                {
                    var e = new RunEngine();
                    e.Reset($"IDLE{d}{encounter:D3}{seed:D4}");
                    e.StartCombat(decks[d], encounter, [RunConstants.StartingRelicFor(PlayerCharacter.Ironclad)],
                        playerHp: 80, playerMaxHp: 80, potionIds: [], playerGold: 99);
                    var rng = new Random(seed);
                    for (int step = 0; step < 200 && e.State.ActiveCombat is { } c && e.State.Phase == RunPhase.Combat; step++)
                    {
                        var actions = CombatEngine.ValidActions(c).ToList();
                        if (actions.Count == 0) break;
                        int target = Math.Max(0, c.Enemies.FindIndex(x => x.Hp > 0));
                        e.Step(actions[rng.Next(actions.Count)], target, out _, out _, out _);
                        if (e.State.ActiveCombat is not { } after) break;
                        bool over = e.State.Phase != RunPhase.Combat;
                        steps++;
                        ends += over ? 1 : 0;
                        queued += after.AutoPlayQueue.Count > 0 || after.PendingSelection != null ? 1 : 0;
                        findings.AddRange(IdleFindings(after, over, $"deck {d}, encounter {encounter}, seed {seed}, step {step}"));
                    }
                }
            }
        }

        // A step that ends the fight in the middle of a run of auto-plays, made sure of rather than left to chance:
        // Battle Trance draws three Strikes under Hellraiser, and the first one kills the last enemy.
        var k = new RunEngine();
        k.Reset("IDLEQUEUE1");
        k.StartCombat(RunConstants.StarterDeckIds, RunConstants.SlitheringStranglerEncounterId,
            [RunConstants.StartingRelicFor(PlayerCharacter.Ironclad)], playerHp: 80, playerMaxHp: 80, potionIds: [], playerGold: 99);
        var kc = k.State.ActiveCombat!;
        kc.Hand.Clear();
        kc.Hand.Add(new CardInstance(IC.BattleTrance, false));
        kc.DrawPile.Clear();
        kc.DrawPile.AddRange(Enumerable.Repeat(new CardInstance(IC.StrikeIronclad, false), 3));
        kc.DiscardPile.Clear();
        kc.KnownTopCount = kc.KnownBottomCount = 0;
        kc.Energy = Math.Max(kc.Energy, 1);
        BuffSystem.Apply(kc.PlayerBuffs, BuffId.Hellraiser, 1);
        for (int i = 0; i < kc.Enemies.Count; i++) kc.Enemies[i].Hp = i == 0 ? 6 : 0;
        k.Step(0, 0, out _, out _, out _);
        Assert.NotEqual(RunPhase.Combat, k.State.Phase);  // the scenario must end the fight, or it checks nothing
        findings.AddRange(IdleFindings(k.State.ActiveCombat!, true, "Hellraiser + Battle Trance kill"));

        output.WriteLine($"{steps} steps checked, {ends} of them ending the fight, {queued} with a queue or a selection open");
        Assert.True(ends > 0 && queued > 0, "the random fights must end and must open selections (the queue itself is only the turn-one list now)");
        var fail = new List<string>();
        foreach (string stage in new[] { "transient", "transient-end" })
        {
            var mine = findings.Where(f => Parse(f).Stage == stage).GroupBy(f => Parse(f).Path).Select(g => g.First()).ToList();
            var (fresh, known, gone, failed) = Judge(stage, mine, Open.Value, Strict);
            foreach (var f in known) output.WriteLine("open " + f);
            fail.AddRange(failed);
        }

        Assert.True(fail.Count == 0, "transient fields not at rest after a step, and not in lifecycle-open.txt (or listed and fixed):\n"
            + string.Join("\n", fail.Take(20)));
    }

    /// <summary>The fields a returned step must leave at rest, as findings ("transient CombatState.X: ..." while the
    /// fight goes on, "transient-end ..." after a step that ended it).</summary>
    internal static List<string> IdleFindings(CombatState after, bool fightOver, string at)
    {
        var f = new List<string>();
        void Add(string field, string what) => f.Add($"{(fightOver ? "transient-end" : "transient")} CombatState.{field}: {what} ({at})");
        if (after.AutoPlaying) Add("AutoPlaying", "is set");
        if (after.ResolvingDefendCard) Add("ResolvingDefendCard", "is set");
        if (after.AutoPlayQueue.Count > 0) Add("AutoPlayQueue", $"holds {after.AutoPlayQueue.Count} cards");
        if (after.HeldPlayedCard != null && after.PendingSelection == null) Add("HeldPlayedCard", "a card held with no selection open");
        return f;
    }

    [Fact]
    public void No_static_field_a_run_changes_stays_changed()
    {
        // State kept in statics outlives every Reset. A run is played and reset; a static whose value
        // differs from before the run is a leak, unless it is a cache named here.
        var statics = typeof(CombatState).Assembly.GetTypes()
            .SelectMany(t => t.GetFields(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
                .Where(f => !f.IsLiteral && !f.Name.Contains("<>") && !typeof(Delegate).IsAssignableFrom(f.FieldType))
                .Select(f => (Name: t.Name + "." + f.Name, Field: f)))
            .ToList();
        _ = FreshEngine();  // static initializers and lazy tables, once
        var before = statics.ToDictionary(s => s.Name, s => SafeCanon(s.Field));
        var e = RunInCombat();
        for (int i = 0; i < 40 && e.State.ActiveCombat != null; i++) e.Step(e.State.ActiveCombat.Hand.Count, -1, out _, out _, out _);
        e.Reset("LIFECYCLE2");
        var changed = statics.Where(s => SafeCanon(s.Field) != before[s.Name]).Select(s => s.Name).Where(n => !StaticAllowed.ContainsKey(n)).ToList();
        Assert.True(changed.Count == 0, "static fields a run left changed: " + string.Join(", ", changed));
    }

    /// <summary>Statics a run may change that carry nothing from one run to the next, with the reason.</summary>
    static readonly Dictionary<string, string> StaticAllowed = new()
    {
        ["CardEffects._shuffleCards"] = "ThreadStatic scratch of ShuffleDiscardIntoDraw: every use writes [0, count) before it reads it",
        ["CardEffects._shuffleKeys"] = "the same scratch's sort keys",
    };

    static string SafeCanon(FieldInfo f)
    {
        try
        {
            return StateWalk.Canon(f.GetValue(null));
        }
        catch (Exception ex)
        {
            return "unreadable: " + ex.GetType().Name;
        }
    }

    static RunEngine FreshEngine()
    {
        var e = new RunEngine();
        e.Reset("LIFECYCLE2");
        return e;
    }

    static string Multiset(IEnumerable<CardInstance> pile) =>
        string.Join("|", pile.Select(c => StateWalk.Canon(c)).OrderBy(x => x, StringComparer.Ordinal));

    // ============================================================================ the game's view

    static readonly Lazy<(Dictionary<int, string> Cards, Dictionary<int, string> Enemies)> Names = new(() =>
    {
        var map = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))!;
        static string Snake(string s) => Regex.Replace(s, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();
        Dictionary<int, string> Rev(string kind)
        {
            var d = new Dictionary<int, string>();
            foreach (var kv in map[kind]!.AsObject()) d.TryAdd((int)kv.Value!, Snake(kv.Key));
            return d;
        }

        return (Rev("cards"), Rev("enemies"));
    });

    /// <summary>The state as the referee records the game's snapshot (referee/driver/Combat.cs), for the write-back.</summary>
    static JsonObject GameView(CombatState c)
    {
        var (cards, enemies) = Names.Value;
        JsonArray Pile(IEnumerable<CardInstance> p) => new(p.Select(x => (JsonNode)new JsonArray(cards[x.DefId], x.Upgraded)).ToArray());
        // A power as the game names it (STRENGTH_POWER): the fixture's buffs are ones Reanchor maps by name.
        JsonArray Powers(List<BuffState> bs) => new(bs.Select(b => (JsonNode)new JsonArray(
            Regex.Replace(b.Id.ToString(), "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant() + "_POWER", b.Magnitude)).ToArray());
        JsonObject Rng(CountingRandom? r) => new() { ["seed"] = (long)(uint)(r?.Seed ?? 0), ["counter"] = r?.CallCount ?? 0 };
        var rng = new JsonObject();
        foreach (var (name, r) in new (string, CountingRandom?)[]
                 {
                     ("Shuffle", c.ShuffleRng), ("CombatTargets", c.TargetRng), ("MonsterAi", c.AiRng as CountingRandom),
                     ("CombatCardSelection", c.CardSelectionRng), ("CombatCardGeneration", c.CardGenerationRng),
                     ("CombatEnergyCosts", c.EnergyCostRng), ("CombatPotionGeneration", c.PotionGenerationRng),
                     ("CombatOrbGeneration", c.OrbGenerationRng), ("Niche", c.NicheHpRng),
                 })
        {
            if (r != null) rng[name] = Rng(r);
        }

        return new JsonObject
        {
            ["in_combat"] = true,
            ["round"] = c.Turn + 1,
            ["player"] = new JsonObject
            {
                ["hp"] = c.PlayerHp, ["max_hp"] = c.PlayerMaxHp, ["block"] = c.PlayerBlock, ["energy"] = c.Energy, ["powers"] = Powers(c.PlayerBuffs),
            },
            ["hand"] = Pile(c.Hand), ["draw"] = Pile(c.DrawPile), ["discard"] = Pile(c.DiscardPile), ["exhaust"] = Pile(c.ExhaustPile),
            ["enemies"] = new JsonArray(c.Enemies.Where(x => x.Hp > 0).Select(x => (JsonNode)new JsonObject
            {
                ["name"] = enemies[x.DefId], ["hp"] = x.Hp, ["max_hp"] = x.MaxHp, ["block"] = x.Block, ["powers"] = Powers(x.Buffs),
            }).ToArray()),
            ["rng"] = rng,
        };
    }

    // ============================================================================ the ledger file

    static void WriteLedger(Ledger l, List<Member> members)
    {
        string[] stages = ["clone", "copyto", "reset", "dump", "load", "reanchor", "run-reset", "run-dump"];
        var sb = new StringBuilder();
        sb.AppendLine("# State lifecycle ledger (generated by LifecycleTests)").AppendLine();
        sb.AppendLine("ok: handled; allowed: left alone on purpose (the reason); open: not handled, listed in lifecycle-open.txt; NEW: not handled and not listed (fails).").AppendLine();
        sb.AppendLine("| field | " + string.Join(" | ", stages) + " |");
        sb.AppendLine("|---|" + string.Concat(stages.Select(_ => "---|")));
        foreach (var m in members)
        {
            sb.Append("| ").Append(m.Path).Append(" |");
            foreach (var st in stages)
            {
                var v = l.Verdict.GetValueOrDefault((st, m.Path), "");
                sb.Append(' ').Append(v.StartsWith("FINDING") ? (IsOpen(st, m.Path) ? "**open**" : "**NEW**") : v.StartsWith("allowed") ? "allowed" : v).Append(" |");
            }

            sb.AppendLine();
        }

        sb.AppendLine().AppendLine("## Findings").AppendLine();
        foreach (var f in l.Findings) sb.AppendLine("- " + f);
        sb.AppendLine().AppendLine("## Allowed, with the reason").AppendLine();
        foreach (var ((st, path), v) in l.Verdict.Where(kv => kv.Value.StartsWith("allowed")).OrderBy(kv => kv.Key.Stage).ThenBy(kv => kv.Key.Path))
        {
            sb.AppendLine($"- {st} {path}: {v["allowed: ".Length..]}");
        }

        File.WriteAllText(Path.Combine(AppContext.BaseDirectory, "lifecycle-ledger.md"), sb.ToString());
    }
}
