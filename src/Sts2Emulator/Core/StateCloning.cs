namespace Sts2Emulator.Core;

using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;

/// <summary>
/// Deep copies of the mutable simulation state, so a search can fork a position
/// instead of replaying the run to reach it.
/// </summary>
/// <remarks>
/// <para>
/// Everything here is hand-written rather than reflected, because the core is
/// NativeAOT. That makes it drift-prone -- a field added to a state class and not
/// added here is a clone that silently shares it -- so <c>StateCloningTests</c>
/// walks every field by reflection and fails when one is missed.
/// </para>
/// <para>
/// A plain clone is a faithful copy, which for a tree search is an oracle: every
/// run-level stream derives from the run seed, so rolling forward reads the exact
/// rewards, shop stock and encounter compositions still to come. Resampling is what
/// makes a clone safe to search with -- see <c>RunEngine.ResampleHiddenState</c> and
/// docs/agent-interface.md.
/// </para>
/// </remarks>
public static class StateCloning
{
    public static GameRng Clone(this GameRng rng) => rng.Copy();

    /// <summary>
    /// A copy of this stream, from a DIFFERENT seed but at the same call count.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The position is kept deliberately. A stream's call count is not hidden information --
    /// it is a fact about how much has happened -- so a determinization should change what
    /// the next value IS, not how far along the sequence stands. Keeping it also means a
    /// reseeded copy stays comparable to the original in the stream-position diff.
    /// </para>
    /// <para>
    /// A non-null <paramref name="streamName" /> is mixed into the seed, so that reseeding
    /// several sibling streams from the same <paramref name="seed" /> -- as
    /// <see cref="Clone(CombatState, int?)" /> does -- gives each of them a DIFFERENT
    /// derived seed instead of the same one. Without it, two streams at the same call
    /// count would draw the identical sequence of values after reseeding, which is not
    /// what an independent stream coming out of a determinization is supposed to look
    /// like. The one caller with no sibling to decorrelate from (<c>NativeCombat</c>'s own
    /// top-level Rng) omits it and keeps the un-derived, single-stream behaviour.
    /// </para>
    /// </remarks>
    public static CountingRandom? Reseeded(
        this CountingRandom? rng,
        int? seed,
        string? streamName = null
    )
    {
        if (rng is null)
        {
            return null;
        }

        if (seed is null)
        {
            return rng.Clone();
        }

        int streamSeed = streamName is null
            ? seed.Value
            : unchecked(
                seed.Value * (int)0x9E3779B1
                + DeterministicHash.GetDeterministicHashCode(streamName)
            );
        var copy = new CountingRandom(streamSeed);
        for (int i = 0; i < rng.CallCount; i++)
        {
            copy.Next();
        }

        return copy;
    }

    public static CountingRandom Clone(this CountingRandom rng) => rng.Copy();

    public static EnemyState Clone(this EnemyState enemy)
    {
        var copy = new EnemyState();
        enemy.CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Every field of <paramref name="enemy" /> onto <paramref name="into" />, as <see cref="Clone(EnemyState)" />
    /// copies it (the one field list both use). <paramref name="inPlace" /> refills <paramref name="into" />'s
    /// lists instead of replacing them (<see cref="CopyTo(CombatState, CombatState, int?, bool)" />).
    /// </summary>
    public static void CopyTo(this EnemyState enemy, EnemyState into, bool inPlace = false)
    {
        into.DefId = enemy.DefId;
        into.Hp = enemy.Hp;
        into.MaxHp = enemy.MaxHp;
        into.Block = enemy.Block;
        into.CurrentIntent = enemy.CurrentIntent;
        into.SecondaryIntent = enemy.SecondaryIntent;
        into.Buffs = Refill(into.Buffs, enemy.Buffs, inPlace);
        into.LastSpawnedDefId = enemy.LastSpawnedDefId;
        into.PoweredHitsThisTurn = enemy.PoweredHitsThisTurn;
        into.HardenedShellDamageThisTurn = enemy.HardenedShellDamageThisTurn;
        into.MoveIndex = enemy.MoveIndex;
        into.LastMove = enemy.LastMove;
        into.LastMoveRepeats = enemy.LastMoveRepeats;
        into.OnceOnlyMoveUsed = enemy.OnceOnlyMoveUsed;
        into.MoveHistory = Refill(into.MoveHistory, enemy.MoveHistory, inPlace);
        into.StartsOnBranch = enemy.StartsOnBranch;
        into.RollsNextMove = enemy.RollsNextMove;
        into.CurlUpArmed = enemy.CurlUpArmed;
        into.NemesisIntangibleOn = enemy.NemesisIntangibleOn;
        into.StarterMove = enemy.StarterMove;
        into.Slot = enemy.Slot;
        into.StolenGold = enemy.StolenGold;
        into.HeistGold = enemy.HeistGold;
        into.StolenCard = enemy.StolenCard;
        into.Escaped = enemy.Escaped;
        into.OffBalance = enemy.OffBalance;
        into.LastBranch = enemy.LastBranch;
        into.RepeatStreak = enemy.RepeatStreak;
        into.BranchCooldown = enemy.BranchCooldown;
    }

    public static PendingCardSelection Clone(this PendingCardSelection selection) =>
        selection.Clone(selection.Replay);

    /// <summary>
    /// A copy of the screen with <paramref name="replay" /> as the step it stopped, and
    /// <paramref name="candidates" /> as its candidates when given.
    /// </summary>
    public static PendingCardSelection Clone(
        this PendingCardSelection selection,
        HookChoiceReplay? replay,
        List<int>? candidates = null
    ) =>
        new()
        {
            Kind = selection.Kind,
            Candidates = candidates ?? [.. selection.Candidates],
            SourceCardDefId = selection.SourceCardDefId,
            Amount = selection.Amount,
            GeneratedCandidates = [.. selection.GeneratedCandidates],
            AfterSelectionToHand = [.. selection.AfterSelectionToHand],
            Skippable = selection.Skippable,
            GeneratedUpgraded = selection.GeneratedUpgraded,
            GeneratedRetained = selection.GeneratedRetained,
            // Never changed once made, so a copy shares it (HookChoiceReplay).
            Replay = replay,
        };

    /// <summary>
    /// Copy a combat, optionally RESEEDING every random stream it owns.
    /// </summary>
    /// <remarks>
    /// Without a seed this is a faithful copy: same streams, same seeds, same call counts,
    /// so the copy plays out exactly as the original would. That is what the differential
    /// tests want and must not change.
    ///
    /// With one, each stream is rebuilt from the new seed and fast-forwarded to the SAME
    /// call count -- different values, identical positions. That is the latent half of a
    /// determinization: every roll still to come (enemy moves, card generation, targeting,
    /// orb and HP rolls) resolves differently in each copy, which is what a search over an
    /// imperfect-information game needs and a plain clone cannot give it. `RunState.Clone`
    /// has taken a `stringSeed` for this reason; combat had no equivalent.
    ///
    /// It does NOT touch the draw pile. That order was materialised by a shuffle that has
    /// already happened, so reseeding cannot undo it -- see `ReshuffleUnknownDrawOrder`.
    /// </remarks>
    public static CombatState Clone(this CombatState combat, int? rngSeed = null)
    {
        var copy = new CombatState();
        combat.CopyTo(copy, rngSeed);
        return copy;
    }

    /// <summary>
    /// Every field of <paramref name="combat" /> onto <paramref name="into" />, as <see cref="Clone" />
    /// copies it: the one field list both use, so an in-place restore cannot drift from the
    /// copy. <c>CombatEngine.Step</c> restores a step this way when it replays it to answer a
    /// screen raised in the middle of it (<see cref="HookChoiceReplay" />); <paramref name="combat" />
    /// is left as it was.
    /// </summary>
    /// <param name="inPlace">
    /// Refill <paramref name="into" />'s lists, potion belt and enemies (each enemy onto the one at
    /// its index) instead of replacing them: for a combat in the middle of a step, whose code holds
    /// them, as when a step from a document puts the answering copy back whole at its screen
    /// (<c>ScreenHiddenState.Whole</c>). The streams are moved back in place either way where they
    /// are the same stream (<see cref="Carry" />).
    /// </param>
    public static void CopyTo(this CombatState combat, CombatState into, int? rngSeed = null, bool inPlace = false)
    {
        into.AscensionLevel = combat.AscensionLevel;
        into.Character = combat.Character;
        into.Unlocks = combat.Unlocks;
        into.PlayerHp = combat.PlayerHp;
        into.PlayerMaxHp = combat.PlayerMaxHp;
        into.PlayerBlock = combat.PlayerBlock;
        into.Energy = combat.Energy;
        into.MaxEnergy = combat.MaxEnergy;
        into.PlayerGold = combat.PlayerGold;
        into.StolenBackGold = combat.StolenBackGold;
        into.RoyaltiesGold = combat.RoyaltiesGold;
        into.FatGremlinEscaped = combat.FatGremlinEscaped;
        into.EscapedEnemies = combat.EscapedEnemies;
        into.DepartedEnemyDefIds = Refill(into.DepartedEnemyDefIds, combat.DepartedEnemyDefIds, inPlace);
        into.BattlewornDummyRanOutOfTime = combat.BattlewornDummyRanOutOfTime;
        into.PaelsLegionCooldown = combat.PaelsLegionCooldown;
        into.PaelsLegionCard = combat.PaelsLegionCard;
        into.MercGoldWasStolen = combat.MercGoldWasStolen;
        into.Hand = Refill(into.Hand, combat.Hand, inPlace);
        into.DrawPile = Refill(into.DrawPile, combat.DrawPile, inPlace);
        into.DiscardPile = Refill(into.DiscardPile, combat.DiscardPile, inPlace);
        into.ExhaustPile = Refill(into.ExhaustPile, combat.ExhaustPile, inPlace);
        into.ReturnToHandBeforeDraw = Refill(into.ReturnToHandBeforeDraw, combat.ReturnToHandBeforeDraw, inPlace);
        into.CopiesToHandBeforeDraw = Refill(into.CopiesToHandBeforeDraw, combat.CopiesToHandBeforeDraw, inPlace);
        into.EndTurnAwaitingSelection = combat.EndTurnAwaitingSelection;
        into.AfterimageBeforePlay = combat.AfterimageBeforePlay;
        into.StormBeforePlay = combat.StormBeforePlay;
        into.SubroutineBeforePlay = combat.SubroutineBeforePlay;
        into.StatusCardsDrawnThisTurn = combat.StatusCardsDrawnThisTurn;
        into.ResolvingDefendCard = combat.ResolvingDefendCard;
        into.PlayedCardCostForCombat = combat.PlayedCardCostForCombat;
        into.AutoPlayQueue = Refill(into.AutoPlayQueue, combat.AutoPlayQueue, inPlace);
        into.HeldPlayedCard = combat.HeldPlayedCard;
        into.HeldPlayedCardExhausts = combat.HeldPlayedCardExhausts;
        into.Orbs = Refill(into.Orbs, combat.Orbs, inPlace);
        into.OrbCapacity = combat.OrbCapacity;
        into.BaseOrbSlots = combat.BaseOrbSlots;
        into.OstyHp = combat.OstyHp;
        into.OstyMaxHp = combat.OstyMaxHp;
        into.Stars = combat.Stars;
        into.StrangleBeforePlay = Refill(into.StrangleBeforePlay, combat.StrangleBeforePlay, inPlace);
        into.PotionSlots = inPlace && into.PotionSlots.Length == combat.PotionSlots.Length && !ReferenceEquals(into.PotionSlots, combat.PotionSlots)
            ? Refill(into.PotionSlots, combat.PotionSlots)
            : (int[])combat.PotionSlots.Clone();
        into.MaxPotionSlots = combat.MaxPotionSlots;
        into.Relics = Refill(into.Relics, combat.Relics, inPlace);
        into.PlayerBuffs = Refill(into.PlayerBuffs, combat.PlayerBuffs, inPlace);
        into.PlayerDebuffsAtRoundStart = Refill(into.PlayerDebuffsAtRoundStart, combat.PlayerDebuffsAtRoundStart, inPlace);
        into.Enemies = RefillEnemies(into.Enemies, combat.Enemies, inPlace);
        into.EncounterId = combat.EncounterId;
        into.IsEliteCombat = combat.IsEliteCombat;
        into.ShuffleRng = Carry(into.ShuffleRng, combat.ShuffleRng);
        into.TargetRng = Carry(into.TargetRng, combat.TargetRng);
        into.CardSelectionRng = Carry(into.CardSelectionRng, combat.CardSelectionRng);
        into.EnergyCostRng = Carry(into.EnergyCostRng, combat.EnergyCostRng);
        into.CardGenerationRng = Carry(into.CardGenerationRng, combat.CardGenerationRng);
        into.PotionGenerationRng = Carry(into.PotionGenerationRng, combat.PotionGenerationRng);
        into.OrbGenerationRng = Carry(into.OrbGenerationRng, combat.OrbGenerationRng);
        // A CountingRandom carries its seed and position, so it clones exactly. A
        // plain Random does not, and only the combat-only environment supplies one
        // -- that path has no clone export, so the shared reference is unreachable.
        into.AiRng = combat.AiRng is CountingRandom counting
            ? Carry(into.AiRng as CountingRandom, counting)
            : combat.AiRng;
        into.NicheHpRng = Carry(into.NicheHpRng, combat.NicheHpRng);
        // Never changed once made, so the copy shares its contents.
        into.LastIntentRoll = combat.LastIntentRoll is { } roll ? roll with { } : null;
        into.AiStateAssumed = combat.AiStateAssumed;
        if (rngSeed is int seed)
        {
            into.ReseedStreams(seed);
        }

        // A screen that stopped a step keeps the step's own copy (HookChoiceReplay), shared: a
        // resample cannot reach it, and need not. The answer replays the step on the streams it
        // started with, and puts this copy's streams back at the screen (ScreenHiddenState).
        into.PendingSelection = combat.PendingSelection?.Clone();
        into.AutoPlaying = combat.AutoPlaying;
        into.PlayedCardBonusDamage = combat.PlayedCardBonusDamage;
        into.PlayedCardBonusBlock = combat.PlayedCardBonusBlock;
        into.EnergySpentThisTurn = combat.EnergySpentThisTurn;
        into.PenNibArmed = combat.PenNibArmed;
        into.EtherealExhaustsThisTurn = combat.EtherealExhaustsThisTurn;
        into.VambraceSpent = combat.VambraceSpent;
        into.VambraceCard = combat.VambraceCard;
        into.UnblockedDamageThisTurn = combat.UnblockedDamageThisTurn;
        into.RainbowRingAttacks = combat.RainbowRingAttacks;
        into.RainbowRingSkills = combat.RainbowRingSkills;
        into.RainbowRingPowers = combat.RainbowRingPowers;
        into.RainbowRingPaidThisTurn = combat.RainbowRingPaidThisTurn;
        into.UnsettlingLampSpent = combat.UnsettlingLampSpent;
        into.UnsettlingLampCard = combat.UnsettlingLampCard;
        into.GamblingChipDiscarded = combat.GamblingChipDiscarded;
        into.TurnStartHeldPhase = combat.TurnStartHeldPhase;
        into.TurnStartHeldRelic = combat.TurnStartHeldRelic;
        into.BurningSticksUsed = combat.BurningSticksUsed;
        into.BeltBuckleApplied = combat.BeltBuckleApplied;
        into.IsEliteRoom = combat.IsEliteRoom;
        into.TookUnblockedDamage = combat.TookUnblockedDamage;
        into.PlayerHitTurn = combat.PlayerHitTurn;
        into.PlayerHitTurnBefore = combat.PlayerHitTurnBefore;
        into.PlayedCardEnchantSpent = combat.PlayedCardEnchantSpent;
        into.PlayedCardEnchantGrew = combat.PlayedCardEnchantGrew;
        into.PlayedCardCostBump = combat.PlayedCardCostBump;
        into.Turn = combat.Turn;
        into.PlayerTurn = combat.PlayerTurn;
        into.LastAttackOrSkillThisTurn = combat.LastAttackOrSkillThisTurn;
        into.LastAttackOrSkillLastTurn = combat.LastAttackOrSkillLastTurn;
        into.SkillPlayedWhileSmoggy = combat.SkillPlayedWhileSmoggy;
        into.CardsPlayedThisTurn = combat.CardsPlayedThisTurn;
        into.CardPlaysStartedThisTurn = combat.CardPlaysStartedThisTurn;
        into.CardPlaySeriesStartedThisTurn = combat.CardPlaySeriesStartedThisTurn;
        into.AttackCardsPlayedThisTurn = combat.AttackCardsPlayedThisTurn;
        into.AttackOrSkillCardsPlayedThisTurn = combat.AttackOrSkillCardsPlayedThisTurn;
        into.CardPlaysThisTurn = combat.CardPlaysThisTurn;
        into.CardsPlayedThisCombat = combat.CardsPlayedThisCombat;
        into.CardsDrawnThisCombat = combat.CardsDrawnThisCombat;
        into.ShivsPlayedThisTurn = combat.ShivsPlayedThisTurn;
        into.ExtraCardRewards = combat.ExtraCardRewards;
        into.StolenCards = Refill(into.StolenCards, combat.StolenCards, inPlace);
        into.StolenBackCards = Refill(into.StolenBackCards, combat.StolenBackCards, inPlace);
        into.ExecutedThisEnemyTurn = Refill(into.ExecutedThisEnemyTurn, combat.ExecutedThisEnemyTurn, inPlace);
        into.RuptureOwed = combat.RuptureOwed;
        into.BlockGainsThisTurn = combat.BlockGainsThisTurn;
        into.PlayerHpLostThisTurn = combat.PlayerHpLostThisTurn;
        into.CardsExhaustedThisTurn = combat.CardsExhaustedThisTurn;
        into.LightningOrbsChanneledThisCombat = combat.LightningOrbsChanneledThisCombat;
        into.EtherealExhaustCount = combat.EtherealExhaustCount;
        into.EtherealCardPlaysThisCombat = combat.EtherealCardPlaysThisCombat;
        into.LethalAttackPlay = combat.LethalAttackPlay;
        into.SkillCardsPlayedThisTurn = combat.SkillCardsPlayedThisTurn;
        into.BoundCardsThisTurn = combat.BoundCardsThisTurn;
        into.BoundCardPlayedThisTurn = combat.BoundCardPlayedThisTurn;
        into.MusicBoxUsedThisTurn = combat.MusicBoxUsedThisTurn;
        into.FurCoatBites = combat.FurCoatBites;
        into.WhisperingEarringSpent = combat.WhisperingEarringSpent;
        into.CardPlaysLastTurn = combat.CardPlaysLastTurn;
        into.ExtraCardsDrawnThisTurn = combat.ExtraCardsDrawnThisTurn;
        into.DoomAppliedThisTurn = combat.DoomAppliedThisTurn;
        into.OstyAttacksThisTurn = combat.OstyAttacksThisTurn;
        into.StarsGainedThisTurn = combat.StarsGainedThisTurn;
        into.PlayedCardStarsSpent = combat.PlayedCardStarsSpent;
        into.VoidFormCardsPlayedThisTurn = combat.VoidFormCardsPlayedThisTurn;
        into.EndTurnAfterPlay = combat.EndTurnAfterPlay;
        into.CardsGeneratedThisCombat = combat.CardsGeneratedThisCombat;
        into.TenderCardsPlayedThisTurn = combat.TenderCardsPlayedThisTurn;
        into.UnblockedDamageHitCount = combat.UnblockedDamageHitCount;
        into.TargetEnemyIndex = combat.TargetEnemyIndex;
        into.KnownTopCount = combat.KnownTopCount;
        into.KnownBottomCount = combat.KnownBottomCount;
    }

    /// <summary>
    /// A stream for <see cref="CopyTo" />: onto a combat that already holds the same stream (a
    /// step restored for a replay), it is moved back in place, so whoever holds the object -- a
    /// trace naming streams by object, a test counting draws -- keeps holding the combat's
    /// stream; anywhere else it is a copy.
    /// </summary>
    /// <summary>
    /// <paramref name="source" />'s items in a list of their own: <paramref name="existing" /> refilled when
    /// <paramref name="inPlace" /> (so whoever holds that list goes on with it), else a new one.
    /// </summary>
    private static List<T> Refill<T>(List<T> existing, List<T> source, bool inPlace)
    {
        if (!inPlace || existing is null || ReferenceEquals(existing, source))
        {
            return [.. source];
        }

        existing.Clear();
        existing.AddRange(source);
        return existing;
    }

    private static int[] Refill(int[] existing, int[] source)
    {
        Array.Copy(source, existing, source.Length);
        return existing;
    }

    /// <summary>
    /// <paramref name="source" />'s enemies as copies; in place, each copied onto the enemy standing at its
    /// index, so a reference held to it (the turn end's <c>Baseline</c>, a play's target) stays good. A roster
    /// of another size is built anew.
    /// </summary>
    private static List<EnemyState> RefillEnemies(List<EnemyState> existing, List<EnemyState> source, bool inPlace)
    {
        if (!inPlace || existing is null || ReferenceEquals(existing, source) || existing.Count != source.Count)
        {
            return [.. source.Select(enemy => enemy.Clone())];
        }

        for (int i = 0; i < source.Count; i++)
        {
            if (ReferenceEquals(existing[i], source[i]))
            {
                existing[i] = source[i].Clone();
            }
            else
            {
                source[i].CopyTo(existing[i], inPlace: true);
            }
        }

        return existing;
    }

    private static CountingRandom? Carry(CountingRandom? existing, CountingRandom? source)
    {
        if (
            existing is not null
            && source is not null
            && !ReferenceEquals(existing, source)
            && existing.Seed == source.Seed
        )
        {
            existing.RestoreFrom(source);
            return existing;
        }

        return source?.Clone();
    }

    /// <summary>
    /// Every stream the combat owns rebuilt from <paramref name="seed" /> at the same call count,
    /// as <see cref="Clone(CombatState, int?)" /> resamples them.
    /// </summary>
    public static void ReseedStreams(this CombatState combat, int seed)
    {
        for (int i = 0; i < StreamCount; i++)
        {
            if (Stream(combat, i) is { } stream)
            {
                SetStream(combat, i, stream.Reseeded(seed, StreamNames[i]));
            }
        }
    }

    /// <summary>
    /// The streams a combat owns, by field name: the one list a resample
    /// (<see cref="ReseedStreams" />) and a stopped step's answer (<c>ScreenHiddenState</c>) go
    /// through. The name is mixed into a resampled stream's seed (<see cref="Reseeded" />).
    /// </summary>
    internal static readonly string[] StreamNames =
    [
        nameof(CombatState.ShuffleRng),
        nameof(CombatState.TargetRng),
        nameof(CombatState.CardSelectionRng),
        nameof(CombatState.EnergyCostRng),
        nameof(CombatState.CardGenerationRng),
        nameof(CombatState.PotionGenerationRng),
        nameof(CombatState.OrbGenerationRng),
        nameof(CombatState.AiRng),
        nameof(CombatState.NicheHpRng),
    ];

    internal static int StreamCount => StreamNames.Length;

    /// <summary>The combat's stream <paramref name="index" /> of <see cref="StreamNames" />, null when unset (or a plain Random).</summary>
    internal static CountingRandom? Stream(CombatState combat, int index) =>
        index switch
        {
            0 => combat.ShuffleRng,
            1 => combat.TargetRng,
            2 => combat.CardSelectionRng,
            3 => combat.EnergyCostRng,
            4 => combat.CardGenerationRng,
            5 => combat.PotionGenerationRng,
            6 => combat.OrbGenerationRng,
            7 => combat.AiRng as CountingRandom,
            8 => combat.NicheHpRng,
            _ => throw new ArgumentOutOfRangeException(nameof(index)),
        };

    internal static void SetStream(CombatState combat, int index, CountingRandom? stream)
    {
        switch (index)
        {
            case 0: combat.ShuffleRng = stream; break;
            case 1: combat.TargetRng = stream; break;
            case 2: combat.CardSelectionRng = stream; break;
            case 3: combat.EnergyCostRng = stream; break;
            case 4: combat.CardGenerationRng = stream; break;
            case 5: combat.PotionGenerationRng = stream; break;
            case 6: combat.OrbGenerationRng = stream; break;
            case 7: combat.AiRng = stream; break;
            case 8: combat.NicheHpRng = stream; break;
            default: throw new ArgumentOutOfRangeException(nameof(index));
        }
    }

    public static RunMapNode Clone(this RunMapNode node)
    {
        var copy = new RunMapNode(node.Col, node.Row)
        {
            NodeType = node.NodeType,
            EncounterId = node.EncounterId,
            CanBeModified = node.CanBeModified,
        };
        copy.Children.AddRange(node.Children);
        copy.Parents.AddRange(node.Parents);
        return copy;
    }

    /// <summary>
    /// A RunRngSet is twelve streams off one seed, so it rebuilds from the seed and
    /// then moves each stream to where the original had got to (an O(1) state copy when the
    /// seed is the same; a resample's new seed still draws its way there).
    /// </summary>
    public static RunRngSet CloneAt(string stringSeed, RunRngSet source)
    {
        var copy = new RunRngSet(stringSeed);
        copy.UpFront.MoveToPositionOf(source.UpFront);
        copy.Shuffle.MoveToPositionOf(source.Shuffle);
        copy.UnknownMapPoint.MoveToPositionOf(source.UnknownMapPoint);
        copy.CombatCardGeneration.MoveToPositionOf(source.CombatCardGeneration);
        copy.CombatPotionGeneration.MoveToPositionOf(source.CombatPotionGeneration);
        copy.CombatCardSelection.MoveToPositionOf(source.CombatCardSelection);
        copy.CombatEnergyCosts.MoveToPositionOf(source.CombatEnergyCosts);
        copy.CombatTargets.MoveToPositionOf(source.CombatTargets);
        copy.MonsterAi.MoveToPositionOf(source.MonsterAi);
        copy.Niche.MoveToPositionOf(source.Niche);
        copy.CombatOrbs.MoveToPositionOf(source.CombatOrbs);
        copy.TreasureRoomRelics.MoveToPositionOf(source.TreasureRoomRelics);
        return copy;
    }

    public static PlayerRngSet CloneAt(RunRngSet runRng, PlayerRngSet source)
    {
        var copy = new PlayerRngSet(runRng);
        copy.Rewards.AdvanceToCallCount(source.Rewards.CallCount);
        copy.Shops.AdvanceToCallCount(source.Shops.CallCount);
        copy.Transformations.AdvanceToCallCount(source.Transformations.CallCount);
        return copy;
    }

    /// <summary>
    /// A deep copy of the run. Pass a different <paramref name="stringSeed" /> to
    /// rebuild every stream off a new seed, which resamples everything the run has
    /// not yet paid out -- rewards, shop stock, encounter composition, shuffles --
    /// while leaving what has already happened exactly as it was.
    /// </summary>
    public static RunState Clone(this RunState state, string? stringSeed = null)
    {
        var rng = CloneAt(stringSeed ?? state.StringSeed, state.Rng);
        var mapNodes = new Dictionary<(int Col, int Row), RunMapNode>(state.MapNodes.Count);
        foreach (var (coord, node) in state.MapNodes)
        {
            mapNodes[coord] = node.Clone();
        }

        return new RunState
        {
            StringSeed = state.StringSeed,
            AscensionLevel = state.AscensionLevel,
            Character = state.Character,
            Unlocks = state.Unlocks,
            Rng = rng,
            PlayerRng = CloneAt(rng, state.PlayerRng),
            PlayerHp = state.PlayerHp,
            PlayerMaxHp = state.PlayerMaxHp,
            Gold = state.Gold,
            Floor = state.Floor,
            Phase = state.Phase,
            Deck = [.. state.Deck],
            Relics = [.. state.Relics],
            UsedUpRelics = [.. state.UsedUpRelics],
            MaxPotionSlots = state.MaxPotionSlots,
            CardRewardRerollAvailable = state.CardRewardRerollAvailable,
            LastingCandyRewardCard = state.LastingCandyRewardCard,
            LastingCandyRewardUpgraded = state.LastingCandyRewardUpgraded,
            CardRemovalsOwed = state.CardRemovalsOwed,
            RoyaltiesGoldOwed = state.RoyaltiesGoldOwed,
            RestOptionsTaken = state.RestOptionsTaken,
            TookUnblockedDamageThisCombat = state.TookUnblockedDamageThisCombat,
            RewardEnchantIndex = state.RewardEnchantIndex,
            RewardEnchantment = state.RewardEnchantment,
            PendingBonusRelicRewards = [.. state.PendingBonusRelicRewards],
            WaxRelics = [.. state.WaxRelics],
            FurCoatRooms = [.. state.FurCoatRooms],
            FurCoatAct = state.FurCoatAct,
            GoldenPathAct = state.GoldenPathAct,
            SpoilsQuestCoord = state.SpoilsQuestCoord,
            ParasolSweepPending = state.ParasolSweepPending,
            LavaRockTriggered = state.LavaRockTriggered,
            SilverCrucibleTreasureRoomsEntered = state.SilverCrucibleTreasureRoomsEntered,
            PendingSpecialCardReward = state.PendingSpecialCardReward,
            ResumeEventId = state.ResumeEventId,
            ResumeEventPage = state.ResumeEventPage,
            ResumeEventDummyEscaped = state.ResumeEventDummyEscaped,
            PendingNeowsBonesCurse = state.PendingNeowsBonesCurse,
            PendingHeftyTabletCurse = state.PendingHeftyTabletCurse,
            PendingPotionRewards = [.. state.PendingPotionRewards],
            PotionSlots = (int[])state.PotionSlots.Clone(),
            CurrentNodeType = state.CurrentNodeType,
            NeowOptions = (int[])state.NeowOptions.Clone(),
            RewardCards = (int[])state.RewardCards.Clone(),
            RewardGold = state.RewardGold,
            RewardPotion = state.RewardPotion,
            RewardCardPending = state.RewardCardPending,
            PendingOtherCharacterCardRewards = state.PendingOtherCharacterCardRewards,
            NeowAwaitingProceed = state.NeowAwaitingProceed,
            EventAwaitingProceed = state.EventAwaitingProceed,
            AwaitingActStartNode = state.AwaitingActStartNode,
            ReturnToRewardScreenAfterCardReward = state.ReturnToRewardScreenAfterCardReward,
            MapNodeTypes = (int[])state.MapNodeTypes.Clone(),
            MapChoices = (int[])state.MapChoices.Clone(),
            ShopCards = (int[])state.ShopCards.Clone(),
            ShopCardUpgraded = (bool[])state.ShopCardUpgraded.Clone(),
            ShopRelics = (int[])state.ShopRelics.Clone(),
            ShopPotions = (int[])state.ShopPotions.Clone(),
            ShopCosts = (int[])state.ShopCosts.Clone(),
            RewardUpgraded = (bool[])state.RewardUpgraded.Clone(),
            RelicReward = state.RelicReward,
            EventId = state.EventId,
            EventValue0 = state.EventValue0,
            EventValue1 = state.EventValue1,
            ActiveCombat = state.ActiveCombat?.Clone(),
            ActiveCombatRng = state.ActiveCombatRng?.Clone(),
            LastPlayerWon = state.LastPlayerWon,
            MapNodes = mapNodes,
            CurrentMapCoord = state.CurrentMapCoord,
            MapOptionCoords = ((int Col, int Row)?[])state.MapOptionCoords.Clone(),
            // The act list carries the sequences and the boss for EVERY act, and the
            // per-act properties are views on it — so cloning the list is what clones
            // them. Copying the views instead would silently drop acts 2 and 3.
            Acts =
            [
                .. state.Acts.Select(act =>
                    act with
                    {
                        Events = (int[])act.Events.Clone(),
                        NormalEncounters = (int[])act.NormalEncounters.Clone(),
                        EliteEncounters = (int[])act.EliteEncounters.Clone(),
                    }
                ),
            ],
            CurrentActIndex = state.CurrentActIndex,
            SeaGlassCharacter = state.SeaGlassCharacter,
            NormalEncountersVisited = state.NormalEncountersVisited,
            EliteEncountersVisited = state.EliteEncountersVisited,
            EventSequenceIndex = state.EventSequenceIndex,
            CardRarityOffset = state.CardRarityOffset,
            PotionRewardOdds = state.PotionRewardOdds,
            PendingRelicReward = state.PendingRelicReward,
            ShopRemovalsUsed = state.ShopRemovalsUsed,
            ShopRemovalUsedThisVisit = state.ShopRemovalUsedThisVisit,
            TransformSelectedDeckIndex = state.TransformSelectedDeckIndex,
            PendingSelectionKind = state.PendingSelectionKind,
            PendingSelectionArg = state.PendingSelectionArg,
            PendingSelectionCount = state.PendingSelectionCount,
            PendingOfferCards = (int[])state.PendingOfferCards.Clone(),
            PendingOfferUpgraded = (bool[])state.PendingOfferUpgraded.Clone(),
            BundleOffer = (int[])state.BundleOffer.Clone(),
            SelectedBundle = state.SelectedBundle,
            PendingOfferPicks = state.PendingOfferPicks,
            PendingOfferSwiftIndexes = [.. state.PendingOfferSwiftIndexes],
            PendingOfferHooked = state.PendingOfferHooked,
            PendingOfferOptional = state.PendingOfferOptional,
            PendingSelectionOptional = state.PendingSelectionOptional,
            EventPage = state.EventPage,
            EventResumesAfterRewards = state.EventResumesAfterRewards,
            CrystalSphere = state.CrystalSphere?.Clone(),
            CrystalSphereRng = state.CrystalSphereRng?.Clone(),
            EventRngStream = state.EventRngStream?.Clone(),
            EventRngName = state.EventRngName,
            EventRelicStock = state.EventRelicStock is null ? null : [.. state.EventRelicStock],
            EventSkippedCards = [.. state.EventSkippedCards],
            EventCombatHps = state.EventCombatHps is null ? null : [.. state.EventCombatHps],
            AddingBingBongClone = state.AddingBingBongClone,
            GuiltyCombatsServed = [.. state.GuiltyCombatsServed],
            VisitedEventIds = [.. state.VisitedEventIds],
            PendingSelectionEventEntry = state.PendingSelectionEventEntry,
            PendingGoldRewards = [.. state.PendingGoldRewards],
            RewardSpecialCards = [.. state.RewardSpecialCards],
            PendingCardOffers =
            [
                .. state.PendingCardOffers.Select(offer => new CardOffer(
                    (int[])offer.Cards.Clone(),
                    (bool[])offer.Upgraded.Clone(),
                    offer.EnchantIndex,
                    offer.LastingCandyCard,
                    offer.LastingCandyUpgraded,
                    offer.Reroll
                )),
            ],
            PendingSelectionFollowUpCard = state.PendingSelectionFollowUpCard,
            PendingSelectionFollowUpCount = state.PendingSelectionFollowUpCount,
            PendingSelectionFollowUpHpLoss = state.PendingSelectionFollowUpHpLoss,
            PendingSelectionFollowUpMaxHpGain = state.PendingSelectionFollowUpMaxHpGain,
            PendingSelectionEnchantAmount = state.PendingSelectionEnchantAmount,
            PendingSelectionCardType = state.PendingSelectionCardType,
            PendingSelectionReturnsToEvent = state.PendingSelectionReturnsToEvent,
            PendingSelectionReturn = state.PendingSelectionReturn,
            EventRandomOffer = (int[])state.EventRandomOffer.Clone(),
            TinkerCardType = state.TinkerCardType,
            RelicBag = state.RelicBag.Clone(),
            SharedRelicBag = state.SharedRelicBag.Clone(),
            PendingRestUpgrade = state.PendingRestUpgrade,
            RestResultPending = state.RestResultPending,
            UnknownMapPointsVisited = state.UnknownMapPointsVisited,
            UnknownMapPointMonsterOdds = state.UnknownMapPointMonsterOdds,
            UnknownMapPointEliteOdds = state.UnknownMapPointEliteOdds,
            UnknownMapPointTreasureOdds = state.UnknownMapPointTreasureOdds,
            UnknownMapPointShopOdds = state.UnknownMapPointShopOdds,
            LastResolvedRoomType = state.LastResolvedRoomType,
        };
    }

    /// <summary>
    /// Shuffle the draw pile into an order the agent has not been shown.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The other half of a combat determinization. <see cref="Clone(CombatState, int?)" />
    /// reseeds every stream, which changes every roll STILL TO COME, but the draw order is
    /// not a future roll -- it was materialised by a shuffle that already happened, and a
    /// reseed cannot undo it. Without this, a search that forks the position reads the true
    /// order off its own copy and plays each fight with foresight.
    /// </para>
    /// <para>
    /// Only the UNKNOWN region moves. <see cref="CombatState.KnownTopCount" /> and
    /// <see cref="CombatState.KnownBottomCount" /> exist for exactly this -- cards the
    /// player has been shown, in order, at each end of the pile -- so the shuffle runs
    /// over the middle and leaves both ends where they are. Shuffling through them
    /// resamples information the agent legitimately holds, which is not a determinization
    /// but a corruption: the engine goes on believing the top N are the cards it put
    /// there, and acts on cards that are no longer where it left them.
    /// </para>
    /// <para>
    /// Deliberately not folded into <see cref="Clone(CombatState, int?)" />: a plain clone
    /// is a faithful copy and the differential tests depend on that, seed or no seed.
    /// </para>
    /// <para>
    /// An open screen over the draw pile offers CARDS the player has been shown -- Seeker
    /// Strike's sample of three, Secret Weapon's Attacks -- at places they have not been: its
    /// candidates follow their cards through the shuffle. Left as indices, the copy offered
    /// whatever the shuffle put there (another sample, a Skill on Secret Weapon's screen).
    /// </para>
    /// </remarks>
    public static void ReshuffleUnknownDrawOrder(this CombatState combat, Random rng)
    {
        int start = combat.KnownTopCount;
        int end = combat.DrawPile.Count - combat.KnownBottomCount;
        var screen = combat.PendingSelection is { } open && open.Kind.ChoosesFromDrawPile() ? open : null;
        // Where each card now in the pile came from, for the screen's candidates.
        int[]? from = screen is null ? null : [.. Enumerable.Range(0, combat.DrawPile.Count)];
        // Fisher-Yates over [start, end), the region nothing has been shown.
        for (int i = end - 1; i > start; i--)
        {
            int j = start + rng.Next(i - start + 1);
            (combat.DrawPile[i], combat.DrawPile[j]) = (combat.DrawPile[j], combat.DrawPile[i]);
            if (from is not null)
            {
                (from[i], from[j]) = (from[j], from[i]);
            }
        }

        if (screen is not null && from is not null)
        {
            var to = new int[from.Length];
            for (int place = 0; place < from.Length; place++)
            {
                to[from[place]] = place;
            }

            var followed = screen.Candidates.Select(index => index >= 0 && index < to.Length ? to[index] : index).Order().ToList();
            if (!followed.SequenceEqual(screen.Candidates))
            {
                combat.PendingSelection = screen.Clone(screen.Replay, candidates: followed);
            }
        }
    }
}
