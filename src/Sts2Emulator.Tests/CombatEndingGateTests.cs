using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// The combat-ending gate (`CombatManager.IsOverOrEnding`, <see cref="CombatEngine.IsOverOrEnding" />):
/// once the last primary enemy is dead, or the player's death is pending, the commands the game
/// gates do nothing. Each stopped command is tested beside the same play while the fight goes on.
/// </summary>
public class CombatEndingGateTests
{
    private static readonly CardInstance[] SixDefends =
    [
        Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad),
        Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad),
    ];

    /// <summary>A fight whose draw pile is empty, so the next draw reshuffles the discard.</summary>
    private static Fight EmptyDrawPile(int enemyHp, params CardInstance[] hand)
    {
        var fight = Fight.Hand(hand).Enemy(hp: enemyHp).PlayerHp(60, 80).Draw().Discard(SixDefends);
        fight.State.ShuffleRng = new CountingRandom(0);
        return fight;
    }

    /// <summary>
    /// `CardPileCmd.Draw` returns at once when the fight is over or ending: a Pommel Strike
    /// that kills the last enemy draws nothing, so it reshuffles nothing on the run's
    /// Shuffle stream (corpus BYL0P7VTKB, BHFCKBYSXS: Rng.Shuffle ahead of the game's).
    /// </summary>
    [Theory]
    [InlineData(5, true)]
    [InlineData(50, false)]
    public void AKillingDrawCardDrawsNothing(int enemyHp, bool ends)
    {
        var fight = EmptyDrawPile(enemyHp, Card(IC.PommelStrike));

        var result = fight.Play(0);

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(ends ? 0 : 1, fight.State.Hand.Count);
        Assert.Equal(ends ? 0 : 5, fight.State.ShuffleRng!.CallCount);
    }

    /// <summary>
    /// Pillage draws until it draws a non-Attack; after the kill its first draw is refused,
    /// which ends the loop.
    /// </summary>
    [Fact]
    public void PillageDrawsNothingAfterTheKill()
    {
        var fight = EmptyDrawPile(1, Card(IC.Pillage));

        var result = fight.Play(0);

        Assert.True(result.Terminal && result.PlayerWon);
        Assert.Empty(fight.State.Hand);
        Assert.Equal(0, fight.State.ShuffleRng!.CallCount);
    }

    /// <summary>
    /// The gate is asked again before each card: Speedster's hit on the first card drawn
    /// kills, and the other two of Battle Trance's three stay in the pile.
    /// </summary>
    [Theory]
    [InlineData(5, 1)]
    [InlineData(50, 3)]
    public void ADrawStopsAtTheCardWhoseHookEndsTheFight(int enemyHp, int drawn)
    {
        var fight = Fight
            .Hand(Card(IC.BattleTrance))
            .Enemy(hp: enemyHp)
            .Draw(Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad));
        fight.PlayerBuff(BuffId.Speedster, 5);

        fight.Play(0);

        Assert.Equal(drawn, fight.State.Hand.Count);
        Assert.Equal(3 - drawn, fight.State.DrawPile.Count);
    }

    /// <summary>
    /// `CardPileCmd.Shuffle` does nothing once the fight is over: no Shuffle draw, and the
    /// discard stays where it is.
    /// </summary>
    [Theory]
    [InlineData(0, 0)]
    [InlineData(30, 5)]
    public void AShuffleOnceTheFightIsOverDoesNothing(int enemyHp, int shuffleDraws)
    {
        var fight = EmptyDrawPile(enemyHp);

        CardEffects.ShuffleDiscardIntoDraw(fight.State, new Random(0));

        Assert.Equal(shuffleDraws, fight.State.ShuffleRng!.CallCount);
        Assert.Equal(shuffleDraws == 0 ? 6 : 0, fight.State.DiscardPile.Count);
    }

    /// <summary>
    /// The player's pending death shuts the gate too (`IsEnding` with a pending loss): a dead
    /// player draws nothing. A Fairy in a Bottle or an unspent Lizard Tail still to answer the
    /// death keeps it open.
    /// </summary>
    [Fact]
    public void APendingLossShutsTheGateUnlessSomethingWillRefuseTheDeath()
    {
        var fight = Fight.Hand().Enemy(hp: 30).PlayerHp(0, 80);
        Assert.True(CombatEngine.IsOverOrEnding(fight.State));

        fight.State.Relics.Add(new RelicInstance(RelicEffects.LizardTail, 0));
        Assert.False(CombatEngine.IsOverOrEnding(fight.State));

        fight.State.Relics[0] = fight.State.Relics[0] with { Counter = 1 };
        Assert.True(CombatEngine.IsOverOrEnding(fight.State));
    }

    /// <summary>
    /// A creature whose death holds the fight open keeps the gate open until its death is
    /// answered (a Gremlin Merc's reinforcements, `ShouldStopCombatFromEnding`).
    /// </summary>
    [Fact]
    public void ADeathThatHoldsTheFightOpenKeepsTheGateOpen()
    {
        var fight = Fight.Hand().Enemy(hp: 0, buffs: new BuffState(BuffId.Surprise, 1));

        Assert.False(CombatEngine.IsOverOrEnding(fight.State));
    }

    /// <summary>
    /// Tools of the Trade's card is part of the hand draw (`ToolsOfTheTradePower.ModifyHandDraw`),
    /// so a Rolling Boulder kill in `AfterPlayerTurnStart` comes after it is dealt: six cards,
    /// the sixth reshuffling the four in the discard on the run's Shuffle stream. As a separate
    /// draw after the Boulder, the gate refused it: five cards, no reshuffle, Rng.Shuffle one
    /// shuffle behind the game's.
    /// </summary>
    [Theory]
    [InlineData(3, true)]
    [InlineData(50, false)]
    public void ToolsOfTheTradesCardIsDealtBeforeATurnStartKill(int enemyHp, bool ends)
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: enemyHp)
            .PlayerHp(60, 80)
            .Draw(SixDefends[..5])
            .Discard(SixDefends[..4]);
        fight.State.ShuffleRng = new CountingRandom(0);
        fight.PlayerBuff(BuffId.RollingBoulderPower, 5);
        fight.PlayerBuff(BuffId.ToolsOfTheTrade, 1);

        var result = fight.EndTurn();

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(6, fight.State.Hand.Count);
        Assert.Empty(fight.State.DiscardPile);
        Assert.Equal(3, fight.State.ShuffleRng!.CallCount);
    }

    /// <summary>
    /// Being the hand draw, Tools of the Trade's card is not one Fiddle refuses
    /// (`Fiddle.ShouldDraw` spares the hand draw): five, Fiddle's two and the Tools card.
    /// </summary>
    [Fact]
    public void FiddleDoesNotRefuseToolsOfTheTradesCard()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 50)
            .PlayerHp(60, 80)
            .Draw([.. SixDefends, .. SixDefends]);
        fight.State.Relics = [new RelicInstance(RelicEffects.Fiddle)];
        fight.PlayerBuff(BuffId.ToolsOfTheTrade, 1);

        fight.EndTurn();

        Assert.Equal(8, fight.State.Hand.Count);
    }

    /// <summary>
    /// Inferno's burn and Rolling Boulder are `AfterPlayerTurnStart`, which `SetupPlayerTurn`
    /// dispatches after the hand draw: a turn start whose burn kills the last enemy has
    /// already dealt the hand and reshuffled for it (corpus L80WXWJ1F5: Rng.Shuffle 16 to 28
    /// across the end turn). They ran before the draw, which the gate would have refused.
    /// </summary>
    [Theory]
    [InlineData(BuffId.InfernoSelfDamage)]
    [InlineData(BuffId.RollingBoulderPower)]
    public void ATurnStartKillComesAfterTheHandDraw(BuffId killer)
    {
        var fight = EmptyDrawPile(3);
        if (killer == BuffId.InfernoSelfDamage)
        {
            fight.PlayerBuff(BuffId.Inferno, 6);
            fight.PlayerBuff(BuffId.InfernoSelfDamage, 1);
        }
        else
        {
            fight.PlayerBuff(BuffId.RollingBoulderPower, 5);
        }

        var result = fight.EndTurn();

        Assert.True(result.Terminal && result.PlayerWon);
        Assert.Equal(5, fight.State.Hand.Count);
        Assert.Equal(5, fight.State.ShuffleRng!.CallCount);
    }

    // ── CardCmd.Exhaust and the gated hook dispatches (backlog C1-F04) ─────────────

    private static Fight InfernoTurnStart(int relicId, int enemyHp)
    {
        var fight = Fight.Hand().Enemy(hp: enemyHp).PlayerHp(999);
        fight.State.Relics = [new RelicInstance(relicId, 1)];
        fight.PlayerBuff(BuffId.Inferno, 6);
        fight.PlayerBuff(BuffId.InfernoSelfDamage, 1);
        return fight;
    }

    /// <summary>
    /// `Hook.AfterSideTurnStart` comes after `SetupPlayerTurn`'s `AfterPlayerTurnStart`, where
    /// Inferno burns: the burn's kill ends the fight before either flower ticks (run referee
    /// CGXZ45F6HL: Happy Flower 0 in the game, 1 here). The same burn that leaves the enemy
    /// standing lets it tick.
    /// </summary>
    [Theory]
    [InlineData(RelicEffects.HappyFlower, 3, true)]
    [InlineData(RelicEffects.FakeHappyFlower, 3, true)]
    [InlineData(RelicEffects.HappyFlower, 50, false)]
    [InlineData(RelicEffects.FakeHappyFlower, 50, false)]
    public void AFlowerTicksOnlyWhileTheFightGoesOn(int relicId, int enemyHp, bool ends)
    {
        var fight = InfernoTurnStart(relicId, enemyHp);

        var result = fight.EndTurn();

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(ends ? 1 : 2, fight.State.Relics[0].Counter);
    }

    /// <summary>
    /// Crossbow and Seal of Gold are `AfterSideTurnStart` relics too: after the turn start's
    /// kill Crossbow rolls nothing on the run's CombatCardGeneration stream and Seal of Gold
    /// takes no gold. While the fight goes on, both run, after the hand draw.
    /// </summary>
    [Theory]
    [InlineData(3, true)]
    [InlineData(50, false)]
    public void CrossbowAndSealOfGoldRunOnlyWhileTheFightGoesOn(int enemyHp, bool ends)
    {
        var fight = InfernoTurnStart(RelicEffects.Crossbow, enemyHp);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.SealOfGold));
        fight.State.PlayerGold = 20;
        int generated = fight.State.CardGenerationRng!.CallCount;

        var result = fight.EndTurn();

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(ends ? 20 : 15, fight.State.PlayerGold);
        if (ends)
        {
            Assert.Equal(generated, fight.State.CardGenerationRng!.CallCount);
        }
        else
        {
            Assert.True(fight.State.CardGenerationRng!.CallCount > generated);
            Assert.True(fight.State.Hand[^1].FreeThisTurn);
        }
    }

    /// <summary>
    /// The player's own `AfterSideTurnStart` powers are the same gated dispatch: after the
    /// burn's kill the poison does not tick (player HP outlives the fight) and Countdown rolls
    /// no target on the run's CombatTargets stream.
    /// </summary>
    [Theory]
    [InlineData(3, true)]
    [InlineData(50, false)]
    public void PoisonAndCountdownRunOnlyWhileTheFightGoesOn(int enemyHp, bool ends)
    {
        var fight = InfernoTurnStart(RelicEffects.HappyFlower, enemyHp);
        fight.PlayerBuff(BuffId.Poison, 4);
        fight.PlayerBuff(BuffId.Countdown, 5);
        int targets = fight.State.TargetRng!.CallCount;

        var result = fight.EndTurn();

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(ends ? 998 : 994, fight.State.PlayerHp);
        Assert.Equal(ends, fight.State.TargetRng!.CallCount == targets);
    }

    /// <summary>
    /// True Grit's block sets off Juggernaut, whose hit kills; the random card it then exhausts
    /// is still picked off CombatCardSelection, but `CardCmd.Exhaust` does nothing, so Joss
    /// Paper does not count it (run referee NQH1R9PA6V: 3 in the game, 4 here). While the fight
    /// goes on, it counts.
    /// </summary>
    [Theory]
    [InlineData(2, true)]
    [InlineData(50, false)]
    public void TheExhaustAfterTheKillingHitIsNotCounted(int enemyHp, bool ends)
    {
        var fight = Fight.Hand(Card(IC.TrueGrit), Card(IC.StrikeIronclad)).Enemy(hp: enemyHp);
        fight.State.Relics = [new RelicInstance(RelicEffects.JossPaper, 3)];
        fight.PlayerBuff(BuffId.Juggernaut, 6);
        int selections = fight.State.CardSelectionRng!.CallCount;

        var result = fight.Play(0);

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(ends ? 3 : 4, fight.State.Relics[0].Counter);
        Assert.Equal(selections + 1, fight.State.CardSelectionRng!.CallCount);
    }

    /// <summary>
    /// A killing card with Exhaust goes to its result pile through `CardCmd.Exhaust` after the
    /// play, so it is not counted either.
    /// </summary>
    [Fact]
    public void AKillingCardThatExhaustsItselfIsNotCounted()
    {
        var fight = Fight.Hand(Card(IC.Feed)).Enemy(hp: 5);
        fight.State.Relics = [new RelicInstance(RelicEffects.JossPaper, 3)];

        var result = fight.Play(0);

        Assert.True(result.Terminal && result.PlayerWon);
        Assert.Equal(3, fight.State.Relics[0].Counter);
    }

    /// <summary>
    /// Forgotten Soul's hit is an exhaust hook: after the leader's death it rolls no target on
    /// the run's CombatTargets stream, though a minion is still standing.
    /// </summary>
    [Fact]
    public void NoExhaustHookRollsAfterTheKill()
    {
        var fight = Fight
            .Hand(Card(IC.Feed))
            .Enemy(hp: 5)
            .Enemy(hp: 20, buffs: new BuffState(BuffId.Minion, 1));
        fight.State.Relics = [new RelicInstance(RelicEffects.ForgottenSoul)];
        var minion = fight.Enemy1;
        int targets = fight.State.TargetRng!.CallCount;

        var result = fight.Play(0, target: 0);

        Assert.True(result.Terminal && result.PlayerWon);
        Assert.Equal(targets, fight.State.TargetRng!.CallCount);
        Assert.Equal(20, minion.Hp);
    }

    /// <summary>
    /// Ethereal exhausts at the turn's end: the first one's Charon's Ashes hit kills, the second
    /// exhaust does nothing, and `Hook.AfterTurnEnd` -- where Joss Paper folds its banked
    /// Ethereal count in -- never runs. `AfterCombatEnd` drops the bank.
    /// </summary>
    [Fact]
    public void EtherealExhaustsBankedBeforeATurnEndKillAreDropped()
    {
        var fight = Fight.Hand(Card(ST.Dazed), Card(ST.Dazed)).Enemy(hp: 3);
        fight.State.Relics =
        [
            new RelicInstance(RelicEffects.JossPaper, 0),
            new RelicInstance(RelicEffects.CharonsAshes),
        ];

        var result = fight.EndTurn();

        Assert.True(result.Terminal && result.PlayerWon);
        Assert.Equal(0, fight.State.Relics[0].Counter);
    }

    /// <summary>
    /// A kill in the end of turn's first phase (an Ethereal exhaust's Charon's Ashes hit) ends
    /// the fight at `EndPlayerTurnPhaseOneInternal`'s `CheckWinCondition`: no flush, no
    /// `AfterFlush`, no `AfterTurnEnd`, so Constrict takes no HP and Bookmark rolls no
    /// CombatCardSelection.
    /// </summary>
    [Theory]
    [InlineData(3, true)]
    [InlineData(50, false)]
    public void NothingAfterTheEndOfTurnsFirstPhaseRunsOnceItKills(int enemyHp, bool ends)
    {
        var fight = Fight
            .Hand(Card(ST.Dazed), Card(IC.StrikeIronclad))
            .Enemy(hp: enemyHp)
            .PlayerHp(50);
        fight.State.Relics =
        [
            new RelicInstance(RelicEffects.CharonsAshes),
            new RelicInstance(RelicEffects.Bookmark),
        ];
        fight.PlayerBuff(BuffId.Constrict, 6);
        fight.PlayerBuff(BuffId.RetainHand, 1);
        int selections = fight.State.CardSelectionRng!.CallCount;

        var result = fight.EndTurn();

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        if (ends)
        {
            Assert.Equal(50, fight.State.PlayerHp);
            Assert.Equal(selections, fight.State.CardSelectionRng!.CallCount);
        }
        else
        {
            Assert.True(fight.State.PlayerHp <= 44);
            Assert.True(fight.State.CardSelectionRng!.CallCount > selections);
        }
    }

    /// <summary>
    /// The first phase's kill is still answered before the fight ends: `Hook.BeforeDeath` is not
    /// gated, so a Thieving Hopper killed by the Ethereal exhaust's Charon's Ashes hit gives its
    /// stolen card back (`SwipePower.BeforeDeath`), and a Fat Gremlin its Heist gold
    /// (`HeistPower.BeforeDeath`). Ending the turn there skipped the enemy phase's answer, and
    /// both were lost to the run.
    /// </summary>
    [Theory]
    [InlineData(KE.ThievingHopper)]
    [InlineData(KE.FatGremlin)]
    public void AFirstPhaseKillStillReturnsWhatTheEnemyStole(int thief)
    {
        var fight = Fight.Hand(Card(ST.Dazed)).Enemy(hp: 2, defId: thief).PlayerHp(60, 80);
        fight.State.Relics = [new RelicInstance(RelicEffects.CharonsAshes)];
        var stolen = Card(IC.StrikeIronclad, upgraded: true);
        if (thief == KE.ThievingHopper)
        {
            fight.Enemy0.StolenCard = stolen;
        }
        else
        {
            fight.Enemy0.HeistGold = 40;
        }

        var result = fight.EndTurn();

        Assert.True(result.Terminal && result.PlayerWon);
        if (thief == KE.ThievingHopper)
        {
            Assert.Contains(fight.State.StolenBackCards, card => card.DefId == stolen.DefId);
        }
        else
        {
            Assert.Equal(40, fight.State.StolenBackGold);
        }
    }

    /// <summary>
    /// An enemy's Doom kills at the end of its OWNER's side turn (`DoomPower.BeforeSideTurnEnd`,
    /// FINDINGS 4W1BFLMWVJ), so in the game a doomed last enemy is alive through the player's
    /// `Hook.AfterTurnEnd`: Constrict hits, Joss Paper folds the Ethereal exhaust in and the
    /// Late pass's Disintegration hits, lethal Doom or not. The emulator's Doom kill at the
    /// player's turn end shut both gates and took all three away.
    /// </summary>
    [Theory]
    [InlineData(10, true)]
    [InlineData(4, false)]
    public void ADoomedLastEnemyLeavesThePlayersAfterTurnEndOpen(int doom, bool ends)
    {
        var fight = Fight
            .Hand(Card(ST.Dazed))
            .Enemy(hp: 5, buffs: new BuffState(BuffId.Doom, doom))
            .PlayerHp(50, 80);
        fight.State.Relics = [new RelicInstance(RelicEffects.JossPaper, 0)];
        fight.PlayerBuff(BuffId.Constrict, 3);
        fight.PlayerBuff(BuffId.Disintegration, 2);

        var result = fight.EndTurn();

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(1, fight.State.Relics[0].Counter);
        Assert.Equal(45, fight.State.PlayerHp);
    }

    /// <summary>
    /// The Doom kill asks the gate itself (`!IsOverOrEnding`), after the player's
    /// `AfterTurnEnd`: a Constrict that kills the player ends the fight before it, so Book
    /// Repair Knife's heal for the doomed enemy never comes and the fight is lost. With the
    /// Doom killing first, the heal stood the player back up to take the hit and live.
    /// </summary>
    [Fact]
    public void AConstrictDeathComesBeforeAnEnemysDoom()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 5, buffs: new BuffState(BuffId.Doom, 10))
            .Enemy(hp: 50)
            .PlayerHp(5, 80);
        fight.State.Relics = [new RelicInstance(RelicEffects.BookRepairKnife)];
        fight.PlayerBuff(BuffId.Constrict, 5);

        var result = fight.EndTurn();

        Assert.True(result.Terminal);
        Assert.False(result.PlayerWon);
        Assert.Equal(0, fight.State.PlayerHp);
        Assert.Equal(5, fight.Enemy0.Hp);
    }

    // ── A pending loss comes first (the C1-F04 rejection) ──────────────────────────

    /// <summary>
    /// Lethal Doom on the player kills in `BeforeTurnEnd`'s BeforeSideTurnEnd pass, before
    /// `DoTurnEnd` exhausts the Ethereal cards, and the turn's `CheckWinCondition` takes the
    /// pending loss first: the Charon's Ashes hit that would have killed the last enemy never
    /// comes, and the fight is lost. With the Doom one short of lethal, the exhaust kills and the
    /// fight is won.
    /// </summary>
    [Theory]
    [InlineData(10, false)]
    [InlineData(9, true)]
    public void LethalDoomLosesBeforeAnEtherealExhaustCanWin(int doom, bool won)
    {
        var fight = Fight.Hand(Card(ST.Dazed)).Enemy(hp: 3).PlayerHp(10, 80);
        fight.State.Relics = [new RelicInstance(RelicEffects.CharonsAshes)];
        fight.PlayerBuff(BuffId.Doom, doom);
        // Held by reference: the killed enemy's corpse is swept as its death is answered.
        var enemy = fight.Enemy0;

        var result = fight.EndTurn();

        Assert.True(result.Terminal);
        Assert.Equal(won, result.PlayerWon);
        Assert.Equal(won ? 0 : 3, enemy.Hp);
        Assert.Equal(won ? 10 : 0, fight.State.PlayerHp);
    }

    /// <summary>
    /// Inside the one pass the player's powers go in the order they were applied, and Doom asks
    /// the gate itself: a Bomb applied ahead of it that kills the last enemy spares the player
    /// (won), while a Doom applied first kills before the Bomb goes off, and the Bomb's kill
    /// comes after the pending loss (lost).
    /// </summary>
    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void TheBombAheadOfLethalDoomWinsAndBehindItLoses(bool bombFirst, bool won)
    {
        var fight = Fight.Hand().Enemy(hp: 20).PlayerHp(10, 80);
        if (bombFirst)
        {
            fight.PlayerBuff(BuffId.TheBombPower, 1, data: 40);
            fight.PlayerBuff(BuffId.Doom, 10);
        }
        else
        {
            fight.PlayerBuff(BuffId.Doom, 10);
            fight.PlayerBuff(BuffId.TheBombPower, 1, data: 40);
        }

        var result = fight.EndTurn();

        Assert.True(result.Terminal);
        Assert.Equal(won, result.PlayerWon);
        Assert.Equal(0, fight.Enemy0.Hp);
    }

    /// <summary>
    /// `DoomKill` is a `CreatureCmd.Kill` without force, so a Fairy in a Bottle refuses it and
    /// the fight goes on. It used to set the player to 0 and end the fight as lost.
    /// </summary>
    [Fact]
    public void AFairyInABottleRefusesTheDoom()
    {
        var fight = Fight.Hand().Enemy(hp: 30).PlayerHp(10, 80);
        int fairy = Sts2Emulator.GeneratedData.Potions.FindId("FairyInABottle")!.Value;
        fight.State.PotionSlots[0] = fairy;
        fight.PlayerBuff(BuffId.Doom, 10);

        var result = fight.EndTurn();

        Assert.False(result.Terminal);
        Assert.Equal(0, fight.State.PotionSlots[0]);
        Assert.True(fight.State.PlayerHp > 0);
    }

    // ── The other gated commands ────────────────────────────────────────────────────

    /// <summary>
    /// The leader dead and its minion still standing, as the emulator holds it until the death
    /// is answered: in the game the fight is already ending (the minion is a secondary enemy).
    /// </summary>
    private static Fight LeaderDeadMinionStanding(bool leaderAlive = false)
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: leaderAlive ? 30 : 0)
            .Enemy(hp: 20, buffs: new BuffState(BuffId.Minion, 1))
            .PlayerHp(60, 80);
        return fight;
    }

    /// <summary>
    /// `CreatureCmd.GainBlock` returns 0 once the fight is ending, before `AfterBlockGained`:
    /// no block, and no Juggernaut hit rolling CombatTargets over the minion.
    /// </summary>
    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 5)]
    public void ABlockGainedAfterTheKillSetsOffNoJuggernaut(bool leaderAlive, int block)
    {
        var fight = LeaderDeadMinionStanding(leaderAlive);
        fight.PlayerBuff(BuffId.Juggernaut, 6);
        int targets = fight.State.TargetRng!.CallCount;

        int gained = CardEffects.GainUnpoweredBlock(fight.State, 5, new Random(0));

        Assert.Equal(block, gained);
        Assert.Equal(block, fight.State.PlayerBlock);
        Assert.Equal(leaderAlive, fight.State.TargetRng!.CallCount > targets);
    }

    /// <summary>
    /// `OrbCmd.Passive`, `OrbCmd.Evoke` and `OrbCmd.Channel` do nothing once the fight is ending:
    /// a Lightning rolls no target, the queue keeps its orb, nothing is channeled.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NoOrbCommandRunsAfterTheKill(bool leaderAlive)
    {
        var fight = LeaderDeadMinionStanding(leaderAlive);
        fight.State.OrbCapacity = 1;
        fight.State.Orbs.Add(new OrbState(OrbType.Lightning));
        int targets = fight.State.TargetRng!.CallCount;

        CardEffects.TriggerOrbPassive(fight.State, 0, new Random(0));
        int afterPassive = fight.State.TargetRng!.CallCount;
        CardEffects.ChannelOrb(fight.State, OrbType.Frost, new Random(0));

        Assert.Equal(leaderAlive, afterPassive > targets);
        Assert.Equal(leaderAlive ? OrbType.Frost : OrbType.Lightning, fight.State.Orbs[0].Type);
        Assert.Equal(leaderAlive, fight.State.TargetRng!.CallCount > afterPassive);
    }

    /// <summary>
    /// `Hook.AfterCardDiscarded` is a gated dispatch: Tingsha's hit rolls no target after the
    /// kill.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TingshaRollsNothingAfterTheKill(bool leaderAlive)
    {
        var fight = LeaderDeadMinionStanding(leaderAlive);
        fight.State.Relics = [new RelicInstance(RelicEffects.Tingsha)];
        int targets = fight.State.TargetRng!.CallCount;

        RelicEffects.ApplyAfterCardDiscarded(fight.State, new Random(0));

        Assert.Equal(leaderAlive, fight.State.TargetRng!.CallCount > targets);
    }

    /// <summary>
    /// `StampedePower` picks once per stack off the run's Shuffle stream whatever the picks
    /// before did: after its first Strike kills, `CardCmd.AutoPlay` does nothing and the second
    /// Strike stays in hand, but the second pick is still drawn. It stopped at the kill here.
    /// </summary>
    [Fact]
    public void StampedeStillPicksAfterItsFirstAutoPlayKills()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad))
            .Enemy(hp: 6)
            .PlayerHp(60, 80);
        fight.State.ShuffleRng = new CountingRandom(0);
        fight.PlayerBuff(BuffId.Stampede, 2);

        var result = fight.EndTurn();

        Assert.True(result.Terminal && result.PlayerWon);
        Assert.Equal(2, fight.State.ShuffleRng!.CallCount);
        Assert.Single(fight.State.Hand);
    }

    /// <summary>
    /// `CombatManager.ExecuteEnemyTurn` checks the win condition after every enemy: once the
    /// leader's own poison has killed it, its minion (secondary, so dying with it) does not
    /// act. It attacked here before the end-of-phase death answer swept it.
    /// </summary>
    [Theory]
    [InlineData(2, 60, true)]
    [InlineData(50, 50, false)]
    public void NoEnemyActsOnceTheFightIsOver(int leaderHp, int playerHpAfter, bool ends)
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: leaderHp, buffs: new BuffState(BuffId.Poison, 5))
            .Enemy(
                hp: 20,
                intent: new Intent(IntentType.Attack, 10),
                buffs: new BuffState(BuffId.Minion, 1)
            )
            .PlayerHp(60, 80);

        var result = fight.EndTurn();

        Assert.Equal(ends, result.Terminal && result.PlayerWon);
        Assert.Equal(playerHpAfter, fight.State.PlayerHp);
    }
}
