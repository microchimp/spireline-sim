using System;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// Turn one is a turn like any other: `CombatManager.StartCombatInternal` runs
/// `BeforeCombatStart` and then the same `StartTurn` every later turn runs. Its relic hooks
/// fire in the order the player holds the relics (`CombatState.IterateHookListeners`), and a
/// screen one raises holds the rest of the turn start until it is answered.
/// </summary>
public class TurnOneStartTests
{
    /// <summary>
    /// `RampartPower.AfterSideTurnStart` fires on the player side's turn start, turn one too:
    /// the game's Turret Operator opens on 25 block (XKSB75348Q, HVL23QXSWR); the emulator's
    /// opened on none, because only `EndTurn` granted it.
    /// </summary>
    [Fact]
    public void TheShieldArmoursTheTurretOnTurnOne()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.TurretOperator, 8);

        Assert.Equal(25, fight.State.Enemies.Single(e => e.DefId == KE.TurretOperator).Block);
        Assert.Equal(0, fight.State.Enemies.Single(e => e.DefId == KE.LivingShield).Block);
    }

    /// <summary>
    /// `MercuryHourglass.AfterPlayerTurnStart` has no turn guard, so it hits on turn one
    /// (U0TNXQZM08: the enemy is 3 lower in the game at step 0).
    /// </summary>
    [Fact]
    public void MercuryHourglassHitsOnTurnOne()
    {
        var plain = Fight.WithRelics();
        var hourglass = Fight.WithRelics(RelicEffects.MercuryHourglass);

        Assert.Equal(
            plain.State.Enemies.Select(e => e.Hp - 3),
            hourglass.State.Enemies.Select(e => e.Hp)
        );
    }

    /// <summary>
    /// A Gambling Chip held BEFORE a Vexing Puzzlebox asks over the five drawn cards, and the
    /// Puzzlebox card arrives after the pitch and redraw (6K928H3WXL). The emulator always ran
    /// the Puzzlebox first.
    /// </summary>
    [Fact]
    public void AChipHeldFirstAsksBeforeThePuzzleboxCardArrives()
    {
        var fight = Fight.WithRelics(RelicEffects.GamblingChip, RelicEffects.VexingPuzzlebox);

        Assert.Equal(CardSelectionKind.DiscardAnyThenDraw, fight.Pending!.Kind);
        Assert.Equal(5, fight.State.Hand.Count);
        var drawn = fight.State.Hand.Select(c => c.DefId).ToList();

        fight.Choose(0);
        fight.Choose(fight.Pending!.Candidates.Count); // decline the rest: draws one

        Assert.Null(fight.Pending);
        Assert.Equal(6, fight.State.Hand.Count);
        Assert.Equal(drawn.Skip(1), fight.State.Hand.Take(4).Select(c => c.DefId));
        Assert.True(fight.State.Hand[^1].FreeThisTurn, "the Puzzlebox card comes last");
        Assert.False(fight.State.Hand[^2].FreeThisTurn, "the redraw comes before it");
    }

    /// <summary>And held the other way round, the Puzzlebox card is in hand for the Chip.</summary>
    [Fact]
    public void APuzzleboxHeldFirstPutsItsCardUnderTheChip()
    {
        var fight = Fight.WithRelics(RelicEffects.VexingPuzzlebox, RelicEffects.GamblingChip);

        Assert.Equal(CardSelectionKind.DiscardAnyThenDraw, fight.Pending!.Kind);
        Assert.Equal(6, fight.Pending.Candidates.Count);
        Assert.True(fight.State.Hand[^1].FreeThisTurn);
    }

    /// <summary>
    /// Bellows upgrades the hand as it stands when its turn comes: a Puzzlebox held before it
    /// has its card upgraded, one held after it does not.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void BellowsUpgradesWhatTheRelicsBeforeItAdded(bool puzzleboxFirst)
    {
        int[] relics = puzzleboxFirst
            ? [RelicEffects.VexingPuzzlebox, RelicEffects.Bellows]
            : [RelicEffects.Bellows, RelicEffects.VexingPuzzlebox];
        var fight = Fight.WithRelics(relics);

        var puzzle = fight.State.Hand.Single(c => c.FreeThisTurn);
        Assert.Equal(puzzleboxFirst, puzzle.Upgraded);
    }

    /// <summary>
    /// Toolbox (`BeforeHandDraw`) and Choices Paradox (`AfterPlayerTurnStart`) both ask on turn
    /// one, one after the other. Choices Paradox's screen used to replace the Toolbox one that
    /// was still standing, and the Toolbox pick was lost.
    /// </summary>
    [Fact]
    public void ToolboxAndChoicesParadoxBothAsk()
    {
        var fight = Fight.WithRelics(RelicEffects.Toolbox, RelicEffects.ChoicesParadox);

        Assert.Equal(RelicEffects.Toolbox, fight.Pending!.SourceCardDefId);
        int tool = fight.Pending.GeneratedCandidates[0];
        fight.Choose(0);

        Assert.True(fight.Pending!.GeneratedRetained, "Choices Paradox asks next");
        int paradox = fight.Pending.GeneratedCandidates[0];
        fight.Choose(0);

        Assert.Null(fight.Pending);
        Assert.Equal(tool, fight.State.Hand[0].DefId);
        Assert.Equal(paradox, fight.State.Hand[^1].DefId);
    }

    /// <summary>
    /// Nothing plays itself while a turn-one screen holds the turn start
    /// (`RunAutoPrePlayPhase` awaits `SetupPlayerTurn`), and the turn-one hooks after the
    /// screen run once it is answered: the Hourglass hits after the Toolbox pick.
    /// </summary>
    [Fact]
    public void TheRestOfTurnOneWaitsForTheScreen()
    {
        var fight = Fight.WithRelics(RelicEffects.Toolbox, RelicEffects.MercuryHourglass);
        var full = Fight.WithRelics().State.Enemies.Select(e => e.Hp).ToList();

        Assert.Equal(full, fight.State.Enemies.Select(e => e.Hp));
        CombatEngine.ResolvePrePlayPhase(fight.State, new Random(0));
        Assert.NotNull(fight.Pending);

        fight.Choose(0);

        Assert.Equal(full.Select(hp => hp - 3), fight.State.Enemies.Select(e => e.Hp));
    }

    /// <summary>
    /// `Pendulum.AfterPlayerTurnStart` draws AFTER the hand draw on every third turn, so its
    /// card comes after the cards `BeforeHandDraw` put in the hand. It drew before the hand
    /// draw, ahead of Infinite Blades' Shiv.
    /// </summary>
    [Fact]
    public void PendulumDrawsAfterTheHandDraw()
    {
        var fight = Fight.WithRelics(RelicEffects.Pendulum);
        fight.State.PlayerHp = 9999;
        fight.EndTurn();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.InfiniteBlades, 1);

        fight.EndTurn(); // turn three: Pendulum's draw

        Assert.Equal(SI.Shiv, fight.State.Hand[0].DefId);
        Assert.Equal(7, fight.State.Hand.Count);
    }

    /// <summary>
    /// `BloodVial` heals in `AfterPlayerTurnStartLate`, after Royal Poison's bite in the
    /// ordinary pass: from full health the player ends 2 down, not 4.
    /// </summary>
    [Fact]
    public void BloodVialHealsAfterRoyalPoisonBites()
    {
        var state = new CombatState { NicheHpRng = new CountingRandom(0) };

        CombatFactory.Reset(
            state,
            new Random(0),
            StarterDeckIds,
            1,
            [RelicEffects.BloodVial, RelicEffects.RoyalPoison],
            playerHp: 80,
            playerMaxHp: 80
        );

        Assert.Equal(78, state.PlayerHp);
    }

    /// <summary>
    /// `FurCoat.BeforeCombatStart` pins the roster before turn one starts, so the Hourglass's
    /// turn-one hit lands on 1 HP.
    /// </summary>
    [Fact]
    public void FurCoatBitesBeforeTurnOne()
    {
        var state = new CombatState { NicheHpRng = new CountingRandom(0), FurCoatBites = true };

        CombatFactory.Reset(state, new Random(0), StarterDeckIds, 1, [RelicEffects.MercuryHourglass]);

        Assert.All(state.Enemies, e => Assert.True(e.Hp <= 0));
    }

    /// <summary>
    /// `TeaOfDiscourtesy.BeforeCombatStart` buries its Dazed before turn one starts, so the
    /// opening hand can draw them: three cards and two Dazed are a hand of five.
    /// </summary>
    [Fact]
    public void TeaOfDiscourtesysDazedCanOpenTheHand()
    {
        var state = new CombatState { NicheHpRng = new CountingRandom(0) };
        CardInstance[] deck = [.. Enumerable.Repeat(new CardInstance(StarterDeckIds[0], false), 3)];

        CombatFactory.Reset(
            state,
            new Random(0),
            deck,
            1,
            [RelicEffects.TeaOfDiscourtesy],
            80,
            80,
            [],
            0,
            runRelics: [new RelicInstance(RelicEffects.TeaOfDiscourtesy, 1)]
        );

        Assert.Equal(5, state.Hand.Count);
        Assert.Equal(2, state.Hand.Count(c => c.DefId == ST.Dazed));
    }

    /// <summary>
    /// `PowerCell.BeforeSideTurnStart` moves its two zero-cost cards before `SetupPlayerTurn`
    /// draws: they open the hand and the five drawn follow them.
    /// </summary>
    [Fact]
    public void PowerCellsCardsAreInHandBeforeTheDraw()
    {
        var state = new CombatState { NicheHpRng = new CountingRandom(0) };
        int[] deck = [IC.Anger, IC.Anger, .. Enumerable.Repeat(StarterDeckIds[0], 8)];

        CombatFactory.Reset(state, new Random(0), deck, 1, [RelicEffects.PowerCell]);

        Assert.Equal(7, state.Hand.Count);
        Assert.Equal([IC.Anger, IC.Anger], state.Hand.Take(2).Select(c => c.DefId));
    }

    /// <summary>
    /// The opening hand is a hand draw like any other (`CardPileCmd.Draw(fromHandDraw: true)`),
    /// so it stops at a full hand: ten, with Ninja Scroll's Shivs already in it.
    /// </summary>
    [Fact]
    public void TheOpeningHandStopsAtTen()
    {
        var state = new CombatState { NicheHpRng = new CountingRandom(0) };
        var deck = Enumerable.Repeat(StarterDeckIds[0], 12).ToArray();

        CombatFactory.Reset(
            state,
            new Random(0),
            deck,
            1,
            [RelicEffects.NinjaScroll, RelicEffects.RingOfTheSnake, RelicEffects.RadiantPearl]
        );

        Assert.Equal(10, state.Hand.Count);
    }
}
