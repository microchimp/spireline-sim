using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// A screen raised in the turn start's setup is picked after the rest of the turn start, as a player
/// using the UI picks it (owner decision, 2026-10-14). `CombatManager.StartTurn` awaits
/// `SetupPlayerTurn` only until it completes or pauses on a choice, then runs on --
/// `Hook.AfterSideTurnStart`, the orbs, the win check -- and the screen shows once its hook action
/// starts. The game settles it (referee/probes.py P19, P20, and pick_timing "ui" recordings of
/// Toolbox, Gambling Chip, Choices Paradox and Tyranny): the referee's old picker answered at once,
/// and the emulator followed it.
/// </summary>
public class TurnStartPickTimingTests
{
    private static Fight Foregone(BuffId other, int amount) =>
        Fight.Hand(Card(IC.StrikeIronclad))
            .Energy(3)
            .Draw([.. Pile(IC.Bash, IC.DefendIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Anger, IC.Inflame, IC.TwinStrike, IC.ShrugItOff, IC.PommelStrike, IC.IronWave)])
            .Enemy(hp: 400)
            .PlayerBuff(BuffId.ForegoneConclusion, 1)
            .PlayerBuff(other, amount);

    /// <summary>
    /// P19: Foregone Conclusion asks in `BeforeHandDraw`, so the side's turn start runs while it waits
    /// -- before the hand draw -- and Draw Cards Next Turn is removed before `ModifyHandDraw` reads it:
    /// one pick and five cards, not seven.
    /// </summary>
    [Fact]
    public void ABeforeHandDrawScreenLetsTheSidesTurnStartRemoveTheNextTurnDraw()
    {
        var fight = Foregone(BuffId.NextTurnDraw, 2);

        fight.EndTurn();
        Assert.Equal(204, fight.Pending!.SourceCardDefId);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.NextTurnDraw));
        fight.Choose(0);

        Assert.Null(fight.Pending);
        Assert.Equal(1 + 5, fight.State.Hand.Count);
    }

    /// <summary>P20: and Clarity's last turn ticks off before the draw reads it: one pick and five.</summary>
    [Fact]
    public void ABeforeHandDrawScreenLetsClarityTickOffBeforeTheDraw()
    {
        var fight = Foregone(BuffId.Clarity, 1);

        fight.EndTurn();
        fight.Choose(0);

        Assert.Equal(1 + 5, fight.State.Hand.Count);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Clarity));
    }

    /// <summary>
    /// The player's Poison ticks in `AfterSideTurnStart`: with Tyranny's screen open it has already
    /// ticked, and a tick that kills ends the fight with the screen never asked.
    /// </summary>
    [Fact]
    public void TheSidesTurnStartRunsBeforeTheScreenIsAskedAndCanEndTheFight()
    {
        static Fight Tyranny(int hp) =>
            Fight.Hand(Card(IC.StrikeIronclad))
                .Energy(3)
                .Draw([.. Pile(IC.Bash, IC.DefendIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Anger, IC.Inflame)])
                .Enemy(hp: 400)
                .PlayerHp(hp)
                .PlayerBuff(BuffId.Tyranny, 1)
                .PlayerBuff(BuffId.Poison, 3);

        var alive = Tyranny(50);
        alive.EndTurn();
        Assert.NotNull(alive.Pending);
        Assert.Equal(50 - 3, alive.State.PlayerHp);
        Assert.Equal(2, alive.PlayerBuffAmount(BuffId.Poison));

        var dying = Tyranny(3);
        var result = dying.EndTurn();
        Assert.True(result.Terminal);
        Assert.False(result.PlayerWon);
        Assert.Null(dying.Pending);
        Assert.Empty(dying.State.ExhaustPile);
    }

    /// <summary>
    /// The screen is built over the hand as it stands once the side's turn start has run: Furnace's
    /// Forge puts a Sovereign Blade in hand before Tyranny's screen shows, and the blade is offered
    /// (the game's pick_timing "ui" recording lists it among Tyranny's options).
    /// </summary>
    [Fact]
    public void AScreenOverTheHandOffersWhatTheSidesTurnStartPutThere()
    {
        var fight = Fight.Hand(Card(IC.StrikeIronclad))
            .Energy(3)
            .Draw([.. Pile(IC.Bash, IC.DefendIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Anger, IC.Inflame)])
            .Enemy(hp: 400)
            .PlayerBuff(BuffId.Tyranny, 1)
            .PlayerBuff(BuffId.Furnace, 1);

        fight.EndTurn();

        Assert.Equal(fight.State.Hand.Count, fight.Pending!.Candidates.Count);
        Assert.Contains(fight.State.Hand, card => card.DefId == 448); // Sovereign Blade
    }

    /// <summary>A combat dealt with these relics, Bone Tea charged for this fight as the run hands it over.</summary>
    private static (CombatState State, CountingRandom Rng) TurnOne(params int[] relics)
    {
        var state = new CombatState { NicheHpRng = new CountingRandom(0) };
        var rng = new CountingRandom(0);
        CombatFactory.Reset(
            state,
            rng,
            [.. StarterDeckIds.ToArray().Select(id => new CardInstance(id, false))],
            1,
            relics,
            80,
            80,
            [],
            99,
            runRelics: [.. relics.Select(id => new RelicInstance(id, id == RelicEffects.BoneTea ? 1 : 0))]
        );
        return (state, rng);
    }

    /// <summary>
    /// Turn one: Gambling Chip's screen holds the turn, and the rest of turn one's start runs first.
    /// Bone Tea upgrades the drawn hand before the pick, so the card pitched was upgraded and the
    /// one drawn for it is not (the game: four upgraded, one not; the old picker: all five).
    /// </summary>
    [Fact]
    public void TurnOnesSideTurnStartRunsBeforeTheChipsPick()
    {
        var (state, rng) = TurnOne(RelicEffects.BoneTea, RelicEffects.GamblingChip);

        Assert.Equal(CardSelectionKind.DiscardAnyThenDraw, state.PendingSelection!.Kind);
        Assert.All(state.Hand, card => Assert.True(card.Upgraded));
        CombatEngine.Step(state, 0, rng);
        CombatEngine.Step(state, state.PendingSelection!.Candidates.Count, rng);

        Assert.Null(state.PendingSelection);
        Assert.Equal(4, state.Hand.Count(card => card.Upgraded));
        Assert.False(state.Hand[^1].Upgraded);
        Assert.True(state.DiscardPile[^1].Upgraded);
    }

    /// <summary>
    /// Gambling Chip's held screen is shown over the hand as it stands once turn one's side has run:
    /// Orange Dough's two colourless cards and Fencing Manual's Sovereign Blade reach the hand while
    /// the chip waits (`AfterSideTurnStart`, TurnNumber &lt;= 1), and the chip offers them, as the UI
    /// does (it selects from the live hand). The screen offered the five cards it counted when it
    /// paused, so the added cards could not be pitched.
    /// </summary>
    [Theory]
    [InlineData(RelicEffects.OrangeDough, 2)]
    [InlineData(RelicEffects.FencingManual, 1)]
    public void TurnOnesHeldChipScreenOffersTheCardsTheSideAdded(int relic, int added)
    {
        var (state, rng) = TurnOne(RelicEffects.GamblingChip, relic);

        var open = state.PendingSelection!;
        Assert.Equal(CardSelectionKind.DiscardAnyThenDraw, open.Kind);
        Assert.Equal(5 + added, state.Hand.Count);
        Assert.Equal(Enumerable.Range(0, state.Hand.Count), open.Candidates);

        // Pitching the last card added, then closing the screen, draws one for it.
        var pitched = state.Hand[^1];
        CombatEngine.Step(state, open.Candidates.Count - 1, rng);
        CombatEngine.Step(state, state.PendingSelection!.Candidates.Count, rng);

        Assert.Null(state.PendingSelection);
        Assert.Equal(pitched, state.DiscardPile[^1]);
        Assert.Equal(5 + added, state.Hand.Count);
    }

    /// <summary>
    /// And Toolbox asks before the opening draw: Bone Tea's turn-one upgrade finds an empty hand, and
    /// nothing in the opening hand is upgraded (the game; the old picker upgraded all six).
    /// </summary>
    [Fact]
    public void BeforeToolboxsPickBoneTeaUpgradesAHandNotYetDrawn()
    {
        var (state, rng) = TurnOne(RelicEffects.BoneTea, RelicEffects.Toolbox);

        Assert.Empty(state.Hand);
        CombatEngine.Step(state, 0, rng);

        Assert.Null(state.PendingSelection);
        Assert.Equal(6, state.Hand.Count);
        Assert.DoesNotContain(state.Hand, card => card.Upgraded);
        Assert.Equal(0, state.Relics.Single(relic => relic.DefId == RelicEffects.BoneTea).Counter);
    }
}
