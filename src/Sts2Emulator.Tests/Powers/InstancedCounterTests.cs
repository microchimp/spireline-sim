using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// Toric Toughness, Orbit, Panache and Monologue (and Automation, in AutomationTests) are
/// <c>PowerInstanceType.Instanced</c>: every play is a power of its own with its own Amount
/// AND its own counter, and the hook runs once per instance, oldest first. The counter is the
/// instance's <c>BuffState.Data</c>. A stacked power would have added the second play's turns,
/// energy or damage to the first one's and shared a single count.
/// </summary>
public class InstancedCounterTests
{
    private static List<BuffState> Of(Fight fight, BuffId id) =>
        fight.State.PlayerBuffs.Where(b => b.Id == id).ToList();

    // ── Toric Toughness ───────────────────────────────────────────────────────────────────

    private static int ToricId => GeneratedData.Cards.FindId("ToricToughness")!.Value;

    private static Fight ToricFight(params CardInstance[] hand)
    {
        var fight = Fight.Hand(hand).Energy(9).Enemy();
        fight.State.PlayerHp = 900;
        fight.State.PlayerMaxHp = 900;
        return fight;
    }

    /// <summary>
    /// `ToricToughness.OnPlay` hands the power the block it GAINED (`SetBlock`) and two turns. A
    /// second play is a second power with its own block and its own two turns: the turns do not
    /// add to four on the first, and the first one's block is not overwritten by the second's.
    /// </summary>
    [Fact]
    public void ASecondPlayIsAnInstanceWithItsOwnBlockAndItsOwnTurns()
    {
        var fight = ToricFight(new CardInstance(ToricId, false), new CardInstance(ToricId, true));

        fight.Play(0);
        fight.Play(0);

        Assert.Equal(
            [new BuffState(BuffId.ToricToughness, 2, 5), new BuffState(BuffId.ToricToughness, 2, 7)],
            Of(fight, BuffId.ToricToughness)
        );
    }

    [Fact]
    public void EveryInstanceGrantsItsOwnBlockWhenBlockClears()
    {
        var fight = ToricFight(new CardInstance(ToricId, false), new CardInstance(ToricId, true));
        fight.Play(0);
        fight.Play(0);

        fight.EndTurn();

        Assert.Equal(12, fight.State.PlayerBlock);
        Assert.Equal(
            [new BuffState(BuffId.ToricToughness, 1, 5), new BuffState(BuffId.ToricToughness, 1, 7)],
            Of(fight, BuffId.ToricToughness)
        );
    }

    /// <summary>
    /// An instance played a turn later runs out a turn later; the first goes when ITS two turns
    /// are spent, and what is left grants only its own block.
    /// </summary>
    [Fact]
    public void AnInstancePlayedLaterRunsOutLater()
    {
        var fight = ToricFight(new CardInstance(ToricId, false));
        fight.Play(0);
        fight.EndTurn();
        fight.State.Hand.Clear();
        fight.State.Hand.Add(new CardInstance(ToricId, true));
        fight.State.Energy = 9;
        fight.Play(0);

        fight.EndTurn();

        Assert.Equal(12, fight.State.PlayerBlock);
        Assert.Equal([new BuffState(BuffId.ToricToughness, 1, 7)], Of(fight, BuffId.ToricToughness));

        fight.EndTurn();

        Assert.Equal(7, fight.State.PlayerBlock);
        Assert.Empty(Of(fight, BuffId.ToricToughness));
    }

    // ── Orbit ─────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// `OrbitPower.AfterEnergySpent` counts the energy spent SINCE THAT INSTANCE landed. After 3
    /// spent a second Orbit lands; the next 1 completes the first one's four and only the first
    /// pays, and the second pays on its own fourth.
    /// </summary>
    [Fact]
    public void EveryOrbitCountsTheEnergySpentSinceItLanded()
    {
        var fight = Fight.Hand().Energy(10).PlayerBuff(BuffId.Orbit, 1);
        CardEffects.ApplyAfterEnergySpent(fight.State, 3);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Orbit, 2);

        CardEffects.ApplyAfterEnergySpent(fight.State, 1);

        Assert.Equal(11, fight.State.Energy);
        Assert.Equal(
            [new BuffState(BuffId.Orbit, 1, 4), new BuffState(BuffId.Orbit, 2, 1)],
            Of(fight, BuffId.Orbit)
        );

        CardEffects.ApplyAfterEnergySpent(fight.State, 3);

        // The second reaches four and pays ITS amount of two; the first is at seven.
        Assert.Equal(13, fight.State.Energy);
        Assert.Equal(
            [new BuffState(BuffId.Orbit, 1, 7), new BuffState(BuffId.Orbit, 2, 4)],
            Of(fight, BuffId.Orbit)
        );
    }

    /// <summary>The Orbit card applies an instance, it does not add an energy to the last one.</summary>
    [Fact]
    public void PlayingOrbitTwiceMakesTwoPowers()
    {
        var fight = Fight.Hand(new CardInstance(335, false), new CardInstance(335, true)).Energy(9);

        fight.Play(0);
        fight.Play(0);

        Assert.Equal(2, Of(fight, BuffId.Orbit).Count);
    }

    // ── Panache ───────────────────────────────────────────────────────────────────────────

    private static void PlayDefends(Fight fight, int count)
    {
        for (int i = 0; i < count; i++)
        {
            fight.State.Hand = [Card(IC.DefendIronclad)];
            fight.Play(0);
        }
    }

    /// <summary>
    /// `PanachePower` counts every card its owner plays, from the one AFTER the play that applied
    /// it (`alreadyApplied`). Each play is its own counter: the first, played alone, counts the
    /// second Panache's play as one of its own five, and the second does not count that play.
    /// </summary>
    [Fact]
    public void EveryPanacheCountsItsOwnFive()
    {
        var fight = Fight.Hand(Card(CL.Panache), Card(CL.Panache, upgraded: true))
            .Energy(30)
            .Enemy(hp: 400);
        fight.Play(0);
        PlayDefends(fight, 2);
        fight.State.Hand = [Card(CL.Panache, upgraded: true)];
        fight.Play(0);

        Assert.Equal(
            [new BuffState(BuffId.PanachePower, 10, 3), new BuffState(BuffId.PanachePower, 14, 0)],
            Of(fight, BuffId.PanachePower)
        );

        int hp = fight.Enemy0.Hp;
        PlayDefends(fight, 1);
        Assert.Equal(hp, fight.Enemy0.Hp);

        // The first reaches five: ITS ten, and the second has counted only two of its own.
        PlayDefends(fight, 1);
        Assert.Equal(hp - 10, fight.Enemy0.Hp);
        Assert.Equal(
            [new BuffState(BuffId.PanachePower, 10, 0), new BuffState(BuffId.PanachePower, 14, 2)],
            Of(fight, BuffId.PanachePower)
        );

        // Two more leave the second one short of its five, and the third reaches it: ITS fourteen.
        PlayDefends(fight, 2);
        Assert.Equal(hp - 10, fight.Enemy0.Hp);
        PlayDefends(fight, 1);
        Assert.Equal(hp - 10 - 14, fight.Enemy0.Hp);
    }

    /// <summary>
    /// A Panache with no count in its Data -- what the live game's snapshot gives -- is armed
    /// and has counted nothing: the fifth card played after the snapshot fires it.
    /// </summary>
    [Fact]
    public void APanacheBuiltWithoutADataCountsFromTheNextCard()
    {
        var fight = Fight.Hand().Energy(30).Enemy(hp: 400).PlayerBuff(BuffId.PanachePower, 10);

        PlayDefends(fight, 4);
        Assert.Equal(400, fight.Enemy0.Hp);

        PlayDefends(fight, 1);
        Assert.Equal(390, fight.Enemy0.Hp);
    }

    // ── Monologue ─────────────────────────────────────────────────────────────────────────

    private const int Monologue = 316;

    /// <summary>
    /// Each Monologue is a power of its own with its own `StrengthApplied`. The second pays only
    /// from the card after it, and the first pays for the second Monologue's own play.
    /// </summary>
    [Fact]
    public void EveryMonologueKeepsItsOwnRunningTotal()
    {
        var fight = Fight.Hand(Card(Monologue), Card(Monologue)).Energy(9).Enemy(hp: 500);
        fight.Play(0);
        fight.Play(0);

        Assert.Equal(
            [new BuffState(BuffId.Monologue, 1, 1), new BuffState(BuffId.Monologue, 1, 0)],
            Of(fight, BuffId.Monologue)
        );
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Strength));

        fight.State.Hand = [Card(IC.DefendIronclad)];
        fight.Play(0);

        Assert.Equal(
            [new BuffState(BuffId.Monologue, 1, 2), new BuffState(BuffId.Monologue, 1, 1)],
            Of(fight, BuffId.Monologue)
        );
        Assert.Equal(3, fight.PlayerBuffAmount(BuffId.Strength));
    }

    /// <summary>
    /// `AfterSideTurnEnd` runs for each instance: each removes itself and takes back its own
    /// total, so every point of Strength the two handed out is returned and no more.
    /// </summary>
    [Fact]
    public void AtTheEndOfTheTurnEveryInstanceTakesBackItsOwn()
    {
        var fight = Fight.Hand(Card(Monologue), Card(Monologue)).Energy(9).Enemy(hp: 500);
        fight.PlayerBuff(BuffId.Strength, 4);
        fight.Play(0);
        fight.Play(0);
        fight.State.Hand = [Card(IC.DefendIronclad)];
        fight.Play(0);
        Assert.Equal(7, fight.PlayerBuffAmount(BuffId.Strength));

        fight.EndTurn();

        Assert.Equal(4, fight.PlayerBuffAmount(BuffId.Strength));
        Assert.Empty(Of(fight, BuffId.Monologue));
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.MonologueApplied));
    }

    /// <summary>
    /// `MonologuePower` listens to `BeforeCardPlayed`/`AfterCardPlayed`, which an auto-play fires
    /// like any play (`CardCmd.AutoPlay` runs `OnPlayWrapper`): a Cascade that plays a Defend off
    /// the draw pile pays for the Defend and for the Cascade.
    /// </summary>
    [Fact]
    public void AnAutoPlayedCardPaysToo()
    {
        const int cascade = 546;
        var fight = Fight.Hand(new CardInstance(cascade, false)).Energy(1).Enemy(hp: 500);
        fight.PlayerBuff(BuffId.Monologue, 1);
        fight.State.DrawPile = [Card(IC.DefendIronclad)];

        fight.Play();

        Assert.Equal(2, fight.PlayerBuffAmount(BuffId.Strength));
        Assert.Equal([new BuffState(BuffId.Monologue, 1, 2)], Of(fight, BuffId.Monologue));
    }
}
