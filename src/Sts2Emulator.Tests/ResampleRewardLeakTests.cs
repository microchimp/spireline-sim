using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Spireline DESIGN.md 7.6 L3, reward contents: a resampled clone of a run redeals the card
/// choices, potion drop and relic reward of a fight not yet won, and keeps the ones of a
/// reward screen the player has already been shown.
/// </summary>
public class ResampleRewardLeakTests
{
    private const string Seed = "84G7XUWQD6";
    private const int Clones = 200;

    private static RunEngine AtMap()
    {
        var e = new RunEngine();
        e.Reset(Seed, 10);
        for (int guard = 0; guard < 10 && e.State.Phase != RunPhase.Map; guard++)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Map, e.State.Phase);
        return e;
    }

    private static int NodeOption(RunEngine e) =>
        Enumerable.Range(0, RunConstants.MapChoices).First(i => e.State.MapNodeTypes[i] == RunConstants.NodeNormal);

    private static int EncounterOf(bool elite) =>
        (int)(elite ? CombatFactory.ActOneEncounter.Byrdonis : CombatFactory.ActOneEncounter.SlimesNormal);

    private static int NodeTypeOf(bool elite) => elite ? RunConstants.NodeElite : RunConstants.NodeNormal;

    /// <summary>The one place a resampled clone is made, so a broken variant is a one-line change.</summary>
    private static RunEngine Resampled(RunEngine e, int seed) => e.Clone(seed);

    private static string J<T>(IEnumerable<T> xs) => string.Join(",", xs);

    private sealed record Contents(string Cards, string Upgraded, int Gold, int Potion, int Relic, string Extras);

    /// <summary>Every reward field a screen shows or queues behind it.</summary>
    private static Contents Read(RunState s) =>
        new(
            J(s.RewardCards),
            J(s.RewardUpgraded.Select(b => b ? 1 : 0)),
            s.RewardGold,
            s.RewardPotion,
            s.RelicReward,
            J(s.PendingPotionRewards)
                + "|"
                + J(s.PendingBonusRelicRewards)
                + "|"
                + J(s.PendingCardOffers.Select(o => J(o.Cards) + "/" + J(o.Upgraded.Select(b => b ? 1 : 0))))
                + "|"
                + J(s.PendingGoldRewards)
        );

    private static void Enter(RunEngine e, bool elite) =>
        Assert.Equal(0, e.EnterEncounterDebug(NodeOption(e), EncounterOf(elite), NodeTypeOf(elite)));

    private static Contents Win(RunEngine e)
    {
        Assert.Equal(RunPhase.Combat, e.State.Phase);
        Assert.Equal(0, e.DebugResolveCombat());
        Assert.NotEqual(RunPhase.Combat, e.State.Phase);
        Assert.NotEqual(RunPhase.Complete, e.State.Phase);
        return Read(e.State);
    }

    /// <summary>
    /// The run stands at <paramref name="source" />; the true run enters the fight and wins it. Clones of
    /// <paramref name="source" /> (before or after entering) win the same fight.
    /// </summary>
    private static void AssertRewardsAreRedealt(bool elite, bool cloneAfterEntry)
    {
        var source = AtMap();
        if (cloneAfterEntry)
        {
            Enter(source, elite);
        }

        RunEngine Finish(RunEngine r)
        {
            if (!cloneAfterEntry)
            {
                Enter(r, elite);
            }

            return r;
        }

        var truth = Win(Finish(source.Clone()));
        Assert.NotEqual("0,0,0", truth.Cards);
        Assert.Equal(elite, Finish(source.Clone()).State.CurrentNodeType == RunConstants.NodeElite);

        // Control: a faithful copy is an oracle and meets the true run's rewards exactly.
        Assert.Equal(truth, Win(Finish(source.Clone())));

        var outcomes = new List<Contents>();
        for (int seed = 0; seed < Clones; seed++)
        {
            var clone = Finish(Resampled(source, seed));
            outcomes.Add(Win(clone));
            if (elite)
            {
                Assert.True(clone.State.RelicReward != 0 || clone.State.Phase != RunPhase.RelicReward);
            }
        }

        string what = $"(elite={elite}, cloneAfterEntry={cloneAfterEntry})";
        Assert.True(outcomes.Distinct().Count() > 1, $"every resampled clone met the true rewards {what}");
        Assert.Contains(outcomes, c => c != truth);
        Assert.True(outcomes.Select(c => c.Cards).Distinct().Count() > 1, $"card choices never varied {what}");
        Assert.True(outcomes.Select(c => c.Potion).Distinct().Count() > 1, $"potion drops never varied {what}");
        if (elite)
        {
            Assert.True(outcomes.Select(c => c.Relic).Distinct().Count() > 1, $"the relic reward never varied {what}");
        }
        else
        {
            Assert.All(outcomes, c => Assert.Equal(0, c.Relic));
        }
    }

    [Fact]
    public void AMonsterFightsRewardsAreRedealtByClonesTakenBeforeTheFightIsEntered() =>
        AssertRewardsAreRedealt(elite: false, cloneAfterEntry: false);

    [Fact]
    public void AMonsterFightsRewardsAreRedealtByClonesTakenInsideTheFight() =>
        AssertRewardsAreRedealt(elite: false, cloneAfterEntry: true);

    [Fact]
    public void AnEliteFightsRewardsAndRelicAreRedealtByClonesTakenBeforeTheFightIsEntered() =>
        AssertRewardsAreRedealt(elite: true, cloneAfterEntry: false);

    [Fact]
    public void AnEliteFightsRewardsAndRelicAreRedealtByClonesTakenInsideTheFight() =>
        AssertRewardsAreRedealt(elite: true, cloneAfterEntry: true);

    [Fact]
    public void ARewardScreenAlreadyOpenedKeepsEveryItemInEveryResampledClone()
    {
        foreach (bool elite in new[] { false, true })
        {
            var e = AtMap();
            Enter(e, elite);
            var shown = Win(e);
            Assert.Equal(RunPhase.RelicReward, e.State.Phase);
            Assert.NotEqual("0,0,0", shown.Cards);

            for (int seed = 0; seed < Clones; seed++)
            {
                Assert.Equal(shown, Read(Resampled(e, seed).State));
            }

            // One screen later (the relic taken, the cards still to pick) the rest stays as shown.
            e.Step(0, -1, out _, out _, out _);
            var later = Read(e.State);
            Assert.Equal(shown.Cards, later.Cards);
            for (int seed = 0; seed < Clones; seed++)
            {
                Assert.Equal(later, Read(Resampled(e, seed).State));
            }
        }
    }
}
