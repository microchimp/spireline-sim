using Sts2Emulator.Core;
using Sts2Emulator.Tests.Invariants;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Each live-state invariant fires on a state built to break it, and none fires on a fight played from
/// the start by a seeded chooser (the relations in MetamorphicTests run that on many more).
/// </summary>
public class StateInvariantsTests
{
    private static CombatState Sound(int seed = 7) =>
        TestFights.Open(CombatFactory.ActOneEncounter.NibbitsWeak, seed).State;

    private static bool Fires(CombatState s, string check) => StateInvariants.Check(s).Any(v => v.Check == check);

    [Fact]
    public void AFreshFightBreaksNothing()
    {
        Assert.Empty(StateInvariants.Check(Sound()));
    }

    [Fact]
    public void HitPointsStayInRange()
    {
        var s = Sound();
        s.PlayerHp = s.PlayerMaxHp + 1;
        Assert.True(Fires(s, "hp_range"));
        s = Sound();
        s.PlayerHp = -1;
        Assert.True(Fires(s, "hp_range"));
        s = Sound();
        s.Enemies[0].Hp = s.Enemies[0].MaxHp + 1;
        Assert.True(Fires(s, "hp_range"));
    }

    [Fact]
    public void EnergyAndBlockAreNeverNegative()
    {
        var s = Sound();
        s.Energy = -1;
        Assert.True(Fires(s, "energy"));
        s = Sound();
        s.PlayerBlock = -4;
        Assert.True(Fires(s, "block"));
        s = Sound();
        s.Enemies[0].Block = -1;
        Assert.True(Fires(s, "block"));
    }

    [Fact]
    public void TheHandHoldsAtMostTen()
    {
        var s = Sound();
        while (s.Hand.Count <= StateInvariants.HandLimit)
        {
            s.Hand.Add(s.Hand[0]);
        }

        Assert.True(Fires(s, "hand_limit"));
    }

    [Fact]
    public void ACardMustBeOneTheTablesKnow()
    {
        var s = Sound();
        s.DiscardPile.Add(new CardInstance(99999, false));
        Assert.True(Fires(s, "card_id"));
    }

    [Fact]
    public void ABuffAtZeroOrTwiceOrNegativeIsBroken()
    {
        var s = Sound();
        s.PlayerBuffs.Add(new BuffState(BuffId.Vulnerable, 0));
        Assert.True(Fires(s, "power_amount"));
        s = Sound();
        s.PlayerBuffs.Add(new BuffState(BuffId.Weak, 1));
        s.PlayerBuffs.Add(new BuffState(BuffId.Weak, 2));
        Assert.True(Fires(s, "power_dup"));
        s = Sound();
        s.PlayerBuffs.Add(new BuffState(BuffId.Weak, -2));
        Assert.True(Fires(s, "power_amount"));
    }

    [Fact]
    public void StrengthMayBeNegativeAndAnInstancedPowerMayRepeat()
    {
        var s = Sound();
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, -3));
        s.PlayerBuffs.Add(new BuffState(BuffId.TheBombPower, 3, 40));
        s.PlayerBuffs.Add(new BuffState(BuffId.TheBombPower, 2, 40));
        Assert.Empty(StateInvariants.Check(s));
    }

    [Fact]
    public void MachineryThatRunsInsideAStepIsAtRestWhenOneReturns()
    {
        var s = Sound();
        s.AutoPlayQueue.Add(s.Hand[0]);
        Assert.True(Fires(s, "at_rest"));
        s = Sound();
        s.HeldPlayedCard = s.Hand[0];
        Assert.True(Fires(s, "at_rest"));
        s = Sound();
        s.AutoPlaying = true;
        Assert.True(Fires(s, "at_rest"));
        s = Sound();
        s.ResolvingDefendCard = true;
        Assert.True(Fires(s, "at_rest"));
    }

    [Fact]
    public void TheKnownRegionsStayInsideThePile()
    {
        var s = Sound();
        s.KnownTopCount = s.DrawPile.Count;
        s.KnownBottomCount = 1;
        Assert.True(Fires(s, "known_order"));
    }

    [Fact]
    public void TwoFieldsMayNotShareOneGenerator()
    {
        var s = Sound();
        s.CardSelectionRng = s.TargetRng;
        Assert.True(Fires(s, "rng_shared"));
    }

    [Fact]
    public void ACounterMayNotGoNegative()
    {
        var s = Sound();
        s.CardsPlayedThisTurn = -1;
        Assert.True(Fires(s, "counter_nonneg"));
    }

    [Fact]
    public void ATrackerCatchesAStreamThatGoesBackAndATurnThatSkips()
    {
        var s = Sound();
        var tracker = new StateInvariants.Tracker();
        Assert.Empty(tracker.Step(s));

        s.ShuffleRng!.Next();
        Assert.Empty(tracker.Step(s));

        s.ShuffleRng = new CountingRandom(1); // a fresh stream: back at zero
        Assert.Contains(tracker.Step(s), v => v.Check == "rng_monotone");

        s.Turn += 2;
        Assert.Contains(tracker.Step(s), v => v.Check == "turn_monotone");

        s.CardsDrawnThisCombat = -5 + s.CardsDrawnThisCombat;
        Assert.Contains(tracker.Step(s), v => v.Check is "counter_monotone" or "counter_nonneg");
    }

    [Theory]
    [InlineData((int)CombatFactory.ActOneEncounter.Cultists, 1)]
    [InlineData((int)CombatFactory.ActOneEncounter.SlimesWeak, 2)]
    [InlineData((int)CombatFactory.ActOneEncounter.NibbitsNormal, 3)]
    [InlineData((int)CombatFactory.ActOneEncounter.CorpseSlugs, 4)]
    [InlineData((int)CombatFactory.ActOneEncounter.FossilStalker, 5)]
    public void AFightPlayedByAChooserBreaksNothing(int encounterId, int seed)
    {
        var encounter = (CombatFactory.ActOneEncounter)encounterId;
        var c = TestFights.Open(encounter, seed);
        var report = new RelationReport();

        Metamorphic.Invariants(() => c, seed, report);

        Assert.Empty(report.Failures);
    }
}
