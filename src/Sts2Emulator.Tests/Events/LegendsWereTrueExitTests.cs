using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Legends Were True's Slowly Find an Exit: never locked (ThatDoesDamage only flags it as
/// lethal, and its potion is offered, so a full belt does not close it), and its potion is ONE
/// PlayerRng.Rewards.NextItem over the character's pool then the shared one -- not a reward's
/// rarity roll. The emulator closed it on a full belt (run referee, seed 0NT10MDKY3).
/// </summary>
public class LegendsWereTrueExitTests
{
    private static RunEngine At()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.EventId = RunConstants.EventTheLegendsWereTrue;
        engine.State.Phase = RunPhase.Event;
        return engine;
    }

    [Fact]
    public void AFullBeltDoesNotCloseTheExit()
    {
        var engine = At();
        var potion = RunNonCombatEffects.NamedPotion("FirePotion");
        for (int i = 0; i < engine.State.MaxPotionSlots; i++) engine.State.PotionSlots[i] = potion;
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        Assert.NotEqual(0, mask[1]);
    }

    [Fact]
    public void ThePotionIsOneUniformPick()
    {
        var engine = At();
        var oracle = engine.State.PlayerRng.Rewards.Copy();
        int expected = oracle.NextItem(RunRewardGenerator.PotionRewardPool(engine.State.Character));

        Assert.Equal(0, engine.Step(1, -1, out _, out _, out _));

        Assert.Equal(expected, engine.State.RewardPotion);
        Assert.Equal(oracle.CallCount, engine.State.PlayerRng.Rewards.CallCount);
    }
}

/// <summary>Wellspring's bottle is the same uniform pick as the exit's (NextPoolPotion).</summary>
public class WellspringPotionTests
{
    [Fact]
    public void TheBottleIsOneUniformPick()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.EventId = RunConstants.EventWellspring;
        engine.State.Phase = RunPhase.Event;
        var oracle = engine.State.PlayerRng.Rewards.Copy();
        int expected = oracle.NextItem(RunRewardGenerator.PotionRewardPool(engine.State.Character));

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(expected, engine.State.RewardPotion);
        Assert.Equal(oracle.CallCount, engine.State.PlayerRng.Rewards.CallCount);
    }
}
