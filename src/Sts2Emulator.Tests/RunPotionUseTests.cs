using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The four `PotionUsage.AnyTime` potions used away from a fight: Foul Potion thrown at the shop's
/// merchant (100 gold), Blood Potion, Fruit Juice, and Entropic Brew (which fills every open slot
/// off `Rng.CombatPotionGeneration`). The run engine had no such move; real histories use them
/// constantly (Foul Potion at 435 shop floors of 21,742 Ironclad A10 runs).
/// </summary>
public class RunPotionUseTests
{
    private static int Potion(string name) => RunNonCombatEffects.NamedPotion(name);

    private static RunEngine At(RunPhase phase, params string[] belt)
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.Phase = phase;
        engine.State.MaxPotionSlots = 3;
        Array.Clear(engine.State.PotionSlots);
        for (int i = 0; i < belt.Length; i++)
        {
            engine.State.PotionSlots[i] = belt[i] == "-" ? 0 : Potion(belt[i]);
        }

        return engine;
    }

    private static int[] Mask(RunEngine engine)
    {
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        return mask;
    }

    private static int Use(RunEngine engine, int slot) =>
        engine.Step(RunConstants.UsePotionAction + slot, -1, out _, out _, out _);

    [Fact]
    public void FoulPotionAtTheShopPaysAHundredGoldAndLeavesTheShop()
    {
        var engine = At(RunPhase.Shop, "FirePotion", "FoulPotion", "FoulPotion");
        engine.State.Gold = 30;

        Assert.Equal(1, Mask(engine)[RunConstants.UsePotionAction + 1]);
        Assert.Equal(0, Mask(engine)[RunConstants.UsePotionAction + 0]); // Fire Potion is CombatOnly

        Assert.Equal(0, Use(engine, 1));

        Assert.Equal(130, engine.State.Gold);
        Assert.Equal(0, engine.State.PotionSlots[1]);
        Assert.Equal(Potion("FoulPotion"), engine.State.PotionSlots[2]); // one potion, one throw
        Assert.Equal(RunPhase.Shop, engine.State.Phase);

        Assert.Equal(0, Use(engine, 2));
        Assert.Equal(230, engine.State.Gold);
    }

    [Theory]
    [InlineData(RunPhase.Map)]
    [InlineData(RunPhase.Rest)]
    [InlineData(RunPhase.Event)]
    [InlineData(RunPhase.RelicReward)]
    public void FoulPotionNeedsTheMerchant(RunPhase phase)
    {
        var engine = At(phase, "FoulPotion");
        engine.State.EventId = RunConstants.EventWellspring;
        engine.State.Gold = 30;

        Assert.Equal(0, Mask(engine)[RunConstants.UsePotionAction]);
        Assert.Equal(-1, Use(engine, 0));
        Assert.Equal(30, engine.State.Gold);
        Assert.Equal(Potion("FoulPotion"), engine.State.PotionSlots[0]);
    }

    [Fact]
    public void BloodPotionHealsTwentyPercentOfMaxHpAndCapsAtMax()
    {
        var engine = At(RunPhase.Rest, "BloodPotion", "BloodPotion");
        engine.State.PlayerMaxHp = 80;
        engine.State.PlayerHp = 30;

        Assert.Equal(0, Use(engine, 0));
        Assert.Equal(46, engine.State.PlayerHp);

        engine.State.PlayerHp = 75;
        Assert.Equal(0, Use(engine, 1));
        Assert.Equal(80, engine.State.PlayerHp);
    }

    [Fact]
    public void FruitJuiceAddsFiveMaxHp()
    {
        var engine = At(RunPhase.Map, "FruitJuice");
        engine.State.PlayerMaxHp = 80;
        engine.State.PlayerHp = 60;

        Assert.Equal(0, Use(engine, 0));

        Assert.Equal(85, engine.State.PlayerMaxHp);
        Assert.Equal(65, engine.State.PlayerHp);
    }

    /// <summary>
    /// EntropicBrew.OnUse: `while (HasOpenPotionSlots)` roll `CreateRandomPotionOutOfCombat` off
    /// `Rng.CombatPotionGeneration` -- two draws a potion -- and procure it. Not the Rewards or the
    /// Shops stream, which every card reward and shelf also reads.
    /// </summary>
    [Fact]
    public void EntropicBrewFillsTheOpenSlotsOffTheCombatPotionStream()
    {
        var engine = At(RunPhase.Shop, "EntropicBrew", "FirePotion", "-");
        var s = engine.State;
        var oracle = s.Rng.CombatPotionGeneration.Copy();
        int first = RunRewardGenerator.NextPotion(s, oracle);
        int second = RunRewardGenerator.NextPotion(s, oracle);
        int rewards = s.PlayerRng.Rewards.CallCount;
        int shops = s.PlayerRng.Shops.CallCount;

        Assert.Equal(1, Mask(engine)[RunConstants.UsePotionAction]);
        Assert.Equal(0, Use(engine, 0));

        Assert.Equal([first, Potion("FirePotion"), second], s.PotionSlots.Take(3));
        Assert.Equal(oracle.CallCount, s.Rng.CombatPotionGeneration.CallCount);
        Assert.Equal(rewards, s.PlayerRng.Rewards.CallCount);
        Assert.Equal(shops, s.PlayerRng.Shops.CallCount);
    }

    [Fact]
    public void EntropicBrewOnAFullBeltRollsNothingButTheBrewsOwnSlot()
    {
        var engine = At(RunPhase.Map, "EntropicBrew", "FirePotion", "FirePotion");
        var s = engine.State;
        var oracle = s.Rng.CombatPotionGeneration.Copy();
        int only = RunRewardGenerator.NextPotion(s, oracle);

        Assert.Equal(0, Use(engine, 0));

        // The brew leaves the belt first, so its own slot is the one open slot it fills.
        Assert.Equal(only, s.PotionSlots[0]);
        Assert.Equal(oracle.CallCount, s.Rng.CombatPotionGeneration.CallCount);
    }

    [Fact]
    public void EntropicBrewUnderSozuRollsOnceAndGainsNothing()
    {
        var engine = At(RunPhase.Map, "EntropicBrew", "-", "-");
        var s = engine.State;
        s.Relics.Add(new RelicInstance(RelicEffects.Sozu));
        int before = s.Rng.CombatPotionGeneration.CallCount;

        Assert.Equal(0, Use(engine, 0));

        Assert.All(s.PotionSlots, slot => Assert.Equal(0, slot));
        Assert.Equal(before + 2, s.Rng.CombatPotionGeneration.CallCount); // rolled, then refused
    }

    /// <summary>
    /// The Future of Potions, Ranwid and the Stone of All Time clear `CanRemovePotions` until they
    /// finish, which also greys out the belt's use button.
    /// </summary>
    [Theory]
    [InlineData(RunConstants.EventTheFutureOfPotions)]
    [InlineData(RunConstants.EventRanwidTheElder)]
    [InlineData(RunConstants.EventStoneOfAllTime)]
    public void AnEventThatHoldsTheBeltBlocksUse(int eventId)
    {
        var engine = At(RunPhase.Event, "BloodPotion");
        engine.State.EventId = eventId;
        engine.State.PlayerHp = 10;

        Assert.Equal(0, Mask(engine)[RunConstants.UsePotionAction]);
        Assert.Equal(-1, Use(engine, 0));
        Assert.Equal(10, engine.State.PlayerHp);
    }

    [Fact]
    public void AnEmptySlotIsNotAMove()
    {
        var engine = At(RunPhase.Shop, "FoulPotion");

        Assert.Equal(-1, Use(engine, 2));
        Assert.Equal(0, Mask(engine)[RunConstants.UsePotionAction + 2]);
    }
}
