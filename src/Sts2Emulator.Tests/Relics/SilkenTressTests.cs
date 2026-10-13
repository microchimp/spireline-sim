using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// SilkenTress.TryModifyCardRewardOptionsLate: the first card reward built after the relic is
/// obtained has every option Glam can take enchanted with it, then IsUsed is set. The emulator
/// only took the gold (run referee, seed 4LS0JLWHUW: Burning Pact arrived with Glam 1 in the game).
/// </summary>
public class SilkenTressTests
{
    private static RunEngine WithTress()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicSilkenTress);
        return engine;
    }

    private static CardInstance PickFirstGlammable(RunEngine engine)
    {
        var s = engine.State;
        RunRewardGenerator.PopulateCardReward(s);
        s.Phase = RunPhase.CardReward;
        int pick = Array.FindIndex(s.RewardCards, c => c != 0
            && Enchantments.CanEnchant(new CardInstance(c, false), Enchantment.Glam));
        Assert.True(pick >= 0);
        Assert.Equal(0, engine.Step(pick, -1, out _, out _, out _));
        return s.Deck[^1];
    }

    [Fact]
    public void TheFirstCardRewardIsGlamEnchanted()
    {
        var engine = WithTress();
        Assert.Equal(0, engine.State.Gold);
        var card = PickFirstGlammable(engine);
        Assert.Equal(Enchantment.Glam, card.Enchantment);
        Assert.Equal(1, card.EnchantAmount);
    }

    [Fact]
    public void OnlyTheFirstOne()
    {
        var engine = WithTress();
        PickFirstGlammable(engine);
        var second = PickFirstGlammable(engine);
        Assert.Equal(Enchantment.None, second.Enchantment);
    }
}
