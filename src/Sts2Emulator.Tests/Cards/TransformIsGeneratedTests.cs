using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// `CardCmd.Transform` in combat records each replacement as generated: `CombatHistory.CardGenerated`
// (which Supermassive counts) and `Hook.AfterCardGeneratedForCombat` (Arsenal's Strength). Every
// transform a card makes in combat goes through it -- Primal Force's Giant Rocks among them.
public class TransformIsGeneratedTests
{
    [Fact]
    public void PrimalForcesRocksCountAsGenerated()
    {
        var fight = Fight
            .Hand(Card(IC.PrimalForce), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash))
            .Energy(3)
            .Enemy(hp: 500)
            .PlayerBuff(BuffId.Arsenal, 1);

        fight.Play();

        Assert.Equal(2, fight.State.CardsGeneratedThisCombat);
        Assert.Equal(2, fight.PlayerBuffAmount(BuffId.Strength));
    }

    [Fact]
    public void EntropysTransformCountsAsGenerated()
    {
        // Entropy's screen comes after the hand draw, over the new hand; its answer transforms.
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Energy(3)
            .Draw(Card(IC.DefendIronclad))
            .Enemy(hp: 500)
            .PlayerBuff(BuffId.RetainHand, 1)
            .PlayerBuff(BuffId.EntropyPower, 1)
            .PlayerBuff(BuffId.Arsenal, 1);

        fight.EndTurn();
        fight.Choose(0);

        Assert.Equal(1, fight.State.CardsGeneratedThisCombat);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Strength));
    }
}
