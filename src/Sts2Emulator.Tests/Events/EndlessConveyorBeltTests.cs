using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// <c>EndlessConveyor.GrabSomethingOffTheBelt</c>: pay (unless the dish is the Golden Fysh), run
/// the dish's action to its end -- a transform's pick and a potion screen are awaited -- and only
/// then <c>RollDish</c> for the next one, off the same event stream.
/// </summary>
[CoversEvent("EndlessConveyor")]
public class EndlessConveyorBeltTests
{
    private static RunEngine Belt(string seed, int dish, int gold = 100)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventEndlessConveyor);
        engine.State.EventValue0 = dish;
        engine.State.Gold = gold;
        return engine;
    }

    [Fact]
    public void TheGoldenFyshIsFreeAndPaysSeventyFive()
    {
        var engine = Belt("ABCDEF", RunNonCombatEffects.DishGoldenFysh);

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(175, engine.State.Gold);
    }

    [Theory]
    [InlineData("ABCDEF")]
    [InlineData("WR2EZCP1YP")]
    [InlineData("HTTGUGQD27")]
    [InlineData("EK395JE92V")]
    public void TheJellyLiversTransformDrawsBeforeTheNextDishIsRolled(string seed)
    {
        var engine = Belt(seed, RunNonCombatEffects.DishJellyLiver);
        var state = engine.State;
        var oracle = state.EventRngStream!.Clone();
        int before = oracle.CallCount;
        int original = state.Deck[0].DefId;
        int expected = oracle.NextItem(RunRewardGenerator.TransformOptionsFor(original));

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));
        Assert.Equal(RunPhase.TransformSelect, state.Phase);
        // The belt has not turned while the pick is open.
        Assert.Equal(before, state.EventRngStream!.CallCount);

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(expected, state.Deck[^1].DefId);
        Assert.Equal(RunPhase.Event, state.Phase);
        Assert.Equal(RunConstants.EventEndlessConveyor, state.EventId);
        // The transform's draw, then the roll's.
        Assert.Equal(before + 2, state.EventRngStream!.CallCount);
    }

    [Fact]
    public void TheCondimentsPotionIsOnAScreenAndTheBeltTurnsWhenItIsAnswered()
    {
        foreach (int answer in new[] { 0, RunConstants.RewardSkipAction })
        {
            var engine = Belt("ABCDEF", RunNonCombatEffects.DishSuspiciousCondiment);
            var state = engine.State;
            int draws = state.EventRngStream!.CallCount;
            int grabs = state.EventValue1!.Value;

            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

            Assert.Equal(RunPhase.RelicReward, state.Phase);
            Assert.NotEqual(0, state.RewardPotion);
            Assert.Empty(state.PendingPotionRewards);
            Assert.Equal(draws, state.EventRngStream!.CallCount);
            Assert.Equal(grabs, state.EventValue1);

            Assert.Equal(0, engine.Step(answer, -1, out _, out _, out _));

            Assert.Equal(RunPhase.Event, state.Phase);
            Assert.Equal(RunConstants.EventEndlessConveyor, state.EventId);
            Assert.Equal(grabs + 1, state.EventValue1);
            Assert.Equal(draws + 1, state.EventRngStream!.CallCount);
            Assert.Equal(answer == 0, state.PotionSlots.Any(potion => potion != 0));
        }
    }
}
