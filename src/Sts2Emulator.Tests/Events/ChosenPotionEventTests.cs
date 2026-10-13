using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Ranwid and the Stone of All Time each roll the potion they will take as the options are made
/// (<c>base.Rng.NextItem(Owner.Potions)</c>) and name it on the option. The emulator spent
/// Ranwid's draw and threw the answer away, and the Stone never drew: both took whatever potion
/// sat first in the belt.
/// </summary>
[CoversEvent("RanwidTheElder")]
[CoversEvent("StoneOfAllTime")]
public class ChosenPotionEventTests
{
    private static RunEngine At(string seed, int eventId, params int[] slots)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        engine.State.Gold = 200;
        for (int i = 0; i < engine.State.PotionSlots.Length; i++)
        {
            engine.State.PotionSlots[i] = i < slots.Length ? slots[i] : 0;
        }

        RunNonCombatEffects.BeginEvent(engine.State, eventId);
        return engine;
    }

    private static int Expected(RunState state, string entry, int[] heldSlots)
    {
        var oracle = new GameRng(
            unchecked(
                state.Rng.Seed + (uint)DeterministicHash.GetDeterministicHashCode(entry)
            )
        );
        return heldSlots[oracle.NextInt(0, heldSlots.Length)];
    }

    [Fact]
    public void RanwidTakesThePotionHeNamedNotTheFirstOne()
    {
        bool notFirst = false;
        foreach (string seed in new[] { "ABCDEF", "JJ39U0LLVY", "M4HT4KF030", "X2QEM46QX8", "0KFMJ8NU8E", "ABB6PCF77A" })
        {
            var engine = At(seed, RunConstants.EventRanwidTheElder, 11, 12, 13);
            int slot = Expected(engine.State, "RANWID_THE_ELDER", [0, 1, 2]);
            notFirst |= slot != 0;

            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(i == slot ? 0 : 11 + i, engine.State.PotionSlots[i]);
            }
        }

        Assert.True(notFirst, "every seed named the first potion");
    }

    [Fact]
    public void TheStoneLiftsThePotionItNamedAndGivesTenMaxHp()
    {
        bool notFirst = false;
        foreach (string seed in new[] { "ABCDEF", "JJ39U0LLVY", "M4HT4KF030", "X2QEM46QX8", "0KFMJ8NU8E", "ABB6PCF77A" })
        {
            var engine = At(seed, RunConstants.EventStoneOfAllTime, 0, 21, 22);
            int maxHp = engine.State.PlayerMaxHp;
            int slot = Expected(engine.State, "STONE_OF_ALL_TIME", [1, 2]);
            notFirst |= slot != 1;

            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

            Assert.Equal(maxHp + 10, engine.State.PlayerMaxHp);
            for (int i = 1; i < 3; i++)
            {
                Assert.Equal(i == slot ? 0 : 20 + i, engine.State.PotionSlots[i]);
            }
        }

        Assert.True(notFirst, "every seed named the first potion");
    }

    [Fact]
    public void AnEmptyBeltDrawsNothingAndLocksTheLift()
    {
        var engine = At("ABCDEF", RunConstants.EventStoneOfAllTime);

        Assert.Null(engine.State.EventRngStream);
        Assert.Equal(-1, engine.Step(0, -1, out _, out _, out _));
    }
}
