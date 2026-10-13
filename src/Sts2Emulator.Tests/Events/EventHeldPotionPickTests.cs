using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Ranwid the Elder and the Stone of All Time each name a held potion while they build their
/// options: <c>Rng.NextItem(Owner.Potions)</c>, one draw off the event's own stream (none on an
/// empty belt). The emulator gave away whichever potion sat first in the belt, so a run holding
/// several lost the wrong one -- and every offer that reads the belt later (The Future of Potions'
/// card types, a shop's room for a potion) worked from the wrong belt.
/// </summary>
public class EventHeldPotionPickTests
{
    private static int Potion(string name) => RunNonCombatEffects.NamedPotion(name);

    private static RunEngine Open(int eventId, string seed, params string[] belt)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        engine.State.MaxPotionSlots = 3;
        for (int i = 0; i < belt.Length; i++)
        {
            engine.State.PotionSlots[i] = belt[i] == "-" ? 0 : Potion(belt[i]);
        }

        RunNonCombatEffects.BeginEvent(engine.State, eventId);
        return engine;
    }

    private static GameRng EventStream(RunEngine engine, string entry) =>
        new(unchecked(engine.State.Rng.Seed + (uint)DeterministicHash.GetDeterministicHashCode(entry)));

    private static readonly string[] Belt = ["FirePotion", "BlockPotion", "FlexPotion"];

    [Fact]
    public void StoneOfAllTimeDrinksThePotionItNamedNotTheFirst()
    {
        var picked = new HashSet<int>();
        for (int n = 0; n < 40; n++)
        {
            var engine = Open(RunConstants.EventStoneOfAllTime, "SEED" + n.ToString("D6"), Belt);
            int expected = EventStream(engine, "STONE_OF_ALL_TIME").NextInt(0, 3);
            var before = engine.State.PotionSlots.ToArray();
            int maxHp = engine.State.PlayerMaxHp;

            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

            Assert.Equal(maxHp + 10, engine.State.PlayerMaxHp);
            for (int slot = 0; slot < 3; slot++)
            {
                Assert.Equal(slot == expected ? 0 : before[slot], engine.State.PotionSlots[slot]);
            }

            picked.Add(expected);
        }

        Assert.Equal(3, picked.Count); // every slot comes up; "the first one" never was the rule
    }

    [Fact]
    public void RanwidGivesThePotionItNamedNotTheFirst()
    {
        var picked = new HashSet<int>();
        for (int n = 0; n < 40; n++)
        {
            var engine = Open(RunConstants.EventRanwidTheElder, "SEED" + n.ToString("D6"), Belt);
            var stream = EventStream(engine, "RANWID_THE_ELDER");
            int expected = stream.NextInt(0, 3);
            var before = engine.State.PotionSlots.ToArray();
            int relics = engine.State.Relics.Count;

            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

            for (int slot = 0; slot < 3; slot++)
            {
                Assert.Equal(slot == expected ? 0 : before[slot], engine.State.PotionSlots[slot]);
            }

            Assert.Equal(relics + 1, engine.State.Relics.Count);
            picked.Add(expected);
        }

        Assert.Equal(3, picked.Count);
    }

    /// <summary>The slot is the k-th HELD potion: gaps in the belt do not count.</summary>
    [Fact]
    public void AGapInTheBeltIsSkipped()
    {
        var engine = Open(RunConstants.EventStoneOfAllTime, "ABCDEF", "FirePotion", "-", "FlexPotion");
        int k = EventStream(engine, "STONE_OF_ALL_TIME").NextInt(0, 2);

        Assert.Equal(k == 0 ? 0 : 2, RunNonCombatEffects.EventPotionSlot(engine.State));
    }

    [Fact]
    public void AnEmptyBeltDrawsNothingForThePotion()
    {
        var engine = Open(RunConstants.EventStoneOfAllTime, "ABCDEF");

        Assert.Equal(-1, RunNonCombatEffects.EventPotionSlot(engine.State));
        Assert.Equal(0, engine.State.EventRngStream?.CallCount ?? 0);
    }
}

/// <summary>
/// War Historian Repy's chest: <c>UnlockChest</c> offers [PotionReward, PotionReward, RelicReward,
/// RelicReward] on one rewards screen. The emulator offered one potion and one relic ("half of it").
/// </summary>
public class RepyChestRewardTests
{
    [Fact]
    public void TheChestOffersTwoPotionsAndTwoRelicsRolledInThatOrder()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var s = engine.State;
        s.Deck.Add(new CardInstance(RunNonCombatEffects.LanternKeyCard, Upgraded: false));
        RunNonCombatEffects.BeginEvent(s, RunConstants.EventWarHistorianRepy);
        var oracle = s.PlayerRng.Rewards.Copy();
        int first = RunRewardGenerator.NextPotion(s, oracle);
        int second = RunRewardGenerator.NextPotion(s, oracle);
        int draws = s.PlayerRng.Rewards.CallCount;

        Assert.Equal(0, engine.Step(1, -1, out _, out _, out _));

        Assert.Equal(RunPhase.RelicReward, s.Phase);
        Assert.Equal(first, s.RewardPotion);
        Assert.Equal([second], s.PendingPotionRewards);
        Assert.NotEqual(0, s.RelicReward);
        Assert.Single(s.PendingBonusRelicRewards);
        Assert.NotEqual(s.RelicReward, s.PendingBonusRelicRewards[0]);
        // Both potions (rarity, pick) before either relic's rarity roll.
        Assert.Equal(oracle.CallCount, draws + 4);
        Assert.True(s.PlayerRng.Rewards.CallCount >= oracle.CallCount + 2);
    }

    [Fact]
    public void LeavingTheScreenReturnsToTheEventsFinishedPage()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var s = engine.State;
        s.Deck.Add(new CardInstance(RunNonCombatEffects.LanternKeyCard, Upgraded: false));
        RunNonCombatEffects.BeginEvent(s, RunConstants.EventWarHistorianRepy);
        engine.Step(1, -1, out _, out _, out _);

        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));

        Assert.Equal(RunPhase.Event, s.Phase);
        Assert.Equal(RunConstants.EventResultPending, s.EventId);
    }
}
