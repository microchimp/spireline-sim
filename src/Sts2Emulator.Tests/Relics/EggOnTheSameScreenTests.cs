using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// <c>CardReward</c> subscribes to its player's <c>RelicObtained</c> as it is built and stays
/// subscribed until it is chosen from or skipped, so a relic obtained while the reward waits runs
/// its card-reward hooks over the options already on the table. An egg claimed off an elite's
/// rewards screen therefore upgrades the card reward still on that screen. The emulator rolled
/// the options with the relics held at that moment and never looked again, so the OFFER was a
/// card short of the game's (the deck, upgraded on arrival, was right) -- 196 recorded Ironclad
/// A10 runs.
/// </summary>
public class EggOnTheSameScreenTests
{
    private static int Card(CardType type)
    {
        foreach (int id in CharacterPools.Cards(PlayerCharacter.Ironclad).ToArray())
        {
            var def = GeneratedData.Cards.Get(id);
            if (
                def.Type == type
                && def.Upgradable
                && def.Rarity is CardRarity.Common or CardRarity.Uncommon or CardRarity.Rare
            )
            {
                return id;
            }
        }

        throw new InvalidOperationException($"no upgradable {type} in the pool");
    }

    private static int Other(CardType type, int not)
    {
        foreach (int id in CharacterPools.Cards(PlayerCharacter.Ironclad).ToArray())
        {
            var def = GeneratedData.Cards.Get(id);
            if (
                id != not
                && def.Type == type
                && def.Upgradable
                && def.Rarity is CardRarity.Common or CardRarity.Uncommon or CardRarity.Rare
            )
            {
                return id;
            }
        }

        throw new InvalidOperationException($"no second {type} in the pool");
    }

    /// <summary>An elite's screen: the relic claimed first, the card reward still waiting.</summary>
    private static RunEngine EliteScreen(int relicId, int[] options)
    {
        var engine = new RunEngine();
        engine.Reset("8QKMNR4T2W");
        engine.State.Phase = RunPhase.RelicReward;
        engine.State.CurrentNodeType = RunConstants.NodeElite;
        engine.State.RewardGold = 0;
        engine.State.RewardPotion = 0;
        engine.State.RelicReward = relicId;
        engine.State.RewardCardPending = true;
        Array.Clear(engine.State.RewardUpgraded);
        for (int i = 0; i < options.Length; i++)
        {
            engine.State.RewardCards[i] = options[i];
        }

        return engine;
    }

    [Theory]
    [InlineData(RunConstants.RelicMoltenEgg, CardType.Attack)]
    [InlineData(RunConstants.RelicToxicEgg, CardType.Skill)]
    [InlineData(RunConstants.RelicFrozenEgg, CardType.Power)]
    public void AnEggClaimedFirstUpgradesTheMatchingOptionsStillWaiting(int egg, CardType type)
    {
        int attack = Card(CardType.Attack);
        int skill = Card(CardType.Skill);
        int power = Card(CardType.Power);
        var engine = EliteScreen(egg, [attack, skill, power]);

        // The relic is the screen's first item (no gold, no potion), the card reward its next.
        Assert.True(engine.Step(0, -1, out _, out _, out _) >= 0);

        Assert.Contains(engine.State.Relics, relic => relic.DefId == egg);
        for (int i = 0; i < 3; i++)
        {
            var matches = GeneratedData.Cards.Get(engine.State.RewardCards[i]).Type == type;
            Assert.Equal(matches, engine.State.RewardUpgraded[i]);
        }

        // And the offer the player then opens shows it.
        Assert.True(engine.Step(0, -1, out _, out _, out _) >= 0);
        Assert.Equal(RunPhase.CardReward, engine.State.Phase);
    }

    /// <summary>The egg's rule is "upgradable and not yet upgraded": it never undoes or doubles.</summary>
    [Fact]
    public void AnOptionThatRolledUpgradedIsLeftAlone()
    {
        int attack = Card(CardType.Attack);
        int second = Other(CardType.Attack, attack);
        var engine = EliteScreen(RunConstants.RelicMoltenEgg, [attack, second, Card(CardType.Skill)]);
        engine.State.RewardUpgraded[0] = true;

        engine.Step(0, -1, out _, out _, out _);

        Assert.True(engine.State.RewardUpgraded[0]);
        Assert.True(engine.State.RewardUpgraded[1]);
        Assert.False(engine.State.RewardUpgraded[2]);
    }

    /// <summary>
    /// Once the card reward is open (or answered) it is no longer waiting: the reward being chosen
    /// from ends with the choice, and the egg that arrives later changes the DECK only.
    /// </summary>
    [Fact]
    public void ARewardAlreadyAnsweredIsNotTouched()
    {
        int attack = Card(CardType.Attack);
        var engine = EliteScreen(RunConstants.RelicMoltenEgg, [attack, Card(CardType.Skill), Card(CardType.Power)]);
        engine.State.RewardCardPending = false;

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicMoltenEgg);

        Assert.All(engine.State.RewardUpgraded, upgraded => Assert.False(upgraded));
    }

    /// <summary>
    /// The Hunt's extra card reward is a second CardReward in the same set, built (and
    /// subscribed) with the first: it queues behind the screen's own and the egg reaches it too.
    /// </summary>
    [Fact]
    public void AnEggAlsoUpgradesTheQueuedCardRewards()
    {
        int attack = Card(CardType.Attack);
        var engine = EliteScreen(RunConstants.RelicMoltenEgg, [attack, Card(CardType.Skill), Card(CardType.Power)]);
        engine.State.PendingCardOffers.Add(
            CardOffer.Plain([Card(CardType.Skill), attack, Card(CardType.Power)])
        );

        engine.Step(0, -1, out _, out _, out _);

        var queued = engine.State.PendingCardOffers[0];
        Assert.False(queued.Upgraded[0]);
        Assert.True(queued.Upgraded[1]);
        Assert.False(queued.Upgraded[2]);
    }

    /// <summary>
    /// Lasting Candy's fourth option is one of the reward's options: the late hooks run over all
    /// four, so Frozen Egg upgrades the Power the candy added -- whether the egg arrives after the
    /// screen is built or was already held when it was.
    /// </summary>
    [Fact]
    public void AnEggUpgradesLastingCandysFourthOptionToo()
    {
        int power = Card(CardType.Power);
        var engine = EliteScreen(RunConstants.RelicFrozenEgg, [Card(CardType.Attack), Card(CardType.Skill), Other(CardType.Power, power)]);
        engine.State.LastingCandyRewardCard = power;
        engine.State.LastingCandyRewardUpgraded = false;

        engine.Step(0, -1, out _, out _, out _);

        Assert.True(engine.State.LastingCandyRewardUpgraded);
    }

    [Fact]
    public void AnEggAlreadyHeldUpgradesLastingCandysFourthOptionWhenTheScreenIsBuilt()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunNonCombatEffects.NamedRelic("LastingCandy"));
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicFrozenEgg);
        for (int i = 0; i < 2; i++)
        {
            RelicEffects.CountCombatForLastingCandy(engine.State);
        }

        RunRewardGenerator.PopulateCardReward(engine.State);

        Assert.NotEqual(0, engine.State.LastingCandyRewardCard);
        Assert.True(engine.State.LastingCandyRewardUpgraded);
    }

    /// <summary>
    /// An egg that comes off a screen with NO card reward waiting changes nothing about the
    /// rewards (nothing to see it), and the hook is silent for any other relic.
    /// </summary>
    [Fact]
    public void OtherRelicsLeaveTheWaitingOptionsAlone()
    {
        var engine = EliteScreen(RunConstants.RelicWingedBoots, [Card(CardType.Attack), Card(CardType.Skill), Card(CardType.Power)]);

        engine.Step(0, -1, out _, out _, out _);

        Assert.All(engine.State.RewardUpgraded, upgraded => Assert.False(upgraded));
    }
}
