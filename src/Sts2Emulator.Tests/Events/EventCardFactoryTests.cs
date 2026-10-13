using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The events that offer cards make them through <c>CardFactory.CreateForReward</c>, and the
/// factory is where the relics that touch an offer act: Silver Crucible (a <c>CardReward</c>
/// only), the egg relics (any set unless <c>NoHookUpgrades</c>), Wing Charm and Glitter. The
/// emulator's events rolled bare ids and skipped all of it (Brain Leech's Rip with a Silver
/// Crucible: 36 real runs, the three colourless cards arrived plain).
/// </summary>
[CoversEvent("BrainLeech")]
[CoversEvent("RoomFullOfCheese")]
[CoversEvent("ColorfulPhilosophers")]
[CoversEvent("InfestedAutomaton")]
[CoversEvent("EndlessConveyor")]
[CoversEvent("Trial")]
public class EventCardFactoryTests
{
    private static RunEngine At(int eventId, string seed, params RelicInstance[] relics)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        engine.State.Relics.AddRange(relics);
        RunNonCombatEffects.BeginEvent(engine.State, eventId);
        return engine;
    }

    private static RelicInstance Crucible => new(RunConstants.RelicSilverCrucible, Counter: 3);

    private static CardType TypeOf(int cardId) => GeneratedData.Cards.Get(cardId).Type;

    private static int CrucibleUses(RunState state) =>
        state.Relics.Single(r => r.DefId == RunConstants.RelicSilverCrucible).Counter;

    // ── Brain Leech: Rip is a CardReward, Share Knowledge a bare grid ─────────

    [Theory]
    [InlineData("ABB6PCF77A")]
    [InlineData("ABCDEF")]
    [InlineData("Q95K0ZL072")]
    public void RipWithSilverCrucibleOffersThreeUpgradedCardsOffTheSameDraws(string seed)
    {
        var plain = At(RunConstants.EventBrainLeech, seed);
        var crucible = At(RunConstants.EventBrainLeech, seed, Crucible);
        Assert.Equal(0, plain.Step(1, -1, out _, out _, out _));
        Assert.Equal(0, crucible.Step(1, -1, out _, out _, out _));

        // The Crucible changes the cards on the screen, never which cards they are or how
        // many draws it took.
        Assert.Equal(plain.State.RewardCards, crucible.State.RewardCards);
        Assert.Equal(
            plain.State.PlayerRng.Rewards.CallCount,
            crucible.State.PlayerRng.Rewards.CallCount
        );
        Assert.All(plain.State.RewardUpgraded, upgraded => Assert.False(upgraded));
        for (int i = 0; i < crucible.State.RewardCards.Length; i++)
        {
            Assert.Equal(
                GeneratedData.Cards.Get(crucible.State.RewardCards[i]).Upgradable,
                crucible.State.RewardUpgraded[i]
            );
        }

        Assert.Equal(2, CrucibleUses(crucible.State));
    }

    [Theory]
    [InlineData("ABCDEF")]
    [InlineData("ABB6PCF77A")]
    [InlineData("Q95K0ZL072")]
    [InlineData("M4HT4KF030")]
    public void ShareKnowledgeLeavesTheCrucibleAloneButTheEggsUpgrade(string seed)
    {
        var engine = At(
            RunConstants.EventBrainLeech,
            seed,
            Crucible,
            new RelicInstance(RunConstants.RelicMoltenEgg)
        );
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        // No IsCardReward on the grid's options: the Crucible keeps its use and upgrades nothing
        // it was not already going to; Molten Egg upgrades the Attacks.
        Assert.Equal(3, CrucibleUses(engine.State));
        var cards = engine.State.PendingOfferCards;
        Assert.Equal(RunConstants.BrainLeechCardChoices, cards.Length);
        for (int i = 0; i < cards.Length; i++)
        {
            bool expected =
                TypeOf(cards[i]) == CardType.Attack && GeneratedData.Cards.Get(cards[i]).Upgradable;
            Assert.Equal(expected, engine.State.PendingOfferUpgraded[i]);
        }
    }

    [Fact]
    public void RipUpgradesWhatTheEggsUpgrade()
    {
        int attacks = 0;
        foreach (string seed in new[] { "ABCDEF", "ABB6PCF77A", "Q95K0ZL072", "M4HT4KF030" })
        {
            var engine = At(
                RunConstants.EventBrainLeech,
                seed,
                new RelicInstance(RunConstants.RelicMoltenEgg),
                new RelicInstance(RunConstants.RelicToxicEgg)
            );
            Assert.Equal(0, engine.Step(1, -1, out _, out _, out _));
            for (int i = 0; i < engine.State.RewardCards.Length; i++)
            {
                int id = engine.State.RewardCards[i];
                bool expected =
                    TypeOf(id) is CardType.Attack or CardType.Skill
                    && GeneratedData.Cards.Get(id).Upgradable;
                attacks += TypeOf(id) == CardType.Attack ? 1 : 0;
                Assert.Equal(expected, engine.State.RewardUpgraded[i]);
            }
        }

        Assert.True(attacks > 0, "the seeds never offered an Attack");
    }

    // ── Wing Charm: one option, off Niche, on every factory set ───────────────

    [Fact]
    public void ShareKnowledgeWithWingCharmDrawsNicheOnceAndEnchantsThatOption()
    {
        var engine = At(
            RunConstants.EventBrainLeech,
            "ABCDEF",
            new RelicInstance(RelicEffects.WingCharm)
        );
        int nicheBefore = engine.State.Rng.Niche.CallCount;
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(nicheBefore + 1, engine.State.Rng.Niche.CallCount);
        int swift = Assert.Single(engine.State.PendingOfferSwiftIndexes);
        Assert.InRange(swift, 0, engine.State.PendingOfferCards.Length - 1);
        int taken = engine.State.PendingOfferCards[swift];
        Assert.Equal(0, engine.Step(swift, -1, out _, out _, out _));
        Assert.Contains(
            engine.State.Deck,
            card => card.DefId == taken && card.Enchantment == Enchantment.Swift
        );
    }

    // ── The set itself ────────────────────────────────────────────────────────

    [Fact]
    public void TheFlagsDecideWhichHooksTouchASet()
    {
        var engine = At(RunConstants.EventBrainLeech, "ABCDEF", Crucible);
        int[] pool = RunRewardGenerator.OwnCardPool(engine.State).ToArray();

        // CardReward: the Crucible upgrades, and counts the set.
        var reward = RunRewardGenerator.CreateForReward(
            engine.State,
            3,
            pool,
            CardCreationFlags.NoUpgradeRoll | CardCreationFlags.IsCardReward
        );
        Assert.All(reward.Upgraded, upgraded => Assert.True(upgraded));
        Assert.Equal(2, CrucibleUses(engine.State));

        // NoModifyHooks: nothing runs, the use stays.
        var bare = RunRewardGenerator.CreateForReward(
            engine.State,
            3,
            pool,
            CardCreationFlags.NoUpgradeRoll
                | CardCreationFlags.IsCardReward
                | CardCreationFlags.NoModifyHooks
        );
        Assert.All(bare.Upgraded, upgraded => Assert.False(upgraded));
        Assert.Equal(2, CrucibleUses(engine.State));

        // The upgrade roll is a Rewards draw per card unless NoUpgradeRoll says otherwise.
        var rewards = engine.State.PlayerRng.Rewards;
        int before = rewards.CallCount;
        RunRewardGenerator.CreateForReward(
            engine.State,
            3,
            pool,
            CardCreationFlags.NoModifyHooks,
            CardRarity.Common
        );
        Assert.Equal(before + 6, rewards.CallCount);
        RunRewardGenerator.CreateForReward(
            engine.State,
            3,
            pool,
            CardCreationFlags.NoModifyHooks | CardCreationFlags.NoUpgradeRoll,
            CardRarity.Common
        );
        Assert.Equal(before + 9, rewards.CallCount);
    }

    [Fact]
    public void ColorfulPhilosophersCrucibleUpgradesAllThreeSetsAndIsSpent()
    {
        var engine = At(RunConstants.EventColorfulPhilosophers, "ABCDEF", Crucible);
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        // Three CardRewards, one use each, every option upgraded.
        Assert.Equal(0, CrucibleUses(engine.State));
        Assert.Equal(2, engine.State.PendingCardOffers.Count);
        var offers = new[]
        {
            new CardOffer(engine.State.RewardCards, engine.State.RewardUpgraded),
        }.Concat(engine.State.PendingCardOffers);
        foreach (var offer in offers)
        {
            for (int i = 0; i < offer.Cards.Length; i++)
            {
                Assert.Equal(GeneratedData.Cards.Get(offer.Cards[i]).Upgradable, offer.Upgraded[i]);
            }
        }
    }

    [Theory]
    [InlineData("ABCDEF")]
    [InlineData("ABB6PCF77A")]
    public void GorgeGivesTheEggsTheirUpgradeButNotTheCrucible(string seed)
    {
        var engine = At(
            RunConstants.EventRoomFullOfCheese,
            seed,
            Crucible,
            new RelicInstance(RunConstants.RelicMoltenEgg)
        );
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(3, CrucibleUses(engine.State));
        var cards = engine.State.PendingOfferCards;
        Assert.Equal(RunConstants.GorgeCardChoices, cards.Length);
        Assert.Contains(cards, id => TypeOf(id) == CardType.Attack);
        for (int i = 0; i < cards.Length; i++)
        {
            Assert.Equal(
                TypeOf(cards[i]) == CardType.Attack && GeneratedData.Cards.Get(cards[i]).Upgradable,
                engine.State.PendingOfferUpgraded[i]
            );
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void InfestedAutomatonsCardIsMadeThroughTheFactory(int option)
    {
        // Wing Charm draws once off Niche for the one-card set, and the card arrives with
        // Swift whenever Swift can be put on it.
        var engine = At(
            RunConstants.EventInfestedAutomaton,
            "ABCDEF",
            new RelicInstance(RelicEffects.WingCharm)
        );
        int nicheBefore = engine.State.Rng.Niche.CallCount;
        int deckBefore = engine.State.Deck.Count;
        Assert.Equal(0, engine.Step(option, -1, out _, out _, out _));

        Assert.Equal(deckBefore + 1, engine.State.Deck.Count);
        var card = engine.State.Deck[^1];
        bool canSwift = Enchantments.CanEnchant(card with { Enchantment = Enchantment.None }, Enchantment.Swift);
        Assert.True(canSwift, GeneratedData.Cards.Get(card.DefId).Name);
        Assert.Equal(canSwift ? 1 : 0, engine.State.Rng.Niche.CallCount - nicheBefore);
        Assert.Equal(canSwift ? Enchantment.Swift : Enchantment.None, card.Enchantment);
    }

    [Fact]
    public void TheTrialsNondescriptVerdictOffersTwoCrucibleSets()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        engine.State.Relics.Add(Crucible);
        engine.State.Phase = RunPhase.Event;
        engine.State.EventId = RunConstants.EventTrial;
        engine.State.EventPage = 3;
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        // Two CardRewards, one Crucible use each: both screens arrive upgraded.
        Assert.Equal(1, CrucibleUses(engine.State));
        var offers = new[]
        {
            new CardOffer(engine.State.RewardCards, engine.State.RewardUpgraded),
        }.Concat(engine.State.PendingCardOffers);
        Assert.Equal(2, offers.Count());
        foreach (var offer in offers)
        {
            for (int i = 0; i < offer.Cards.Length; i++)
            {
                Assert.Equal(GeneratedData.Cards.Get(offer.Cards[i]).Upgradable, offer.Upgraded[i]);
            }
        }
    }

    [Fact]
    public void SeaGlassMakesThreeSetsAndWingCharmEnchantsOneOptionInEach()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.Relics.Add(new RelicInstance(RelicEffects.WingCharm));
        RunNonCombatEffects.GenerateAncientOptions(engine.State, RunConstants.AncientOrobas);
        int nicheBefore = engine.State.Rng.Niche.CallCount;
        int rewardsBefore = engine.State.PlayerRng.Rewards.CallCount;

        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicSeaGlass);

        // Fifteen picks and no upgrade rolls; one Niche draw per five-card set.
        Assert.Equal(rewardsBefore + 15, engine.State.PlayerRng.Rewards.CallCount);
        Assert.Equal(15, engine.State.PendingOfferCards.Length);
        Assert.Equal(nicheBefore + 3, engine.State.Rng.Niche.CallCount);
        var swift = engine.State.PendingOfferSwiftIndexes;
        Assert.Equal(3, swift.Count);
        Assert.Equal(new[] { 0, 1, 2 }, swift.Select(index => index / 5));

        // Taking an earlier card shifts the others down by one.
        engine.State.Phase = RunPhase.TransformSelect;
        int other = Enumerable.Range(0, 15).First(i => !swift.Contains(i));
        var expected = swift.Select(index => index > other ? index - 1 : index).ToList();
        Assert.Equal(0, engine.Step(other, -1, out _, out _, out _));
        Assert.Equal(expected, engine.State.PendingOfferSwiftIndexes);
    }

    [Fact]
    public void EndlessConveyorsFriedEelGoesThroughTheFactory()
    {
        // The colourless card goes to the deck as the factory leaves it: Wing Charm's Niche draw
        // is made and the card arrives enchanted (it is the only option there is).
        var engine = At(
            RunConstants.EventEndlessConveyor,
            "ABCDEF",
            new RelicInstance(RelicEffects.WingCharm)
        );
        int nicheBefore = engine.State.Rng.Niche.CallCount;
        int deckBefore = engine.State.Deck.Count;
        RunNonCombatEffects.ApplyEndlessConveyorDish(
            engine.State,
            RunNonCombatEffects.DishFriedEel
        );

        Assert.Equal(deckBefore + 1, engine.State.Deck.Count);
        Assert.Equal(nicheBefore + 1, engine.State.Rng.Niche.CallCount);
        Assert.Equal(Enchantment.Swift, engine.State.Deck[^1].Enchantment);
    }
}
