using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// The last of the shop pool: five relics that hook the REWARD list, plus Toolbox, whose
// screen is a combat one.

public class LavaLampTests
{
    /// <summary>
    /// Every upgradable option on the card-reward screen is upgraded after a combat that
    /// landed no unblocked damage — the same shape as Silver Crucible's, read once for the
    /// whole screen.
    /// </summary>
    [Fact]
    public void ACleanCombatUpgradesTheWholeScreen()
    {
        var clean = new RunEngine();
        clean.Reset("LAVA");
        clean.State.Relics.Add(new RelicInstance(RelicEffects.LavaLamp));
        clean.State.TookUnblockedDamageThisCombat = false;
        RunRewardGenerator.PopulateCardReward(clean.State);

        var upgradable = Enumerable
            .Range(0, clean.State.RewardCards.Length)
            .Where(i =>
                RunConstants.IsRunCardUpgradable(
                    new CardInstance(clean.State.RewardCards[i], false)
                )
            )
            .ToList();

        Assert.NotEmpty(upgradable);
        Assert.All(upgradable, i => Assert.True(clean.State.RewardUpgraded[i]));
    }

    /// <summary>
    /// `AfterDamageReceived` ignores Unblockable damage, so only damage the player could
    /// have blocked spoils it — which is why the flag is set from the blockable path.
    /// </summary>
    [Fact]
    public void UnblockableDamageDoesNotSpoilIt()
    {
        var fight = Fight.Hand().Energy(3).Enemy(hp: 200);
        fight.State.PlayerHp = 200;

        CardEffects.LoseHp(fight.State, 10);
        Assert.False(fight.State.TookUnblockedDamage);

        fight.State.PlayerBlock = 0;
        CardEffects.DealDamageToPlayer(fight.State, 10);
        Assert.True(fight.State.TookUnblockedDamage);
    }

    /// <summary>
    /// A monster's attack that gets through the block spoils it too -- the commonest way, and
    /// the one the enemy-attack path forgot (run referee, seed RXVFG7FZ86: after Vantom the
    /// emulator upgraded the whole screen, the game none of it). A fully blocked hit does not.
    /// </summary>
    [Fact]
    public void AMonstersHitThroughBlockSpoilsIt()
    {
        var fight = Fight.Hand().Energy(3).Enemy(hp: 200);
        fight.State.PlayerHp = 200;
        var enemy = fight.State.Enemies[0];

        fight.State.PlayerBlock = 20;
        EnemyAI.DealAttackDamageForTests(enemy, fight.State, 10);
        Assert.False(fight.State.TookUnblockedDamage);

        fight.State.PlayerBlock = 0;
        EnemyAI.DealAttackDamageForTests(enemy, fight.State, 10);
        Assert.True(fight.State.TookUnblockedDamage);
    }
}

public class DingyRugTests
{
    /// <summary>
    /// The colourless pool is ADDED to the reward pool, not swapped for it — so the
    /// character's own cards are still on offer alongside.
    /// </summary>
    [Fact]
    public void ColourlessCardsBecomeReachable()
    {
        var colourless = GeneratedData.CardPools.Colorless.ToArray().ToHashSet();
        bool sawColourless = false;

        for (int seed = 0; seed < 25 && !sawColourless; seed++)
        {
            var engine = new RunEngine();
            engine.Reset($"RUG{seed}");
            engine.State.Relics.Add(new RelicInstance(RelicEffects.DingyRug));
            RunRewardGenerator.PopulateCardReward(engine.State);
            sawColourless = engine.State.RewardCards.Any(colourless.Contains);
        }

        Assert.True(sawColourless, "no seed offered a colourless card with Dingy Rug");
    }

    [Fact]
    public void WithoutItTheyAreNot()
    {
        var colourless = GeneratedData.CardPools.Colorless.ToArray().ToHashSet();

        for (int seed = 0; seed < 25; seed++)
        {
            var engine = new RunEngine();
            engine.Reset($"RUG{seed}");
            RunRewardGenerator.PopulateCardReward(engine.State);
            Assert.DoesNotContain(engine.State.RewardCards, colourless.Contains);
        }
    }
}

public class WingCharmTests
{
    /// <summary>
    /// Finds a seed whose reward screen holds at least one Power -- a screen the relic
    /// enchants whether Swift is read as Powers-only or as the game's base rule.
    /// </summary>
    private static RunEngine ScreenWithAPower(bool withCharm)
    {
        for (int seed = 0; seed < 60; seed++)
        {
            var engine = new RunEngine();
            engine.Reset($"WING{seed}");
            if (withCharm)
            {
                engine.State.Relics.Add(new RelicInstance(RelicEffects.WingCharm));
            }

            RunRewardGenerator.PopulateCardReward(engine.State);
            bool hasPower = engine.State.RewardCards.Any(id =>
                id != 0 && GeneratedData.Cards.Get(id).Type == CardType.Power
            );
            if (hasPower)
            {
                return engine;
            }
        }

        Assert.Fail("no seed in 60 produced a reward screen holding a Power");
        return null!;
    }

    [Fact]
    public void OneRewardOptionCarriesSwift()
    {
        var engine = ScreenWithAPower(withCharm: true);

        Assert.InRange(engine.State.RewardEnchantIndex, 0, engine.State.RewardCards.Length - 1);
        Assert.Equal(Enchantment.Swift, engine.State.RewardEnchantment);
    }

    [Fact]
    public void WithoutItNoOptionIsEnchanted()
    {
        var engine = ScreenWithAPower(withCharm: false);

        Assert.Equal(-1, engine.State.RewardEnchantIndex);
    }

    /// <summary>The pick is filtered to a card Swift can go on (`canonicalSwift.CanEnchant`).</summary>
    [Fact]
    public void ItPicksACardThatCanTakeSwift()
    {
        var engine = ScreenWithAPower(withCharm: true);

        int index = engine.State.RewardEnchantIndex;
        var chosen = new CardInstance(engine.State.RewardCards[index], false);
        Assert.True(Enchantments.CanEnchant(chosen, Enchantment.Swift));
    }

    /// <summary>
    /// A screen with no Power on it is enchanted all the same. `WingCharm` filters the
    /// options through `canonicalSwift.CanEnchant`, and Swift overrides neither
    /// `CanEnchantCardType` nor `CanEnchant` -- an Attack or a Skill takes it as readily
    /// as a Power. The emulator used to read Swift as Powers-only and left these alone.
    /// </summary>
    [Fact]
    public void AScreenWithNoPowerIsEnchantedAllTheSame()
    {
        for (int seed = 0; seed < 60; seed++)
        {
            var engine = new RunEngine();
            engine.Reset($"NOPOWER{seed}");
            engine.State.Relics.Add(new RelicInstance(RelicEffects.WingCharm));
            RunRewardGenerator.PopulateCardReward(engine.State);

            bool hasPower = engine.State.RewardCards.Any(id =>
                id != 0 && GeneratedData.Cards.Get(id).Type == CardType.Power
            );
            if (!hasPower)
            {
                Assert.InRange(
                    engine.State.RewardEnchantIndex,
                    0,
                    engine.State.RewardCards.Length - 1
                );
                Assert.Equal(Enchantment.Swift, engine.State.RewardEnchantment);
                return;
            }
        }

        Assert.Fail("every seed in 60 produced a Power; the no-Power path went untested");
    }
}

public class OrreryTests
{
    /// <summary>
    /// FIVE whole card-reward screens, not five cards on one — `RewardsCmd.OfferCustom`
    /// takes a list of five `CardReward`s.
    /// </summary>
    [Fact]
    public void ItOwesFiveCardRewards()
    {
        var state = new RunState();

        var followUp = RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.Orrery);

        Assert.Equal(RunFollowUp.PreRolledCardReward, followUp);
        // All five are populated as the set is offered: one on the screen, four waiting.
        Assert.Equal(4, state.PendingCardOffers.Count);
        Assert.NotEqual(0, state.RewardCards[0]);
    }
}

public class CauldronTests
{
    [Fact]
    public void ItOwesFivePotionRewards()
    {
        var state = new RunState();

        var followUp = RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.Cauldron);

        Assert.Equal(RunFollowUp.BonusRelicRewards, followUp);
        Assert.Equal(5, state.PendingPotionRewards.Count);
    }
}

public class ToolboxTests
{
    /// <summary>Three DISTINCT colourless cards on turn one, one of which joins the hand.</summary>
    [Fact]
    public void ThreeColourlessCardsAreOffered()
    {
        var fight = Fight.WithRelics(RelicEffects.Toolbox);

        Assert.NotNull(fight.Pending);
        Assert.Equal(CardSelectionKind.GeneratedCardToHand, fight.Pending!.Kind);
        Assert.Equal(3, fight.Pending.GeneratedCandidates.Count);
        Assert.Equal(3, fight.Pending.GeneratedCandidates.Distinct().Count());

        var colourless = GeneratedData.CardPools.Colorless.ToArray().ToHashSet();
        Assert.All(fight.Pending.GeneratedCandidates, id => Assert.Contains(id, colourless));
    }

    /// <summary>
    /// Toolbox.cs: GetDistinctForCombat over the Colorless pool -- a shuffle of the whole
    /// FilterForCombat pool on the generation stream, first three. Not three rolls with
    /// duplicates rejected.
    /// </summary>
    [Fact]
    public void TheOfferIsTheFirstThreeOfAShuffleOfTheFilteredColorlessPool()
    {
        var fight = Fight.WithRelics(RelicEffects.Toolbox);
        fight.State.PendingSelection = null;
        fight.State.CardGenerationRng = new CountingRandom(3);
        var pool = GeneratedData.CardPools.Colorless.ToArray()
            .Where(id =>
            {
                var d = GeneratedData.Cards.Get(id);
                return d.CanBeGeneratedInCombat
                    && !d.MultiplayerOnly
                    && d.Rarity is not (CardRarity.Basic or CardRarity.Ancient or CardRarity.Event);
            })
            .ToList();
        var oracle = new CountingRandom(3);
        for (int i = pool.Count - 1; i > 0; i--)
        {
            int j = oracle.Next(i + 1);
            (pool[i], pool[j]) = (pool[j], pool[i]);
        }

        CardEffects.OpenToolboxSelection(fight.State, new Random(0));

        Assert.Equal(pool.Take(3).ToList(), fight.Pending!.GeneratedCandidates);
        Assert.Equal(pool.Count - 1, fight.State.CardGenerationRng!.CallCount);
    }

    [Fact]
    public void TheChosenCardJoinsTheHand()
    {
        var fight = Fight.WithRelics(RelicEffects.Toolbox);
        int picked = fight.Pending!.GeneratedCandidates[1];

        fight.Choose(1);

        Assert.Null(fight.Pending);
        Assert.Contains(fight.State.Hand, c => c.DefId == picked);
    }

    /// <summary>
    /// Only one screen can be up at a time: a Gambling Chip held alongside asks once the
    /// Toolbox pick is made, because the turn start waits on the pick (the Chip is an
    /// `AfterPlayerTurnStart` relic, after the hand draw that waits too).
    /// </summary>
    [Fact]
    public void AGamblingChipScreenFollowsIt()
    {
        var fight = Fight.WithRelics(RelicEffects.Toolbox, RelicEffects.GamblingChip);

        Assert.Equal(CardSelectionKind.GeneratedCardToHand, fight.Pending!.Kind);
        Assert.Equal((int)TurnStartPhase.BeforeHandDraw, fight.State.TurnStartHeldPhase);

        fight.Choose(0);

        Assert.NotNull(fight.Pending);
        Assert.Equal(CardSelectionKind.DiscardAnyThenDraw, fight.Pending!.Kind);
        Assert.Equal((int)TurnStartPhase.AfterPlayerTurnStart, fight.State.TurnStartHeldPhase);
    }

    /// <summary>
    /// `Toolbox.BeforeHandDraw` awaits its screen BEFORE `SetupPlayerTurn` draws the hand, so
    /// the pick is the first card in hand and the five follow it. The pick was added after the
    /// opening hand had been dealt (seed 8UAGRV20G8: the game's hand starts with Purity).
    /// </summary>
    [Fact]
    public void ThePickIsInHandBeforeTheHandDraw()
    {
        var fight = Fight.WithRelics(RelicEffects.Toolbox);

        Assert.Empty(fight.State.Hand);
        int picked = fight.Pending!.GeneratedCandidates[2];

        fight.Choose(2);

        Assert.Null(fight.Pending);
        Assert.Equal(6, fight.State.Hand.Count);
        Assert.Equal(picked, fight.State.Hand[0].DefId);
        Assert.Equal((int)TurnStartPhase.None, fight.State.TurnStartHeldPhase);
    }
}
