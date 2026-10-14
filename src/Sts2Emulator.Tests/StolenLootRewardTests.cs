using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A combat room's <c>ExtraRewards</c> go through the same populate pass as its own rewards.
/// <c>RewardsSet.WithRewardsFromRoom</c> appends them behind the room's list (gold, potion,
/// cards, an elite's relic) and <c>GenerateWithoutOffering</c> calls <c>Populate()</c> on
/// every reward in that order: a fixed-amount <c>GoldReward</c> still draws
/// <c>Rewards.NextInt(amount, amount + 1)</c>, a <c>CardReward</c> rolls its cards, and a
/// <c>SpecialCardReward</c> draws nothing and is a row the player claims or leaves.
/// </summary>
/// <remarks>
/// The extras are a Heist's gold (<c>HeistPower.BeforeDeath</c>), a Thieving Hopper's stolen
/// card (<c>SwipePower.BeforeDeath</c>), The Hunt's card reward, Royalties' gold
/// (<c>RoyaltiesPower.AfterCombatEnd</c>, in CombatEndRewardPowerTests), Punch Off's relic
/// and potion (PunchOffFightRewardsTests) and the Lantern Key. And
/// <c>EncounterModel.CalculateGoldProportion</c> is <c>1 - escaped creatures / spawned
/// kinds</c>, so an escaped Hopper's room has no gold row at all.
/// </remarks>
public class StolenLootRewardTests
{
    private static RunEngine AtFirstCombat(string seed = "QS2GYXRKWN")
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        for (int i = 0; i < 16 && engine.State.Phase != RunPhase.Map; i++)
        {
            engine.Step(0, -1, out _, out _, out _);
        }

        int action = Array.FindIndex(
            engine.State.MapNodeTypes,
            nodeType => nodeType == RunConstants.NodeNormal
        );
        Assert.Equal(0, engine.Step(action, -1, out _, out _, out _));
        Assert.Equal(RunPhase.Combat, engine.State.Phase);
        return engine;
    }

    private static int SpecialCardAction(RunState s) =>
        (s.RewardGold != 0 ? 1 : 0)
        + (s.RewardPotion != 0 ? 1 : 0)
        + (s.RelicReward != 0 ? 1 : 0)
        + RunRewardGenerator.CardItemCount(s);

    /// <summary>
    /// The Heist's row draws its value AFTER the room's own: the room's gold, potion and cards
    /// are the control's, and the stream is one further on. It was queued with no draw, so
    /// every Rewards value after a Gremlin Merc fight came one early.
    /// </summary>
    [Fact]
    public void AHeistsGoldDrawsOneRewardsValueAfterTheRoomsOwn()
    {
        var control = AtFirstCombat();
        var heist = AtFirstCombat();

        control.DebugResolveCombat();
        heist.DebugResolveCombat(stolenBackGold: 40);

        Assert.Equal(control.State.RewardGold, heist.State.RewardGold);
        Assert.Equal(control.State.RewardPotion, heist.State.RewardPotion);
        Assert.Equal(control.State.RewardCards, heist.State.RewardCards);
        Assert.Equal(new[] { 40 }, heist.State.PendingGoldRewards);
        Assert.Equal(
            control.State.PlayerRng.Rewards.CallCount + 1,
            heist.State.PlayerRng.Rewards.CallCount
        );
    }

    /// <summary>
    /// The Hopper's death offers the card back as a row; it is not in the deck until the
    /// player claims it, and the claim is a deck add like any other -- Lucky Fysh pays.
    /// </summary>
    [Fact]
    public void AStolenCardIsARowTheClaimPutsBackInTheDeck()
    {
        var engine = AtFirstCombat();
        engine.State.Relics.Add(new RelicInstance(RelicEffects.LuckyFysh));
        var card = engine.State.Deck[^1];
        int copies = engine.State.Deck.Count(c => c == card);
        var control = AtFirstCombat();
        control.DebugResolveCombat();

        engine.DebugResolveCombat(stolenCards: [card], stolenBackCards: [card]);

        Assert.Equal(new[] { card }, engine.State.RewardSpecialCards);
        Assert.Equal(copies - 1, engine.State.Deck.Count(c => c == card));
        // A SpecialCardReward's Populate draws nothing.
        Assert.Equal(control.State.PlayerRng.Rewards.CallCount, engine.State.PlayerRng.Rewards.CallCount);
        // One more action than the control's screen, after its card item.
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        int action = SpecialCardAction(engine.State);
        Assert.Equal(1, mask[action]);

        int gold = engine.State.Gold;
        Assert.True(RunRewardGenerator.ClaimRewardAtIndex(engine.State, action));

        Assert.Empty(engine.State.RewardSpecialCards);
        Assert.Equal(copies, engine.State.Deck.Count(c => c == card));
        Assert.Equal(gold + 15, engine.State.Gold);
    }

    /// <summary>Left on the screen, the card is gone: the deck lost it at the steal.</summary>
    [Fact]
    public void AStolenCardLeftOnTheScreenIsLost()
    {
        var engine = AtFirstCombat();
        var card = engine.State.Deck[^1];
        int copies = engine.State.Deck.Count(c => c == card);

        engine.DebugResolveCombat(stolenCards: [card], stolenBackCards: [card]);
        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));

        Assert.Empty(engine.State.RewardSpecialCards);
        Assert.Equal(copies - 1, engine.State.Deck.Count(c => c == card));
    }

    /// <summary>
    /// The row is the card's DECK version: an upgrade the fight gave its copy (Armaments) is
    /// the fight's, and the deck card it was dealt from is the one that comes back.
    /// </summary>
    [Fact]
    public void TheRowOffersTheDeckVersionNotTheFightsCopy()
    {
        var engine = AtFirstCombat();
        var deckCard = engine.State.Deck.First(c => !c.Upgraded);
        var fightCopy = deckCard with { Upgraded = true, CostBump = 1 };

        engine.DebugResolveCombat(stolenCards: [fightCopy], stolenBackCards: [fightCopy]);

        Assert.Equal(new[] { deckCard }, engine.State.RewardSpecialCards);
    }

    /// <summary>
    /// An escaped Hopper keeps the card -- gone from the deck, no row -- and, alone in its
    /// encounter, takes the room's gold with it: <c>1 - 1/1</c> is no <c>GoldReward</c> and no
    /// draw, so the potion and the cards come one value earlier.
    /// </summary>
    [Fact]
    public void AnEscapedHopperTakesTheCardAndTheRoomsGold()
    {
        var control = AtFirstCombat();
        var escaped = AtFirstCombat();
        foreach (var e in new[] { control, escaped })
        {
            var combat = e.State.ActiveCombat!;
            combat.EncounterId = (int)CombatFactory.ActOneEncounter.ThievingHopper;
            combat.Enemies.RemoveRange(1, combat.Enemies.Count - 1);
        }

        var card = escaped.State.Deck[^1];
        int copies = escaped.State.Deck.Count(c => c == card);
        control.DebugResolveCombat();
        escaped.DebugResolveCombat(escapedEnemies: 1, stolenCards: [card]);

        Assert.NotEqual(0, control.State.RewardGold);
        Assert.Equal(0, escaped.State.RewardGold);
        Assert.Empty(escaped.State.PendingGoldRewards);
        Assert.Empty(escaped.State.RewardSpecialCards);
        Assert.Equal(copies - 1, escaped.State.Deck.Count(c => c == card));
        Assert.Equal(
            control.State.PlayerRng.Rewards.CallCount - 1,
            escaped.State.PlayerRng.Rewards.CallCount
        );
    }

    /// <summary>A roster of the seed's first enemy and a second creature of another kind or the same.</summary>
    private static RunEngine AtFirstCombatWithTwo(bool sameKind)
    {
        var engine = AtFirstCombat();
        var combat = engine.State.ActiveCombat!;
        combat.Enemies.RemoveRange(1, combat.Enemies.Count - 1);
        var first = combat.Enemies[0];
        int other = first.DefId == KE.SludgeSpinner ? KE.GasBomb : KE.SludgeSpinner;
        combat.Enemies.Add(
            sameKind
                ? first.Clone()
                : CombatFactory.CreateEnemy(other, new Random(0), new Intent(IntentType.Attack, 9))
        );
        return engine;
    }

    /// <summary>
    /// <c>SpawnedEnemies</c> holds KINDS of monster: <c>EncounterModel.OnCreatureSpawned</c>
    /// adds a creature's <c>Monster.CanonicalInstance</c> only if the list lacks it. One of
    /// two creatures of one kind escaping is <c>1 - 1/1</c>: no gold row and no draw.
    /// </summary>
    [Fact]
    public void OneOfTwoOfAKindEscapingLeavesNoGold()
    {
        var control = AtFirstCombatWithTwo(sameKind: true);
        var engine = AtFirstCombatWithTwo(sameKind: true);

        control.DebugResolveCombat();
        engine.DebugResolveCombat(escapedEnemies: 1);

        Assert.NotEqual(0, control.State.RewardGold);
        Assert.Equal(0, engine.State.RewardGold);
        Assert.Equal(
            control.State.PlayerRng.Rewards.CallCount - 1,
            engine.State.PlayerRng.Rewards.CallCount
        );
    }

    /// <summary>One of two KINDS escaping halves the room's gold bounds; the row is still rolled.</summary>
    [Fact]
    public void OneEscapeAmongTwoKindsHalvesTheGold()
    {
        var engine = AtFirstCombatWithTwo(sameKind: false);

        engine.DebugResolveCombat(escapedEnemies: 1);

        Assert.InRange(engine.State.RewardGold, 4, 10);
    }

    /// <summary>
    /// A creature the fight swept out of the roster still took part, as its kind and not as
    /// one more creature: a corpse of the escaper's kind leaves one kind (no gold), a corpse
    /// of another kind two (half). The sweep counted creatures, so both came out at half.
    /// </summary>
    [Theory]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void ASweptCorpseCountsAsItsKind(bool sameKind, bool paysGold)
    {
        var engine = AtFirstCombatWithTwo(sameKind);
        var combat = engine.State.ActiveCombat!;
        combat.Enemies[1].Hp = 0;

        // End the turn: the corpse sweep takes the dead creature out of the roster.
        engine.Step(combat.Hand.Count, -1, out _, out _, out _);
        Assert.Equal(RunPhase.Combat, engine.State.Phase);
        Assert.Single(engine.State.ActiveCombat!.Enemies);

        engine.DebugResolveCombat(escapedEnemies: 1);

        Assert.Equal(paysGold, engine.State.RewardGold != 0);
    }

    /// <summary>
    /// The Hunt's extra CardReward is one of the room's ExtraRewards, so its three cards are
    /// rolled AFTER the room's potion and cards: the room's own offer is the control's. It
    /// was rolled before them, while its comment said "after".
    /// </summary>
    [Fact]
    public void TheHuntsOfferIsRolledAfterTheRoomsOwnRewards()
    {
        var control = AtFirstCombat();
        var hunt = AtFirstCombat();
        hunt.State.ActiveCombat!.ExtraCardRewards = 1;

        control.DebugResolveCombat();
        hunt.DebugResolveCombat();

        Assert.Equal(control.State.RewardPotion, hunt.State.RewardPotion);
        Assert.Equal(control.State.RewardCards, hunt.State.RewardCards);
        Assert.Single(hunt.State.PendingCardOffers);
    }

    /// <summary>
    /// The Lantern Key's knight fight hangs a <c>SpecialCardReward(LanternKey)</c> on its room.
    /// It took the card screen's first slot, which the room's own card roll then cleared, so
    /// the key never reached the screen; now it is its own row beside an untouched offer.
    /// </summary>
    [Fact]
    public void TheLanternKeyIsAClaimableRowBesideTheRoomsCards()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventTheLanternKey);
        Assert.Equal(0, engine.Step(1, -1, out _, out _, out _)); // Keep The Key
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _)); // the fight
        Assert.Equal(0, engine.DebugResolveCombat());

        int key = RunNonCombatEffects.NamedCard("LanternKey");
        Assert.Equal(new[] { new CardInstance(key, false) }, engine.State.RewardSpecialCards);
        Assert.DoesNotContain(key, engine.State.RewardCards);
        Assert.All(engine.State.RewardCards, id => Assert.NotEqual(0, id));

        Assert.True(
            RunRewardGenerator.ClaimRewardAtIndex(engine.State, SpecialCardAction(engine.State))
        );
        Assert.Contains(engine.State.Deck, c => c.DefId == key);
    }

    /// <summary>The rows travel with a clone and do not share their list.</summary>
    [Fact]
    public void TheRowsTravelWithAClone()
    {
        var engine = AtFirstCombat();
        var card = engine.State.Deck[^1];
        engine.DebugResolveCombat(stolenCards: [card], stolenBackCards: [card]);

        var clone = engine.State.Clone();

        Assert.Equal(engine.State.RewardSpecialCards, clone.RewardSpecialCards);
        Assert.NotSame(engine.State.RewardSpecialCards, clone.RewardSpecialCards);
    }
}
