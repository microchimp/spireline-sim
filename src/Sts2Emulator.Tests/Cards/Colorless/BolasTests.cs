using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

public class BolasTests
{
    [Fact]
    public void DamagesTargetAndReturnsBeforeNextDraw()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(CL.Bolas, false)];
        state.DrawPile =
        [
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.DefendIronclad, false),
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.DefendIronclad, false),
            new CardInstance(IC.StrikeIronclad, false),
        ];
        state.DiscardPile.Clear();
        state.Energy = 0;
        state.Enemies =
        [
            new EnemyState
            {
                DefId = 16,
                Hp = 20,
                MaxHp = 20,
                CurrentIntent = new Intent(IntentType.Defend, 0),
                Buffs = [],
            },
        ];

        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(17, state.Enemies[0].Hp);
        Assert.Contains(state.DiscardPile, card => card.DefId == CL.Bolas);
        Assert.Single(state.ReturnToHandBeforeDraw);

        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(CL.Bolas, state.Hand[0].DefId);
        Assert.Equal(6, state.Hand.Count);
        Assert.DoesNotContain(state.DiscardPile, card => card.DefId == CL.Bolas);
        Assert.Empty(state.ReturnToHandBeforeDraw);
    }

    [Fact]
    public void UpgradedUsesUpgradedDamageAndReturnsUpgraded()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(CL.Bolas, true)];
        state.DrawPile.Clear();
        state.DiscardPile.Clear();
        state.Energy = 0;
        state.Enemies =
        [
            new EnemyState
            {
                DefId = 16,
                Hp = 20,
                MaxHp = 20,
                CurrentIntent = new Intent(IntentType.Defend, 0),
                Buffs = [],
            },
        ];

        CombatEngine.Step(state, 0, new Random(0));
        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(16, state.Enemies[0].Hp);
        Assert.Single(state.Hand);
        Assert.Equal(CL.Bolas, state.Hand[0].DefId);
        Assert.True(state.Hand[0].Upgraded);
    }

    [Fact]
    public void ReturnsOnceWhenAttackEffectIsDuplicated()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(IC.OneTwoPunch, false), new CardInstance(CL.Bolas, false)];
        state.DrawPile.Clear();
        state.DiscardPile.Clear();
        state.Energy = 1;
        state.Enemies =
        [
            new EnemyState
            {
                DefId = 16,
                Hp = 20,
                MaxHp = 20,
                CurrentIntent = new Intent(IntentType.Defend, 0),
                Buffs = [],
            },
        ];

        CombatEngine.Step(state, 0, new Random(0));
        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(14, state.Enemies[0].Hp);
        Assert.Single(state.ReturnToHandBeforeDraw);

        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(1, state.Hand.Count(card => card.DefId == CL.Bolas));
        Assert.Empty(state.ReturnToHandBeforeDraw);
    }

    // `Bolas.BeforeHandDraw` (`ThrummingHatchet.BeforeHandDraw` has the same body): the card
    // that finished a play last turn moves ITSELF to the hand, from whichever pile holds it,
    // and does nothing when it is already there. One move of one card; nothing is created.
    // Found by the per-combat card uid audit (docs/kits/Bolas.md): the return queue held one
    // note per PLAY and made a new card for a note no pile answered.

    /// <summary>
    /// The first test's combat (the card alone in hand, five cards to draw, energy 0, one
    /// defending enemy) with every card numbered, so the played card can be followed by uid.
    /// </summary>
    private static CombatState NumberedFight(
        int defId,
        int energy = 0,
        params CardInstance[] alsoInDraw
    )
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(defId, false)];
        state.DrawPile =
        [
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.DefendIronclad, false),
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.DefendIronclad, false),
            new CardInstance(IC.StrikeIronclad, false),
            .. alsoInDraw,
        ];
        state.DiscardPile.Clear();
        state.Energy = energy;
        state.Enemies =
        [
            new EnemyState
            {
                DefId = 16,
                Hp = 60,
                MaxHp = 60,
                CurrentIntent = new Intent(IntentType.Defend, 0),
                Buffs = [],
            },
        ];
        CardUids.NumberAll(state);
        return state;
    }

    /// <summary>Moves the card with <paramref name="uid"/> from the discard pile back into the hand, as a card that returns it mid-turn would.</summary>
    private static void ReturnFromDiscard(CombatState state, int uid, bool retainThisTurn = false)
    {
        int at = state.DiscardPile.FindIndex(card => card.UidOf() == uid);
        Assert.True(at >= 0, "the played card is in the discard pile");
        state.Hand.Add(state.DiscardPile[at] with { RetainThisTurn = retainThisTurn });
        state.DiscardPile.RemoveAt(at);
    }

    private static List<CardInstance> Copies(CombatState state, int defId) =>
        CardUids.LiveCards(state).Where(card => card.DefId == defId).ToList();

    private static void AssertPlayedTwiceInATurnReturnsOnce(int defId, int energy)
    {
        var state = NumberedFight(defId, energy);
        int uid = state.Hand[0].UidOf();
        var rng = new Random(0);

        CombatEngine.Step(state, 0, rng);
        ReturnFromDiscard(state, uid);
        CombatEngine.Step(state, 0, rng);
        CombatEngine.Step(state, state.Hand.Count, rng);

        var copies = Copies(state, defId);
        Assert.Single(copies);
        Assert.Equal(uid, copies[0].UidOf());
        Assert.Contains(state.Hand, card => card.UidOf() == uid);
        Assert.Empty(state.ReturnToHandBeforeDraw);
    }

    [Fact]
    public void PlayedTwiceInATurnReturnsOnce() =>
        AssertPlayedTwiceInATurnReturnsOnce(CL.Bolas, energy: 0);

    [Fact]
    public void ThrummingHatchetPlayedTwiceInATurnReturnsOnce() =>
        AssertPlayedTwiceInATurnReturnsOnce(CL.ThrummingHatchet, energy: 2);

    [Fact]
    public void ARetainedBolasIsNotReturnedAgain()
    {
        var state = NumberedFight(CL.Bolas);
        int uid = state.Hand[0].UidOf();
        var rng = new Random(0);

        CombatEngine.Step(state, 0, rng);
        ReturnFromDiscard(state, uid, retainThisTurn: true);
        CombatEngine.Step(state, state.Hand.Count, rng);

        var copies = Copies(state, CL.Bolas);
        Assert.Single(copies);
        Assert.Equal(uid, copies[0].UidOf());
        Assert.Equal(1, state.Hand.Count(card => card.DefId == CL.Bolas));
        Assert.Empty(state.ReturnToHandBeforeDraw);
    }

    [Fact]
    public void ReturnsThePlayedCopyNotAnotherOne()
    {
        var state = NumberedFight(CL.Bolas, 0, new CardInstance(CL.Bolas, false));
        int played = state.Hand[0].UidOf();
        int other = state.DrawPile[^1].UidOf();
        Assert.NotEqual(played, other);
        var rng = new Random(0);

        CombatEngine.Step(state, 0, rng);
        CombatEngine.Step(state, state.Hand.Count, rng);

        Assert.Equal(2, Copies(state, CL.Bolas).Count);
        Assert.Equal(CL.Bolas, state.Hand[0].DefId);
        Assert.Equal(played, state.Hand[0].UidOf());
        // The five draws take the top of the pile; the other Bolas is the sixth card and stays.
        Assert.Contains(state.DrawPile, card => card.UidOf() == other);
        Assert.Equal(1, state.Hand.Count(card => card.DefId == CL.Bolas));
    }

    [Fact]
    public void TheNextHandDisplayDoesNotPromiseACardAlreadyInTheHand()
    {
        var state = NumberedFight(CL.Bolas);
        int uid = state.Hand[0].UidOf();
        var rng = new Random(0);

        CombatEngine.Step(state, 0, rng);
        Assert.Equal(1, IncomingCount(state, CL.Bolas));

        ReturnFromDiscard(state, uid, retainThisTurn: true);

        Assert.Equal(0, IncomingCount(state, CL.Bolas));
    }

    /// <summary>How many cards of <paramref name="defId"/> the observation's incoming slots promise.</summary>
    private static int IncomingCount(CombatState state, int defId)
    {
        var obs = new int[CombatObservation.ObsSize];
        CombatObservation.Write(state, obs);
        int count = 0;
        for (int i = 0; i < CombatObservation.MaxIncomingCards; i++)
        {
            if (obs[CombatObservation.IncomingOffset + i * CombatObservation.CardSlotSize] == defId)
            {
                count++;
            }
        }

        return count;
    }
}
