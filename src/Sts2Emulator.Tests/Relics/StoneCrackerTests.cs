using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Rng;
using Xunit;

namespace Sts2Emulator.Tests;

// `StoneCracker.AfterRoomEntered(CombatRoom)`: upgrade CardsVar(2) cards of the draw pile,
// chosen by `Where(IsUpgradable).StableShuffle(Rng.CombatCardSelection).Take(2)`.
//
// The emulator took the first two upgradable cards in pile order and drew nothing. The pile
// is dealt shuffled, so that was "two random cards" off the WRONG stream: every card the
// selection stream picks afterwards sat one or more draws late, and the upgrades landed on
// other cards (night04 3JFLX5V4BH opened with Coordinate+ in the game and Defend+ here).
public class StoneCrackerTests
{
    private static readonly int[] Cards =
    [
        IC.StrikeIronclad,
        IC.DefendIronclad,
        IC.Bash,
        IC.Anger,
        IC.PommelStrike,
        IC.ShrugItOff,
        IC.Armaments,
        IC.TwinStrike,
    ];

    private static (CombatState State, CountingRandom Selection) Open(IEnumerable<int> pileOrder)
    {
        var selection = new CountingRandom(11);
        var state = new CombatState
        {
            NicheHpRng = new CountingRandom(0),
            CardSelectionRng = selection,
        };
        var deck = pileOrder.Select(id => new CardInstance(id, false)).ToArray();
        CombatFactory.Reset(
            state,
            new Random(0),
            deck,
            1,
            [RelicEffects.StoneCracker],
            80,
            80,
            [],
            0,
            deckPreShuffled: true
        );
        return (state, selection);
    }

    private static IEnumerable<int> Upgraded(CombatState state) =>
        state.DrawPile.Concat(state.Hand).Where(card => card.Upgraded).Select(card => card.DefId);

    [Fact]
    public void UpgradesTwoCardsAndRollsOncePerCandidateButTheFirst()
    {
        var (state, selection) = Open(Cards);

        Assert.Equal(2, Upgraded(state).Count());
        // Eight upgradable candidates, a Fisher-Yates over them: seven draws.
        Assert.Equal(Cards.Length - 1, selection.CallCount);
    }

    /// <summary>
    /// `StableShuffle` sorts before it shuffles, so the same cards in any pile order give the
    /// same two upgrades.
    /// </summary>
    [Fact]
    public void WhichTwoDoesNotDependOnThePileOrder()
    {
        var forward = Upgraded(Open(Cards).State).Order().ToList();
        var backward = Upgraded(Open(Cards.Reverse()).State).Order().ToList();
        var rotated = Upgraded(Open(Cards.Skip(3).Concat(Cards.Take(3))).State).Order().ToList();

        Assert.Equal(forward, backward);
        Assert.Equal(forward, rotated);
    }

    [Fact]
    public void AnAlreadyUpgradedCardIsNotACandidate()
    {
        var selection = new CountingRandom(11);
        var state = new CombatState
        {
            NicheHpRng = new CountingRandom(0),
            CardSelectionRng = selection,
        };
        CardInstance[] deck =
        [
            new(IC.StrikeIronclad, true),
            new(IC.DefendIronclad, false),
            new(IC.Bash, false),
        ];
        CombatFactory.Reset(
            state,
            new Random(0),
            deck,
            1,
            [RelicEffects.StoneCracker],
            80,
            80,
            [],
            0,
            deckPreShuffled: true
        );

        Assert.Equal(1, selection.CallCount);
        Assert.Equal(3, Upgraded(state).Count());
    }
}
