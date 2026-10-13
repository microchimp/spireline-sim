using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The events and relics that upgrade "random" deck cards do
/// <c>Deck.Where(IsUpgradable).StableShuffle(rng).Take(n)</c>: <c>List.Sort()</c> on the cards
/// (model id, then upgrade level -- an introsort, not stable above sixteen items) and then the
/// Fisher-Yates. Which of several equal cards is upgraded is the sort's doing, and with it the
/// order of upgraded and plain copies in the deck. The emulator sorted with a stable LINQ
/// <c>OrderBy</c> in two places and on the table id (which is not entry order) in two others.
/// </summary>
[CoversEvent("DoorsOfLightAndDark")]
[CoversEvent("BattlewornDummy")]
public class DeckStableShuffleTests
{
    private static int Card(string name) => GeneratedData.Cards.FindId(name)!.Value;

    /// <summary>A card as the game's <c>List&lt;CardModel&gt;.Sort()</c> sees it.</summary>
    private sealed class Probe(string entry, int level, int position) : IComparable<Probe>
    {
        public string Entry { get; } = entry;

        public int Level { get; } = level;

        public int Position { get; } = position;

        public int CompareTo(Probe? other)
        {
            int byEntry = string.CompareOrdinal(Entry, other!.Entry);
            return byEntry != 0 ? byEntry : Level.CompareTo(other.Level);
        }
    }

    private static RunState Deck(params string[] names)
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.Deck.Clear();
        foreach (string name in names)
        {
            engine.State.Deck.Add(new CardInstance(Card(name), Upgraded: false));
        }

        return engine.State;
    }

    private static string[] TwentyOneCards() =>
        [
            .. Enumerable.Repeat("StrikeIronclad", 5),
            .. Enumerable.Repeat("DefendIronclad", 4),
            "Bash",
            "Anger",
            "Anger",
            "ShrugItOff",
            "ShrugItOff",
            "ShrugItOff",
            "IronWave",
            "IronWave",
            "BodySlam",
            "BodySlam",
            "BodySlam",
            "AscendersBane",
        ];

    /// <summary>The positions the game would upgrade: the deck's upgradable cards, sorted, shuffled.</summary>
    private static List<int> Expected(RunState state, GameRng rng, int count)
    {
        var probes = new List<Probe>();
        for (int i = 0; i < state.Deck.Count; i++)
        {
            if (RunConstants.IsRunCardUpgradable(state.Deck[i]))
            {
                var def = GeneratedData.Cards.Get(state.Deck[i].DefId);
                probes.Add(new Probe(def.Entry, state.Deck[i].Upgraded ? 1 : 0, i));
            }
        }

        probes.Sort();
        rng.Shuffle(probes);
        return [.. probes.Take(count).Select(p => p.Position)];
    }

    private static List<int> Upgraded(RunState state) =>
        [
            .. Enumerable
                .Range(0, state.Deck.Count)
                .Where(i => state.Deck[i].Upgraded),
        ];

    [Theory]
    [InlineData("ABCDEF")]
    [InlineData("JJ39U0LLVY")]
    [InlineData("M4HT4KF030")]
    [InlineData("X2QEM46QX8")]
    [InlineData("0KFMJ8NU8E")]
    public void TheLightDoorUpgradesWhatTheGamesSortAndShufflePick(string seed)
    {
        var state = Deck(TwentyOneCards());
        state.Rng = new RunRngSet(seed);
        var oracle = new GameRng(
            unchecked(state.Rng.Seed + (uint)DeterministicHash.GetDeterministicHashCode("DOORS_OF_LIGHT_AND_DARK"))
        );
        var expected = Expected(state, oracle, 2);

        RunNonCombatEffects.UpgradeTwoRandomCardsForLightDoor(state);

        Assert.Equal(expected.Order(), Upgraded(state));
    }

    [Theory]
    [InlineData("ABCDEF")]
    [InlineData("JJ39U0LLVY")]
    [InlineData("M4HT4KF030")]
    public void SandCastleAndTheHammerUpgradeWhatTheGamesSortAndShufflePick(string seed)
    {
        var state = Deck(TwentyOneCards());
        var rng = new GameRng((uint)(seed.GetHashCode() & 0x7fffffff));
        var expected = Expected(state, rng.Copy(), 6);

        RunNonCombatEffects.UpgradeRandomCards(state, 6, rng);

        Assert.Equal(expected.Order(), Upgraded(state));
    }

    [Fact]
    public void TheTableIdOrderIsNotTheModelIdOrder()
    {
        // JACKPOT sorts before JACK_OF_ALL_TRADES by entry, after it by table id. A deck holding
        // both (two colourless cards) has to be sorted by entry.
        string[] names = [.. TwentyOneCards(), "Jackpot", "JackOfAllTrades"];
        for (int seed = 1; seed <= 40; seed++)
        {
            var state = Deck(names);
            var rng = new GameRng((uint)seed);
            var expected = Expected(state, rng.Copy(), 3);

            RunNonCombatEffects.UpgradeRandomDeckCardsForEvent(state, 3, rng);

            Assert.Equal(expected.Order(), Upgraded(state));
        }
    }
}
