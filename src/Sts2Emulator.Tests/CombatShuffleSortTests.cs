using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `ListExtensions.StableShuffle` is `List&lt;CardModel&gt;.Sort()` and then the Fisher-Yates, and
/// `CardPileCmd.Shuffle` is that over discard-then-draw, with `Hook.AfterShuffle` after it.
/// </summary>
/// <remarks>
/// The sort is the runtime's introsort, which does NOT keep equal cards in their order:
/// [B, B, A] sorts to [A, second B, first B]. The emulator sorted stably (LINQ's `OrderBy`
/// in `CardEffects.StableShuffle`, a position-keyed `Array.Sort` in the reshuffle), so two
/// equal copies came out of every shuffle swapped -- Stone Cracker upgraded the other Strike,
/// the auto-plays took the other copy. And the reshuffle fired `AfterShuffle` BEFORE it
/// shuffled: Biiig Hug's Soot was shuffled in with the rest (probe P09: the game shuffled
/// ten and drew the Soot; the emulator shuffled eleven and left it in the pile).
/// </remarks>
public class CombatShuffleSortTests
{
    private static int Id(string name) => GeneratedData.Cards.FindId(name)!.Value;

    /// <summary>A copy told apart by a field the sort does not read.</summary>
    private static CardInstance Copy(string name, int tag, bool upgraded = false) =>
        new(Id(name), upgraded, ReplayCount: tag);

    /// <summary>A card as the game's <c>List&lt;CardModel&gt;.Sort()</c> sees it.</summary>
    private sealed class Probe(CardInstance card) : IComparable<Probe>
    {
        public CardInstance Card { get; } = card;

        public int CompareTo(Probe? other)
        {
            int byEntry = string.CompareOrdinal(
                GeneratedData.Cards.Get(Card.DefId).Entry,
                GeneratedData.Cards.Get(other!.Card.DefId).Entry
            );
            return byEntry != 0 ? byEntry : Card.Upgraded.CompareTo(other.Card.Upgraded);
        }
    }

    /// <summary>The game's `StableShuffle`: sort the list as it is, then Fisher-Yates.</summary>
    private static List<CardInstance> GameStableShuffle(IEnumerable<CardInstance> list, Random rng)
    {
        var probes = list.Select(card => new Probe(card)).ToList();
        probes.Sort();
        CardEffects.ShufflePile(probes, rng);
        return [.. probes.Select(probe => probe.Card)];
    }

    /// <summary>Fisher-Yates with every swap a no-op, so what comes out is the sort's order.</summary>
    private sealed class NoSwapRandom : Random
    {
        public override int Next(int maxValue) => maxValue - 1;
    }

    private static List<int> Tags(IEnumerable<CardInstance> cards) => [.. cards.Select(card => card.ReplayCount)];

    /// <summary>The singles-review case, as the game ran it: [B, B, A] sorts to positions [2, 1, 0].</summary>
    [Fact]
    public void TheGamesSortPutsTheSecondOfTwoEqualCardsFirst()
    {
        CardInstance[] pile = [Copy("StrikeIronclad", 0), Copy("StrikeIronclad", 1), Copy("Bash", 2)];

        Assert.Equal([2, 1, 0], Tags(GameStableShuffle(pile, new NoSwapRandom())));
        Assert.Equal([2, 1, 0], CardEffects.StableShuffle([.. pile], [0, 1, 2], new NoSwapRandom()));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void TheReshuffleSortsEqualCardsAsTheGamesSortDoes(int seed)
    {
        CardInstance[] discard = [Copy("StrikeIronclad", 0), Copy("StrikeIronclad", 1), Copy("Bash", 2)];
        var fight = Fight.Hand().Enemy(hp: 100).Draw().Discard(discard);
        fight.State.ShuffleRng = new CountingRandom(seed);

        var expected = GameStableShuffle(discard, new CountingRandom(seed));
        CardEffects.ShuffleDiscardIntoDraw(fight.State, new Random(0));

        Assert.Equal(Tags(expected), Tags(fight.State.DrawPile));
    }

    /// <summary>
    /// Past sixteen cards the introsort partitions instead of insertion-sorting, and ties move
    /// further: a full reshuffle of discard plus draw, against the game's sort and shuffle.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(42)]
    [InlineData(1234)]
    public void ALargeReshuffleMatchesTheGamesSortAndShuffle(int seed)
    {
        string[] discardNames =
        [
            .. Enumerable.Repeat("StrikeIronclad", 5),
            .. Enumerable.Repeat("DefendIronclad", 4),
            "Bash",
            "Anger",
            "ShrugItOff",
            "Anger",
            "ShrugItOff",
            "IronWave",
        ];
        string[] drawNames = ["StrikeIronclad", "IronWave", "BodySlam", "BodySlam", "DefendIronclad"];
        var discard = discardNames.Select((name, i) => Copy(name, i, upgraded: i % 7 == 6)).ToArray();
        var draw = drawNames.Select((name, i) => Copy(name, discard.Length + i)).ToArray();
        var fight = Fight.Hand().Enemy(hp: 100).Draw(draw).Discard(discard);
        fight.State.ShuffleRng = new CountingRandom(seed);

        var expected = GameStableShuffle(discard.Concat(draw), new CountingRandom(seed));
        CardEffects.ShuffleDiscardIntoDraw(fight.State, new Random(0));

        Assert.Equal(Tags(expected), Tags(fight.State.DrawPile));
        Assert.Empty(fight.State.DiscardPile);
    }

    /// <summary>
    /// `StoneCracker`: `Draw.Where(IsUpgradable).StableShuffle(CombatCardSelection).Take(2)`.
    /// Which of the equal copies it upgrades is the sort's doing.
    /// </summary>
    [Theory]
    [InlineData(3)]
    [InlineData(11)]
    [InlineData(29)]
    [InlineData(77)]
    public void StoneCrackerUpgradesTheCopiesTheGamesSortAndShufflePick(int seed)
    {
        // Twenty upgradable cards: past sixteen the introsort partitions, and equal copies
        // move. (From four to sixteen it insertion-sorts, which keeps them in order.)
        string[] names =
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
        ];
        var pile = names.Select((name, i) => Copy(name, i)).ToArray();
        var state = new CombatState
        {
            NicheHpRng = new CountingRandom(0),
            CardSelectionRng = new CountingRandom(seed),
        };
        CombatFactory.Reset(
            state,
            new Random(0),
            pile,
            1,
            [RelicEffects.StoneCracker],
            80,
            80,
            [],
            0,
            deckPreShuffled: true
        );

        var expected = GameStableShuffle(pile, new CountingRandom(seed)).Take(2).Select(card => card.ReplayCount);
        var upgraded = state.DrawPile.Concat(state.Hand).Where(card => card.Upgraded).Select(card => card.ReplayCount);

        Assert.Equal(expected.Order(), upgraded.Order());
    }

    /// <summary>
    /// Biiig Hug's Soot goes in AFTER the shuffle: ten cards shuffled (nine draws), then the
    /// Soot's place, `Shuffle.NextInt(10 + 1)` off the same stream.
    /// </summary>
    [Theory]
    [InlineData(5)]
    [InlineData(19)]
    [InlineData(300)]
    public void BiiigHugsSootGoesIntoThePileTheShuffleLeft(int seed)
    {
        var discard = Enumerable
            .Range(0, 10)
            .Select(i => Copy(i % 3 == 0 ? "Bash" : i % 2 == 0 ? "DefendIronclad" : "StrikeIronclad", i))
            .ToArray();
        var fight = Fight.Hand().Enemy(hp: 100).Draw().Discard(discard);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.BiiigHug));
        var shuffle = new CountingRandom(seed);
        fight.State.ShuffleRng = shuffle;

        var oracle = new CountingRandom(seed);
        var expected = GameStableShuffle(discard, oracle);
        expected.Insert(oracle.Next(expected.Count + 1), new CardInstance(Id("Soot"), false));
        CardEffects.ShuffleDiscardIntoDraw(fight.State, new Random(0));

        Assert.Equal(10, shuffle.CallCount);
        Assert.Equal(
            expected.Select(card => (card.DefId, card.ReplayCount)),
            fight.State.DrawPile.Select(card => (card.DefId, card.ReplayCount))
        );
    }

    /// <summary>
    /// `Hook.AfterShuffle` visits the player's powers before the player's relics: Stratagem
    /// takes its card out of the shuffled pile, and only then does Biiig Hug place its Soot.
    /// </summary>
    [Fact]
    public void StratagemTakesItsCardBeforeBiiigHugPlacesTheSoot()
    {
        var discard = Enumerable.Range(0, 6).Select(i => Copy(i % 2 == 0 ? "Bash" : "StrikeIronclad", i)).ToArray();
        var fight = Fight.Hand().Enemy(hp: 100).Draw().Discard(discard);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.BiiigHug));
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.StratagemPower, 1);
        fight.State.ShuffleRng = new CountingRandom(8);

        var oracle = new CountingRandom(8);
        var expected = GameStableShuffle(discard, oracle);
        var taken = expected[0];
        expected.RemoveAt(0);
        int sootAt = oracle.Next(expected.Count + 1);
        CardEffects.ShuffleDiscardIntoDraw(fight.State, new Random(0));

        Assert.Equal(taken.ReplayCount, Assert.Single(fight.State.Hand).ReplayCount);
        Assert.Equal(Id("Soot"), fight.State.DrawPile[sootAt].DefId);
        Assert.Equal(6, fight.State.DrawPile.Count);
    }
}
