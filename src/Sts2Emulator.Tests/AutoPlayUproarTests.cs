using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `Uproar.OnPlay`: two hits, then `drawPile.Where(Attack && !Unplayable)
/// .StableShuffle(Rng.Shuffle).FirstOrDefault()` auto-played through `CardCmd.AutoPlay`.
/// The nested play answers its own kills and must leave Uproar's to Uproar's play.
/// </summary>
public class AutoPlayUproarTests
{
    private const int Uproar = 530;

    private static CardInstance Card(int id) => new(id, false);

    /// <summary>
    /// Uproar kills the first enemy with its hits, the Strike it plays kills a second; both
    /// deaths pay Gremlin Horn (energy and a card each). The nested play's corpse sweep used
    /// to remove Uproar's unanswered corpse, so its death hooks never fired.
    /// </summary>
    [Theory]
    [InlineData("hand")]
    [InlineData("havoc")]
    public void UproarsOwnKillIsAnsweredWhenItsAutoPlayedAttackAlsoKills(string how)
    {
        var fight = Fight
            .Hand(Card(how == "hand" ? Uproar : IC.Havoc))
            .Enemy(hp: 12)
            .Enemy(hp: 6)
            .Enemy(hp: 6)
            .Draw(
                [
                    .. how == "havoc" ? new[] { Card(Uproar) } : [],
                    Card(IC.DefendIronclad),
                    Card(IC.StrikeIronclad),
                    Card(IC.DefendIronclad),
                    Card(IC.DefendIronclad),
                ]
            )
            .Energy(3);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.GremlinHorn));
        // A Havoc-played Uproar rolls its own target: the seed aims it at the first enemy.
        fight.State.TargetRng = new CountingRandom(
            Enumerable.Range(1, 100).First(seed => new CountingRandom(seed).Next(3) == 0)
        );

        fight.Play(0);

        Assert.Equal(1, fight.State.Enemies.Count(enemy => enemy.Hp > 0));
        // 3, less the card played from hand (Uproar 2, Havoc 1), plus one per death.
        Assert.Equal(how == "hand" ? 3 : 4, fight.State.Energy);
        Assert.Equal([IC.DefendIronclad, IC.DefendIronclad], Fight.Ids(fight.State.Hand));
    }

    /// <summary>
    /// StableShuffle sorts before it shuffles, so the pick does not depend on pile order, and
    /// it draws n-1 times from the Shuffle stream.
    /// </summary>
    [Fact]
    public void UproarPicksItsAttackWithAStableShuffleOnTheShuffleStream()
    {
        int[] attacks = [IC.StrikeIronclad, IC.Bash, IC.Anger, IC.IronWave];
        var picked = new HashSet<int>();
        foreach (var order in new[] { attacks, attacks.Reverse().ToArray() })
        {
            var fight = Fight
                .Hand(Card(Uproar))
                .Enemy(hp: 200)
                .Draw([.. order.Select(Card), Card(IC.DefendIronclad)])
                .Energy(3);
            var shuffle = new CountingRandom(11);
            fight.State.ShuffleRng = shuffle;
            fight.State.TargetRng = new CountingRandom(7);

            fight.Play(0);

            Assert.Equal(attacks.Length - 1, shuffle.CallCount);
            picked.Add(attacks.Single(id => !fight.State.DrawPile.Any(card => card.DefId == id)));
        }

        Assert.Single(picked);
    }

    /// <summary>An upgraded copy sorts after the plain one (`CardModel.CompareTo`).</summary>
    [Fact]
    public void StableShuffleSortsByEntryThenUpgradeLevel()
    {
        var pile = new List<CardInstance>
        {
            new(IC.StrikeIronclad, true),
            new(IC.Bash, false),
            new(IC.StrikeIronclad, false),
        };

        var order = CardEffects.StableShuffle(pile, [0, 1, 2], new NoShuffle());

        Assert.Equal([1, 2, 0], order);
    }

    /// <summary>A stream whose Fisher-Yates leaves every element where it is.</summary>
    private sealed class NoShuffle : Random
    {
        public override int Next(int maxValue) => maxValue - 1;
    }
}
