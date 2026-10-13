using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `ListExtensions.StableShuffle` sorts by `CardModel.CompareTo` -- ModelId, then upgrade
/// level -- before it shuffles, so the same cards in a different pile order shuffle alike.
/// Catastrophe and Beat Down sorted by Entry alone, so a plain and an upgraded copy kept
/// their pile order and the same draws picked a different copy.
/// </summary>
public class StableShuffleSitesTests
{
    private static readonly CardInstance Strike = new(IC.StrikeIronclad, false);
    private static readonly CardInstance StrikePlus = new(IC.StrikeIronclad, true);
    private static readonly CardInstance Defend = new(IC.DefendIronclad, false);

    /// <summary>What the auto-plays did: the enemy's HP and the player's block.</summary>
    private static (int Hp, int Block) Outcome(Fight fight, int seed)
    {
        fight.State.ShuffleRng = new CountingRandom(seed);
        fight.State.TargetRng = new CountingRandom(7);
        fight.Energy(3);

        fight.Play(0);

        return (fight.State.Enemies[0].Hp, fight.State.PlayerBlock);
    }

    public static IEnumerable<object[]> Seeds() => Enumerable.Range(1, 8).Select(s => new object[] { s });

    [Theory]
    [MemberData(nameof(Seeds))]
    public void CatastrophePicksTheSameCopiesWhateverThePileOrder(int seed)
    {
        Fight Make(params CardInstance[] draw) =>
            Fight.Hand(new CardInstance(CL.Catastrophe, false)).Enemy(hp: 200).Draw(draw);

        Assert.Equal(
            Outcome(Make(StrikePlus, Strike, Defend), seed),
            Outcome(Make(Strike, StrikePlus, Defend), seed)
        );
    }

    [Theory]
    [MemberData(nameof(Seeds))]
    public void BeatDownPicksTheSameCopiesWhateverThePileOrder(int seed)
    {
        Fight Make(params CardInstance[] discard) =>
            Fight.Hand(new CardInstance(CL.BeatDown, false)).Enemy(hp: 200).Draw(Defend).Discard(discard);

        // Three of the four Attacks; which one stays behind is the pick.
        Assert.Equal(
            Outcome(Make(StrikePlus, Strike, Strike, StrikePlus), seed),
            Outcome(Make(Strike, StrikePlus, StrikePlus, Strike), seed)
        );
    }
}
