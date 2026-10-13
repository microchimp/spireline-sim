using Sts2Emulator.Core;
using Sts2Emulator.Tests.Invariants;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// DeepState is what every metamorphic relation compares with, so it has to see what a snapshot does
/// not and must not see what is not state.
/// </summary>
public class DeepStateTests
{
    private static Combat Opened(CombatFactory.ActOneEncounter encounter = CombatFactory.ActOneEncounter.Chompers, int seed = 3)
    {
        return TestFights.Open(encounter, seed);
    }

    [Fact]
    public void TwoOpeningsOfOneFightAreOneState()
    {
        Assert.Equal(DeepState.Digest(Opened().Whole), DeepState.Digest(Opened().Whole));
    }

    [Fact]
    public void ADifferentSeedIsADifferentState()
    {
        Assert.NotEqual(DeepState.Digest(Opened(seed: 3).Whole), DeepState.Digest(Opened(seed: 4).Whole));
    }

    [Fact]
    public void ACloneIsTheSameStateAndNamesNoDifference()
    {
        var c = Opened();
        var clone = c.Clone();

        Assert.Equal(DeepState.Digest(c.Whole), DeepState.Digest(clone.Whole));
        Assert.Empty(DeepState.Diff(c.Whole, clone.Whole));
    }

    [Fact]
    public void AFieldNoSnapshotKeepsStillShowsAsADifference()
    {
        var c = Opened();
        var other = c.Clone();
        other.State.Enemies[0].LastMoveRepeats += 1; // enemy AI scratch: not in the snapshot's visible fields
        other.State.CardsPlayedThisCombat += 1;

        var diff = DeepState.Diff(c.Whole, other.Whole);

        Assert.Contains(diff, d => d.StartsWith(".Item1.Enemies[0].LastMoveRepeats"));
        Assert.Contains(diff, d => d.StartsWith(".Item1.CardsPlayedThisCombat"));
    }

    [Fact]
    public void APerCardModifierShowsAsADifference()
    {
        var c = Opened();
        var other = c.Clone();
        other.State.Hand[0] = other.State.Hand[0] with { CostBump = 1 };

        Assert.Contains(DeepState.Diff(c.Whole, other.Whole), d => d.StartsWith(".Item1.Hand[0].CostBump"));
    }

    [Fact]
    public void AGeneratorsPositionShowsAsADifferenceEvenWithTheSameCallCount()
    {
        // Two streams that were advanced the same number of times but are different streams.
        var a = new CountingRandom(1);
        var b = new CountingRandom(2);
        a.Next();
        b.Next();

        Assert.NotEqual(DeepState.Digest(a), DeepState.Digest(b));
        Assert.NotEmpty(DeepState.Diff(a, b));
    }

    [Fact]
    public void ACopyOfAGeneratorIsTheSameStateUntilEitherMoves()
    {
        var a = new CountingRandom(5);
        a.Next();
        var b = a.Copy();

        Assert.Equal(DeepState.Digest(a), DeepState.Digest(b));
        b.Next();
        Assert.NotEqual(DeepState.Digest(a), DeepState.Digest(b));
    }

    [Fact]
    public void ASharedObjectIsNotTheSameAsTwoEqualOnes()
    {
        var shared = new List<int> { 1, 2 };
        var twoRefs = (shared, shared);
        var twoCopies = (new List<int> { 1, 2 }, new List<int> { 1, 2 });

        Assert.NotEqual(DeepState.Digest(twoRefs), DeepState.Digest(twoCopies));
    }

    [Fact]
    public void SkippedPathsAreLeftOutOfTheDigestAndTheDiff()
    {
        var c = Opened();
        var other = c.Clone();
        other.State.PlayerBlock += 5;

        Assert.NotEqual(DeepState.Digest(c.Whole), DeepState.Digest(other.Whole));
        Assert.Equal(DeepState.Digest(c.Whole, ".Item1.PlayerBlock"), DeepState.Digest(other.Whole, ".Item1.PlayerBlock"));
        Assert.Empty(DeepState.Diff(c.Whole, other.Whole, 6, ".Item1.PlayerBlock"));
    }

    [Fact]
    public void ListOrderIsPartOfTheState()
    {
        var c = Opened();
        var other = c.Clone();
        other.State.Hand.Reverse();

        Assert.NotEqual(DeepState.Digest(c.Whole), DeepState.Digest(other.Whole));
    }

    [Fact]
    public void ReadingTheStateChangesNothing()
    {
        var c = Opened();
        string before = DeepState.Digest(c.Whole);
        for (int i = 0; i < 3; i++)
        {
            DeepState.Digest(c.Whole);
            DeepState.Lines(c.Whole);
        }

        Assert.Equal(before, DeepState.Digest(c.Whole));
    }

    [Fact]
    public void ACloneKeepsNothingSharedWithItsOriginal()
    {
        var c = Opened();
        var clone = c.Clone();
        string frozen = DeepState.Digest(clone.Whole);

        RandomPlay.Play(c, new Random(1), 30);

        Assert.Equal(frozen, DeepState.Digest(clone.Whole));
    }
}
