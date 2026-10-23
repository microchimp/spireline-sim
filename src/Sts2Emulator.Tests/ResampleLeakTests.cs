using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// What a resampled clone hides (Spireline DESIGN.md 7.6, L2 and L8): the fights not yet entered,
/// and the moves behind the intents on screen. What the player has seen stays put.
/// </summary>
public class ResampleLeakTests
{
    private const string Seed = "84G7XUWQD6";

    private static RunEngine AtMap()
    {
        var e = new RunEngine();
        e.Reset(Seed, 10);
        for (int guard = 0; guard < 10 && e.State.Phase != RunPhase.Map; guard++)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Map, e.State.Phase);
        return e;
    }

    private static int FirstOption(RunEngine e, int nodeType) =>
        Enumerable.Range(0, RunConstants.MapChoices).First(i => e.State.MapNodeTypes[i] == nodeType);

    [Fact]
    public void ResampledClonesEnteringOneMonsterNodeMeetMoreThanOneEncounter()
    {
        var e = AtMap();
        int option = FirstOption(e, RunConstants.NodeNormal);
        var seen = new HashSet<int>();
        for (int seed = 0; seed < 200; seed++)
        {
            var clone = e.Clone(seed);
            Assert.Equal(clone.State.MapChoices[option], clone.State.NormalEncounterSequence[0]);
            clone.Step(option, -1, out _, out _, out _);
            seen.Add(clone.ActiveEncounterId);
        }

        Assert.True(seen.Count > 1, $"every resampled clone met encounter {string.Join(",", seen)}");
    }

    [Fact]
    public void AResampleKeepsTheFightsAlreadyEnteredAndTheBossesAndAFaithfulCloneKeepsEverything()
    {
        var e = AtMap();
        int option = FirstOption(e, RunConstants.NodeNormal);
        e.Step(option, -1, out _, out _, out _);
        int entered = e.State.NormalEncountersVisited;
        Assert.Equal(1, entered);

        var faithful = e.Clone();
        Assert.Equal(e.State.NormalEncounterSequence, faithful.State.NormalEncounterSequence);
        Assert.Equal(e.State.EliteEncounterSequence, faithful.State.EliteEncounterSequence);

        bool moved = false;
        for (int seed = 0; seed < 20; seed++)
        {
            var clone = e.Clone(seed);
            Assert.Equal(e.State.NormalEncounterSequence[..entered], clone.State.NormalEncounterSequence[..entered]);
            Assert.Equal(e.State.NormalEncounterSequence.Length, clone.State.NormalEncounterSequence.Length);
            for (int act = 0; act < e.State.Acts.Count; act++)
            {
                Assert.Equal(e.State.Acts[act].BossEncounterId, clone.State.Acts[act].BossEncounterId);
            }

            moved |= !clone.State.NormalEncounterSequence.SequenceEqual(e.State.NormalEncounterSequence);
        }

        Assert.True(moved);
    }

    [Fact]
    public void AResampledForkRollsTheCurrentMovesAgainBehindTheSameIntents()
    {
        var e = AtMap();
        int option = FirstOption(e, RunConstants.NodeNormal);
        Assert.Equal(
            0,
            e.EnterEncounterDebug(option, (int)CombatFactory.ActOneEncounter.SlimesNormal, RunConstants.NodeNormal)
        );
        var root = e.State.ActiveCombat!;
        Assert.NotNull(root.LastIntentRoll);

        for (int seed = 0; seed < 50; seed++)
        {
            var (fork, _) = e.ForkActiveCombat(null)!.Value;
            Assert.True(IntentParticles.Resample(fork, new Random(seed)));
            Assert.Equal(root.Enemies.Select(x => x.CurrentIntent.Type), fork.Enemies.Select(x => x.CurrentIntent.Type));
            Assert.Equal(
                root.Enemies.Select(x => (x.Hp, x.CurrentIntent.Type == IntentType.Attack ? x.CurrentIntent.Magnitude : 0)),
                fork.Enemies.Select(x => (x.Hp, x.CurrentIntent.Type == IntentType.Attack ? x.CurrentIntent.Magnitude : 0))
            );
            // A fork of a fork rolls again from the same pass.
            Assert.True(IntentParticles.Resample(fork.Clone(), new Random(seed + 1000)));
        }
    }

    [Fact]
    public void AFightThatMovedAMonstersMoveSinceThePassKeepsItsMoves()
    {
        var e = AtMap();
        int option = FirstOption(e, RunConstants.NodeNormal);
        e.Step(option, -1, out _, out _, out _);
        var (fork, _) = e.ForkActiveCombat(null)!.Value;
        fork.Enemies[0].LastMove += 7;
        var before = fork.Enemies.Select(x => x.Clone()).ToList();

        Assert.False(IntentParticles.Resample(fork, new Random(1)));
        Assert.Equal(before.Select(x => x.LastMove), fork.Enemies.Select(x => x.LastMove));
    }
}
