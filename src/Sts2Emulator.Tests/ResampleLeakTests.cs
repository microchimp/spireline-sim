using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
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
            // The current act's boss is on the map and stays; the later acts' are not shown yet.
            Assert.Equal(e.State.BossEncounterId, clone.State.BossEncounterId);
            Assert.Equal(e.State.Ancient, clone.State.Ancient);

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

    /// <summary>
    /// A fork's kernel stands in the fork (KernelState.CopyTo binds a combat's copy to no run): a forked step whose Strike
    /// kills an Axebot with Stock respawns it off the fork's own Niche stream, and the live run's combat stream and the
    /// live fight's Niche stream stay where they were.
    /// </summary>
    [Fact]
    public void AForkedStepWithAStockDeathLeavesTheLiveRunsStreamsUntouched()
    {
        var e = AtMap();
        e.Kernel = new Sts2Emulator.Core2.Kernel(new KernelOptions([Switch.F08, Switch.F10]),
            () => Sts2Emulator.Tests.Core2.Models.ModelTestKit.Ids, Sts2Emulator.Core2.Models.Registry.L3b);
        int option = FirstOption(e, RunConstants.NodeNormal);
        Assert.Equal(
            0,
            e.EnterEncounterDebug(option, (int)CombatFactory.ActOneEncounter.SlimesNormal, RunConstants.NodeNormal)
        );
        var live = e.State.ActiveCombat!;
        Assert.Same(e.State, live.Kernel!.World!.Run);
        int liveCombat = e.State.ActiveCombatRng!.CallCount;
        int liveNiche = live.NicheHpRng!.CallCount;

        var (fork, forkRng) = e.ForkActiveCombat(null)!.Value;
        Assert.Null(fork.Kernel!.World!.Run);
        var bot = fork.Enemies[0];
        bot.DefId = KE.Axebot;
        bot.Hp = 1;
        bot.Buffs.Add(new BuffState(BuffId.Stock, 2));
        fork.Hand.Insert(0, new CardInstance(473, false));
        fork.Energy = 9;
        CombatEngine.Step(fork, 0, forkRng, 0);

        Assert.DoesNotContain(bot, fork.Enemies);
        Assert.Equal(KE.Axebot, fork.Enemies[^1].DefId);
        Assert.Equal(liveNiche + 1, fork.NicheHpRng!.CallCount);
        Assert.Equal(liveCombat, e.State.ActiveCombatRng.CallCount);
        Assert.Equal(liveNiche, live.NicheHpRng.CallCount);
        Assert.NotEqual(KE.Axebot, live.Enemies[0].DefId);
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

    private static IEnumerable<RunEngine> Clones(RunEngine e, int n = 200) =>
        Enumerable.Range(0, n).Select(seed => e.Clone(seed));

    [Fact]
    public void LaterActsBossesAreRedealtAndTheCurrentOneAndAFaithfulCloneKeepEverything()
    {
        var e = AtMap();
        var faithful = e.Clone();
        for (int act = 0; act < e.State.Acts.Count; act++)
        {
            Assert.Equal(e.State.Acts[act].BossEncounterId, faithful.State.Acts[act].BossEncounterId);
            Assert.Equal(e.State.Acts[act].SecondBossEncounterId, faithful.State.Acts[act].SecondBossEncounterId);
        }

        for (int act = 1; act < e.State.Acts.Count; act++)
        {
            int a = act;
            var bosses = Clones(e).Select(c => c.State.Acts[a].BossEncounterId).ToHashSet();
            Assert.True(bosses.Count > 1, $"act {a}: every clone dealt boss {string.Join(",", bosses)}");
        }

        Assert.All(Clones(e), c => Assert.Equal(e.State.Acts[0].BossEncounterId, c.State.Acts[0].BossEncounterId));
    }

    [Fact]
    public void LaterActsAncientsAreRedealtAndTheCurrentOneIsKept()
    {
        var e = AtMap();
        var faithful = e.Clone();
        Assert.Equal(e.State.Acts.Select(a => a.Ancient), faithful.State.Acts.Select(a => a.Ancient));
        for (int act = 1; act < e.State.Acts.Count; act++)
        {
            int a = act;
            var ancients = Clones(e).Select(c => c.State.Acts[a].Ancient).ToHashSet();
            Assert.True(ancients.Count > 1, $"act {a}: every clone dealt ancient {string.Join(",", ancients)}");
        }

        Assert.All(Clones(e), c => Assert.Equal(e.State.Ancient, c.State.Ancient));
    }

    [Fact]
    public void EventSequencesAreRedealtFromTheCursorOnAndTheEventsPassedStayPut()
    {
        var e = AtMap();
        int cursor = e.State.EventSequenceIndex;
        var faithful = e.Clone();
        Assert.Equal(e.State.EventSequence, faithful.State.EventSequence);
        Assert.Equal(cursor, faithful.State.EventSequenceIndex);

        var firsts = new HashSet<int>();
        var laterActs = new HashSet<int>();
        foreach (var c in Clones(e))
        {
            // Slot 0 is the Ancient room's and was never shown: it is dealt again. What the run met stays (nothing yet here; see ResampleLeakFollowupTests).
            Assert.Equal(e.State.EventSequence[1..cursor], c.State.EventSequence[1..cursor]);
            Assert.Equal(cursor, c.State.EventSequenceIndex);
            // Same events, reordered: the pool is the generator's, not a new one.
            Assert.Equal(e.State.EventSequence.Order(), c.State.EventSequence.Order());
            firsts.Add(c.State.EventSequence[cursor]);
            laterActs.Add(c.State.Acts[1].Events[1]);
        }

        Assert.True(firsts.Count > 1, "the current act's next event never varied");
        Assert.True(laterActs.Count > 1, "the next act's events never varied");
    }

    [Fact]
    public void ShopStockIsRolledOnEntryOffTheReseededStreamSoClonesDiffer()
    {
        var e = AtMap();
        string Stock(RunEngine r)
        {
            RunRewardGenerator.EnterShop(r.State);
            return string.Join(",", r.State.ShopCards) + "|" + string.Join(",", r.State.ShopRelics);
        }

        Assert.Equal(Stock(e.Clone()), Stock(e.Clone()));
        Assert.True(Clones(e, 50).Select(Stock).ToHashSet().Count > 1, "every clone stocked the same shop");
    }

    private static RunEngine InFight()
    {
        var e = AtMap();
        e.EnterEncounterDebug(
            FirstOption(e, RunConstants.NodeNormal),
            (int)CombatFactory.ActOneEncounter.SlimesNormal,
            RunConstants.NodeNormal
        );
        return e;
    }

    [Fact]
    public void AResampleThatKeepsTheTrueMovesSetsTheFlagAndForksOfItStayFlagged()
    {
        var root = InFight().State.ActiveCombat!;
        Assert.False(root.AiStateAssumed);

        var ok = root.Clone();
        Assert.True(IntentParticles.Resample(ok, new Random(1)));
        Assert.False(ok.AiStateAssumed);
        Assert.False(ok.Clone().AiStateAssumed);

        // A loaded or imported fight has no recorded pass.
        var loaded = root.Clone();
        loaded.LastIntentRoll = null;
        Assert.False(IntentParticles.Resample(loaded, new Random(1)));
        Assert.True(loaded.AiStateAssumed);
        Assert.True(loaded.Clone().AiStateAssumed);
        Assert.True(loaded.Clone(7).AiStateAssumed);

        // AI fields moved since the pass.
        var moved = root.Clone();
        moved.Enemies[0].LastMove += 7;
        Assert.False(IntentParticles.Resample(moved, new Random(1)));
        Assert.True(moved.AiStateAssumed);
    }

    [Fact]
    public void ARunForkOfAFlaggedFightIsFlaggedAndOfAnUnflaggedFightIsNot()
    {
        var e = InFight();
        Assert.False(e.Clone(3).State.ActiveCombat!.AiStateAssumed);
        Assert.False(e.ForkActiveCombat(3)!.Value.Combat.AiStateAssumed);

        e.State.ActiveCombat!.LastIntentRoll = null;
        var flagged = e.Clone(3);
        Assert.True(flagged.State.ActiveCombat!.AiStateAssumed);
        Assert.True(flagged.Clone(4).State.ActiveCombat!.AiStateAssumed);
        Assert.True(flagged.Clone().State.ActiveCombat!.AiStateAssumed);
        Assert.True(e.ForkActiveCombat(3)!.Value.Combat.AiStateAssumed);
        // The faithful source is never flagged by being forked.
        Assert.False(e.State.ActiveCombat!.AiStateAssumed);
    }
}
