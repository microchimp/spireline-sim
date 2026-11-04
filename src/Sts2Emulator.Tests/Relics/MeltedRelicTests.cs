using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A melted relic (Toy Box's wax relics, taken back one by one). The game keeps it in the player's relic list with
/// <c>IsMelted</c> set and every hook, counter and offer skips it; the copy used to drop it from the list, so the two
/// lists differed and a re-anchor from the game's list brought it back alive. The copy keeps it now, in
/// <see cref="RunState.MeltedRelics" />, out of the list every effect reads.
/// </summary>
public class MeltedRelicTests
{
    private const string Seed = "NXV45HW43K";

    private static RunEngine Run()
    {
        var engine = new RunEngine();
        engine.Reset(Seed);
        return engine;
    }

    private static CombatState FirstFight(RunEngine engine)
    {
        engine.State.Phase = RunPhase.Map;
        for (int i = 0; i < engine.State.MapNodeTypes.Length; i++)
        {
            if (engine.State.MapNodeTypes[i] == RunConstants.NodeNormal)
            {
                engine.Step(i, -1, out _, out _, out _);
                break;
            }
        }

        return engine.State.ActiveCombat!;
    }

    /// <summary>A run holding Toy Box one fight short of its melt, and the wax relic it will melt (a Sai, 7 block every turn).</summary>
    private static RunEngine WithAWaxSai(int saiCounter = 0)
    {
        var engine = Run();
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicToyBox, Counter: RunConstants.ToyBoxCombats - 1));
        engine.State.Relics.Add(new RelicInstance(RelicEffects.Sai, saiCounter));
        engine.State.WaxRelics.Add(RelicEffects.Sai);
        return engine;
    }

    [Fact]
    public void AMeltedWaxRelicLeavesTheActiveListAndStaysInTheRunWithItsCounter()
    {
        var engine = WithAWaxSai(saiCounter: 4);

        RunNonCombatEffects.MeltAToyBoxRelic(engine.State);

        Assert.DoesNotContain(engine.State.Relics, relic => relic.DefId == RelicEffects.Sai);
        Assert.Equal(new RelicInstance(RelicEffects.Sai, 4), Assert.Single(engine.State.MeltedRelics));
        Assert.Empty(engine.State.WaxRelics);
        Assert.Equal(RunConstants.ToyBoxCombats, engine.State.Relics.Single(relic => relic.DefId == RunConstants.RelicToyBox).Counter);
    }

    [Fact]
    public void AMeltedRelicHasNoEffectInTheNextFight()
    {
        var held = FirstFight(WithAWaxSai());
        Assert.Equal(7, held.PlayerBlock); // the control: the same relic, not melted

        var engine = WithAWaxSai();
        RunNonCombatEffects.MeltAToyBoxRelic(engine.State);
        var melted = FirstFight(engine);

        Assert.Equal(0, melted.PlayerBlock);
        Assert.DoesNotContain(melted.Relics, relic => relic.DefId == RelicEffects.Sai);
    }

    [Fact]
    public void ALoadFromTheGamesListKeepsTheMeltedFlagAndTheRelicStaysOutOfPlay()
    {
        var engine = Run();
        int starter = engine.State.Relics[0].DefId;
        engine.DebugSetPlayer(-1, -1, -1, null, [starter, -RelicEffects.Sai], null);

        Assert.Equal([starter], engine.State.Relics.Select(relic => relic.DefId));
        Assert.Equal([RelicEffects.Sai], engine.State.MeltedRelics.Select(relic => relic.DefId));
        Assert.Equal(0, FirstFight(engine).PlayerBlock);
    }

    [Fact]
    public void ALoadFollowsTheGamesFlagBothWaysAndKeepsTheCounter()
    {
        var engine = WithAWaxSai(saiCounter: 4);
        int starter = engine.State.Relics[0].DefId;

        engine.DebugSetPlayer(-1, -1, -1, null, [starter, -RelicEffects.Sai], null);
        Assert.Equal(new RelicInstance(RelicEffects.Sai, 4), Assert.Single(engine.State.MeltedRelics));
        Assert.DoesNotContain(engine.State.Relics, relic => relic.DefId == RelicEffects.Sai);

        // The game lists the relic as held again (a copy that melted too early is put right): alive, counter kept.
        engine.DebugSetPlayer(-1, -1, -1, null, [starter, RelicEffects.Sai], null);
        Assert.Empty(engine.State.MeltedRelics);
        Assert.Equal(new RelicInstance(RelicEffects.Sai, 4), engine.State.Relics.Single(relic => relic.DefId == RelicEffects.Sai));
    }

    [Fact]
    public void AMeltedWaxRelicIsNotMeltedTwiceAfterALoad()
    {
        // Toy Box melts "the first wax relic still held": a relic the game's list shows melted is not held, so the next
        // melt takes the next one, as the game's FirstOrDefault(IsWax && !IsMelted) does.
        var engine = WithAWaxSai();
        int other = RelicEffects.Anchor;
        engine.State.Relics.Add(new RelicInstance(other));
        engine.State.WaxRelics.Add(other);
        int starter = engine.State.Relics[0].DefId;
        engine.DebugSetPlayer(-1, -1, -1, null,
            [starter, RunConstants.RelicToyBox, -RelicEffects.Sai, other], null,
            relicCounters: [RunConstants.RelicToyBox, RunConstants.ToyBoxCombats - 1]);

        RunNonCombatEffects.MeltAToyBoxRelic(engine.State);

        Assert.Equal([RelicEffects.Sai, other], engine.State.MeltedRelics.Select(relic => relic.DefId));
        Assert.DoesNotContain(engine.State.Relics, relic => relic.DefId == other);
    }

    [Fact]
    public void AClonedRunKeepsItsMeltedRelics()
    {
        var engine = WithAWaxSai(saiCounter: 4);
        RunNonCombatEffects.MeltAToyBoxRelic(engine.State);

        var clone = engine.State.Clone();

        Assert.Equal(engine.State.MeltedRelics, clone.MeltedRelics);
        Assert.NotSame(engine.State.MeltedRelics, clone.MeltedRelics);
    }

    [Fact]
    public unsafe void TheNativeLoadAndStateListCarryTheFlag()
    {
        int handle = RunNativeExports.Sts2Run_Create();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(handle, out var run));
            run.Reset(Seed);
            int starter = run.State.Relics[0].DefId;
            int[] relics = [starter, -RelicEffects.Sai];
            var obs = new int[RunConstants.RunObsSize];
            fixed (int* r = relics, o = obs)
            {
                Assert.Equal(0, RunNativeExports.Sts2Run_DebugSetPlayer(handle, -1, -1, -1, null, -1, r, relics.Length, null, -1, -1, o));
            }

            Assert.Equal(new[] { starter }, StateList(handle, 1));
            Assert.Equal(new[] { RelicEffects.Sai }, StateList(handle, 24));
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    private static unsafe int[] StateList(int handle, int listId)
    {
        var buffer = new int[16];
        int count;
        fixed (int* b = buffer)
        {
            count = RunNativeExports.Sts2Run_GetStateList(handle, listId, b, buffer.Length);
        }

        Assert.True(count >= 0);
        return buffer.Take(count).ToArray();
    }
}
