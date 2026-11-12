using System.Linq;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Lists 26 and 27 of <c>Sts2Run_GetStateList</c>: what a player tracks about the "?" rooms (how many were visited this act and the
/// odds of a fight, a treasure room and a shop, in thousandths) and the merchant's removals bought and the removals owed. Read by
/// the run-level learner's state (the three state gaps), so they are pinned here against the engine's own fields.
/// </summary>
public class RunStateGapsListTests
{
    private const string Seed = "NXV45HW43K";

    [Fact]
    public unsafe void ARunStartsWithTheBaseOddsAndNoRemovals()
    {
        int handle = RunNativeExports.Sts2Run_Create();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(handle, out var run));
            run.Reset(Seed);
            Assert.Equal(new[] { 0, 100, 20, 30 }, StateList(handle, 26));
            Assert.Equal(new[] { 0, 0 }, StateList(handle, 27));
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    [Fact]
    public unsafe void TheListsReadTheEnginesFieldsRoundedToThousandths()
    {
        int handle = RunNativeExports.Sts2Run_Create();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(handle, out var run));
            run.Reset(Seed);
            run.State.UnknownMapPointsVisited = 3;
            run.State.UnknownMapPointMonsterOdds = 0.1 + 0.1 + 0.1; // 0.30000000000000004
            run.State.UnknownMapPointTreasureOdds = 0.02 + 0.02;
            run.State.UnknownMapPointShopOdds = 0.09;
            run.State.ShopRemovalsUsed = 2;
            run.State.CardRemovalsOwed = 1;
            Assert.Equal(new[] { 3, 300, 40, 90 }, StateList(handle, 26));
            Assert.Equal(new[] { 2, 1 }, StateList(handle, 27));
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    [Fact]
    public unsafe void AQuestionMarkRoomMovesTheOddsAndANewActResetsThem()
    {
        int handle = RunNativeExports.Sts2Run_Create();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(handle, out var run));
            run.Reset(Seed);
            run.State.Phase = RunPhase.Map;
            int option = -1;
            for (int i = 0; i < run.State.MapNodeTypes.Length; i++)
            {
                if (run.State.MapNodeTypes[i] != RunConstants.NodeNone)
                {
                    option = i;
                    break;
                }
            }

            Assert.True(option >= 0);
            run.State.MapNodeTypes[option] = RunConstants.NodeEvent; // a "?"
            Assert.Equal(0, run.Step(option, -1, out _, out _, out _));
            var list = StateList(handle, 26);
            Assert.Equal(1, list[0]);
            Assert.Equal((int)System.Math.Round(run.State.UnknownMapPointMonsterOdds * 1000.0), list[1]);
            Assert.Equal((int)System.Math.Round(run.State.UnknownMapPointTreasureOdds * 1000.0), list[2]);
            Assert.Equal((int)System.Math.Round(run.State.UnknownMapPointShopOdds * 1000.0), list[3]);
            // one "?" rolled something: that room's odds went back to the base, every other still allowed one grew by its base
            Assert.Contains(list[1], new[] { 100, 200 });
            Assert.Contains(list[2], new[] { 20, 40 });
            Assert.Contains(list[3], new[] { 30, 60 });
            Assert.NotEqual(new[] { 1, 100, 20, 30 }, list);

            Assert.True(run.EnterNextAct());
            Assert.Equal(new[] { 0, 100, 20, 30 }, StateList(handle, 26));
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
