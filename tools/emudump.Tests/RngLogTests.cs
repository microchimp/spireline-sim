// EmuDump fight --rng-log (RngLog.cs over IEngine.StartRngLog and TakeRngLog, the old engine's OldRngLog.cs) over a step stopped on a screen raised in the middle of it: the
// answer replays the step from its start with the streams moved back, and draws the stopped run's draws
// again. The log holds each draw once, as trace mode holds the step once (OldTrace.cs); it held the stopped
// run's and then the replay's, so rng_diff.py counted the replayed draws twice (CombatCardGeneration
// 77 -> 154) and reported 387 fights where 101 differ.
using System.Text.Json.Nodes;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

public class RngLogTests
{
    [Fact]
    public void AStoppedStepsDrawsAreLoggedOnceAsTheReplayDrewThem()
    {
        var e = new RunEngine();
        e.Reset("RNGLOG001");
        e.StartCombat(RunConstants.StarterDeckIds, RunConstants.SlitheringStranglerEncounterId,
            [RunConstants.StartingRelicFor(PlayerCharacter.Ironclad)], playerHp: 64, playerMaxHp: 80, potionIds: [], playerGold: 99);
        var c = e.State.ActiveCombat!;
        // Tyranny's pick is asked at the next turn start: the end turn (the enemies' moves, the next
        // intents, the hand draw and its shuffle) stops there, and the answer replays all of it.
        BuffSystem.Apply(c.PlayerBuffs, BuffId.Tyranny, 1);

        var (rngCall, mark) = (CombatTrace.RngCall, CombatTrace.Mark);
        try
        {
            var engine = new OldEngine(e);
            engine.StartRngLog();
            e.Step(c.Hand.Count, -1, out _, out _, out _);
            Assert.NotNull(e.State.ActiveCombat!.PendingSelection?.Replay);
            var stopped = RngLog.Take(engine);
            Assert.NotEmpty(stopped);
            e.Step(0, -1, out _, out _, out _);
            Assert.Null(e.State.ActiveCombat!.PendingSelection);
            var answered = RngLog.Take(engine);

            // The stopped run's draws went with the replay: the step that stopped holds none of them,
            // and every stream's draws count up one at a time, none twice.
            Assert.Empty(stopped);
            Assert.NotEmpty(answered);
            foreach (var stream in answered.Select(d => d!.AsArray()).GroupBy(d => (string)d[0]!))
            {
                var indices = stream.Select(d => (int)d[1]!).ToList();
                Assert.Equal(Enumerable.Range(indices[0], indices.Count), indices);
            }
        }
        finally
        {
            (CombatTrace.RngCall, CombatTrace.Mark) = (rngCall, mark);
        }
    }
}
