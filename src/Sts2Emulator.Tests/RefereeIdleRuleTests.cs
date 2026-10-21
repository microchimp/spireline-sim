using System.Collections.Generic;
using System.Linq;
using Spireline.Referee;
using Xunit;

namespace Sts2Emulator.Tests
{
    /// <summary>
    /// When the driver's six-frame Idle counts the game as settled (referee/driver/IdleRule.cs): nothing runs or
    /// is queued, and in a combat the player may act, which includes the combat's own turn flow being in neither
    /// the turn-start setup nor the end of turn. The frame sequences follow CombatManager's order of flags (see
    /// IdleRule.cs); no game code is involved.
    /// </summary>
    public class RefereeIdleRuleTests
    {
        private const int IdleFrames = 6;

        /// <summary>The player's turn, open: in the Play phase, actions enabled, nothing running.</summary>
        private static readonly TurnFlags Open = new(
            ExecutorRunning: false, QueueEmpty: true, CombatInProgress: true, PlayerActionsDisabled: false,
            PlayerSide: true, InPlayPhase: true, InPlayerTurnSetup: false, ReadyToEndTurn: false,
            EndingPhaseOne: false, EndingPhaseTwo: false);

        private static IEnumerable<(TurnFlags, string)> Frames(TurnFlags f, int n, string sig = "s") =>
            Enumerable.Repeat((f, sig), n);

        [Fact]
        public void OpenTurnIsNotBusy()
        {
            Assert.False(IdleRule.Busy(Open));
            Assert.False(IdleRule.TurnFlowBusy(Open));
        }

        [Fact]
        public void EachTurnFlowFlagIsBusy()
        {
            Assert.True(IdleRule.Busy(Open with { InPlayerTurnSetup = true }));
            Assert.True(IdleRule.Busy(Open with { ReadyToEndTurn = true }));
            Assert.True(IdleRule.Busy(Open with { EndingPhaseOne = true }));
            Assert.True(IdleRule.Busy(Open with { EndingPhaseTwo = true }));
        }

        [Fact]
        public void EachEarlierConditionIsStillBusy()
        {
            Assert.True(IdleRule.Busy(Open with { ExecutorRunning = true }));
            Assert.True(IdleRule.Busy(Open with { QueueEmpty = false }));
            Assert.True(IdleRule.Busy(Open with { PlayerActionsDisabled = true }));
            Assert.True(IdleRule.Busy(Open with { PlayerSide = false }));
            Assert.True(IdleRule.Busy(Open with { InPlayPhase = false }));
        }

        [Fact]
        public void OutsideACombatOnlyTheQueueCounts()
        {
            var none = new TurnFlags(false, true, false, false, false, false, false, false, false, false);
            Assert.False(IdleRule.Busy(none));
            // Leftover end-of-turn flags of a finished combat do not hold a reward screen or map step.
            Assert.False(IdleRule.Busy(none with { ReadyToEndTurn = true, EndingPhaseTwo = true }));
            Assert.True(IdleRule.Busy(none with { QueueEmpty = false }));
        }

        /// <summary>
        /// The end-of-turn gap: the end-turn action has finished (queue empty, executor idle) and the synchronizer
        /// is still in PlayPhase, but the player is ready to end the turn while the game waits for that action's
        /// completion before it marks EndingPlayerTurnPhaseOne. A record taken there is the "GAME RECORD
        /// UNSETTLED" kind; Idle must not settle until the next turn is open.
        /// </summary>
        [Fact]
        public void DoesNotSettleWhileTheEndOfTurnHasBegun()
        {
            var ready = Open with { ReadyToEndTurn = true };
            var frames = Frames(ready, 10, "before")
                .Concat(Frames(ready with { EndingPhaseOne = true }, 3, "before"))
                .Concat(Frames(Open with { EndingPhaseTwo = true, PlayerSide = false, InPlayPhase = false }, 5, "enemy"))
                .Concat(Frames(Open, IdleFrames + 1, "next"))
                .ToList();
            int at = IdleRule.SettlesAt(frames, IdleFrames);
            Assert.True(at >= 18, $"settled at frame {at}, inside the end of turn");
            Assert.Equal(18 + IdleFrames, at);
        }

        /// <summary>
        /// A turn start whose setup flag is still set holds Idle off, whatever the other flags read.
        /// </summary>
        [Fact]
        public void DoesNotSettleDuringTurnStartSetup()
        {
            var frames = Frames(Open with { InPlayerTurnSetup = true }, 20).Concat(Frames(Open, IdleFrames + 1));
            Assert.Equal(20 + IdleFrames, IdleRule.SettlesAt(frames, IdleFrames));
        }

        [Fact]
        public void SettlesAfterSixQuietFramesWithOneSignature()
        {
            Assert.Equal(IdleFrames, IdleRule.SettlesAt(Frames(Open, IdleFrames + 1), IdleFrames));
            // A changing signature (cards drawn over several frames outside the queue) restarts the count.
            var drawing = Enumerable.Range(0, 10).Select(i => (Open, "d" + i));
            Assert.Equal(-1, IdleRule.SettlesAt(drawing, IdleFrames));
        }
    }
}
