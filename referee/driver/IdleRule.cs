// When the driver's six-frame Idle counts the game as settled, as a rule over the flags it reads each
// frame. Combat.cs (Idle) and Settle.cs (the flag settle) read the game's state into TurnFlags; the rule
// itself has no game types, so it is tested on its own (Sts2Emulator.Tests links this file,
// RefereeIdleRuleTests).
//
// The game's turn flow, as CombatManager runs it (local decompile, by member name):
// - StartTurn, player side: sets every player's phase to Start, clears PlayerActionsDisabled and the
//   ready-to-end set, and sets _inPlayerTurnSetup; then SetupPlayerTurn (draw, AfterPlayerTurnStart) and
//   RunAutoPrePlayPhase (phase AutoPrePlay, then Play) run; only then is the action synchronizer put in
//   PlayPhase, and _inPlayerTurnSetup is cleared right after it (ReleaseDeferredEndTurnTransitionIfNeeded
//   also clears it on every exit path).
// - SetReadyToEndTurn adds the player to the ready-to-end set. If the running action is player-driven (the
//   end-turn action itself) the end-of-turn flow first awaits that action's completion
//   (WaitForActionThenEndTurn), so for a moment the queue is empty, the executor idle, the synchronizer
//   still in PlayPhase and the phase still Play: only the ready-to-end set says the turn is ending.
// - AfterAllPlayersReadyToEndTurn holds EndingPlayerTurnPhaseOne; AfterAllPlayersReadyToBeginEnemyTurn holds
//   EndingPlayerTurnPhaseTwo across the end of turn, the enemy turn and the awaited next StartTurn.

using System.Collections.Generic;

namespace Spireline.Referee;

/// <summary>The game's flags Idle reads on one frame. The combat-only ones are false outside a combat.</summary>
internal readonly record struct TurnFlags(
    bool ExecutorRunning,
    bool QueueEmpty,
    bool CombatInProgress,
    bool PlayerActionsDisabled,
    bool PlayerSide,
    bool InPlayPhase,
    bool InPlayerTurnSetup,
    bool ReadyToEndTurn,
    bool EndingPhaseOne,
    bool EndingPhaseTwo);

internal static class IdleRule
{
    /// <summary>
    /// The combat's own turn flow is between the player's phases: the turn start is still being set up, or the
    /// end of turn has begun (ready to end, or either end-of-turn phase).
    /// </summary>
    internal static bool TurnFlowBusy(in TurnFlags f) =>
        f.InPlayerTurnSetup || f.ReadyToEndTurn || f.EndingPhaseOne || f.EndingPhaseTwo;

    /// <summary>Something runs or is queued, or a combat is on and the player may not act yet.</summary>
    internal static bool Busy(in TurnFlags f) =>
        f.ExecutorRunning
        || !f.QueueEmpty
        || (f.CombatInProgress && (f.PlayerActionsDisabled || !f.PlayerSide || !f.InPlayPhase || TurnFlowBusy(f)));

    /// <summary>Quiet frames in a row: a busy frame or a changed state signature starts the count again.</summary>
    internal static int NextStable(int stable, bool busy, string sig, string last) =>
        !busy && sig == last ? stable + 1 : 0;

    /// <summary>
    /// Idle's loop over recorded frames (flags and the state signature of each): the frame index at which it
    /// counts <paramref name="idleFrames" /> quiet frames in a row, or -1 if it never does.
    /// </summary>
    internal static int SettlesAt(IEnumerable<(TurnFlags Flags, string Sig)> frames, int idleFrames)
    {
        string last = "";
        int stable = 0, i = 0;
        foreach (var (flags, sig) in frames)
        {
            bool busy = Busy(flags);
            string s = busy ? "busy" : sig;
            stable = NextStable(stable, busy, s, last);
            last = s;
            if (stable >= idleFrames)
            {
                return i;
            }

            i++;
        }

        return -1;
    }
}
