namespace Sts2Emulator.Core;

/// <summary>
/// Observation points for diagnosing a divergence from the game (tools/emudump's trace mode, which
/// pairs them with the referee's trace of the same fight, referee/driver/Trace.cs). Both sinks are
/// null in every normal run, so a probe costs one static read and changes nothing.
/// </summary>
/// <remarks>
/// The marks sit where the game's own trace marks its combat: <c>play</c> where
/// <c>CardModel.OnPlayWrapper</c> logs CardPlayStarted (after BeforeCardPlayed), <c>played</c> where
/// it logs CardPlayFinished (after the card's and its enchantment's effects, before AfterCardPlayed),
/// <c>turn_end</c> at the player's BeforeTurnEnd, <c>enemy_turn</c> and <c>turn_start</c> at each
/// side's BeforeSideTurnStart, and <c>move</c>/<c>moved</c> around <c>MonsterModel.PerformMove</c>.
/// </remarks>
public static class CombatTrace
{
    /// <summary>(what, subject def id or -1, state) at each mark.</summary>
    public static Action<string, int, CombatState>? Mark;

    /// <summary>Every draw from a <see cref="CountingRandom" />, after its counter moved.</summary>
    public static Action<CountingRandom>? RngCall;

    internal static void At(string what, int subject, CombatState state)
    {
        if (Mark is { } mark && _quiet == 0)
        {
            mark(what, subject, state);
        }
    }

    internal static void OnRng(CountingRandom rng)
    {
        if (RngCall is { } call && _quiet == 0)
        {
            call(rng);
        }
    }

    /// <summary>
    /// A scope on this thread in which neither sink hears anything: a check that replays a step on a
    /// copy (<c>CombatEngine.CheckStoppedStep</c>) is not part of the fight being traced.
    /// </summary>
    internal static QuietScope Quiet()
    {
        _quiet++;
        return default;
    }

    [ThreadStatic]
    private static int _quiet;

    internal readonly struct QuietScope : IDisposable
    {
        public void Dispose() => _quiet--;
    }
}
