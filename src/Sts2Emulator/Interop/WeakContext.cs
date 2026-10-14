using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Interop;

/// <summary>
/// The encounter a native caller means when it names Corpse Slugs or Seapunk by the one id
/// each pair once shared, and the one place the C ABI reads that.
/// </summary>
/// <remarks>
/// <para>
/// The two fights of each pair used to be one emulator id, and the caller picked the
/// variant with a "weak context": <c>completedCombatRooms</c> in [0,3) was the weak one,
/// anything else (-1 by default, and every reset that takes no context) the normal one.
/// They are two encounters in the game (<c>CorpseSlugsWeak</c> and <c>CorpseSlugsNormal</c>,
/// <c>SeapunkWeak</c> and <c>SeapunkNormal</c>; <c>ActModel.GenerateRooms</c> deals the weak
/// ones first from their own list) and two ids here now, and the engine reads only the id.
/// </para>
/// <para>
/// Every C ABI entry point that takes an encounter id goes through <see cref="Resolve" />:
/// the combat resets with a context, the ones without (with <see cref="None" />) and the
/// run's <c>Sts2Run_StartCombat</c>. So the gym, the capture sweeps and the value-net
/// samplers keep getting what they asked for, and a caller that names a normal id gets it
/// whatever the context says. The rolled first combat (<c>Sts2_Reset</c>, an arena reset
/// with no encounter) names nothing: it deals from the act-1 weak pools, so it is the weak
/// encounter, as the game's would be.
/// </para>
/// <para>
/// Nothing inside the emulator reads a count of fights any more. The context was never the
/// game's rule, and the count it was computed from (fights AND elites) was wrong the moment
/// an elite or an event fight came first.
/// </para>
/// </remarks>
internal static class WeakContext
{
    /// <summary>
    /// The context of a reset that takes none: the normal encounter, which is what those
    /// resets have always dealt for the shared ids.
    /// </summary>
    public const int None = -1;

    /// <summary>The encounter id for <paramref name="encounterId" /> under the context.</summary>
    public static int Resolve(int encounterId, int completedCombatRooms)
    {
        if (completedCombatRooms is >= 0 and < 3)
        {
            return encounterId;
        }

        return encounterId switch
        {
            RunConstants.CorpseSlugsEncounterId => RunConstants.CorpseSlugsNormalEncounterId,
            RunConstants.SeapunkEncounterId => RunConstants.SeapunkNormalEncounterId,
            _ => encounterId,
        };
    }
}
