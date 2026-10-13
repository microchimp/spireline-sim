using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Interop;

/// <summary>
/// The encounter a native caller means when it names Corpse Slugs or Seapunk and says
/// which fight it is with a "weak context".
/// </summary>
/// <remarks>
/// <para>
/// The two fights used to be one emulator id each, and the caller picked the variant:
/// <c>completedCombatRooms</c> in [0,3) was the weak one, -1 the normal one. They are two
/// encounters in the game and two ids here now, and the run engine names the one it
/// means. This is where the old convention is read, once, at the edge: the gym and the
/// capture sweeps still pass the shared id and the context across the C ABI, and keep
/// getting what they asked for. A caller that names the normal id gets it whatever the
/// context says.
/// </para>
/// <para>
/// Nothing inside the emulator reads a count of fights any more. The context was never the
/// game's rule -- the game takes the weak variant from the encounter's slot in the act's
/// list -- and the count it was computed from (fights AND elites) was wrong the moment an
/// elite or an event fight came first.
/// </para>
/// </remarks>
internal static class WeakContext
{
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
