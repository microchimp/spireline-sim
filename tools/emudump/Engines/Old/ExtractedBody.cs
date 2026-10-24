// An inline model body that was moved out of its dispatch site into a named function (docs/rebuild/MIGRATION_F08.md, round R5:
// CardEffects.PersonalHive_AfterDamageReceived out of LandDamageOnEnemy, CombatEngine.Ravenous_AfterDeath out of AnswerDeaths, ...).
// The oracle's origin labels (rng-log, trace: "CardEffects.AddCardToDrawPileRandomly < CardEffects.LandDamageOnEnemy") name the
// first two engine frames above a draw. The extraction must not move a label, so the frame of such a function is left out and the
// label names the method the body came from, as it did while the body sat inline.
namespace EmuDump.Engines.Old;

static class ExtractedBody
{
    /// <summary>
    /// Whether <paramref name="method" /> (a method of <paramref name="type" />, a lambda or local function already read as its
    /// enclosing method) is an extracted body: named <c>Model_Hook</c>, in a class of the old engine (<c>Sts2Emulator.Core</c>,
    /// not Core2). No old-engine method had such a name before round R5.
    /// </summary>
    internal static bool Is(Type type, string method)
    {
        int u = method.IndexOf('_');
        if (u <= 0 || u + 1 >= method.Length || !char.IsUpper(method[0]) || !char.IsUpper(method[u + 1]))
        {
            return false;
        }

        string ns = type.Namespace ?? "";
        return ns.StartsWith("Sts2Emulator.Core", StringComparison.Ordinal)
            && !ns.StartsWith("Sts2Emulator.Core2", StringComparison.Ordinal);
    }
}
