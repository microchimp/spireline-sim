// The old engine's loader of a game snapshot into its combat, as the oracle tools reach it. The loader itself is one
// class in the engine DLL, Sts2Emulator.Core.FightAnchor (src/Sts2Emulator/Core/FightAnchor.cs), because the native export
// Sts2Run_ReanchorFight puts a run's fight on a record with the same code; this file only adapts its lists to the tools'
// JsonArrays and keeps the names Engines/Old/ and the tests call. The id tables are read by OldReanchor from
// data/id_map.json and tools/fidelity/power_aliases.json and handed in as <see cref="Tables"/>.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;

namespace EmuDump.Engines.Old;

static class OldAnchor
{
    /// <summary>The game's id tables, as the loader reads them (FightAnchor.Tables).</summary>
    internal sealed class Tables(JsonObject idMap, JsonObject aliases) : FightAnchor.Tables(idMap, aliases);

    internal static readonly HashSet<BuffId> NotPowers = FightAnchor.NotPowers;

    internal static JsonObject PowerKeys(FightAnchor.Tables t) => FightAnchor.PowerKeys(t);

    internal static BuffId? BuffFor(string gamePower) => FightAnchor.BuffFor(gamePower);

    /// <summary>The loader's patched list, copied into the tools' array as it stands (also when the loader throws).</summary>
    static T Patching<T>(JsonArray patched, Func<List<string>, T> call)
    {
        var list = new List<string>();
        try
        {
            return call(list);
        }
        finally
        {
            foreach (var what in list) patched.Add(what);
        }
    }

    internal static string? Anchor(FightAnchor.Tables t, CombatState q, JsonObject g, JsonArray patched) =>
        Patching(patched, list => FightAnchor.Anchor(t, q, g, list));

    internal static bool PatchBuffs(FightAnchor.Tables t, List<BuffState> cur, JsonArray powers, out List<BuffState> result, JsonArray patched, string where)
    {
        List<BuffState> r = cur;
        bool changed = Patching(patched, list => FightAnchor.PatchBuffs(t, cur, powers, out r, list, where));
        result = r;
        return changed;
    }

    internal static (string Status, JsonArray? Unknown) AnchorMoves(CombatState q, JsonObject g, JsonArray patched, Func<int, List<string>> later)
    {
        var (status, unknown) = Patching(patched, list => FightAnchor.AnchorMoves(q, g, list, later));
        return (status, unknown == null ? null : new JsonArray(unknown.Select(i => (JsonNode)i).ToArray()));
    }

    internal static string? GameMove(JsonNode enemy) => FightAnchor.GameMove(enemy);

    internal static string EmuMove(EnemyState en) => FightAnchor.EmuMove(en);

    internal static List<string> LaterMoves(JsonArray gsteps, int j, int i) => FightAnchor.LaterMoves(gsteps, j, i);

    internal static bool Reselect(CombatState q, EnemyState en, int moveIndex, out EnemyState c) => FightAnchor.Reselect(q, en, moveIndex, out c);
}
