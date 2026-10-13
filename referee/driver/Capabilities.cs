// What this driver build is and can do, reported once at job start so a recording can be refused when it
// was not made by a clean build of a committed driver with the fixes a reference needs (referee/refset.py
// record). The commit and the dirty flag are stamped into the DLL at build time (SpirelineReferee.csproj,
// target SpirelineStamp); the capabilities are read off the compiled code itself by reflection, so a merge
// that drops a fix drops its capability too, and no list can claim code the DLL does not have.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json.Nodes;

namespace Spireline.Referee;

public static partial class Driver
{
    private const BindingFlags AnyStatic = BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public;

    private static bool Has(string member) => typeof(Driver).GetMember(member, AnyStatic).Length > 0;

    private static bool HasNested(string type) => typeof(Driver).GetNestedType(type, BindingFlags.NonPublic | BindingFlags.Public) != null;

    /// <summary>
    /// The driver's capabilities, each named after the fix or feature it stands for and present only when the
    /// code that implements it is compiled in:
    ///   settle_flags        the flag settle (Settle.cs: the game's turn flags, SettleMode "flags")
    ///   play_phase_gate     actions only in the player's Play phase (Combat.cs InPlayPhase, fix/referee-stalls)
    ///   watchdog_v2         the frame-counting watchdog and Budget waits (Watchdog.cs)
    ///   console_setup       setup items through the console's commands (Setup.cs)
    ///   refset_setup        the reference set's deck, relic, HP and potion items (RefsetSetup.cs)
    ///   player_load         a real run's holdings put in place through the game's save-load path (RefsetSetup.cs)
    ///   act_entry           the fight's own act entered first, as the console's act command (RefsetSetup.cs)
    ///   deck_residual_fails a deck the setup cannot build fails its item (RefsetSetup.cs)
    ///   relic_flashes       which relics fired, per step (Chooser.cs)
    ///   heuristic_chooser   the "heuristic" chooser for realistic fights (Chooser.cs)
    /// </summary>
    internal static List<string> Capabilities()
    {
        var caps = new List<string>();
        void Add(string name, bool present)
        {
            if (present) caps.Add(name);
        }

        Add("settle_flags", Has("CheckSettleFields") && Has("SettleMode"));
        Add("play_phase_gate", Has("InPlayPhase"));
        Add("watchdog_v2", Has("BudgetFrames") && HasNested("Budget") && Has("ArmWatchdog"));
        Add("console_setup", Has("SetupItem") && Has("EnterFight"));
        Add("refset_setup", Has("RefsetItem"));
        Add("player_load", Has("PlayerLoad"));
        Add("act_entry", Has("EnterActItem"));
        Add("deck_residual_fails", Has("DeckResidualFails"));
        Add("relic_flashes", Has("RecordRelicFlashes"));
        Add("heuristic_chooser", Has("ChooseHeuristic"));
        return caps;
    }

    private static string Stamp(string key) => typeof(Driver).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
        .FirstOrDefault(a => a.Key == key)?.Value ?? "";

    /// <summary>The build and the job's choices, one line in the game's log: "[referee] driver {...}".</summary>
    internal static JsonObject DriverInfo() => new()
    {
        ["commit"] = Stamp("spireline.commit"),
        // Whether the driver's sources (*.cs, *.csproj) differed from that commit when the DLL was built.
        ["dirty"] = Stamp("spireline.dirty").Trim() != "",
        ["capabilities"] = new JsonArray(Capabilities().Select(c => (JsonNode)c).ToArray()),
        ["settle"] = SettleMode,
    };
}
