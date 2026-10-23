// The referee's setup fights (EmuDump fight, Setup.EnterFight and OldSetup.EnterEncounter): the game records the encounter it
// entered by its place in the act's lists (act, kind, index), and EmuDump deals the id at that place
// in the emulator's copy of the list. A weak encounter and its normal twin are two places, so they
// are two ids and two rosters with no flag beside them: the entry used to pass "kind == weak" to
// the run engine, because Corpse Slugs' two fights shared one id, and the normal place dealt that
// shared id (ActModel.GenerateRooms deals the weak and normal encounters from two lists).
using System.Text.Json.Nodes;
using EmuDump.Codex;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

public class SetupFightTests
{
    /// <summary>(act, kind, index in the game's list, the emulator entry for that model, monsters).</summary>
    public static TheoryData<string, string, int, string, int> Fights =>
        new()
        {
            { "UNDERDOCKS", "weak", 0, "CorpseSlugs", 2 },
            { "UNDERDOCKS", "normal", 0, "CorpseSlugsNormal", 3 },
            { "UNDERDOCKS", "weak", 1, "Seapunk", 1 },
            { "UNDERDOCKS", "normal", 7, "SeapunkNormal", 2 },
        };

    [Theory]
    [MemberData(nameof(Fights))]
    public void TheListPlaceAloneNamesTheEncounter(string act, string kind, int index, string entry, int monsters)
    {
        var e = AtMap();
        // Rooms already fought, elites among them, more than any act's weak share: not consulted.
        e.State.NormalEncountersVisited = 5;
        e.State.EliteEncountersVisited = 2;
        int option = Enumerable.Range(0, RunConstants.MapChoices).First(i => e.State.MapNodeTypes[i] == RunConstants.NodeNormal);
        var game = new JsonObject
        {
            ["setup"] = new JsonArray(new JsonObject { ["cmd"] = "fight" }),
            ["fight"] = new JsonObject { ["act"] = act, ["kind"] = kind, ["index"] = index, ["id"] = entry },
        };
        var log = new JsonObject();

        Assert.True(Setup.EnterFight(new OldEngine(e), game, option, log));

        var fight = e.State.ActiveCombat!;
        Assert.Equal(entry, Names.EncounterName(fight.EncounterId));
        Assert.Equal(entry, (string?)log["encounter"]);
        Assert.Equal(monsters, fight.Enemies.Count);
    }

    static RunEngine AtMap()
    {
        var e = new RunEngine();
        e.Reset("84G7XUWQD6", 10);
        for (int guard = 0; guard < 10 && e.State.Phase != RunPhase.Map; guard++)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Map, e.State.Phase);
        return e;
    }
}
