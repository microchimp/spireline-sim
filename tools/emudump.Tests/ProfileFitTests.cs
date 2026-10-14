// The codex walker generates a recorded run for the profile its history fits (ProfileFit): records
// written here from the emulator's own run under a profile the record does not name, so the fit
// must find a profile under which every mark agrees, and leave the mature one when nothing
// disagrees or too little is recorded to tell.
using System.Text.Json;
using System.Text.Json.Nodes;
using EmuDump.Codex;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;
using Epochs = Sts2Emulator.Core.UnlockState.Epochs;

public class ProfileFitTests
{
    const int Asc = 10;

    static string Model(int encounter) => "ENCOUNTER." + Names.GameEntry(Names.EncounterName(encounter));

    static JsonObject Room(string type, string model) => new()
    {
        ["map_point_type"] = type,
        ["rooms"] = new JsonArray(new JsonObject { ["model_id"] = model, ["room_type"] = type }),
    };

    /// <summary>
    /// The marks of the run <paramref name="seed" /> generates for <paramref name="profile" />:
    /// its acts, Neow's options, each act's ancient, its first <paramref name="fights" /> normal
    /// fights, an elite and its boss(es).
    /// </summary>
    static JsonElement History(string seed, UnlockState profile, int fights = 4, int acts = 3)
    {
        var e = new RunEngine();
        e.Reset(seed, Asc, PlayerCharacter.Ironclad, profile);
        var s = e.State;
        var history = new JsonArray();
        for (int act = 0; act < acts; act++)
        {
            var rooms = s.Acts[act];
            var points = new JsonArray();
            var ancient = Room("ancient", "EVENT." + rooms.Ancient);
            if (act == 0)
            {
                ancient["player_stats"] = new JsonArray(new JsonObject
                {
                    ["ancient_choice"] = new JsonArray(s.NeowOptions.Where(x => x != 0).Select(x => (JsonNode)new JsonObject
                    {
                        ["TextKey"] = Names.Relic(x),
                        ["title"] = new JsonObject { ["key"] = Names.Relic(x) + ".title", ["table"] = "relics" },
                    }).ToArray()),
                });
            }

            points.Add(ancient);
            foreach (int fight in rooms.NormalEncounters.Take(fights))
            {
                points.Add(Room("monster", Model(fight)));
            }

            points.Add(Room("elite", Model(rooms.EliteEncounters[0])));
            points.Add(Room("boss", Model(rooms.BossEncounterId)));
            if (rooms.SecondBossEncounterId != 0)
            {
                points.Add(Room("boss", Model(rooms.SecondBossEncounterId)));
            }

            history.Add(points);
        }

        var record = new JsonObject
        {
            ["seed"] = seed,
            ["acts"] = new JsonArray(s.Acts.Select(a => (JsonNode)("ACT." + Names.Act(a.Act))).ToArray()),
            ["map_point_history"] = history,
        };
        return JsonDocument.Parse(record.ToJsonString()).RootElement;
    }

    static UnlockState Fit(string seed, JsonElement history) =>
        ProfileFit.Fit(history, new RunEngine(), seed, Asc, PlayerCharacter.Ironclad);

    static IEnumerable<string> Seeds(int count) => Enumerable.Range(0, count).Select(i => $"PROFILEFIT{i:D3}");

    /// <summary>The first seed whose run under <paramref name="profile" /> differs from the mature one where <paramref name="differs" /> looks.</summary>
    static string SeedWhere(UnlockState profile, Func<RunState, RunState, bool> differs)
    {
        foreach (string seed in Seeds(200))
        {
            var mature = new RunEngine();
            mature.Reset(seed, Asc);
            var young = new RunEngine();
            young.Reset(seed, Asc, PlayerCharacter.Ironclad, profile);
            if (differs(mature.State, young.State))
            {
                return seed;
            }
        }

        throw new InvalidOperationException("no seed shows the profile");
    }

    [Fact]
    public void AMatureRunFitsTheMatureProfile()
    {
        foreach (string seed in Seeds(10))
        {
            Assert.True(Fit(seed, History(seed, UnlockState.All)).IsAll);
        }
    }

    [Fact]
    public void NeowWithoutKaleidoscopeFitsAProfileShortOfTheDefect()
    {
        var young = UnlockState.All.WithLockedEpochs(Epochs.Defect1);
        string seed = SeedWhere(young, (a, b) => !a.NeowOptions.SequenceEqual(b.NeowOptions));
        var fitted = Fit(seed, History(seed, young));
        Assert.Equal([Epochs.Defect1], fitted.LockedEpochs);
    }

    [Fact]
    public void AnUnexpectedBossFitsItsDiscoveryOrder()
    {
        // Overgrowth meets Vantom, the Ceremonial Beast, then the Kin.
        int theKin = RunConstants.BossDiscoveryOrder(RunConstants.ActOvergrowth)[2];
        var kin = UnlockState.All.WithUnseenEncounters(theKin);
        string seed = SeedWhere(kin, (a, b) => a.Acts[0].BossEncounterId != b.Acts[0].BossEncounterId);
        var fitted = Fit(seed, History(seed, kin));
        Assert.Equal([theKin], fitted.UnseenEncounters);
        Assert.Empty(fitted.LockedEpochs);
    }

    [Fact]
    public void FightsDealtFromAShorterEventShuffleFitTheLockedEventEpoch()
    {
        var noPhilosophers = UnlockState.All.WithLockedEpochs(Epochs.Event3);
        string seed = Seeds(1).Single();
        Assert.Equal([Epochs.Event3], Fit(seed, History(seed, noPhilosophers)).LockedEpochs);

        // Two fights are not enough to tell a locked event from a coincidence.
        var two = History(seed, noPhilosophers, fights: 1, acts: 1);
        Assert.Empty(Fit(seed, two).LockedEpochs);
    }

    [Fact]
    public void ALaterAncientFitsOrobasOrDarvLocked()
    {
        var noOrobas = UnlockState.All.WithLockedEpochs(Epochs.Orobas);
        string seed = SeedWhere(noOrobas, (a, b) => a.Acts[1].Ancient != b.Acts[1].Ancient);
        Assert.Equal([Epochs.Orobas], Fit(seed, History(seed, noOrobas)).LockedEpochs);

        var noDarv = UnlockState.All.WithLockedEpochs(Epochs.Darv);
        seed = SeedWhere(noDarv, (a, b) => a.Acts.Any(x => x.Ancient == RunConstants.AncientDarv));
        Assert.Equal([Epochs.Darv], Fit(seed, History(seed, noDarv)).LockedEpochs);
    }

    [Fact]
    public void TheActsFitUnderdocksLockedOrUndiscovered()
    {
        var locked = UnlockState.All.WithLockedEpochs(Epochs.Underdocks);
        string seed = SeedWhere(locked, (a, b) => a.Acts[0].Act != b.Acts[0].Act);
        Assert.Equal([Epochs.Underdocks], Fit(seed, History(seed, locked)).LockedEpochs);

        var undiscovered = UnlockState.All.WithUndiscoveredActs(RunConstants.ActUnderdocks);
        seed = SeedWhere(undiscovered, (a, b) => a.Acts[0].Act != b.Acts[0].Act);
        Assert.Equal([RunConstants.ActUnderdocks], Fit(seed, History(seed, undiscovered)).UndiscoveredActs);
    }

    /// <summary>
    /// The whole walk: a run whose Neow could not offer Kaleidoscope is walked to its end under the
    /// fitted profile and reports it, where the mature profile stopped at Neow.
    /// </summary>
    [Fact]
    public void TheWalkerWalksAYoungProfilesRunAndSaysSo()
    {
        var young = UnlockState.All.WithLockedEpochs(Epochs.Defect1);
        foreach (string seed in Seeds(60))
        {
            var mature = new RunEngine();
            mature.Reset(seed, Asc);
            var run = new RunEngine();
            run.Reset(seed, Asc, PlayerCharacter.Ironclad, young);
            JsonObject record;
            try
            {
                if (mature.State.NeowOptions.SequenceEqual(run.State.NeowOptions))
                {
                    continue;
                }

                record = WalkerTests.Record(seed: seed, profile: young);
            }
            catch (Xunit.Sdk.XunitException)
            {
                continue; // no plain relic blessing or monster node on this seed
            }

            var result = WalkerTests.Walk(record);
            Assert.Equal("complete", (string?)result["status"]);
            Assert.Equal(young.ToString(), (string?)result["profile"]);
            return;
        }

        Assert.Fail("no seed shows a young profile's Neow");
    }
}
