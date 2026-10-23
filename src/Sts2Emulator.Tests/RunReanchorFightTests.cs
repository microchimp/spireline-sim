using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// <c>Sts2Run_ReanchorFight</c> puts the fight a run stands in on the referee's record of it, as an agent
/// sees that record (no streams, the draw pile as a sorted list). Called through the unmanaged entry points
/// ctypes uses. The record is built here from a second run that played a card, so the first run's fight
/// differs from it in the hand, the discard, the energy and the enemy, and must end up equal to the second's.
/// </summary>
public class RunReanchorFightTests
{
    private static readonly Lazy<(JsonObject IdMap, JsonObject Aliases)> Tables = new(() =>
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            string map = Path.Combine(d.FullName, "data", "id_map.json");
            string aliases = Path.Combine(d.FullName, "tools", "fidelity", "power_aliases.json");
            if (File.Exists(map) && File.Exists(aliases))
            {
                return (
                    JsonNode.Parse(File.ReadAllText(map))!.AsObject(),
                    JsonNode.Parse(File.ReadAllText(aliases))!.AsObject()
                );
            }
        }

        throw new DirectoryNotFoundException("data/id_map.json not found above " + AppContext.BaseDirectory);
    });

    private static unsafe void SetTables()
    {
        var (map, aliases) = Tables.Value;
        var doc = new JsonObject { ["id_map"] = JsonNode.Parse(map.ToJsonString()), ["aliases"] = JsonNode.Parse(aliases.ToJsonString()) };
        byte[] text = Encoding.UTF8.GetBytes(doc.ToJsonString());
        delegate* unmanaged<byte*, int, int> set = &NativeExports.Sts2Run_SetAnchorTables;
        fixed (byte* t = text)
        {
            Assert.Equal(0, set(t, text.Length));
        }
    }

    private static string Snake(string name) =>
        Regex.Replace(name, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

    /// <summary>The fight as the referee's observation of it (Env.Obs): the draw pile sorted, no streams.</summary>
    private static string Record(CombatState c)
    {
        var tables = new FightAnchor.Tables(Tables.Value.IdMap, Tables.Value.Aliases);
        var cards = tables.Cards.GroupBy(kv => kv.Value).ToDictionary(g => g.Key, g => g.First().Key);
        var enemies = tables.Enemies.GroupBy(kv => kv.Value).ToDictionary(g => g.Key, g => g.First().Key);
        JsonArray Pile(IEnumerable<CardInstance> p) => new(p.Select(x => (JsonNode)new JsonArray(cards[x.DefId], x.Upgraded)).ToArray());
        JsonArray Powers(List<BuffState> bs) => new(bs.Select(b => (JsonNode)new JsonArray(Snake(b.Id.ToString()) + "_POWER", b.Magnitude)).ToArray());
        var draw = new JsonArray(Pile(c.DrawPile).Select(x => x!.ToJsonString()).Order(StringComparer.Ordinal).Select(x => JsonNode.Parse(x)).ToArray());
        return new JsonObject
        {
            ["in_combat"] = true,
            ["round"] = c.Turn + 1,
            ["player"] = new JsonObject
            {
                ["hp"] = c.PlayerHp, ["max_hp"] = c.PlayerMaxHp, ["block"] = c.PlayerBlock, ["energy"] = c.Energy, ["powers"] = Powers(c.PlayerBuffs),
            },
            ["hand"] = Pile(c.Hand), ["draw"] = draw, ["discard"] = Pile(c.DiscardPile), ["exhaust"] = Pile(c.ExhaustPile),
            ["enemies"] = new JsonArray(c.Enemies.Where(x => x.Hp > 0).Select(x => (JsonNode)new JsonObject
            {
                ["name"] = enemies[x.DefId], ["hp"] = x.Hp, ["max_hp"] = x.MaxHp, ["block"] = x.Block, ["powers"] = Powers(x.Buffs),
            }).ToArray()),
        }.ToJsonString();
    }

    private static unsafe (int Status, string Message) Reanchor(int run, string record)
    {
        delegate* unmanaged<int, byte*, int, byte*, int, int*, int> reanchor = &NativeExports.Sts2Run_ReanchorFight;
        byte[] text = Encoding.UTF8.GetBytes(record);
        var message = new byte[512];
        var obs = new int[RunConstants.RunObsSize];
        fixed (byte* t = text)
        fixed (byte* m = message)
        fixed (int* o = obs)
        {
            int status = reanchor(run, t, text.Length, m, message.Length, o);
            return (status, Encoding.UTF8.GetString(message, 0, Array.IndexOf(message, (byte)0)));
        }
    }

    private static CombatState Fight(int run)
    {
        Assert.True(RunNativeExports.TryGetRun(run, out var engine));
        return engine!.State.ActiveCombat!;
    }

    private static string Piles(CombatState c) =>
        string.Join("|", new[] { c.Hand, c.DiscardPile, c.ExhaustPile }.Select(p => string.Join(",", p.Select(x => $"{x.DefId}{(x.Upgraded ? "+" : "")}"))))
        + $"|{string.Join(",", c.DrawPile.Select(x => x.DefId).Order())}|hp{c.PlayerHp}b{c.PlayerBlock}e{c.Energy}|"
        + string.Join(",", c.Enemies.Select(e => $"{e.DefId}:{e.Hp}:{e.Block}"));

    private static unsafe int Play(int run, int action)
    {
        delegate* unmanaged<int, int, int, int*, float*, int*, int*, int> step = &NativeExports.Sts2Run_Step;
        var obs = new int[RunConstants.RunObsSize];
        float reward = 0;
        int terminal = 0, truncated = 0;
        fixed (int* o = obs)
        {
            return step(run, action, 0, o, &reward, &terminal, &truncated);
        }
    }

    [Fact]
    public void AFightIsPutOnTheRecordOfAFightThatWentOn()
    {
        SetTables();
        int a = RunApi.InCombat();
        int b = RunApi.InCombat();
        try
        {
            Assert.Equal(Piles(Fight(a)), Piles(Fight(b)));
            Play(b, 0);  // the second run plays its first card: another hand, discard, energy
            Assert.NotEqual(Piles(Fight(a)), Piles(Fight(b)));
            int shuffleCalls = Fight(a).ShuffleRng!.CallCount;
            var drawBefore = Fight(a).DrawPile.Select(x => x.DefId).ToList();

            var (status, message) = Reanchor(a, Record(Fight(b)));

            Assert.True(status > 0, $"status {status}: {message}");
            Assert.Equal(Piles(Fight(b)), Piles(Fight(a)));
            // What the record cannot say stays: the streams, and the draw order of cards the record gives only as a list.
            Assert.Equal(shuffleCalls, Fight(a).ShuffleRng!.CallCount);
            Assert.Equal(drawBefore, Fight(a).DrawPile.Select(x => x.DefId).ToList());
            Assert.Contains("patched=", message);
        }
        finally
        {
            RunApi.Destroy(a);
            RunApi.Destroy(b);
        }
    }

    [Fact]
    public void ACardTheRecordHoldsAndTheFightDidNotGetsAFreshUidAndTheOthersKeepTheirs()
    {
        SetTables();
        int run = RunApi.InCombat();
        try
        {
            var combat = Fight(run);
            var before = CardUids.LiveCards(combat).ToDictionary(c => c.UidOf(), c => c.DefId);
            var held = CardUids.LiveCards(combat).Select(c => c.DefId).ToHashSet();
            var tables = new FightAnchor.Tables(Tables.Value.IdMap, Tables.Value.Aliases);
            var (newName, newId) = tables.Cards.OrderBy(kv => kv.Value).First(kv => !held.Contains(kv.Value));
            var record = JsonNode.Parse(Record(combat))!.AsObject();
            record["hand"]!.AsArray().Add(new JsonArray(newName, false));

            var (status, message) = Reanchor(run, record.ToJsonString());

            Assert.True(status > 0, $"status {status}: {message}");
            combat = Fight(run);
            var added = combat.Hand.Single(c => c.DefId == newId);
            Assert.NotEqual(0, added.UidOf());
            Assert.DoesNotContain(added.UidOf(), before.Keys);
            Assert.True(CardUidInvariant.Holds(combat));
            foreach (var c in CardUids.LiveCards(combat).Where(c => c.DefId != newId))
            {
                Assert.True(before.TryGetValue(c.UidOf(), out var def) && def == c.DefId, "a card the fight held keeps its uid");
            }
        }
        finally
        {
            RunApi.Destroy(run);
        }
    }

    [Fact]
    public void ARefusedRecordLeavesTheRunAsItWas()
    {
        SetTables();
        int run = RunApi.InCombat();
        try
        {
            var before = Piles(Fight(run));
            var record = JsonNode.Parse(Record(Fight(run)))!.AsObject();
            record["hand"]!.AsArray().Add(new JsonArray("NO_SUCH_CARD", false));
            var (status, message) = Reanchor(run, record.ToJsonString());
            Assert.Equal(-4, status);
            Assert.StartsWith("no emulator id for", message);
            Assert.Equal(before, Piles(Fight(run)));

            // A roster the fight cannot be paired with: an enemy the run's fight does not have.
            record = JsonNode.Parse(Record(Fight(run)))!.AsObject();
            var extra = JsonNode.Parse(record["enemies"]![0]!.ToJsonString())!.AsObject();
            extra["name"] = new FightAnchor.Tables(Tables.Value.IdMap, Tables.Value.Aliases).Enemies.First(kv => kv.Value != Fight(run).Enemies[0].DefId).Key;
            record["enemies"]!.AsArray().Add(extra);
            (status, message) = Reanchor(run, record.ToJsonString());
            Assert.Equal(-4, status);
            Assert.Contains("where the emulator has none", message);
            Assert.Equal(before, Piles(Fight(run)));
        }
        finally
        {
            RunApi.Destroy(run);
        }
    }

    /// <summary>The export with the record pointer and length as given, no message and no observation buffer.</summary>
    private static unsafe int RawReanchor(int run, byte* record, int recordLen)
    {
        delegate* unmanaged<int, byte*, int, byte*, int, int*, int> reanchor = &NativeExports.Sts2Run_ReanchorFight;
        return reanchor(run, record, recordLen, null, 0, null);
    }

    [Fact]
    public unsafe void NoFightNoRunNoJsonAreRefusedWithTheirOwnCodes()
    {
        SetTables();
        Assert.Equal(-1, Reanchor(-1, "{}").Status);
        int run = RunApi.Create();
        int fighting = RunApi.InCombat();
        try
        {
            Assert.Equal(-1, Reanchor(run, "{}").Status);  // a run on its map
            Assert.Equal(-2, Reanchor(fighting, "not json").Status);
            Assert.Equal(-2, RawReanchor(fighting, null, 0));
            byte[] text = Encoding.UTF8.GetBytes("{}");
            fixed (byte* t = text)
            {
                Assert.Equal(-2, RawReanchor(fighting, t, -1));  // a negative length is refused, not thrown
            }
        }
        finally
        {
            RunApi.Destroy(fighting);
            RunApi.Destroy(run);
        }
    }

    [Fact]
    public unsafe void TablesThatAreNotTheTwoObjectsAreRefused()
    {
        delegate* unmanaged<byte*, int, int> set = &NativeExports.Sts2Run_SetAnchorTables;
        byte[] text = Encoding.UTF8.GetBytes("{\"id_map\": 1}");
        fixed (byte* t = text)
        {
            Assert.Equal(-1, set(t, text.Length));
        }
    }

    private static string Seq(IEnumerable<CardInstance> pile) => string.Join(",", pile.Select(x => $"{x.DefId}{(x.Upgraded ? "+" : "")}"));

    /// <summary>
    /// The record the referee's driver hands the emulator side (Env.AnchorStateOf) joined to the public one: the draw pile in
    /// the game's order and every stream's seed and counter.
    /// </summary>
    private static string Hidden(CombatState c)
    {
        var tables = new FightAnchor.Tables(Tables.Value.IdMap, Tables.Value.Aliases);
        var cards = tables.Cards.GroupBy(kv => kv.Value).ToDictionary(g => g.Key, g => g.First().Key);
        var record = JsonNode.Parse(Record(c))!.AsObject();
        record["draw"] = new JsonArray(c.DrawPile.Select(x => (JsonNode)new JsonArray(cards[x.DefId], x.Upgraded)).ToArray());
        var rng = new JsonObject();
        void Add(string name, Random? r)
        {
            if (r is CountingRandom s) rng[name] = new JsonObject { ["seed"] = (long)(uint)s.Seed, ["counter"] = s.CallCount };
        }

        Add("Shuffle", c.ShuffleRng);
        Add("CombatTargets", c.TargetRng);
        Add("CombatCardSelection", c.CardSelectionRng);
        Add("CombatCardGeneration", c.CardGenerationRng);
        Add("CombatEnergyCosts", c.EnergyCostRng);
        Add("CombatPotionGeneration", c.PotionGenerationRng);
        Add("Niche", c.NicheHpRng);
        Add("MonsterAi", c.AiRng);
        record["rng"] = rng;
        return record.ToJsonString();
    }

    /// <summary>A fight of the same enemies as <paramref name="model" /> shuffled another way: a run of another seed stands for the game.</summary>
    private static int GameLike(CombatState model)
    {
        for (int i = 0; i < 60; i++)
        {
            int run = RunApi.InCombat($"QS2GYXRK{i:D2}");
            var c = Fight(run);
            if (c.Enemies.Select(e => e.DefId).SequenceEqual(model.Enemies.Select(e => e.DefId)) && Seq(c.DrawPile) != Seq(model.DrawPile))
            {
                return run;
            }

            RunApi.Destroy(run);
        }

        throw new InvalidOperationException("no seed gives the same enemies and another shuffle");
    }

    private static unsafe (int Status, string Message) ReanchorHidden(int run, string record, int flags)
    {
        delegate* unmanaged<int, byte*, int, int, byte*, int, int*, int> reanchor = &NativeExports.Sts2Run_ReanchorFightHidden;
        byte[] text = Encoding.UTF8.GetBytes(record);
        var message = new byte[512];
        var obs = new int[RunConstants.RunObsSize];
        fixed (byte* t = text)
        fixed (byte* m = message)
        fixed (int* o = obs)
        {
            int status = reanchor(run, t, text.Length, flags, m, message.Length, o);
            return (status, Encoding.UTF8.GetString(message, 0, Array.IndexOf(message, (byte)0)));
        }
    }

    /// <summary>
    /// M1 of the F11 diagnosis: a re-anchor from the observation (no streams, a sorted draw pile) leaves the emulator drawing from
    /// its own order and its own Shuffle stream, so the next hand and the reshuffle differ from the game's. With the game's
    /// streams and draw order the fight draws what the game draws. Three runs on two seeds stand for the game and two
    /// emulators; the control is anchored the old way and must not match.
    /// </summary>
    [Fact]
    public void AFightAnchoredWithStreamsAndDrawOrderDrawsTheGamesNextCards()
    {
        SetTables();
        int hidden = RunApi.InCombat();
        int control = RunApi.InCombat();
        int game = GameLike(Fight(hidden));
        try
        {
            Assert.NotEqual(Seq(Fight(game).DrawPile), Seq(Fight(hidden).DrawPile));  // two shuffles of one deck

            var (status, message) = ReanchorHidden(hidden, Hidden(Fight(game)), RunNativeExports.AnchorStreams | RunNativeExports.AnchorDrawOrder);
            Assert.True(status > 0, $"status {status}: {message}");
            Assert.Equal(Seq(Fight(game).Hand), Seq(Fight(hidden).Hand));
            Assert.Equal(Seq(Fight(game).DrawPile), Seq(Fight(hidden).DrawPile));
            Assert.Equal(Fight(game).ShuffleRng!.CallCount, Fight(hidden).ShuffleRng!.CallCount);
            Assert.Equal(Fight(game).ShuffleRng!.Seed, Fight(hidden).ShuffleRng!.Seed);

            Assert.True(Reanchor(control, Record(Fight(game))).Status >= 0);
            Assert.Equal(Seq(Fight(game).Hand), Seq(Fight(control).Hand));  // the visible part agrees on both routes

            bool controlDiffers = Seq(Fight(game).DrawPile) != Seq(Fight(control).DrawPile);
            for (int turn = 0; turn < 3; turn++)
            {
                foreach (int run in new[] { game, hidden, control })
                {
                    Play(run, Fight(run).Hand.Count);  // end the turn: the enemies act, the next hand is drawn (a reshuffle on the third)
                }

                Assert.Equal(Seq(Fight(game).Hand), Seq(Fight(hidden).Hand));
                Assert.Equal(Seq(Fight(game).DrawPile), Seq(Fight(hidden).DrawPile));
                Assert.Equal(Seq(Fight(game).DiscardPile), Seq(Fight(hidden).DiscardPile));
                controlDiffers |= Seq(Fight(game).Hand) != Seq(Fight(control).Hand);
            }

            Assert.True(controlDiffers, "the observation-only route happened to draw what the game drew: pick other seeds");
        }
        finally
        {
            RunApi.Destroy(game);
            RunApi.Destroy(hidden);
            RunApi.Destroy(control);
        }
    }

    [Fact]
    public void WithoutTheFlagsTheHiddenExportReadsNeitherTheStreamsNorTheOrder()
    {
        SetTables();
        int run = RunApi.InCombat();
        int game = GameLike(Fight(run));
        try
        {
            // the record as an agent sees it (draw pile sorted), with the streams beside it
            var record = JsonNode.Parse(Hidden(Fight(game)))!.AsObject();
            record["draw"] = JsonNode.Parse(JsonNode.Parse(Record(Fight(game)))!["draw"]!.ToJsonString());
            int shuffleCalls = Fight(run).ShuffleRng!.CallCount;
            int shuffleSeed = Fight(run).ShuffleRng!.Seed;
            var (status, message) = ReanchorHidden(run, record.ToJsonString(), 0);
            Assert.True(status >= 0, message);
            Assert.Equal(shuffleCalls, Fight(run).ShuffleRng!.CallCount);
            Assert.Equal(shuffleSeed, Fight(run).ShuffleRng!.Seed);
            Assert.NotEqual(Seq(Fight(game).DrawPile), Seq(Fight(run).DrawPile));
            Assert.Equal(Seq(Fight(game).Hand), Seq(Fight(run).Hand));

            // streams alone: the combat's streams are the game's, the draw order still is not
            (status, message) = ReanchorHidden(run, record.ToJsonString(), RunNativeExports.AnchorStreams);
            Assert.True(status >= 0, message);
            Assert.Equal(Fight(game).ShuffleRng!.Seed, Fight(run).ShuffleRng!.Seed);
            Assert.Equal(Fight(game).ShuffleRng!.CallCount, Fight(run).ShuffleRng!.CallCount);
            Assert.NotEqual(Seq(Fight(game).DrawPile), Seq(Fight(run).DrawPile));
        }
        finally
        {
            RunApi.Destroy(game);
            RunApi.Destroy(run);
        }
    }

    private static unsafe int SetStreams(int run, string json)
    {
        delegate* unmanaged<int, byte*, int, int> set = &NativeExports.Sts2Run_DebugSetStreams;
        byte[] text = Encoding.UTF8.GetBytes(json);
        fixed (byte* t = text)
        {
            return set(run, t, text.Length);
        }
    }

    [Fact]
    public void TheRunsStreamsAreSetToTheGamesCountersForwardAndBack()
    {
        int run = RunApi.Create();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(run, out var engine));
            var shuffle = engine!.State.Rng.Shuffle;
            long seed = (long)(uint)shuffle.RawSeed;
            Sts2Emulator.Core.Rng.GameRng At(int n)
            {
                var r = new Sts2Emulator.Core.Rng.GameRng(unchecked((uint)shuffle.RawSeed));
                r.AdvanceToCallCount(n);
                return r;
            }

            // Niche is given another seed (not the run's stream) and Nope is no stream: only Shuffle moves.
            Assert.Equal(1, SetStreams(run, $"{{\"Shuffle\":{{\"seed\":{seed},\"counter\":7}},\"Niche\":{{\"seed\":1,\"counter\":9}},\"Nope\":{{\"seed\":1,\"counter\":3}}}}"));
            Assert.Equal(7, shuffle.CallCount);
            Assert.Equal(At(7).NextInt(100000), shuffle.NextInt(100000));

            // behind where it stands: rewound from the seed, not refused
            Assert.Equal(1, SetStreams(run, $"{{\"Shuffle\":{{\"seed\":{seed},\"counter\":3}}}}"));
            Assert.Equal(3, shuffle.CallCount);
            Assert.Equal(At(3).NextInt(100000), shuffle.NextInt(100000));

            Assert.Equal(0, SetStreams(run, $"{{\"Shuffle\":{{\"seed\":{seed},\"counter\":4}}}}"));  // already there
            Assert.Equal(-2, SetStreams(run, "[1]"));
            Assert.Equal(-1, SetStreams(-1, "{}"));
        }
        finally
        {
            RunApi.Destroy(run);
        }
    }

    [Fact]
    public void ABadStreamPayloadIsRefusedWithACodeAndLeavesTheRunAsItWas()
    {
        SetTables();
        int run = RunApi.InCombat();
        int game = GameLike(Fight(run));
        try
        {
            Assert.True(RunNativeExports.TryGetRun(run, out var engine));
            var shuffle = engine!.State.Rng.Shuffle;
            int before = shuffle.CallCount;
            long seed = (long)(uint)shuffle.RawSeed;
            long upSeed = (long)(uint)engine.State.Rng.UpFront.RawSeed;
            // all or nothing: a good Shuffle entry beside a bad one moves nothing
            Assert.Equal(-2, SetStreams(run, $"{{\"Shuffle\":{{\"seed\":{seed},\"counter\":{before + 5}}},\"UpFront\":{{\"seed\":{upSeed},\"counter\":-3}}}}"));
            Assert.Equal(before, shuffle.CallCount);
            Assert.Equal(-2, SetStreams(run, $"{{\"Shuffle\":{{\"seed\":\"x\",\"counter\":1}}}}"));
            Assert.Equal(-2, SetStreams(run, "{\"Niche\":{\"counter\":1}}"));
            Assert.Equal(before, shuffle.CallCount);

            // through a fight re-anchor: the combat loader ignores UpFront, so only the check before the swap catches it
            var combat = Fight(run);
            string piles = Piles(combat);
            var record = JsonNode.Parse(Hidden(Fight(game)))!.AsObject();
            record["rng"]!.AsObject()["UpFront"] = new JsonObject { ["seed"] = upSeed, ["counter"] = -3 };
            var (status, message) = ReanchorHidden(run, record.ToJsonString(), RunNativeExports.AnchorStreams | RunNativeExports.AnchorDrawOrder);
            Assert.Equal(-2, status);
            Assert.Contains("UpFront", message);
            Assert.Same(combat, Fight(run));
            Assert.Equal(piles, Piles(Fight(run)));
            Assert.Equal(before, shuffle.CallCount);
        }
        finally
        {
            RunApi.Destroy(run);
            RunApi.Destroy(game);
        }
    }

    /// <summary>
    /// A search forks the lockstep state with a resampling seed: every combat stream is reseeded, so a leaked Shuffle seed
    /// would predict the next reshuffle. The counts are history and stay the game's by design; the seeds must not.
    /// </summary>
    [Fact]
    public void AResampledForkOfAStreamAnchoredFightHasOtherStreamSeedsAndTheGamesCounts()
    {
        SetTables();
        int hidden = RunApi.InCombat();
        int game = GameLike(Fight(hidden));
        try
        {
            var (status, message) = ReanchorHidden(hidden, Hidden(Fight(game)), RunNativeExports.AnchorStreams | RunNativeExports.AnchorDrawOrder);
            Assert.True(status > 0, message);
            Assert.True(RunNativeExports.TryGetRun(hidden, out var engine));
            var g = Fight(game);
            Assert.Equal(g.ShuffleRng!.Seed, Fight(hidden).ShuffleRng!.Seed);  // the lockstep state stands on the game's streams
            var seeds = new HashSet<int>();
            var orders = new HashSet<string>();
            for (int s = 1; s <= 50; s++)
            {
                var fork = engine!.ForkActiveCombat(s);
                Assert.NotNull(fork);
                var c = fork!.Value.Combat;
                Assert.NotEqual(g.ShuffleRng.Seed, c.ShuffleRng!.Seed);
                Assert.Equal(g.ShuffleRng.CallCount, c.ShuffleRng.CallCount);
                Assert.NotEqual(g.CardGenerationRng!.Seed, c.CardGenerationRng!.Seed);
                Assert.Equal(g.CardGenerationRng.CallCount, c.CardGenerationRng.CallCount);
                if (g.AiRng is CountingRandom gai && c.AiRng is CountingRandom cai)
                {
                    Assert.NotEqual(gai.Seed, cai.Seed);
                }

                seeds.Add(c.ShuffleRng.Seed);
                orders.Add(Seq(c.DrawPile));
            }

            Assert.Equal(50, seeds.Count);
            Assert.True(orders.Count > 5, "the clones' draw orders vary");
            Assert.Equal(Seq(g.DrawPile), Seq(Fight(hidden).DrawPile));  // forking moved nothing
        }
        finally
        {
            RunApi.Destroy(hidden);
            RunApi.Destroy(game);
        }
    }

    /// <summary>The run exports a fight is made and destroyed through (as RunForkCombatTests does).</summary>
    private static unsafe class RunApi
    {
        public static int Create(string seed = "QS2GYXRKWN")
        {
            delegate* unmanaged<int> create = &NativeExports.Sts2Run_Create;
            int run = create();
            delegate* unmanaged<int, byte*, int, int, int*, int> reset = &NativeExports.Sts2Run_Reset;
            byte[] text = Encoding.UTF8.GetBytes(seed);
            var obs = new int[RunConstants.RunObsSize];
            fixed (byte* s = text)
            fixed (int* o = obs)
            {
                Assert.Equal(0, reset(run, s, text.Length, 10, o));
            }

            return run;
        }

        public static int InCombat(string seed = "QS2GYXRKWN")
        {
            int run = Create(seed);
            delegate* unmanaged<int, int*, int, int, int*, int, int, int, int*, int, int, int, int*, int> start =
                &NativeExports.Sts2Run_StartCombat;
            int[] deck = [.. RunConstants.StarterDeckIds];
            int[] potions = [5];
            var obs = new int[RunConstants.RunObsSize];
            fixed (int* d = deck)
            fixed (int* p = potions)
            fixed (int* o = obs)
            {
                Assert.Equal(0, start(run, d, deck.Length, RunConstants.SlitheringStranglerEncounterId, null, 0, 64, 80, p, potions.Length, 99, 0, o));
            }

            return run;
        }

        public static void Destroy(int run)
        {
            delegate* unmanaged<int, void> destroy = &NativeExports.Sts2Run_Destroy;
            destroy(run);
        }
    }
}
