using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
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

    [Fact]
    public void NoFightNoRunNoJsonAreRefusedWithTheirOwnCodes()
    {
        SetTables();
        Assert.Equal(-1, Reanchor(-1, "{}").Status);
        int run = RunApi.Create();
        int fighting = RunApi.InCombat();
        try
        {
            Assert.Equal(-1, Reanchor(run, "{}").Status);  // a run on its map
            Assert.Equal(-2, Reanchor(fighting, "not json").Status);
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

    /// <summary>The run exports a fight is made and destroyed through (as RunForkCombatTests does).</summary>
    private static unsafe class RunApi
    {
        public static int Create()
        {
            delegate* unmanaged<int> create = &NativeExports.Sts2Run_Create;
            int run = create();
            delegate* unmanaged<int, byte*, int, int, int*, int> reset = &NativeExports.Sts2Run_Reset;
            byte[] text = Encoding.UTF8.GetBytes("QS2GYXRKWN");
            var obs = new int[RunConstants.RunObsSize];
            fixed (byte* s = text)
            fixed (int* o = obs)
            {
                Assert.Equal(0, reset(run, s, text.Length, 10, o));
            }

            return run;
        }

        public static int InCombat()
        {
            int run = Create();
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
