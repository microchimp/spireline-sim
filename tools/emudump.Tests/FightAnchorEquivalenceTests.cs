// The snapshot loader moved from tools/emudump (OldAnchor) into the engine DLL (FightAnchor, behind Sts2Run_ReanchorFight).
// What it builds must not have moved: for a sample of fights, every state the emulator reaches (end turn after end turn) is
// written over the state before it, from the game-shaped snapshot of the state after, as Reanchor does for each recorded
// step, and the result is hashed (the dump the comparer reads, the draw order, the streams, the potions, and what was
// patched). EXPECTED holds the hashes the tools' own loader produced before the move (commit ccb04363); the tools' wrapper and
// the DLL's class must both give them.
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using EmuDump.Engine;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Xunit;

public class FightAnchorEquivalenceTests
{
    static readonly string[] Fixtures =
    [
        "kifuda", "kifuda_then_royal_stamp", "royal_stamp_on_other_cards", "dollys_mirror_on_other_cards", "gnarled_hammer_then_punch_dagger",
    ];

    static JsonObject Fixture(string name) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", name + ".json")))!.AsObject();

    static string Snake(string name) => Regex.Replace(name, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

    /// <summary>The state as the referee records the game's snapshot (referee/driver/Combat.cs), built from the emulator's.</summary>
    static JsonObject GameView(CombatState c, bool observation)
    {
        var cards = OldReanchor.Tables.Cards.GroupBy(kv => kv.Value).ToDictionary(g => g.Key, g => g.First().Key);
        var enemies = OldReanchor.Tables.Enemies.GroupBy(kv => kv.Value).ToDictionary(g => g.Key, g => g.First().Key);
        JsonArray Pile(IEnumerable<CardInstance> p) => new(p.Select(x => (JsonNode)new JsonArray(cards[x.DefId], x.Upgraded)).ToArray());
        JsonArray Powers(List<BuffState> bs) => new(bs.Select(b => (JsonNode)new JsonArray(Snake(b.Id.ToString()) + "_POWER", b.Magnitude)).ToArray());
        JsonObject Rng(CountingRandom? r) => new() { ["seed"] = (long)(uint)(r?.Seed ?? 0), ["counter"] = r?.CallCount ?? 0 };
        var rng = new JsonObject();
        foreach (var (name, r) in new (string, CountingRandom?)[]
                 {
                     ("Shuffle", c.ShuffleRng), ("CombatTargets", c.TargetRng), ("MonsterAi", c.AiRng as CountingRandom),
                     ("CombatCardSelection", c.CardSelectionRng), ("CombatCardGeneration", c.CardGenerationRng),
                     ("CombatEnergyCosts", c.EnergyCostRng), ("CombatPotionGeneration", c.PotionGenerationRng),
                     ("CombatOrbGeneration", c.OrbGenerationRng), ("Niche", c.NicheHpRng),
                 })
        {
            if (r != null) rng[name] = Rng(r);
        }

        var draw = Pile(c.DrawPile);
        if (observation) draw = new JsonArray(draw.Select(x => x!.ToJsonString()).Order(StringComparer.Ordinal).Select(x => JsonNode.Parse(x)).ToArray());
        var view = new JsonObject
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
        };
        if (!observation) view["rng"] = rng;
        return view;
    }

    /// <summary>Each state of a fight played by ending turns, with the one before it.</summary>
    internal static IEnumerable<(string Label, CombatState Prior, CombatState After)> Steps()
    {
        foreach (var name in Fixtures)
        {
            var engine = new OldEngine();
            engine.StartFight(new FightSpec(Fixture(name)), new JsonObject());
            var states = new List<CombatState> { engine.Run.State.ActiveCombat!.Clone() };
            for (int k = 0; k < 4 && engine.CombatOpen && engine.Screen == null; k++)
            {
                engine.EndTurn();
                if (engine.Run.State.ActiveCombat is { } c) states.Add(c.Clone());
            }

            for (int k = 1; k < states.Count; k++) yield return ($"{name}#{k}", states[k - 1], states[k]);
        }
    }

    static string Hash(CombatState q, string? problem, IEnumerable<string> patched)
    {
        var text = new StringBuilder();
        text.Append(OldReanchor.DumpLite(q, true).ToJsonString()).Append('|').Append(problem).Append('|').Append(string.Join(",", patched));
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text.ToString())))[..16].ToLowerInvariant();
    }

    static List<string> ViaTools(bool observationRecord = false)
    {
        var got = new List<string>();
        foreach (var (label, prior, after) in Steps())
        {
            var q = prior.Clone();
            var patched = new JsonArray();
            string? problem = OldAnchor.Anchor(OldReanchor.Tables, q, GameView(after, observationRecord), patched);
            got.Add($"{label} {Hash(q, problem, patched.Select(x => (string)x!))}");
        }

        return got;
    }

    static List<string> ViaDll()
    {
        var got = new List<string>();
        foreach (var (label, prior, after) in Steps())
        {
            var q = prior.Clone();
            var patched = new List<string>();
            string? problem = FightAnchor.Anchor(OldReanchor.Tables, q, GameView(after, false), patched);
            got.Add($"{label} {Hash(q, problem, patched)}");
        }

        return got;
    }

    static readonly string[] Expected =
    [
        "kifuda#1 c65378105504ac1f",
        "kifuda#2 76d39750b6e0cf50",
        "kifuda#3 4e615ecb7085160f",
        "kifuda#4 fdc1b9e78a179187",
        "kifuda_then_royal_stamp#1 780b90e2866b1529",
        "kifuda_then_royal_stamp#2 845168614ede3695",
        "kifuda_then_royal_stamp#3 ccf2be07e6891ec8",
        "kifuda_then_royal_stamp#4 ccf2be07e6891ec8",
        "royal_stamp_on_other_cards#1 e5eaf8d6f72f2eb9",
        "royal_stamp_on_other_cards#2 c32a1ed8c7f69aa7",
        "royal_stamp_on_other_cards#3 498a9ff2c3e4be6d",
        "royal_stamp_on_other_cards#4 2fc87d692c3b4185",
        // Dolly's Mirror's record starts from Neow's Bones: a relic claimed off its screen opens its own selection since
        // fix/f16-gold-screens, so the setup deck (and every state after it) is the game's now, not the loader's of ccb04363.
        "dollys_mirror_on_other_cards#1 7d21b176f963bba0",
        "dollys_mirror_on_other_cards#2 dcf5432a1e58b482",
        "dollys_mirror_on_other_cards#3 69a58a1b6de29b32",
        "dollys_mirror_on_other_cards#4 63fe7e6a9ae8b8b4",
        "gnarled_hammer_then_punch_dagger#1 cc799c4d4708182e",
        "gnarled_hammer_then_punch_dagger#2 17ba6d38dea5ce94",
        "gnarled_hammer_then_punch_dagger#3 7b34fba6076cfdb3",
        "gnarled_hammer_then_punch_dagger#4 0621aafa33ef24e7",
    ];

    [Fact]
    public void TheToolsWrapperBuildsWhatTheToolsLoaderBuiltBeforeTheMove() => Assert.Equal(Expected, ViaTools());

    [Fact]
    public void TheDllLoaderBuildsWhatTheToolsLoaderBuiltBeforeTheMove() => Assert.Equal(Expected, ViaDll());
}
