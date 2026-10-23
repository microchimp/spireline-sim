// EmuDump kernel-order <scenarios.json> <out.json>: the kernel's listener list (Core2/OldState.cs) for each of
// tools/fidelity/hook_order.py's scenarios, built as `hookprobe` builds them (relics, potions, powers on the player and
// one enemy, cards in the piles) on the plain engine. tools/kernel/listener_order.py compares the order with
// hook_table.game_order_key. The engine's behaviour is not touched: the kernel only reads the state.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core2;

internal static class KernelOrderMode
{
    public static int Run(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("usage: EmuDump kernel-order <scenarios.json> <out.json>");
            return 2;
        }

        var ids = OldModelIds.Load(Path.Combine(AppContext.BaseDirectory, "id_map.json"),
            Path.Combine(AppContext.BaseDirectory, "power_aliases.json"));
        var spec = JsonNode.Parse(File.ReadAllText(args[0]))!;
        var results = new JsonArray();
        foreach (var sc in spec["scenarios"]!.AsArray())
        {
            var row = new JsonObject { ["name"] = sc!["name"]!.GetValue<string>() };
            try
            {
                var state = Build(sc, out int playerOffset);
                var gaps = new List<OldGap>();
                var list = OldListeners.Enumerate(new OldWorld(state, null, ids), HookIterator.CombatDirect, gaps);
                row["listeners"] = new JsonArray(list.Select(e => (JsonNode)new JsonObject
                {
                    ["kind"] = e.Kind.ToString().ToLowerInvariant(),
                    ["model"] = e.Who.ModelId,
                    ["creature"] = e.Creature,
                    ["index"] = e.Index,
                    ["pile"] = e.Pile == OldPile.None ? null : e.Pile.ToString().ToLowerInvariant(),
                    ["buff"] = e.Kind == ListenerKind.Power ? e.Buff.ToString() : null,
                    ["enemy"] = e.Enemy is null ? null : state.Enemies.IndexOf(e.Enemy),
                    ["enchantment"] = e.Kind == ListenerKind.Enchantment ? ((Enchantment)e.Who.ModelId).ToString() : null,
                }).ToArray());
                row["buffs"] = new JsonObject
                {
                    ["player"] = new JsonArray(state.PlayerBuffs.Select(b => (JsonNode)b.Id.ToString()).ToArray()),
                    ["enemy"] = new JsonArray(state.Enemies.SelectMany(x => x.Buffs).Select(b => (JsonNode)b.Id.ToString()).ToArray()),
                };
                // The scenario's player powers start here (a relic may have put the same power on first).
                row["player_offset"] = playerOffset;
                row["gaps"] = new JsonArray(gaps.Select(g => (JsonNode)$"{g.Why}:{g.What}").Distinct().ToArray());
            }
            catch (Exception e)
            {
                row["error"] = $"{e.GetType().Name}: {e.Message}";
            }

            results.Add(row);
        }

        File.WriteAllText(args[1], new JsonObject { ["scenarios"] = results }.ToJsonString());
        return 0;
    }

    private static CardInstance Card(JsonNode node)
    {
        var a = node.AsArray();
        var ench = a.Count > 1 && a[1] is not null ? Enum.Parse<Enchantment>(a[1]!.GetValue<string>()) : Enchantment.None;
        return new CardInstance(a[0]!.GetValue<int>(), false, Enchantment: ench, EnchantAmount: ench == Enchantment.None ? 0 : 1);
    }

    private static List<BuffState> Buffs(JsonNode? node) =>
        node?.AsArray().Select(b => new BuffState(Enum.Parse<BuffId>(b![0]!.GetValue<string>()), b[1]!.GetValue<int>())).ToList()
        ?? [];

    /// <summary>The scenario's combat as HookProbe's RunScenario builds it, before its actions.</summary>
    private static CombatState Build(JsonNode sc, out int playerOffset)
    {
        var rng = new Random(0);
        var state = new CombatState { NicheHpRng = new CountingRandom(0) };
        var deck = (sc["deck"]?.AsArray() ?? []).Select(n => Card(n!)).ToArray();
        int[] relics = sc["relics"]?.AsArray().Select(x => x!.GetValue<int>()).ToArray() ?? [];
        int[] potions = sc["potions"]?.AsArray().Select(x => x!.GetValue<int>()).ToArray() ?? [];
        int hp = sc["hp"]?.GetValue<int>() ?? 999;
        CombatFactory.Reset(state, rng, deck, sc["encounter"]?.GetValue<int>() ?? 1, relics, hp, hp, potions, 99);
        int enemyHp = sc["enemy_hp"]?.GetValue<int>() ?? 9999;
        state.Enemies =
        [
            new EnemyState
            {
                DefId = sc["enemy_def"]?.GetValue<int>() ?? 16, Hp = enemyHp, MaxHp = enemyHp, Buffs = Buffs(sc["enemy_buffs"]),
                CurrentIntent = new Intent(IntentType.Attack, 6),
            },
        ];
        playerOffset = state.PlayerBuffs.Count;
        state.PlayerBuffs.AddRange(Buffs(sc["player_buffs"]));
        if (sc["hand"] is JsonArray hand)
        {
            state.Hand = hand.Select(n => state.Fresh(Card(n!))).ToList();
        }

        if (sc["draw"] is JsonArray draw)
        {
            state.DrawPile = draw.Select(n => state.Fresh(Card(n!))).ToList();
        }

        if (sc["discard"] is JsonArray discard)
        {
            state.DiscardPile = discard.Select(n => state.Fresh(Card(n!))).ToList();
        }

        return state;
    }
}
