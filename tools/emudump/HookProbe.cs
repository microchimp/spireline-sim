// EmuDump hookprobe <scenarios.json> <out.json>: run synthetic combats on an engine that
// tools/fidelity/hook_order.py instrumented, and write down the order its listener probes fired in.
//
// The probes (`Sts2Emulator.HookProbe`, written into the scratch copy of the engine only) sit before
// every statement of a dispatch point that names a listener (tools/fidelity/hook_dispatch.json). This
// mode builds one combat per scenario holding the scenario's listeners -- relics in obtain order,
// potions by slot, powers in application order on the player and on one dummy enemy, cards in the
// piles -- runs its actions and records the probe ids, with marks between the actions and at
// CombatTrace's seams (play, played, turn_end, enemy_turn, move, moved, turn_start). The engine is
// read through reflection, so this compiles against the plain engine too and says so when run there.
using System.Reflection;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;

internal static class HookProbeMode
{
    // CombatTrace.Mark names -> the negative codes the log carries; an action boundary is -(100 + i).
    private static readonly Dictionary<string, int> MarkCodes = new()
    {
        ["turn_end"] = -1, ["enemy_turn"] = -2, ["turn_start"] = -3, ["play"] = -4, ["played"] = -5,
        ["move"] = -6, ["moved"] = -7,
    };

    public static int Run(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("usage: EmuDump hookprobe <scenarios.json> <out.json>");
            return 2;
        }

        var probe = typeof(CombatState).Assembly.GetType("Sts2Emulator.HookProbe");
        if (probe is null)
        {
            Console.Error.WriteLine("hookprobe: this engine is not instrumented (no Sts2Emulator.HookProbe); "
                + "build it with `python -m tools.fidelity.hook_table order`");
            return 2;
        }

        var on = probe.GetField("On", BindingFlags.Public | BindingFlags.Static)!;
        var log = (List<int>)probe.GetField("Log", BindingFlags.Public | BindingFlags.Static)!.GetValue(null)!;
        var activeMutant = typeof(CombatState).Assembly.GetType("Sts2Emulator.MutCov")
            ?.GetField("Active", BindingFlags.Public | BindingFlags.Static);
        void Mark(int code) => log.Add(code);

        var spec = JsonNode.Parse(File.ReadAllText(args[0]))!;
        var mutants = spec["mutants"]?.AsArray().Select(x => x!.GetValue<int>()).ToArray() ?? [-1];
        var results = new JsonArray();
        foreach (int mutant in mutants)
        {
            activeMutant?.SetValue(null, mutant);
            foreach (var sc in spec["scenarios"]!.AsArray())
            {
                log.Clear();
                string? error = null;
                CombatTrace.Mark = (what, _, _) =>
                {
                    if (MarkCodes.TryGetValue(what, out int code))
                    {
                        Mark(code);
                    }
                };
                on.SetValue(null, true);
                try
                {
                    RunScenario(sc!, Mark);
                }
                catch (Exception e)
                {
                    error = $"{e.GetType().Name}: {e.Message}";
                }

                on.SetValue(null, false);
                CombatTrace.Mark = null;
                var row = new JsonObject
                {
                    ["name"] = sc!["name"]!.GetValue<string>(),
                    ["log"] = new JsonArray(log.Select(x => (JsonNode)JsonValue.Create(x)).ToArray()),
                    ["error"] = error,
                };
                if (activeMutant is not null && spec["mutants"] is not null)
                {
                    row["mutant"] = mutant;
                }

                results.Add(row);
            }
        }

        activeMutant?.SetValue(null, -1);
        File.WriteAllText(args[1], new JsonObject { ["scenarios"] = results }.ToJsonString());
        return 0;
    }

    private static CardInstance Card(JsonNode node)
    {
        var a = node.AsArray();
        int id = a[0]!.GetValue<int>();
        var ench = a.Count > 1 && a[1] is not null
            ? Enum.Parse<Enchantment>(a[1]!.GetValue<string>())
            : Enchantment.None;
        return new CardInstance(id, false, Enchantment: ench, EnchantAmount: ench == Enchantment.None ? 0 : 1);
    }

    private static List<BuffState> Buffs(JsonNode? node) =>
        node?.AsArray().Select(b => new BuffState(Enum.Parse<BuffId>(b![0]!.GetValue<string>()), b[1]!.GetValue<int>())).ToList()
        ?? [];

    private static void RunScenario(JsonNode sc, Action<int> mark)
    {
        var rng = new Random(0);
        var state = new CombatState { NicheHpRng = new CountingRandom(0) };
        var deck = (sc["deck"]?.AsArray() ?? []).Select(n => Card(n!)).ToArray();
        int[] relics = sc["relics"]?.AsArray().Select(x => x!.GetValue<int>()).ToArray() ?? [];
        int[] potions = sc["potions"]?.AsArray().Select(x => x!.GetValue<int>()).ToArray() ?? [];
        int hp = sc["hp"]?.GetValue<int>() ?? 999;
        int encounter = sc["encounter"]?.GetValue<int>() ?? 1;
        mark(-100);
        CombatFactory.Reset(state, rng, deck, encounter, relics, hp, hp, potions, 99);
        state.TargetRng = new CountingRandom(0);
        state.CardSelectionRng = new CountingRandom(0);
        state.CardGenerationRng = new CountingRandom(0);
        state.PotionGenerationRng = new CountingRandom(0);
        state.OrbGenerationRng = new CountingRandom(0);
        Settle(state, rng);

        // After the combat start: the listeners that cannot be there at the start of a fight.
        int enemyHp = sc["enemy_hp"]?.GetValue<int>() ?? 9999;
        int enemyDef = sc["enemy_def"]?.GetValue<int>() ?? 16;
        state.Enemies =
        [
            new EnemyState
            {
                DefId = enemyDef, Hp = enemyHp, MaxHp = enemyHp, Buffs = Buffs(sc["enemy_buffs"]),
                CurrentIntent = new Intent(IntentType.Attack, sc["enemy_attack"]?.GetValue<int>() ?? 6),
            },
        ];
        state.PlayerBuffs.AddRange(Buffs(sc["player_buffs"]));
        if (sc["hand"] is JsonArray hand)
        {
            state.Hand = hand.Select(n => Card(n!)).ToList();
        }

        if (sc["draw"] is JsonArray draw)
        {
            state.DrawPile = draw.Select(n => Card(n!)).ToList();
        }

        if (sc["discard"] is JsonArray discard)
        {
            state.DiscardPile = discard.Select(n => Card(n!)).ToList();
        }

        int i = 0;
        foreach (var act in sc["actions"]?.AsArray() ?? [])
        {
            i++;
            mark(-(100 + i));
            state.Energy = 99;
            string what = act!["do"]!.GetValue<string>();
            StepResult result;
            if (what == "play")
            {
                state.Hand.Insert(0, Card(act["card"]!));
                result = CombatEngine.Step(state, 0, rng, act["target"]?.GetValue<int>() ?? 0);
            }
            else if (what == "end")
            {
                result = CombatEngine.Step(state, state.Hand.Count, rng);
            }
            else if (what == "potion")
            {
                result = CombatEngine.Step(state, state.Hand.Count + 1 + (act["slot"]?.GetValue<int>() ?? 0), rng, 0);
            }
            else
            {
                throw new InvalidOperationException($"unknown action {what}");
            }

            if (result.Terminal || Settle(state, rng))
            {
                break;
            }
        }
    }

    /// <summary>Answers every open screen with its first valid choice. True when the fight ended.</summary>
    private static bool Settle(CombatState state, Random rng)
    {
        for (int guard = 0; guard < 40 && state.PendingSelection is not null; guard++)
        {
            int[] valid = CombatEngine.ValidActions(state);
            if (valid.Length == 0)
            {
                return false;
            }

            if (CombatEngine.Step(state, valid[0], rng, 0).Terminal)
            {
                return true;
            }
        }

        return false;
    }
}
