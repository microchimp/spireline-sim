using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace EmuDump.Engines.Old;

/// <summary>The old engine's native dump of a fight step, built from the run engine's own data.</summary>
static class OldDump
{
    internal static JsonObject Dump(RunEngine e) =>
        DumpCombat(e.State.ActiveCombat, e.State.ActiveCombat != null && e.State.Phase == RunPhase.Combat, e.State.Phase.ToString(), e.State);

    /// <summary>One step's state as `EmuDump fight` prints it (tools/emudump.Tests/LifecycleTests.cs reads it too).
    /// <paramref name="run"/> gives a finished fight its <c>final</c> block.</summary>
    internal static JsonObject DumpCombat(CombatState? c, bool inCombat, string phase, RunState? run = null)
    {
        var o = new JsonObject { ["in_combat"] = inCombat, ["run_phase"] = phase };
        if (c == null) return o;
        // A finished fight keeps its CombatState on the run (SyncAfterCombat copies out what the run keeps).
        if (!inCombat && run != null) o["final"] = Final(run, run.LastPlayerWon);
        byte[] written;
        using (Prof.Scope("dump.snapshot_write")) written = Sts2Emulator.Interop.CombatSnapshot.WriteUnchecked(c);
        using var parse = Prof.Scope("dump.parse_and_rest");
        var snap = JsonNode.Parse(written)!;
        // Buff ids are the emulator's BuffId enum, not id_map's power ids: name them here.
        foreach (var holder in new[] { snap["player"]! }.Concat(snap["enemies"]!.AsArray().Select(x => x!)))
        {
            foreach (var b in holder["buffs"]!.AsArray())
            {
                b!["name"] = ((BuffId)(int)b["id"]!).ToString();
            }
        }

        // The snapshot's intent lacks the move's second declared intent (it derives it on load).
        var se = snap["enemies"]!.AsArray();
        for (int i = 0; i < se.Count && i < c.Enemies.Count; i++) se[i]!["intent"] = IntentJson(c.Enemies[i]);
        o["snapshot"] = snap;
        if (Setup.Extended) o["potions"] = Setup.Belt(c.PotionSlots, c.MaxPotionSlots);
        o["draw_order"] = new JsonArray(c.DrawPile.Select(x => (JsonNode)new JsonArray(x.DefId, x.Upgraded)).ToArray());
        int N(CountingRandom? r) => r?.CallCount ?? -1;
        o["rng"] = new JsonObject
        {
            ["Shuffle"] = N(c.ShuffleRng), ["CombatTargets"] = N(c.TargetRng), ["MonsterAi"] = N(c.AiRng as CountingRandom),
            ["CombatCardSelection"] = N(c.CardSelectionRng), ["CombatCardGeneration"] = N(c.CardGenerationRng),
            ["CombatEnergyCosts"] = N(c.EnergyCostRng), ["CombatPotionGeneration"] = N(c.PotionGenerationRng),
            ["CombatOrbGeneration"] = N(c.OrbGenerationRng), ["Niche"] = N(c.NicheHpRng),
        };
        return o;
    }

    /// <summary>
    /// What a fight leaves once it is over, written beside the dump of the step that ended it.
    /// The game tears a WON fight down inside that step (Player.AfterCombatEnd: the player's powers, block and
    /// piles go, the roster with the combat), so its snapshot then shows what the run holds: hit points
    /// with the after-combat heals in, the potion belt, the relics and the RNG streams. The emulator's
    /// run is that state (the CombatState it keeps is not torn down), so this writes the run's. A LOST
    /// fight ends the run and writes only that: the game freezes the combat where the player died, the
    /// emulator finishes the enemy phase, and what either left goes nowhere.
    /// Gold and the relics' counters are written too; the game's snapshot has neither yet, and
    /// tools/fidelity/fight_compare.py compares them once a record has them.
    /// </summary>
    internal static JsonObject Final(RunState s, bool won)
    {
        var f = new JsonObject { ["won"] = won };
        if (!won) return f;

        f["player"] = new JsonObject { ["hp"] = s.PlayerHp, ["max_hp"] = s.PlayerMaxHp, ["gold"] = s.Gold };
        f["rng"] = Setup.RunRng(s);
        if (Setup.Extended)
        {
            f["potions"] = Setup.Belt(s.PotionSlots, s.MaxPotionSlots);
            f["relics"] = new JsonArray(s.Relics.Select(r => (JsonNode)Setup.RelicEntry(r.DefId)).ToArray());
            f["relic_counters"] = new JsonArray(s.Relics.Select(r => (JsonNode)r.Counter).ToArray());
        }

        return f;
    }

    /// <summary>
    /// The emulator's own move for one enemy: its announced intent (type is the IntentType enum, magnitude
    /// the base damage per hit for an attack), the second intent its move declares, and the AI position
    /// that chose it. tools/fidelity/fight_compare.py maps this and the game's MoveState intents to one
    /// vocabulary; move_index/last_move only explain, they are not compared.
    /// </summary>
    internal static JsonObject IntentJson(EnemyState en)
    {
        static JsonObject One(Intent it) => new()
        {
            ["type"] = (int)it.Type, ["magnitude"] = it.Magnitude, ["hits"] = it.Hits, ["carries_damage"] = it.CarriesDamage,
        };
        var o = One(en.CurrentIntent);
        o["secondary"] = en.SecondaryIntent is { } s ? One(s) : null;
        o["move_index"] = en.MoveIndex;
        o["last_move"] = en.LastMove;
        return o;
    }
}
