using System.Reflection;
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;

namespace Sts2Emulator.Tests.Invariants;

/// <summary>One broken invariant: which check, and what was seen.</summary>
public readonly record struct Violation(string Check, string Detail)
{
    public override string ToString() => $"{Check}: {Detail}";
}

/// <summary>
/// What must hold of a live combat between two steps, whatever the fight: the engine's own fields,
/// including the ones no snapshot or recording shows (queues, held cards, generator objects).
/// </summary>
/// <remarks>
/// The recorded-fight checks (tools/fidelity/invariants.py) judge what both sides show. These judge
/// what only the emulator has, on any state, so they run on every step of every fight the metamorphic
/// runner plays. <see cref="Tracker"/> adds the checks that compare a state with the one before it.
/// </remarks>
public static class StateInvariants
{
    public const int HandLimit = CombatObservation.MaxHand;
    public const int MaxEnemies = 8;
    public const int EnergyCap = 30;
    public const int BlockCap = 999;
    public const int MaxPile = 150;

    private static readonly FieldInfo[] TurnCounters = typeof(CombatState)
        .GetFields(BindingFlags.Public | BindingFlags.Instance)
        .Where(f => f.FieldType == typeof(int) && (f.Name.EndsWith("ThisTurn") || f.Name.EndsWith("ThisCombat") || f.Name.EndsWith("LastTurn")))
        .ToArray();

    /// <summary>Violations of the single state <paramref name="s"/> (a state between two steps).</summary>
    public static List<Violation> Check(CombatState s)
    {
        var v = new List<Violation>();
        void Bad(string check, string detail) => v.Add(new Violation(check, detail));

        if (s.PlayerMaxHp <= 0 || s.PlayerHp < 0 || s.PlayerHp > s.PlayerMaxHp)
        {
            Bad("hp_range", $"player hp {s.PlayerHp}/{s.PlayerMaxHp}");
        }

        if (s.Energy < 0 || s.Energy > EnergyCap)
        {
            Bad("energy", $"energy {s.Energy} (max {s.MaxEnergy})");
        }

        if (s.Stars < 0)
        {
            Bad("energy", $"stars {s.Stars}");
        }

        if (s.PlayerBlock < 0 || s.PlayerBlock > BlockCap)
        {
            Bad("block", $"player block {s.PlayerBlock}");
        }

        if (s.Hand.Count > HandLimit)
        {
            Bad("hand_limit", $"hand {s.Hand.Count}");
        }

        foreach (var (name, pile) in Piles(s))
        {
            if (pile.Count > MaxPile)
            {
                Bad("hand_limit", $"{name} pile {pile.Count}");
            }

            for (int i = 0; i < pile.Count; i++)
            {
                if (!KnownCard(pile[i].DefId))
                {
                    Bad("card_id", $"{name}[{i}] has unknown card id {pile[i].DefId}");
                }
            }
        }

        if (s.Enemies.Count > MaxEnemies)
        {
            Bad("enemy_count", $"{s.Enemies.Count} enemies");
        }

        for (int i = 0; i < s.Enemies.Count; i++)
        {
            var e = s.Enemies[i];
            if (e.Hp < 0 || e.MaxHp <= 0 || e.Hp > e.MaxHp)
            {
                Bad("hp_range", $"enemy {i} def {e.DefId} hp {e.Hp}/{e.MaxHp}");
            }

            if (e.Block < 0 || e.Block > BlockCap)
            {
                Bad("block", $"enemy {i} def {e.DefId} block {e.Block}");
            }

            Buffs($"enemy {i} def {e.DefId}", e.Buffs, Bad);
        }

        Buffs("player", s.PlayerBuffs, Bad);

        // Machinery that only runs inside a step: all of it must be back to rest when a step returns.
        if (s.AutoPlayQueue.Count != 0)
        {
            Bad("at_rest", $"auto-play queue holds {s.AutoPlayQueue.Count} cards");
        }

        if (s.HeldPlayedCard is not null && s.PendingSelection is null)
        {
            Bad("at_rest", "a played card is held with no selection open");
        }

        // The other half of the slot's rule: the held card stands for the game's Play pile, so no
        // other pile has it (a card is in exactly one pile). A stale slot beside its filed card is
        // what made the next step file the card twice.
        if (s.HeldPlayedCard is { } held && held.UidOf() != 0)
        {
            foreach (var (name, pile) in Piles(s))
            {
                if (
                    name is "hand" or "draw" or "discard" or "exhaust"
                    && pile.Exists(card => card.UidOf() == held.UidOf())
                )
                {
                    Bad(
                        "at_rest",
                        $"the held played card (uid {held.UidOf()}) is also in the {name} pile"
                    );
                }
            }
        }

        if (s.AutoPlaying)
        {
            Bad("at_rest", "AutoPlaying is set");
        }

        if (s.ResolvingDefendCard)
        {
            Bad("at_rest", "ResolvingDefendCard is set");
        }


        if (s.KnownTopCount < 0 || s.KnownBottomCount < 0 || s.KnownTopCount + s.KnownBottomCount > s.DrawPile.Count)
        {
            Bad("known_order", $"known top {s.KnownTopCount} + bottom {s.KnownBottomCount} of {s.DrawPile.Count}");
        }

        if (s.Orbs.Count > Math.Max(s.OrbCapacity, 0))
        {
            Bad("orbs", $"{s.Orbs.Count} orbs in {s.OrbCapacity} slots");
        }

        foreach (var f in TurnCounters)
        {
            int x = (int)f.GetValue(s)!;
            if (x < 0 && x != int.MinValue)
            {
                Bad("counter_nonneg", $"{f.Name} = {x}");
            }
        }

        // A stream shared by two fields desynchronises both: each must be its own object.
        var streams = Streams(s).Where(x => x.Rng is not null).ToList();
        for (int i = 0; i < streams.Count; i++)
        {
            for (int j = i + 1; j < streams.Count; j++)
            {
                if (ReferenceEquals(streams[i].Rng, streams[j].Rng))
                {
                    Bad("rng_shared", $"{streams[i].Name} and {streams[j].Name} are one generator");
                }
            }
        }

        if (s.Turn < 0)
        {
            Bad("turn", $"turn {s.Turn}");
        }

        return v;
    }

    private static void Buffs(string who, List<BuffState> buffs, Action<string, string> bad)
    {
        var seen = new HashSet<BuffId>();
        foreach (var b in buffs)
        {
            if (b.Magnitude == 0)
            {
                bad("power_amount", $"{who} holds {b.Id} at 0");
            }
            else if (b.Magnitude < 0 && !BuffSystem.AllowsNegative(b.Id) && !SignedByDesign.Contains(b.Id))
            {
                bad("power_amount", $"{who} holds {b.Id} at {b.Magnitude}");
            }

            if (!seen.Add(b.Id) && !BuffSystem.IsInstanced(b.Id))
            {
                bad("power_dup", $"{who} holds {b.Id} twice");
            }
        }
    }

    /// <summary>
    /// Buffs the emulator stores as a signed number on purpose: the temporary stats carry their
    /// direction in the sign, the energy and draw owed carry nothing negative but are listed for
    /// the owner to extend when a new one is found.
    /// </summary>
    private static readonly HashSet<BuffId> SignedByDesign =
    [
        BuffId.TemporaryStrength,
        BuffId.TemporaryFocus,
    ];

    /// <summary>The five card lists plus the queues that hold cards between turns.</summary>
    public static IEnumerable<(string Name, List<CardInstance> Pile)> Piles(CombatState s)
    {
        yield return ("hand", s.Hand);
        yield return ("draw", s.DrawPile);
        yield return ("discard", s.DiscardPile);
        yield return ("exhaust", s.ExhaustPile);
        yield return ("return_to_hand", s.ReturnToHandBeforeDraw);
        yield return ("copies_to_hand", s.CopiesToHandBeforeDraw);
        yield return ("stolen_back", s.StolenBackCards);
    }

    /// <summary>The combat's named random streams, for the monotone and shared-object checks.</summary>
    public static IEnumerable<(string Name, CountingRandom? Rng)> Streams(CombatState s)
    {
        yield return ("Shuffle", s.ShuffleRng);
        yield return ("CombatTargets", s.TargetRng);
        yield return ("CombatCardSelection", s.CardSelectionRng);
        yield return ("CombatCardGeneration", s.CardGenerationRng);
        yield return ("CombatPotionGeneration", s.PotionGenerationRng);
        yield return ("CombatEnergyCosts", s.EnergyCostRng);
        yield return ("CombatOrbGeneration", s.OrbGenerationRng);
        yield return ("Niche", s.NicheHpRng);
        yield return ("MonsterAi", s.AiRng as CountingRandom);
    }

    private static readonly HashSet<int> CardIds = [.. Cards.All.ToArray().Select(c => c.Id)];

    private static bool KnownCard(int id) => CardIds.Contains(id);

    /// <summary>Keeps the previous state's figures to judge the next one.</summary>
    public sealed class Tracker
    {
        private Dictionary<string, int>? _rng;
        private int _turn;
        private int _played;
        private int _drawn;
        private int _generated;

        /// <summary>Single-state checks plus the ones that need the step before.</summary>
        public List<Violation> Step(CombatState s)
        {
            var v = Check(s);
            var rng = Streams(s).Where(x => x.Rng is not null).ToDictionary(x => x.Name, x => x.Rng!.CallCount);
            if (_rng is not null)
            {
                foreach (var (name, was) in _rng)
                {
                    if (!rng.TryGetValue(name, out int now))
                    {
                        v.Add(new Violation("rng_monotone", $"{name} stream disappeared (was at {was})"));
                    }
                    else if (now < was)
                    {
                        v.Add(new Violation("rng_monotone", $"{name} {was} -> {now}"));
                    }
                }

                if (s.Turn < _turn || s.Turn > _turn + 1)
                {
                    v.Add(new Violation("turn_monotone", $"turn {_turn} -> {s.Turn}"));
                }

                if (s.CardsPlayedThisCombat < _played || s.CardsDrawnThisCombat < _drawn || s.CardsGeneratedThisCombat < _generated)
                {
                    v.Add(new Violation(
                        "counter_monotone",
                        $"played {_played}->{s.CardsPlayedThisCombat} drawn {_drawn}->{s.CardsDrawnThisCombat} generated {_generated}->{s.CardsGeneratedThisCombat}"));
                }
            }

            _rng = rng;
            _turn = s.Turn;
            _played = s.CardsPlayedThisCombat;
            _drawn = s.CardsDrawnThisCombat;
            _generated = s.CardsGeneratedThisCombat;
            return v;
        }
    }
}
