namespace Sts2Emulator.Solver;

using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;

/// <summary>
/// The tight utility bound (FIGHT_SOLVER.md section 19). L0 is 0 for a death, else 0.5 + 0.5 * min(1, (HP + 10 * potions)
/// / max HP). With no heal, no max-HP change and no potion gain able to happen in the rest of the combat, HP and the potions
/// held only fall and max HP stays, so every value under a position is at most L0 of its own HP (plus what the winning step
/// heals before the terminal is scored) and potions. The sites that could break it are the scanned registry
/// (<c>tools/solver/scan_utility_gains.py</c>, <c>utility_gains.json</c>); <see cref="UtilityGains" />'s names are written
/// from it into <c>UtilityGains.g.cs</c>. A holder is reachable in this combat when it is present anywhere: a card in any
/// pile, hold or screen candidate, a relic, a potion in a slot, a buff on the player or any enemy, an enemy.
/// </summary>
internal static partial class UtilityGains
{
    // Resolved on first use, in a class of their own: the names (UtilityGains.g.cs) are this class's static fields in another
    // file, and the order of static initialisers across a partial class's files is not defined.
    static class R
    {
        internal static readonly HashSet<int> CardIds = [.. CardNames.Select(Cards.FindId).Where(i => i is not null).Select(i => i!.Value)];
        internal static readonly (int Id, string Name)[] RelicIds = [.. RelicNames.Select(n => (Relics.FindId(n) ?? -1, n))];
        internal static readonly (int Id, string Name)[] PotionIds = [.. PotionNames.Select(n => (Potions.FindId(n) ?? -1, n))];
        internal static readonly (BuffId Id, string Name)[] BuffIds = [.. BuffNames.Select(n => (Enum.TryParse<BuffId>(n, out var b) ? b : (BuffId)(-1), n))];
        internal static readonly (int Id, string Name)[] EnemyIds = [.. EnemyNames.Select(n => (EnemyDefId(n), n))];
        internal static readonly (int Id, int Constant)[] EndBeforeTerminal = [.. EndHeals.Where(e => e.BeforeTerminal).Select(e => (Relics.FindId(e.Relic) ?? -1, e.Constant))];
    }

    /// <summary>Holder names that name nothing (UtilityGainTests requires none).</summary>
    internal static IEnumerable<string> Unresolved() =>
        CardNames.Where(n => Cards.FindId(n) is null).Select(n => "card " + n)
            .Concat(R.RelicIds.Where(r => r.Id < 0).Select(r => "relic " + r.Name))
            .Concat(R.PotionIds.Where(p => p.Id < 0).Select(p => "potion " + p.Name))
            .Concat(R.BuffIds.Where(b => (int)b.Id < 0).Select(b => "buff " + b.Name))
            .Concat(R.EnemyIds.Where(e => e.Id < 0).Select(e => "enemy " + e.Name))
            .Concat(EndHeals.Where(e => Relics.FindId(e.Relic) is null).Select(e => "end relic " + e.Relic));

    static int EnemyDefId(string name)
    {
        for (int id = 1; id < 1024; id++)
        {
            try
            {
                if (Enemies.Get(id).Name == name)
                {
                    return id;
                }
            }
            catch (ArgumentException)
            {
                // no enemy with this id
            }
        }

        return -1;
    }

    /// <summary>The first gain holder present in the combat (<c>kind:Name</c>), or null when no utility gain is reachable.</summary>
    internal static string? Off(CombatState s)
    {
        if (R.CardIds.Count > 0)
        {
            foreach (var c in StateKey.AllCards(s))
            {
                if (R.CardIds.Contains(Math.Abs(c.DefId)))
                {
                    return "card:" + Cards.Get(Math.Abs(c.DefId)).Name;
                }
            }

            if (s.PendingSelection is { } p)
            {
                foreach (int id in p.GeneratedCandidates)
                {
                    if (R.CardIds.Contains(Math.Abs(id)))
                    {
                        return "card:" + Cards.Get(Math.Abs(id)).Name;
                    }
                }
            }
        }

        foreach (var (id, name) in R.RelicIds)
        {
            if (s.Relics.Exists(r => r.DefId == id))
            {
                return "relic:" + name;
            }
        }

        foreach (var (id, name) in R.PotionIds)
        {
            if (Array.IndexOf(s.PotionSlots, id) >= 0)
            {
                return "potion:" + name;
            }
        }

        foreach (var (id, name) in R.BuffIds)
        {
            if (s.PlayerBuffs.Exists(b => b.Id == id) || s.Enemies.Exists(e => e.Buffs.Exists(b => b.Id == id)))
            {
                return "buff:" + name;
            }
        }

        foreach (var (id, name) in R.EnemyIds)
        {
            if (s.Enemies.Exists(e => e.DefId == id))
            {
                return "enemy:" + name;
            }
        }

        return null;
    }

    /// <summary>What the winning step heals before the terminal is scored, at most: the combat's-end heals applied then.</summary>
    internal static int EndConstant(CombatState s)
    {
        int c = 0;
        foreach (var (id, constant) in R.EndBeforeTerminal)
        {
            if (s.Relics.Exists(r => r.DefId == id))
            {
                c += constant;
            }
        }

        return c;
    }

    /// <summary>
    /// L0 of the position's HP plus the end constant with <paramref name="potionsDelta" /> potions more held: the tight bound
    /// when no gain is reachable (the caller checks <see cref="Off" />).
    /// </summary>
    internal static Rational Tight(CombatState s, int potionsDelta = 0)
    {
        if (s.PlayerHp <= 0)
        {
            return Utility.Loss;
        }

        int max = Math.Max(1, s.PlayerMaxHp);
        long score = Math.Min((long)s.PlayerHp + EndConstant(s) + 10L * Math.Max(0, Utility.Potions(s) + potionsDelta), max);
        return new Rational(max + score, 2L * max);
    }

    /// <summary>A sound upper bound on every value under the position: <see cref="Tight" /> when no gain is reachable, else L0's maximum.</summary>
    internal static Rational Upper(CombatState s) => Off(s) is null ? Tight(s) : Utility.Max;
}
