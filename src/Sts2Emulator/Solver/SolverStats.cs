namespace Sts2Emulator.Solver;

using System.Diagnostics;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;

/// <summary>
/// The solver's debug switches (the baseline round, FIGHT_SOLVER.md section 16). Static, off by default, compiled into the
/// native library and the JIT builds alike; neither changes a value, a key or an engine byte, only what is measured.
/// </summary>
public static class SolverDebug
{
    /// <summary>The per-phase stopwatch counters (<see cref="SolverProf" />).</summary>
    public static bool PhaseTimers
    {
        get => SolverProf.On;
        set => SolverProf.On = value;
    }

    /// <summary>The merge audit: a second key per expanded end-turn move with the candidate dead fields masked.</summary>
    public static bool MergeAudit;

    /// <summary>Bit 0 the phase timers, bit 1 the merge audit (the native switch's flags).</summary>
    public static int Flags
    {
        get => (PhaseTimers ? 1 : 0) | (MergeAudit ? 2 : 0);
        set
        {
            PhaseTimers = (value & 1) != 0;
            MergeAudit = (value & 2) != 0;
        }
    }
}

/// <summary>The phases the stopwatch counters split a solve into. Each phase's time is exclusive of the phases nested in it.</summary>
public enum Phase
{
    /// <summary>Everything not in another phase: the recursion, move lists, dictionaries, allocation.</summary>
    Remainder = 0,

    /// <summary>The engine step (<c>CombatEngine.Step</c> or the kernel's), less its clones and the chance hook's calls.</summary>
    Step,

    /// <summary>Whole-state clones (<c>StateCloning.Clone</c>): the solver's per-run copy and the engine's pre-step record.</summary>
    Clone,

    /// <summary>The state key's canonical bytes (<c>StateKey.Encode</c>).</summary>
    KeyEncode,

    /// <summary>The key's SHA-256 (<c>StateKey.HashOf</c>).</summary>
    KeyHash,

    /// <summary><c>Rational</c> arithmetic and comparison, wherever it runs.</summary>
    Rational,

    /// <summary>Transposition table lookups and inserts.</summary>
    Table,

    /// <summary>Chance expansion: <c>Successors.Expand</c>'s own work, <c>DrawSteer</c>'s layout, the odds and the chance hook's site, tie and width checks.</summary>
    Chance,

    /// <summary>Bound checks: the max node's value, bound and exactness rule (<c>NodeValue.Max</c>) and the budget check.</summary>
    Bound,

    Count,
}

/// <summary>
/// Exclusive per-phase time on a stack: entering a phase charges the time since the last mark to the phase below it, leaving
/// charges it to the phase left. Per thread; off by default (one static read per call when off). A scope abandoned by an
/// exception is closed by the enclosing scope's exit, which pops to its own depth.
/// </summary>
public static class SolverProf
{
    public static volatile bool On;

    [ThreadStatic]
    static long[]? t_ticks;

    [ThreadStatic]
    static Phase[]? t_stack;

    [ThreadStatic]
    static int t_depth;

    [ThreadStatic]
    static long t_mark;

    /// <summary>Enters <paramref name="p" />; returns the depth to pass to <see cref="Exit" /> (-1 when off).</summary>
    public static int Enter(Phase p)
    {
        if (!On)
        {
            return -1;
        }

        var ticks = t_ticks ??= new long[(int)Phase.Count];
        var stack = t_stack ??= new Phase[64];
        long now = Stopwatch.GetTimestamp();
        int d = t_depth;
        if (d > 0)
        {
            ticks[(int)stack[d - 1]] += now - t_mark;
        }

        if (d == stack.Length)
        {
            Array.Resize(ref t_stack, d * 2);
            stack = t_stack!;
        }

        stack[d] = p;
        t_depth = d + 1;
        t_mark = now;
        return d;
    }

    /// <summary>Leaves the phase entered at <paramref name="depth" /> (and any left open above it by an exception).</summary>
    public static void Exit(int depth)
    {
        if (depth < 0 || t_stack is null || t_depth <= depth)
        {
            return;
        }

        long now = Stopwatch.GetTimestamp();
        t_ticks![(int)t_stack[t_depth - 1]] += now - t_mark;
        t_depth = depth;
        t_mark = now;
    }

    /// <summary>The thread's running totals in ticks (a copy).</summary>
    public static long[] Ticks() => t_ticks is { } t ? (long[])t.Clone() : new long[(int)Phase.Count];

    public static double Seconds(long ticks) => ticks / (double)Stopwatch.Frequency;
}

/// <summary>One horizon of a deepening solve: where it ended, at what cumulative cost, and whether it was exact.</summary>
public sealed record HorizonMark(int Horizon, string End, long Nodes, long Steps, double Seconds, bool Exact, double EstimateLog10);

/// <summary>Move kinds for the per-kind counters.</summary>
public enum MoveKind
{
    Play = 0,
    EndTurn = 1,
    Screen = 2,
    Potion = 3,
}

/// <summary>
/// The merge audit (optimisation review change 2): at each expanded end-turn move, the pre-step key with the candidate dead
/// fields masked. Distinct masked keys against distinct full keys is the share of end-turn expansions a dead-field
/// transposition could merge. Two masks: every candidate (block, energy, per-turn counters, per-combat counters with no
/// reader in the combat), and the same without block (block is dead only when nothing hits this enemy turn and no block
/// reader or keeper is present, which this audit does not check: the first mask is the upper bound, the second the floor).
/// Readers are looked for in every pile at the node; a reader generated later in the fight is not seen.
/// </summary>
public static class MergeAudit
{
    /// <summary>Per-combat counters and the cards that read them (CardEffects and CombatEngine read sites).</summary>
    static readonly (string Field, string[] Readers)[] CombatCounters =
    [
        ("CardsPlayedThisCombat", ["GoldAxe"]),
        ("CardsDrawnThisCombat", ["Murder"]),
        ("LightningOrbsChanneledThisCombat", ["Voltaic"]),
        ("EtherealCardPlaysThisCombat", ["BansheesCry", "PullFromBelow"]),
        ("CardsGeneratedThisCombat", ["Supermassive"]),
    ];

    static readonly Dictionary<string, int[]> ReaderIds = CombatCounters.ToDictionary(
        x => x.Field, x => x.Readers.Select(Cards.FindId).Where(i => i is not null).Select(i => i!.Value).ToArray());

    /// <summary>The masked position: a copy with the candidate dead fields set to their turn-start value.</summary>
    public static CombatState Masked(CombatState s, bool maskBlock)
    {
        var present = new HashSet<int>();
        foreach (var c in StateKey.AllCards(s))
        {
            present.Add(Math.Abs(c.DefId));
        }

        bool Read(string field) => ReaderIds[field].Any(present.Contains);
        var m = s.Clone();
        if (maskBlock)
        {
            m.PlayerBlock = 0;
        }

        m.Energy = 0;
        m.StatusCardsDrawnThisTurn = 0;
        m.EnergySpentThisTurn = 0;
        m.EtherealExhaustsThisTurn = 0;
        m.UnblockedDamageThisTurn = 0;
        m.RainbowRingPaidThisTurn = false;
        m.CardsPlayedThisTurn = 0;
        m.CardPlaysStartedThisTurn = 0;
        m.CardPlaySeriesStartedThisTurn = 0;
        m.AttackCardsPlayedThisTurn = 0;
        m.AttackOrSkillCardsPlayedThisTurn = 0;
        m.CardPlaysThisTurn = 0;
        m.LastAttackOrSkillThisTurn = null;
        m.ShivsPlayedThisTurn = 0;
        m.BlockGainsThisTurn = 0;
        m.PlayerHpLostThisTurn = 0;
        m.CardsExhaustedThisTurn = 0;
        m.SkillCardsPlayedThisTurn = 0;
        m.BoundCardsThisTurn = 0;
        m.BoundCardPlayedThisTurn = false;
        m.MusicBoxUsedThisTurn = false;
        m.ExtraCardsDrawnThisTurn = 0;
        m.DoomAppliedThisTurn = false;
        m.OstyAttacksThisTurn = 0;
        m.StarsGainedThisTurn = 0;
        m.VoidFormCardsPlayedThisTurn = 0;
        m.TenderCardsPlayedThisTurn = 0;
        if (!Read("CardsPlayedThisCombat")) m.CardsPlayedThisCombat = 0;
        if (!Read("CardsDrawnThisCombat")) m.CardsDrawnThisCombat = 0;
        if (!Read("LightningOrbsChanneledThisCombat")) m.LightningOrbsChanneledThisCombat = 0;
        if (!Read("EtherealCardPlaysThisCombat")) m.EtherealCardPlaysThisCombat = 0;
        if (!Read("CardsGeneratedThisCombat")) m.CardsGeneratedThisCombat = 0;
        return m;
    }

    /// <summary>Records one expanded end-turn move's three keys (full, masked, masked but block).</summary>
    public static void Record(CombatState s, SolveCounters c)
    {
        try
        {
            c.AuditFull.Add(StateKey.Hash(s, KeyMode.Expected));
            c.AuditMasked.Add(StateKey.Hash(Masked(s, maskBlock: true), KeyMode.Expected));
            c.AuditMaskedKeepBlock.Add(StateKey.Hash(Masked(s, maskBlock: false), KeyMode.Expected));
        }
        catch (StateKeyRefusedException)
        {
            // The node's own key refused first; nothing to audit.
        }
    }
}

/// <summary>The stats object (EmuDump's <c>stats</c>, the native <c>Sts2_SolverStats</c>): every counter of the baseline round.</summary>
public static class SolverStatsJson
{
    static readonly string[] KindNames = ["play", "end_turn", "screen", "potion"];

    static readonly string[] PhaseNames = ["remainder", "step", "clone", "key_encode", "key_hash", "rational", "table", "chance", "bound"];

    public static JsonObject Of(ExpectedResult r)
    {
        var c = r.Counters;
        var moves = new JsonObject();
        var runs = new JsonObject();
        for (int k = 0; k < KindNames.Length; k++)
        {
            moves[KindNames[k]] = c.MovesByKind[k];
            runs[KindNames[k]] = c.RunsByKind[k];
        }

        var horizons = new JsonArray();
        foreach (var h in c.Horizons)
        {
            horizons.Add((JsonNode)new JsonObject
            {
                ["horizon"] = h.Horizon, ["end"] = h.End, ["nodes"] = h.Nodes, ["steps"] = h.Steps, ["seconds"] = h.Seconds, ["exact"] = h.Exact,
                ["estimate_log10"] = double.IsFinite(h.EstimateLog10) ? h.EstimateLog10 : null,
            });
        }

        static JsonObject MacroOffJson(SolveCounters c)
        {
            var o = new JsonObject();
            foreach (var (k, v) in c.MacroOff.OrderBy(x => x.Key, StringComparer.Ordinal))
            {
                o[k] = v;
            }

            return o;
        }

        var refusals = new JsonObject();
        foreach (var (k, v) in c.Refusals.OrderBy(x => x.Key, StringComparer.Ordinal))
        {
            refusals[k] = v;
        }

        var o = new JsonObject
        {
            ["nodes"] = c.Nodes,
            ["engine_steps"] = c.Steps,
            ["moves_by_kind"] = moves,
            ["runs_by_kind"] = runs,
            ["chance_levels"] = c.ChanceLevels,
            ["chance_level_max"] = c.ChanceLevelMax,
            ["end_turn_leaves"] = c.EndTurnLeaves,
            ["table_hits"] = c.TableHits,
            ["table_hits_exact"] = c.TableHitsExact,
            ["table_hits_same_turns"] = c.TableHitsSameTurns,
            ["table_misses"] = c.TableMisses,
            ["merged_outcomes"] = c.Merged,
            ["terminal_leaves"] = c.Leaves,
            ["horizon_leaves"] = c.HorizonLeaves,
            ["wide_roll_leaves"] = c.WideRollLeaves,
            ["macro_nodes"] = c.MacroNodes,
            ["macro_draws_avoided"] = c.MacroDrawsAvoided,
            ["draw_nodes_single"] = c.DrawNodesSingle,
            ["macro_off"] = MacroOffJson(c),
            ["macro_root"] = c.MacroRoot,
            ["screens"] = c.Screens,
            ["refusals"] = refusals,
            ["wall_seconds"] = r.Seconds,
            ["horizons"] = horizons,
        };
        if (c.PhaseTicks is { } t)
        {
            var phases = new JsonObject();
            long sum = 0;
            for (int i = 0; i < PhaseNames.Length; i++)
            {
                phases[PhaseNames[i]] = SolverProf.Seconds(t[i]);
                sum += t[i];
            }

            o["phase_seconds"] = phases;
            o["phase_total_seconds"] = SolverProf.Seconds(sum);
        }

        if (c.AuditFull.Count > 0)
        {
            o["merge_audit"] = new JsonObject
            {
                ["end_turn_expansions"] = c.AuditExpansions,
                ["distinct_full"] = c.AuditFull.Count,
                ["distinct_masked"] = c.AuditMasked.Count,
                ["distinct_masked_keep_block"] = c.AuditMaskedKeepBlock.Count,
            };
        }

        return o;
    }
}
