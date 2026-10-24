// The damage command's plain results and the per-combat log their refs index (MIGRATION_F08.md section 3 items 1-2).
// Game code is cited by class and method only.
namespace Sts2Emulator.Core2;

/// <summary>The game's ValueProp bits (game: ValueProp) as <see cref="ValueProp.Bits" /> carries them.</summary>
public static class ValueProps
{
    public const int Unblockable = 2;
    public const int Unpowered = 4;
    public const int Move = 8;
    public const int SkipHurtAnim = 16;

    /// <summary>A card's or a monster's attack: Move, powered.</summary>
    public static readonly ValueProp PoweredAttack = new(Move);

    public static bool Has(this ValueProp p, int bit) => (p.Bits & bit) != 0;

    /// <summary>game: ValuePropExtensions.IsPoweredAttack.</summary>
    public static bool IsPoweredAttack(this ValueProp p) => p.Has(Move) && !p.Has(Unpowered);
}

/// <summary>
/// One damage event's result (game: DamageResult): the receiver, the props, and the int amounts the game keeps (its one
/// cast is Creature.LoseHpInternal's; the blocked amount is cast where CreatureCmd.Damage stores it).
/// </summary>
public sealed record DamageResult(OldCreature Receiver, ValueProp Props)
{
    public int BlockedDamage { get; set; }
    public int UnblockedDamage { get; init; }
    public int OverkillDamage { get; init; }
    public bool WasTargetKilled { get; init; }
    public bool WasBlockBroken { get; set; }
    public bool WasFullyBlocked { get; set; }

    /// <summary>The ref a dispatcher's arguments carry (AfterDamageGiven, AfterDamageReceived).</summary>
    public DamageResultRef Ref { get; internal set; }
}

/// <summary>
/// The commands' per-combat log, held by <see cref="KernelState.Commands" />: the damage results by ref (a handler reads
/// one through <see cref="Result" />), the attacks executing, and the kill hook point's call count. Copied with the
/// state (results re-pointed at the copy's enemies by roster index); attacks are live only during their Execute.
/// </summary>
public sealed class CommandLog
{
    private readonly List<DamageResult> _results = [];
    private readonly Dictionary<int, AttackCmd> _attacks = [];
    private int _nextAttack;

    /// <summary>How many times Damage reached <see cref="CreatureCmd.Kill" /> (one per Damage call that ran).</summary>
    public int KillCalls { get; internal set; }

    /// <summary>Test seam until R3a: what <see cref="CreatureCmd.Kill" /> runs instead of its stub. Not copied.</summary>
    internal Action<KernelState, IReadOnlyList<OldCreature>>? KillStandIn { get; set; }

    internal DamageResultRef Add(DamageResult r)
    {
        r.Ref = new DamageResultRef(_results.Count);
        _results.Add(r);
        return r.Ref;
    }

    /// <summary>The result a <see cref="DamageResultRef" /> names.</summary>
    public DamageResult Result(DamageResultRef r) => _results[r.Id];

    internal AttackRef Begin(AttackCmd a)
    {
        var r = new AttackRef(++_nextAttack);
        _attacks[r.Id] = a;
        return r;
    }

    internal void End(AttackRef r) => _attacks.Remove(r.Id);

    /// <summary>The executing attack an <see cref="AttackRef" /> names.</summary>
    public AttackCmd Attack(AttackRef r) => _attacks[r.Id];

    internal CommandLog CopyFor(Core.CombatState? from, Core.CombatState? to)
    {
        var c = new CommandLog { KillCalls = KillCalls, _nextAttack = _nextAttack };
        foreach (var r in _results)
        {
            var who = r.Receiver;
            if (who.Enemy is { } e && from is not null && to is not null)
            {
                int at = from.Enemies.FindIndex(x => ReferenceEquals(x, e));
                if (at >= 0 && at < to.Enemies.Count)
                    who = OldCreature.Of(to.Enemies[at]);
            }

            c._results.Add(r with { Receiver = who });
        }

        return c;
    }
}

/// <summary>The game's CardPreviewMode values as <see cref="PreviewMode.Kind" /> carries them.</summary>
public static class PreviewModes
{
    public static readonly PreviewMode None = new(0);
    public static readonly PreviewMode Normal = new(1);
    public static readonly PreviewMode Upgrade = new(2);
    public static readonly PreviewMode MultiCreatureTargeting = new(3);
}
