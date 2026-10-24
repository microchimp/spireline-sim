// game: AttackCommand, the game-shaped attack over the old state (docs/rebuild/MIGRATION_F08.md section 3 item 1).
// Round R2: written and tested, called from no old site yet (R4 reroutes the callers). Game code is cited by class and
// method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

/// <summary>
/// The F06 bridge of an attack: the old engine's own BeforeAttack / AfterAttack points for this attack, which
/// <see cref="AttackCmd.Execute" /> runs once each at the game's places while F06 is off (and never with F06 on, when
/// it dispatches the hooks and F06's switch gates the old points off). A rerouted caller (R4) passes its old point here
/// instead of running it: for a monster's attack the post-loop bodies of <c>EnemyAI.DealAttack</c> (Suck's
/// <c>TriggerSuck</c> with the hits that landed, Painful Stabs' wounds), for Osty's
/// <c>RelicEffects.ApplyAfterOstyAttack</c>. A card's attack passes none: its old points
/// (<c>CombatEngine.ResolvePlayRepetition</c> [effect]/[after], <c>QueueAttackPlayLifecycleEffects</c>) stay at the play
/// site, once per play, until F06 migrates.
/// </summary>
public sealed record AttackBridge(Action<AttackCmd>? BeforeAttack = null, Action<AttackCmd>? AfterAttack = null);

/// <summary>
/// game: AttackCommand (the builder's state and its Execute). Targeting is one of: a single target
/// (<see cref="SingleTarget" />), all opponents (<see cref="AllOpponents" />), random opponents
/// (<see cref="RandomPick" />, the old engine's target pick, with or without duplicates).
/// </summary>
public sealed class AttackCmd(OldCreature attacker, decimal damagePerHit, int hitCount = 1)
{
    public OldCreature Attacker { get; } = attacker;
    public decimal DamagePerHit { get; } = damagePerHit;
    public int HitCount { get; } = hitCount;

    /// <summary>game: the CalculatedDamageVar's Calculate(singleTarget), when the amount depends on the target.</summary>
    public Func<OldCreature?, decimal>? Calculate { get; init; }

    public OldCreature? SingleTarget { get; init; }
    public bool AllOpponents { get; init; }

    /// <summary>Random targeting: the pick among the valid targets (the game's Rng.CombatTargets.NextItem; the caller
    /// passes the old engine's).</summary>
    public Func<IReadOnlyList<OldCreature>, OldCreature>? RandomPick { get; init; }

    public bool RandomAllowsDuplicates { get; init; } = true;
    public ValueProp Props { get; init; } = ValueProps.PoweredAttack;
    public CardHandle? CardSource { get; init; }
    public AttackBridge? Bridge { get; init; }

    /// <summary>The results of each hit's Damage call, in order (game: AttackCommand's results).</summary>
    public List<IReadOnlyList<DamageResult>> Results { get; } = [];

    /// <summary>The hits that ran (one Damage call each).</summary>
    public int Hits => Results.Count;

    /// <summary>
    /// game: AttackCommand.Execute. The combat over or ending: return at once (no hook). Attacker dead: nothing.
    /// BeforeAttack; the hit count (ModifyAttackHitCount); per hit: the attacker dead stops it; the living possible
    /// targets, none left in a live combat stops it; the target (single when one is valid, all valid ones in one call,
    /// or the random pick); one <see cref="CreatureCmd.Damage" />. Then AfterAttack. BeforeAttack, the hit count and
    /// AfterAttack are F06's: with F06 off the count is the one given and the <see cref="Bridge" />'s old points run.
    /// </summary>
    public AttackCmd Execute(KernelState k, ChoiceCtx? ctx = null)
    {
        var w = k.World ?? throw new InvalidOperationException("AttackCmd needs a switched-on kernel state");
        var s = w.Combat ?? throw new InvalidOperationException("AttackCmd outside a combat");
        // game: CombatManager.IsOverOrEnding in a live combat (the old engine's flag; every old combat is live).
        if (CombatEngine.IsOverOrEnding(s))
            return this;
        if (OldCreatures.IsDead(s, Attacker))
            return this;
        if (SingleTarget is null && !AllOpponents && RandomPick is null)
            throw new InvalidOperationException("No targets set.");
        bool f06 = k.On(Switch.F06);
        var self = k.Commands.Begin(this);
        try
        {
            if (f06)
                CreatureCmd.Settle(k, Hooks.BeforeAttack(k, new BeforeAttackArgs(self)),
                    st => Hooks.BeforeAttackResume(k, new BeforeAttackArgs(self), st));
            else
                Bridge?.BeforeAttack?.Invoke(this);
            int count = f06 ? Hooks.ModifyAttackHitCount(k, new ModifyAttackHitCountArgs(self, HitCount)) : HitCount;
            for (int i = 0; i < count; i++)
            {
                if (OldCreatures.IsDead(s, Attacker))
                    break;
                var valid = PossibleTargets(w).Where(c => !OldCreatures.IsDead(s, c)).ToList();
                if (valid.Count == 0)
                    break;
                OldCreature? single;
                if (RandomPick is null)
                {
                    single = valid.Count == 1 ? valid[0] : null;
                }
                else
                {
                    if (!RandomAllowsDuplicates)
                    {
                        valid = valid.Where(c => Results.SelectMany(r => r).All(r => r.Receiver != c)).ToList();
                        if (valid.Count == 0)
                            throw new InvalidOperationException("No valid targets for attack with duplicates disallowed.");
                    }

                    single = RandomPick(valid);
                }

                decimal amount = Calculate is { } calc ? calc(single) : DamagePerHit;
                Results.Add(CreatureCmd.Damage(k, single is { } t ? [t] : valid, amount, Props, Attacker, CardSource,
                    ctx));
            }

            // game: CombatManager.History.CreatureAttacked: the old state keeps no combat history.
            if (f06)
            {
                var after = new AfterAttackArgs(ctx ?? new ChoiceCtx(), self);
                CreatureCmd.Settle(k, Hooks.AfterAttack(k, after), st => Hooks.AfterAttackResume(k, after, st));
            }
            else
            {
                Bridge?.AfterAttack?.Invoke(this);
            }
        }
        finally
        {
            k.Commands.End(self);
        }

        return this;
    }

    /// <summary>game: AttackCommand.GetPossibleTargets.</summary>
    private List<OldCreature> PossibleTargets(OldWorld w) =>
        SingleTarget is { } t ? [t] : OldCreatures.Opponents(w, Attacker);
}
