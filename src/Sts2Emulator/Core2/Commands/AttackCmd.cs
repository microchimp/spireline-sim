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
/// (<see cref="RandomlyTargeted" />: one draw of the CombatTargets stream per hit, with or without duplicates).
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

    /// <summary>
    /// game: IsRandomlyTargeted. Each hit picks its one target among the living opponents, after the duplicates filter
    /// when duplicates are disallowed, by drawing the attacker's player's CombatTargets stream (the pet owner's for
    /// Osty; the old state's <c>TargetRng</c>) exactly once per hit: <c>Next(count)</c>, drawn even over one candidate
    /// (the game's Rng.NextItem). An enemy attacker has no such stream and throws.
    /// </summary>
    public bool RandomlyTargeted { get; init; }

    /// <summary>Test seam: replaces the stream draw of <see cref="RandomlyTargeted" /> (and implies random targeting).</summary>
    public Func<IReadOnlyList<OldCreature>, OldCreature>? RandomPick { get; init; }

    /// <summary>
    /// game: AttackCommand.BeforeDamage, the per-hit callback run after the target pick and before the damage amount is
    /// calculated and Damage runs. The game's seven users (Dagger Spray, Fiend Fire, Hyperbeam, Sovereign Blade,
    /// Sweeping Beam, Calcified Cultist, Damp Cultist) only play effects and sounds; none changes simulation state.
    /// </summary>
    public Action? BeforeDamage { get; init; }

    public bool RandomAllowsDuplicates { get; init; } = true;
    public ValueProp Props { get; init; } = ValueProps.PoweredAttack;
    public CardHandle? CardSource { get; init; }
    public AttackBridge? Bridge { get; init; }

    /// <summary>The results of each hit's Damage call, in order (game: AttackCommand's results).</summary>
    public List<IReadOnlyList<DamageResult>> Results { get; } = [];

    /// <summary>The hits that ran (one Damage call each).</summary>
    public int Hits => Results.Count;

    // construct: game AttackCommand.Execute
    //   -        Hook.BeforeAttack
    //   -        Hook.ModifyAttackHitCount
    //   for/if   BeforeDamage
    //   for      CreatureCmd.Damage
    //   -        Hook.AfterAttack
    // end construct
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
        if (SingleTarget is null && !AllOpponents && !RandomlyTargeted && RandomPick is null)
            throw new InvalidOperationException("No targets set.");
        bool f06 = k.On(Switch.F06);
        var self = k.Commands.Begin(this);
        try
        {
            if (f06)
                CreatureCmd.Settle(k, Hooks.BeforeAttack(k, new BeforeAttackArgs(self)),
                    st => Hooks.BeforeAttackResume(k, new BeforeAttackArgs(self), st));
            else
            {
                k.Shared.BridgePoint?.Invoke(HookId.BeforeAttack);
                Bridge?.BeforeAttack?.Invoke(this);
            }
            int count = f06 ? Hooks.ModifyAttackHitCount(k, new ModifyAttackHitCountArgs(self, HitCount)) : HitCount;
            for (int i = 0; i < count; i++)
            {
                if (OldCreatures.IsDead(s, Attacker))
                    break;
                // game: `GetPossibleTargets().Where(c => c.IsAlive)`; a target a nested kill or escape took out of the combat
                // (a single target no roster lists any more) is not one either.
                var valid = PossibleTargets(w).Where(c => !OldCreatures.IsDead(s, c) && OldCreatures.InCombat(w, c)).ToList();
                if (valid.Count == 0)
                    break;
                OldCreature? single;
                if (!RandomlyTargeted && RandomPick is null)
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

                    single = RandomPick?.Invoke(valid) ?? DrawTarget(s, valid);
                }

                BeforeDamage?.Invoke();
                decimal amount = Calculate is { } calc ? calc(single) : DamagePerHit;
                Results.Add(CreatureCmd.Damage(k, single is { } t ? [t] : valid, amount, Props, Attacker, CardSource,
                    ctx));
            }

            // game: CombatManager.History.CreatureAttacked, one entry per command after its last hit, before AfterAttack.
            OldCreatures.NoteAttacked(w, Attacker);
            if (f06)
            {
                var after = new AfterAttackArgs(ctx ?? new ChoiceCtx(), self);
                CreatureCmd.Settle(k, Hooks.AfterAttack(k, after), st => Hooks.AfterAttackResume(k, after, st));
            }
            else
            {
                // game: AttackCommand's end runs AfterAttack whether or not its target died; with F06 off that point is
                // the bridge.
                k.Shared.BridgePoint?.Invoke(HookId.AfterAttack);
                Bridge?.AfterAttack?.Invoke(this);
            }
        }
        finally
        {
            k.Commands.End(self);
        }

        return this;
    }

    /// <summary>game: <c>(Attacker.Player ?? Attacker.PetOwner).RunState.Rng.CombatTargets.NextItem(valid)</c>.</summary>
    private OldCreature DrawTarget(CombatState s, List<OldCreature> valid)
    {
        if (Attacker.Kind == CreatureKind.Enemy)
            throw new InvalidOperationException("Random targeting draws the player's CombatTargets stream: the attacker must be the player or its pet.");
        var rng = s.TargetRng ?? throw new InvalidOperationException("No CombatTargets stream in this combat.");
        return valid[rng.Next(valid.Count)];
    }

    /// <summary>game: AttackCommand.GetPossibleTargets.</summary>
    private List<OldCreature> PossibleTargets(OldWorld w) =>
        SingleTarget is { } t ? [t] : OldCreatures.Opponents(w, Attacker);
}
