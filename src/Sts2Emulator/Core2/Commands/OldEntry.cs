// The old engine's damage, kill and escape entry points under the F08+F10 switch (docs/rebuild/MIGRATION_F08.md section
// 4, round R4). Each old site gates with `if (Kernel.If(Switch.F08, state) is { } k)` and calls one of these with the old
// call's arguments; each runs the game-shaped command (AttackCmd, CreatureCmd.Damage, Kill, Escape) and hands back the
// number the old caller reads. The old site's inline model bodies do not run here: their handlers do (the batches).
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

public static class OldEntry
{
    /// <summary>The old call's flags as the game's props: powered is a move (an attack), unpowered otherwise.</summary>
    public static ValueProp Props(bool powered, bool unblockable) =>
        new((powered ? ValueProps.Move : ValueProps.Unpowered) | (unblockable ? ValueProps.Unblockable : 0));

    /// <summary>Damage the old engine called unblockable and unpowered (self-damage, HP loss).</summary>
    public static readonly ValueProp HpLoss = new(ValueProps.Unblockable | ValueProps.Unpowered);

    /// <summary>Unpowered, blockable damage (Thorns-like retaliation, a status's burn).</summary>
    public static readonly ValueProp Unpowered = new(ValueProps.Unpowered);

    /// <summary>The card being played (the game's Play pile), as a damage's card source.</summary>
    public static CardHandle? InFlight(KernelState k) => k.World?.InFlight is { } c ? new CardHandle(c.Uid) : null;

    /// <summary>
    /// One hit on an enemy (old CardEffects.DealDamageToEnemy / LandDamageOnEnemy / DealUnpoweredDamageToEnemy):
    /// CreatureCmd.Damage, the base amount folded by the hooks. <paramref name="dealer" /> and <paramref name="cardSource" />
    /// are the caller's: the player as dealer for a card's, relic's or player power's hit, null for the damage-over-time
    /// powers whose game call passes no dealer (Poison, Strangle, Haunt, Demise: no AfterDamageGiven for anyone, and no
    /// dealer-dead skip); the card in flight (<see cref="InFlight" />) as source for a card's own hit, null for a power,
    /// relic, orb or potion (each of the game's passes null). Returns the HP lost; <paramref name="total" /> is blocked plus
    /// unblocked (the game's TotalDamage).
    /// </summary>
    public static int HitEnemy(KernelState k, EnemyState target, int amount, ValueProp props, OldCreature? dealer,
        CardHandle? cardSource, out int total)
    {
        var rs = CreatureCmd.Damage(k, [OldCreature.Of(target)], amount, props, dealer, cardSource);
        total = rs.Count > 0 ? rs[0].BlockedDamage + rs[0].UnblockedDamage : 0;
        return rs.Count > 0 ? rs[0].UnblockedDamage : 0;
    }

    /// <summary>
    /// Damage to every hittable enemy in ONE CreatureCmd.Damage call (old CardEffects.DealUnpoweredDamageToAll and the
    /// per-enemy loops of the game's list-damage bodies: Hailstorm, Inferno, Outbreak, Black Hole, Speedster, Panache,
    /// Smokestack, The Bomb, the relics and potions that hit every enemy). game: <c>CreatureCmd.Damage(ctx,
    /// CombatState.HittableEnemies, ...)</c>: every target is hit before any death resolves (Damage's one Kill after the
    /// list), so a listener of the second target sees the first still standing. The dealer and the source are the
    /// caller's, as <see cref="HitEnemy" />: every list caller's game call has the player (the owner) as dealer; the
    /// source is null for every power, relic, orb and potion list (the game passes null), the card in flight only for a
    /// card's own list.
    /// </summary>
    public static IReadOnlyList<DamageResult> HitEnemies(KernelState k, decimal amount, ValueProp props,
        OldCreature? dealer, CardHandle? cardSource) =>
        CreatureCmd.Damage(k, OldCreatures.HittableEnemies(k.World!), amount, props, dealer, cardSource);

    /// <summary>Damage on the player (old DealDamageToPlayer, LosePlayerHp, LoseHp): returns the HP lost.</summary>
    public static int HitPlayer(KernelState k, int amount, ValueProp props, EnemyState? dealer = null,
        CardHandle? cardSource = null)
    {
        var by = dealer is null ? (OldCreature?)null : OldCreature.Of(dealer);
        var rs = CreatureCmd.Damage(k, [OldCreature.Player], amount, props, by, cardSource);
        return rs.Count > 0 ? rs[0].UnblockedDamage : 0;
    }

    /// <summary>Damage on Osty (old CardEffects.DamageOsty): unblockable, unpowered; returns the HP lost.</summary>
    public static int HitOsty(KernelState k, int amount)
    {
        if (amount <= 0)
            return 0;
        var rs = CreatureCmd.Damage(k, [OldCreature.Osty], amount, HpLoss, null, null);
        return rs.Count > 0 ? rs[0].UnblockedDamage : 0;
    }

    /// <summary>
    /// A monster's attack on the player (old EnemyAI.DealAttack): AttackCmd with <paramref name="hits" /> hits of
    /// <paramref name="baseDamage" />, the old post-loop bodies (Suck with the hits that landed, Painful Stabs) run once as
    /// the F06 bridge's AfterAttack. A hit lands when it took HP.
    /// </summary>
    public static void EnemyAttack(KernelState k, EnemyState enemy, int baseDamage, int hits, Action<int> afterAttack)
    {
        var cmd = new AttackCmd(OldCreature.Of(enemy), baseDamage, hits)
        {
            SingleTarget = OldCreature.Player,
            Bridge = new AttackBridge(AfterAttack: a => afterAttack(Landed(a))),
        };
        cmd.Execute(k);
    }

    /// <summary>One hit of a monster's attack outside its move's loop (old EnemyAI.DealAttackDamage): true when it took HP.</summary>
    public static bool EnemyHit(KernelState k, EnemyState enemy, int baseDamage)
    {
        var cmd = new AttackCmd(OldCreature.Of(enemy), baseDamage) { SingleTarget = OldCreature.Player };
        cmd.Execute(k);
        return Landed(cmd) > 0;
    }

    /// <summary>
    /// game: AttackContext (AttackCommand.CreateContextAsync), a card's own Damage calls grouped as one attack for the
    /// attack hooks (Omnislice, Echoing Slash). Opening it runs BeforeAttack, <see cref="AddHit" /> counts a hit and keeps
    /// its results on the context's command (game: an AttackCommand of 0 from the card, all opponents), disposing it runs
    /// AfterAttack. No CreatureAttacked entry and no hit-count hook: the game's context writes and asks neither. With F06
    /// off the bridge's old points run instead of the hooks, as in <see cref="AttackCmd.Execute" />.
    /// </summary>
    public sealed class AttackContext : IDisposable
    {
        private readonly KernelState _k;
        private readonly AttackRef _self;
        private readonly AttackBridge? _bridge;
        private bool _disposed;

        /// <summary>The context's command: its <see cref="AttackCmd.Results" /> are the hits added.</summary>
        public AttackCmd Attack { get; }

        public AttackContext(KernelState k, AttackBridge? bridge = null)
        {
            _k = k;
            _bridge = bridge;
            Attack = new AttackCmd(OldCreature.Player, 0m) { AllOpponents = true, CardSource = InFlight(k), Bridge = bridge };
            _self = k.Commands.Begin(Attack);
            if (k.On(Switch.F06))
                CreatureCmd.Settle(k, Hooks.BeforeAttack(k, new BeforeAttackArgs(_self)),
                    st => Hooks.BeforeAttackResume(k, new BeforeAttackArgs(_self), st));
            else
                bridge?.BeforeAttack?.Invoke(Attack);
        }

        /// <summary>game: AttackContext.AddHit (IncrementHitsInternal, AddResultsInternal); hands the results back.</summary>
        public IReadOnlyList<DamageResult> AddHit(IReadOnlyList<DamageResult> results)
        {
            Attack.Results.Add(results);
            return results;
        }

        /// <summary>game: AttackContext.DisposeAsync, AfterAttack once.</summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            try
            {
                if (_k.On(Switch.F06))
                {
                    var after = new AfterAttackArgs(new ChoiceCtx(), _self);
                    CreatureCmd.Settle(_k, Hooks.AfterAttack(_k, after), st => Hooks.AfterAttackResume(_k, after, st));
                }
                else
                {
                    _bridge?.AfterAttack?.Invoke(Attack);
                }
            }
            finally
            {
                _k.Commands.End(_self);
            }
        }
    }

    private static int Landed(AttackCmd a) => a.Results.Count(r => r.Any(x => x.UnblockedDamage > 0));

    /// <summary>
    /// The kill of one creature, one Kill call (old HP zeroed in place: GasBomb, Waterfall Giant's explosion, Osty). Doom
    /// too: the game's DoomPower.DoomKill calls Kill once per creature, so old KillDoomedEnemies calls this per enemy.
    /// </summary>
    public static void Kill(KernelState k, OldCreature c, bool force = false) => CreatureCmd.Kill(k, [c], force);

    /// <summary>An enemy's escape (old HP zeroed with the escape flag: Thieving Hopper, Fat Gremlin, Battleworn Dummy).</summary>
    public static void Escape(KernelState k, EnemyState enemy) => CreatureCmd.Escape(k, OldCreature.Of(enemy));
}
