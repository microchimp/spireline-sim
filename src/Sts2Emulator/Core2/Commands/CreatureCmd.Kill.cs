// game: CreatureCmd.Kill, KillWithoutCheckingWinCondition and Escape, CombatManager.RemoveCreature and
// MonsterModel.PerformMove's end, over the old state (docs/rebuild/MIGRATION_F08.md section 3 items 4-6 and 8). Dispatch
// is through Hooks.* and, for the four pseudo-hooks (section 5), to the one model each belongs to. Round R3a: written and
// tested, called from no old site yet (R4 reroutes the callers). Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

public static partial class CreatureCmd
{
    /// <summary>
    /// game: CreatureCmd.Kill(creatures, force). Nobody to kill: nothing. Each creature through
    /// <see cref="KillWithoutCheckingWinCondition" /> in list order; then, when the list held the player and the player
    /// is dead, the loss (<see cref="LoseCombat" />, R3b's point). No win check here: the game checks it at its own points
    /// (CombatManager.CheckWinCondition, R3b). The dead-player EndTurn of the other branch needs a second player.
    /// </summary>
    /// <remarks>Every call is counted in <see cref="CommandLog.KillCalls" /> (R2's contract: one per Damage call, the
    /// chained kills of teammates and Osty counted too); the test seam <see cref="CommandLog.KillStandIn" /> runs in its
    /// place when set.</remarks>
    public static void Kill(KernelState k, IReadOnlyList<OldCreature> killed, bool force = false)
    {
        k.Commands.KillCalls++;
        if (k.Commands.KillStandIn is { } standIn)
        {
            standIn(k, killed);
            return;
        }

        if (killed.Count == 0)
            return;
        var s = (k.World ?? throw new InvalidOperationException("CreatureCmd.Kill needs a switched-on kernel state"))
            .Combat ?? throw new InvalidOperationException("CreatureCmd.Kill outside a combat");
        bool player = killed.Any(c => c.Kind == CreatureKind.Player);
        foreach (var c in killed.ToList())
            KillWithoutCheckingWinCondition(k, c, force, 0);
        if (player && OldCreatures.IsDead(s, OldCreature.Player))
            LoseCombat(k);
    }

    /// <summary>
    /// game: CreatureCmd.KillWithoutCheckingWinCondition. A creature no longer attached to the combat (an enemy out of the
    /// roster): nothing. The creature is marked dying (still in the combat) to the kill's end; HP above 0 is lost first
    /// (unblockable, unpowered) with AfterCurrentHpChanged, and the player's hooks are kept active over the
    /// player's own passes. BeforeDeath; ShouldDie (both passes) unless forced or max HP &lt;= 0.
    /// Death: the Died event (pseudo-hook), ShouldCreatureBeRemovedFromCombatAfterDeath, AfterDeath(false), the living
    /// teammates read; an enemy due for removal leaves the room (<see cref="RemoveCreature" />, always) and the roster
    /// (<see cref="OldRoster.RemoveFromRoster" />, skipped while it performs its move); whether it is a primary enemy is
    /// read before its powers go; the powers the predicates allow are removed, AfterRemoved per removed power; then a
    /// primary enemy whose living teammates are all secondary takes them with it, and a player loses its orbs, Osty, and
    /// its hooks (<see cref="OldWorld.DeactivatePlayerHooks" />, for the rest of the combat).
    /// Prevented: at most 10 rounds of AfterDeath(true), AfterPreventingDeath(preventer) and again while still dead.
    /// </summary>
    internal static void KillWithoutCheckingWinCondition(KernelState k, OldCreature c, bool force, int recursion)
    {
        var w = k.World!;
        var s = w.Combat!;
        if (!OldCreatures.InRoster(s, c))
            return;
        // Marked from the HP loss on: the game keeps the creature in the combat at 0 HP until CombatState.RemoveCreature.
        var dying = c.Enemy is { } enemy ? w.MarkDying(enemy) : null;
        var keep = c.Kind == CreatureKind.Player ? w.KeepPlayerHooks() : null;
        try
        {
            int hp = OldCreatures.Hp(s, c);
            if (hp > 0)
            {
                OldCreatures.SetHp(s, c, 0);
                var a = new AfterCurrentHpChangedArgs(R(w, c), -hp);
                Settle(k, Hooks.AfterCurrentHpChanged(k, a), st => Hooks.AfterCurrentHpChangedResume(k, a, st));
            }

            var self = R(w, c);
            var before = new BeforeDeathArgs(self);
            Settle(k, Hooks.BeforeDeath(k, before), st => Hooks.BeforeDeathResume(k, before, st));
            ListenerRef? preventer = null;
            if (force || OldCreatures.MaxHp(s, c) <= 0 || Hooks.ShouldDie(k, new ShouldDieArgs(R(w, c)), out preventer))
            {
                Died(k, c);
                bool remove = Hooks.ShouldCreatureBeRemovedFromCombatAfterDeath(k,
                    new ShouldCreatureBeRemovedFromCombatAfterDeathArgs(R(w, c)), out _);
                Hooks.AfterDeath(k, new AfterDeathArgs(R(w, c), false, 0f));
                var teammates = c.Kind == CreatureKind.Enemy
                    ? OldCreatures.Enemies(w).Where(t => t != c && !OldCreatures.IsDead(s, t)).ToList()
                    : new List<OldCreature>();
                if (remove && c.Enemy is { } gone && OldCreatures.InRoster(s, c))
                {
                    RemoveCreature(k, c);
                    if (!w.IsPerformingMove(gone))
                        OldRoster.RemoveFromRoster(w, gone);
                }

                bool primary = OldCreatures.IsPrimaryEnemy(w, c);
                RemoveAllPowersAfterDeath(k, c);
                if (c.Kind == CreatureKind.Enemy)
                {
                    if (primary && teammates.Count != 0 && teammates.All(t => OldCreatures.IsSecondaryEnemy(w, t)))
                        Kill(k, teammates);
                }
                else if (c.Kind == CreatureKind.Player)
                {
                    s.Orbs.Clear();
                    if (s.OstyMaxHp > 0 && s.OstyHp > 0)
                        Kill(k, [OldCreature.Osty], force);
                    keep?.Dispose();
                    keep = null;
                    // game: Player.DeactivateHooks, for the rest of the combat: the mark is not scoped back.
                    w.DeactivatePlayerHooks();
                    // game: CombatManager.HandlePlayerDeath runs only while another player lives (multiplayer).
                }
            }
            else
            {
                if (recursion >= 10)
                    throw new InvalidOperationException(
                        "Combat is ending, but something is continually preventing the last creature from being killed");
                Hooks.AfterDeath(k, new AfterDeathArgs(R(w, c), true, 0f));
                var after = new AfterPreventingDeathArgs(preventer!.Value, R(w, c));
                Settle(k, Hooks.AfterPreventingDeath(k, after), st => Hooks.AfterPreventingDeathResume(k, after, st));
                if (OldCreatures.IsDead(s, c))
                    KillWithoutCheckingWinCondition(k, c, force, recursion + 1);
            }
        }
        finally
        {
            keep?.Dispose();
            dying?.Dispose();
        }
    }

    /// <summary>
    /// game: CreatureCmd.Escape. A dead creature, or one no longer attached to the combat: nothing. Its powers removed
    /// with no AfterRemoved (Creature.RemoveAllPowersInternalExcept, with nothing excepted: every old buff, the engine's
    /// bookkeeping ones going with the creature), it leaves the room
    /// (<see cref="RemoveCreature" />) and goes on the escaped list (<see cref="OldRoster.Escaped" />: the roster
    /// too). No death hook runs: no BeforeDeath, ShouldDie, Died, AfterDeath. The win check follows at the next point.
    /// </summary>
    public static void Escape(KernelState k, OldCreature c)
    {
        var w = k.World ?? throw new InvalidOperationException("CreatureCmd.Escape needs a switched-on kernel state");
        var s = w.Combat ?? throw new InvalidOperationException("CreatureCmd.Escape outside a combat");
        if (OldCreatures.IsDead(s, c) || !OldCreatures.InRoster(s, c))
            return;
        if (c.Enemy is not { } enemy)
            throw new NotSupportedException("CreatureCmd.Escape of the player's side: the game's callers escape enemies");
        enemy.Buffs.Clear();
        RemoveCreature(k, c);
        OldRoster.Escaped(w, enemy);
    }

    /// <summary>
    /// game: MonsterModel.PerformMove around the move's actions: the monster performs its move (IsPerformingMove) while
    /// <paramref name="move" /> runs; then, the flag cleared, a monster that died during it is asked
    /// ShouldCreatureBeRemovedFromCombatAfterDeath again and, removal due, leaves the roster now (its room removal ran at
    /// the kill). R4's enemy-turn site wraps the old move (EnemyAI.ExecuteIntent) in it.
    /// </summary>
    public static void PerformMove(KernelState k, OldCreature monster, Action move)
    {
        var w = k.World ?? throw new InvalidOperationException("CreatureCmd.PerformMove needs a switched-on kernel state");
        var enemy = monster.Enemy ?? throw new ArgumentException("only a monster performs a move", nameof(monster));
        using (w.PerformingMove(enemy))
            move();
        var s = w.Combat!;
        if (!OldCreatures.IsDead(s, monster) || !OldCreatures.InRoster(s, monster))
            return;
        bool remove;
        using (w.MarkDying(enemy))
            remove = Hooks.ShouldCreatureBeRemovedFromCombatAfterDeath(k,
                new ShouldCreatureBeRemovedFromCombatAfterDeathArgs(R(w, monster)), out _);
        if (remove)
            OldRoster.RemoveFromRoster(w, enemy);
    }

    /// <summary>
    /// game: CombatManager.RemoveCreature: a monster's BeforeRemovedFromRoom (pseudo-hook, to its own model) and its move
    /// state reset (MonsterModel.ResetStateMachine, <see cref="OldRoster.ResetMoveState" />). StateTracker.Unsubscribe is
    /// the visuals'; the kernel's listener tables are built from the state, so they need no entry dropped.
    /// </summary>
    internal static void RemoveCreature(KernelState k, OldCreature c)
    {
        if (c.Enemy is not { } enemy)
            return;
        var w = k.World!;
        if (Pseudo<HookEvent<BeforeRemovedFromRoomArgs>>(k, HookId.BeforeRemovedFromRoom, ListenerKind.Monster,
                enemy.DefId) is { } fn)
            Run(fn(Monster(w, c), new BeforeRemovedFromRoomArgs(c), null), HookId.BeforeRemovedFromRoom);
        OldRoster.ResetMoveState(enemy);
    }

    /// <summary>
    /// game: Creature.RemoveAllPowersAfterDeath, then each removed power's AfterRemoved(oldOwner). Every power is judged
    /// first, in application order (Enumerable.Except reads its exceptions before it yields): kept when its own
    /// ShouldPowerBeRemovedAfterOwnerDeath (pseudo-hook) says no, else when Hook.ShouldPowerBeRemovedOnDeath says no (the
    /// OwnerInCombat step: an owner that left the combat removes all, no listener asked). Then the rest are removed in
    /// order, then AfterRemoved runs per removed power, in removal order.
    /// </summary>
    private static void RemoveAllPowersAfterDeath(KernelState k, OldCreature c)
    {
        var w = k.World!;
        var removed = new List<OldListener>();
        foreach (var p in OldCreatures.Powers(w, c))
        {
            bool own = Pseudo<HookPredicate<PowerDeathArgs>>(k, HookId.ShouldPowerBeRemovedAfterOwnerDeath,
                ListenerKind.Power, p.Who.ModelId) is not { } fn || fn(p.Who, new PowerDeathArgs(c, p.Buff));
            if (own && (!OldCreatures.InCombat(w, c) || Hooks.ShouldPowerBeRemovedOnDeath(k,
                    new ShouldPowerBeRemovedOnDeathArgs(new PowerRef(R(w, c), p.Who.ModelId, p.Who.Instance)), out _)))
                removed.Add(p);
        }

        var gone = removed.Select(p => (p, Buffs: RemovePower(w, c, p))).ToList();
        foreach (var (p, buffs) in gone)
        {
            if (Pseudo<HookEvent<AfterRemovedArgs>>(k, HookId.AfterRemoved, ListenerKind.Power, p.Who.ModelId) is { } fn)
                Run(fn(p.Who, new AfterRemovedArgs(c, p.Buff, buffs), null), HookId.AfterRemoved);
        }
    }

    /// <summary>game: PowerModel.RemoveInternal over the old buffs: every buff entry of the power's game model on the
    /// creature (one game power can span several BuffIds). Returns the entries removed.</summary>
    private static List<BuffState> RemovePower(OldWorld w, OldCreature c, OldListener p)
    {
        var buffs = OldCreatures.Buffs(w.Combat!, c)!;
        var ids = buffs.Select(b => b.Id).Distinct().ToList();
        var taken = new List<BuffState>();
        foreach (var id in ids)
        {
            bool same = id == p.Buff || (w.Ids.Power(id) is { Kind: not (PowerMapKind.EngineOnly or PowerMapKind.Folded or PowerMapKind.NotModelled) } m
                && m.ModelId == p.Who.ModelId);
            if (!same)
                continue;
            taken.AddRange(buffs.Where(b => b.Id == id));
            BuffSystem.Remove(buffs, id);
        }

        return taken;
    }

    /// <summary>The creature's Died event (Creature.InvokeDiedEvent), raised before the removal query: to the dying
    /// monster's own model (SlumberingBeetle and SoulNexus subscribe).</summary>
    private static void Died(KernelState k, OldCreature c)
    {
        if (c.Enemy is not { } enemy)
            return;
        if (Pseudo<HookEvent<DiedArgs>>(k, HookId.Died, ListenerKind.Monster, enemy.DefId) is { } fn)
            Run(fn(Monster(k.World!, c), new DiedArgs(c), null), HookId.Died);
    }

    /// <summary>R3b's point: game CombatManager.LoseCombat (the pending loss, processed at the next
    /// CheckWinCondition) and the F17 hand-off. Until R3b only counted (<see cref="CommandLog.LossCalls" />); the old
    /// state already shows the loss (the player at 0 HP: CombatEngine.IsAboutToLose).</summary>
    private static void LoseCombat(KernelState k) => k.Commands.LossCalls++;

    private static ListenerRef Monster(OldWorld w, OldCreature c) =>
        new((int)ListenerKind.Monster, c.Enemy!.DefId, OldCreatures.TryRef(w, c)?.Index ?? -1);

    private static T? Pseudo<T>(KernelState k, HookId hook, ListenerKind kind, int model) where T : Delegate =>
        k.Handlers?.Find(hook, kind, model) as T;

    private static void Run(HookStep step, HookId hook)
    {
        if (step.IsPaused)
            throw new NotSupportedException($"{hook}: a pseudo-hook handler cannot pause (the game calls it directly)");
    }
}

/// <summary>Arguments of the monster's BeforeRemovedFromRoom (game: MonsterModel.BeforeRemovedFromRoom, from
/// CombatManager.RemoveCreature). Pseudo-hook: no Hook dispatcher, called on the leaving monster only.</summary>
public readonly record struct BeforeRemovedFromRoomArgs(OldCreature Creature);

/// <summary>Arguments of the creature's Died event (game: Creature.InvokeDiedEvent). Pseudo-hook.</summary>
public readonly record struct DiedArgs(OldCreature Creature);

/// <summary>Arguments of a power's own ShouldPowerBeRemovedAfterOwnerDeath (game: PowerModel). Pseudo-hook: the power
/// on its dead owner, by its first buff entry.</summary>
public readonly record struct PowerDeathArgs(OldCreature Owner, BuffId Buff);

/// <summary>Arguments of a removed power's AfterRemoved(oldOwner) (game: PowerModel.AfterRemoved). Pseudo-hook: the
/// owner (which may have left the combat), the power's first BuffId and the buff entries removed with it.</summary>
public readonly record struct AfterRemovedArgs(OldCreature OldOwner, BuffId Buff, IReadOnlyList<BuffState> Removed);
