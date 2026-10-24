// game: CreatureCmd.Add over the old roster (docs/rebuild/MIGRATION_F08.md, batch L3b): the command StockPower.AfterDeath
// lists its new Axebot with. Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

public static partial class CreatureCmd
{
    /// <summary>
    /// game: CreatureCmd.Add(Creature): refused outside a combat in progress; CombatState.AddCreature puts the creature at
    /// the END of its side's list (an enemy: the old roster, <see cref="CombatState.Enemies" />) and refuses one the combat
    /// already holds; then CombatManager.AfterCreatureAdded (the creature's AfterAddedToRoom and, on the player's side of the
    /// turn, its move roll). Tier 1: the caller's old body built the creature (CombatState.CreateCreature's HP roll) and gave
    /// it what AfterAddedToRoom and the roll give it (an Axebot's Stock, its opening intent), so the command lists it and
    /// returns it. The game's IsLiveCombat test is always true over the old state. A creature that dies in the same Kill
    /// leaves after its AfterDeath, so a respawn added there stands at the end of the roster, as in the game. Last, the
    /// game's Hook.AfterCreatureAddedToCombat (PhilosophersStone, FurCoat, Sandpit, Murderous), fired behind its own
    /// family's switch (F01); with F01 off the old engine's body for it (RelicEffects.Spawned) runs where the old spawn
    /// sites call it, and the old in-place Axebot respawn calls it nowhere (MIGRATION_F08.md R7).
    /// </summary>
    public static OldCreature Add(KernelState k, EnemyState monster)
    {
        var s = Combat(k);
        if (!CombatEnd.IsInProgress(k))
            throw new InvalidOperationException("CreatureCmd.Add: the combat is not in progress");
        if (s.Enemies.Any(e => ReferenceEquals(e, monster)))
            throw new InvalidOperationException("CreatureCmd.Add: the creature is already in the combat");
        s.Enemies.Add(monster);
        var added = OldCreature.Of(monster);
        if (k.On(Switch.F01))
        {
            var a = new AfterCreatureAddedToCombatArgs(R(k.World!, added));
            Settle(k, Hooks.AfterCreatureAddedToCombat(k, a), st => Hooks.AfterCreatureAddedToCombatResume(k, a, st));
        }

        return added;
    }
}
