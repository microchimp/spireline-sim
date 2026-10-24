// game: BurrowedPower (MegaCrit.Sts2.Core.Models.Powers), its F08/F10 overrides (docs/rebuild/port-f08-f10.md, batch power-4;
// docs/rebuild/MIGRATION_F08.md, batch L2). Registered under the F08+F10 switch from Core2/Models/Registry.cs.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

/// <summary>Burrowed: the Tunneler keeps its block until the block is broken, which stuns it.</summary>
internal static class BurrowedPower
{
    /// <summary>game: BurrowedPower.AfterBlockBroken. When the creature is the owner and a Tunneler: it is stunned into its
    /// still-dizzy move and the power removed. The old body is <c>EnemyAI.BreakBurrowIfBlockGone</c> (Burrowed removed,
    /// the block cleared, Stunned, the move reset).</summary>
    internal static HookStep AfterBlockBroken(HookCtx c, ListenerRef self, in AfterBlockBrokenArgs a, ChoiceCtx? ctx)
    {
        if (c.Power(self) is not { } p || a.Creature != p.Owner || !c.IsMonster(p.Owner, "Tunneler")
            || c.Enemy(p.Owner) is not { } e)
            return HookStep.Done;
        EnemyAI.BreakBurrowIfBlockGone(e);
        return HookStep.Done;
    }

    /// <summary>game: BurrowedPower.AfterRemoved (pseudo-hook): CreatureCmd.LoseBlock of all the old owner's block. Core2
    /// dispatches AfterRemoved only from the owner's death removal; a dead Burrowed owner leaves the combat (no power
    /// keeps it), so its block is read by nothing and the handler changes nothing. Not modelled: a removal outside a death
    /// (PowerCmd.Remove; the old <c>EnemyAI.BreakBurrowIfBlockGone</c> clears the block there).</summary>
    internal static HookStep AfterRemoved(HookCtx c, ListenerRef self, in AfterRemovedArgs a, ChoiceCtx? ctx) =>
        HookStep.Done;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "BurrowedPower")
            .On(HookId.AfterBlockBroken, (HookEvent<AfterBlockBrokenArgs>)AfterBlockBroken)
            .On(HookId.AfterRemoved, (HookEvent<AfterRemovedArgs>)AfterRemoved);
}
