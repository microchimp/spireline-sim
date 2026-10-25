// Round R7g (docs/rebuild/MIGRATION_F08.md, "R7g"): a pair `port_list --unreached` lists (no record in the manifest reaches
// the model; the old engine plays it). Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2.Models.Powers;

internal static class NecroMasteryPower
{
    /// <summary>
    /// game: NecroMasteryPower.AfterCurrentHpChanged. An event: when the HP change is a loss, the creature is Osty and its
    /// pet owner is the power's owner, one CreatureCmd.Damage of the loss times the amount at every hittable enemy,
    /// unblockable and unpowered, from the owner, no card (a re-entry). The old site, CardEffects.DamageOsty, returns early
    /// under F08 (HitOsty) before its Necro Mastery loop, so only this handler deals it; the old loop hit each enemy in its
    /// own call, the game's one call is the kernel's.
    /// </summary>
    internal static HookStep AfterCurrentHpChanged(HookCtx c, ListenerRef self, in AfterCurrentHpChangedArgs a, ChoiceCtx? ctx)
    {
        if (a.Delta >= 0m || c.Power(self) is not { } p || c.PetOwner(a.Creature) is not { } petOwner || petOwner != p.Owner
            || Old(c, p.Owner) is not { } owner)
            return HookStep.Done;
        var targets = c.HittableEnemies().Select(r => c.Enemy(r)).OfType<EnemyState>().Select(OldCreature.Of).ToList();
        CreatureCmd.Damage(c.K, targets, -a.Delta * p.Amount, OldEntry.HpLoss, owner, null);
        return HookStep.Done;
    }

    /// <summary>The old creature a reference names: an enemy, the player, or Osty (a creature with a pet owner).</summary>
    private static OldCreature? Old(HookCtx c, CreatureRef r) =>
        c.Enemy(r) is { } e ? OldCreature.Of(e) : c.IsPlayer(r) ? OldCreature.Player : c.PetOwner(r) is not null ? OldCreature.Osty : null;

    internal static void Register(HandlerTable t) =>
        t.Model(ListenerKind.Power, "NecroMasteryPower")
            .On(HookId.AfterCurrentHpChanged, (HookEvent<AfterCurrentHpChangedArgs>)AfterCurrentHpChanged);
}
