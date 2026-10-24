using EmuDump.Engine;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;

namespace EmuDump.Engines.Old;

/// <summary>The old engine's open moves for IEngine.LegalActions: CombatEngine.ValidActions, with the targets each can take.</summary>
static class OldActions
{
    internal static IReadOnlyList<LegalAction> Of(RunEngine run)
    {
        var c = run.State.ActiveCombat;
        if (c == null || run.State.Phase != RunPhase.Combat || c.PendingSelection != null) return [];
        var targets = Enumerable.Range(0, c.Enemies.Count).Where(i => EnemyAI.IsTargetable(c.Enemies[i])).ToList();
        var none = new[] { -1 };
        var all = new List<LegalAction>();
        foreach (int action in CombatEngine.ValidActions(c))
        {
            if (action < c.Hand.Count)
            {
                var card = c.Hand[action];
                bool aims = CardEffects.TargetTypeOf(c, card, Sts2Emulator.GeneratedData.Cards.Get(card.DefId)) == CardTarget.AnyEnemy;
                if (aims && targets.Count == 0) continue;
                all.Add(new LegalAction("play", action, aims ? targets : none));
            }
            else if (action == c.Hand.Count)
            {
                all.Add(new LegalAction("end_turn", 0, none));
            }
            else
            {
                int slot = action - c.Hand.Count - 1;
                bool aims = PotionEffects.NeedsEnemyTarget(c.PotionSlots[slot]);
                if (aims && targets.Count == 0) continue;
                all.Add(new LegalAction("potion", slot, aims ? targets : none));
            }
        }

        return all;
    }
}
