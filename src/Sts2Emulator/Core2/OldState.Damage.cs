// The old state's damage history (docs/rebuild/MIGRATION_F08.md, "Fixes after R2's review"): the counters the old damage
// sites kept and old code still reads, owned by Core2's commands at the game's points. Each is the old field of a game
// history query; the AfterDamageReceived listeners' tallies and the old player-side body are their handlers' (batch L2).
// Game code is cited by class and method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

public static partial class OldCreatures
{
    /// <summary>
    /// game: CombatManager.History.DamageReceived, one entry per result, read by the old fields: Spite's "lost HP this
    /// turn" (<c>PlayerHpLostThisTurn</c>: unblocked &gt; 0 on the player's side turn, the entry's round and side), Tear
    /// Asunder's hits (<c>UnblockedDamageHitCount</c>: every entry with unblocked &gt; 0), Emotion Chip's previous-turn hit
    /// (<c>PlayerHitTurn</c>: every entry not fully blocked), Beat Into Shape's powered hits on a target this turn
    /// (<c>EnemyState.PoweredHitsThisTurn</c>: the player itself as dealer). Unguarded: the old engine has no IsEnding flag
    /// inside a Damage call (the game sets it at the win check, after Kill).
    /// </summary>
    public static void NoteDamageReceived(OldWorld w, DamageResult r, OldCreature? dealer)
    {
        var s = w.Combat!;
        if (r.Receiver.Kind == CreatureKind.Player)
        {
            if (r.UnblockedDamage > 0)
            {
                if (s.PlayerTurn)
                    s.PlayerHpLostThisTurn += r.UnblockedDamage;
                s.UnblockedDamageHitCount++;
            }

            if (!r.WasFullyBlocked)
            {
                int turn = s.Turn + 1;
                if (turn != s.PlayerHitTurn)
                {
                    s.PlayerHitTurnBefore = s.PlayerHitTurn;
                    s.PlayerHitTurn = turn;
                }
            }
        }
        else if (r.Receiver.Enemy is { } enemy && dealer is { Kind: CreatureKind.Player } && r.Props.IsPoweredAttack()
                 && s.PlayerTurn)
        {
            enemy.PoweredHitsThisTurn++;
        }
    }

    /// <summary>game: CombatManager.History.CreatureAttacked, read by Flatten and Rattle (Osty's attack entries this
    /// turn): one per attack command, <c>OstyAttacksThisTurn</c>.</summary>
    public static void NoteAttacked(OldWorld w, OldCreature attacker)
    {
        var s = w.Combat!;
        if (attacker.Kind == CreatureKind.Osty && s.PlayerTurn)
            s.OstyAttacksThisTurn++;
    }
}
