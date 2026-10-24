// The old state's damage history (docs/rebuild/MIGRATION_F08.md, "Fixes after R2's review"): the counters the old damage
// sites kept and old code still reads, owned by Core2's commands at the game's points. Each is the old field of a game
// history query or of a listener's tally; a ported model's handler takes its line over. Game code is cited by class and
// method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;

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

    /// <summary>
    /// The old state of the AfterDamageReceived listeners, for a result whose receiver was not killed: Beating
    /// Remnant's turn tally (<c>UnblockedDamageThisTurn</c>, held relic or not: the old field is global), Lava Lamp's
    /// latch (<c>TookUnblockedDamage</c>: unblocked &gt; 0, not Unblockable), Hardened Shell's tally on its owner, and the
    /// old player-side body (<c>CardEffects.AfterPlayerDamageReceived</c>: Rupture, Inferno, Centennial Puzzle,
    /// Self-Forming Clay, Demon Tongue, Red Skull) with Rupture's fromPlayedCard as the game reads it: the damage's
    /// card source is the card being played, on the player's turn.
    /// </summary>
    public static void AfterDamageReceived(OldWorld w, DamageResult r, ValueProp props, CardHandle? card)
    {
        var s = w.Combat!;
        if (r.Receiver.Kind == CreatureKind.Player)
        {
            s.UnblockedDamageThisTurn += r.UnblockedDamage;
            if (r.UnblockedDamage > 0 && !props.Has(ValueProps.Unblockable))
                s.TookUnblockedDamage = true;
            bool fromPlayedCard = card is { } h && w.InFlight is { } flying && flying.Uid == h.Uid && s.PlayerTurn;
            CardEffects.AfterPlayerDamageReceived(s, r.UnblockedDamage, fromPlayedCard);
        }
        else if (r.Receiver.Enemy is { } enemy && !r.WasFullyBlocked
                 && BuffSystem.Get(enemy.Buffs, BuffId.HardenedShell) > 0)
        {
            enemy.HardenedShellDamageThisTurn += r.UnblockedDamage;
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
