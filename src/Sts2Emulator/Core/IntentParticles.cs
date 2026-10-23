namespace Sts2Emulator.Core;

/// <summary>
/// The roster as one <see cref="EnemyAI.ChooseIntents" /> pass found it and as it left it: what a
/// resampled fork needs to roll the enemies' current moves again (<see cref="IntentParticles" />).
/// Never changed once made, so clones share it.
/// </summary>
public sealed record IntentRoll(EnemyState[] Before, EnemyState[] After, int Ascension);

/// <summary>
/// The enemies' current moves as hidden information (Spireline DESIGN.md 1.1, AI-state particles;
/// leak test L8). The player sees each intent's category, and for an attack its damage and hits;
/// not which move is behind it, nor the AI stream that picked it. A resampled fork therefore
/// rolls the last intent pass again from the state the pass started in, with a fresh stream,
/// and keeps a roll only when every enemy shows the same intent the player is looking at. That
/// is rejection sampling from the AI's own prior, so the moves come out in posterior proportion.
/// </summary>
/// <remarks>
/// What came before the pass (the moves already executed, which the player watched) is kept as
/// it is. Only the fields the pass itself writes are replaced. When the fight has moved those
/// fields since the pass (a summon, a forced move), or no roll survives
/// <see cref="MaxTries" />, the fork keeps the moves it has.
/// </remarks>
public static class IntentParticles
{
    /// <summary>DESIGN.md 1.1's K.</summary>
    public const int MaxTries = 4096;

    /// <summary>Roll the current moves again off <paramref name="rng" />.</summary>
    /// <returns>True when a consistent roll replaced them.</returns>
    public static bool Resample(CombatState combat, Random rng)
    {
        var roll = combat.LastIntentRoll;
        var live = combat.Enemies;
        if (roll is null || live.Count != roll.After.Length)
        {
            return false;
        }

        for (int i = 0; i < live.Count; i++)
        {
            if (!SameAiState(live[i], roll.After[i]))
            {
                return false;
            }
        }

        for (int attempt = 0; attempt < MaxTries; attempt++)
        {
            var particle = roll.Before.Select(e => e.Clone()).ToList();
            var stream = new Random(rng.Next());
            EnemyAI.ChooseIntents(particle, 0, stream, stream, roll.Ascension);
            if (!Enumerable.Range(0, live.Count).All(i => ShowsTheSame(particle[i], roll.After[i])))
            {
                continue;
            }

            for (int i = 0; i < live.Count; i++)
            {
                // A creature the pass skipped (dead, or spending the turn on a revive) rolled
                // nothing; its secondary intent is the turn's, not the pass's.
                if (roll.Before[i].Hp > 0 && BuffSystem.Get(roll.Before[i].Buffs, BuffId.Reviving) == 0)
                {
                    Adopt(particle[i], live[i]);
                }
            }

            combat.LastIntentRoll = roll with { After = [.. live.Select(e => e.Clone())] };
            return true;
        }

        return false;
    }

    /// <summary>What the screen shows of both intents: the category, and damage and hits where shown.</summary>
    private static bool ShowsTheSame(EnemyState a, EnemyState b) =>
        Displayed(a.CurrentIntent, b.CurrentIntent)
        && (
            a.SecondaryIntent is { } x
                ? b.SecondaryIntent is { } y && Displayed(x, y)
                : b.SecondaryIntent is null
        );

    private static bool Displayed(Intent a, Intent b) =>
        a.Type == b.Type
        && a.CarriesDamage == b.CarriesDamage
        && (
            (a.Type != IntentType.Attack && !a.CarriesDamage)
            || (a.Magnitude == b.Magnitude && a.Hits == b.Hits)
        );

    /// <summary>The creature and every field an intent pass writes are as the pass left them.</summary>
    private static bool SameAiState(EnemyState live, EnemyState after) =>
        live.DefId == after.DefId
        && live.Slot == after.Slot
        && live.CurrentIntent == after.CurrentIntent
        && live.MoveIndex == after.MoveIndex
        && live.LastMove == after.LastMove
        && live.LastMoveRepeats == after.LastMoveRepeats
        && live.OnceOnlyMoveUsed == after.OnceOnlyMoveUsed
        && live.MoveHistory.SequenceEqual(after.MoveHistory)
        && live.RollsNextMove == after.RollsNextMove
        && live.OffBalance == after.OffBalance
        && live.LastBranch == after.LastBranch
        && live.RepeatStreak == after.RepeatStreak
        && live.BranchCooldown == after.BranchCooldown;

    private static void Adopt(EnemyState particle, EnemyState live)
    {
        live.CurrentIntent = particle.CurrentIntent;
        live.SecondaryIntent = particle.SecondaryIntent;
        live.MoveIndex = particle.MoveIndex;
        live.LastMove = particle.LastMove;
        live.LastMoveRepeats = particle.LastMoveRepeats;
        live.OnceOnlyMoveUsed = particle.OnceOnlyMoveUsed;
        live.MoveHistory = [.. particle.MoveHistory];
        live.RollsNextMove = particle.RollsNextMove;
        live.OffBalance = particle.OffBalance;
        live.LastBranch = particle.LastBranch;
        live.RepeatStreak = particle.RepeatStreak;
        live.BranchCooldown = particle.BranchCooldown;
    }
}
