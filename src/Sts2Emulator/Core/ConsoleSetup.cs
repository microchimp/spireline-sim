namespace Sts2Emulator.Core;

/// <summary>
/// State changes the game's developer-console setup commands make, for the referee's setup fights
/// (referee/driver/Setup.cs): the same game commands, so the emulator can build the same start.
/// None of them is a rule of its own; each is the path an existing effect already takes, minus
/// the applier or source card the console never has.
/// </summary>
public static class ConsoleSetup
{
    /// <summary>
    /// <c>PowerCmd.Apply</c> / <c>PowerCmd.ModifyAmount</c> with no applier and no source card, as
    /// <c>ApplyPowerConsoleCmd</c> calls them: on the player (<paramref name="enemyIndex"/> -1) or on
    /// <c>state.Enemies[enemyIndex]</c>. With no applier, <c>ModifyPowerAmountGiven</c> is never asked
    /// (Snecko Skull, Unsettling Lamp stay out), and the after-change listeners that look for their
    /// owner as the applier (Vicious, Outbreak, Sleight of Flesh, ...) do not fire. What is left is
    /// <c>ModifyPowerAmountReceived</c>: Artifact eats a debuff application whole (a negative
    /// Strength or Dexterity is one), and Ruined Helmet doubles the player's first Strength.
    /// </summary>
    /// <returns>false when the target is not a living creature.</returns>
    public static bool ApplyPowerWithoutApplier(CombatState state, int enemyIndex, BuffId id, int amount)
    {
        bool player = enemyIndex < 0;
        if (!player && (enemyIndex >= state.Enemies.Count || state.Enemies[enemyIndex].Hp <= 0))
        {
            return false;
        }

        if (amount == 0)
        {
            return true;
        }

        var buffs = player ? state.PlayerBuffs : state.Enemies[enemyIndex].Buffs;
        if (BuffSystem.IsDebuffApplication(id, amount) && BuffSystem.TryConsumeArtifact(buffs))
        {
            return true;
        }

        if (player && id == BuffId.Strength)
        {
            amount = Effects.RelicEffects.DoubleFirstStrengthReceived(state, amount);
        }

        BuffSystem.Apply(buffs, id, amount);
        return true;
    }
}
