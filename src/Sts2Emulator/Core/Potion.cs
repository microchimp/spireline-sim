namespace Sts2Emulator.Core;

/// <summary>
/// `PotionModel.Usage`: when the belt's use button works. `CombatOnly` potions are offered in a fight only,
/// `AnyTime` ones also away from it, and `Automatic` ones (Fairy in a Bottle) have no use button at all -- they
/// fire from their own hook (a death), never from a player action. See <see cref="Effects.PotionEffects.UsageOf" />.
/// </summary>
public enum PotionUsage
{
    CombatOnly,
    AnyTime,
    Automatic,
}

public enum PotionRarity
{
    None,
    Common,
    Uncommon,
    Rare,
    Event,
    Token,
}

public readonly record struct PotionDef(
    int Id,
    string Name,
    PotionRarity Rarity = PotionRarity.None,
    // The game's `PotionModel.TargetType` -- the same `CardTarget` enum cards use
    // (`MegaCrit.Sts2.Core.Entities.Cards.TargetType` is shared between the two). Only
    // `AnyEnemy` performs target selection, which is what `PotionEffects.NeedsEnemyTarget`
    // reads.
    CardTarget TargetType = CardTarget.Self
);
