using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CombatManager.StartTurn` rolls every enemy's next move (`Creature.PrepareForNextTurn`,
/// guarded by `!isExtraPlayerTurn`) right after `Hook.BeforeSideTurnStart` and before
/// anything else the turn start does -- the block clear, the hand draw, `AfterPlayerTurnStart`
/// (Inferno's burn, Rolling Boulder, Entropy), the orbs.
/// </summary>
/// <remarks>
/// The emulator rolled at the END of its turn start. An enemy that the turn start killed had
/// by then no HP and no roll, where the game had already drawn its move off `Rng.MonsterAi`:
/// night04's GVCYB1LRQJ and PPAD21YZ3F, a Leaf Slime S at 1 HP burned down by Inferno, are
/// one MonsterAi draw short at the first end of turn after the power lands.
/// </remarks>
public class TurnStartMoveRollTests
{
    /// <summary>Two Leaf Slime S: a RandomBranchState both, so every roll costs exactly one draw.</summary>
    private static Fight TwoSlimes()
    {
        var fight = Fight
            .Hand()
            .Enemy(
                hp: 1,
                defId: KE.LeafSlimeS,
                intent: new Intent(IntentType.Attack, 1)
            )
            .Enemy(
                hp: 40,
                defId: KE.LeafSlimeS,
                intent: new Intent(IntentType.Attack, 1)
            )
            .Draw(Enumerable.Repeat(new CardInstance(IC.DefendIronclad, false), 12).ToArray())
            .PlayerHp(60, 60);
        fight.State.AiRng = new CountingRandom(7);
        return fight;
    }

    /// <summary>
    /// The burn kills the slime, and it still rolled: two draws for two enemies, both of them
    /// standing when the turn began.
    /// </summary>
    [Fact]
    public void AnEnemyTheTurnStartKillsHasAlreadyRolled()
    {
        var fight = TwoSlimes().PlayerBuff(BuffId.Inferno, 6).PlayerBuff(BuffId.InfernoSelfDamage, 1);

        fight.EndTurn();

        // Gone from the roster or left at zero, either way no longer standing.
        var standing = Assert.Single(fight.State.Enemies, enemy => enemy.Hp > 0);
        Assert.Equal(34, standing.Hp);
        Assert.Equal(2, ((CountingRandom)fight.State.AiRng!).CallCount);
    }

    /// <summary>The baseline the test above is measured against: no burn, same two draws.</summary>
    [Fact]
    public void WithoutTheBurnBothEnemiesRollToo()
    {
        var fight = TwoSlimes();

        fight.EndTurn();

        Assert.Equal(2, ((CountingRandom)fight.State.AiRng!).CallCount);
    }

    /// <summary>
    /// Paels Eye's extra turn: the enemy side did not act, so `!isExtraPlayerTurn` skips the
    /// roll and the intents the player is looking at stay the intents they were.
    /// </summary>
    [Fact]
    public void AnExtraTurnRollsNothing()
    {
        var fight = TwoSlimes();
        fight.State.Relics.Add(new RelicInstance(RelicEffects.PaelsEye));

        fight.EndTurn();

        Assert.Equal(0, ((CountingRandom)fight.State.AiRng!).CallCount);
        Assert.Equal(new Intent(IntentType.Attack, 1), fight.State.Enemies[0].CurrentIntent);
    }
}
