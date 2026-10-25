// game: the batch L2 player-side AfterDamageReceived listeners (docs/rebuild/MIGRATION_F08.md, batch L2): Inferno, Rupture,
// Self-Forming Clay, Centennial Puzzle, Demon Tongue and Red Skull, which Core2's damage command ran as one old body
// before. Each test binds the kernel with the F08+F10 switch and the model's own Register, runs the game's command and
// asserts the game's effect; with the handler deleted the effect is missing, so each fails.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Powers;
using Sts2Emulator.Core2.Models.Relics;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models;

public class L2PlayerDamageTests
{
    internal static KernelState K(CombatState s, params Action<HandlerTable>[] models) =>
        new Kernel(new KernelOptions(On), () => Ids, models.ToList()).Bind(s);

    internal static EnemyState Foe(int hp = 30) => Enemy("Crusher", hp);

    internal static void EnemyHitsPlayer(KernelState k, EnemyState e, decimal amount) =>
        CreatureCmd.Damage(k, [OldCreature.Player], amount, ValueProps.PoweredAttack, OldCreature.Of(e), null);

    /// <summary>HP the player takes from its own card (Unblockable, Unpowered), as Offering or Blood Wall deal it.</summary>
    internal static void SelfDamage(KernelState k, decimal amount) =>
        CreatureCmd.Damage(k, [OldCreature.Player], amount, new ValueProp(ValueProps.Unblockable | ValueProps.Unpowered),
            OldCreature.Player, null);

    [Fact]
    public void InfernoBurnsEveryEnemyWhenItsOwnerIsHurtOnItsOwnTurn()
    {
        var a = Foe();
        var b = Foe();
        var s = Combat(a, b);
        BuffSystem.Apply(s.PlayerBuffs, BuffId.Inferno, 3);
        var k = K(s, InfernoPower.Register);
        SelfDamage(k, 2m);
        Assert.Equal((48, 27, 27), (s.PlayerHp, a.Hp, b.Hp));
        s.PlayerTurn = false;
        EnemyHitsPlayer(k, a, 2m);
        Assert.Equal((46, 27, 27), (s.PlayerHp, a.Hp, b.Hp));
    }

    [Fact]
    public void RuptureGivesStrengthForHpLostOnItsOwnersTurn()
    {
        var e = Foe();
        var s = Combat(e);
        BuffSystem.Apply(s.PlayerBuffs, BuffId.RupturePower, 2);
        var k = K(s, RupturePower.Register);
        SelfDamage(k, 3m);
        Assert.Equal(2, BuffSystem.Get(s.PlayerBuffs, BuffId.Strength));
        s.PlayerTurn = false;
        EnemyHitsPlayer(k, e, 3m);
        Assert.Equal(2, BuffSystem.Get(s.PlayerBuffs, BuffId.Strength));
    }

    [Fact]
    public void SelfFormingClayGivesBlockNextTurnForUnblockedDamage()
    {
        var e = Foe();
        var s = Combat(e);
        s.Relics.Add(new RelicInstance(RelicEffects.SelfFormingClay));
        var k = K(s, SelfFormingClay.Register);
        s.PlayerBlock = 10;
        EnemyHitsPlayer(k, e, 5m);
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.BlockNextTurn));
        EnemyHitsPlayer(k, e, 8m);
        Assert.Equal(3, BuffSystem.Get(s.PlayerBuffs, BuffId.BlockNextTurn));
    }

    [Fact]
    public void CentennialPuzzleIsUsedByTheFirstUnblockedHit()
    {
        var e = Foe();
        var s = Combat(e);
        s.Relics.Add(new RelicInstance(RelicEffects.CentennialPuzzle));
        var k = K(s, CentennialPuzzle.Register);
        s.PlayerBlock = 10;
        EnemyHitsPlayer(k, e, 5m);
        Assert.Equal(0, s.Relics[0].Counter);
        EnemyHitsPlayer(k, e, 8m);
        Assert.Equal(1, s.Relics[0].Counter);
    }

    [Fact]
    public void DemonTongueHealsTheFirstUnblockedHitOfItsOwnersTurn()
    {
        var e = Foe();
        var s = Combat(e);
        s.Relics.Add(new RelicInstance(RelicEffects.DemonTongue));
        var k = K(s, DemonTongue.Register);
        s.PlayerTurn = false;
        EnemyHitsPlayer(k, e, 5m);
        Assert.Equal((45, 0), (s.PlayerHp, s.Relics[0].Counter));
        s.PlayerTurn = true;
        SelfDamage(k, 4m);
        Assert.Equal((45, 1), (s.PlayerHp, s.Relics[0].Counter));
        SelfDamage(k, 4m);
        Assert.Equal(41, s.PlayerHp);
    }

    [Fact]
    public void RedSkullFollowsItsOwnersHpAcrossHalfOnDamageAndHeal()
    {
        var e = Foe();
        var s = Combat(e);
        s.Relics.Add(new RelicInstance(RelicEffects.RedSkull));
        var k = K(s, RedSkull.Register);
        EnemyHitsPlayer(k, e, 20m);
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.Strength));
        EnemyHitsPlayer(k, e, 5m);
        Assert.Equal(3, BuffSystem.Get(s.PlayerBuffs, BuffId.Strength));
        CreatureCmd.Heal(k, OldCreature.Player, 10m);
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.Strength));
    }
}
