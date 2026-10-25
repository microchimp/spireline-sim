// game: the batch L1 models (docs/rebuild/MIGRATION_F08.md, batch L1): Paper Cuts, The Gambit, Hard to Kill, Slow, Lizard
// Tail, Fairy in a Bottle and the five powers kept after their owner's death. Each test binds the kernel with the F08+F10
// switch and the model's own Register, runs the game's command or dispatcher, and asserts the game's effect; with the
// handler deleted the pair is unregistered and the effect is missing, so each fails.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models.Potions;
using Sts2Emulator.Core2.Models.Powers;
using Sts2Emulator.Core2.Models.Relics;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.Monsters.MonsterHarness;

namespace Sts2Emulator.Tests.Core2.Models;

public class L1ModelTests
{
    private static KernelState K(CombatState s, params Action<HandlerTable>[] models) =>
        new Kernel(new KernelOptions(On), () => Ids, models.ToList()).Bind(s);

    private static EnemyState Foe(int hp = 30) => Enemy("Crusher", hp);

    private static void EnemyHitsPlayer(KernelState k, EnemyState e, decimal amount) =>
        CreatureCmd.Damage(k, [OldCreature.Player], amount, ValueProps.PoweredAttack, OldCreature.Of(e), null);

    private static ModifyDamageArgs Attack(int target, decimal damage, ValueProp? props = null) =>
        new(new CreatureRef(target), new CreatureRef(0), damage, props ?? ValueProps.PoweredAttack, null,
            ModifyDamageHookType.All, PreviewModes.None);

    [Fact]
    public void PaperCutsTakesMaxHpWhenItsOwnersAttackGetsThrough()
    {
        var e = Foe();
        BuffSystem.Apply(e.Buffs, BuffId.PaperCuts, 2);
        var s = Combat(e);
        var k = K(s, PaperCutsPower.Register);
        EnemyHitsPlayer(k, e, 6m);
        Assert.Equal((48, 44), (s.PlayerMaxHp, s.PlayerHp));

        s.PlayerBlock = 10;
        EnemyHitsPlayer(k, e, 6m);
        Assert.Equal((48, 44), (s.PlayerMaxHp, s.PlayerHp));
    }

    [Fact]
    public void TheGambitKillsItsOwnerOnAnUnblockedPoweredHit()
    {
        var e = Foe();
        var s = Combat(e);
        BuffSystem.Apply(s.PlayerBuffs, BuffId.TheGambitPower, 1);
        var k = K(s, TheGambitPower.Register);
        s.PlayerBlock = 10;
        EnemyHitsPlayer(k, e, 5m);
        Assert.Equal(50, s.PlayerHp);
        s.PlayerBlock = 0;
        EnemyHitsPlayer(k, e, 5m);
        Assert.Equal(0, s.PlayerHp);
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.TheGambitPower));
    }

    [Fact]
    public void HardToKillCapsTheDamageItsOwnerTakes()
    {
        var e = Foe();
        BuffSystem.Apply(e.Buffs, BuffId.HardToKill, 5);
        var k = K(Combat(e), HardToKillPower.Register);
        Assert.Equal(5m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 20m), out _));
        Assert.Equal(20m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 20m), out _));
        Assert.Equal(5m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 20m, new ValueProp(ValueProps.Unpowered)), out _));
    }

    [Fact]
    public void SlowScalesPoweredAttacksByTheCardsPlayedThisTurn()
    {
        var e = Foe();
        BuffSystem.Apply(e.Buffs, BuffId.Slow, 1);
        BuffSystem.Apply(e.Buffs, BuffId.SlowCount, 3);
        var k = K(Combat(e), SlowPower.Register);
        Assert.Equal(13m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 10m), out _));
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 10m, new ValueProp(ValueProps.Unpowered)), out _));
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 10m), out _));
    }

    [Fact]
    public void LizardTailRefusesTheFirstDeathAndHealsHalf()
    {
        var e = Foe();
        var s = Combat(e);
        s.Relics.Add(new RelicInstance(RelicEffects.LizardTail));
        var k = K(s, LizardTail.Register);
        EnemyHitsPlayer(k, e, 60m);
        Assert.Equal(25, s.PlayerHp);
        Assert.False(RelicEffects.LizardTailUnspent(s));
        EnemyHitsPlayer(k, e, 60m);
        Assert.Equal(0, s.PlayerHp);
    }

    [Fact]
    public void FairyInABottleRefusesTheDeathIsUsedUpAndHealsThirtyPercent()
    {
        var e = Foe();
        var s = Combat(e);
        s.PotionSlots[0] = GeneratedData.Potions.FindId("FairyInABottle")!.Value;
        var k = K(s, FairyInABottle.Register);
        EnemyHitsPlayer(k, e, 60m);
        Assert.Equal(15, s.PlayerHp);
        Assert.Equal(0, s.PotionSlots[0]);
        EnemyHitsPlayer(k, e, 60m);
        Assert.Equal(0, s.PlayerHp);
    }

    [Theory]
    [InlineData("MinionPower")]
    [InlineData("ReattachPower")]
    [InlineData("AdaptablePower")]
    [InlineData("SteamEruptionPower")]
    [InlineData("PainfulStabsPower")]
    public void ThePowerStaysOnItsDeadOwner(string power)
    {
        Action<HandlerTable> register = power switch
        {
            "MinionPower" => MinionPower.Register,
            "ReattachPower" => ReattachPower.Register,
            "AdaptablePower" => AdaptablePower.Register,
            "SteamEruptionPower" => SteamEruptionPower.Register,
            _ => PainfulStabsPower.Register,
        };
        // Batch L2 added the models' other pairs (L2DeathTests); this pair is the pseudo-hook's.
        Assert.Contains((power, HookId.ShouldPowerBeRemovedAfterOwnerDeath), Pairs(register).Select(h => (h.GameId, h.Hook)));
        var t = new HandlerTable(Ids, _ => true);
        register(t);
        var fn = (HookPredicate<PowerDeathArgs>)t.Find(HookId.ShouldPowerBeRemovedAfterOwnerDeath, ListenerKind.Power,
            Ids.Resolve(ListenerKind.Power, power))!;
        var s = Combat(Foe());
        var k = K(s, register);
        Assert.False(fn(k.Ctx, default, new PowerDeathArgs(OldCreature.Of(s.Enemies[0]), default)));
    }
}
