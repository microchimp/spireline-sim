// Round R7b2 (rebuild/f08-r7b2), the R7 checks of docs/rebuild/MIGRATION_F08.md that the handlers inherited from the old
// sites: each handler takes the game's condition under the switch. Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class R7b2Tests
{
    private const int ShivDef = 430;
    private const int KnifeTrapDef = 276;

    private static int Buff(List<BuffState> buffs, BuffId id) => BuffSystem.Get(buffs, id);

    // ---------------------------------------------------------------------------------------- Phantom Blades

    [Fact]
    public void PhantomBladesAnswersTheGamesShivTagWhichOnlyShivCarries()
    {
        // game: PhantomBladesPower.ModifyDamageAdditive tests CardTag.Shiv on the card source; Shiv is the one card the game
        // tags Shiv. CardDef.ShivTag also marks Knife Trap (the extractor matches the tag's name in the card's text), so a
        // Knife Trap card source gets no bonus, as in the game; a Shiv gets the power's amount.
        Assert.True(GeneratedData.Cards.Get(KnifeTrapDef).ShivTag);
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.PhantomBlades, 3));
        var shiv = Add(s, s.Hand, ShivDef);
        var trap = Add(s, s.Hand, KnifeTrapDef);
        var k = Kernel(s, Registry.Pilot);
        Assert.Equal(4m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: new CardHandle(trap.Uid)), out _));
        Assert.Equal(7m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 4m, card: new CardHandle(shiv.Uid)), out _));
    }

    // ---------------------------------------------------------------------------------------- Buffer

    [Fact]
    public void AnEnemysBufferTakesItsHpLossAndSpendsAStack()
    {
        // game: BufferPower.ModifyHpLostAfterOstyLate and AfterModifyingHpLostAfterOsty act for any owner.
        var s = Combat(40);
        var theirs = GivePower(s, "BufferPower", BuffId.Buffer, 2, s.Enemies[0]);
        var k = Kernel(s, Registry.Power2);

        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 7m), out var changed));
        Assert.Equal([theirs], changed);
        Hit(k, OldCreature.Of(s.Enemies[0]), 7m);
        Assert.Equal((40, 1), (s.Enemies[0].Hp, Buff(s.Enemies[0].Buffs, BuffId.Buffer)));
        Hit(k, OldCreature.Of(s.Enemies[0]), 7m);
        Assert.Equal(40, s.Enemies[0].Hp);
        Assert.DoesNotContain(s.Enemies[0].Buffs, b => b.Id == BuffId.Buffer);
        Hit(k, OldCreature.Of(s.Enemies[0]), 7m);
        Assert.Equal(33, s.Enemies[0].Hp);
    }

    // ---------------------------------------------------------------------------------------- Personal Hive

    [Fact]
    public void PersonalHiveShufflesNoDazedForAPoweredHitWithNoDealer()
    {
        // game: PersonalHivePower.AfterDamageReceived tests dealer != null.
        var s = Combat(40);
        GivePower(s, "PersonalHivePower", BuffId.PersonalHive, 2, s.Enemies[0]);
        var k = Kernel(s, Registry.Power3);
        CreatureCmd.Damage(k, [OldCreature.Of(s.Enemies[0])], 5m, ValueProps.PoweredAttack, null, null);
        Assert.Equal(35, s.Enemies[0].Hp);
        Assert.Empty(s.DrawPile);
        Hit(k, OldCreature.Of(s.Enemies[0]), 5m);
        Assert.Equal(2, s.DrawPile.Count);
    }

    // ---------------------------------------------------------------------------------------- Intangible

    [Fact]
    public void IntangibleLeavesAnHpLossAloneOnceTheCombatIsNotInProgress()
    {
        // game: IntangiblePower.ModifyHpLostAfterOsty returns the amount while CombatManager.IsInProgress is false.
        var s = Combat(40);
        var power = GivePower(s, "IntangiblePower", BuffId.Intangible, 1, s.Enemies[0]);
        var k = Kernel(s, Registry.Power2);
        Assert.Equal(1m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 9m), out var changed));
        Assert.Equal([power], changed);
        k.Progress.InProgress = false;
        Assert.Equal(9m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 9m), out changed));
        Assert.Empty(changed);
    }

    // ---------------------------------------------------------------------------------------- Thorns

    [Fact]
    public void AnEnemysThornsHitsOstyWhenOstyStrikes()
    {
        // game: ThornsPower.BeforeDamageReceived damages the dealer.
        var s = WithOsty(Combat(40), 10);
        GivePower(s, "ThornsPower", BuffId.Thorns, 3, s.Enemies[0]);
        var k = Kernel(s, Registry.L3a);
        Hit(k, OldCreature.Of(s.Enemies[0]), 6m, dealer: OldCreature.Osty);
        Assert.Equal((50, 7, 34), (s.PlayerHp, s.OstyHp, s.Enemies[0].Hp));
        Hit(k, OldCreature.Of(s.Enemies[0]), 6m);
        Assert.Equal((47, 7, 28), (s.PlayerHp, s.OstyHp, s.Enemies[0].Hp));
    }

    // ---------------------------------------------------------------------------------------- Crab Rage

    [Fact]
    public void CrabRageAnswersTheDeathOfAnAllyThatHoldsNoCrabRage()
    {
        // game: CrabRagePower.AfterDeath answers any death on the owner's side but its own.
        var s = Combat(30, 30);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.CrabRage, 1));
        s.PlayerBuffs.Add(new BuffState(BuffId.CrabRage, 1));
        var k = Kernel(s, Registry.L3a);
        using (Dying(k, s.Enemies[1]))
            Hooks.AfterDeath(k, AfterDeathOf(2));
        Assert.Equal(RunConstants.CrabRageStrength, Buff(s.Enemies[0].Buffs, BuffId.Strength));
        Assert.Equal(RunConstants.CrabRageBlock, s.Enemies[0].Block);
        Assert.Equal(0, Buff(s.Enemies[0].Buffs, BuffId.CrabRage));
        // The player's copy: an enemy's death is on the other side.
        Assert.Equal(1, Buff(s.PlayerBuffs, BuffId.CrabRage));
        Assert.Equal(0, s.PlayerBlock);
    }

    // ---------------------------------------------------------------------------------------- Surrounded

    [Fact]
    public void SurroundedMultipliesAnUnpoweredHitFromTheBack()
    {
        // game: SurroundedPower.ModifyDamageMultiplicative has no props test.
        var s = Combat(30, 30);
        var mine = GivePower(s, "SurroundedPower", BuffId.Surrounded, RunConstants.FacingRight);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.BackAttackLeft, 1));
        s.Enemies[1].Buffs.Add(new BuffState(BuffId.BackAttackRight, 1));
        var k = Kernel(s, Registry.L3a);
        Assert.Equal(15m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 10m, props: Unpowered, dealer: 1), out var changed));
        Assert.Equal([mine], changed);
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 10m, props: Unpowered, dealer: 2), out _));
    }

    // ---------------------------------------------------------------------------------------- Die For You

    [Fact]
    public void ARevivedOstyStillTakesThePlayersPoweredHit()
    {
        // game: OstyCmd.Summon applies DieForYouPower at the first add; ShouldPowerBeRemovedAfterOwnerDeath is false, so the
        // dead Osty keeps it and the revive (the same creature re-added) holds it.
        var s = WithOsty(Combat(40), 10);
        var k = Kernel(s, Registry.R7a);
        CreatureCmd.Kill(k, [OldCreature.Osty]);
        Assert.Equal(0, s.OstyHp);
        CardEffects.SummonOsty(s, 8);
        Assert.Equal(8, s.OstyHp);
        Hit(k, OldCreature.Player, 6m, dealer: OldCreature.Of(s.Enemies[0]));
        Assert.Equal((50, 2), (s.PlayerHp, s.OstyHp));
    }
}
