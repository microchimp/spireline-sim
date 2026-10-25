// Round R7f (docs/rebuild/MIGRATION_F08.md, "R7f"): the R7 proof's worse corpus fights. Two F08 damage folds the port list
// left out (no record in its manifest reaches the models themselves): Diamond Diadem's and Undying Sigil's halving of an
// enemy's attack.
// Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.GeneratedData;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class R7fTests
{
    private static KernelState Bind(CombatState s) =>
        new Kernel(new KernelOptions([Switch.F08, Switch.F10]), () => ListenerOrderTests.Ids,
            [.. Sts2Emulator.Core2.Models.Registry.All]).Bind(s);

    private static int HpAfterHit(Fight fight, bool core2, int damage)
    {
        if (core2)
            Bind(fight.State);
        int hp = fight.State.PlayerHp;
        EnemyAI.DealAttackDamageForTests(fight.State.Enemies[0], fight.State, damage);
        return hp - fight.State.PlayerHp;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DiamondDiademHalvesAnEnemyAttack(bool core2)
    {
        // game: DiamondDiademPower.ModifyDamageMultiplicative, one half on a powered attack on the owner: 13 -> 6.
        var fight = Fight.Hand(new CardInstance(IC.StrikeIronclad, false)).Enemy(hp: 40).PlayerBuff(BuffId.DiamondDiadem, 1);
        Assert.Equal(6, HpAfterHit(fight, core2, 13));
    }

    [Theory]
    [InlineData(false, 5)]
    [InlineData(true, 5)]
    [InlineData(false, 0)]
    [InlineData(true, 0)]
    public void UndyingSigilHalvesTheAttackOfACreatureAtItsDoom(bool core2, int doom)
    {
        // game: UndyingSigil.ModifyDamageMultiplicative: an attacker whose HP is at or below its Doom deals half.
        var fight = Fight.WithRelics(RelicEffects.UndyingSigil).Enemy(hp: 5, buffs: new BuffState(BuffId.Doom, doom));
        Assert.Equal(doom > 0 ? 5 : 11, HpAfterHit(fight, core2, 11));
    }
}
