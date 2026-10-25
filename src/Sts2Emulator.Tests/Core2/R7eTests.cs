// Round R7e (docs/rebuild/MIGRATION_F08.md, "R7e"): a hit's target that BeforeDamageReceived's nested damage killed and took
// out of the combat (an enemy's Thorns hit the player, whose Inferno burst kills it). The game's CreatureCmd.Damage goes on
// with the dead target, which has no HP left to lose; under the switch the kernel named the removed creature and threw.
// Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class R7eTests
{
    private static KernelState Bind(CombatState s) =>
        new Kernel(new KernelOptions([Switch.F08, Switch.F10]), () => ListenerOrderTests.Ids,
            [.. Sts2Emulator.Core2.Models.Registry.All]).Bind(s);

    [Fact]
    public void AHitWhoseTargetTheThornsInfernoChainKilledLosesNothingMoreAndEndsTheCombat()
    {
        var fight = Fight.Hand(new CardInstance(IC.StrikeIronclad, false)).Enemy(hp: 4, buffs: new BuffState(BuffId.Thorns, 2))
            .Energy(3).PlayerBuff(BuffId.Inferno, 6).PlayerBuff(BuffId.InfernoSelfDamage, 1);
        Bind(fight.State);
        int hp = fight.State.PlayerHp;
        var r = fight.Play(0, 0);
        Assert.DoesNotContain(fight.State.Enemies, e => e.Hp > 0);  // dead, and the corpse removed
        Assert.Equal(hp - 2, fight.State.PlayerHp);  // Thorns' 2 only: the Strike landed on a creature already dead
        Assert.True(r.Terminal);
        Assert.True(r.PlayerWon);
    }
}
