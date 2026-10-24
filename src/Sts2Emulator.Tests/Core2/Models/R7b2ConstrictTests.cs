// Round R7b2 (rebuild/f08-r7b2), the R7 check "Constrict: the Applier test" of docs/rebuild/MIGRATION_F08.md: the game removes
// the power when its Applier dies; the old state records the Applier under the switch. Game code is cited by class and method
// only.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class R7b2ConstrictTests
{
    private static int Buff(List<BuffState> buffs, BuffId id) => BuffSystem.Get(buffs, id);

    // ---------------------------------------------------------------------------------------- Constrict

    /// <summary>Two Slithering Stranglers (creatures 1 and 2), the player's Constrict 3 applied by the first.</summary>
    private static CombatState TwoStranglers()
    {
        var s = Combat(30, 40);
        foreach (var e in s.Enemies)
            e.DefId = KE.SlitheringStrangler;
        GivePower(s, "ConstrictPower", BuffId.Constrict, 3);
        s.Enemies[0].AppliedPlayersConstrict = true;
        return s;
    }

    [Fact]
    public void ConstrictStaysWhenTheStranglerThatDidNotApplyItDiesAndGoesWithItsApplier()
    {
        // game: ConstrictPower.AfterDeath removes the power when the dead creature is its Applier.
        var s = TwoStranglers();
        var k = Kernel(s, Registry.Power4);
        using (Dying(k, s.Enemies[1]))
            Hooks.AfterDeath(k, AfterDeathOf(2));
        Assert.Equal(3, Buff(s.PlayerBuffs, BuffId.Constrict));
        using (Dying(k, s.Enemies[0]))
            Hooks.AfterDeath(k, AfterDeathOf(1));
        Assert.Equal(0, Buff(s.PlayerBuffs, BuffId.Constrict));
    }

    [Fact]
    public void TheStranglersConstrictMoveRecordsTheApplierOnlyUnderTheSwitchAndOnlyWhenItCreatesThePower()
    {
        // game: PowerCmd.Apply sets Applier when it creates the power; a later Apply stacks onto it (ModifyAmount).
        var s = Combat(30, 40);
        foreach (var e in s.Enemies)
        {
            e.DefId = KE.SlitheringStrangler;
            e.CurrentIntent = new Intent(IntentType.Debuff, 0);
        }

        _ = Kernel(s, Registry.All);
        EnemyAI.ExecuteIntent(s.Enemies[1], s, new Random(1));
        EnemyAI.ExecuteIntent(s.Enemies[0], s, new Random(1));
        Assert.Equal(6, Buff(s.PlayerBuffs, BuffId.Constrict));
        Assert.Equal((false, true), (s.Enemies[0].AppliedPlayersConstrict, s.Enemies[1].AppliedPlayersConstrict));

        var off = Combat(30);
        off.Enemies[0].DefId = KE.SlitheringStrangler;
        off.Enemies[0].CurrentIntent = new Intent(IntentType.Debuff, 0);
        EnemyAI.ExecuteIntent(off.Enemies[0], off, new Random(1));
        Assert.Equal(3, Buff(off.PlayerBuffs, BuffId.Constrict));
        Assert.False(off.Enemies[0].AppliedPlayersConstrict);
    }
}
