// Round R7i (docs/rebuild/MIGRATION_F08.md, "R7i"): with F08+F10 on, the combat's kernel state is bound before the
// combat is set up, so a kill at turn one's start goes through CreatureCmd.Kill and leaves the roster as the game's does;
// and the turn end's relic pass asks each relic with the player's hooks still active. Game code is cited by class and
// method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.Core2;

public class R7iTests
{
    private static readonly Switch[] On = [Switch.F08, Switch.F10];

    private static CombatState StartSlimes(Kernel? kernel)
    {
        var engine = new RunEngine { Kernel = kernel };
        engine.Reset("0");
        Assert.Equal(0, engine.StartCombat(
            RunConstants.StarterDeckIds.ToArray(),
            encounterId: (int)CombatFactory.ActOneEncounter.SlimesWeak,
            [RunConstants.RelicBurningBlood, RelicEffects.FestivePopper, RelicEffects.MercuryHourglass],
            playerHp: 64,
            playerMaxHp: 80,
            [0, 0, 0],
            playerGold: 99));
        return engine.State.ActiveCombat!;
    }

    [Fact]
    public void ACombatStartKillLeavesTheRosterUnderTheSwitch()
    {
        // 2b49779387ccd8b9: Festive Popper's 9 and Mercury Hourglass's 3 at turn one's start kill the Twig Slime S (8 to 12
        // HP). game: CreatureCmd.Kill removes it from the combat there. The kernel's state was bound after the setup, so
        // the kill ran the old code, whose corpse sweep the switch turns off, and the corpse stayed listed: a later
        // target index named it.
        var old = StartSlimes(null);
        var core2 = StartSlimes(new Kernel(new KernelOptions(On), () => ListenerOrderTests.Ids,
            Sts2Emulator.Core2.Models.Registry.All));

        Assert.NotNull(core2.Kernel);
        Assert.Contains(old.Enemies, e => e.Hp <= 0);  // the old engine sweeps its corpse at the first step
        Assert.All(core2.Enemies, e => Assert.True(e.Hp > 0, $"enemy {e.DefId} listed at {e.Hp} HP"));
        Assert.Equal(old.Enemies.Where(e => e.Hp > 0).Select(e => (e.DefId, e.Hp)),
            core2.Enemies.Select(e => (e.DefId, e.Hp)));
        Assert.Equal(old.Enemies.Count - old.Enemies.Count(e => e.Hp <= 0), core2.Enemies.Count);
    }

    [Fact]
    public void AConstrictKillAtTheTurnEndShutsTheRelicsAfterIt()
    {
        // game: Hook.AfterTurnEnd walks the player's powers before its relics; Constrict's kill deactivates the player's
        // hooks, and Lunar Pastry is not visited.
        var fight = Fight.Hand()
            .Draw(Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad).ToArray())
            .Enemy(hp: 50).PlayerHp(3, 80).PlayerBuff(BuffId.Constrict, 5);
        fight.State.Relics = [new RelicInstance(RelicEffects.LunarPastry)];
        int stars = fight.State.Stars;

        var kernel = new Kernel(new KernelOptions(On), () => ListenerOrderTests.Ids, Sts2Emulator.Core2.Models.Registry.All);
        var engine = new RunEngine { Kernel = kernel };
        CardUids.NumberNew(fight.State);
        kernel.Attach(fight.State, engine.State);
        engine.StepCombat(fight.State, fight.State.Hand.Count, new CountingRandom(1), -1);

        Assert.True(fight.State.PlayerHp <= 0);
        Assert.Equal(stars, fight.State.Stars);
    }
}
