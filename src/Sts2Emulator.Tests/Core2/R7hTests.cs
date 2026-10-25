// Round R7h (docs/rebuild/MIGRATION_F08.md, "R7h"): the last R7 proof gates. With F08+F10 on, the play's aim is the
// play's (game: CardPlay.Target), so no copy of the combat keeps it past the play; and the player's AfterSideTurnEnd
// relics and HP powers run in Hook.AfterTurnEnd, after FlushPlayerHand, as the game's do. Game code is cited by class and
// method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;
using Kit = Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2;

public class R7hTests
{
    private static readonly Switch[] On = [Switch.F08, Switch.F10];

    // ------------------------------------------------------------------ the aim belongs to the play

    [Fact]
    public void ACopyTakenInsideAPlayKeepsNoAimAfterIt()
    {
        // 2b49779387ccd8b9: step 1's Strike aimed enemies[0]; a copy taken inside that play kept the aim, and the Beetle
        // Juice aimed at enemies[2] four steps later shrank enemies[0].
        var s = Kit.Combat(10, 10, 10);
        var k = Kit.Kernel(s, [], On);
        CombatState copy;
        using (k.World!.Aiming(s.Enemies[0]))
        {
            Assert.Same(s.Enemies[0], CardEffects.TargetedEnemyFor(s));
            copy = s.Clone();
        }

        Assert.Null(copy.Kernel!.World!.Aimed);
        copy.TargetEnemyIndex = 2;
        Assert.Same(copy.Enemies[2], CardEffects.TargetedEnemyFor(copy));
        s.TargetEnemyIndex = 1;
        Assert.Same(s.Enemies[1], CardEffects.TargetedEnemyFor(s));
    }

    [Fact]
    public void AnInPlaceRestoreInsideAPlayKeepsThePlaysAim()
    {
        // A copy put back on the live combat inside a play (a hidden-state restore at a screen) is not the play: the
        // open play's target stands until its scope ends.
        var s = Kit.Combat(10, 10, 10);
        var k = Kit.Kernel(s, [], On);
        var before = s.Clone();
        using (k.World!.Aiming(s.Enemies[1]))
        {
            before.CopyTo(s, inPlace: true);
            Assert.Same(s.Enemies[1], s.Kernel!.World!.Aimed);
        }

        Assert.Null(s.Kernel!.World!.Aimed);
    }

    // ------------------------------------------------------------------ AfterSideTurnEnd after the flush

    private static StepResult EndTurnSwitchedOn(Fight fight, int seed = 1)
    {
        var kernel = new Kernel(new KernelOptions(On), () => ListenerOrderTests.Ids, Sts2Emulator.Core2.Models.Registry.All);
        var engine = new RunEngine { Kernel = kernel };
        CardUids.NumberNew(fight.State);
        kernel.Attach(fight.State, engine.State);
        return engine.StepCombat(fight.State, fight.State.Hand.Count, new CountingRandom(seed), -1);
    }

    [Fact]
    public void AParryingShieldKillAtTheTurnEndDrawsGremlinHornsCardAfterTheFlush()
    {
        // 7171bcd6dd0b0bd6. game: ParryingShield.AfterSideTurnEnd runs in Hook.AfterTurnEnd, after FlushPlayerHand: the
        // card Gremlin Horn's AfterDeath draws stays in the hand, and the next turn's five come on top of it.
        var fight = Fight.Hand()
            .Draw(Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad,
                IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad).ToArray())
            .Enemy(hp: 1).Enemy(hp: 1);
        fight.State.Relics = [new RelicInstance(RelicEffects.ParryingShield), new RelicInstance(RelicEffects.GremlinHorn),
            new RelicInstance(RelicEffects.LunarPastry)];
        fight.State.PlayerBlock = 12;
        int stars = fight.State.Stars;

        var result = EndTurnSwitchedOn(fight);

        Assert.False(result.Terminal);
        Assert.Single(fight.State.Enemies, e => e.Hp > 0);
        Assert.Equal(6, fight.State.Hand.Count);
        Assert.Empty(fight.State.DiscardPile);
        Assert.Equal(4, fight.State.DrawPile.Count);
        Assert.Equal(stars + 1, fight.State.Stars);  // Lunar Pastry pays once, in the same pass
    }

    [Fact]
    public void RegenHealsAfterTheHandsTurnEndDamage()
    {
        // game: RegenPower.AfterSideTurnEnd (Hook.AfterTurnEnd) heals after the turn-end cards' damage (DoTurnEnd): 78 -2
        // (Burn) +5 caps at 80. The old order healed first (capped at 80) and Burn took it to 78.
        var fight = Fight.Hand(Card(ST.Burn))
            .Draw(Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad).ToArray())
            .Enemy(hp: 50).PlayerHp(78, 80).PlayerBuff(BuffId.Regen, 5);

        EndTurnSwitchedOn(fight);

        Assert.Equal(80, fight.State.PlayerHp);
        Assert.Equal(4, fight.PlayerBuffAmount(BuffId.Regen));
    }
}
