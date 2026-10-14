using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// `HardenedShellPower` keeps its Amount (the cap, 20) for the whole fight and counts what the
/// owner has lost this turn in a private `damageReceivedThisTurn`: `ModifyHpLostBeforeOstyLate`
/// returns `Math.Min(amount, Amount - damageReceivedThisTurn)`, `AfterDamageReceived` adds the
/// result's `UnblockedDamage`, and `BeforeSideTurnStart` zeroes the tally for EITHER side's
/// turn. The emulator used to spend the Amount itself and drop the power at zero, after which
/// nothing capped and the rest of the turn's damage landed in full.
/// </summary>
public class HardenedShellPoolTests
{
    private static BuffState Shell => new(BuffId.HardenedShell, 20, 0);

    /// <summary>
    /// The mechanism: a shell spent by one swing stays spent, so a second swing the same turn
    /// does nothing, however big it is.
    /// </summary>
    [Fact]
    public void ASpentShellStillCapsEverythingLaterThatTurn()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SkulkingColony);
        var colony = fight.Enemy0;
        colony.Block = 0;
        int hp = colony.Hp;

        CardEffects.DealDamage(fight.State, 50);
        Assert.Equal(hp - 20, colony.Hp);

        CardEffects.DealDamage(fight.State, 50);

        Assert.Equal(hp - 20, colony.Hp);
    }

    /// <summary>
    /// The cap is the Amount and it never moves: the readout (and the recorded game, which
    /// reports `Amount`) stays at 20 through the turn.
    /// </summary>
    [Fact]
    public void TheAmountStaysTheCapWhileTheTallyRises()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SkulkingColony);
        var colony = fight.Enemy0;
        colony.Block = 0;

        CardEffects.DealDamage(fight.State, 12);

        Assert.Equal(20, BuffSystem.Get(colony.Buffs, BuffId.HardenedShell));
        Assert.Equal(12, colony.HardenedShellDamageThisTurn);

        CardEffects.DealDamage(fight.State, 12);

        Assert.Equal(20, BuffSystem.Get(colony.Buffs, BuffId.HardenedShell));
        Assert.Equal(20, colony.HardenedShellDamageThisTurn);
    }

    /// <summary>
    /// A multi-hit attack against a shell it spends part-way: 15 lands, 5 lands, and the third
    /// 15 lands nothing (it landed in full before).
    /// </summary>
    [Fact]
    public void AMultiHitCardSpendsTheShellMidAttack()
    {
        var fight = Fight
            .Hand(Card(IC.Whirlwind))
            .Energy(3)
            .PlayerBuff(BuffId.Strength, 10)
            .Enemy(hp: 100, buffs: Shell);

        fight.Play();

        Assert.Equal(80, fight.Enemy0.Hp);
    }

    /// <summary>
    /// An all-enemies attack spends each creature's own tally: the shelled one stops at 20, the
    /// plain one beside it takes every hit.
    /// </summary>
    [Fact]
    public void AnAreaAttackSpendsOnlyTheShelledCreaturesPool()
    {
        var fight = Fight
            .Hand(Card(IC.Whirlwind))
            .Energy(3)
            .PlayerBuff(BuffId.Strength, 10)
            .Enemy(hp: 100, buffs: Shell)
            .Enemy(hp: 100);

        fight.Play();

        Assert.Equal(80, fight.Enemy0.Hp);
        Assert.Equal(55, fight.Enemy1.Hp);
    }

    /// <summary>Two shelled creatures each have a pool of their own.</summary>
    [Fact]
    public void EachShelledCreatureHasItsOwnPool()
    {
        var fight = Fight
            .Hand(Card(IC.Whirlwind))
            .Energy(3)
            .PlayerBuff(BuffId.Strength, 10)
            .Enemy(hp: 100, buffs: Shell)
            .Enemy(hp: 100, buffs: Shell);

        fight.Play();

        Assert.Equal(80, fight.Enemy0.Hp);
        Assert.Equal(80, fight.Enemy1.Hp);
    }

    /// <summary>
    /// Unpowered damage draws on the same pool as a card's (the hook reads no props): a poison
    /// tick or Combust after the swing that spent the shell lands nothing.
    /// </summary>
    [Fact]
    public void UnpoweredDamageAfterASpentShellLandsNothing()
    {
        var fight = Fight.Hand().Enemy(hp: 100, buffs: Shell);

        CardEffects.DealDamage(fight.State, 50);
        Assert.Equal(80, fight.Enemy0.Hp);

        CardEffects.DealUnpoweredDamageToEnemy(fight.State, fight.Enemy0, 30);

        Assert.Equal(80, fight.Enemy0.Hp);
    }

    /// <summary>
    /// Block comes off first and only the HP actually lost counts against the shell
    /// (`DamageResult.UnblockedDamage`): 10 block turns a 15 hit into 5 off the pool.
    /// </summary>
    [Fact]
    public void OnlyTheHpActuallyLostSpendsTheShell()
    {
        var fight = Fight.Hand().Enemy(hp: 100, block: 10, buffs: Shell);

        CardEffects.DealDamage(fight.State, 15);
        Assert.Equal(95, fight.Enemy0.Hp);
        Assert.Equal(5, fight.Enemy0.HardenedShellDamageThisTurn);

        CardEffects.DealDamage(fight.State, 30);

        Assert.Equal(80, fight.Enemy0.Hp);
        Assert.Equal(20, fight.Enemy0.HardenedShellDamageThisTurn);
    }

    /// <summary>
    /// A hit that kills counts only the HP the creature had (`LoseHpInternal` reports
    /// `currentHp - CurrentHp`), though the overkill is no concern of the tally either way.
    /// </summary>
    [Fact]
    public void ALethalHitCountsTheHpTheCreatureHad()
    {
        var fight = Fight.Hand().Enemy(hp: 8, buffs: Shell);

        CardEffects.DealDamage(fight.State, 50);

        Assert.Equal(0, fight.Enemy0.Hp);
        Assert.Equal(8, fight.Enemy0.HardenedShellDamageThisTurn);
    }

    /// <summary>
    /// `BeforeSideTurnStart` zeroes the tally and leaves the Amount: the player's turn end
    /// hands the colony its next turn's 20, and the turn after that starts fresh again.
    /// </summary>
    [Fact]
    public void TheTurnBoundaryZeroesTheTallyAndKeepsTheCap()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SkulkingColony).PlayerHp(200);
        var colony = fight.Enemy0;
        colony.Block = 0;
        CardEffects.DealDamage(fight.State, 50);
        Assert.Equal(20, colony.HardenedShellDamageThisTurn);

        fight.EndTurn();

        Assert.Equal(20, BuffSystem.Get(colony.Buffs, BuffId.HardenedShell));
        Assert.Equal(0, colony.HardenedShellDamageThisTurn);
    }

    /// <summary>
    /// The enemy phase is one pool too: poison spends it (30 takes 20), so the Flame Barrier
    /// the colony's own attack sets off afterwards lands nothing. With the old representation
    /// the poison tick dropped the shell and the whole 30 of retaliation landed.
    /// </summary>
    [Fact]
    public void ThePoolIsSharedBetweenAPoisonTickAndTheRetaliationThatFollows()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SkulkingColony).PlayerHp(200);
        var colony = fight.Enemy0;
        colony.Block = 0;
        int hp = colony.Hp;
        BuffSystem.Apply(colony.Buffs, BuffId.Poison, 30);
        fight.PlayerBuff(BuffId.FlameBarrier, 30);

        fight.EndTurn();

        Assert.Equal(hp - 20, colony.Hp);
    }

    /// <summary>
    /// A search or a rollout copies the state mid-turn: the clone has to carry the tally, or
    /// its colony would have a fresh shell the original's has already spent.
    /// </summary>
    [Fact]
    public void ACloneCarriesTheTally()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SkulkingColony);
        fight.Enemy0.Block = 0;
        CardEffects.DealDamage(fight.State, 50);
        int hp = fight.Enemy0.Hp;

        var copy = fight.State.Clone();
        CardEffects.DealDamage(copy, 50);

        Assert.Equal(20, copy.Enemies.First().HardenedShellDamageThisTurn);
        Assert.Equal(hp, copy.Enemies.First().Hp);
    }
}
