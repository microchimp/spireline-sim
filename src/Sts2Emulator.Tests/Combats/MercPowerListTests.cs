using System;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// The powers the Gremlin Merc and its Fat Gremlin carry show up in the power list, as in the game.
/// </summary>
/// <remarks>
/// <c>GremlinMerc.AfterAddedToRoom</c> applies <c>SurprisePower(1)</c> and then one
/// <c>ThieveryPower(20)</c> per player, and <c>SurprisePower.AfterDeath</c> gives the Fat
/// Gremlin a <c>HeistPower</c> holding what the Merc stole (<c>PowerCmd.Apply</c> adds nothing
/// for an amount of 0). The emulator kept the steal rate as a literal and the stolen gold as
/// <c>EnemyState</c> fields, so neither power was ever in a state's power list: every state of
/// every Merc fight differed from the game's at its first look.
/// </remarks>
public class MercPowerListTests
{
    private static CombatState MercFight(int ascension = 10)
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN", ascension);
        engine.StartCombat(
            RunConstants.StarterDeckIds,
            RunConstants.GremlinMercEncounterId,
            [],
            playerHp: 80,
            playerMaxHp: 80,
            potionIds: [],
            playerGold: 99
        );
        return engine.State.ActiveCombat!;
    }

    private static EnemyState Merc(CombatState state) =>
        state.Enemies.Single(enemy => enemy.DefId == KE.GremlinMerc);

    /// <summary>The Merc's list on the first state: Surprise and Thievery, the steal rate 20.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(10)]
    public void TheMercOpensWithSurpriseAndThievery(int ascension)
    {
        var merc = Merc(MercFight(ascension));

        Assert.Equal(1, BuffSystem.Get(merc.Buffs, BuffId.Surprise));
        Assert.Equal(20, BuffSystem.Get(merc.Buffs, BuffId.Thievery));
        Assert.Equal(0, BuffSystem.Get(merc.Buffs, BuffId.Heist));
    }

    /// <summary>
    /// <c>ThieveryPower.Amount</c> is the rate; what it has taken goes into its GoldVar, so a
    /// steal moves the gold and <c>StolenGold</c> and leaves the power at 20.
    /// </summary>
    [Fact]
    public void AStealLeavesTheThieveryRateAlone()
    {
        var state = MercFight();
        var merc = Merc(state);
        state.PlayerBlock = 99;

        EnemyAI.ExecuteIntent(merc, state, new Random(0));
        EnemyAI.ExecuteIntent(merc, state, new Random(0));

        Assert.Equal(40, merc.StolenGold);
        Assert.Equal(20, BuffSystem.Get(merc.Buffs, BuffId.Thievery));
    }

    private static CombatState KillTheMerc(int stolen)
    {
        var state = MercFight();
        var merc = Merc(state);
        merc.Hp = 1;
        merc.StolenGold = stolen;
        state.Hand.Clear();
        state.Hand.Add(new CardInstance(IC.StrikeIronclad, false));
        state.Energy = 3;

        CombatEngine.Step(state, 0, new Random(0));
        return state;
    }

    /// <summary>The Fat Gremlin arrives holding a Heist of what the Merc took, beside HeistGold.</summary>
    [Fact]
    public void TheFatGremlinHoldsAHeistOfTheStolenGold()
    {
        var state = KillTheMerc(stolen: 40);

        var fat = state.Enemies.Single(enemy => enemy.DefId == 28 && enemy.Hp > 0);
        Assert.Equal(40, BuffSystem.Get(fat.Buffs, BuffId.Heist));
        Assert.Equal(40, fat.HeistGold);
        // The Sneaky Gremlin takes no part in it.
        var sneaky = state.Enemies.Single(enemy => enemy.DefId == 78 && enemy.Hp > 0);
        Assert.Equal(0, BuffSystem.Get(sneaky.Buffs, BuffId.Heist));
    }

    /// <summary>A Merc that stole nothing leaves a Fat Gremlin with no Heist at all.</summary>
    [Fact]
    public void ANothingStolenMercLeavesNoHeist()
    {
        var state = KillTheMerc(stolen: 0);

        var fat = state.Enemies.Single(enemy => enemy.DefId == 28 && enemy.Hp > 0);
        Assert.Equal(0, BuffSystem.Get(fat.Buffs, BuffId.Heist));
        Assert.Equal(0, fat.HeistGold);
    }

    /// <summary>
    /// A Merc killed inside its own attack (Flame Barrier hitting back) never reaches the steal:
    /// <c>CreatureCmd.Kill</c> runs <c>SurprisePower.AfterDeath</c> and then strips the dead
    /// creature's powers, so the loop over its <c>ThieveryPower</c> instances is empty. The Fat
    /// Gremlin's Heist is what the Merc took before that move, and the player keeps the rest.
    /// </summary>
    [Fact]
    public void AMercKilledByItsOwnAttackStealsNothingMore()
    {
        var state = MercFight();
        var merc = Merc(state);
        merc.Hp = 1;
        merc.StolenGold = 40;
        state.PlayerGold = 99;
        state.PlayerBlock = 0;
        BuffSystem.Apply(state.PlayerBuffs, BuffId.FlameBarrier, 4);
        state.Hand.Clear();

        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(99, state.PlayerGold);
        var fat = state.Enemies.Single(enemy => enemy.DefId == 28 && enemy.Hp > 0);
        Assert.Equal(40, BuffSystem.Get(fat.Buffs, BuffId.Heist));
        Assert.Equal(40, fat.HeistGold);
    }

    /// <summary>
    /// <c>AttackCommand.Execute</c> asks <c>Attacker.IsDead</c> before every hit, so an attacker
    /// that a Flame Barrier killed on its first hit lands no second one. The Merc's double
    /// attack is one case; the helper every multi-hit enemy attack goes through is the mechanism.
    /// </summary>
    [Theory]
    [InlineData(KE.GremlinMerc)]
    [InlineData(1)]
    public void AnAttackerKilledMidAttackLandsNoFurtherHits(int defId)
    {
        var state = MercFight();
        state.Enemies.Clear();
        var enemy = new EnemyState
        {
            DefId = defId,
            Hp = 3,
            MaxHp = 40,
            CurrentIntent = new Intent(IntentType.Attack, 8, Hits: 3),
            Buffs = [],
        };
        state.Enemies.Add(enemy);
        state.PlayerBlock = 0;
        int hp = state.PlayerHp;
        BuffSystem.Apply(state.PlayerBuffs, BuffId.FlameBarrier, 4);

        EnemyAI.ExecuteIntent(enemy, state, new Random(0));

        Assert.True(enemy.Hp <= 0);
        Assert.Equal(hp - 8, state.PlayerHp);
    }

    /// <summary>The Heist is paid out once: the Fat Gremlin's death banks it and drops the power.</summary>
    [Fact]
    public void KillingTheFatGremlinSpendsTheHeist()
    {
        var state = KillTheMerc(stolen: 40);
        var fat = state.Enemies.Single(enemy => enemy.DefId == 28 && enemy.Hp > 0);
        fat.Hp = 1;
        state.Hand.Clear();
        state.Hand.Add(new CardInstance(IC.StrikeIronclad, false));
        state.Energy = 3;

        CombatEngine.Step(state, 0, new Random(0), targetEnemyIndex: state.Enemies.IndexOf(fat));

        Assert.Equal(40, state.StolenBackGold);
        Assert.Equal(0, fat.HeistGold);
        Assert.Equal(0, BuffSystem.Get(fat.Buffs, BuffId.Heist));
    }
}
