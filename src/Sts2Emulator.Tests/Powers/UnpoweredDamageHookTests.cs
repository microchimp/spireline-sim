using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CreatureCmd.Damage` is the game's one damage pipeline, and each hook decides for itself
/// whether the damage's props matter. Inferno's `AfterDamageReceived` reads none, so it
/// answers an enemy's Thorns or a Constrict tick as much as a self-damage card; Hardened
/// Shell, Slumber and the rest read none either, so a poison tick or Flame Barrier's
/// retaliation trips them as a card does. The Static audit tests (Audit/) pin the
/// enemy-side hooks one by one against the unpowered helper.
/// </summary>
public class UnpoweredDamageHookTests
{
    /// <summary>
    /// Strike into a Thorns Toadpole with Inferno up (Speedtest Q2J7LXCT99, step 5). Thorns
    /// retaliates in `BeforeDamageReceived`, before the Strike lands; the 2 the player takes
    /// is unblocked damage on their own turn, so Inferno hits every enemy for 6, the struck
    /// Toadpole included -- which the Strike then finishes. Game: one Toadpole dead, the
    /// other at 11.
    /// </summary>
    [Fact]
    public void InfernoBurnsWhenAnEnemysThornsHitsThePlayer()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Enemy(hp: 12, defId: KE.Toadpole, buffs: new BuffState(BuffId.Thorns, 2, 0))
            .Enemy(hp: 17, defId: KE.Toadpole)
            .PlayerHp(60)
            .PlayerBuff(BuffId.Inferno, 6);
        var struck = fight.Enemy0;
        var other = fight.Enemy1;

        fight.Play(0, target: 0);

        Assert.Equal(58, fight.State.PlayerHp);
        Assert.Equal(0, struck.Hp);
        Assert.Equal(11, other.Hp);
    }

    /// <summary>
    /// `CreatureCmd.Damage` skips a target that is already dead before `ModifyDamage` and
    /// Thorns. Thunderclap walks both Toadpoles; the first one's Thorns (2) sets off Inferno,
    /// whose burst kills the second before its turn. Game: the corpse runs no Thorns, so one
    /// burst, not two -- player 58, the first Toadpole 30 - 6 - 4 = 20.
    /// </summary>
    [Fact]
    public void AnAoeSkipsAnEnemyThatAnEarlierTargetsInfernoBurstKilled()
    {
        var fight = Fight
            .Hand(Card(IC.Thunderclap))
            .Enemy(hp: 30, defId: KE.Toadpole, buffs: new BuffState(BuffId.Thorns, 2, 0))
            .Enemy(hp: 5, defId: KE.Toadpole, buffs: new BuffState(BuffId.Thorns, 2, 0))
            .PlayerHp(60)
            .PlayerBuff(BuffId.Inferno, 6);
        var first = fight.Enemy0;
        var second = fight.Enemy1;

        fight.Play(0);

        Assert.Equal(58, fight.State.PlayerHp);
        Assert.Equal(20, first.Hp);
        Assert.True(second.Hp <= 0);
    }

    /// <summary>
    /// The unpowered sibling: Omnislice's splash is one `CreatureCmd.Damage` over the other
    /// hittable enemies, and it is the unpowered hit Thorns answers. The first splashed
    /// Toadpole's Thorns sets off Inferno, which kills the second; the corpse is skipped.
    /// Game: player 58, the struck enemy 50 - 8 - 6 = 36, the first Toadpole 30 - 6 - 8 = 16.
    /// </summary>
    [Fact]
    public void OmnislicesSplashSkipsAnEnemyThatAnInfernoBurstKilled()
    {
        var fight = Fight
            .Hand(Card(CL.Omnislice))
            .Enemy(hp: 50)
            .Enemy(hp: 30, defId: KE.Toadpole, buffs: new BuffState(BuffId.Thorns, 2, 0))
            .Enemy(hp: 5, defId: KE.Toadpole, buffs: new BuffState(BuffId.Thorns, 2, 0))
            .PlayerHp(60)
            .PlayerBuff(BuffId.Inferno, 6);
        var struck = fight.State.Enemies[0];
        var thorny = fight.State.Enemies[1];
        var burnt = fight.State.Enemies[2];

        fight.Play(0, target: 0);

        Assert.Equal(58, fight.State.PlayerHp);
        Assert.Equal(36, struck.Hp);
        Assert.Equal(16, thorny.Hp);
        Assert.True(burnt.Hp <= 0);
    }

    /// <summary>
    /// Echoing Slash counts `WasTargetKilled` over its own volley's results, and a target
    /// skipped as already dead has no result. The second enemy dies to the burst the first
    /// one's Thorns set off, not to the volley, so there is no second volley: the first
    /// enemy ends at 30 - 6 - 10 = 14 and the player took one Thorns.
    /// </summary>
    [Fact]
    public void EchoingSlashDoesNotCountAnEnemyItsVolleySkippedAsDead()
    {
        var fight = Fight
            .Hand(Card(SI.EchoingSlash))
            .Energy(1)
            .Enemy(hp: 30, defId: KE.Toadpole, buffs: new BuffState(BuffId.Thorns, 2, 0))
            .Enemy(hp: 5)
            .PlayerHp(60)
            .PlayerBuff(BuffId.Inferno, 6);
        var first = fight.Enemy0;
        var second = fight.Enemy1;

        fight.Play(0);

        Assert.Equal(58, fight.State.PlayerHp);
        Assert.Equal(14, first.Hp);
        Assert.True(second.Hp <= 0);
    }

    /// <summary>
    /// The same hook on a sibling route: `ConstrictPower.AfterSideTurnEnd` damages the
    /// player while the side is still the player's, so Inferno answers it.
    /// </summary>
    [Fact]
    public void InfernoBurnsWhenConstrictHitsThePlayerAtTheTurnsEnd()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 100)
            .PlayerHp(60)
            .PlayerBuff(BuffId.Inferno, 6)
            .PlayerBuff(BuffId.Constrict, 3);
        var enemy = fight.Enemy0;

        fight.EndTurn();

        Assert.Equal(94, enemy.Hp);
    }

    /// <summary>
    /// The player's own Poison: `PoisonPower.AfterSideTurnStart` deals the tick through
    /// `CreatureCmd.Damage(Unblockable | Unpowered)` on the player's side turn, so Inferno
    /// answers it like a self-damage card. The tick wrote HP directly and set nothing off.
    /// </summary>
    [Fact]
    public void InfernoBurnsWhenThePlayersPoisonTicks()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 100)
            .PlayerHp(60)
            .PlayerBuff(BuffId.Inferno, 6)
            .PlayerBuff(BuffId.Poison, 4);
        var enemy = fight.Enemy0;

        fight.EndTurn();

        Assert.Equal(56, fight.State.PlayerHp);
        Assert.Equal(94, enemy.Hp);
        Assert.Equal(3, fight.PlayerBuffAmount(BuffId.Poison));
    }

    /// <summary>
    /// The tick lands in `AfterSideTurnStart`, after `DemonTongue.BeforeSideTurnStart` has
    /// cleared the once-per-turn flag: the tick is the turn's first unblocked hit on the
    /// player's own side, so it is healed back and the flag stays spent, and a self-damage
    /// card later that turn is not healed.
    /// </summary>
    [Fact]
    public void DemonTongueHealsThePoisonTickAndNothingElseThatTurn()
    {
        var fight = Fight
            .WithRelics(RelicEffects.DemonTongue)
            .Enemy(hp: 100)
            .PlayerHp(60)
            .PlayerBuff(BuffId.Poison, 4);

        fight.EndTurn();
        Assert.Equal(60, fight.State.PlayerHp);

        CardEffects.LoseHp(fight.State, 5);
        Assert.Equal(55, fight.State.PlayerHp);
    }

    /// <summary>
    /// `PoisonPower`'s tick is `CreatureCmd.Damage(Unblockable | Unpowered)`, and
    /// `HardenedShellPower.ModifyHpLostBeforeOstyLate` reads no props: 30 poison takes 20
    /// off a Skulking Colony, not 30.
    /// </summary>
    [Fact]
    public void HardenedShellCapsAPoisonTick()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SkulkingColony);
        var colony = fight.Enemy0;
        int hp = colony.Hp;
        BuffSystem.Apply(colony.Buffs, BuffId.Poison, 30);

        fight.EndTurn();

        Assert.Equal(hp - 20, colony.Hp);
    }

    /// <summary>
    /// Flame Barrier's retaliation is the player's unpowered damage, landing in the ENEMY
    /// phase, and `HardenedShellPower.BeforeSideTurnStart` refills the shell for that side's
    /// turn too: a shell the player's turn spent to nothing caps the retaliation at a fresh
    /// 20, where it used to land whole.
    /// </summary>
    [Fact]
    public void FlameBarrierDrawsOnAShellRefilledForTheEnemyTurn()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.SkulkingColony).PlayerHp(200);
        var colony = fight.Enemy0;
        colony.Block = 0;
        int hp = colony.Hp;
        CardEffects.DealDamage(fight.State, 50);
        Assert.Equal(hp - 20, colony.Hp);
        fight.PlayerBuff(BuffId.FlameBarrier, 50);

        fight.EndTurn();

        Assert.Equal(hp - 40, colony.Hp);
    }

    /// <summary>
    /// The hooks that ask `props.IsPoweredAttack()` stay a card's: unpowered damage arms no
    /// Curl Up and strips no Flutter.
    /// </summary>
    [Fact]
    public void UnpoweredDamageLeavesThePoweredOnlyHooksAlone()
    {
        var fight = Fight
            .Hand()
            .Enemy(
                hp: 100,
                buffs: [new BuffState(BuffId.CurlUp, 5, 0), new BuffState(BuffId.Flutter, 1, 0)]
            );

        CardEffects.DealUnpoweredDamageToEnemy(fight.State, fight.Enemy0, 10);

        Assert.Equal(90, fight.Enemy0.Hp);
        Assert.False(fight.Enemy0.CurlUpArmed);
        Assert.Equal(1, fight.EnemyBuffAmount(BuffId.Flutter));
    }
}
