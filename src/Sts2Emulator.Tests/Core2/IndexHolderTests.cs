// The R7 checks before switch-on (docs/rebuild/MIGRATION_F08.md, "R7 checks before switch-on"; the F08 review,
// docs/reviews/2026-10-24-f08-migration.md, recommendations 3, 4 and 7 and its 1.3 A-D), each under the F08+F10 switch
// with the registered handlers: the play's target and Oblivion held by reference across the kernel's corpse removal, the
// in-move death, Curl Up's own card, and the Fogmog's primary rule.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class IndexHolderTests
{
    private const int Slug = 16;
    private const int Bash = 30;
    private const int Strike = 472;
    private const int Defend = 131;

    private static KernelState Bind(CombatState s, params Action<HandlerTable>[] extra) =>
        new Kernel(new KernelOptions([Switch.F08, Switch.F10]), () => ListenerOrderTests.Ids,
            [.. Sts2Emulator.Core2.Models.Registry.All, .. extra]).Bind(s);

    /// <summary>Three slugs at these HPs, the given hand, the first enemy turn not yet run.</summary>
    private static Fight Three(int a, int b, int c, params int[] hand) =>
        Fight.Hand([.. hand.Select(id => new CardInstance(id, false))]).Enemy(hp: a).Enemy(hp: b).Enemy(hp: c);

    [Fact]
    public void ACardsSecondTargetedEffectFollowsTheCreatureNotTheIndex()
    {
        // [A, B, C], Bash at B: the hit kills B and the kernel removes it, so index 1 is C. The game's Vulnerable goes to
        // CurrentTarget, the dead B (nobody); by index it went to C.
        var f = Three(20, 5, 20, Bash);
        var a = f.State.Enemies[0];
        var c = f.State.Enemies[2];
        Bind(f.State);
        f.Play(0, target: 1);
        Assert.Equal(2, f.State.Enemies.Count);
        Assert.Equal(0, BuffSystem.Get(c.Buffs, BuffId.Vulnerable));
        Assert.Equal(0, BuffSystem.Get(a.Buffs, BuffId.Vulnerable));
    }

    [Fact]
    public void OblivionIsPaidToTheCreatureNotTheSlotAfterAMidCardRemoval()
    {
        // [A, B, C(Oblivion 3)], Strike kills B: C moves to slot 1 before AfterCardPlayed; its own Oblivion pays it 3 Doom.
        var f = Three(20, 5, 20, Strike);
        var c = f.State.Enemies[2];
        BuffSystem.Apply(c.Buffs, BuffId.Oblivion, 3);
        Bind(f.State);
        f.Play(0, target: 1);
        Assert.Equal(2, f.State.Enemies.Count);
        Assert.Equal(3, BuffSystem.Get(c.Buffs, BuffId.Doom));
        Assert.Equal(0, BuffSystem.Get(f.State.Enemies[0].Buffs, BuffId.Doom));
    }

    [Fact]
    public void APositionalDeathCheckDoesNotRunUnderTheSwitch()
    {
        // [A, B, C], Strike kills B: the death is answered inside the hit; the old positional pairing (HandleEnemyDeaths,
        // enemyHpsBefore[i] against the shifted roster) is off whole, so C, now in B's slot and alive, is not "newly dead"
        // and B is answered once (one kill call for the one Damage call).
        var f = Three(20, 5, 20, Strike);
        var k = Bind(f.State);
        f.Play(0, target: 1);
        Assert.Equal(1, k.Commands.KillCalls);
        Assert.Equal(20, f.State.Enemies[1].Hp);
        Assert.Equal([20, 20], f.State.Enemies.Select(e => e.Hp));
    }

    /// <summary>A stand-in for the player's ThornsPower (batch L3a ports the real one): the monster performing its move
    /// takes 5 unpowered damage when the player is hit.</summary>
    private static HookStep StandInThorns(HookCtx c, ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? ctx)
    {
        if (c.World.Moving is { Hp: > 0 } mover)
            CardEffects.DealUnpoweredDamageToEnemy(c.OldCombat, mover, 5);
        return HookStep.Done;
    }

    private static void RegisterStandInThorns(HandlerTable t) =>
        t.Model(ListenerKind.Power, "ThornsPower")
            .On(HookId.AfterDamageReceived, (HookEvent<AfterDamageReceivedArgs>)StandInThorns);

    [Fact]
    public void AMoverKilledInItsOwnMoveGetsNoPostAttackBookkeeping()
    {
        // [A, mover, C]: the mover's attack meets the player's Thorns and it dies inside its move. The game's move ends
        // there: no MoveIndex step and no Ritual Strength (the forced move a death sets, ABOUT_TO_BLOW, then stands).
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        var mover = new EnemyState { DefId = Slug, Hp = 3, MaxHp = 3, CurrentIntent = new Intent(IntentType.Attack, 5) };
        BuffSystem.Apply(mover.Buffs, BuffId.Ritual, 2);
        s.Enemies = [new EnemyState { DefId = Slug, Hp = 20, MaxHp = 20 }, mover, new EnemyState { DefId = Slug, Hp = 20, MaxHp = 20 }];
        BuffSystem.Apply(s.PlayerBuffs, BuffId.Thorns, 5);
        var k = Bind(s, RegisterStandInThorns);
        int moveIndex = mover.MoveIndex;
        CreatureCmd.PerformMove(k, OldCreature.Of(mover), () => EnemyAI.ExecuteIntent(mover, s, new Random(1)));
        Assert.True(mover.Hp <= 0);
        Assert.Equal(moveIndex, mover.MoveIndex);
        Assert.Equal(0, BuffSystem.Get(mover.Buffs, BuffId.Strength));
        Assert.False(k.World!.MoverDied); // the scope handed the outer state back
    }

    [Fact]
    public void CurlUpPaysOnlyAtTheCardThatArmedIt()
    {
        // [A, louse, C]: a powered hit from a card that is not being played arms Curl Up; a Defend played next is not that
        // card, so the game's CurlUpPower.AfterCardPlayed pays nothing (the old flag paid at the next card's end).
        var f = Three(20, 20, 20, Defend);
        var louse = f.State.Enemies[1];
        BuffSystem.Apply(louse.Buffs, BuffId.CurlUp, 5);
        var k = Bind(f.State);
        CreatureCmd.Damage(k, [OldCreature.Of(louse)], 1, ValueProps.PoweredAttack, OldCreature.Player, new CardHandle(9999));
        Assert.True(louse.CurlUpArmed);
        f.Play(0);
        Assert.Equal(0, louse.Block);
        Assert.Equal(5, BuffSystem.Get(louse.Buffs, BuffId.CurlUp));
    }

    [Fact]
    public void AnIllusionHolderIsNotAPrimaryEnemy()
    {
        // The game's IllusionPower.AfterApplied gives its owner MinionPower: the Fogmog's Eye With Teeth is secondary, so
        // the Fogmog's death is the last primary's and the fight ends (CombatManager.IsEnding).
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        var fogmog = new EnemyState { DefId = Slug, Hp = 30, MaxHp = 30 };
        var eye = new EnemyState { DefId = Slug, Hp = 6, MaxHp = 6 };
        BuffSystem.Apply(eye.Buffs, BuffId.Illusion, 1);
        s.Enemies = [new EnemyState { DefId = Slug, Hp = 10, MaxHp = 10 }, fogmog, eye];
        var k = Bind(s);
        Assert.False(OldCreatures.IsPrimaryEnemy(k.World!, OldCreature.Of(eye)));
        Assert.True(OldCreatures.IsPrimaryEnemy(k.World!, OldCreature.Of(fogmog)));
    }
}
