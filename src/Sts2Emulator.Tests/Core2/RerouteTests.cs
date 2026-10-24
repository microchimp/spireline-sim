// The old engine's damage, kill and win-check sites rerouted onto Core2 under the F08+F10 switch (docs/rebuild/
// MIGRATION_F08.md, round R4). No model handler is registered: the old entry points run the game-shaped commands with an
// empty handler set, so only the commands' own behaviour shows (deaths inside the hit, removal, the win check's point).
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class RerouteTests
{
    private static readonly Switch[] DamageOnly = [Switch.F08, Switch.F10];
    private const int Slug = 16;

    private static CombatState Combat(params int[] enemyHp)
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = enemyHp.Select(hp => new EnemyState { DefId = Slug, Hp = hp, MaxHp = hp }).ToList();
        return s;
    }

    private static KernelState On(CombatState s) =>
        new Kernel(new KernelOptions(DamageOnly), () => ListenerOrderTests.Ids, []).Bind(s);

    private sealed class Host : ICombatEndHost
    {
        public List<bool> Ends { get; } = [];
        public void EndCombat(KernelState k, bool playerWon) => Ends.Add(playerWon);
    }

    [Fact]
    public void CardDamageGoesThroughCore2()
    {
        var s = Combat(20, 20);
        var k = On(s);
        CardEffects.DealDamage(s, 6);
        Assert.Equal(14, s.Enemies[0].Hp);
        Assert.Equal(20, s.Enemies[1].Hp);
        Assert.Equal(1, k.Commands.KillCalls); // one CreatureCmd.Damage call, its Kill hook point reached once
    }

    [Fact]
    public void WithTheSwitchOffTheOldPathRuns()
    {
        var s = Combat(5, 20);
        CardEffects.DealDamage(s, 10);
        Assert.Null(s.Kernel);
        Assert.Equal(2, s.Enemies.Count); // the old engine sweeps its corpse after the card, not inside the hit
        Assert.Equal(0, s.Enemies[0].Hp);
    }

    [Fact]
    public void TheDeathIsInsideTheHitAndTheCorpseGoneBeforeTheNextSelection()
    {
        // G2891c25e0d (Headbutt): game: CreatureCmd.Damage's Kill removes the slug before the card's next effect, so the
        // next pick (the screen, a second hit's first enemy) sees only the survivor.
        var s = Combat(5, 20);
        var k = On(s);
        var survivor = s.Enemies[1];
        CardEffects.DealDamage(s, 10);
        Assert.Single(s.Enemies);
        Assert.Same(survivor, s.Enemies[0]);
        CardEffects.DealDamage(s, 3);
        Assert.Equal(17, survivor.Hp);
        Assert.Equal(2, k.Commands.KillCalls);
    }

    [Fact]
    public void PlayerDamageAndHpLossGoThroughCore2()
    {
        var s = Combat(20);
        s.PlayerBlock = 4;
        var k = On(s);
        CardEffects.DealDamageToPlayer(s, 10);
        Assert.Equal(0, s.PlayerBlock);
        Assert.Equal(44, s.PlayerHp);
        CardEffects.LoseHp(s, 7);
        Assert.Equal(37, s.PlayerHp);
        EnemyAI.DealAttackDamageForTests(s.Enemies[0], s, 9);
        Assert.Equal(28, s.PlayerHp);
        Assert.Equal(3, k.Commands.KillCalls);
    }

    [Fact]
    public void TheWinCheckWaitsForTheBoundary()
    {
        // game: Kill does not check the win; CombatManager.CheckWinCondition does, after the action (K4's boundary).
        var s = Combat(5);
        var k = On(s);
        CardEffects.DealDamage(s, 10);
        Assert.Empty(s.Enemies);
        Assert.True(CombatEnd.IsInProgress(k));
        Assert.True(CombatEngine.IsOverOrEnding(s)); // the old reading is Core2's: ending, not over
        var host = new Host();
        CombatEnd.AfterAction(k, host);
        Assert.Equal([true], host.Ends);
        Assert.False(CombatEnd.IsInProgress(k));
    }

    [Fact]
    public void ThePlayersDeathIsALossAtTheNextCheck()
    {
        var s = Combat(20);
        var k = On(s);
        CardEffects.LoseHp(s, 60);
        Assert.Equal(0, s.PlayerHp);
        Assert.True(CombatEngine.IsAboutToLose(s));
        Assert.Equal(1, k.Commands.LossCalls);
        var host = new Host();
        CombatEnd.AfterAction(k, host);
        Assert.Equal([false], host.Ends);
    }

    [Fact]
    public void CleaveIsOneDamageCallAndItsDeathResolvesAfterBothAreHit()
    {
        // game: AttackCommand.Execute calls Damage once per hit over every live target, and Damage calls Kill once after
        // the whole list, so the slug Cleave kills dies only once the second slug has taken its share.
        var s = Combat(5, 20);
        var k = On(s);
        var hpOfTheSecondAtTheKill = new List<int>();
        k.Commands.KillStandIn = (_, killed) =>
        {
            if (killed.Count > 0)
                hpOfTheSecondAtTheKill.Add(s.Enemies[1].Hp);
        };
        CardEffects.DealDamageToAll(s, 8);
        Assert.Equal([12], hpOfTheSecondAtTheKill);
        Assert.Equal(1, k.Commands.KillCalls);
    }

    [Fact]
    public void TwinStrikeIsOneCommandOfTwoHitsAndADeadTargetEatsTheRest()
    {
        var s = Combat(3, 20);
        var k = On(s);
        var second = s.Enemies[1];
        CardEffects.DealDamageMultiHit(s, 5, 2, new Random(1));
        Assert.Single(s.Enemies);
        Assert.Equal(20, second.Hp); // as the old engine: the second hit is not rolled onto the survivor
        Assert.Equal(1, k.Commands.KillCalls);

        var t = Combat(20);
        var kt = On(t);
        CardEffects.DealDamageMultiHit(t, 5, 2, new Random(1));
        Assert.Equal(10, t.Enemies[0].Hp);
        Assert.Equal(2, kt.Commands.KillCalls);
    }

    [Fact]
    public void WhirlwindHitsEveryLiveEnemyXTimesOneDamageCallPerHit()
    {
        var s = Combat(12, 30);
        var k = On(s);
        var second = s.Enemies[1];
        CardEffects.DealDamageToAllMultiHit(s, 5, 3);
        Assert.Equal(15, second.Hp);
        Assert.Single(s.Enemies);
        Assert.Equal(3, k.Commands.KillCalls);
    }

    [Fact]
    public void SwordBoomerangDrawsTheSameStreamOncePerHitOverTheLiveEnemies()
    {
        bool sawTheDeath = false;
        for (int seed = 1; seed <= 20; seed++)
        {
            var old = Combat(1, 40, 40);
            old.TargetRng = new CountingRandom(seed);
            var oldEnemies = old.Enemies.ToList();
            CardEffects.DealDamageToRandomEnemiesMultiHit(old, 3, 4, new Random(seed));

            var s = Combat(1, 40, 40);
            s.TargetRng = new CountingRandom(seed);
            var enemies = s.Enemies.ToList();
            On(s);
            CardEffects.DealDamageToRandomEnemiesMultiHit(s, 3, 4, new Random(seed));

            Assert.Equal(oldEnemies.Select(e => e.Hp), enemies.Select(e => e.Hp));
            Assert.Equal(old.TargetRng!.CallCount, s.TargetRng!.CallCount);
            Assert.Equal(4, s.TargetRng.CallCount);
            sawTheDeath |= enemies[0].Hp == 0;
        }

        Assert.True(sawTheDeath);
    }

    [Fact]
    public void SkittishAnswersEachCardAttackOnceAfterItsLastHit()
    {
        // F06 off: the card attack's AfterAttack point (Skittish) runs once per AttackCmd, through its bridge.
        var s = Combat(30);
        On(s);
        var e = s.Enemies[0];
        BuffSystem.Apply(e.Buffs, BuffId.Skittish, 4);
        CardEffects.DealDamageMultiHit(s, 5, 3, new Random(1));
        Assert.Equal((15, 4), (e.Hp, e.Block)); // one command: the block comes after all three hits

        var t = Combat(30);
        On(t);
        var g = t.Enemies[0];
        BuffSystem.Apply(g.Buffs, BuffId.Skittish, 4);
        CardEffects.DealDamageAsSeparateAttacks(t, 5, 3, new Random(1));
        Assert.Equal((19, 0), (g.Hp, g.Block)); // three commands: the second hit lands on the first's block
    }

    [Fact]
    public void OmnisliceSplashIsOneDamageCallSoThornsAnswersBothBeforeTheDeath()
    {
        // game: Omnislice.OnPlay: the aimed hit (Move), then ONE CreatureCmd.Damage over the target's other hittable
        // teammates for TotalDamage + OverkillDamage with Unpowered | Move and the card as source.
        // ThornsPower.BeforeDamageReceived answers a powered attack or an Omnislice: both splashed enemies answer, and
        // the one the splash kills dies only after the other has been hit (Damage's one Kill after its list).
        const int uid = 77;
        var s = Combat(40, 5, 20);
        s.Enemies[1].Buffs.Add(new BuffState(BuffId.Thorns, 3));
        s.Enemies[2].Buffs.Add(new BuffState(BuffId.Thorns, 3));
        var second = s.Enemies[2];
        KernelState? kk = null;
        var answered = new List<int>();
        var k = kk = new Kernel(new KernelOptions(DamageOnly), () => ListenerOrderTests.Ids,
        [
            t => t.Model(ListenerKind.Power, "ThornsPower").On(HookId.BeforeDamageReceived,
                (HookEvent<BeforeDamageReceivedArgs>)((HookCtx _, ListenerRef self, in BeforeDamageReceivedArgs a, ChoiceCtx? c) =>
                {
                    if (a.Target.Index == self.Instance >> 8 && a.Dealer is not null
                        && (a.Props.IsPoweredAttack() || a.CardSource == new CardHandle(uid)))
                    {
                        answered.Add(a.Props.Bits);
                        CreatureCmd.Damage(kk!, [OldCreature.Player], 3m, OldEntry.Unpowered, null, null);
                    }

                    return HookStep.Done;
                })),
        ]).Bind(s);
        var atTheKill = new List<int>();
        k.Commands.KillStandIn = (_, killed) =>
        {
            if (killed.Count > 0)
                atTheKill.Add(second.Hp);
        };
        var card = new CardInstance(CL.Omnislice, false) { Uid = uid };
        using (k.World!.Flying(card))
            CardEffects.Apply(GeneratedData.Cards.Get(CL.Omnislice), false, s, new Random(1), card);
        int splash = ValueProps.Unpowered | ValueProps.Move;
        Assert.Equal([splash, splash], answered);
        Assert.Equal(44, s.PlayerHp);
        Assert.Equal(32, s.Enemies[0].Hp);
        Assert.Equal([12], atTheKill);
    }

    [Fact]
    public void SeekingEdgeSweepIsOneAttackOverEveryEnemyAtTheBaseAmount()
    {
        // game: SovereignBlade.OnPlay with Seeking Edge: one AttackCommand targeting all opponents, so one Damage call
        // over both and the first enemy's death after the second is hit; Conqueror's doubling is its
        // ModifyDamageMultiplicative hook's (Core2/Models/Powers/ConquerorPower.cs, registered here), not the amount's.
        const int sovereignBlade = 448;
        var plain = Combat(5, 30);
        BuffSystem.Apply(plain.PlayerBuffs, BuffId.SeekingEdge, 1);
        CardEffects.Apply(GeneratedData.Cards.Get(sovereignBlade), false, plain, new Random(1),
            new CardInstance(sovereignBlade, false) { Uid = 5 });
        int blade = 30 - plain.Enemies[1].Hp;
        Assert.True(blade > 0);

        var s = Combat(5, 30);
        var k = WithModels(s);
        BuffSystem.Apply(s.PlayerBuffs, BuffId.SeekingEdge, 1);
        BuffSystem.Apply(s.Enemies[1].Buffs, BuffId.Conqueror, 1);
        var second = s.Enemies[1];
        var atTheKill = new List<int>();
        k.Commands.KillStandIn = (_, killed) =>
        {
            if (killed.Count > 0)
                atTheKill.Add(second.Hp);
        };
        PlayBlade(k, s);
        Assert.Equal(30 - 2 * blade, second.Hp);
        Assert.Equal([30 - 2 * blade], atTheKill);
        Assert.Equal(1, k.Commands.KillCalls);
    }

    private const int SovereignBlade = 448;

    private static KernelState WithModels(CombatState s) =>
        new Kernel(new KernelOptions(DamageOnly), () => ListenerOrderTests.Ids, Sts2Emulator.Core2.Models.Registry.All)
            .Bind(s);

    /// <summary>Sovereign Blade played as the game plays it: in flight (the Play pile), so its attack's card source.</summary>
    private static void PlayBlade(KernelState k, CombatState s)
    {
        var card = new CardInstance(SovereignBlade, false) { Uid = 5 };
        using (k.World!.Flying(card))
            CardEffects.Apply(GeneratedData.Cards.Get(SovereignBlade), false, s, new Random(1), card);
    }

    private static int PlainBlade()
    {
        var plain = Combat(40);
        CardEffects.Apply(GeneratedData.Cards.Get(SovereignBlade), false, plain, new Random(1),
            new CardInstance(SovereignBlade, false) { Uid = 5 });
        return 40 - plain.Enemies[0].Hp;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void ConquerorDoublesOnlyTheBladeOnItsOwnerUnderSeekingEdge(int conquered)
    {
        // game: ConquerorPower.ModifyDamageMultiplicative: 2 when the card source is Sovereign Blade, the damage a powered
        // attack and the target the power's owner. One AttackCmd over both enemies: only the Conqueror one is doubled.
        int blade = PlainBlade();
        Assert.True(blade > 0);
        var s = Combat(40, 40);
        var k = WithModels(s);
        BuffSystem.Apply(s.PlayerBuffs, BuffId.SeekingEdge, 1);
        BuffSystem.Apply(s.Enemies[conquered].Buffs, BuffId.Conqueror, 1);
        PlayBlade(k, s);
        Assert.Equal(40 - 2 * blade, s.Enemies[conquered].Hp);
        Assert.Equal(40 - blade, s.Enemies[1 - conquered].Hp);
    }

    [Theory]
    [InlineData(0, 2)]
    [InlineData(1, 1)]
    public void SovereignBladeAtItsTargetIsDoubledOnlyWhenTheTargetIsTheConqueror(int conquered, int factor)
    {
        // game: SovereignBlade.OnPlay without Seeking Edge: one AttackCommand at the target (the first enemy here).
        int blade = PlainBlade();
        var s = Combat(40, 40);
        var k = WithModels(s);
        BuffSystem.Apply(s.Enemies[conquered].Buffs, BuffId.Conqueror, 1);
        PlayBlade(k, s);
        Assert.Equal(40 - factor * blade, s.Enemies[0].Hp);
        Assert.Equal(40, s.Enemies[1].Hp);
    }

    /// <summary>ThornsPower.BeforeDamageReceived's condition as the game has it (a powered attack or an Omnislice), with
    /// every card source its owner is hit with recorded; with batch L2's InfernoPower handler, which took over the old
    /// player-side body Core2's damage command called (MIGRATION_F08.md, batch L2, item 1).</summary>
    private static KernelState WithThorns(CombatState s, int omnislice, List<CardHandle?> seen, List<int> answered) =>
        new Kernel(new KernelOptions(DamageOnly), () => ListenerOrderTests.Ids,
        [
            t => t.Model(ListenerKind.Power, "ThornsPower").On(HookId.BeforeDamageReceived,
                (HookEvent<BeforeDamageReceivedArgs>)((HookCtx c, ListenerRef self, in BeforeDamageReceivedArgs a,
                    ChoiceCtx? ctx) =>
                {
                    if (a.Target.Index != self.Instance >> 8)
                        return HookStep.Done;
                    seen.Add(a.CardSource);
                    if (a.Dealer is not null && (a.Props.IsPoweredAttack() || a.CardSource == new CardHandle(omnislice)))
                    {
                        answered.Add(a.Props.Bits);
                        CreatureCmd.Damage(c.K, [OldCreature.Player], 3m, OldEntry.Unpowered, null, null);
                    }

                    return HookStep.Done;
                })),
            Sts2Emulator.Core2.Models.Powers.InfernoPower.Register,
        ]).Bind(s);

    [Fact]
    public void OmnisliceIntoThornsWithInfernoAnswersOnlyTheCardsOwnHit()
    {
        // game: Thorns answers Omnislice's aimed hit (a powered attack); the 3 the player loses fires InfernoPower, whose
        // CreatureCmd.Damage over the hittable enemies passes no card source, so Thorns does not answer it. With the card
        // in flight as Inferno's source it would, and Inferno would fire again.
        const int uid = 81;
        var s = Combat(40, 30);
        s.PlayerTurn = true;
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Thorns, 3));
        s.PlayerBuffs.Add(new BuffState(BuffId.Inferno, 6));
        var seen = new List<CardHandle?>();
        var answered = new List<int>();
        var k = WithThorns(s, uid, seen, answered);
        var card = new CardInstance(CL.Omnislice, false) { Uid = uid };
        using (k.World!.Flying(card))
            CardEffects.Apply(GeneratedData.Cards.Get(CL.Omnislice), false, s, new Random(1), card);
        Assert.Equal([new CardHandle(uid), null], seen);
        Assert.Single(answered);
        Assert.Equal(47, s.PlayerHp);
    }

    [Fact]
    public void PowerRelicAndPotionListsHitWithNoCardSource()
    {
        // game: InfernoPower.AfterDamageReceived, MercuryHourglass.AfterPlayerTurnStart and ExplosiveAmpoule.OnUse each
        // call CreatureCmd.Damage over the hittable enemies with cardSource null, even with a card in flight.
        var s = Combat(60, 60);
        s.PlayerTurn = true;
        foreach (var e in s.Enemies)
            e.Buffs.Add(new BuffState(BuffId.Thorns, 1));
        s.PlayerBuffs.Add(new BuffState(BuffId.Inferno, 2));
        s.Relics.Add(new RelicInstance(RelicEffects.MercuryHourglass));
        var seen = new List<CardHandle?>();
        var answered = new List<int>();
        var k = WithThorns(s, -1, seen, answered);
        var card = new CardInstance(SovereignBlade, false) { Uid = 90 };
        using (k.World!.Flying(card))
        {
            CardEffects.AfterPlayerDamageReceived(s, 4, fromPlayedCard: true);
            Assert.Equal([null, null], seen);
            RelicEffects.ApplyAfterPlayerTurnStart(s, 2, new Random(1));
            Assert.Equal([null, null, null, null], seen);
            PotionEffects.Apply(21, s);
        }

        Assert.Equal(6, seen.Count);
        Assert.All(seen, x => Assert.Null(x));
        Assert.Empty(answered);
        Assert.Equal(60 - 2 - 3 - 10, s.Enemies[0].Hp);
    }

    [Fact]
    public void EchoingSlashCountsTheKillsOfEachVolleyFromItsResults()
    {
        // game: EchoingSlash.OnPlay: each volley is one Damage over the hittable enemies and adds a volley per result
        // that killed its target. 10 into 5, 5, 30: the first volley kills two, the third kills the last, the fourth
        // finds no one (no Damage call reaches Kill).
        var s = Combat(5, 5, 30);
        var k = On(s);
        CardEffects.Apply(GeneratedData.Cards.Get(SI.EchoingSlash), false, s, new Random(1),
            new CardInstance(SI.EchoingSlash, false) { Uid = 9 });
        Assert.Empty(s.Enemies);
        Assert.Equal(3, k.Commands.KillCalls);
    }

    [Fact]
    public void TearAsunderHitsOncePlusTheUnblockedHitsInTheHistory()
    {
        // game: TearAsunder's hit count is 1 + the DamageReceived entries on its owner with unblocked damage, which
        // Core2's Damage writes; the hits are one AttackCommand (one Damage call each).
        var s = Combat(60);
        var k = On(s);
        OldEntry.HitPlayer(k, 3, OldEntry.Unpowered, OldCreature.Of(s.Enemies[0]));
        OldEntry.HitPlayer(k, 3, OldEntry.Unpowered, OldCreature.Of(s.Enemies[0]));
        int before = k.Commands.KillCalls;
        CardEffects.Apply(GeneratedData.Cards.Get(IC.TearAsunder), false, s, new Random(1),
            new CardInstance(IC.TearAsunder, false) { Uid = 3 });
        Assert.Equal(3, k.Commands.KillCalls - before);
        Assert.Equal(60 - 3 * 5, s.Enemies[0].Hp);
    }

    [Fact]
    public void UnpoweredDamageSkipsTheOldCapsUnderTheSwitch()
    {
        // The Hard to Kill cap is ModifyDamage's hook (no handler registered here), not the old body before the gate.
        var s = Combat(30);
        On(s);
        var e = s.Enemies[0];
        BuffSystem.Apply(e.Buffs, BuffId.HardToKill, 2);
        CardEffects.DealUnpoweredDamageToEnemy(s, e, 9);
        Assert.Equal(21, e.Hp);
    }

    [Fact]
    public void TheHpCommandsTakeTheOldWrites()
    {
        var s = Combat(20);
        s.PlayerHp = 30;
        On(s);
        CardEffects.HealPlayer(s, 5);
        Assert.Equal(35, s.PlayerHp);
        CardEffects.GainMaxHp(s, 4);
        Assert.Equal((54, 39), (s.PlayerMaxHp, s.PlayerHp));
        CardEffects.LoseMaxHp(s, 20);
        Assert.Equal((34, 34), (s.PlayerMaxHp, s.PlayerHp));
    }

    /// <summary>Every AfterDamageGiven the player's listener (an Inferno buff) is asked, by its dealer: the player, none
    /// or another creature.</summary>
    private static KernelState WithDealerLog(CombatState s, List<string> dealers) =>
        new Kernel(new KernelOptions(DamageOnly), () => ListenerOrderTests.Ids,
        [
            t => t.Model(ListenerKind.Power, "InfernoPower").On(HookId.AfterDamageGiven,
                (HookEvent<AfterDamageGivenArgs>)((HookCtx c, ListenerRef self, in AfterDamageGivenArgs a, ChoiceCtx? ctx) =>
                {
                    dealers.Add(a.Dealer is not { } d ? "none" : d == c.Owner(self) ? "player" : "other");
                    return HookStep.Done;
                })),
        ]).Bind(s);

    [Fact]
    public void PoisonTickHasNoDealerSoNoAfterDamageGivenForThePlayerAndNoDeadDealerSkip()
    {
        // game: PoisonPower's tick (as StranglePower's, HauntPower's and DemisePower's Damage) passes a null dealer and a
        // null source: no listener sees the player as the dealer, and a dead player does not void the tick.
        var fight = Fight.Hand(new CardInstance(CL.Omnislice, false)).Energy(0).Enemy(hp: 60);
        fight.State.PlayerBuffs.Add(new BuffState(BuffId.Inferno, 1));
        fight.State.PlayerHp = 999;
        BuffSystem.Apply(fight.Enemy0.Buffs, BuffId.Poison, 5);
        var dealers = new List<string>();
        WithDealerLog(fight.State, dealers);
        fight.EndTurn();
        Assert.Equal(55, fight.Enemy0.Hp);
        Assert.Contains("none", dealers);
        Assert.DoesNotContain("player", dealers);

        // The same entry with a dead player: the tick lands (no dealer to be dead); with the player as dealer it would be
        // one empty result.
        var s = Combat(30);
        s.PlayerHp = 0;
        WithDealerLog(s, []);
        CardEffects.DealUnblockableUnpoweredDamageToEnemy(s, s.Enemies[0], 4, noDealer: true);
        Assert.Equal(26, s.Enemies[0].Hp);
        CardEffects.DealUnblockableUnpoweredDamageToEnemy(s, s.Enemies[0], 4);
        Assert.Equal(26, s.Enemies[0].Hp);
    }
}
