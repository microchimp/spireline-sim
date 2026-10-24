// Core2's kill sequence (docs/rebuild/MIGRATION_F08.md section 3 items 4-6 and 8, round R3a): CreatureCmd.Kill,
// KillWithoutCheckingWinCondition, Escape, PerformMove's end and the old roster's removal, with test-only handlers
// standing in for the models they name (registered under the F08+F10 switch). No model is migrated here; each handler
// returns what the case needs, and each test cites the game method it is read from.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class KillCommandTests
{
    private static readonly OldModelIds Ids = ListenerOrderTests.Ids;
    private static readonly Switch[] DamageOnly = [Switch.F08, Switch.F10];
    private static readonly JsonObject IdMap = LoadIdMap();
    private const int Slug = 16;

    private static JsonObject LoadIdMap()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (!File.Exists(Path.Combine(dir!.FullName, "data", "id_map.json")))
            dir = dir.Parent;
        return JsonNode.Parse(File.ReadAllText(Path.Combine(dir.FullName, "data", "id_map.json")))!.AsObject();
    }

    private static string Name(string kind, int id) => IdMap[kind]!.AsObject().First(kv => (int)kv.Value! == id).Key;

    private static readonly string Relic = Name("relics", 1);
    private static readonly string Monster = Name("enemies", Slug);

    private static CombatState Combat(params int[] enemyHp)
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = enemyHp.Select(hp => new EnemyState { DefId = Slug, Hp = hp, MaxHp = hp }).ToList();
        s.Relics = [new RelicInstance(1)];
        return s;
    }

    private static KernelState K(CombatState s, Action<HandlerTable> models) =>
        new Kernel(new KernelOptions(DamageOnly), () => Ids, [models]).Bind(s);

    private static int PowerId(BuffId b) => Ids.Power(b).ModelId;

    private static HookEvent<T> Log<T>(List<string> log, Func<T, string> what) where T : struct =>
        (ListenerRef self, in T a, ChoiceCtx? c) =>
        {
            log.Add(what(a));
            return HookStep.Done;
        };

    private static HookPredicate<T> Ask<T>(List<string> log, Func<T, string> what, Func<T, bool>? answer = null)
        where T : struct =>
        (ListenerRef self, in T a) =>
        {
            log.Add(what(a));
            return answer?.Invoke(a) ?? true;
        };

    /// <summary>Every death hook on the player's relic and the slug's monster model, logged.</summary>
    private static void DeathHooks(HandlerTable t, List<string> log, Func<ShouldDieArgs, bool>? shouldDie = null,
        Func<ShouldCreatureBeRemovedFromCombatAfterDeathArgs, bool>? remove = null)
    {
        t.Model(ListenerKind.Relic, Relic)
            .On(HookId.BeforeDeath, Log<BeforeDeathArgs>(log, a => $"BeforeDeath {a.Creature.Index}"))
            .On(HookId.ShouldDie, Ask(log, (ShouldDieArgs a) => $"ShouldDie {a.Creature.Index}", shouldDie))
            .On(HookId.ShouldCreatureBeRemovedFromCombatAfterDeath,
                Ask(log, (ShouldCreatureBeRemovedFromCombatAfterDeathArgs a) => $"ShouldRemove {a.Creature.Index}", remove))
            .On(HookId.AfterDeath, Log<AfterDeathArgs>(log, a => $"AfterDeath {a.Creature.Index} {a.WasRemovalPrevented}"))
            .On(HookId.ShouldPowerBeRemovedOnDeath,
                Ask(log, (ShouldPowerBeRemovedOnDeathArgs a) => $"PowerOnDeath {a.Power.ModelId}"));
        t.Model(ListenerKind.Monster, Monster)
            .On(HookId.Died, Log<DiedArgs>(log, a => "Died"))
            .On(HookId.BeforeRemovedFromRoom, Log<BeforeRemovedFromRoomArgs>(log, a => "BeforeRemovedFromRoom"));
    }

    private static IReadOnlyList<DamageResult> Hit(KernelState k, OldCreature target, decimal amount, OldCreature? dealer = null) =>
        CreatureCmd.Damage(k, [target], amount, ValueProps.PoweredAttack, dealer ?? OldCreature.Player, null);

    [Fact]
    public void AKillRunsTheGamesSequenceAndTheEnemyLeavesTheRosterInsideTheDamageCall()
    {
        // game: CreatureCmd.KillWithoutCheckingWinCondition: BeforeDeath, ShouldDie, InvokeDiedEvent,
        // ShouldCreatureBeRemovedFromCombatAfterDeath, AfterDeath(false), CombatManager.RemoveCreature
        // (BeforeRemovedFromRoom) and CombatState.RemoveCreature, then RemoveAllPowersAfterDeath (the owner left the
        // combat: Hook.ShouldPowerBeRemovedOnDeath returns true with no listener asked) and AfterRemoved per power.
        var s = Combat(5, 40);
        var slug = s.Enemies[0];
        slug.Buffs.Add(new BuffState(BuffId.Vulnerable, 2));
        slug.MoveIndex = 3;
        var log = new List<string>();
        int listedAtAfterDeath = -1;
        KernelState? kk = null;
        var k = kk = K(s, t =>
        {
            DeathHooks(t, log);
            t.Model(ListenerKind.Power, "VulnerablePower")
                .On(HookId.ShouldPowerBeRemovedAfterOwnerDeath, Ask(log, (PowerDeathArgs a) => $"OwnDeath {a.Buff}"))
                .On(HookId.AfterRemoved, Log<AfterRemovedArgs>(log, a => $"AfterRemoved {a.Buff} {a.Removed.Count}"));
            t.Model(ListenerKind.Monster, Name("enemies", Slug)).On(HookId.AfterDeath,
                (HookEvent<AfterDeathArgs>)((ListenerRef self, in AfterDeathArgs a, ChoiceCtx? c) =>
                {
                    listedAtAfterDeath = OldCreatures.Enemies(kk!.World!).Count;
                    return HookStep.Done;
                }));
        });
        Hit(k, OldCreature.Of(slug), 9m);
        Assert.Equal(["BeforeDeath 1", "ShouldDie 1", "Died", "ShouldRemove 1", "AfterDeath 1 False", "BeforeRemovedFromRoom",
            "OwnDeath Vulnerable", "AfterRemoved Vulnerable 1"], log);
        Assert.Equal(2, listedAtAfterDeath);
        Assert.Equal([s.Enemies[0]], s.Enemies.Where(e => e != slug));
        Assert.Single(s.Enemies);
        Assert.Contains(Slug, s.DepartedEnemyDefIds);
        Assert.Empty(slug.Buffs);
        Assert.Equal(0, slug.MoveIndex);
        Assert.Equal(1, k.Commands.KillCalls);
    }

    [Fact]
    public void AShouldDiePreventerRevivesThroughAfterDeathTrueAndAfterPreventingDeath()
    {
        // game: KillWithoutCheckingWinCondition's prevented branch: AfterDeath(wasRemovalPrevented: true), then
        // Hook.AfterPreventingDeath on the preventer only (Fairy in a Bottle heals there), and no Died, no removal.
        var s = Combat(30);
        var log = new List<string>();
        var k = K(s, t =>
        {
            DeathHooks(t, log, shouldDie: a => a.Creature.Index != 0);
            t.Model(ListenerKind.Relic, Relic).On(HookId.AfterPreventingDeath,
                (HookEvent<AfterPreventingDeathArgs>)((ListenerRef self, in AfterPreventingDeathArgs a, ChoiceCtx? c) =>
                {
                    log.Add("AfterPreventingDeath");
                    s.PlayerHp = 10;
                    return HookStep.Done;
                }));
        });
        Hit(k, OldCreature.Player, 80m, OldCreature.Of(s.Enemies[0]));
        Assert.Equal(["BeforeDeath 0", "ShouldDie 0", "AfterDeath 0 True", "AfterPreventingDeath"], log);
        Assert.Equal(10, s.PlayerHp);
        Assert.Equal(0, k.Commands.LossCalls);
        Assert.NotEqual(PlayerHooks.Deactivated, k.World!.PlayerHooks);
    }

    [Fact]
    public void APreventerThatNeverRevivesStopsAfterTenRounds()
    {
        // game: KillWithoutCheckingWinCondition: recursion >= 10 throws InvalidOperationException.
        var s = Combat(30);
        var log = new List<string>();
        var k = K(s, t => DeathHooks(t, log, shouldDie: _ => false));
        Assert.Throws<InvalidOperationException>(() => CreatureCmd.Kill(k, [OldCreature.Player]));
        Assert.Equal(11, log.Count(x => x == "ShouldDie 0"));
    }

    [Fact]
    public void PowersAreKeptOrRemovedByBothPredicatesAllJudgedFirstThenAfterRemovedInOrder()
    {
        // game: Creature.RemoveAllPowersAfterDeath: Except(powers where !ShouldPowerBeRemovedAfterOwnerDeath() ||
        // !Hook.ShouldPowerBeRemovedOnDeath(p)), every power judged before any goes; then AfterRemoved per removed power.
        // The removal is vetoed (ShouldCreatureBeRemovedFromCombatAfterDeath false), so the owner is in the combat and
        // Hook.ShouldPowerBeRemovedOnDeath asks its listeners.
        var s = Combat(5);
        var slug = s.Enemies[0];
        slug.Buffs.Add(new BuffState(BuffId.Vulnerable, 1));
        slug.Buffs.Add(new BuffState(BuffId.Weak, 1));
        slug.Buffs.Add(new BuffState(BuffId.Strength, 3));
        var log = new List<string>();
        int weak = PowerId(BuffId.Weak);
        var k = K(s, t =>
        {
            t.Model(ListenerKind.Relic, Relic)
                .On(HookId.ShouldCreatureBeRemovedFromCombatAfterDeath,
                    (HookPredicate<ShouldCreatureBeRemovedFromCombatAfterDeathArgs>)((ListenerRef self,
                        in ShouldCreatureBeRemovedFromCombatAfterDeathArgs a) => false))
                .On(HookId.ShouldPowerBeRemovedOnDeath,
                    Ask(log, (ShouldPowerBeRemovedOnDeathArgs a) => $"OnDeath {a.Power.ModelId == weak}",
                        a => a.Power.ModelId != weak));
            foreach (var (power, keep) in new[] { ("VulnerablePower", true), ("WeakPower", false), ("StrengthPower", false) })
                t.Model(ListenerKind.Power, power)
                    .On(HookId.ShouldPowerBeRemovedAfterOwnerDeath, Ask(log, (PowerDeathArgs a) => $"Own {a.Buff}", _ => !keep))
                    .On(HookId.AfterRemoved, Log<AfterRemovedArgs>(log, a => $"AfterRemoved {a.Buff}"));
        });
        Hit(k, OldCreature.Of(slug), 9m);
        Assert.Equal(["Own Vulnerable", "Own Weak", "OnDeath True", "Own Strength", "OnDeath False", "AfterRemoved Strength"], log);
        Assert.Equal([BuffId.Vulnerable, BuffId.Weak], slug.Buffs.Select(b => b.Id));
        Assert.Contains(slug, s.Enemies);
    }

    [Fact]
    public void AMonsterKilledDuringItsMoveLeavesTheRoomAtOnceAndTheRosterAtTheMovesEnd()
    {
        // game: KillWithoutCheckingWinCondition calls CombatManager.RemoveCreature (BeforeRemovedFromRoom,
        // ResetStateMachine) always, CombatState.RemoveCreature only when !monster.IsPerformingMove; MonsterModel.PerformMove
        // clears the flag after the move and asks ShouldCreatureBeRemovedFromCombatAfterDeath again, removing it then.
        var s = Combat(5, 40);
        var mover = s.Enemies[0];
        mover.Buffs.Add(new BuffState(BuffId.Strength, 2));
        mover.MoveIndex = 2;
        var log = new List<string>();
        var k = K(s, t => DeathHooks(t, log));
        var w = k.World!;
        bool listedAfterKill = false, inRosterAfterKill = false;
        CreatureCmd.PerformMove(k, OldCreature.Of(mover), () =>
        {
            Hit(k, OldCreature.Of(mover), 9m); // Thorns-like: the mover's own attack kills it
            inRosterAfterKill = s.Enemies.Contains(mover);
            listedAfterKill = OldCreatures.Enemies(w).Contains(OldCreature.Of(mover));
        });
        Assert.True(inRosterAfterKill);
        Assert.True(listedAfterKill);
        Assert.Equal(["BeforeDeath 1", "ShouldDie 1", "Died", "ShouldRemove 1", "AfterDeath 1 False", "BeforeRemovedFromRoom",
            $"PowerOnDeath {PowerId(BuffId.Strength)}", "ShouldRemove 1"], log);
        Assert.Equal(0, mover.MoveIndex);
        Assert.DoesNotContain(mover, s.Enemies);
        Assert.Contains(Slug, s.DepartedEnemyDefIds);
        Assert.Null(w.Moving);
    }

    [Fact]
    public void TwinStrikesFirstHitKillsAndRemovesItsTargetSoTheSecondHitFindsNone()
    {
        // game: AttackCommand.Execute per hit: the living possible targets, none in a live combat: break; Kill ran
        // inside hit 1's CreatureCmd.Damage (MIGRATION_F08.md section 1's Twin Strike trace).
        var s = Combat(5, 40);
        var slug = s.Enemies[0];
        var k = K(s, _ => { });
        var a = new AttackCmd(OldCreature.Player, 6m, 2) { SingleTarget = OldCreature.Of(slug) }.Execute(k);
        Assert.Equal(1, a.Hits);
        Assert.DoesNotContain(slug, s.Enemies);
        Assert.Equal(40, s.Enemies[0].Hp);
        // Whirlwind: hit 2 reaches only the survivor.
        var s2 = Combat(5, 40);
        var k2 = K(s2, _ => { });
        var all = new AttackCmd(OldCreature.Player, 6m, 2) { AllOpponents = true }.Execute(k2);
        Assert.Equal(2, all.Hits);
        Assert.Single(s2.Enemies);
        Assert.Equal(28, s2.Enemies[0].Hp);
    }

    [Fact]
    public void HeadbuttsCorpseIsGoneBeforeTheCardsNextEffect()
    {
        // G2891c25e0d. game: the removal is inside KillWithoutCheckingWinCondition, so when CreatureCmd.Damage returns to
        // the card (Headbutt's discard-pile screen next) the dead slug is out of the combat; an AfterDeath listener on the
        // survivor (Ravenous) ran while it was still listed.
        var s = Combat(5, 40);
        var dead = s.Enemies[0];
        var survivor = s.Enemies[1];
        survivor.Buffs.Add(new BuffState(BuffId.Ravenous, 5));
        var seen = new List<int>();
        KernelState? kk = null;
        var k = kk = K(s, t => t.Model(ListenerKind.Power, "RavenousPower").On(HookId.AfterDeath,
            (HookEvent<AfterDeathArgs>)((ListenerRef self, in AfterDeathArgs a, ChoiceCtx? c) =>
            {
                seen.Add(OldCreatures.Enemies(kk!.World!).Count);
                return HookStep.Done;
            })));
        Hit(k, OldCreature.Of(dead), 9m);
        Assert.Equal([2], seen);
        var w = k.World!;
        Assert.Equal([OldCreature.Of(survivor)], OldCreatures.Enemies(w));
        Assert.Equal([OldCreature.Of(survivor)], OldCreatures.HittableEnemies(w));
        Assert.Equal(new CreatureRef(1), OldCreatures.Ref(w, OldCreature.Of(survivor)));
    }

    [Fact]
    public void EscapeRunsNoDeathHookButLeavesTheRoom()
    {
        // game: CreatureCmd.Escape: RemoveAllPowersInternalExcept (no AfterRemoved), CombatManager.RemoveCreature
        // (BeforeRemovedFromRoom), CombatState.CreatureEscaped (escaped list, roster). No BeforeDeath, ShouldDie, Died,
        // AfterDeath.
        var s = Combat(20, 40);
        var hopper = s.Enemies[0];
        hopper.Buffs.Add(new BuffState(BuffId.Vulnerable, 1));
        var log = new List<string>();
        var k = K(s, t =>
        {
            DeathHooks(t, log);
            t.Model(ListenerKind.Power, "VulnerablePower")
                .On(HookId.AfterRemoved, Log<AfterRemovedArgs>(log, a => "AfterRemoved"));
        });
        CreatureCmd.Escape(k, OldCreature.Of(hopper));
        Assert.Equal(["BeforeRemovedFromRoom"], log);
        Assert.True(hopper.Escaped);
        Assert.Equal(1, s.EscapedEnemies);
        Assert.DoesNotContain(hopper, s.Enemies);
        Assert.Equal(20, hopper.Hp);
        Assert.Empty(hopper.Buffs);
        CreatureCmd.Escape(k, OldCreature.Of(hopper));
        Assert.Equal(1, s.EscapedEnemies);
    }

    [Fact]
    public void ThePlayersDeathKillsOstyThenDeactivatesThePlayersHooksAndReachesTheLoss()
    {
        // game: KillWithoutCheckingWinCondition for a player: its own death passes visit its listeners at 0 HP; after
        // AfterDeath and the powers, the orb queue cleared, Osty killed, Player.DeactivateHooks; then CreatureCmd.Kill's
        // all-players-dead branch: CombatManager.LoseCombat (R3b's point).
        var s = Combat(30);
        s.OstyHp = 7;
        s.OstyMaxHp = 7;
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 1));
        var log = new List<string>();
        KernelState? kk = null;
        var k = kk = K(s, t =>
        {
            DeathHooks(t, log);
            t.Model(ListenerKind.Relic, Relic).On(HookId.AfterCurrentHpChanged,
                Log<AfterCurrentHpChangedArgs>(log, a => $"HpChanged {a.Creature.Index} {kk!.World!.PlayerHooks}"));
            t.Model(ListenerKind.Power, "StrengthPower")
                .On(HookId.AfterRemoved, Log<AfterRemovedArgs>(log, a => $"AfterRemoved {kk!.World!.PlayerHooks}"));
        });
        Hit(k, OldCreature.Player, 80m, OldCreature.Of(s.Enemies[0]));
        Assert.Equal(["HpChanged 0 Active", "BeforeDeath 0", "ShouldDie 0", "ShouldRemove 0", "AfterDeath 0 False",
            $"PowerOnDeath {PowerId(BuffId.Strength)}", "AfterRemoved Active", "HpChanged 1 Active", "BeforeDeath 1", "ShouldDie 1", "ShouldRemove 1",
            "AfterDeath 1 False"], log);
        Assert.Equal(0, s.OstyHp);
        Assert.Empty(s.PlayerBuffs);
        Assert.Equal(PlayerHooks.Deactivated, k.World!.PlayerHooks);
        Assert.False(OldState.PlayerActive(k.World!));
        Assert.Equal(1, k.Commands.LossCalls);
    }

    [Fact]
    public void APrimaryEnemysDeathTakesItsSecondaryTeammatesWithIt()
    {
        // game: KillWithoutCheckingWinCondition: a primary enemy (read before its powers go) whose living teammates are
        // all secondary (Creature.IsSecondaryEnemy: MinionPower.OwnerIsSecondaryEnemy) kills them.
        var s = Combat(5, 30, 30);
        s.Enemies[1].Buffs.Add(new BuffState(BuffId.Minion, 1));
        s.Enemies[2].Buffs.Add(new BuffState(BuffId.Minion, 1));
        var k = K(s, _ => { });
        Hit(k, OldCreature.Of(s.Enemies[0]), 9m);
        Assert.Empty(s.Enemies);
        Assert.Equal(2, k.Commands.KillCalls);
        // A secondary enemy's death takes nobody.
        var s2 = Combat(5, 30);
        s2.Enemies[0].Buffs.Add(new BuffState(BuffId.Minion, 1));
        Hit(K(s2, _ => { }), OldCreature.Of(s2.Enemies[0]), 9m);
        Assert.Equal([30], s2.Enemies.Select(e => e.Hp));
    }

    [Fact]
    public void ADealerANestedDamageKilledIsNoLongerNamedByLaterHooks()
    {
        // game: CreatureCmd.Damage keeps the dealer Creature after a nested command killed it (Thorns-like retaliation
        // inside AfterCurrentHpChanged); its position is gone once the kill removed it, so later hooks get no dealer.
        var s = Combat(5, 40);
        var attacker = s.Enemies[0];
        CreatureRef? dealerSeen = new(99);
        KernelState? kk = null;
        var k = kk = K(s, t => t.Model(ListenerKind.Relic, Relic)
            .On(HookId.AfterCurrentHpChanged,
                (HookEvent<AfterCurrentHpChangedArgs>)((ListenerRef self, in AfterCurrentHpChangedArgs a, ChoiceCtx? c) =>
                {
                    if (a.Creature.Index == 0)
                        Hit(kk!, OldCreature.Of(attacker), 99m, OldCreature.Player);
                    return HookStep.Done;
                }))
            .On(HookId.AfterDamageGiven,
                (HookEvent<AfterDamageGivenArgs>)((ListenerRef self, in AfterDamageGivenArgs a, ChoiceCtx? c) =>
                {
                    if (a.Target.Index == 0)
                        dealerSeen = a.Dealer;
                    return HookStep.Done;
                })));
        Hit(k, OldCreature.Player, 6m, OldCreature.Of(attacker));
        Assert.Equal(44, s.PlayerHp);
        Assert.DoesNotContain(attacker, s.Enemies);
        Assert.Null(dealerSeen);
    }
}
