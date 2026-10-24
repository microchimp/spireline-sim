// Core2's game-shaped damage commands (docs/rebuild/MIGRATION_F08.md section 3 items 1-3, round R2): CreatureCmd.Damage
// and AttackCmd over the old state, with test-only handlers standing in for the powers they name (registered in the
// test under the F08+F10 switch, or every switch where the hook is another family's). No model is migrated here; the
// handlers return what the game's override returns for the case. Each test cites the game method it is read from.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class DamageCommandTests
{
    private static readonly OldModelIds Ids = ListenerOrderTests.Ids;
    private static readonly Switch[] Every = Enum.GetValues<Switch>().Where(s => s != Switch.None).ToArray();
    private static readonly Switch[] DamageOnly = [Switch.F08, Switch.F10];
    private static readonly JsonObject IdMap = LoadIdMap();

    private static JsonObject LoadIdMap()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (!File.Exists(Path.Combine(dir!.FullName, "data", "id_map.json")))
            dir = dir.Parent;
        return JsonNode.Parse(File.ReadAllText(Path.Combine(dir.FullName, "data", "id_map.json")))!.AsObject();
    }

    private static string RelicName(int id) => IdMap["relics"]!.AsObject().First(kv => (int)kv.Value! == id).Key;

    private static CombatState Combat(params int[] enemyHp)
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = enemyHp.Select(hp => new EnemyState { DefId = 16, Hp = hp, MaxHp = hp }).ToList();
        s.Relics = [new RelicInstance(1)];
        return s;
    }

    private static KernelState K(CombatState s, Action<HandlerTable> models, IEnumerable<Switch>? on = null, bool countKills = true)
    {
        var k = new Kernel(new KernelOptions(on ?? Every), () => Ids, [models]).Bind(s);
        if (countKills)
            k.Commands.KillStandIn = (_, killed) => Killed.Value!.AddRange(killed);
        Killed.Value = [];
        return k;
    }

    private static readonly ThreadLocal<List<OldCreature>> Killed = new(() => []);

    // A power listener's owner: OldState.PowerInstance(creature, instance).
    private static int Owner(ListenerRef self) => self.Instance >> 8;

    private static bool OnOwner(ListenerRef self, CreatureRef? target) => target is { } t && t.Index == Owner(self);

    private static HookFold<ModifyDamageArgs, decimal> MulOnOwner(decimal factor) =>
        (HookCtx _, ListenerRef self, in ModifyDamageArgs a, decimal v) => OnOwner(self, a.Target) ? factor : 1m;

    private static IReadOnlyList<DamageResult> Hit(KernelState k, OldCreature target, decimal amount,
        OldCreature? dealer = null, ValueProp? props = null, CardHandle? card = null) =>
        CreatureCmd.Damage(k, [target], amount, props ?? ValueProps.PoweredAttack, dealer ?? OldCreature.Player, card);

    // ---------------------------------------------------------------------------------------- the decimal fold

    [Fact]
    public void TheFoldIsAdditiveThenMultiplicativeInDecimalAndCastOnceAtTheHpLoss()
    {
        // game: Hook.ModifyDamageInternal (additive, multiplicative, cap passes over IterateHookListeners, decimal) and
        // Creature.LoseHpInternal (the one cast). Strength 2 on the dealer (StrengthPower.ModifyDamageAdditive), the
        // target Vulnerable (x1.5) and Slow at x1.1 (SlowPower.ModifyDamageMultiplicative, one card played): 3 + 2 = 5, 5 x 1.5 x 1.1 =
        // 8.25, 8 lost. The batch-3 RoundingOrder case: an int fold casting after Vulnerable gives 7 x 1.1 = 7.
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 2));
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, 1));
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Slow, 1));
        decimal seen = -1;
        var k = K(s, t =>
        {
            t.Model(ListenerKind.Power, "StrengthPower").On(HookId.ModifyDamageAdditive,
                (HookFold<ModifyDamageArgs, decimal>)((HookCtx _, ListenerRef self, in ModifyDamageArgs a, decimal v) =>
                    a.Dealer is { } d && d.Index == Owner(self) ? 2m : 0m));
            t.Model(ListenerKind.Power, "VulnerablePower").On(HookId.ModifyDamageMultiplicative, MulOnOwner(1.5m));
            t.Model(ListenerKind.Power, "SlowPower").On(HookId.ModifyDamageMultiplicative, MulOnOwner(1.1m));
            t.Model(ListenerKind.Relic, RelicName(1)).On(HookId.BeforeDamageReceived,
                (HookEvent<BeforeDamageReceivedArgs>)((HookCtx _, ListenerRef self, in BeforeDamageReceivedArgs a, ChoiceCtx? c) =>
                {
                    seen = a.Amount;
                    return HookStep.Done;
                }));
        });
        var r = Assert.Single(Hit(k, OldCreature.Of(s.Enemies[0]), 3m));
        Assert.Equal(8.25m, seen);
        Assert.Equal(8, r.UnblockedDamage);
        Assert.Equal(32, s.Enemies[0].Hp);
    }

    [Fact]
    public void VulnerableAndSlowOnNineGiveSixteen()
    {
        // MIGRATION_F08.md section 6's case (game: Hook.ModifyDamageInternal, LoseHpInternal): 9 x 1.5 = 13.5, Slow at
        // x1.2 = 16.2, cast once: 16.
        var s = Combat(40);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, 1));
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Slow, 2));
        var k = K(s, t =>
        {
            t.Model(ListenerKind.Power, "VulnerablePower").On(HookId.ModifyDamageMultiplicative, MulOnOwner(1.5m));
            t.Model(ListenerKind.Power, "SlowPower").On(HookId.ModifyDamageMultiplicative, MulOnOwner(1.2m));
        });
        Assert.Equal(16, Assert.Single(Hit(k, OldCreature.Of(s.Enemies[0]), 9m)).UnblockedDamage);
    }

    [Fact]
    public void TheIntangibleCapLowersOnlyValuesAboveOneAndJoinsTheChangedSetOnlyThen()
    {
        // game: Hook.ModifyDamageInternal's cap pass (a listener counts as a modifier only when it lowers the value),
        // IntangiblePower.ModifyDamageCap (1 on its owner), Hook.AfterModifyingDamageAmount (filter: the modifiers).
        var s = Combat(40);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Intangible, 1));
        var told = new List<int>();
        var k = K(s, t =>
            t.Model(ListenerKind.Power, "IntangiblePower").On(HookId.ModifyDamageCap,
                    (HookCap<ModifyDamageArgs, decimal>)((HookCtx _, ListenerRef self, in ModifyDamageArgs a) =>
                        OnOwner(self, a.Target) ? 1m : decimal.MaxValue))
                .On(HookId.AfterModifyingDamageAmount,
                    (HookEvent<AfterModifyingDamageAmountArgs>)((HookCtx _, ListenerRef self, in AfterModifyingDamageAmountArgs a,
                        ChoiceCtx? c) =>
                    {
                        told.Add(a.Modifiers.Count);
                        return HookStep.Done;
                    })));
        var enemy = OldCreature.Of(s.Enemies[0]);
        Assert.Equal(1, Assert.Single(Hit(k, enemy, 10m)).UnblockedDamage);
        Assert.Equal(0, Assert.Single(Hit(k, enemy, 0.5m)).UnblockedDamage);
        Assert.Equal([1], told);
        Assert.Equal(39, s.Enemies[0].Hp);
    }

    [Fact]
    public void HardToKillsCapComesAfterSlow()
    {
        // game: Hook.ModifyDamageInternal runs the cap pass last: 10 x 1.2 (Slow, two cards played) = 12, capped at 5 (HardToKillPower
        // .ModifyDamageCap). A cap before the multiplier would give 6.
        var s = Combat(40);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Slow, 2));
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.HardToKill, 5));
        var k = K(s, t =>
        {
            t.Model(ListenerKind.Power, "SlowPower").On(HookId.ModifyDamageMultiplicative, MulOnOwner(1.2m));
            t.Model(ListenerKind.Power, "HardToKillPower").On(HookId.ModifyDamageCap,
                (HookCap<ModifyDamageArgs, decimal>)((HookCtx _, ListenerRef self, in ModifyDamageArgs a) =>
                    OnOwner(self, a.Target) ? 5m : decimal.MaxValue));
        });
        Assert.Equal(5, Assert.Single(Hit(k, OldCreature.Of(s.Enemies[0]), 10m)).UnblockedDamage);
    }

    [Fact]
    public void TheEnchantmentStepsComeFirstAndTheClampLast()
    {
        // game: Hook.ModifyDamage: the card's EnchantDamageAdditive then EnchantDamageMultiplicative before the passes
        // (Sharp 3 then the target's Vulnerable: (6 + 3) x 1.5 = 13.5 -> 13; Corrupted x1.5 on 6, then +1 = 10), only
        // on a powered attack (an unpowered hit of the same card gets none), and Math.Max(0, ...) at the end.
        var s = Combat(100);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, 1));
        var sharp = s.Fresh(new CardInstance(IC.StrikeIronclad, false, Enchantment: Enchantment.Sharp, EnchantAmount: 3));
        var corrupt = s.Fresh(new CardInstance(IC.StrikeIronclad, false, Enchantment: Enchantment.Corrupted));
        s.Hand.Add(sharp);
        s.Hand.Add(corrupt);
        bool vulnerable = true;
        decimal add = 0m;
        var k = K(s, t =>
        {
            t.Model(ListenerKind.Power, "VulnerablePower").On(HookId.ModifyDamageMultiplicative,
                (HookFold<ModifyDamageArgs, decimal>)((HookCtx _, ListenerRef self, in ModifyDamageArgs a, decimal v) =>
                    vulnerable && OnOwner(self, a.Target) ? 1.5m : 1m));
            t.Model(ListenerKind.Relic, RelicName(1)).On(HookId.ModifyDamageAdditive,
                (HookFold<ModifyDamageArgs, decimal>)((HookCtx _, ListenerRef self, in ModifyDamageArgs a, decimal v) => add));
        });
        var enemy = OldCreature.Of(s.Enemies[0]);
        Assert.Equal(13, Assert.Single(Hit(k, enemy, 6m, card: new CardHandle(sharp.Uid))).UnblockedDamage);
        vulnerable = false;
        add = 1m;
        Assert.Equal(10, Assert.Single(Hit(k, enemy, 6m, card: new CardHandle(corrupt.Uid))).UnblockedDamage);
        Assert.Equal(7, Assert.Single(Hit(k, enemy, 6m, props: new ValueProp(ValueProps.Move | ValueProps.Unpowered),
            card: new CardHandle(sharp.Uid))).UnblockedDamage);
        add = -20m;
        var none = Assert.Single(Hit(k, enemy, 6m));
        Assert.Equal(0, none.UnblockedDamage);
        Assert.Equal(100 - 13 - 10 - 7, s.Enemies[0].Hp);
    }

    [Fact]
    public void ThePreviewOfAnAllEnemiesCardFoldsOncePerHittableEnemyAndKeepsAnAgreedValue()
    {
        // game: Hook.ModifyDamage with no target and CardPreviewMode.MultiCreatureTargeting for an AllEnemies card in
        // Hand: the passes per CombatState.HittableEnemies entry; agreeing after an int cast, that value; else the plain
        // fold (no target: the Vulnerable listener does not apply).
        var s = Combat(30, 30);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, 1));
        var whirl = s.Fresh(new CardInstance(IC.Whirlwind, false)); // a uid under the counter: the listing keeps it
        s.Hand.Add(whirl);
        decimal factor = 1.5m;
        var k = K(s, t => t.Model(ListenerKind.Power, "VulnerablePower").On(HookId.ModifyDamageMultiplicative,
            (HookFold<ModifyDamageArgs, decimal>)((HookCtx _, ListenerRef self, in ModifyDamageArgs a, decimal v) =>
                OnOwner(self, a.Target) ? factor : 1m)));
        var preview = new ModifyDamageArgs(null, new CreatureRef(0), 5m, ValueProps.PoweredAttack, new CardHandle(whirl.Uid),
            ModifyDamageHookType.All, PreviewModes.MultiCreatureTargeting);
        Assert.Equal(5m, Hooks.ModifyDamage(k, preview, out var ch));
        Assert.Empty(ch);
        factor = 1.1m; // 5.5 and 5 agree after the cast: the first enemy's value and its modifier
        Assert.Equal(5.5m, Hooks.ModifyDamage(k, preview, out ch));
        Assert.Single(ch);
        s.Enemies[1].Buffs.Add(new BuffState(BuffId.Vulnerable, 1)); // both 5.5
        Assert.Equal(5.5m, Hooks.ModifyDamage(k, preview, out ch));
        Assert.Equal(2, ch.Count);
    }

    // ---------------------------------------------------------------------------------------- block and the HP loss

    [Fact]
    public void BlockAbsorbsInDecimalLosesTheCastAmountAndAfterBlockBrokenFiresOnce()
    {
        // game: Creature.DamageBlockInternal (blocked = min(Block, amount), Block -= (int)blocked), CreatureCmd.Damage's
        // wasBlockBroken (block now 0 and something blocked) and wasFullyBlocked, then AfterBlockBroken per broken
        // result, AfterCurrentHpChanged only with HP lost.
        var s = Combat(40);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, 1));
        var broke = new List<int>();
        var hpChanged = new List<decimal>();
        var k = K(s, t =>
        {
            t.Model(ListenerKind.Power, "VulnerablePower").On(HookId.ModifyDamageMultiplicative, MulOnOwner(1.5m));
            t.Model(ListenerKind.Relic, RelicName(1))
                .On(HookId.AfterBlockBroken, (HookEvent<AfterBlockBrokenArgs>)((HookCtx _, ListenerRef self, in AfterBlockBrokenArgs a,
                    ChoiceCtx? c) =>
                {
                    broke.Add(a.Creature.Index);
                    return HookStep.Done;
                }))
                .On(HookId.AfterCurrentHpChanged, (HookEvent<AfterCurrentHpChangedArgs>)((HookCtx _, ListenerRef self,
                    in AfterCurrentHpChangedArgs a, ChoiceCtx? c) =>
                {
                    hpChanged.Add(a.Delta);
                    return HookStep.Done;
                }));
        });
        var enemy = OldCreature.Of(s.Enemies[0]);
        s.Enemies[0].Block = 10;
        var full = Assert.Single(Hit(k, enemy, 5m)); // 7.5: block loses 7, nothing through
        Assert.Equal((3, 40, 7, true, false), (s.Enemies[0].Block, s.Enemies[0].Hp, full.BlockedDamage, full.WasFullyBlocked,
            full.WasBlockBroken));
        var through = Assert.Single(Hit(k, enemy, 5m)); // 7.5 on 3 block: 4.5 through, 4 lost
        Assert.Equal((0, 36, 3, true), (s.Enemies[0].Block, s.Enemies[0].Hp, through.BlockedDamage, through.WasBlockBroken));
        Assert.Equal([1], broke);
        Assert.Equal([-4m], hpChanged);
    }

    [Fact]
    public void OstyTakesTheRedirectedHitAndTheOverkillGoesBackToThePlayer()
    {
        // game: CreatureCmd.Damage: ModifyUnblockedDamageTarget moves the unblocked damage to Osty, ModifyHpLost after
        // Osty folds it there, Osty's LoseHpInternal keeps the overkill, which a second after-Osty fold sends to the
        // original target; two results, the block on the original's. Osty killed joins the killed list.
        var s = Combat(40);
        s.OstyMaxHp = 10;
        s.OstyHp = 3;
        s.PlayerBlock = 2;
        var afterOsty = new List<(int, decimal)>();
        var k = K(s, t =>
        {
            t.Model(ListenerKind.Monster, "Osty").On(HookId.ModifyUnblockedDamageTarget,
                (HookFold<ModifyUnblockedDamageTargetArgs, CreatureRef>)((HookCtx _, ListenerRef self,
                    in ModifyUnblockedDamageTargetArgs a, CreatureRef v) => a.OriginalTarget.Index == 0 ? new CreatureRef(1) : v));
            t.Model(ListenerKind.Relic, RelicName(1)).On(HookId.ModifyHpLostAfterOsty,
                (HookFold<ModifyHpLostArgs, decimal>)((HookCtx _, ListenerRef self, in ModifyHpLostArgs a, decimal v) =>
                {
                    afterOsty.Add((a.Target.Index, v));
                    return v;
                }));
        });
        var results = CreatureCmd.Damage(k, [OldCreature.Player], 12m, ValueProps.PoweredAttack,
            OldCreature.Of(s.Enemies[0]), null);
        Assert.Equal(2, results.Count);
        Assert.Equal((CreatureKind.Osty, 3, 7, true), (results[0].Receiver.Kind, results[0].UnblockedDamage,
            results[0].OverkillDamage, results[0].WasTargetKilled));
        Assert.Equal((CreatureKind.Player, 7, 2), (results[1].Receiver.Kind, results[1].UnblockedDamage, results[1].BlockedDamage));
        Assert.Equal([(1, 10m), (0, 7m)], afterOsty);
        Assert.Equal((0, 0, 43), (s.OstyHp, s.PlayerBlock, s.PlayerHp));
        Assert.Equal([OldCreature.Osty], Killed.Value!);
    }

    // ---------------------------------------------------------------------------------------- the sequence

    private static Action<HandlerTable> Recorder(List<string> log) => t => t.Model(ListenerKind.Relic, RelicName(1))
        .On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)((HookCtx _, ListenerRef s, in ModifyDamageArgs a, decimal v) =>
            Note(log, "add", a.Target, 0m)))
        .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)((HookCtx _, ListenerRef s, in ModifyDamageArgs a,
            decimal v) => Note(log, "mul", a.Target, 1m)))
        .On(HookId.ModifyDamageCap, (HookCap<ModifyDamageArgs, decimal>)((HookCtx _, ListenerRef s, in ModifyDamageArgs a) =>
            Note(log, "cap", a.Target, decimal.MaxValue)))
        .On(HookId.AfterModifyingDamageAmount, Ev<AfterModifyingDamageAmountArgs>(log, _ => "afterAmount"))
        .On(HookId.BeforeDamageReceived, Ev<BeforeDamageReceivedArgs>(log, a => $"before{a.Target.Index}"))
        .On(HookId.ModifyHpLostBeforeOsty, (HookFold<ModifyHpLostArgs, decimal>)((HookCtx _, ListenerRef s, in ModifyHpLostArgs a,
            decimal v) => Note(log, "lostBefore", a.Target, v)))
        .On(HookId.ModifyUnblockedDamageTarget, (HookFold<ModifyUnblockedDamageTargetArgs, CreatureRef>)((HookCtx _, ListenerRef s,
            in ModifyUnblockedDamageTargetArgs a, CreatureRef v) =>
        {
            log.Add($"redirect{a.OriginalTarget.Index}");
            return v;
        }))
        .On(HookId.ModifyHpLostAfterOsty, (HookFold<ModifyHpLostArgs, decimal>)((HookCtx _, ListenerRef s, in ModifyHpLostArgs a,
            decimal v) => Note(log, "lostAfter", a.Target, v)))
        .On(HookId.AfterBlockBroken, Ev<AfterBlockBrokenArgs>(log, a => $"broken{a.Creature.Index}"))
        .On(HookId.AfterCurrentHpChanged, Ev<AfterCurrentHpChangedArgs>(log, a => $"hp{a.Creature.Index}"))
        .On(HookId.AfterDamageGiven, Ev<AfterDamageGivenArgs>(log, a => $"given{a.Target.Index}"))
        .On(HookId.AfterDamageReceived, Ev<AfterDamageReceivedArgs>(log, a => $"received{a.Target.Index}"));

    private static decimal Note(List<string> log, string what, CreatureRef? target, decimal v)
    {
        log.Add($"{what}{target?.Index}");
        return v;
    }

    private static HookEvent<T> Ev<T>(List<string> log, Func<T, string> what) where T : struct =>
        (HookCtx _, ListenerRef self, in T a, ChoiceCtx? c) =>
        {
            log.Add(what(a));
            return HookStep.Done;
        };

    [Fact]
    public void EachTargetRunsTheGamesSequenceThenTheResultsRunTheirsInOrderThenOneKill()
    {
        // game: CreatureCmd.Damage: per target ModifyDamage (3 passes), AfterModifyingDamageAmount (the relic modified
        // nothing: not told), BeforeDamageReceived, ModifyHpLost before Osty, ModifyUnblockedDamageTarget, ModifyHpLost
        // after Osty; after every target, per result AfterBlockBroken, AfterCurrentHpChanged, AfterDamageGiven,
        // AfterDamageReceived (not for a killed one); then Kill once with the killed (MIGRATION_F08.md section 1).
        var s = Combat(5, 40);
        s.Enemies[1].Block = 2;
        var log = new List<string>();
        var k = K(s, Recorder(log));
        var results = CreatureCmd.Damage(k, [OldCreature.Of(s.Enemies[0]), OldCreature.Of(s.Enemies[1])], 6m,
            ValueProps.PoweredAttack, OldCreature.Player, null);
        string[] per(int i) => [$"add{i}", $"mul{i}", $"cap{i}", $"before{i}", $"lostBefore{i}", $"redirect{i}", $"lostAfter{i}"];
        Assert.Equal([.. per(1), .. per(2), "hp1", "given1", "broken2", "hp2", "given2", "received2"], log);
        Assert.Equal(2, results.Count);
        Assert.Equal(1, k.Commands.KillCalls);
        Assert.Equal([OldCreature.Of(s.Enemies[0])], Killed.Value!);
        Assert.Equal((0, 36), (s.Enemies[0].Hp, s.Enemies[1].Hp));
    }

    [Fact]
    public void ADeadTargetIsSkippedAndADeadDealerDealsNothing()
    {
        // game: CreatureCmd.Damage: `originalTarget.IsDead` continue; a dead dealer returns one empty result per target
        // before any hook (and before Kill).
        var s = Combat(0, 40);
        var log = new List<string>();
        var k = K(s, Recorder(log));
        var r = CreatureCmd.Damage(k, [OldCreature.Of(s.Enemies[0]), OldCreature.Of(s.Enemies[1])], 4m,
            ValueProps.PoweredAttack, OldCreature.Player, null);
        Assert.Equal(OldCreature.Of(s.Enemies[1]), Assert.Single(r).Receiver);
        s.PlayerHp = 0;
        var none = CreatureCmd.Damage(k, [OldCreature.Of(s.Enemies[1])], 4m, ValueProps.PoweredAttack, OldCreature.Player, null);
        Assert.Equal(0, Assert.Single(none).UnblockedDamage);
        Assert.Equal(1, k.Commands.KillCalls);
        Assert.Equal(36, s.Enemies[1].Hp);
    }

    [Fact]
    public void KillIsReachedOncePerDamageCallAndAnswersADeath()
    {
        // Damage always reaches Kill once (game: `await Kill(killedCreatures)`, empty or not); since R3a the kill runs
        // (KillCommandTests): the dead enemy leaves the roster.
        var s = Combat(5);
        var k = K(s, _ => { }, DamageOnly, countKills: false);
        Hit(k, OldCreature.Of(s.Enemies[0]), 1m);
        Hit(k, OldCreature.Of(s.Enemies[0]), 1m);
        Assert.Equal(2, k.Commands.KillCalls);
        var enemy = s.Enemies[0];
        Hit(k, OldCreature.Of(enemy), 9m);
        Assert.Equal(3, k.Commands.KillCalls);
        Assert.DoesNotContain(enemy, s.Enemies);
    }

    // ---------------------------------------------------------------------------------------- AttackCmd

    [Fact]
    public void TwinStrikeStopsAtTheSecondHitWhenTheFirstKilledItsTarget()
    {
        // game: AttackCommand.Execute: per hit the living possible targets; none in a live combat: break. Kill ran inside
        // hit 1's Damage (section 1's Twin Strike trace): one Damage call, one Kill.
        var s = Combat(5, 40);
        var k = K(s, _ => { }, DamageOnly);
        var a = new AttackCmd(OldCreature.Player, 6m, 2) { SingleTarget = OldCreature.Of(s.Enemies[0]) }.Execute(k);
        Assert.Equal(1, a.Hits);
        Assert.Equal(1, k.Commands.KillCalls);
        Assert.Equal([OldCreature.Of(s.Enemies[0])], Killed.Value!);
        Assert.Equal(40, s.Enemies[1].Hp);
    }

    [Fact]
    public void WhirlwindHitsAllLivingOpponentsPerHitInOneCallAndStopsWhenNoneIsLeft()
    {
        // game: AttackCommand.Execute: an all-opponents hit is one CreatureCmd.Damage over every valid target (single when
        // one is left); the loop stops when the last one died.
        var s = Combat(5, 12);
        var k = K(s, _ => { }, DamageOnly);
        var a = new AttackCmd(OldCreature.Player, 6m, 3) { AllOpponents = true }.Execute(k);
        Assert.Equal(2, a.Hits);
        Assert.Equal([2, 1], a.Results.Select(r => r.Count));
        Assert.Equal(2, k.Commands.KillCalls);
        Assert.Equal([OldCreature.Of(s.Enemies[0]), OldCreature.Of(s.Enemies[1])], Killed.Value!);
    }

    [Fact]
    public void AnAttackOnAnOverCombatOrByADeadAttackerDoesNothing()
    {
        // game: AttackCommand.Execute returns before BeforeAttack when CombatManager.IsOverOrEnding or the attacker is
        // dead.
        var s = Combat(0);
        int before = 0, after = 0;
        var bridge = new AttackBridge(_ => before++, _ => after++);
        var k = K(s, _ => { }, DamageOnly);
        Assert.Equal(0, new AttackCmd(OldCreature.Player, 6m) { AllOpponents = true, Bridge = bridge }.Execute(k).Hits);
        s.Enemies.Add(new EnemyState { DefId = 16, Hp = 10, MaxHp = 10 });
        Assert.Equal(0, new AttackCmd(OldCreature.Of(s.Enemies[0]), 6m) { AllOpponents = true, Bridge = bridge }.Execute(k).Hits);
        Assert.Equal((0, 0, 50, 0), (before, after, s.PlayerHp, k.Commands.KillCalls));
    }

    [Fact]
    public void WithF06OffTheBridgesOldPointsRunOncePerAttackAndNoHookIsDispatched()
    {
        // MIGRATION_F08.md section 3 item 1 (the F06 bridge): F08 on, F06 off. A two-hit card attack, a three-hit
        // Whirlwind and a monster's three-hit move each run BeforeAttack's and AfterAttack's old point once, with the
        // count given (game: AttackCommand.Execute's one BeforeAttack, ModifyAttackHitCount and AfterAttack).
        var s = Combat(100, 100);
        int before = 0, after = 0, landed = -1;
        var k = K(s, _ => { }, DamageOnly);
        var bridge = new AttackBridge(_ => before++, a =>
        {
            after++;
            landed = a.Results.SelectMany(r => r).Count(r => r.UnblockedDamage > 0);
        });
        new AttackCmd(OldCreature.Player, 3m, 2) { SingleTarget = OldCreature.Of(s.Enemies[0]), Bridge = bridge }.Execute(k);
        Assert.Equal((1, 1, 2), (before, after, landed));
        new AttackCmd(OldCreature.Player, 3m, 3) { AllOpponents = true, Bridge = bridge }.Execute(k);
        Assert.Equal((2, 2, 6), (before, after, landed));
        new AttackCmd(OldCreature.Of(s.Enemies[1]), 2m, 3) { AllOpponents = true, Bridge = bridge }.Execute(k);
        Assert.Equal((3, 3, 3), (before, after, landed));
        Assert.Equal(44, s.PlayerHp);
    }

    [Fact]
    public void WithF06OnTheAttackDispatchesItsHooksOnceAndTheBridgeIsGatedOff()
    {
        // F06 on as well: BeforeAttack, ModifyAttackHitCount and AfterAttack through Hooks (game: AttackCommand.Execute),
        // once per attack; the bridge's old points do not run; the modified count is the loop's.
        var s = Combat(100);
        var log = new List<string>();
        int bridged = 0;
        KernelState? k0 = null;
        var k = K(s, t => t.Model(ListenerKind.Relic, RelicName(1))
            .On(HookId.BeforeAttack, Ev<BeforeAttackArgs>(log, _ => "beforeAttack"))
            .On(HookId.ModifyAttackHitCount, (HookFold<ModifyAttackHitCountArgs, int>)((HookCtx _, ListenerRef self,
                in ModifyAttackHitCountArgs a, int v) => v + 1))
            .On(HookId.AfterAttack, Ev<AfterAttackArgs>(log, a => $"afterAttack{k0!.Commands.Attack(a.Command).Hits}")));
        k0 = k;
        var a = new AttackCmd(OldCreature.Player, 3m, 2)
        {
            SingleTarget = OldCreature.Of(s.Enemies[0]),
            Bridge = new AttackBridge(_ => bridged++, _ => bridged++),
        }.Execute(k);
        Assert.Equal(3, a.Hits);
        Assert.Equal(["beforeAttack", "afterAttack3"], log);
        Assert.Equal(0, bridged);
        Assert.Equal(3, k.Commands.KillCalls);
    }

}
