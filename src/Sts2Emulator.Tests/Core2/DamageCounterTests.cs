// The old state's damage history owned by Core2's commands, and AttackCmd's per-hit seams (docs/rebuild/MIGRATION_F08.md,
// "Fixes after R2's review"). One test per counter, each read from the game reader that the old field stands for; the
// game's code is cited by class and method only.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class DamageCounterTests
{
    private static readonly Switch[] DamageOnly = [Switch.F08, Switch.F10];

    private static string RelicName(int id)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (!File.Exists(Path.Combine(dir!.FullName, "data", "id_map.json")))
            dir = dir.Parent;
        var map = JsonNode.Parse(File.ReadAllText(Path.Combine(dir.FullName, "data", "id_map.json")))!.AsObject();
        return map["relics"]!.AsObject().First(kv => (int)kv.Value! == id).Key;
    }

    private static CombatState Combat(params int[] enemyHp)
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = enemyHp.Select(hp => new EnemyState { DefId = 16, Hp = hp, MaxHp = hp }).ToList();
        s.Relics = [new RelicInstance(1)];
        return s;
    }

    private static KernelState K(CombatState s, Action<HandlerTable>? models = null)
    {
        var k = new Kernel(new KernelOptions(DamageOnly), () => ListenerOrderTests.Ids, [models ?? (_ => { })]).Bind(s);
        k.Commands.KillStandIn = (_, _) => { };
        return k;
    }

    private static OldCreature E(CombatState s, int i = 0) => OldCreature.Of(s.Enemies[i]);

    private static IReadOnlyList<DamageResult> Hit(KernelState k, OldCreature target, decimal amount, OldCreature dealer,
        ValueProp? props = null, CardHandle? card = null) =>
        CreatureCmd.Damage(k, [target], amount, props ?? ValueProps.PoweredAttack, dealer, card);

    private static readonly ValueProp SelfCost = new(ValueProps.Unblockable | ValueProps.Unpowered);

    // ---------------------------------------------------------------------------------------- the history counters

    [Fact]
    public void PlayerHpLostThisTurnIsSpitesQueryOverTheTurnsEntries()
    {
        // game: Spite reads DamageReceivedEntry (receiver the owner, UnblockedDamage > 0, HappenedThisTurn: the round and
        // the side). An enemy's hit on the player's turn counts, a fully blocked one does not, and an entry stamped on
        // the enemy's side turn is not this (player) turn's.
        var s = Combat(30);
        var k = K(s);
        Hit(k, OldCreature.Player, 4m, E(s));
        Assert.Equal(4, s.PlayerHpLostThisTurn);
        s.PlayerBlock = 10;
        Hit(k, OldCreature.Player, 5m, E(s));
        Assert.Equal(4, s.PlayerHpLostThisTurn);
        s.PlayerBlock = 0;
        s.PlayerTurn = false;
        Hit(k, OldCreature.Player, 6m, E(s));
        Assert.Equal(4, s.PlayerHpLostThisTurn);
    }

    [Fact]
    public void UnblockedDamageHitCountCountsEveryEntryTheGameCounts()
    {
        // game: Tear Asunder counts DamageReceivedEntry with receiver the owner and UnblockedDamage > 0, whatever dealt it
        // and whatever the props: a monster's hit and the player's own Unblockable cost both count, a blocked hit does not.
        var s = Combat(30);
        var k = K(s);
        Hit(k, OldCreature.Player, 4m, E(s));
        Hit(k, OldCreature.Player, 3m, OldCreature.Player, SelfCost);
        s.PlayerBlock = 20;
        Hit(k, OldCreature.Player, 5m, E(s));
        Assert.Equal(2, s.UnblockedDamageHitCount);
    }

    [Fact]
    public void TookUnblockedDamageIsLavaLampsLatchAndRefusesUnblockable()
    {
        // game: LavaLamp.AfterDamageReceived: the owner, UnblockedDamage > 0, and not ValueProp.Unblockable.
        var s = Combat(30);
        var k = K(s);
        Hit(k, OldCreature.Player, 3m, OldCreature.Player, SelfCost);
        Assert.False(s.TookUnblockedDamage);
        s.PlayerBlock = 9;
        Hit(k, OldCreature.Player, 5m, E(s));
        Assert.False(s.TookUnblockedDamage);
        Hit(k, OldCreature.Player, 12m, E(s));
        Assert.True(s.TookUnblockedDamage);
    }

    [Fact]
    public void UnblockedDamageThisTurnIsBeatingRemnantsTally()
    {
        // game: BeatingRemnant.AfterDamageReceived adds the result's UnblockedDamage on any damage to its owner.
        var s = Combat(30);
        var k = K(s);
        Hit(k, OldCreature.Player, 3m, E(s));
        Hit(k, OldCreature.Player, 4m, OldCreature.Player, SelfCost);
        Assert.Equal(7, s.UnblockedDamageThisTurn);
    }

    [Fact]
    public void HardenedShellDamageThisTurnAddsTheUnblockedDamageOfItsOwnerOnly()
    {
        // game: HardenedShellPower.AfterDamageReceived (target its owner, not fully blocked): += UnblockedDamage.
        var s = Combat(60, 60);
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.HardenedShell, 20));
        var k = K(s);
        Hit(k, E(s, 0), 6m, OldCreature.Player);
        Hit(k, E(s, 1), 7m, OldCreature.Player);
        s.Enemies[0].Block = 10;
        Hit(k, E(s, 0), 8m, OldCreature.Player);
        Assert.Equal(6, s.Enemies[0].HardenedShellDamageThisTurn);
        Assert.Equal(0, s.Enemies[1].HardenedShellDamageThisTurn);
        Hit(k, E(s, 0), 20m, OldCreature.Player);
        Assert.Equal(6 + 18, s.Enemies[0].HardenedShellDamageThisTurn);
    }

    [Fact]
    public void PoweredHitsThisTurnIsBeatIntoShapesCountOnTheTarget()
    {
        // game: Beat Into Shape counts DamageReceivedEntry with the target as receiver, the owner's creature as dealer, a
        // powered attack, this turn: one per result (a blocked hit too), none for an unpowered hit or another dealer.
        var s = Combat(60, 60);
        var k = K(s);
        Hit(k, E(s, 0), 5m, OldCreature.Player);
        s.Enemies[0].Block = 50;
        Hit(k, E(s, 0), 5m, OldCreature.Player);
        Hit(k, E(s, 0), 5m, OldCreature.Player, new ValueProp(ValueProps.Move | ValueProps.Unpowered));
        Hit(k, E(s, 0), 5m, E(s, 1));
        Hit(k, E(s, 1), 5m, OldCreature.Player);
        Assert.Equal(2, s.Enemies[0].PoweredHitsThisTurn);
        Assert.Equal(1, s.Enemies[1].PoweredHitsThisTurn);
    }

    [Fact]
    public void PlayerHitTurnIsEmotionChipsStampForAnyEntryNotFullyBlocked()
    {
        // game: EmotionChip reads DamageReceivedEntry (receiver the owner, not WasFullyBlocked, HappenedLastPlayerTurn):
        // the stamp is the player turn (Turn + 1) the entry lands in; a fully blocked hit stamps nothing.
        var s = Combat(30);
        s.Turn = 2;
        var k = K(s);
        s.PlayerBlock = 10;
        Hit(k, OldCreature.Player, 4m, E(s));
        Assert.False(RelicEffects.PlayerHitInTurn(s, 3));
        Hit(k, OldCreature.Player, 14m, E(s));
        Assert.True(RelicEffects.PlayerHitInTurn(s, 3));
        s.Turn = 3;
        Hit(k, OldCreature.Player, 2m, E(s));
        Assert.True(RelicEffects.PlayerHitInTurn(s, 3));
        Assert.True(RelicEffects.PlayerHitInTurn(s, 4));
    }

    [Fact]
    public void OstyAttacksThisTurnIsOneEntryPerCommandAfterItsLastHit()
    {
        // game: AttackCommand.Execute adds one CreatureAttackedEntry after its hits; Flatten and Rattle read Osty's.
        var s = Combat(60);
        s.OstyHp = 10;
        s.OstyMaxHp = 10;
        var k = K(s);
        new AttackCmd(OldCreature.Osty, 2m, 3) { SingleTarget = E(s) }.Execute(k);
        Assert.Equal(1, s.OstyAttacksThisTurn);
        new AttackCmd(OldCreature.Player, 2m, 2) { SingleTarget = E(s) }.Execute(k);
        Assert.Equal(1, s.OstyAttacksThisTurn);
    }

    // ---------------------------------------------------------------------------------------- Rupture

    [Fact]
    public void RuptureBanksTheCostOfTheCardBeingPlayedAndPaysAnyOtherHpLossAtOnce()
    {
        // game: RupturePower.AfterDamageReceived: the card source is a card being played (BeforeCardPlayed added it) ->
        // the amount is banked until AfterCardPlayed; otherwise Strength at once. The old body is called with that.
        var s = Combat(30);
        s.PlayerBuffs.Add(new BuffState(BuffId.RupturePower, 2));
        var card = s.Fresh(new CardInstance(IC.StrikeIronclad, false));
        var k = K(s);
        using (k.World!.Flying(card))
            Hit(k, OldCreature.Player, 3m, OldCreature.Player, SelfCost, new CardHandle(card.Uid));
        Assert.Equal(2, s.RuptureOwed);
        Assert.Equal(0, BuffSystem.Get(s.PlayerBuffs, BuffId.Strength));
        Hit(k, OldCreature.Player, 3m, OldCreature.Player, SelfCost);
        Assert.Equal(2, s.RuptureOwed);
        Assert.Equal(2, BuffSystem.Get(s.PlayerBuffs, BuffId.Strength));
    }

    // ---------------------------------------------------------------------------------------- AttackCmd

    [Fact]
    public void RandomTargetingDrawsTheCombatTargetsStreamOncePerHit()
    {
        // game: AttackCommand.Execute: Rng.CombatTargets.NextItem(validTargets) per hit, after the duplicates filter,
        // drawn even over one candidate.
        var s = Combat(1000, 1000, 1000);
        s.TargetRng = new CountingRandom(77);
        var twin = new CountingRandom(77);
        var k = K(s);
        var a = new AttackCmd(OldCreature.Player, 1m, 5) { RandomlyTargeted = true }.Execute(k);
        Assert.Equal(5, s.TargetRng.CallCount);
        for (int i = 0; i < 5; i++)
            Assert.Equal(E(s, twin.Next(3)), Assert.Single(a.Results[i]).Receiver);

        var one = Combat(1000);
        one.TargetRng = new CountingRandom(5);
        new AttackCmd(OldCreature.Player, 1m, 2) { RandomlyTargeted = true }.Execute(K(one));
        Assert.Equal(2, one.TargetRng.CallCount);
    }

    [Fact]
    public void RandomTargetingWithoutDuplicatesFiltersBeforeTheDrawAndThrowsWhenNoneIsLeft()
    {
        // game: the filter drops creatures already in the attack's results, then the draw is over the rest; with none
        // left the game throws.
        var s = Combat(1000, 1000, 1000);
        s.TargetRng = new CountingRandom(9);
        var twin = new CountingRandom(9);
        var k = K(s);
        var a = new AttackCmd(OldCreature.Player, 1m, 3) { RandomlyTargeted = true, RandomAllowsDuplicates = false }.Execute(k);
        var left = new List<int> { 0, 1, 2 };
        for (int i = 0; i < 3; i++)
        {
            int pick = left[twin.Next(left.Count)];
            left.Remove(pick);
            Assert.Equal(E(s, pick), Assert.Single(a.Results[i]).Receiver);
        }

        Assert.Equal(3, s.TargetRng.CallCount);
        var again = Combat(1000, 1000);
        again.TargetRng = new CountingRandom(9);
        Assert.Throws<InvalidOperationException>(() =>
            new AttackCmd(OldCreature.Player, 1m, 3) { RandomlyTargeted = true, RandomAllowsDuplicates = false }.Execute(K(again)));
    }

    [Fact]
    public void AnEnemyAttackerHasNoCombatTargetsStream()
    {
        var s = Combat(30);
        s.TargetRng = new CountingRandom(1);
        Assert.Throws<InvalidOperationException>(() =>
            new AttackCmd(E(s), 1m) { RandomlyTargeted = true }.Execute(K(s)));
        Assert.Equal(0, s.TargetRng.CallCount);
    }

    [Fact]
    public void BeforeDamageRunsPerHitAfterTheTargetPickAndBeforeTheAmountIsCalculated()
    {
        // game: AttackCommand.Execute: pick the target, `_beforeDamage()`, then Calculate(single) and Damage.
        var s = Combat(1000, 1000);
        s.TargetRng = new CountingRandom(3);
        var log = new List<string>();
        var k = K(s);
        new AttackCmd(OldCreature.Player, 1m, 2)
        {
            RandomlyTargeted = true,
            BeforeDamage = () => log.Add($"before:{s.TargetRng.CallCount}"),
            Calculate = _ => { log.Add("calc"); return 1m; },
        }.Execute(k);
        Assert.Equal(new[] { "before:1", "calc", "before:2", "calc" }, log);
    }

    [Fact]
    public void CalculateRunsPerHitWithTheSingleTargetOrNull()
    {
        // game: CalculatedDamageVar.Calculate(singleTarget) per hit: the target when the hit has one, null when the hit
        // goes to every valid target. A lone living opponent is a single target even for an all-opponents attack.
        var s = Combat(100);
        var seen = new List<OldCreature?>();
        int n = 0;
        var k = K(s);
        new AttackCmd(OldCreature.Player, 0m, 3) { SingleTarget = E(s), Calculate = t => { seen.Add(t); return 3m + n++; } }.Execute(k);
        Assert.Equal(new OldCreature?[] { E(s), E(s), E(s) }, seen);
        Assert.Equal(100 - (3 + 4 + 5), s.Enemies[0].Hp);

        var two = Combat(100, 100);
        seen.Clear();
        new AttackCmd(OldCreature.Player, 0m, 2) { AllOpponents = true, Calculate = t => { seen.Add(t); return 5m; } }.Execute(K(two));
        Assert.Equal(new OldCreature?[] { null, null }, seen);
        Assert.Equal(new[] { 90, 90 }, two.Enemies.Select(e => e.Hp).ToArray());

        var lone = Combat(100);
        seen.Clear();
        new AttackCmd(OldCreature.Player, 0m) { AllOpponents = true, Calculate = t => { seen.Add(t); return 5m; } }.Execute(K(lone));
        Assert.Equal(new OldCreature?[] { E(lone) }, seen);
    }

    [Fact]
    public void ZeroDamageIntoBlockIsFullyBlockedAndAfterDamageReceivedStillFires()
    {
        // game: CreatureCmd.Damage: 0 damage against standing block has blocked 0 but WasFullyBlocked (the block is
        // there and no HP is lost); the result is not killed, so AfterDamageReceived runs, and block is untouched.
        var s = Combat(30);
        s.PlayerBlock = 10;
        int fired = 0;
        var k = K(s, t => t.Model(ListenerKind.Relic, RelicName(1)).On(HookId.AfterDamageReceived,
            (HookEvent<AfterDamageReceivedArgs>)((ListenerRef self, in AfterDamageReceivedArgs a, ChoiceCtx? c) =>
            {
                fired++;
                return HookStep.Done;
            })));
        var r = Assert.Single(Hit(k, OldCreature.Player, 0m, E(s)));
        Assert.Equal(1, fired);
        Assert.True(r.WasFullyBlocked);
        Assert.False(r.WasBlockBroken);
        Assert.Equal(0, r.UnblockedDamage);
        Assert.Equal(10, s.PlayerBlock);
        Assert.Equal(50, s.PlayerHp);
        Assert.False(s.TookUnblockedDamage);
        Assert.Equal(0, s.UnblockedDamageHitCount);
    }


}
