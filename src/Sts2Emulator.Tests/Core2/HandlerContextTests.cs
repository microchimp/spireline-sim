// The handler context (docs/rebuild/TIER1_KERNEL.md section 4) and the worked example the handler batches copy
// (docs/rebuild/MIGRATION_F08.md section 5, "How to write a handler"): test-only handlers for two real powers, registered
// under the F08+F10 switch and run by the generated dispatcher over an old CombatState. A handler reaches its own model
// only through the HookCtx it is given. No model is migrated here.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class HandlerContextTests
{
    private static readonly OldModelIds Ids = ListenerOrderTests.Ids;
    private static readonly Switch[] DamageOnly = [Switch.F08, Switch.F10];

    // ---------------------------------------------------------------------------------------- the template

    /// <summary>
    /// game: StrengthPower.ModifyDamageAdditive. A fold-add: the addend, which is the power's Amount for its owner's
    /// powered attacks and 0 otherwise. The shape every handler has: the context first (the only way to the combat),
    /// the listener it runs for, the dispatcher's argument record (read-only), then the fold's running value. It reads its
    /// own amount and owner through the context (<c>c.Power(self)</c>), never by counting on the old representation.
    /// </summary>
    private static decimal StrengthAdditive(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Dealer != me.Owner || !a.Props.IsPoweredAttack())
            return 0m;
        return me.Amount;
    }

    /// <summary>
    /// game: VulnerablePower.ModifyDamageMultiplicative. A fold-mul: the factor, 1 when the power does not apply. Where the
    /// old engine already has the body as an extracted function (<c>BuffSystem.Vulnerable_ModifyDamageMultiplicative</c>,
    /// round R5), the handler keeps the game's condition and calls it with the old buff lists the context hands over
    /// (<c>c.OldBuffs</c>, Tier 1 only).
    /// </summary>
    private static decimal VulnerableMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack())
            return 1m;
        var dealer = a.Dealer is { } d ? c.OldBuffs(d) : [];
        return (decimal)BuffSystem.Vulnerable_ModifyDamageMultiplicative(dealer, c.OldBuffs(me.Owner), 0f);
    }

    private static void Register(HandlerTable t)
    {
        t.Model(ListenerKind.Power, "StrengthPower").On(HookId.ModifyDamageAdditive, (HookFold<ModifyDamageArgs, decimal>)StrengthAdditive);
        t.Model(ListenerKind.Power, "VulnerablePower")
            .On(HookId.ModifyDamageMultiplicative, (HookFold<ModifyDamageArgs, decimal>)VulnerableMultiplicative);
    }

    private static CombatState Combat()
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = [new EnemyState { DefId = 16, Hp = 40, MaxHp = 40 }, new EnemyState { DefId = 16, Hp = 40, MaxHp = 40 }];
        return s;
    }

    private static KernelState K(CombatState s, IEnumerable<Switch>? on = null) =>
        new Kernel(new KernelOptions(on ?? DamageOnly), () => Ids, [Register]).Bind(s);

    private static ModifyDamageArgs Attack(int target, decimal damage, ValueProp? props = null, int? dealer = 0) =>
        new(new CreatureRef(target), dealer is { } d ? new CreatureRef(d) : null, damage, props ?? ValueProps.PoweredAttack, null,
            ModifyDamageHookType.All, PreviewModes.None);

    private static ListenerRef Power(string name, int creature, int instance = 0) =>
        new((int)ListenerKind.Power, Ids.Resolve(ListenerKind.Power, name), OldState.PowerInstance(creature, instance));

    // ---------------------------------------------------------------------------------------- the worked example

    [Fact]
    public void TheFoldReadsEachPowersOwnStateThroughTheContextAndReturnsTheGamesValue()
    {
        // Strength 3 on the player (the dealer), Vulnerable on the first enemy (creature 1): 6 + 3 = 9 (the additive pass),
        // 9 x 1.5 = 13.5 (the multiplicative pass), kept as a decimal for the caller's one cast. Both listeners changed it.
        var s = Combat();
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 3));
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, 2));
        var k = K(s);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed);

        Assert.Equal(13.5m, v);
        Assert.Equal([Power("StrengthPower", 0), Power("VulnerablePower", 1)], changed);
    }

    [Fact]
    public void AHandlerAnswersTheIdentityWhereItsPowerDoesNotApply()
    {
        // Strength is the player's: an enemy dealer gets nothing from it. Vulnerable is the first enemy's: a hit on the
        // second enemy, or an unpowered one (thorns, poison), takes no factor. Nothing changed, so the changed set is empty.
        var s = Combat();
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 3));
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, 2));
        var k = K(s);

        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 2, damage: 6m, dealer: null), out changed));
        Assert.Empty(changed);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m, new ValueProp(ValueProps.Unpowered)), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TheAmountIsTheOldStatesNow()
    {
        // The handler reads the state at the call, not at registration or at the pass's snapshot: a stack applied between
        // two calls changes the next value.
        var s = Combat();
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 1));
        var k = K(s);
        Assert.Equal(7m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out _));
        s.PlayerBuffs[0] = new BuffState(BuffId.Strength, 4);
        Assert.Equal(10m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out _));
    }

    [Fact]
    public void WithNoSwitchOnNoHandlerIsRegisteredAndTheDispatcherTakesTheFastPath()
    {
        var s = Combat();
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 3));
        var k = K(s, []);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Empty(changed);
    }

    // ---------------------------------------------------------------------------------------- the context itself

    [Fact]
    public void TheContextFindsAListenersOwnerAndAmountAndNothingForOneThatIsNotThere()
    {
        var s = Combat();
        s.Relics = [new RelicInstance(1)];
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 3));
        s.Enemies[1].Buffs.Add(new BuffState(BuffId.Vulnerable, 2));
        var c = K(s).Ctx;

        Assert.Equal(new PowerView(new CreatureRef(0), 3m), c.Power(Power("StrengthPower", 0)));
        Assert.Equal(new PowerView(new CreatureRef(2), 2m), c.Power(Power("VulnerablePower", 2)));
        Assert.Equal(new CreatureRef(2), c.Owner(Power("VulnerablePower", 2)));
        Assert.Equal(new CreatureRef(0), c.Owner(new ListenerRef((int)ListenerKind.Relic, 1, 0)));
        // Not there: Vulnerable is on the other enemy; a second Strength instance; a creature that does not exist.
        Assert.Null(c.Power(Power("VulnerablePower", 1)));
        Assert.Null(c.Power(Power("StrengthPower", 0, instance: 1)));
        Assert.Null(c.Power(Power("StrengthPower", 7)));
        Assert.Null(c.Owner(new ListenerRef((int)ListenerKind.Relic, 1, 5)));
        // The old buff list a creature carries.
        Assert.Same(s.PlayerBuffs, c.OldBuffs(new CreatureRef(0)));
        Assert.Same(s.Enemies[1].Buffs, c.OldBuffs(new CreatureRef(2)));
    }

    [Fact]
    public void ARuntimeThatIsNoKernelStateGivesAHandlerNoState()
    {
        var c = new NoState().Ctx;
        Assert.Throws<InvalidOperationException>(() => c.K);
        Assert.Throws<InvalidOperationException>(() => c.Power(Power("StrengthPower", 0)));
    }

    private sealed class NoState : HookRuntime
    {
        public override bool HasAny(HookId hook) => false;

        public override HookCursor<THandler> Listeners<THandler>(HookIterator iterator, HookId hook) =>
            HookCursor<THandler>.Empty();

        public override ChoiceCtx NewHookContext(HookId dispatcher, ListenerRef who) => new(dispatcher, who);

        public override void OnPaused(in PauseSite site, HookStep step)
        {
        }

        public override long HookActionMark() => 0;

        public override void WaitAll(HookId dispatcher, long mark)
        {
        }
    }
}
