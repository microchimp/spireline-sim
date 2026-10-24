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
    /// (<c>c.OldBuffs</c>, Tier 1 only) and the combat state (<c>c.OldCombat</c>, Tier 1 only) for the one input it does not
    /// carry itself: the relics' shift of the multiplier, which the old fold takes as <c>vulnerableDelta</c> (the site passes
    /// <c>RelicEffects.VulnerableMultiplierDeltaAgainstEnemies(state)</c>). A batch passes every such input the same way.
    /// </summary>
    private static decimal VulnerableMultiplicative(HookCtx c, ListenerRef self, in ModifyDamageArgs a, decimal current)
    {
        if (c.Power(self) is not { } me || a.Target != me.Owner || !a.Props.IsPoweredAttack())
            return 1m;
        var dealer = a.Dealer is { } d ? c.OldBuffs(d) : [];
        return (decimal)BuffSystem.Vulnerable_ModifyDamageMultiplicative(
            dealer, c.OldBuffs(me.Owner), RelicEffects.VulnerableMultiplierDeltaAgainstEnemies(c.OldCombat));
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
    public void TheVulnerableTemplateTakesTheRelicDeltaTheOldSitePasses()
    {
        // Paper Phrog: Vulnerable on an enemy is 1.75 where it is 1.5, the shift the old fold takes as vulnerableDelta.
        // A handler that passed 0 would answer 6 x 1.5 = 9 here. The old fold's own answer for the state is the oracle.
        var s = Combat();
        s.Relics = [new RelicInstance(RelicEffects.PaperPhrog)];
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, 2));
        var k = K(s);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed);

        Assert.Equal(10.5m, v);
        Assert.Equal(
            BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs, 1f,
                vulnerableDelta: RelicEffects.VulnerableMultiplierDeltaAgainstEnemies(s)),
            (int)v);
        Assert.Equal([Power("VulnerablePower", 1)], changed);
    }

    [Fact]
    public void TheContextNamesTheOldCombatAndTheOldEnemyOfACreature()
    {
        var s = Combat();
        var c = K(s).Ctx;
        Assert.Same(s, c.OldCombat);
        Assert.Same(s.Enemies[0], c.Enemy(new CreatureRef(1)));
        Assert.Same(s.Enemies[1], c.Enemy(new CreatureRef(2)));
        Assert.Null(c.Enemy(new CreatureRef(0))); // the player has no enemy state
        Assert.Null(c.Enemy(new CreatureRef(3))); // nobody there
        s.OstyMaxHp = s.OstyHp = 10; // Osty is creature 1 and pushes the enemies up
        Assert.Null(c.Enemy(new CreatureRef(1)));
        Assert.Same(s.Enemies[0], c.Enemy(new CreatureRef(2)));
    }

    private static CardInstance Card(CombatState s, int def, int enchant = 0, bool bound = false) =>
        s.Fresh(new CardInstance(def, false, Enchantment: (Enchantment)enchant, EnchantAmount: enchant == 0 ? 0 : 1,
            Bound: bound));

    private static ListenerRef CardListener(CardInstance x) => new((int)ListenerKind.Card, x.DefId, x.Uid);

    [Fact]
    public void TheContextFindsACardByItsHandleAndAListenersOwnCopyByItsUid()
    {
        // Three copies of one model (def 5) in three piles: a handler acts on the copy it runs for.
        var s = Combat();
        var a = Card(s, 5);
        var b = Card(s, 5);
        var d = Card(s, 5, enchant: (int)Enchantment.Sharp);
        var other = Card(s, 6);
        s.Hand = [other, a];
        s.DrawPile = [b];
        s.DiscardPile = [Card(s, 6), d];
        var k = K(s);
        var c = k.Ctx;

        Assert.Equal(new CardView(new CardHandle(a.Uid), 5, CardPile.Hand, 1), c.Card(new CardHandle(a.Uid)));
        Assert.Equal(new CardView(new CardHandle(b.Uid), 5, CardPile.Draw, 0), c.Card(new CardHandle(b.Uid)));
        Assert.Equal(new CardView(new CardHandle(d.Uid), 5, CardPile.Discard, 1), c.Card(new CardHandle(d.Uid)));
        Assert.Null(c.Card(new CardHandle(999)));
        Assert.Null(c.Card(new CardHandle(0)));

        Assert.Equal(new CardView(new CardHandle(a.Uid), 5, CardPile.Hand, 1), c.Card(CardListener(a)));
        Assert.Equal(new CardView(new CardHandle(b.Uid), 5, CardPile.Draw, 0), c.Card(CardListener(b)));
        // The enchantment listener sits on its card: the view is the card's.
        Assert.Equal(new CardView(new CardHandle(d.Uid), 5, CardPile.Discard, 1),
            c.Card(new ListenerRef((int)ListenerKind.Enchantment, (int)Enchantment.Sharp, d.Uid)));
        Assert.Null(c.Card(CardListener(a) with { ModelId = 6 })); // another model than the card has
        Assert.Null(c.Card(new ListenerRef((int)ListenerKind.Relic, 5, a.Uid))); // not a card kind

        // The card in play is in Play at 0, not in the hand it may still be listed in.
        using (k.World.Flying(a))
            Assert.Equal(new CardView(new CardHandle(a.Uid), 5, CardPile.Play, 0), c.Card(CardListener(a)));

        Assert.Equal(new CreatureRef(0), c.Owner(CardListener(b)));
    }

    [Fact]
    public void TheContextGivesARelicItsCounterAndNothingForOneNotHeld()
    {
        var s = Combat();
        s.Relics = [new RelicInstance(1), new RelicInstance(2, 7)];
        var k = K(s);
        var c = k.Ctx;
        Assert.Equal(new RelicView(2, 1, 7), c.Relic(new ListenerRef((int)ListenerKind.Relic, 2, 1)));
        Assert.Equal(new RelicView(1, 0, 0), c.Relic(new ListenerRef((int)ListenerKind.Relic, 1, 0)));
        Assert.Null(c.Relic(new ListenerRef((int)ListenerKind.Relic, 2, 0))); // another relic at that place
        Assert.Null(c.Relic(new ListenerRef((int)ListenerKind.Relic, 2, 5))); // no place
        Assert.Null(c.Relic(Power("StrengthPower", 0)));
        s.Relics[1] = new RelicInstance(2, 8); // the counter is the old state's now
        Assert.Equal(8, c.Relic(new ListenerRef((int)ListenerKind.Relic, 2, 1))!.Value.Counter);
        // A player whose hooks are deactivated is on no listener list: its relic is not held as far as a handler sees.
        k.World.DeactivatePlayerHooks();
        Assert.Null(c.Relic(new ListenerRef((int)ListenerKind.Relic, 1, 0)));
    }

    /// <summary>A combat with every kind of listener the lookup reads by position: two entries of one power on the player,
    /// powers on both enemies, relics, a potion, orbs, and cards in every pile with an enchantment and an affliction.</summary>
    private static CombatState Crowd()
    {
        var s = Combat();
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 3));
        s.PlayerBuffs.Add(new BuffState(BuffId.Dexterity, 1));
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 2)); // a second entry of the model: instance 1
        s.Enemies[0].Buffs.Add(new BuffState(BuffId.Vulnerable, 2));
        s.Enemies[1].Buffs.Add(new BuffState(BuffId.Weak, 1));
        s.Enemies[1].Buffs.Add(new BuffState(BuffId.Vulnerable, 1));
        s.Relics = [new RelicInstance(1), new RelicInstance(2, 5)];
        s.PotionSlots[1] = 4;
        s.Orbs = [new OrbState(OrbType.Lightning), new OrbState(OrbType.Frost)];
        s.Hand = [Card(s, 5, enchant: (int)Enchantment.Sharp), Card(s, 6, bound: true)];
        s.DrawPile = [Card(s, 7)];
        s.DiscardPile = [Card(s, 8), Card(s, 5)];
        s.ExhaustPile = [Card(s, 9)];
        return s;
    }

    private static readonly ListenerKind[] ReadByPosition =
    [
        ListenerKind.Power, ListenerKind.Relic, ListenerKind.Potion, ListenerKind.Orb, ListenerKind.Card,
        ListenerKind.Affliction, ListenerKind.Enchantment,
    ];

    [Fact]
    public void LocateAnswersEachListenerTheCombatListsWithItsOwnEntryAndNothingElse()
    {
        // The positional lookup (no listing built) against the listing itself, every entry, in the plain state and with a
        // card in flight; then with the player's hooks deactivated (the listing then holds the creatures' powers only).
        var w = K(Crowd()).World;
        var strength = Ids.Resolve(ListenerKind.Power, "StrengthPower");

        void Agree()
        {
            var all = new List<OldListener>();
            OldState.Combat(w, all, []);
            var read = all.Where(e => ReadByPosition.Contains(e.Kind)).ToList();
            Assert.Equal(ReadByPosition.Length, read.Select(e => e.Kind).Distinct().Count()); // every kind is in the crowd
            foreach (var e in read)
                Assert.Equal(e, OldState.Locate(w, e.Who));
            // Not listed: answered null, as the listing would not have it.
            Assert.Null(OldState.Locate(w, new((int)ListenerKind.Power, strength, OldState.PowerInstance(0, 2))));
            Assert.Null(OldState.Locate(w, new((int)ListenerKind.Relic, 1, 7)));
            Assert.Null(OldState.Locate(w, new((int)ListenerKind.Potion, 4, 0)));
            Assert.Null(OldState.Locate(w, new((int)ListenerKind.Orb, (int)OrbType.Frost, 0)));
            Assert.Null(OldState.Locate(w, new((int)ListenerKind.Card, 5, 0)));
        }

        Agree();
        using (w.Flying(w.Combat!.Hand[0]))
            Agree();

        w.DeactivatePlayerHooks();
        var left = new List<OldListener>();
        OldState.Combat(w, left, []);
        Assert.DoesNotContain(left, e => e.Kind is ListenerKind.Relic or ListenerKind.Card or ListenerKind.Potion);
        Assert.Null(OldState.Locate(w, new((int)ListenerKind.Relic, 1, 0)));
        Assert.Null(OldState.Locate(w, new((int)ListenerKind.Card, 7, w.Combat.DrawPile[0].Uid)));
        foreach (var e in left.Where(e => e.Kind == ListenerKind.Power))
            Assert.Equal(e, OldState.Locate(w, e.Who));
    }

    [Fact]
    public void ThePowerAndOwnerLookupsAllocateNothing()
    {
        var s = Crowd();
        var c = K(s).Ctx;
        var strength = Power("StrengthPower", 0, instance: 1);
        var relic = new ListenerRef((int)ListenerKind.Relic, 2, 1);
        var card = CardListener(s.ExhaustPile[0]);
        for (int i = 0; i < 200; i++)
        {
            c.Power(strength);
            c.Owner(relic);
            c.Owner(card);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        decimal sum = 0;
        for (int i = 0; i < 20_000; i++)
        {
            sum += c.Power(strength)!.Value.Amount;
            sum += c.Owner(relic)!.Value.Index + c.Owner(card)!.Value.Index;
        }

        long used = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(40_000m, sum);
        Assert.True(used < 4_000, $"a per-call allocation is back in Power or Owner ({used} bytes over 20000 calls)");
    }

    [Fact]
    public void ARuntimeThatIsNoKernelStateGivesAHandlerNoState()
    {
        var c = new NoState().Ctx;
        Assert.Throws<InvalidOperationException>(() => c.K);
        Assert.Throws<InvalidOperationException>(() => c.OldCombat);
        Assert.Throws<InvalidOperationException>(() => c.Enemy(new CreatureRef(1)));
        Assert.Throws<InvalidOperationException>(() => c.Card(new CardHandle(1)));
        Assert.Throws<InvalidOperationException>(() => c.Relic(new ListenerRef((int)ListenerKind.Relic, 1, 0)));
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
