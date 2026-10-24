// The test kit the handler batches share (docs/rebuild/CLOUD_BRIEFS.md section 3, "Tests"; the C1 pilot sets it, the
// later batches reuse it unchanged): build an old CombatState holding a power, a relic or a card, bind a Kernel with the
// F08+F10 test switch and the batch's registrations, read the handler table, and make the dispatchers' arguments. The
// worked example it follows is HandlerContextTests (MIGRATION_F08.md section 5, "How to write a handler"); the kernel is
// built as KillCommandTests builds it.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2.Models;

public static class ModelTestKit
{
    /// <summary>The model ids of data/id_map.json and tools/fidelity/power_aliases.json, as the engine loads them.</summary>
    public static readonly OldModelIds Ids = ListenerOrderTests.Ids;

    /// <summary>The handler batches' test switch: the damage and death families on, every other off.</summary>
    public static readonly Switch[] DamageOnly = [Switch.F08, Switch.F10];

    /// <summary>Every family but the damage and death ones: the table exists and the F08/F10 handlers are absent.</summary>
    public static readonly Switch[] AllButDamage =
        Enum.GetValues<Switch>().Where(s => s != Switch.None && !DamageOnly.Contains(s)).ToArray();

    /// <summary>The enemy the kit's combats hold (a slug: no model of its own takes part in a test).</summary>
    public const int Slug = 16;

    // ---------------------------------------------------------------------------------------- the combat

    /// <summary>A combat of the player (50/50) against one slug per HP given, no relic, nothing in the piles.</summary>
    public static CombatState Combat(params int[] enemyHp)
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = enemyHp.Select(hp => new EnemyState { DefId = Slug, Hp = hp, MaxHp = hp }).ToList();
        return s;
    }

    /// <summary>Osty joins the combat (the player's pet, <paramref name="hp" /> of <paramref name="hp" />): creature 1, so the
    /// enemies move up to 2 and on, as in the engine. Returns <paramref name="s" /> for chaining:
    /// <c>WithOsty(Combat(40))</c>. Game: the player's Osty is a pet of the player (<c>Creature.PetOwner</c>).</summary>
    public static CombatState WithOsty(CombatState s, int hp = 10)
    {
        s.OstyMaxHp = s.OstyHp = hp;
        return s;
    }

    /// <summary>The player (creature 0) holds <paramref name="relicId" /> (the old def id) at the end of the obtain order;
    /// returns its listener.</summary>
    public static ListenerRef Hold(CombatState s, int relicId, int counter = 0)
    {
        s.Relics.Add(new RelicInstance(relicId, counter));
        return new ListenerRef((int)ListenerKind.Relic, relicId, s.Relics.Count - 1);
    }

    /// <summary>A fresh card of def <paramref name="defId" /> added at the end of <paramref name="pile" /> (one of the
    /// combat's four lists); returns it, with its uid.</summary>
    public static CardInstance Add(CombatState s, List<CardInstance> pile, int defId, bool upgraded = false)
    {
        var card = s.Fresh(new CardInstance(defId, upgraded));
        pile.Add(card);
        return card;
    }

    // ---------------------------------------------------------------------------------------- the kernel

    /// <summary>A kernel state over <paramref name="s" /> with the test switch (or <paramref name="on" />) and these
    /// registrations, the way KillCommandTests builds one.</summary>
    public static KernelState Kernel(CombatState s, IEnumerable<Action<HandlerTable>> models, IEnumerable<Switch>? on = null) =>
        new Kernel(new KernelOptions(on ?? DamageOnly), () => Ids, models).Bind(s);

    /// <summary>As above, for one registration.</summary>
    public static KernelState Kernel(CombatState s, Action<HandlerTable> model, IEnumerable<Switch>? on = null) =>
        Kernel(s, [model], on);

    /// <summary><paramref name="models" /> without <paramref name="model" /> (a static Register method group compares
    /// equal to itself): the delete-mutant of one model.</summary>
    public static IReadOnlyList<Action<HandlerTable>> Without(IEnumerable<Action<HandlerTable>> models, Action<HandlerTable> model) =>
        models.Where(m => !m.Equals(model)).ToList();

    /// <summary>The handler table the kernel builds for these registrations under the test switch (or
    /// <paramref name="on" />): what EmuDump kernel-handlers reads.</summary>
    public static HandlerTable Table(IEnumerable<Action<HandlerTable>> models, IEnumerable<Switch>? on = null) =>
        new Kernel(new KernelOptions(on ?? AllButDamage.Concat(DamageOnly)), () => Ids, models).Handlers
        ?? throw new InvalidOperationException("no switch on: the kernel builds no table");

    /// <summary>The (kind, game model, hook) pairs a table holds, in its own order (kind, name, hook).</summary>
    public static IReadOnlyList<(ListenerKind Kind, string Model, HookId Hook)> Pairs(HandlerTable t) =>
        t.Registered().Select(r => (r.Kind, r.GameId, r.Hook)).ToList();

    /// <summary>Asserts that each of <paramref name="hooks" /> is registered for the model and that none is a stub.</summary>
    public static void AssertRegistered(HandlerTable t, ListenerKind kind, string model, params HookId[] hooks)
    {
        var mine = t.Registered().Where(r => r.Kind == kind && r.GameId == model).ToList();
        foreach (var hook in hooks)
        {
            var r = Assert.Single(mine, x => x.Hook == hook);
            Assert.False(r.Stub, $"{kind} {model} {hook} is a [StubHandler]");
        }
    }

    /// <summary>Asserts that no hook of <paramref name="hooks" /> is registered for the model.</summary>
    public static void AssertAbsent(HandlerTable t, ListenerKind kind, string model, params HookId[] hooks)
    {
        var mine = t.Registered().Where(r => r.Kind == kind && r.GameId == model).Select(r => r.Hook).ToList();
        foreach (var hook in hooks)
            Assert.DoesNotContain(hook, mine);
    }

    // ---------------------------------------------------------------------------------------- listeners and arguments

    /// <summary>A power listener: the model's game class name, the creature it sits on (the player 0, Osty 1 when
    /// present, then the enemies in roster order) and which of that creature's instances of the model.</summary>
    public static ListenerRef Power(string model, int creature, int instance = 0) =>
        new((int)ListenerKind.Power, Ids.Resolve(ListenerKind.Power, model), OldState.PowerInstance(creature, instance));

    /// <summary>A relic listener: the model's game class name and its position in the obtain order.</summary>
    public static ListenerRef Relic(string model, int index) =>
        new((int)ListenerKind.Relic, Ids.Resolve(ListenerKind.Relic, model), index);

    /// <summary>A card listener: the copy's def id and uid.</summary>
    public static ListenerRef Card(CardInstance card) => new((int)ListenerKind.Card, card.DefId, card.Uid);

    /// <summary>The creature reference a dispatcher carries for an old creature of the bound combat.</summary>
    public static CreatureRef Ref(KernelState k, OldCreature c) => OldCreatures.Ref(k.World!, c);

    /// <summary>The player's powered attack on <paramref name="target" /> for <paramref name="damage" />, as
    /// Hooks.ModifyDamage takes it (all passes, no preview).</summary>
    public static ModifyDamageArgs Attack(int target, decimal damage, ValueProp? props = null, int? dealer = 0,
        CardHandle? card = null) =>
        new(new CreatureRef(target), dealer is { } d ? new CreatureRef(d) : null, damage, props ?? ValueProps.PoweredAttack,
            card, ModifyDamageHookType.All, PreviewModes.None);

    /// <summary>An HP loss of <paramref name="amount" /> on <paramref name="target" /> from the player's powered attack, as
    /// Hooks.ModifyHpLost takes it, in the given phase (after Osty unless said).</summary>
    public static ModifyHpLostArgs HpLoss(int target, decimal amount, ValueProp? props = null, int? dealer = 0,
        CardHandle? card = null, HpLossHookPhase phase = HpLossHookPhase.AfterOsty) =>
        new(new CreatureRef(target), amount, props ?? ValueProps.PoweredAttack, dealer is { } d ? new CreatureRef(d) : null,
            card, phase);

    /// <summary>One CreatureCmd.Damage call: <paramref name="dealer" /> (the player unless said) hits
    /// <paramref name="target" /> for <paramref name="amount" /> with <paramref name="props" /> (a powered attack unless
    /// said) and the card source <paramref name="card" />.</summary>
    public static IReadOnlyList<DamageResult> Hit(KernelState k, OldCreature target, decimal amount, OldCreature? dealer = null,
        ValueProp? props = null, CardHandle? card = null) =>
        CreatureCmd.Damage(k, [target], amount, props ?? ValueProps.PoweredAttack, dealer ?? OldCreature.Player, card);

    /// <summary>The unpowered props (thorns, poison, a relic's damage): Move off, Unpowered on.</summary>
    public static readonly ValueProp Unpowered = new(ValueProps.Unpowered);

    // ---------------------------------------------------------------------------------------- holders

    /// <summary>The creature index of <paramref name="enemy" /> in the combat (the player 0, Osty 1 when present, then the
    /// enemies in roster order). It counts every earlier entry of s.Enemies, dead ones included, which TryAt skips, so it is the
    /// creature index only while no earlier enemy is dead; the enemy itself must be listed.</summary>
    public static int IndexOf(CombatState s, EnemyState enemy) =>
        1 + (s.OstyMaxHp > 0 ? 1 : 0) + s.Enemies.FindIndex(e => ReferenceEquals(e, enemy));

    /// <summary>A holder of a power: <paramref name="buff" /> at <paramref name="amount" /> goes on <paramref name="on" />
    /// (an enemy) or on the player (null, the default); returns the power's listener, the first instance of
    /// <paramref name="model" /> on that creature (a creature holds a power once). Enemy-owned and player-owned holders are
    /// the two calls.</summary>
    public static ListenerRef GivePower(CombatState s, string model, BuffId buff, int amount, EnemyState? on = null)
    {
        (on?.Buffs ?? s.PlayerBuffs).Add(new BuffState(buff, amount));
        return Power(model, on is null ? 0 : IndexOf(s, on));
    }

    /// <summary>The N-holder setup (docs/rebuild/CLOUD_BRIEFS.md section 3, rule 5): <paramref name="model" /> is held by the
    /// player and by every enemy of the combat, each at <paramref name="amount" />; returns the listeners, the player's first,
    /// then the enemies in roster order. A handler must act for its own holder only: a dispatch that names one creature must
    /// change that creature's listener and no other.</summary>
    public static IReadOnlyList<ListenerRef> Holders(CombatState s, string model, BuffId buff, int amount) =>
        [GivePower(s, model, buff, amount), .. s.Enemies.Select(e => GivePower(s, model, buff, amount, e))];

    // ---------------------------------------------------------------------------------------- death hooks

    /// <summary>The arguments of Hooks.AfterDeath for the creature at <paramref name="creature" />.</summary>
    public static AfterDeathArgs AfterDeathOf(int creature, bool removalPrevented = false, float animLength = 0f) =>
        new(new CreatureRef(creature), removalPrevented, animLength);

    /// <summary>The arguments of Hooks.BeforeDeath for the creature at <paramref name="creature" />.</summary>
    public static BeforeDeathArgs BeforeDeathOf(int creature) => new(new CreatureRef(creature));

    /// <summary>The arguments of Hooks.ShouldDie for the creature at <paramref name="creature" />.</summary>
    public static ShouldDieArgs ShouldDieOf(int creature) => new(new CreatureRef(creature));

    /// <summary>
    /// The state the game's death overrides see: <paramref name="enemy" /> is dead (0 HP) but not yet removed, so it is still
    /// listed in the combat (OldWorld.MarkDying, as CreatureCmd.Kill sets it). Dispose to restore the HP and clear the mark.
    /// A handler that reads the dying creature (its powers, its position) is tested inside this scope.
    /// </summary>
    public static IDisposable Dying(KernelState k, EnemyState enemy)
    {
        int hp = enemy.Hp;
        enemy.Hp = 0;
        var mark = k.World!.MarkDying(enemy);
        return new Scope(() =>
        {
            mark.Dispose();
            enemy.Hp = hp;
        });
    }

    private sealed class Scope(Action end) : IDisposable
    {
        private Action? _end = end;

        public void Dispose()
        {
            _end?.Invoke();
            _end = null;
        }
    }
}
