// The shared setup of the monster handler tests (docs/rebuild/MIGRATION_F08.md section 5, "How to write a handler", step 6):
// an old CombatState with the monsters named by their game class, a Kernel bound with the F08+F10 switch and the model's
// own Register, and each handler that Register adds wrapped to log its call before it runs. The kill goes through the
// real command path (CreatureCmd.Damage, then the kill), so the log says which pass called the handler and in what order;
// a test whose model registers no handler for the pair finds nothing logged.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;

namespace Sts2Emulator.Tests.Core2.Models.Monsters;

internal static class MonsterHarness
{
    internal static readonly Switch[] On = [Switch.F08, Switch.F10];

    internal static OldModelIds Ids => ListenerOrderTests.Ids;

    internal static int Id(string monster) => Ids.Resolve(ListenerKind.Monster, monster);

    internal static EnemyState Enemy(string monster, int hp = 30) => new() { DefId = Id(monster), Hp = hp, MaxHp = hp };

    internal static CombatState Combat(params EnemyState[] enemies) =>
        new() { PlayerHp = 50, PlayerMaxHp = 50, Enemies = [.. enemies] };

    /// <summary>The kernel over <paramref name="s" /> with <paramref name="models" /> registered as they register
    /// themselves, each handler logged into <paramref name="log" /> as "Hook self&lt;-creature" (creature indices: the
    /// listener's own, and the creature the hook is about).</summary>
    internal static KernelState Bind(CombatState s, List<string> log, IEnumerable<Switch>? on,
        params Action<HandlerTable>[] models) =>
        new Kernel(new KernelOptions(on ?? On), () => Ids, models.Select(m => Spied(m, log)).ToList()).Bind(s);

    internal static KernelState Bind(CombatState s, List<string> log, params Action<HandlerTable>[] models) =>
        Bind(s, log, null, models);

    /// <summary>A powered hit from the player for <paramref name="amount" /> (the kill follows inside the command).</summary>
    internal static void Hit(KernelState k, EnemyState target, decimal amount) =>
        CreatureCmd.Damage(k, [OldCreature.Of(target)], amount, ValueProps.PoweredAttack, OldCreature.Player, null);

    /// <summary>The (kind, game class, hook) pairs the model registers with every switch on.</summary>
    internal static IReadOnlyList<RegisteredHandler> Pairs(Action<HandlerTable> register)
    {
        var t = new HandlerTable(Ids, _ => true);
        register(t);
        return t.Registered();
    }

    private static Action<HandlerTable> Spied(Action<HandlerTable> register, List<string> log) => t =>
    {
        var probe = new HandlerTable(Ids, _ => true);
        register(probe);
        // Each handler once, under the name the old id resolves (an inheritor's name has no old id of its own).
        foreach (var h in probe.Registered())
        {
            if (TryId(h.Kind, h.GameId) is not { } id || probe.Find(h.Hook, h.Kind, id) is not { } fn)
                continue;
            t.Model(h.Kind, h.GameId).On(h.Hook, Wrap(fn, h.Hook, log));
        }
    };

    private static int? TryId(ListenerKind kind, string gameId)
    {
        try
        {
            return Ids.Resolve(kind, gameId);
        }
        catch (ArgumentException)
        {
            return null;
        }
    }

    private static Delegate Wrap(Delegate fn, HookId hook, List<string> log) => fn switch
    {
        HookEvent<AfterDeathArgs> f => (HookEvent<AfterDeathArgs>)((HookCtx c, ListenerRef self, in AfterDeathArgs a,
            ChoiceCtx? ctx) =>
        {
            log.Add($"{hook} {self.Instance}<-{a.Creature.Index}");
            return f(c, self, in a, ctx);
        }),
        HookEvent<DiedArgs> f => (HookEvent<DiedArgs>)((HookCtx c, ListenerRef self, in DiedArgs a, ChoiceCtx? ctx) =>
        {
            log.Add($"{hook} {self.Instance}");
            return f(c, self, in a, ctx);
        }),
        HookEvent<BeforeRemovedFromRoomArgs> f => (HookEvent<BeforeRemovedFromRoomArgs>)((HookCtx c, ListenerRef self,
            in BeforeRemovedFromRoomArgs a, ChoiceCtx? ctx) =>
        {
            log.Add($"{hook} {self.Instance}");
            return f(c, self, in a, ctx);
        }),
        HookEvent<BeforeDeathArgs> f => (HookEvent<BeforeDeathArgs>)((HookCtx c, ListenerRef self, in BeforeDeathArgs a,
            ChoiceCtx? ctx) =>
        {
            log.Add($"{hook} {self.Instance}<-{a.Creature.Index}");
            return f(c, self, in a, ctx);
        }),
        // The HP change's delta is logged too: the damage pass and the kill's own HP loss each raise one.
        HookEvent<AfterCurrentHpChangedArgs> f => (HookEvent<AfterCurrentHpChangedArgs>)((HookCtx c, ListenerRef self,
            in AfterCurrentHpChangedArgs a, ChoiceCtx? ctx) =>
        {
            log.Add($"{hook} {self.Instance}<-{a.Creature.Index} {a.Delta}");
            return f(c, self, in a, ctx);
        }),
        HookEvent<AfterDamageReceivedArgs> f => (HookEvent<AfterDamageReceivedArgs>)((HookCtx c, ListenerRef self,
            in AfterDamageReceivedArgs a, ChoiceCtx? ctx) =>
        {
            log.Add($"{hook} {self.Instance}<-{a.Target.Index}");
            return f(c, self, in a, ctx);
        }),
        HookEvent<AfterRemovedArgs> f => (HookEvent<AfterRemovedArgs>)((HookCtx c, ListenerRef self,
            in AfterRemovedArgs a, ChoiceCtx? ctx) =>
        {
            log.Add($"{hook} {self.Instance}");
            return f(c, self, in a, ctx);
        }),
        HookEvent<AfterBlockBrokenArgs> f => (HookEvent<AfterBlockBrokenArgs>)((HookCtx c, ListenerRef self,
            in AfterBlockBrokenArgs a, ChoiceCtx? ctx) =>
        {
            log.Add($"{hook} {self.Instance}<-{a.Creature.Index}");
            return f(c, self, in a, ctx);
        }),
        HookFold<ModifyDamageArgs, decimal> f => f,
        _ => throw new NotSupportedException($"{hook}: the harness does not wrap {fn.GetType().Name} yet"),
    };
}
