// The kernel's runtime over the old state (docs/rebuild/TIER1_KERNEL.md sections 1 and 4): the handler table, the
// switches and a dispatch through Hooks.g.cs. The handlers here are test-only, registered in the test; no model is
// migrated by them.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class KernelTests
{
    private static readonly OldModelIds Ids = ListenerOrderTests.Ids;
    private static readonly JsonObject IdMap = LoadIdMap();
    private static readonly Switch[] Every = Enum.GetValues<Switch>().Where(s => s != Switch.None).ToArray();

    private static JsonObject LoadIdMap()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (!File.Exists(Path.Combine(dir!.FullName, "data", "id_map.json")))
            dir = dir.Parent;
        return JsonNode.Parse(File.ReadAllText(Path.Combine(dir.FullName, "data", "id_map.json")))!.AsObject();
    }

    private static string Name(string section, int id) =>
        IdMap[section]!.AsObject().First(kv => (int)kv.Value! == id).Key;

    private sealed record TestResume : HookResume;

    // One group per family, the turn-start flow kept whole: lets a test switch one family on and leave another off.
    private static readonly Switch[][] FineGroups =
        [.. Every.Where(s => s is not (Switch.F02 or Switch.F03 or Switch.F13)).Select(s => new[] { s }),
            [Switch.F02, Switch.F03, Switch.F13]];

    private static Kernel K(IEnumerable<Switch> on, Action<HandlerTable> models, IReadOnlyList<Switch[]>? groups = null) =>
        new(new KernelOptions(on, groups), () => Ids, [models]);

    private static CombatState Combat()
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = [new EnemyState { DefId = 16, Hp = 40, MaxHp = 40 }];
        s.Relics = [new RelicInstance(3), new RelicInstance(1), new RelicInstance(2)];
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 1));
        return s;
    }

    [Fact]
    public void SwitchGroupsKeepTheTurnStartFlowWholeAndEveryFamilyInOneGroup()
    {
        Assert.Single(SwitchGroups.Groups, g => g.Contains(Switch.F02) && g.Contains(Switch.F03) && g.Contains(Switch.F13));
        foreach (var f in Every)
            Assert.Single(SwitchGroups.Groups, g => g.Contains(f));
        Assert.Equal(Switch.F06, SwitchGroups.FamilyOf(HookId.AfterCardPlayed));
        Assert.Equal(Switch.F06, SwitchGroups.FamilyOf(HookId.Dispatch_AfterCardPlayed));
        Assert.Equal(Switch.F03, SwitchGroups.FamilyOf(HookId.ModifyHandDraw));
    }

    [Fact]
    public void AGroupSwitchesWhole()
    {
        var shared = SwitchGroups.Groups.First(g => g.Length > 1);
        var ex = Assert.Throws<ArgumentException>(() => new KernelOptions([shared[0]]));
        Assert.Contains("switch together", ex.Message);
        Assert.Equal(shared.Length, new KernelOptions(shared).Switches.Count);
        Assert.Throws<ArgumentException>(() => KernelOptions.FromFamilies(["F99"]));
    }

    [Fact]
    public void HandlersAreKeyedByGameModelWithAPerModelMask()
    {
        HookEvent<AfterCardPlayedArgs> ev = (ListenerRef _, in AfterCardPlayedArgs _, ChoiceCtx? _) => HookStep.Done;
        string akabeko = Name("relics", 1);
        var k = K(Every, t => t.Model(ListenerKind.Relic, akabeko).On(HookId.AfterCardPlayed, ev)
            .On(HookId.AfterCardPlayedLate, ev));
        var t = k.Handlers!;
        Assert.Equal(2, t.Count);
        Assert.Equal(new[] { HookId.AfterCardPlayed, HookId.AfterCardPlayedLate }, t.Mask(ListenerKind.Relic, akabeko).Hooks);
        Assert.Equal(t.Mask(ListenerKind.Relic, akabeko).Hooks, t.Mask(ListenerKind.Relic, 1).Hooks);
        Assert.True(t.Mask(ListenerKind.Relic, 2).IsEmpty);
        Assert.Same(ev, t.Find(HookId.AfterCardPlayed, ListenerKind.Relic, 1));
        Assert.Null(t.Find(HookId.AfterCardPlayed, ListenerKind.Card, 1));
        Assert.True(t.AnyOf(HookId.AfterCardPlayed, ListenerKind.Relic));
        Assert.False(t.AnyOf(HookId.AfterCardPlayed, ListenerKind.Power));
        Assert.True(k.HasAny(HookId.AfterCardPlayed));
        Assert.False(k.HasAny(HookId.BeforeCombatStart));
        Assert.Equal(akabeko, t.GameId(ListenerKind.Relic, 1));
        Assert.Equal(Ids.Power(BuffId.Strength).ModelId, t.Model(ListenerKind.Power, "StrengthPower").OldId);
        Assert.Equal((int)OrbType.Frost, t.Model(ListenerKind.Orb, "FrostOrb").OldId);
        // Broken cases: a second handler, an unknown model, a dispatcher id, a model kind the old state cannot name.
        Assert.Throws<ArgumentException>(() => t.Model(ListenerKind.Relic, akabeko).On(HookId.AfterCardPlayed, ev));
        Assert.Throws<ArgumentException>(() => t.Model(ListenerKind.Relic, "NoSuchRelic"));
        Assert.Throws<ArgumentException>(() => t.Model(ListenerKind.Relic, akabeko).On(HookId.Dispatch_AfterCardPlayed, ev));
        Assert.Throws<ArgumentException>(() => t.Model(ListenerKind.Modifier, "Anything"));
    }

    [Fact]
    public void WithNoSwitchOnTheKernelBuildsNothingAndDispatchIsTheFastPath()
    {
        bool ran = false;
        var k = K([], t => ran = true);
        Assert.False(k.AnyOn);
        Assert.Null(k.Handlers);
        Assert.Null(k.World);
        Assert.False(ran);
        var s = Combat();
        using (Kernel.Enter(k))
        {
            Assert.Null(Kernel.If(Switch.F06, s));
            using (Kernel.Marks.Setup())
            using (Kernel.Marks.Flying(new CardInstance(1, false)))
                Assert.Null(k.World);
        }

        Assert.Equal(HookStatus.Done, Hooks.AfterCardPlayed(k, new AfterCardPlayedArgs(new ChoiceCtx(), new CardPlayRef(1))));
        Assert.Empty(k.HookActions);
        Assert.Null(Kernel.Current);
    }

    [Fact] // game: CombatState.IterateHookListeners through Hook.AfterCardPlayed (passed context, pushed per listener)
    public void ATestHandlerIsDispatchedInGameOrderWithItsArguments()
    {
        var seen = new List<(ListenerRef Who, int Play, bool Pushed)>();
        HookEvent<AfterCardPlayedArgs> ev = (ListenerRef self, in AfterCardPlayedArgs a, ChoiceCtx? ctx) =>
        {
            seen.Add((self, a.CardPlay.Id, ctx is not null && ctx.Models.Count == 1 && ctx.Models[0] == self));
            return HookStep.Done;
        };
        var k = K(Every, t =>
        {
            foreach (int r in new[] { 1, 2, 3 })
                t.Model(ListenerKind.Relic, Name("relics", r)).On(HookId.AfterCardPlayed, ev);
            t.Model(ListenerKind.Power, "StrengthPower").On(HookId.AfterCardPlayed, ev);
            t.Model(ListenerKind.Monster, Name("enemies", 16)).On(HookId.AfterCardPlayed, ev);
        });
        var s = Combat();
        var ctx = new ChoiceCtx();
        using (Kernel.Enter(k))
        {
            var kk = Kernel.If(Switch.F06, s)!;
            Assert.Same(s, kk.World!.Combat);
            Assert.Equal(HookStatus.Done, Hooks.AfterCardPlayed(kk, new AfterCardPlayedArgs(ctx, new CardPlayRef(7))));
        }

        int str = Ids.Power(BuffId.Strength).ModelId;
        Assert.Equal(new ListenerRef[]
        {
            new((int)ListenerKind.Power, str, OldState.PowerInstance(0, 0)),
            new((int)ListenerKind.Relic, 3, 0), new((int)ListenerKind.Relic, 1, 1), new((int)ListenerKind.Relic, 2, 2),
            new((int)ListenerKind.Monster, 16, 1),
        }, seen.Select(x => x.Who));
        Assert.All(seen, x => Assert.Equal(7, x.Play));
        Assert.All(seen, x => Assert.True(x.Pushed));
        Assert.Empty(ctx.Models);
    }

    [Fact] // game: Hook.ModifyHandDraw (fold-set over two passes, changed by int-ne)
    public void AFoldRunsThroughTheRuntimeAndReportsWhoChangedIt()
    {
        HookFold<ModifyHandDrawArgs, decimal> plusOne = (ListenerRef _, in ModifyHandDrawArgs _, decimal v) => v + 1;
        HookFold<ModifyHandDrawArgs, decimal> same = (ListenerRef _, in ModifyHandDrawArgs _, decimal v) => v + 0.5m;
        var k = K(Every, t =>
        {
            t.Model(ListenerKind.Relic, Name("relics", 2)).On(HookId.ModifyHandDraw, plusOne);
            t.Model(ListenerKind.Relic, Name("relics", 3)).On(HookId.ModifyHandDraw, same);
            t.Model(ListenerKind.Relic, Name("relics", 1)).On(HookId.ModifyHandDrawLate, plusOne);
        });
        k.At(Combat());
        decimal v = Hooks.ModifyHandDraw(k, new ModifyHandDrawArgs(new PlayerRef(0), 5), out var changed);
        Assert.Equal(7.5m, v);
        Assert.Equal(new ListenerRef[] { new((int)ListenerKind.Relic, 2, 2), new((int)ListenerKind.Relic, 1, 1) }, changed);
    }

    [Fact]
    public void AHandlerOfASwitchedOffGroupNeverRuns()
    {
        int draws = 0, plays = 0;
        HookFold<ModifyHandDrawArgs, decimal> draw = (ListenerRef _, in ModifyHandDrawArgs _, decimal v) =>
        {
            draws++;
            return v + 1;
        };
        HookEvent<AfterCardPlayedArgs> play = (ListenerRef _, in AfterCardPlayedArgs _, ChoiceCtx? _) =>
        {
            plays++;
            return HookStep.Done;
        };
        var k = K([Switch.F06], t => t.Model(ListenerKind.Relic, Name("relics", 1))
            .On(HookId.ModifyHandDraw, draw).On(HookId.AfterCardPlayed, play), FineGroups);
        k.At(Combat());
        Assert.True(k.On(Switch.F06));
        Assert.False(k.On(Switch.F03));
        Assert.Equal(1, k.Handlers!.Count);
        Assert.False(k.HasAny(HookId.ModifyHandDraw));
        Assert.Equal(5m, Hooks.ModifyHandDraw(k, new ModifyHandDrawArgs(new PlayerRef(0), 5), out var changed));
        Assert.Empty(changed);
        Hooks.AfterCardPlayed(k, new AfterCardPlayedArgs(new ChoiceCtx(), new CardPlayRef(1)));
        Assert.Equal((0, 1), (draws, plays));
        using (Kernel.Enter(k))
        {
            Assert.Null(Kernel.If(Switch.F03, k.World!.Combat!));
            Assert.NotNull(Kernel.If(Switch.F06, k.World!.Combat!));
        }
    }

    [Fact] // game: HookPlayerChoiceContext (per-listener: a hook action, the dispatch moves on) and a passed context (stops)
    public void PausesAreRecordedAsHookActionsOrStops()
    {
        var order = new List<int>();
        HookEvent<AfterTurnEndArgs> end = (ListenerRef self, in AfterTurnEndArgs _, ChoiceCtx? ctx) =>
        {
            order.Add(self.ModelId);
            Assert.Equal(HookId.Dispatch_AfterTurnEnd, ctx!.Dispatcher);
            return self.ModelId == 3 ? HookStep.Paused("choose", new TestResume()) : HookStep.Done;
        };
        HookEvent<AfterCardPlayedArgs> play = (ListenerRef self, in AfterCardPlayedArgs _, ChoiceCtx? _) =>
        {
            order.Add(-self.ModelId);
            return self.ModelId == 1 ? HookStep.Paused("choose", new TestResume()) : HookStep.Done;
        };
        var k = K(Every, t =>
        {
            foreach (int r in new[] { 1, 2, 3 })
                t.Model(ListenerKind.Relic, Name("relics", r)).On(HookId.AfterSideTurnEnd, end).On(HookId.AfterCardPlayed, play);
        });
        k.At(Combat());
        Hooks.AfterTurnEndPasses(k, new AfterTurnEndArgs(Side.Player, []));
        Assert.Equal(new[] { 3, 1, 2 }, order);
        var act = Assert.Single(k.HookActions);
        Assert.Equal((1L, 0, new ListenerRef((int)ListenerKind.Relic, 3, 0)), (act.Id, act.Site.Index, act.Site.Who));
        Assert.Equal(new[] { new WaitPoint(HookId.Dispatch_AfterTurnEnd, 0, 1), new WaitPoint(HookId.Dispatch_AfterTurnEnd, 0, 1) },
            k.Waits);

        order.Clear();
        var ctx = new ChoiceCtx();
        var a = new AfterCardPlayedArgs(ctx, new CardPlayRef(1));
        var st = Hooks.AfterCardPlayed(k, a);
        Assert.True(st.IsPaused);
        Assert.Equal(new[] { -3, -1 }, order);
        Assert.Equal(st.Site, Assert.Single(k.Stops));
        Assert.Equal(1, st.Site!.Value.Index);
        Assert.Single(ctx.Models);
        Assert.Equal(HookStatus.Done, Hooks.AfterCardPlayedResume(k, a, st));
        Assert.Equal(new[] { -3, -1, -2 }, order);
        Assert.Empty(ctx.Models);
    }

    [Fact] // TIER1_KERNEL.md section 3: the marks are scoped setters the old engine's sites open
    public void MarksAreScopedSettersOnTheCurrentKernel()
    {
        var k = K(Every, _ => { });
        var s = Combat();
        var card = new CardInstance(1, false);
        using (Kernel.Enter(k))
        {
            k.At(s);
            using (Kernel.Marks.Setup())
            {
                Assert.True(k.World!.InSetup);
                using (Kernel.Marks.Flying(card))
                    Assert.Equal(card, k.World.InFlight);
                Assert.Null(k.World.InFlight);
            }

            Assert.False(k.World.InSetup);
            using (Kernel.Marks.DeactivatePlayerHooks())
                Assert.False(OldState.PlayerActive(k.World));
            Assert.True(OldState.PlayerActive(k.World));
        }

        Assert.Null(Kernel.Current);
    }
}
