// The plain-data rule of a hook step (docs/rebuild/TIER1_KERNEL.md section 5, Core2/HookPlain.cs): a request and a resume
// record are shared by reference when a state holding them is copied (KernelState.CopyTo), so neither holds a live object.
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

public class HookPlainTests
{
    private sealed record Plain(int Amount, string Who, int[] Handles, ListenerRef Owner, RelicInstance Relic, List<string> Notes) : HookResume;

    private sealed record Nested(Plain Inner, object Open) : HookResume;

    private sealed record HoldsEnemy(EnemyState Target) : HookResume;

    private sealed record HoldsCombat(CombatState State) : HookResume;

    private sealed record HoldsCards(List<CardInstance> Cards) : HookResume;

    private sealed record HoldsCardArray(CardInstance[] Cards);

    private sealed record HoldsCtx(ChoiceCtx Ctx) : HookResume;

    private sealed record HoldsDelegate(Func<int> Later) : HookResume;

    private sealed class NotARecord
    {
        public int X;
    }

    private sealed record HoldsClass(NotARecord Thing) : HookResume;

    private static readonly ListenerRef Who = new((int)ListenerKind.Relic, 1, 1);

    private static Plain APlainRecord() => new(1, "x", [1, 2], Who, new RelicInstance(3), ["n"]);

    [Fact]
    public void PlainRecordsPass()
    {
        Assert.Null(HookPlain.Violation(null));
        Assert.Null(HookPlain.Violation(APlainRecord()));
        Assert.Null(HookPlain.Violation(new Nested(APlainRecord(), 5)));
        Assert.Null(HookPlain.Violation(new Nested(APlainRecord(), APlainRecord())));
    }

    [Fact]
    public void LiveTypesAreRejectedInResumeRecords()
    {
        // game: the context's task holds handles (a creature, a card) and re-reads them; a record that holds the live
        // object itself would act on its source's state from a copy.
        Assert.Contains("EnemyState", HookPlain.Violation(new HoldsEnemy(new EnemyState()))!);
        Assert.Contains("CombatState", HookPlain.Violation(new HoldsCombat(new CombatState()))!);
        Assert.Contains("CardInstance", HookPlain.Violation(new HoldsCards([]))!);
        Assert.Contains("ChoiceCtx", HookPlain.Violation(new HoldsCtx(new ChoiceCtx()))!);
        Assert.NotNull(HookPlain.Violation(new HoldsDelegate(() => 1)));
        Assert.Contains("not a record", HookPlain.Violation(new HoldsClass(new NotARecord()))!);
    }

    [Fact]
    public void LiveTypesAreRejectedInRequestsAndInOpenMembers()
    {
        Assert.Contains("CardInstance", HookPlain.Violation(new HoldsCardArray([]))!);
        Assert.NotNull(HookPlain.Violation(new CombatState()));
        Assert.NotNull(HookPlain.Violation(new List<EnemyState>()));
        Assert.NotNull(HookPlain.Violation(new Nested(APlainRecord(), new EnemyState())));
        Assert.NotNull(HookPlain.Violation(new Nested(APlainRecord(), new List<object> { new CombatState() })));
    }

    [Fact]
    public void ADebugBuildRefusesToPauseWithALiveRequestOrResumeRecord()
    {
        // Every pause the other tests make (their Screen, Log and PickResume records) passes through the same check.
#if DEBUG
        Assert.Throws<InvalidOperationException>(() => HookStep.Paused(new HoldsCardArray([]), APlainRecord()));
        Assert.Throws<InvalidOperationException>(() => HookStep.Paused("request", new HoldsEnemy(new EnemyState())));
#endif
        Assert.True(HookStep.Paused("request", APlainRecord()).IsPaused);
    }
}
