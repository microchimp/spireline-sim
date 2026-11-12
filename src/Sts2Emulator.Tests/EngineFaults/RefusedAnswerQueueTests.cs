// Engine fault class 4 (docs/ENGINE_FAULTS.md): "PlayCard#N / UsePotion#N Executing at the front of queue 0 is in an invalid
// state" at ActionQueueSet.GetReadyAction. Under core2 an answer to a screen raised inside a card play or a potion resumes the
// paused action (a new id) and the executor starts it before the body replays the step. A refused replay
// (StoppedStepMismatch) put the old engine's combat back on its screen, but the kernel state it put back was the copy taken
// inside the body, with the action resumed and executing; the next step's executor met that front and threw, ending the
// worker process. The refusal now puts the kernel state back as it stood at the screen.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Tests.Solver;
using Xunit;

namespace Sts2Emulator.Tests.EngineFaults;

public class RefusedAnswerQueueTests
{
    static int Card(string name) => Cards.FindId(name)!.Value;

    static int PotionId(string name) => Potions.FindId(name)!.Value;

    /// <summary>
    /// Turn one under core2, stopped on a screen raised inside the action: Armaments' upgrade pick (a card play), or Gambler's
    /// Brew's discard pick (a potion).
    /// </summary>
    static (CombatState State, CountingRandom Rng) AtScreen(string source)
    {
        var (s, rng) = SolverPositions.Fight("core2", 46,
            [Card("Armaments"), Card("StrikeIronclad"), Card("StrikeIronclad"), Card("DefendIronclad"), Card("DefendIronclad")], [],
            source == "potion" ? [PotionId("GamblersBrew")] : []);
        foreach (var e in s.Enemies)
        {
            e.Hp = e.MaxHp = 200;
        }

        int action = source == "potion" ? s.Hand.Count + 1 : s.Hand.FindIndex(c => c.DefId == Card("Armaments"));
        KernelGate.Step(s.Kernel!.Shared, s, rng, action);
        Assert.NotNull(s.PendingSelection?.Replay);
        return (s, rng);
    }

    static string Queue(CombatState s) =>
        $"{string.Join(",", s.Kernel!.Queue!.QueueOf(0).Select(a => $"{a.Kind}#{a.Id} {a.State}"))} next {s.Kernel.Queue.NextId}";

    /// <summary>The stopped step's record with one answer too many: its replay ends with an answer left over, and is refused.</summary>
    static HookChoiceReplay LeftOver(HookChoiceReplay stopped) =>
        new()
        {
            Before = stopped.Before,
            RngBefore = stopped.RngBefore,
            Action = stopped.Action,
            TargetEnemyIndex = stopped.TargetEnemyIndex,
            Answers = [.. stopped.Answers, new StepAnswer(0, null)],
            Hidden = [.. stopped.Hidden, ScreenHiddenState.None],
            Taken = stopped.Taken,
            Screens = [.. stopped.Screens, stopped.Screens[^1]],
            Peeks = stopped.Peeks,
        };

    [Theory]
    [InlineData("card")]
    [InlineData("potion")]
    public void ARefusedAnswerLeavesTheActionGatheringAndTheNextAnswerResolvesIt(string source)
    {
        var (s, rng) = AtScreen(source);
        var kernel = s.Kernel!.Shared;
        var stopped = s.PendingSelection!.Replay!;
        string atScreen = Queue(s);
        Assert.Contains("GatheringPlayerChoice", atScreen);

        s.PendingSelection = s.PendingSelection.Clone(LeftOver(stopped));
        Assert.Throws<StoppedStepMismatch>(() => KernelGate.Step(kernel, s, rng, 0));

        // On its screen as it was, the queue included.
        Assert.Equal(atScreen, Queue(s));

        // The step goes on from there: the true record's answer resolves the screen (Gambler's Brew asks again until it is
        // declined) and the action finishes.
        s.PendingSelection = s.PendingSelection!.Clone(stopped);
        KernelGate.Step(kernel, s, rng, 0);
        while (s.PendingSelection is { Skippable: true } again)
        {
            KernelGate.Step(kernel, s, rng, again.Candidates.Count);
        }

        Assert.Null(s.PendingSelection?.Replay);
        Assert.Empty(s.Kernel!.Queue!.QueueOf(0));
    }
}
