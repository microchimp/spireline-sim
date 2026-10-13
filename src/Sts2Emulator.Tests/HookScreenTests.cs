using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Interop;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// A screen a HOOK raises in the middle of a step (HookChoiceReplay): the step stops on it, and
// the answer runs the step again from where it started, with the answer taken where the game
// awaits it. Tyranny's exhaust is the probe here: `TyrannyPower.AfterPlayerTurnStart` asks after
// the enemy turn and the hand draw, and the turn start goes on after the pick.
public class HookScreenReplayTests
{
    private static Fight TyrannyFight() =>
        Fight
            .Hand(Card(IC.StrikeIronclad))
            .Energy(3)
            .Draw([.. Pile(IC.Bash, IC.DefendIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Anger, IC.Inflame, IC.TwinStrike)])
            .Enemy(hp: 500, intent: new Intent(IntentType.Attack, 7))
            .PlayerBuff(BuffId.Tyranny, 1);

    [Fact]
    public void TheStepStopsOnTheScreenAndTheAnswerFinishesIt()
    {
        var fight = TyrannyFight();
        int hp = fight.State.PlayerHp;

        fight.EndTurn();

        Assert.Equal(CardSelectionKind.ExhaustFromHandRepeated, fight.Pending!.Kind);
        Assert.NotNull(fight.State.HookChoice);
        Assert.Equal(hp - 7, fight.State.PlayerHp);
        var picked = fight.State.Hand[2];

        fight.Choose(2);

        Assert.Null(fight.Pending);
        Assert.Null(fight.State.HookChoice);
        Assert.Equal([picked], fight.State.ExhaustPile);
        // The answer replayed the end of the turn from its start: the enemy hit once, not twice.
        Assert.Equal(hp - 7, fight.State.PlayerHp);
    }

    [Fact]
    public void ACopyOfAStoppedStepIsAnsweredTheSameWay()
    {
        var fight = TyrannyFight();
        var rng = new CountingRandom(7);
        CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
        var copy = fight.State.Clone();
        var copyRng = rng.Clone();

        CombatEngine.Step(fight.State, 3, rng);
        CombatEngine.Step(copy, 3, copyRng);

        Assert.Equal(CombatSnapshot.Write(fight.State), CombatSnapshot.Write(copy));
        Assert.Equal(rng.CallCount, copyRng.CallCount);
    }

    [Fact]
    public void AnAnswerOutsideTheScreenIsRefusedAndChangesNothing()
    {
        var fight = TyrannyFight();
        fight.EndTurn();
        var before = CombatSnapshot.Write(fight.State);

        Assert.Equal(StepResult.Invalid, fight.Choose(fight.Pending!.Candidates.Count));

        Assert.Equal(before, CombatSnapshot.Write(fight.State));
        Assert.NotNull(fight.State.HookChoice);
    }

    [Fact]
    public void AStepOnStreamsThatCannotBePutBackRefusesToStopHalfway()
    {
        var fight = TyrannyFight();

        Assert.Throws<InvalidOperationException>(
            () => CombatEngine.Step(fight.State, fight.State.Hand.Count, new Random(0))
        );
    }
}

// `EntropyPower.AfterPlayerTurnStart`: `CardSelectCmd.FromHand(prefs(prompt, Amount))`, then
// `CardCmd.TransformToRandom` on Rng.CombatCardSelection. After the hand draw, over the new hand,
// and the turn start goes on after the pick.
public class EntropyScreenTests
{
    private static Fight WithEntropy(int amount = 1) =>
        Fight
            .Hand()
            .Energy(3)
            .Draw([.. Pile(IC.Bash, IC.DefendIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Anger, IC.Inflame)])
            .Enemy(hp: 500)
            .PlayerBuff(BuffId.EntropyPower, amount);

    [Fact]
    public void TheScreenComesAfterTheDrawOverTheNewHand()
    {
        var fight = WithEntropy();

        fight.EndTurn();

        Assert.Equal(CardSelectionKind.TransformInHand, fight.Pending!.Kind);
        Assert.Equal(CL.Entropy, fight.Pending.SourceCardDefId);
        Assert.Equal(5, fight.Pending.Candidates.Count);
        Assert.Equal(5, fight.State.Hand.Count);
    }

    [Fact]
    public void TheChosenCardIsTransformedInPlaceAndCountsAsGenerated()
    {
        var fight = WithEntropy();
        fight.EndTurn();
        var before = fight.State.Hand.ToList();
        int generated = fight.State.CardsGeneratedThisCombat;

        fight.Choose(2);

        var after = fight.State.Hand;
        Assert.Equal(5, after.Count);
        Assert.NotEqual(before[2].DefId, after[2].DefId);
        Assert.False(after[2].Upgraded);
        Assert.Contains(after[2].DefId, Core.Run.RunRewardGenerator.TransformOptionsFor(before[2].DefId));
        Assert.Equal(before.Where((_, i) => i != 2), after.Where((_, i) => i != 2));
        // `CardCmd.Transform` records the replacement as generated, which Supermassive counts.
        Assert.Equal(generated + 1, fight.State.CardsGeneratedThisCombat);
    }

    [Fact]
    public void TwoStacksAskTwiceAndNeverForTheSameCard()
    {
        var fight = WithEntropy(amount: 2);
        fight.EndTurn();
        fight.Choose(1);

        Assert.Equal(CardSelectionKind.TransformInHand, fight.Pending!.Kind);
        Assert.Equal([0, 2, 3, 4], fight.Pending.Candidates);
        fight.Choose(0);

        Assert.Null(fight.Pending);
    }

    [Fact]
    public void TyrannyStillAsksAfterIt()
    {
        var fight = WithEntropy().PlayerBuff(BuffId.Tyranny, 1);

        fight.EndTurn();
        Assert.Equal(CardSelectionKind.TransformInHand, fight.Pending!.Kind);
        fight.Choose(0);

        Assert.Equal(CardSelectionKind.ExhaustFromHandRepeated, fight.Pending!.Kind);
        fight.Choose(0);

        Assert.Null(fight.Pending);
        Assert.Single(fight.State.ExhaustPile);
    }
}
