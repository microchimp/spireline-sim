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
        Assert.NotNull(fight.Pending.Replay);
        Assert.Equal(hp - 7, fight.State.PlayerHp);
        var picked = fight.State.Hand[2];

        fight.Choose(2);

        Assert.Null(fight.Pending);
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
        Assert.NotNull(fight.Pending!.Replay);
    }

    [Fact]
    public void AStepOnStreamsThatCannotBePutBackLeavesTheScreenStanding()
    {
        var fight = TyrannyFight();

        CombatEngine.Step(fight.State, fight.State.Hand.Count, new Random(0));

        // Nothing to replay it from: the screen stands over the finished step, as screens used to.
        Assert.Equal(CardSelectionKind.ExhaustFromHandRepeated, fight.Pending!.Kind);
        Assert.Null(fight.Pending.Replay);
        CombatEngine.Step(fight.State, 0, new Random(0));
        Assert.Null(fight.Pending);
        Assert.Single(fight.State.ExhaustPile);
    }
}

// `StratagemPower.AfterShuffle` asks (`CardSelectCmd.FromCombatPile(Draw, prefs(prompt, Amount))`)
// from inside the shuffle, which runs from inside `CardPileCmd.Draw` when the pile is empty. The
// pick joins the hand and the draw goes on after it. The emulator took the top card of the new
// pile without asking.
public class StratagemShuffleScreenTests
{
    private static readonly int[] Six =
    [
        IC.Bash,
        IC.StrikeIronclad,
        IC.DefendIronclad,
        CL.Entropy,
        CL.Stratagem,
        IC.Inflame,
    ];

    private static Fight WithStratagem(params CardInstance[] hand) =>
        Fight.Hand(hand).Energy(9).Enemy(hp: 500).PlayerBuff(BuffId.StratagemPower, 1);

    private static List<int> Ids(List<CardInstance> pile) => [.. pile.Select(c => c.DefId)];

    [Fact]
    public void AShuffleInTheMiddleOfADrawAsksAndTheDrawGoesOnAfterThePick()
    {
        var fight = WithStratagem(Card(IC.BattleTrance))
            .Draw(Card(IC.Anger))
            .Discard([.. Pile(Six)]);

        fight.Play();

        // One card came off the old pile; the second needed the shuffle, which asked.
        Assert.Equal([IC.Anger], Ids(fight.State.Hand));
        Assert.Equal(CardSelectionKind.DrawPileToHand, fight.Pending!.Kind);
        Assert.Equal(CL.Stratagem, fight.Pending.SourceCardDefId);
        Assert.Equal(6, fight.Pending.Candidates.Count);
        var pile = Ids(fight.State.DrawPile);

        fight.Choose(3);

        // Anger, the pick, then the two cards still to draw -- off the pile WITHOUT the pick.
        // Battle Trance's gag came after its draw, so it stopped none of them.
        var rest = pile.Where((_, i) => i != 3).ToList();
        Assert.Equal([IC.Anger, pile[3], rest[0], rest[1]], Ids(fight.State.Hand));
        Assert.Null(fight.Pending);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.NoDraw));
    }

    [Fact]
    public void PillageKeepsDrawingAfterThePick()
    {
        // Pillage draws until it draws a non-Attack. The old pile is one Strike; the shuffle
        // asks, and the draw goes on into the new pile: Anger, Anger, then the Defend.
        var fight = WithStratagem(Card(IC.Pillage))
            .Draw(Card(IC.StrikeIronclad))
            .Discard([.. Pile(IC.Anger, IC.Anger, IC.DefendIronclad, IC.Bash)]);

        fight.Play();
        var pile = Ids(fight.State.DrawPile);
        int bash = pile.IndexOf(IC.Bash);
        fight.Choose(bash);

        var rest = pile.Where((_, i) => i != bash).ToList();
        int stop = rest.IndexOf(IC.DefendIronclad);
        Assert.Equal(
            [IC.StrikeIronclad, IC.Bash, .. rest.Take(stop + 1)],
            Ids(fight.State.Hand)
        );
        Assert.Null(fight.Pending);
    }

    [Fact]
    public void EscapePlanJudgesTheCardItDrewAfterThePick()
    {
        // The draw that shuffled returns the card it drew, not the pick: a Defend, a Skill.
        var fight = WithStratagem(Card(SI.EscapePlan))
            .Draw()
            .Discard([.. Pile(IC.DefendIronclad, IC.DefendIronclad, IC.Bash)]);

        fight.Play();
        int bash = Ids(fight.State.DrawPile).IndexOf(IC.Bash);
        fight.Choose(bash);

        Assert.Equal([IC.Bash, IC.DefendIronclad], Ids(fight.State.Hand));
        Assert.Equal(3, fight.State.PlayerBlock);
    }

    [Fact]
    public void ThinkingAheadsOwnScreenFollowsThePick()
    {
        var fight = WithStratagem(Card(CL.ThinkingAhead))
            .Draw(Card(IC.Anger))
            .Discard([.. Pile(Six)]);

        fight.Play();
        Assert.Equal(CL.Stratagem, fight.Pending!.SourceCardDefId);
        fight.Choose(0);

        // Anger, the pick, one more card: then Thinking Ahead puts one back.
        Assert.Equal(3, fight.State.Hand.Count);
        Assert.Equal(CardSelectionKind.HandToDrawPileTop, fight.Pending!.Kind);
        var back = fight.State.Hand[0];
        fight.Choose(0);

        Assert.Null(fight.Pending);
        Assert.Equal(2, fight.State.Hand.Count);
        Assert.Equal(back, fight.State.DrawPile[0]);
    }

    [Fact]
    public void CascadeKeepsTakingAfterThePick()
    {
        // `CardPileCmd.AutoPlayFromDrawPile(2, Top)` takes both cards first, running
        // `ShuffleIfNecessary` before each take: the second take shuffles, Stratagem asks, and
        // the take goes on off the new pile.
        const int cascade = 546;
        var fight = WithStratagem(Card(cascade))
            .Energy(2)
            .Draw(Card(IC.DefendIronclad))
            .Discard([.. Pile(IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.Bash)]);

        fight.Play();
        Assert.Equal(CL.Stratagem, fight.Pending!.SourceCardDefId);
        int bash = Ids(fight.State.DrawPile).IndexOf(IC.Bash);
        fight.Choose(bash);

        // Both Defends were played: 2 x 5 block, and the pick is in hand.
        Assert.Equal([IC.Bash], Ids(fight.State.Hand));
        Assert.Equal(10, fight.State.PlayerBlock);
        Assert.Equal(2, fight.State.DrawPile.Count);
    }

    [Fact]
    public void ThePileIsTakenWholeWithoutAskingWhenItIsNoBiggerThanTheCount()
    {
        var fight = WithStratagem(Card(IC.PommelStrike)).Draw().Discard(Card(IC.Bash));

        fight.Play();

        // `FromCombatPile` hands the pile back when it is no bigger than the minimum; the draw
        // then finds nothing left to take.
        Assert.Null(fight.Pending);
        Assert.Equal([IC.Bash], Ids(fight.State.Hand));
        Assert.Empty(fight.State.DrawPile);
    }

    [Fact]
    public void TwoEmptyPilesAreNotAShuffle()
    {
        var fight = Fight.Hand(Card(IC.PommelStrike)).Energy(3).Enemy(hp: 500).Draw().Discard();
        fight.State.Relics.Add(new RelicInstance(RelicEffects.TheAbacus));

        fight.Play();

        // `ShuffleIfNecessary` shuffles only when the discard pile has cards, so the Abacus
        // (and every other `AfterShuffle`) does not fire. Pommel Strike itself is not
        // discarded until after its draw.
        Assert.Equal(0, fight.State.PlayerBlock);
    }

    [Fact]
    public void TheAbacusPaysAfterThePickAndForegoneConclusionAsksAfterBoth()
    {
        // `ForegoneConclusionPower.BeforeHandDraw` runs `ShuffleIfNecessary` first. Its shuffle
        // asks Stratagem's question; the relics' `AfterShuffle` follows the powers'; then
        // Foregone Conclusion asks its own, over the new pile, before the hand is drawn.
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .Energy(3)
            .Enemy(hp: 500)
            .Draw()
            .Discard([.. Pile(Six)])
            .PlayerBuff(BuffId.StratagemPower, 1)
            .PlayerBuff(BuffId.ForegoneConclusion, 1);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.TheAbacus));

        fight.EndTurn();
        Assert.Equal(CL.Stratagem, fight.Pending!.SourceCardDefId);
        Assert.Equal(8, fight.Pending.Candidates.Count);
        Assert.Equal(0, fight.State.PlayerBlock);
        var stratagemPick = fight.State.DrawPile[5];
        fight.Choose(5);

        Assert.Equal(204, fight.Pending!.SourceCardDefId); // Foregone Conclusion
        Assert.Equal(6, fight.State.PlayerBlock);
        var foregonePick = fight.State.DrawPile[0];
        fight.Choose(0);

        Assert.Null(fight.Pending);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.ForegoneConclusion));
        Assert.Equal(stratagemPick, fight.State.Hand[0]);
        Assert.Equal(foregonePick, fight.State.Hand[1]);
        // Then the hand draw: five more, as many as the six left in the pile allow.
        Assert.Equal(7, fight.State.Hand.Count);
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

// `CardPileCmd.Draw` and `Shuffle` return at once when the combat `IsOverOrEnding`, and
// `CardSelectCmd.FromHand`/`FromCombatPile` return nothing without asking: a fight won by the card
// that would draw, shuffle or ask ends there. The emulator drew, shuffled and asked anyway, so
// a lethal Pommel Strike under Stratagem left a won fight open on a screen.
public class NoScreenOnceTheFightIsWonTests
{
    [Fact]
    public void ALethalDrawOverAnEmptyPileEndsTheFightWithoutAsking()
    {
        var fight = Fight
            .Hand(Card(IC.PommelStrike))
            .Energy(3)
            .Draw()
            .Discard(Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash))
            .Enemy(hp: 5)
            .PlayerBuff(BuffId.StratagemPower, 1);

        var result = fight.Play();

        Assert.True(result.Terminal);
        Assert.True(result.PlayerWon);
        Assert.Null(fight.Pending);
        // No shuffle and no draw: the discard is where it was.
        Assert.Equal(4, fight.State.DiscardPile.Count + fight.State.ExhaustPile.Count);
        Assert.Empty(fight.State.DrawPile);
    }

    [Fact]
    public void ALethalDrawDrawsNothing()
    {
        var fight = Fight
            .Hand(Card(IC.PommelStrike))
            .Energy(3)
            .Draw(Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .Enemy(hp: 5);

        Assert.True(fight.Play().Terminal);

        Assert.Equal(2, fight.State.DrawPile.Count);
        Assert.Empty(fight.State.Hand);
    }
}
