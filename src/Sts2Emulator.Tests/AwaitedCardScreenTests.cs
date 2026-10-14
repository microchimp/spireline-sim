using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// A card's own screen is awaited inside its `OnPlay` (CombatEngine.AwaitSelection), like a hook's:
/// the step stops on it, and the answer replays the step with the pick taken where the card asks.
/// The emulator used to end the play first -- every after-play hook, the card's placement -- and
/// leave the screen standing; a second play of the same card overwrote the first screen; and a
/// card auto-played by Havoc or Mayhem took a fixed pick for the player.
/// </summary>
public class AwaitedCardScreenTests
{
    [Fact]
    public void ABurstSurvivorAsksForEachPlaysDiscard()
    {
        // `BurstPower.ModifyCardPlayCount`: Survivor resolves twice, and each resolution awaits
        // its own `FromHandForDiscard`.
        var fight = Fight
            .Hand(Card(SI.Survivor), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash))
            .Energy(3)
            .Enemy(hp: 40)
            .PlayerBuff(BuffId.Burst, 1);

        fight.Play(0);
        Assert.Equal(CardSelectionKind.DiscardFromHandRepeated, fight.Pending!.Kind);
        Assert.Equal(3, fight.Pending.Candidates.Count);
        fight.Choose(0);
        Assert.Equal(CardSelectionKind.DiscardFromHandRepeated, fight.Pending!.Kind);
        Assert.Equal(2, fight.Pending.Candidates.Count);
        fight.Choose(0);

        Assert.Null(fight.Pending);
        Assert.Equal([IC.Bash], Fight.Ids(fight.State.Hand));
        Assert.Equal(2 * 8, fight.State.PlayerBlock);
    }

    [Fact]
    public void AStoppedPlayHasNotFinished()
    {
        // Stopped on Headbutt's screen, the play is in the middle of its OnPlay: the damage is
        // dealt, the card is in no pile yet (the game's play pile), and nothing after it has run.
        var fight = Fight
            .Hand(Card(IC.Headbutt))
            .Energy(3)
            .Discard(Card(IC.StrikeIronclad), Card(IC.Bash))
            .Enemy(hp: 40);

        fight.Play(0);

        Assert.Equal(CardSelectionKind.DiscardToDrawPileTop, fight.Pending!.Kind);
        Assert.Equal(40 - 9, fight.Enemy0.Hp);
        Assert.DoesNotContain(fight.State.DiscardPile, c => c.DefId == IC.Headbutt);
        Assert.Equal(0, fight.State.CardPlaysThisTurn);

        fight.Choose(1);

        Assert.Equal(IC.Bash, fight.State.DrawPile[0].DefId);
        Assert.Contains(fight.State.DiscardPile, c => c.DefId == IC.Headbutt);
        Assert.Equal(1, fight.State.CardPlaysThisTurn);
    }

    [Fact]
    public void AMayhemCardsScreenStopsTheTurnStartAndTheTurnGoesOnAfterIt()
    {
        // `MayhemPower.AfterAutoPrePlayPhaseEntered` plays the top card; Headbutt asks inside it.
        var fight = Fight
            .Hand()
            .Energy(3)
            .Draw([.. Pile(IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.Headbutt)])
            .Discard(Card(IC.StrikeIronclad), Card(IC.Bash))
            .Enemy(hp: 40)
            .PlayerBuff(BuffId.MayhemPower, 1);

        fight.EndTurn();

        Assert.Equal(CardSelectionKind.DiscardToDrawPileTop, fight.Pending!.Kind);
        fight.Choose(1);

        Assert.Null(fight.Pending);
        Assert.Equal(IC.Bash, fight.State.DrawPile[0].DefId);
        Assert.Equal(5, fight.State.Hand.Count);
        Assert.Equal(40 - 9, fight.Enemy0.Hp);
    }

    /// <summary>
    /// `HellraiserPower.AfterCardDrawnEarly` plays a drawn Strike inside the hand draw, so a Seeker
    /// Strike it plays asks there -- before the rest of the draw and before `AfterPlayerTurnStart`'s
    /// Entropy -- as the game did in TCU1SMLQHY (jobs-pickorder, steps 10 and 18: the Seeker Strike
    /// pick, then Entropy's). The Strike used to be queued and played once the step was over, so
    /// Entropy asked first and the recorded Seeker Strike pick was refused.
    /// </summary>
    [Fact]
    public void AStrikeHellraiserPlaysAsksInsideTheHandDrawBeforeEntropy()
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Energy(3)
            .Draw([.. Pile(CL.SeekerStrike, IC.DefendIronclad, IC.Bash, IC.Anger, IC.Inflame, IC.ShrugItOff, IC.IronWave, IC.Thunderclap, IC.Armaments)])
            .Enemy(hp: 400)
            .PlayerBuff(BuffId.Hellraiser, 1)
            .PlayerBuff(BuffId.EntropyPower, 1);

        fight.EndTurn();

        // The Seeker Strike's pick, with nothing drawn after it yet.
        Assert.Equal(CardSelectionKind.DrawPileToHand, fight.Pending!.Kind);
        Assert.Equal(CL.SeekerStrike, fight.Pending.SourceCardDefId);
        Assert.Empty(fight.State.Hand);
        Assert.Equal(400 - 9, fight.Enemy0.Hp);
        fight.Choose(0);

        // The draw goes on (the played Strike was one of the five), then Entropy asks over it.
        Assert.Equal(CardSelectionKind.TransformInHand, fight.Pending!.Kind);
        Assert.Equal(1 + 4, fight.State.Hand.Count);
        Assert.Equal(5, fight.Pending.Candidates.Count);
        fight.Choose(0);
        Assert.Null(fight.Pending);
    }

    [Theory]
    [InlineData(61)] // Touch of Insanity: `FromHand(prefs(prompt, 1))`
    [InlineData(48)] // Power Potion: `FromChooseACardScreen(..., canSkip: true)`
    [InlineData(53)] // Skill Potion
    [InlineData(10)] // Colorless Potion
    public void APotionsScreenStopsTheUseAndTheAnswerFinishesIt(int potionId)
    {
        // `PotionModel.OnUseWrapper` awaits `OnUse`, which awaits the screen: the potion is gone
        // from the belt (RemoveBeforeUse) and nothing after the use has run while it stands.
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad), Card(IC.Bash))
            .Energy(3)
            .Enemy(hp: 40);
        fight.State.PotionSlots[0] = potionId;

        fight.Potion(0);

        Assert.NotNull(fight.Pending?.Replay);
        Assert.Equal(0, fight.State.PotionSlots[0]);
        int hand = fight.State.Hand.Count;
        fight.Choose(0);

        Assert.Null(fight.Pending);
        Assert.Equal(0, fight.State.PotionSlots[0]);
        Assert.True(fight.State.Hand.Count >= hand);
    }
}
