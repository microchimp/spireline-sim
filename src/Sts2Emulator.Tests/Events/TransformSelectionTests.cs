using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// <c>CardSelectCmd.FromDeckForTransformation</c> offers <c>c.Type != Quest &amp;&amp; c.IsTransformable</c>,
/// and a deck card is transformable exactly when it is not Eternal. Every event and relic that
/// transforms a picked card (Aroma of Chaos, Morphic Grove, Symbiote, the Trial, Whispering Hollow,
/// the Endless Conveyor's Jelly Liver, Astrolabe, Claws, New Leaf) goes through it. The emulator
/// accepted any card, so Ascender's Bane (every Ascension 10 deck) could be transformed away.
/// </summary>
[CoversEvent("AromaOfChaos")]
public class TransformSelectionTests
{
    [Fact]
    public void ATransformSelectionDoesNotOfferEternalOrQuestCards()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventAromaOfChaos);
        int bane = GeneratedData.Cards.FindId("AscendersBane")!.Value;
        int egg = RunNonCombatEffects.ByrdonisEggCard;
        engine.State.Deck.Add(new CardInstance(bane, Upgraded: false));
        engine.State.Deck.Add(new CardInstance(egg, Upgraded: false));
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        for (int i = 0; i < engine.State.Deck.Count; i++)
        {
            bool offered = RunNonCombatEffects.CanSelectCard(engine.State, i);
            int id = engine.State.Deck[i].DefId;
            Assert.Equal(id != bane && id != egg, offered);
        }

        // And the engine refuses the Bane outright.
        int baneIndex = engine.State.Deck.FindIndex(card => card.DefId == bane);
        Assert.Equal(-1, engine.Step(baneIndex, -1, out _, out _, out _));
    }
}
