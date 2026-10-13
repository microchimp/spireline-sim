using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CardModel.OnPlay` is awaited before the played card is moved to its result pile, so a
/// card that opens a selection screen or queues auto-plays lands in the exhaust or discard
/// pile AFTER the cards that screen and that queue send there (Purity.OnPlay,
/// Catastrophe.OnPlay). Found by the night03 referee replay: PURITY exhaust order
/// [picked, PURITY] in the game, [PURITY, picked] here.
/// </summary>
public class PlayedCardPlacementTests
{
    [Fact]
    public void Purity_ExhaustsItsPickBeforeItself()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand =
        [
            new CardInstance(CL.Purity, false),
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.Bash, false),
        ];
        state.Energy = 3;

        CombatEngine.Step(state, 0, new Random(0));
        Assert.NotNull(state.PendingSelection);
        // The card is in the play pile while the screen is up, not in the exhaust pile yet.
        Assert.Empty(state.ExhaustPile);

        CombatEngine.Step(state, 0, new Random(0)); // pick the Strike
        CombatEngine.Step(state, 0, new Random(0)); // pick the Bash, which empties the hand

        Assert.Null(state.PendingSelection);
        Assert.Equal(
            new[] { IC.StrikeIronclad, IC.Bash, CL.Purity },
            state.ExhaustPile.Select(c => c.DefId).ToArray()
        );
    }

    [Fact]
    public void Purity_DecliningStillExhaustsItself()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(CL.Purity, false), new CardInstance(IC.Bash, false)];
        state.Energy = 3;

        CombatEngine.Step(state, 0, new Random(0));
        var skip = CombatEngine.ValidActions(state).Max();
        CombatEngine.Step(state, skip, new Random(0));

        Assert.Null(state.PendingSelection);
        Assert.Equal(new[] { CL.Purity }, state.ExhaustPile.Select(c => c.DefId).ToArray());
    }

    [Fact]
    public void Catastrophe_LandsInTheDiscardAfterTheCardsItAutoPlayed()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [new CardInstance(CL.Catastrophe, false)];
        state.DrawPile =
        [
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.StrikeIronclad, false),
            new CardInstance(IC.StrikeIronclad, false),
        ];
        state.DiscardPile = [];
        state.Energy = 3;

        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(CL.Catastrophe, state.DiscardPile[^1].DefId);
        Assert.Equal(3, state.DiscardPile.Count);
    }
}
