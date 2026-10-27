using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Sapphire Seed's Plant has no lock: `SapphireSeed.Plant` filters the deck for cards
/// Sown can enchant, awaits the selection over that list, enchants only if the pick is
/// not null, and calls `SetEventFinished`. A deck with no card Sown can take (every card
/// already carries an enchantment, say) gets an empty selection and the event ends.
///
/// The emulator offered Plant in the mask in that deck and then refused the step, so the
/// agent was told it had a move it did not.
/// </summary>
public class SapphireSeedPlantTests
{
    private static RunEngine AtTheSeedWithNothingToPlantOn()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.EventId = RunConstants.EventSapphireSeed;
        engine.State.Phase = RunPhase.Event;
        for (int i = 0; i < engine.State.Deck.Count; i++)
        {
            engine.State.Deck[i] = engine.State.Deck[i] with
            {
                Enchantment = Enchantment.Glam,
                EnchantAmount = 1,
            };
        }

        Assert.False(
            RunNonCombatEffects.CanBeginDeckSelection(
                engine.State,
                DeckSelection.Enchant,
                (int)Enchantment.Sown
            )
        );
        return engine;
    }

    [Fact]
    public void PlantWithNothingToEnchantFinishesTheEvent()
    {
        var engine = AtTheSeedWithNothingToPlantOn();
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        Assert.NotEqual(0, mask[1]);

        Assert.Equal(0, engine.Step(1, -1, out _, out _, out _));

        Assert.Equal(RunPhase.Event, engine.State.Phase);
        Assert.Equal(RunConstants.EventResultPending, engine.State.EventId);
    }
}
