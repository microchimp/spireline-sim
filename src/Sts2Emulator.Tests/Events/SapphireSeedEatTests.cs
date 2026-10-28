using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Sapphire Seed's Eat heals, asks which card to upgrade, and then the event is over --
/// whether or not the player had a card to pick. `SapphireSeed.Eat` awaits the upgrade
/// selection, upgrades only if the pick is not null, and calls `SetEventFinished` on the
/// line after, so an empty selection costs nothing and closes the page.
///
/// The emulator healed, opened the selection, and ignored `BeginDeckSelection` refusing to
/// open it. With nothing upgradable that left the event page up with the same options, so
/// an agent that chose Eat again got the same screen back: a run of the generation-5 agent
/// (a deck the Tablet of Truth had upgraded in full) sat on it for the whole wall-clock cap.
/// </summary>
public class SapphireSeedEatTests
{
    private static RunEngine AtTheSeed(bool fullyUpgraded)
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        engine.State.EventId = RunConstants.EventSapphireSeed;
        engine.State.Phase = RunPhase.Event;
        engine.State.PlayerMaxHp = 80;
        engine.State.PlayerHp = 20;
        if (fullyUpgraded)
        {
            for (int i = 0; i < engine.State.Deck.Count; i++)
            {
                engine.State.Deck[i] = engine.State.Deck[i] with { Upgraded = true };
            }

            Assert.DoesNotContain(
                engine.State.Deck,
                RunConstants.IsRunCardUpgradable
            );
        }

        return engine;
    }

    private static int[] Offered(RunEngine engine)
    {
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        return Enumerable.Range(0, mask.Length).Where(index => mask[index] != 0).ToArray();
    }

    [Fact]
    public void EatWithNothingToUpgradeStillHeals()
    {
        var engine = AtTheSeed(fullyUpgraded: true);

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(29, engine.State.PlayerHp);
    }

    /// <summary>
    /// The page closes: the event is on its result page, which offers Proceed (and the generic leave) and nothing
    /// of the Sapphire Seed's own, and Proceed leaves for the map.
    /// </summary>
    [Fact]
    public void EatWithNothingToUpgradeFinishesTheEvent()
    {
        var engine = AtTheSeed(fullyUpgraded: true);

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(RunPhase.Event, engine.State.Phase);
        Assert.Equal(RunConstants.EventResultPending, engine.State.EventId);
        Assert.All(Offered(engine), a => Assert.True(a is 0 or RunConstants.EventSkipAction));

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(RunPhase.Map, engine.State.Phase);
        Assert.NotEqual(RunConstants.EventSapphireSeed, engine.State.EventId);
    }

    /// <summary>With a card to upgrade the selection opens first, as before.</summary>
    [Fact]
    public void EatWithAnUpgradableCardOpensTheSelection()
    {
        var engine = AtTheSeed(fullyUpgraded: false);

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));

        Assert.Equal(29, engine.State.PlayerHp);
        Assert.Equal(RunPhase.TransformSelect, engine.State.Phase);
        Assert.Equal(DeckSelection.Upgrade, engine.State.PendingSelectionKind);
    }
}
