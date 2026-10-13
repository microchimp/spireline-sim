using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Welcome to Wongos has four options and the fourth, Leave, is not free: <c>Leave</c> downgrades
/// one upgraded card, <c>base.Rng.NextItem(Deck.Where(IsUpgraded))</c>. The emulator took the
/// generic way out and left the deck alone.
/// </summary>
[CoversEvent("WelcomeToWongos")]
public class WongosLeaveTests
{
    private static RunEngine Wongos(string seed, int upgraded)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        for (int i = 0; i < upgraded; i++)
        {
            engine.State.Deck[i] = engine.State.Deck[i] with { Upgraded = true };
        }

        engine.State.Gold = 50;
        RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventWelcomeToWongos);
        return engine;
    }

    [Theory]
    [InlineData("ABCDEF")]
    [InlineData("JJ39U0LLVY")]
    [InlineData("M4HT4KF030")]
    public void LeavingDowngradesOneUpgradedCardPickedOffTheEventsStream(string seed)
    {
        var engine = Wongos(seed, upgraded: 4);
        var candidates = Enumerable
            .Range(0, engine.State.Deck.Count)
            .Where(i => engine.State.Deck[i].Upgraded)
            .ToList();
        var oracle = new GameRng(
            unchecked(
                engine.State.Rng.Seed
                    + (uint)DeterministicHash.GetDeterministicHashCode("WELCOME_TO_WONGOS")
            )
        );
        int expected = oracle.NextItem(candidates);

        Assert.Equal(0, engine.Step(RunConstants.EventSkipAction, -1, out _, out _, out _));

        var after = Enumerable
            .Range(0, engine.State.Deck.Count)
            .Where(i => engine.State.Deck[i].Upgraded)
            .ToList();
        Assert.Equal(candidates.Where(i => i != expected), after);
    }

    [Fact]
    public void LeavingWithNothingUpgradedDrawsNothing()
    {
        var engine = Wongos("ABCDEF", upgraded: 0);

        Assert.Equal(0, engine.Step(RunConstants.EventSkipAction, -1, out _, out _, out _));

        Assert.DoesNotContain(engine.State.Deck, card => card.Upgraded);
        Assert.Null(engine.State.EventRngStream);
    }
}
