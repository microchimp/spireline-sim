using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Slippery Bridge's re-roll rule (<c>SlipperyBridge.GetNewRandomCard</c>). The first roll
/// takes the deck's non-Basic cards. Each Hold On then notes the threatened card as skipped
/// and draws ONCE from every removable card of a different id that has not been skipped --
/// Basic cards included. The emulator re-drew EventPage + 1 times from the non-Basic cards
/// every time, which moved both the pick and the event's stream (real runs: the game removed
/// a Strike the emulator kept, 663 runs).
/// </summary>
[CoversEvent("SlipperyBridge")]
public class SlipperyBridgeRerollTests
{
    private static int Card(string name) => GeneratedData.Cards.FindId(name)!.Value;

    private static RunEngine Bridge(string seed)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        var deck = engine.State.Deck;
        deck.Clear();
        foreach (
            string name in new[]
            {
                "StrikeIronclad",
                "StrikeIronclad",
                "StrikeIronclad",
                "DefendIronclad",
                "DefendIronclad",
                "Bash",
                "Anger",
                "BodySlam",
                "BodySlam",
                "ShrugItOff",
                "IronWave",
                "AscendersBane",
            }
        )
        {
            deck.Add(new CardInstance(Card(name), Upgraded: false));
        }

        RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventSlipperyBridge);
        return engine;
    }

    /// <summary>
    /// The rule written out again, independently of the engine, on its own copy of the
    /// event's stream (seed + the entry's deterministic hash).
    /// </summary>
    private sealed class Oracle(RunState state)
    {
        private readonly GameRng _rng = new(
            unchecked(
                state.Rng.Seed + (uint)DeterministicHash.GetDeterministicHashCode("SLIPPERY_BRIDGE")
            )
        );
        private readonly List<int> _skipped = [];
        private int _held = -1;

        private bool Removable(int i) =>
            !GeneratedData.Cards.Get(state.Deck[i].DefId).Eternal && !_skipped.Contains(i);

        public int Next()
        {
            List<int> list;
            if (_held < 0)
            {
                list = Enumerable
                    .Range(0, state.Deck.Count)
                    .Where(i =>
                        Removable(i)
                        && GeneratedData.Cards.Get(state.Deck[i].DefId).Rarity != CardRarity.Basic
                    )
                    .ToList();
            }
            else
            {
                _skipped.Add(_held);
                list = Enumerable
                    .Range(0, state.Deck.Count)
                    .Where(i => Removable(i) && state.Deck[i].DefId != state.Deck[_held].DefId)
                    .ToList();
            }

            if (list.Count == 0)
            {
                list = Enumerable
                    .Range(0, state.Deck.Count)
                    .Where(i => !GeneratedData.Cards.Get(state.Deck[i].DefId).Eternal)
                    .ToList();
            }

            _held = _rng.NextItem(list);
            return _held;
        }
    }

    [Theory]
    [InlineData("ABCDEF")]
    [InlineData("JJ39U0LLVY")]
    [InlineData("M4HT4KF030")]
    [InlineData("X2QEM46QX8")]
    [InlineData("0KFMJ8NU8E")]
    public void EveryHoldOnDrawsOnceFromTheOtherCardsBasicsIncluded(string seed)
    {
        var engine = Bridge(seed);
        var oracle = new Oracle(engine.State);

        Assert.Equal(oracle.Next(), RunNonCombatEffects.SlipperyBridgeCardIndex(engine.State));
        for (int hold = 0; hold < 8; hold++)
        {
            Assert.Equal(0, engine.Step(1, -1, out _, out _, out _));
            Assert.Equal(oracle.Next(), RunNonCombatEffects.SlipperyBridgeCardIndex(engine.State));
        }
    }

    [Fact]
    public void ABasicCardCanBeThreatenedAfterAHoldOn()
    {
        bool sawBasic = false;
        foreach (string seed in new[] { "ABCDEF", "JJ39U0LLVY", "M4HT4KF030", "X2QEM46QX8" })
        {
            var engine = Bridge(seed);
            for (int hold = 0; hold < 8; hold++)
            {
                engine.Step(1, -1, out _, out _, out _);
                int index = RunNonCombatEffects.SlipperyBridgeCardIndex(engine.State);
                sawBasic |=
                    GeneratedData.Cards.Get(engine.State.Deck[index].DefId).Rarity
                    == CardRarity.Basic;
            }
        }

        Assert.True(sawBasic, "a re-roll never named a Basic card");
    }

    [Fact]
    public void TheCardNamedAfterAHoldOnIsAlwaysADifferentCard()
    {
        var engine = Bridge("ABCDEF");
        for (int hold = 0; hold < 8; hold++)
        {
            int before = engine.State.Deck[
                RunNonCombatEffects.SlipperyBridgeCardIndex(engine.State)
            ].DefId;
            engine.Step(1, -1, out _, out _, out _);
            int after = engine.State.Deck[
                RunNonCombatEffects.SlipperyBridgeCardIndex(engine.State)
            ].DefId;
            Assert.NotEqual(before, after);
        }
    }
}
