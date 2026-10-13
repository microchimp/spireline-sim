using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Per-event <c>IsAllowed</c> rules once Acts 2 and 3 are dealt.
///
/// <para>
/// <c>IsEventAllowed</c> used to refuse nine shared events and four later-act events
/// outright ("the emulator models Act 1 only"), so Acts 2 and 3 never offered Doll Room,
/// Symbiote, Relic Trader and the rest, and every refusal moved the ring's cursor. The
/// game gates each of them on its own state, <c>CurrentActIndex</c> included
/// (decompiled <c>Models.Events.*.IsAllowed</c>); the ring walk is
/// <c>RoomSet.EnsureNextEventIsValid</c> and <c>ActModel.PullNextEvent</c>.
/// </para>
/// </summary>
public class EventEligibilityTests
{
    private static RunState StockedAt(int actIndex)
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var state = engine.State;
        state.CurrentActIndex = actIndex;
        state.Gold = 999;
        state.PotionSlots[0] = RunNonCombatEffects.NamedPotion("Ashwater");
        state.Relics.Clear();
        foreach (int id in TradableRelics(6))
        {
            state.Relics.Add(new RelicInstance(id));
        }

        return state;
    }

    private static IEnumerable<int> TradableRelics(int count)
    {
        var found = new List<int>();
        foreach (var def in GeneratedData.Relics.All)
        {
            if (def.Id != 0 && def.IsTradable)
            {
                found.Add(def.Id);
                if (found.Count == count)
                {
                    break;
                }
            }
        }

        Assert.Equal(count, found.Count);
        return found;
    }

    /// <summary>
    /// Expected result at act index 0, 1 and 2, read off each class's <c>IsAllowed</c>:
    /// Crystal Sphere, Potion Courier and Symbiote are <c>&gt; 0</c>; Doll Room, Stone of
    /// All Time and Welcome to Wongos are <c>== 1</c>; Ranwid, Relic Trader and Fake
    /// Merchant refuse index 0 only.
    /// </summary>
    [Theory]
    [InlineData(RunConstants.EventCrystalSphere, false, true, true)]
    [InlineData(RunConstants.EventDollRoom, false, true, false)]
    [InlineData(RunConstants.EventFakeMerchant, false, true, true)]
    [InlineData(RunConstants.EventPotionCourier, false, true, true)]
    [InlineData(RunConstants.EventRanwidTheElder, false, true, true)]
    [InlineData(RunConstants.EventRelicTrader, false, true, true)]
    [InlineData(RunConstants.EventStoneOfAllTime, false, true, false)]
    [InlineData(RunConstants.EventSymbiote, false, true, true)]
    [InlineData(RunConstants.EventWelcomeToWongos, false, true, false)]
    public void TheNineSharedEventsFollowTheirActIndexRule(
        int eventId,
        bool act0,
        bool act1,
        bool act2
    )
    {
        bool[] expected = [act0, act1, act2];
        for (int act = 0; act < 3; act++)
        {
            var state = StockedAt(act);
            Assert.Equal(
                expected[act],
                RunNonCombatEffects.IsEventAllowedForTests(state, eventId)
            );
        }
    }

    [Fact]
    public void RanwidNeedsGoldAPotionAndATradableRelicAllAtOnce()
    {
        // Game: every one of the three tests must pass. It was an OR.
        var state = StockedAt(1);
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventRanwidTheElder));

        state.Gold = 99;
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventRanwidTheElder));

        state = StockedAt(1);
        Array.Clear(state.PotionSlots);
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventRanwidTheElder));

        state = StockedAt(1);
        state.Relics.Clear();
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventRanwidTheElder));
    }

    [Fact]
    public void RelicTraderCountsTradableRelicsNotRelics()
    {
        var state = StockedAt(1);
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventRelicTrader));

        // Four tradable ones plus the starter relic: five relics, four tradable.
        state.Relics.Clear();
        foreach (int id in TradableRelics(4))
        {
            state.Relics.Add(new RelicInstance(id));
        }

        state.Relics.Add(new RelicInstance(RunConstants.RelicBurningBlood));
        Assert.Equal(5, state.Relics.Count);
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventRelicTrader));
    }

    [Fact]
    public void StoneOfAllTimeWantsAPotionAndNothingElse()
    {
        var state = StockedAt(1);
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventStoneOfAllTime));

        // A deck full of enchantable cards was an alternative; the game has none.
        Array.Clear(state.PotionSlots);
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventStoneOfAllTime));
    }

    [Fact]
    public void FakeMerchantWantsAHundredGoldOrAFoulPotion()
    {
        var state = StockedAt(1);
        state.Gold = 99;
        Array.Clear(state.PotionSlots);
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventFakeMerchant));

        state.PotionSlots[0] = RunNonCombatEffects.NamedPotion("FoulPotion");
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventFakeMerchant));

        Array.Clear(state.PotionSlots);
        state.Gold = 100;
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventFakeMerchant));
    }

    [Theory]
    [InlineData(RunConstants.EventCrystalSphere)]
    [InlineData(RunConstants.EventWelcomeToWongos)]
    public void GoldGatedSharedEventsWantAHundredGold(int eventId)
    {
        var state = StockedAt(1);
        state.Gold = 99;
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, eventId));
        state.Gold = 100;
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, eventId));
    }

    /// <summary>
    /// The four later-act events declare no act test of their own: the pool decides where
    /// they appear. They were refused unconditionally by <c>IsEventAllowed</c>, so Hive and
    /// Glory could never show them.
    /// </summary>
    [Theory]
    [InlineData(RunConstants.EventColorfulPhilosophers, 1)]
    [InlineData(RunConstants.EventColossalFlower, 1)]
    [InlineData(RunConstants.EventGraveOfTheForgotten, 2)]
    [InlineData(RunConstants.EventRoundTeaParty, 2)]
    public void LaterActPoolEventsAreAllowedInTheirActs(int eventId, int actIndex)
    {
        var state = StockedAt(actIndex);
        state.PlayerHp = state.PlayerMaxHp;
        // Grave of the Forgotten wants a card Souls Power will take, i.e. one that exhausts.
        state.Deck.Add(new(RunNonCombatEffects.NamedCard("Afterlife"), false));
        Assert.True(RunNonCombatEffects.IsEventAllowedForTests(state, eventId));
    }

    [Fact]
    public void LaterActPoolEventsKeepTheirOwnRunStateGates()
    {
        var state = StockedAt(1);
        state.PlayerHp = 18;
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventColossalFlower));
        state.PlayerHp = 19;
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventColossalFlower));

        state.PlayerHp = 11;
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventRoundTeaParty));
        state.PlayerHp = 12;
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventRoundTeaParty));
    }

    [Theory]
    [InlineData(0, true)]
    [InlineData(1, true)]
    [InlineData(2, false)]
    public void TeaMasterIsOnlyInTheFirstTwoActs(int actIndex, bool allowed)
    {
        var state = StockedAt(actIndex);
        Assert.Equal(allowed, RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventTeaMaster));
    }

    [Theory]
    [InlineData(0, true)]
    [InlineData(1, false)]
    [InlineData(2, false)]
    public void TheLegendsWereTrueIsOnlyInTheFirstAct(int actIndex, bool allowed)
    {
        var state = StockedAt(actIndex);
        Assert.Equal(
            allowed,
            RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventTheLegendsWereTrue)
        );
    }

    [Theory]
    [InlineData(148, false)]
    [InlineData(149, true)]
    public void LuminousChoirWantsItsCanonicalPriceInGold(int gold, bool allowed)
    {
        // Game: `Gold >= GoldVar(149)`, tested on the canonical model. It was
        // `Gold >= 100 || Deck.Count >= 3`, which a starting deck always passed.
        var state = StockedAt(0);
        state.Gold = gold;
        Assert.Equal(
            allowed,
            RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventLuminousChoir)
        );
    }

    [Fact]
    public void AmalgamatorNeedsTwoRemovableStrikesAndTwoRemovableDefends()
    {
        var state = StockedAt(0);
        int strike = RunNonCombatEffects.NamedCard("StrikeIronclad");
        int defend = RunNonCombatEffects.NamedCard("DefendIronclad");
        state.Deck = [new(strike, false), new(strike, false), new(defend, false)];
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventAmalgamator));
        state.Deck.Add(new(defend, false));
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventAmalgamator));
    }

    /// <summary>
    /// <c>CardModel.IsRemovable</c> is <c>!Keywords.Contains(Eternal)</c>: every Eternal
    /// curse stays (the helper only knew Ascender's Bane) and a quest card does not
    /// (the helper wrongly refused those). Morphic Grove counts the removable cards.
    /// </summary>
    [Fact]
    public void RemovabilityIsTheEternalKeywordAndNothingElse()
    {
        var state = StockedAt(0);
        int strike = RunNonCombatEffects.NamedCard("StrikeIronclad");

        state.Deck = [new(strike, false), new(RunNonCombatEffects.NamedCard("BadLuck"), false)];
        Assert.False(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventMorphicGrove));

        state.Deck = [new(strike, false), new(RunNonCombatEffects.NamedCard("SpoilsMap"), false)];
        Assert.True(RunNonCombatEffects.IsEventAllowed(state, RunConstants.EventMorphicGrove));
    }

    [Fact]
    public void EventsWithNoIsAllowedOverrideAreAlwaysAllowed()
    {
        // DrowningBeacon, SelfHelpBook, SpiritGrafter and InfestedAutomaton carry no rule
        // in the game (base EventModel.IsAllowed is true); the emulator had invented
        // `Hp > 13` and deck-content tests for them.
        var state = StockedAt(0);
        state.PlayerHp = 1;
        state.Deck = [new(RunNonCombatEffects.NamedCard("BadLuck"), false)];
        foreach (int id in new[]
        {
            RunConstants.EventDrowningBeacon,
            RunConstants.EventSelfHelpBook,
            RunConstants.EventSpiritGrafter,
            RunConstants.EventInfestedAutomaton,
        })
        {
            Assert.True(RunNonCombatEffects.IsEventAllowed(state, id));
        }
    }

    /// <summary>
    /// <c>RoomSet.EnsureNextEventIsValid</c>: a disallowed candidate costs a ring
    /// position (<c>eventsVisited++</c>); an allowed one is pulled and
    /// <c>MarkRoomVisited</c> steps past it. No RNG is drawn either way.
    /// </summary>
    [Fact]
    public void ActTwoRingOffersASharedEventAndActOneRingPassesItOver()
    {
        foreach (int act in new[] { 0, 1 })
        {
            var state = StockedAt(act);
            state.EventSequence =
            [
                RunConstants.EventDollRoom,
                RunConstants.EventBrainLeech,
                RunConstants.EventSymbiote,
            ];
            state.EventSequenceIndex = 0;
            state.VisitedEventIds.Clear();
            int draws = state.Rng.UpFront.CallCount;

            RunNonCombatEffects.EnterEvent(state);

            if (act == 1)
            {
                Assert.Equal(RunConstants.EventDollRoom, state.EventId);
                Assert.Equal(1, state.EventSequenceIndex);
            }
            else
            {
                // Doll Room refused (one position), Brain Leech pulled (one more).
                Assert.Equal(RunConstants.EventBrainLeech, state.EventId);
                Assert.Equal(2, state.EventSequenceIndex);
            }

            Assert.Equal(draws, state.Rng.UpFront.CallCount);
        }
    }

    [Fact]
    public void TheEventRingStartsAtItsSecondEntryInEveryAct()
    {
        // Each act opens on its ancient, a RoomType.Event room that RunManager marks
        // visited, so events[1] is the first real event -- as Neow does for Act 1.
        var engine = new RunEngine();
        engine.Reset("QEY9JXP3KR", 2);
        Assert.Equal(1, engine.State.EventSequenceIndex);

        Assert.True(engine.EnterNextAct());
        Assert.Equal(1, engine.State.EventSequenceIndex);
        Assert.True(engine.EnterNextAct());
        Assert.Equal(1, engine.State.EventSequenceIndex);
    }

    [Fact]
    public void NewLeafDoesNotForceAnEvent()
    {
        // The game's NewLeaf has no event hook; the emulator used to open the first
        // event as The Legends Were True whenever the relic was held.
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        var state = engine.State;
        state.Relics.Add(new RelicInstance(RunConstants.RelicNewLeaf));
        state.EventSequence = [RunConstants.EventWhisperingHollow];
        state.EventSequenceIndex = 0;
        state.Gold = 999;

        RunNonCombatEffects.EnterEvent(state);

        Assert.Equal(RunConstants.EventWhisperingHollow, state.EventId);
    }

    [Fact]
    public void ALanternKeyTurnsTheLastActsNextEventIntoRepy()
    {
        // LanternKey.ModifyNextEvent, act index 2 only. The ring still steps past the
        // displaced event, and Repy -- not it -- goes on the visited list.
        var state = StockedAt(2);
        state.EventSequence = [RunConstants.EventWhisperingHollow];
        state.EventSequenceIndex = 0;
        state.Deck.Add(new(RunNonCombatEffects.LanternKeyCard, false));

        RunNonCombatEffects.EnterEvent(state);

        Assert.Equal(RunConstants.EventWarHistorianRepy, state.EventId);
        Assert.Equal(1, state.EventSequenceIndex);
        Assert.Contains(RunConstants.EventWarHistorianRepy, state.VisitedEventIds);
        Assert.DoesNotContain(RunConstants.EventWhisperingHollow, state.VisitedEventIds);

        state = StockedAt(1);
        state.EventSequence = [RunConstants.EventWhisperingHollow];
        state.EventSequenceIndex = 0;
        state.Deck.Add(new(RunNonCombatEffects.LanternKeyCard, false));
        RunNonCombatEffects.EnterEvent(state);
        Assert.Equal(RunConstants.EventWhisperingHollow, state.EventId);
    }
}
