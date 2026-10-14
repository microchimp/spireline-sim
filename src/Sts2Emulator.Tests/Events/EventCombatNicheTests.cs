using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// When the game spends Niche draws outside a fight's own start.
///
/// A Combat-layout event (Punch Off, The Lantern Key) builds its encounter's creatures as the
/// room is entered (<c>EventRoom.EnterInternal</c> -> <c>EventModel.GenerateInternalCombatState</c>),
/// and <c>CombatState.CreateCreature</c> rolls each one's HP on <c>Rng.Niche</c> there; the fight
/// reuses them (<c>CombatRoom.ShouldCreateCombat</c> false). The emulator rolled them when the fight
/// started, so Niche was behind after entry and stayed behind when no fight followed (run referee:
/// 84G7XUWQD6 Punch Off +2, MU4GCA5UPC Lantern Key +1). Royal Stamp shuffles its candidates on Niche
/// before its screen, which the emulator skipped.
/// </summary>
public class EventCombatNicheTests
{
    private static RunEngine Entered(int eventId, string seed = "ABCDEF")
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        RunNonCombatEffects.BeginEvent(engine.State, eventId);
        return engine;
    }

    private static int[] FightHps(RunEngine engine)
    {
        Assert.Equal(0, engine.Step(1, -1, out _, out _, out _)); // the fight's page
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _)); // the fight
        Assert.Equal(RunPhase.Combat, engine.State.Phase);
        return engine.State.ActiveCombat!.Enemies.Select(e => e.MaxHp).ToArray();
    }

    [Theory]
    [InlineData(RunConstants.EventPunchOff, 2)]
    [InlineData(RunConstants.EventTheLanternKey, 1)]
    public void EnteringTheRoomRollsEachCreatureOnNiche(int eventId, int creatures)
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        int before = engine.State.Rng.Niche.CallCount;

        RunNonCombatEffects.BeginEvent(engine.State, eventId);

        Assert.Equal(before + creatures, engine.State.Rng.Niche.CallCount);
    }

    [Fact]
    public void AnEventWithNoCombatLayoutDrawsNothing()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        int before = engine.State.Rng.Niche.CallCount;

        RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventEndlessConveyor);

        Assert.Equal(before, engine.State.Rng.Niche.CallCount);
    }

    [Theory]
    [InlineData(RunConstants.EventPunchOff)]
    [InlineData(RunConstants.EventTheLanternKey)]
    public void TheFightTakesTheHpRolledOnEntryAndDrawsNoMore(int eventId)
    {
        var plain = Entered(eventId);
        int[] entryHps = FightHps(plain);

        // A Niche draw between entry and the fight moves nothing the room already rolled.
        var moved = Entered(eventId);
        int afterEntry = moved.State.Rng.Niche.CallCount;
        moved.State.Rng.Niche.NextInt(1000);
        int[] hps = FightHps(moved);

        Assert.Equal(entryHps, hps);
        Assert.Equal(afterEntry + 1, moved.State.ActiveCombat!.NicheHpRng!.CallCount);
        Assert.Equal(afterEntry + 1, moved.State.Rng.Niche.CallCount);
    }

    [Fact]
    public void ReturningTheKeyKeepsTheKnightsDraw()
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        int before = engine.State.Rng.Niche.CallCount;
        RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventTheLanternKey);

        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _)); // Return The Key: no fight

        Assert.Equal(before + 1, engine.State.Rng.Niche.CallCount);
    }

    [Fact]
    public void TheEntryRollTravelsWithAClone()
    {
        var engine = Entered(RunConstants.EventPunchOff);

        var clone = engine.State.Clone();

        Assert.Equal(engine.State.EventCombatHps, clone.EventCombatHps);
        Assert.NotSame(engine.State.EventCombatHps, clone.EventCombatHps);
    }

    /// <summary>
    /// <c>RoyalStamp.AfterObtained</c>: the in-place unstable shuffle of the stampable deck cards on
    /// Niche, one draw per card after the first; a Power is not a candidate.
    /// </summary>
    [Theory]
    [InlineData(1, 0)]
    [InlineData(4, 3)]
    public void RoyalStampShufflesItsCandidatesOnNiche(int stampable, int draws)
    {
        var deck = Enumerable.Range(0, stampable).Select(_ => new CardInstance(SI.Slice, false)).ToList();
        deck.Add(new CardInstance(IC.Inflame, false));
        var state = new RunState { Deck = deck };
        int before = state.Rng.Niche.CallCount;

        RunNonCombatEffects.ApplyRelicPickup(state, RelicEffects.RoyalStamp);

        Assert.Equal(before + draws, state.Rng.Niche.CallCount);
        Assert.Equal(RunPhase.TransformSelect, state.Phase);
    }
}
