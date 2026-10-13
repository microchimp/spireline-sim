using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// What the Spoils Map's chest pays: <c>OneOffSynchronizer.DoTreasureRoomRewards</c> adds
/// <c>TryHandleSpoilsMap</c> to the chest's own gold, and the point the run stands on has to
/// carry the quest marker for it to pay. See <see cref="SpoilsMapQuestTests" /> for the
/// map the marker sits on.
/// </summary>
public class SpoilsMapPayoutTests
{
    private const int ChestCol = SpoilsMapQuestTests.ChestCol;

    private static RunEngine ActTwo(string seed) => SpoilsMapQuestTests.ActTwo(seed);

    private static int ChestRow(RunState state) => SpoilsMapQuestTests.ChestRow(state);

    /// <summary>The run on the map one row under the chest, so the next move opens it.</summary>
    private static RunEngine BelowTheChest(string seed, Action<RunState>? tweak = null)
    {
        var engine = SpoilsMapQuestTests.ActTwo(seed);
        var state = engine.State;
        int rowBelow = ChestRow(state) - 1;
        var parent = state
            .MapNodes.Values.Where(node => node.Row == rowBelow && node.Children.Count > 0)
            .OrderBy(node => node.Col)
            .First();
        state.AwaitingActStartNode = false;
        state.CurrentMapCoord = (parent.Col, parent.Row);
        state.Floor = 16 + parent.Row;
        state.Phase = RunPhase.Map;
        RunMapGenerator.RefreshMapOptions(state);
        tweak?.Invoke(state);
        return engine;
    }

    private static void OpenTheChest(RunEngine engine)
    {
        int action = Array.IndexOf(engine.State.MapNodeTypes, RunConstants.NodeRelic);
        Assert.True(action >= 0, "the row under the chest offers no chest");
        Assert.Equal(0, engine.Step(action, -1, out _, out _, out _));
        Assert.Equal(RunPhase.Treasure, engine.State.Phase);
    }

    private static bool HoldsTheCard(RunState state) => SpoilsMapQuestTests.HoldsTheCard(state);

    /// <summary>
    /// The chest pays its own gold off the Rewards stream and the card adds 600 on top, so
    /// the same chest opened with the marker cleared is the baseline: the two runs are the
    /// same in everything else.
    /// </summary>
    [Fact]
    public void TheChestPaysSixHundredOnTopOfItsOwnGoldAndTheCardLeavesTheDeck()
    {
        var quest = BelowTheChest("SPOILS0000");
        var plain = BelowTheChest("SPOILS0000", state => state.SpoilsQuestCoord = null);
        int before = quest.State.Gold;
        Assert.Equal(before, plain.State.Gold);

        OpenTheChest(quest);
        OpenTheChest(plain);

        Assert.Equal(plain.State.Gold + RunConstants.SpoilsMapGold, quest.State.Gold);
        Assert.True(plain.State.Gold > before, "the chest's own gold was not paid");
        Assert.False(HoldsTheCard(quest.State));
        Assert.True(HoldsTheCard(plain.State));
        Assert.Null(quest.State.SpoilsQuestCoord);
        // The card is no stream draw: the Rewards stream stands where the plain chest left it.
        Assert.Equal(
            plain.State.PlayerRng.Rewards.CallCount,
            quest.State.PlayerRng.Rewards.CallCount
        );
    }

    /// <summary>
    /// 600 is the card's <c>GoldVar</c> base value, gained as it stands. Poverty (from A3)
    /// cuts the chest's own roll and the monsters' gold, not this.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(10)]
    public void PovertyDoesNotCutTheSixHundred(int ascension)
    {
        var quest = BelowTheChest("SPOILS0000", s => s.AscensionLevel = ascension);
        var plain = BelowTheChest(
            "SPOILS0000",
            s =>
            {
                s.AscensionLevel = ascension;
                s.SpoilsQuestCoord = null;
            }
        );

        OpenTheChest(quest);
        OpenTheChest(plain);

        Assert.Equal(RunConstants.SpoilsMapGold, quest.State.Gold - plain.State.Gold);
    }

    /// <summary>
    /// <c>TryHandleSpoilsMap</c> pays for the Spoils Maps in the DECK. A card removed
    /// since the map was made takes its marker with it (<c>BeforeCardRemoved</c>), and a
    /// chest opened after that pays nothing extra.
    /// </summary>
    [Fact]
    public void ACardRemovedFromTheDeckBeforeTheChestPaysNothing()
    {
        var removed = BelowTheChest(
            "SPOILS0000",
            state => state.Deck.RemoveAll(card => card.DefId == RunConstants.SpoilsMapCard)
        );
        var plain = BelowTheChest(
            "SPOILS0000",
            state =>
            {
                state.Deck.RemoveAll(card => card.DefId == RunConstants.SpoilsMapCard);
                state.SpoilsQuestCoord = null;
            }
        );

        OpenTheChest(removed);
        OpenTheChest(plain);

        Assert.Equal(plain.State.Gold, removed.State.Gold);
    }

    /// <summary>
    /// The marker is on a map POINT. A "?" that turns out to be a chest is some other
    /// point, so the chest it opens pays nothing extra and the quest stays on the real one.
    /// </summary>
    [Fact]
    public void AChestOpenedFromADifferentPointPaysNothing()
    {
        var state = ActTwo("SPOILS0000").State;
        state.CurrentMapCoord = (ChestCol, ChestRow(state) - 1);
        int gold = state.Gold;

        RunNonCombatEffects.CompleteSpoilsMapQuest(state);

        Assert.Equal(gold, state.Gold);
        Assert.True(HoldsTheCard(state));
        Assert.NotNull(state.SpoilsQuestCoord);
    }

    [Fact]
    public void TheQuestPaysOnlyOnce()
    {
        var engine = BelowTheChest("SPOILS0000");
        OpenTheChest(engine);
        int gold = engine.State.Gold;

        RunNonCombatEffects.CompleteSpoilsMapQuest(engine.State);

        Assert.Equal(gold, engine.State.Gold);
    }
}
