using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Spoils Map: a quest card (The Legends Were True) that replaces act 2's map with an
/// hourglass and pays 600 gold at its one chest.
/// </summary>
/// <remarks>
/// <para>
/// <c>SpoilsMap.ModifyGeneratedMap</c> hands act 2 a <c>SpoilsActMap</c> in place of the
/// standard map whenever the card is in the DECK as the map is made. It is drawn from a
/// stream of its own (<c>spoils_map</c>) with the standard recipe -- seven paths, point
/// types dealt out, duplicate segments pruned -- but the paths are steered through a
/// single chest in the middle column, and nothing centres, spreads or straightens the
/// result. The card's <c>Late</c> and <c>After</c> hooks then put the quest marker on the
/// map's first Treasure point, and <c>OneOffSynchronizer.TryHandleSpoilsMap</c> pays it
/// when that point's chest is opened.
/// </para>
/// <para>
/// The exact map is checked against the game's own in
/// <see cref="ActMapShapeTests.TheActMapIsTheMapTheGameDrew" /> (a real run's act 2 is
/// flagged <c>spoils_map</c> in the fixture). The tests here pin the properties that
/// the hourglass was built to have, on seeds the game has not been asked about, and the
/// quest around it. Without the map, 436 of the 445 act-2 maps that real runs walked and
/// the emulator could not were this one.
/// </para>
/// </remarks>
public class SpoilsMapQuestTests
{
    internal const int ChestCol = RunConstants.MapStartCol;

    public static IEnumerable<object[]> Seeds() =>
        Enumerable.Range(0, 16).Select(i => new object[] { $"SPOILS{i:D4}" });

    internal static RunEngine ActTwo(string seed, bool withCard = true, int ascension = 10)
    {
        var engine = new RunEngine();
        engine.Reset(seed, ascension);
        if (withCard)
        {
            engine.State.Deck.Add(new CardInstance(RunConstants.SpoilsMapCard, Upgraded: false));
        }

        Assert.True(engine.EnterNextAct());
        return engine;
    }

    internal static int ChestRow(RunState state) => RunConstants.MapBossRowFor(state.Act) - 7;

    internal static bool HoldsTheCard(RunState state) =>
        state.Deck.Any(card => card.DefId == RunConstants.SpoilsMapCard);

    [Theory]
    [MemberData(nameof(Seeds))]
    public void TheMapIsAnHourglassAroundOneChestInTheMiddleColumn(string seed)
    {
        var state = ActTwo(seed).State;
        int chestRow = ChestRow(state);
        int bossRow = RunConstants.MapBossRowFor(state.Act);

        var chest = Assert.Single(
            state.MapNodes.Values.Where(n => n.NodeType == RunConstants.NodeRelic)
        );
        Assert.Equal((ChestCol, chestRow), (chest.Col, chest.Row));

        // A row reaches as far from the middle column as it is from the chest's row, and
        // as it is from the boss: the glass is narrow at the chest and again at the top.
        foreach (
            var node in state.MapNodes.Values.Where(n =>
                n.Row > 0 && n.NodeType != RunConstants.NodeBoss
            )
        )
        {
            int reach = Math.Min(3, Math.Min(Math.Abs(node.Row - chestRow), bossRow - node.Row));
            Assert.True(
                Math.Abs(node.Col - ChestCol) <= reach,
                $"({node.Col},{node.Row}) is outside the hourglass: reach {reach}"
            );
            Assert.All(node.Children, c => Assert.True(Math.Abs(c.Col - node.Col) <= 1));
            Assert.True(node.Children.Count <= 3 && node.Parents.Count <= 3);
        }
    }

    [Theory]
    [MemberData(nameof(Seeds))]
    public void EveryPathFromTheAncientToTheBossCrossesTheChest(string seed)
    {
        var state = ActTwo(seed).State;
        var chest = (ChestCol, ChestRow(state));
        int bossRow = RunConstants.MapBossRowFor(state.Act);

        int paths = 0;
        void Walk((int Col, int Row) at, bool crossed)
        {
            crossed |= at == chest;
            if (at.Row == bossRow)
            {
                Assert.True(crossed, "a path reached the boss without opening the chest");
                paths++;
                return;
            }

            var children = state.MapNodes[at].Children;
            Assert.NotEmpty(children);
            foreach (var child in children)
            {
                Walk(child, crossed);
            }
        }

        Walk((RunConstants.MapStartCol, 0), crossed: false);
        Assert.True(paths > 0);
    }

    [Theory]
    [MemberData(nameof(Seeds))]
    public void ThePointTypesFollowTheStandardRules(string seed)
    {
        var state = ActTwo(seed).State;
        int bossRow = RunConstants.MapBossRowFor(state.Act);

        Assert.All(
            state.MapNodes.Values.Where(n => n.Row == 1),
            n => Assert.Equal(RunConstants.NodeNormal, n.NodeType)
        );
        Assert.All(
            state.MapNodes.Values.Where(n => n.Row == bossRow - 1),
            n => Assert.Equal(RunConstants.NodeRest, n.NodeType)
        );
        Assert.All(
            state.MapNodes.Values.Where(n => n.NodeType == RunConstants.NodeElite),
            n => Assert.True(n.Row >= 6)
        );
        Assert.DoesNotContain(
            state.MapNodes.Values,
            n => n.NodeType == RunConstants.NodeNone && n.Row > 0 && n.Row < bossRow
        );
    }

    [Fact]
    public void WithoutTheCardActTwoKeepsItsStandardMap()
    {
        var withCard = ActTwo("SPOILS0000").State;
        var without = ActTwo("SPOILS0000", withCard: false).State;

        Assert.Null(without.SpoilsQuestCoord);
        // The standard map has a chest in every column it reaches; it is not the hourglass.
        Assert.True(
            without.MapNodes.Values.Count(n => n.Row == ChestRow(without)) > 1,
            "the standard map's chest row is one node wide"
        );
        Assert.NotEqual(
            withCard.MapNodes.Values.Select(n => (n.Col, n.Row, n.NodeType)).Order().ToList(),
            without.MapNodes.Values.Select(n => (n.Col, n.Row, n.NodeType)).Order().ToList()
        );
    }

    /// <summary>
    /// <c>SpoilsMap.SpoilsActIndex</c> is 1: the hourglass is act 2's, and the card that
    /// is still in the deck in act 3 does nothing to that map.
    /// </summary>
    [Fact]
    public void ActThreeIsStandardWhateverTheDeckHolds()
    {
        var engine = ActTwo("SPOILS0000");
        Assert.NotNull(engine.State.SpoilsQuestCoord);

        Assert.True(engine.EnterNextAct());

        var state = engine.State;
        Assert.True(HoldsTheCard(state));
        Assert.Null(state.SpoilsQuestCoord);
        Assert.True(state.MapNodes.Values.Count(n => n.NodeType == RunConstants.NodeRelic) > 1);
    }

    [Fact]
    public void ActOneIsStandardWhateverTheDeckHolds()
    {
        var engine = new RunEngine();
        engine.Reset("SPOILS0000", 10);
        engine.State.Deck.Add(new CardInstance(RunConstants.SpoilsMapCard, Upgraded: false));

        RunMapGenerator.GenerateActMap(engine.State);

        Assert.Null(engine.State.SpoilsQuestCoord);
        Assert.True(
            engine.State.MapNodes.Values.Count(n => n.Row == RunConstants.MapTreasureRow) > 1
        );
    }

    /// <summary>
    /// The marker is put on the map AS IT IS MADE. A card picked up once act 2's map
    /// exists changes nothing: no hook regenerates the map, so there is no quest on it.
    /// </summary>
    [Fact]
    public void ACardPickedUpAfterTheMapIsMadeMarksNothing()
    {
        var engine = ActTwo("SPOILS0000", withCard: false);
        engine.State.Deck.Add(new CardInstance(RunConstants.SpoilsMapCard, Upgraded: false));

        Assert.Null(engine.State.SpoilsQuestCoord);
    }

    [Fact]
    public void TheQuestMarkerSitsOnTheChest()
    {
        var state = ActTwo("SPOILS0000").State;

        Assert.Equal((ChestCol, ChestRow(state)), state.SpoilsQuestCoord);
    }

    /// <summary>
    /// A Golden Compass is a relic and a relic is a listener after the deck's cards, so
    /// its path replaces the hourglass the card made. The card's <c>Late</c> hook still
    /// runs on the map that is left, and marks that map's first Treasure point.
    /// </summary>
    [Fact]
    public void AGoldenPathReplacesTheHourglassAndTheMarkerMovesOntoIt()
    {
        var engine = ActTwo("SPOILS0000");
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicGoldenCompass));
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicGoldenCompass);

        var state = engine.State;
        Assert.All(state.MapNodes.Values, n => Assert.Equal(RunConstants.MapStartCol, n.Col));
        var firstChest = state
            .MapNodes.Values.Where(n => n.NodeType == RunConstants.NodeRelic)
            .OrderBy(n => n.Row)
            .First();
        Assert.Equal((firstChest.Col, firstChest.Row), state.SpoilsQuestCoord);
    }

    [Fact]
    public void ResettingTheRunClearsTheMarker()
    {
        var engine = ActTwo("SPOILS0000");
        Assert.NotNull(engine.State.SpoilsQuestCoord);

        engine.Reset("SPOILS0000", 10);

        Assert.Null(engine.State.SpoilsQuestCoord);
    }

    [Fact]
    public void ACloneKeepsTheMarker()
    {
        var engine = ActTwo("SPOILS0000");

        var copy = engine.State.Clone();

        Assert.Equal(engine.State.SpoilsQuestCoord, copy.SpoilsQuestCoord);
        Assert.NotNull(copy.SpoilsQuestCoord);
    }
}
