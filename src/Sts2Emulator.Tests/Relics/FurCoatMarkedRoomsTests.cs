using System.Collections.Generic;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Fur Coat marks the rooms the game marks. The coat's own stream is seeded from its ModelId
/// entry, FUR_COAT; seeded from the class name the copy marked seven other rooms, so a fight
/// the game opened at 1 HP opened at full HP here and the other way round (the nights of
/// 2026-11-03 to 11-05: the enemy-HP records at a fight's first decision, e.g. a Devoted
/// Sculptor at 1 against 172, a Mecha Knight at 320 against 1).
/// </summary>
/// <remarks>
/// The coordinates are the map choices the game's run took into those fights (the real
/// arm's log), the coat being the run's Act 3 Ancient pick; marked or not is what the game
/// showed at the fight's first decision. The map is the copy's own Act 3 map for the seed,
/// which the lockstep had matched choice by choice up to there.
/// </remarks>
public class FurCoatMarkedRoomsTests
{
    private static RunEngine ActThreeWithTheCoat(string seed)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        engine.State.CurrentActIndex = 2;
        RunMapGenerator.GenerateActMap(engine.State);
        engine.State.Relics.Add(new RelicInstance(RunConstants.RelicFurCoat));
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunConstants.RelicFurCoat);
        return engine;
    }

    [Theory]
    [InlineData("XH5XRL65FK", 6, 2, true)] // Devoted Sculptor: the game 1 HP, the copy 172
    [InlineData("XH5XRL65FK", 3, 10, false)] // Mecha Knight: the game 320, the copy 1
    [InlineData("XJ9TH8JPUZ", 6, 1, true)] // Scrolls of Biting
    [InlineData("XJ9TH8JPUZ", 5, 4, true)] // Living Shield and Turret Operator
    [InlineData("XJ9TH8JPUZ", 5, 6, true)] // Soul Nexus
    [InlineData("9CBCG8L86A", 6, 2, false)] // Devoted Sculptor: the game 172, the copy 1
    public void TheCoatMarksTheRoomsTheGameMarked(string seed, int col, int row, bool marked)
    {
        var engine = ActThreeWithTheCoat(seed);

        Assert.True(engine.State.MapNodes.ContainsKey((col, row)), $"no map point at ({col},{row})");
        Assert.Contains(
            engine.State.MapNodes[(col, row)].NodeType,
            new[] { RunConstants.NodeNormal, RunConstants.NodeElite }
        );
        Assert.Equal(RunConstants.FurCoatCombats, engine.State.FurCoatRooms.Count);
        Assert.Equal(marked, engine.State.FurCoatRooms.Contains((col, row)));
    }
}
