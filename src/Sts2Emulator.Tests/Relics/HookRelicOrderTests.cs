using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Relics a reward hook ADDS (Lava Rock's two, Black Star's, Wongo's ticket's) are populated
/// after the room's own rewards: GenerateWithoutOffering fills the base list, runs
/// Hook.ModifyRewards, then fills what the hooks added. So the card offer is the same with or
/// without the hook (run referee, seed E2KJGJ1SMR: Lava Rock's act-1 boss offered other cards).
/// </summary>
public class HookRelicOrderTests
{
    private static RunEngine AtBossRewards(bool lavaRock)
    {
        var engine = new RunEngine();
        engine.Reset("LAVAROCK01");
        var s = engine.State;
        s.CurrentNodeType = RunConstants.NodeBoss;
        s.LastResolvedRoomType = RunConstants.NodeBoss;
        if (lavaRock) s.Relics.Add(new RelicInstance(RunConstants.RelicLavaRock));
        RunRewardGenerator.GenerateCombatRewards(s);
        return engine;
    }

    [Fact]
    public void LavaRocksRelicsDoNotMoveTheCards()
    {
        var plain = AtBossRewards(lavaRock: false);
        var rock = AtBossRewards(lavaRock: true);
        Assert.Equal(plain.State.RewardCards, rock.State.RewardCards);
        Assert.NotEqual(0, rock.State.RelicReward);
        Assert.Single(rock.State.PendingBonusRelicRewards);
    }
}
