// What EmuDump writes for the step that ends a fight (Fight.Dump, Fight.Final). The comparer used to look at
// nothing but "in_combat" there, because the dump said nothing else; a fight the game left at 76 hit
// points and the emulator at 68 was scored identical. The dump now carries what the run holds after a
// won fight (hit points with the after-combat heals in, gold, RNG positions and, for the records that
// have them, the belt, the relics and their counters) and the hit points of a lost one.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

public class EndOfFightDumpTests
{
    static RunEngine InAFight()
    {
        var e = new RunEngine();
        e.Reset("84G7XUWQD6", 10);
        for (int guard = 0; guard < 10 && e.State.Phase != RunPhase.Map; guard++)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        int option = Enumerable.Range(0, RunConstants.MapChoices).First(i => e.State.MapNodeTypes[i] == RunConstants.NodeNormal);
        e.Step(option, -1, out _, out _, out _);
        Assert.Equal(RunPhase.Combat, e.State.Phase);
        return e;
    }

    [Fact]
    public void AFightInProgressWritesNoFinalState()
    {
        Setup.Extended = false;
        var d = Fight.Dump(InAFight());

        Assert.True((bool)d["in_combat"]!);
        Assert.Null(d["final"]);
        Assert.NotNull(d["snapshot"]);
    }

    [Fact]
    public void AWonFightWritesWhatTheRunHoldsWithTheAfterCombatHealIn()
    {
        Setup.Extended = true;
        var e = InAFight();
        e.State.ActiveCombat!.PlayerHp = 50;  // Burning Blood heals six of the thirty missing
        Assert.NotEqual(-1, e.DebugResolveCombat());

        var d = Fight.Dump(e);

        Assert.False((bool)d["in_combat"]!);
        var final = d["final"]!.AsObject();
        Assert.True((bool)final["won"]!);
        Assert.Equal(56, (int)final["player"]!["hp"]!);
        Assert.Equal(e.State.PlayerMaxHp, (int)final["player"]!["max_hp"]!);
        Assert.Equal(e.State.Gold, (int)final["player"]!["gold"]!);
        Assert.Equal(e.State.Rng.Shuffle.CallCount, (int)final["rng"]!["Shuffle"]!);
        Assert.Contains("BURNING_BLOOD", final["relics"]!.AsArray().Select(r => (string?)r));
        Assert.Equal(final["relics"]!.AsArray().Count, final["relic_counters"]!.AsArray().Count);
        Assert.Equal(e.State.MaxPotionSlots, final["potions"]!.AsArray().Count);
    }

    [Fact]
    public void ARecordWithoutTheSetupExtrasGetsNoBeltAndNoRelics()
    {
        Setup.Extended = false;
        var e = InAFight();
        e.DebugResolveCombat();

        var final = Fight.Dump(e)["final"]!.AsObject();

        Assert.Null(final["potions"]);
        Assert.Null(final["relics"]);
        Assert.NotNull(final["player"]!["hp"]);
    }

    [Fact]
    public void ALostFightWritesThatAndNothingTheRunKeeps()
    {
        Setup.Extended = true;
        var e = InAFight();
        e.State.ActiveCombat!.PlayerHp = -4;  // an overkill: the game stops at 0
        e.State.Phase = RunPhase.Map;  // the fight is over (the engine's own game-over phase is not the point)
        e.State.LastPlayerWon = false;

        var final = Fight.Dump(e)["final"]!.AsObject();

        Assert.False((bool)final["won"]!);
        Assert.Null(final["player"]);
        Assert.Null(final["rng"]);
    }
}
