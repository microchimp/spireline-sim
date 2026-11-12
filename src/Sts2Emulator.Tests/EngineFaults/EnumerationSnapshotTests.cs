// Engine fault class 3 (docs/ENGINE_FAULTS.md): "Collection was modified; enumeration operation may not execute" at
// CardEffects.ApplyGeneratedCardApproximation <- ApplyDefectCard (22 traces in the judge corpus's worker logs, every one a
// Chill handed to an Ironclad). Chill channelled one Frost per living enemy by walking the live enemy list; a channel into
// full slots evokes the oldest orb, a Lightning's evoke can kill, and under core2 the kill takes the corpse out of that list
// at once. The game walks a copy of its hittable enemies taken as the card is played; so does the copy now, and Tesla Coil
// walks a copy of its Lightning orbs. The sweep plays every card into the same hazard and holds that no effect throws.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Tests.Solver;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.EngineFaults;

public class EnumerationSnapshotTests
{
    private const int Chill = 86;

    /// <summary>
    /// <paramref name="hand" /> against three one-HP enemies and a big one, four orb slots full of Lightning (so every channel
    /// evokes one, and every evoke can kill), a draw and a discard pile; bound to a core2 kernel when asked.
    /// </summary>
    private static (CombatState State, CountingRandom Rng, Sts2Emulator.Core2.Kernel? Kernel) Hazard(bool core2, params CardInstance[] hand)
    {
        var f = Fight.Hand(hand)
            .Energy(10)
            .Draw([.. Pile(IC.StrikeIronclad, IC.DefendIronclad, IC.Anger, IC.ShrugItOff, IC.PommelStrike, IC.Inflame, IC.Havoc, IC.TrueGrit)])
            .Discard([.. Pile(IC.Bash, IC.IronWave, IC.Headbutt)])
            .Enemy(hp: 1).Enemy(hp: 1).Enemy(hp: 1).Enemy(hp: 400);
        f.State.OrbCapacity = 4;
        f.State.Orbs = [new OrbState(OrbType.Lightning), new OrbState(OrbType.Lightning), new OrbState(OrbType.Lightning), new OrbState(OrbType.Lightning)];
        var kernel = core2 ? SolverPositions.KernelFor("core2") : null;
        if (kernel is not null)
        {
            CardUids.NumberNew(f.State);
            kernel.Bind(f.State);
        }

        return (f.State, new CountingRandom(3), kernel);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ChillChannelsAFrostForEachEnemyHittableAsItIsPlayed(bool core2)
    {
        var (s, rng, kernel) = Hazard(core2, Card(Chill), Card(IC.DefendIronclad));

        KernelGate.Step(kernel, s, rng, 0);

        // Four hittable enemies when Chill was played: four Frosts, each evoking a Lightning, though the evokes killed.
        Assert.True(s.Enemies.Count(EnemyAI.IsTargetable) < 4, "no evoke killed: the hazard did not happen");
        Assert.Equal([OrbType.Frost, OrbType.Frost, OrbType.Frost, OrbType.Frost], s.Orbs.Select(o => o.Type));
    }

    [Fact]
    public void NoCardsEffectThrowsWhenItsHitsKillUnderCore2()
    {
        var failed = new List<string>();
        foreach (var def in GeneratedData.Cards.All.ToArray())
        {
            if (def.Type is CardType.Status or CardType.Curse)
            {
                continue;
            }

            foreach (bool upgraded in new[] { false, true })
            {
                var (s, rng, kernel) = Hazard(true, Card(def.Id, upgraded), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash), Card(IC.Anger));
                var thrown = Record.Exception(() =>
                {
                    KernelGate.Step(kernel, s, rng, 0);
                    for (int i = 0; i < 12 && s.PendingSelection is { } screen; i++)
                    {
                        KernelGate.Step(kernel, s, rng, 0 < screen.Candidates.Count ? 0 : screen.Candidates.Count);
                    }
                });
                if (thrown is not null)
                {
                    failed.Add($"{def.Name}{(upgraded ? "+" : "")}: {thrown.GetType().Name}: {thrown.Message}");
                }
            }
        }

        Assert.Empty(failed);
    }
}
