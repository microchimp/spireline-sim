using Sts2Emulator.Core;
using Sts2Emulator.Tests.Invariants;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Each relation of tools/emudump's metamorphic runner, tried twice: on the engine, where it must hold, and
/// on a copy of the engine with one fault put in, where it must fail. A relation that passes both is
/// checking nothing.
/// </summary>
public class MetamorphicTests
{
    private const CombatFactory.ActOneEncounter Slimes = CombatFactory.ActOneEncounter.SlimesWeak;
    private const CombatFactory.ActOneEncounter Slugs = CombatFactory.ActOneEncounter.CorpseSlugs;

    private static Combat Open(CombatFactory.ActOneEncounter encounter, int seed) => TestFights.Open(encounter, seed);

    /// <summary>The engine with one fault: <paramref name="after"/> runs after each real step.</summary>
    private sealed class Faulty : Combat
    {
        private readonly Action<Faulty, Move> _after;
        private readonly Action<Faulty, Faulty>? _cloned;

        public Faulty(Combat from, Action<Faulty, Move> after, Action<Faulty, Faulty>? cloned = null)
            : base(from.State, from.Rng)
        {
            _after = after;
            _cloned = cloned;
        }

        /// <summary>What the player had just before the last step, for a fault that reads a change.</summary>
        public int BlockBefore { get; private set; }

        public int TurnBefore { get; private set; }

        public override StepResult Step(Move m)
        {
            BlockBefore = State.PlayerBlock;
            TurnBefore = State.Turn;
            var r = base.Step(m);
            _after(this, m);
            return r;
        }

        public override Combat Clone()
        {
            var real = base.Clone();
            var copy = new Faulty(real, _after, _cloned) { Over = Over, Won = Won };
            _cloned?.Invoke(this, copy);
            return copy;
        }
    }

    private static Func<Combat> Opening(CombatFactory.ActOneEncounter e, int seed, Action<Faulty, Move> after, Action<Faulty, Faulty>? cloned = null) =>
        () => new Faulty(Open(e, seed), after, cloned);

    private static List<Failure> Run(string relation, Func<Combat> open, int seeds = 12)
    {
        var all = new List<Failure>();
        for (int seed = 1; seed <= seeds; seed++)
        {
            var report = new RelationReport();
            Metamorphic.Run(open, seed, report, [relation]);
            all.AddRange(report.Failures);
        }

        return all;
    }

    private static void NoFault(Move m, Faulty c) { }

    // ---- the engine as it is ---------------------------------------------------------------------------

    public static IEnumerable<object[]> Fights() =>
        new[]
        {
            (Slimes, 1), (Slimes, 2), (Slugs, 3), (CombatFactory.ActOneEncounter.Cultists, 4),
            (CombatFactory.ActOneEncounter.NibbitsNormal, 5), (CombatFactory.ActOneEncounter.Toadpoles, 6),
            (CombatFactory.ActOneEncounter.FossilStalker, 7), (CombatFactory.ActOneEncounter.Mawler, 8),
        }.Select(f => new object[] { (int)f.Item1, f.Item2 });

    [Theory]
    [MemberData(nameof(Fights))]
    public void EveryRelationHoldsOnTheEngine(int encounterId, int seed)
    {
        var encounter = (CombatFactory.ActOneEncounter)encounterId;
        var all = new List<Failure>();
        for (int chooser = 0; chooser < 4; chooser++)
        {
            var report = new RelationReport();
            Metamorphic.Run(() => Open(encounter, seed), chooser, report);
            all.AddRange(report.Failures);
        }

        Assert.True(all.Count == 0, string.Join("\n", all.Select(f => $"{f.Relation} step {f.Step} {f.Card}: {f.Detail}")));
    }

    [Fact]
    public void TheRelationsAreAllCounted()
    {
        var report = new RelationReport();
        for (int chooser = 0; chooser < 6; chooser++)
        {
            Metamorphic.Run(() => Open(Slugs, 3), chooser, report);
        }

        foreach (string relation in Metamorphic.Names)
        {
            Assert.True(report.Instances.GetValueOrDefault(relation) > 0, $"{relation} never ran on a fight it could judge");
        }
    }

    [Fact]
    public void AnEngineThatThrowsIsAFailureNotACrash()
    {
        var report = new RelationReport();
        Metamorphic.Run(() => new Faulty(Open(Slimes, 1), (_, _) => throw new InvalidOperationException("boom")), 1, report, ["invariants"]);

        Assert.Contains(report.Failures, f => f.Relation == "exception" && f.Detail.Contains("boom"));
    }

    // ---- one fault each --------------------------------------------------------------------------------

    private static int _leak;

    [Fact]
    public void DeterminismSeesStateTwoOpeningsShare()
    {
        // A static counter folded into the state: two copies played in step see different values.
        var failures = Run("determinism", Opening(Slimes, 1, (c, _) => c.State.PlayerBlock += ++_leak % 5));

        Assert.Contains(failures, f => f.Relation == "determinism");
    }

    [Fact]
    public void CloneSeesACopyThatSharesAPile()
    {
        var failures = Run("clone", Opening(Slimes, 1, (c, m) => { }, (original, copy) => copy.State.Hand = original.State.Hand));

        Assert.Contains(failures, f => f.Relation == "clone");
    }

    [Fact]
    public void CloneSeesACopyThatForgetsAField()
    {
        var failures = Run("clone", Opening(Slimes, 1, (c, m) => { }, (_, copy) => copy.State.Enemies[0].LastMoveRepeats += 3));

        Assert.Contains(failures, f => f.Relation == "clone");
    }

    [Fact]
    public void PadSeesAnEffectThatReadsTheDrawPileSize()
    {
        var failures = Run("pad", Opening(Slimes, 1, (c, m) =>
        {
            if (c.State.DrawPile.Count % 2 == 0)
            {
                c.State.CardSelectionRng!.Next(); // rolls only on an even pile
            }
        }));

        Assert.Contains(failures, f => f.Relation == "pad");
    }

    [Fact]
    public void TwinsSeesAnEffectThatReadsTheHandIndex()
    {
        var failures = Run("twins", Opening(Slimes, 1, (c, m) => c.State.PlayerBlock += m.Action % 2));

        Assert.Contains(failures, f => f.Relation == "twins");
    }

    [Fact]
    public void SymmetrySeesAnEnemyThatIsHitHarderInTheSecondSlot()
    {
        var failures = Run("symmetry", Opening(Slugs, 3, (c, m) =>
        {
            if (m.Target == 1)
            {
                c.State.Enemies[1].Hp = Math.Max(1, c.State.Enemies[1].Hp - 1);
            }
        }));

        Assert.Contains(failures, f => f.Relation == "symmetry");
    }

    [Fact]
    public void StreamsSeesAReadOfAStreamThatIsNotCounted()
    {
        // The value is read off a copy, so the call count never moves, but the state depends on it.
        var failures = Run("streams", Opening(Slimes, 1, (c, m) => c.State.PlayerBlock += c.State.CardGenerationRng!.Copy().Next(4)));

        Assert.Contains(failures, f => f.Relation == "streams");
    }

    [Fact]
    public void HandOrderSeesAnEffectThatReadsThePlayedIndex()
    {
        var failures = Run("handorder", Opening(Slimes, 1, (c, m) => c.State.PlayerBlock += m.Action == 0 ? 1 : 0));

        Assert.Contains(failures, f => f.Relation == "handorder");
    }

    [Fact]
    public void DiscardOrderSeesAnEffectThatReadsTheTopOfTheDiscard()
    {
        var failures = Run("discardorder", Opening(Slimes, 1, (c, m) =>
        {
            if (c.State.DiscardPile.Count > 0)
            {
                c.State.PlayerBlock += c.State.DiscardPile[0].DefId % 2;
            }
        }), seeds: 30);

        Assert.Contains(failures, f => f.Relation == "discardorder");
    }

    [Fact]
    public void EnergySeesAnEffectThatReadsWhatIsLeft()
    {
        var failures = Run("energy", Opening(Slimes, 1, (c, m) =>
        {
            if (c.State.Energy >= 4)
            {
                c.State.PlayerBlock += 1;
            }
        }));

        Assert.Contains(failures, f => f.Relation == "energy");
    }

    [Fact]
    public void MonotoneSeesBlockThatCostsHitPoints()
    {
        var failures = Run("monotone", Opening(Slimes, 1, (c, m) =>
        {
            if (c.State.Turn != c.TurnBefore) // the enemies' turn just ran: the more block, the more it hurt
            {
                c.State.PlayerHp = Math.Max(1, c.State.PlayerHp - c.BlockBefore);
            }
        }));

        Assert.Contains(failures, f => f.Relation == "monotone");
    }

    [Fact]
    public void MonotoneSeesStrengthThatHealsEnemies()
    {
        var failures = Run("monotone", Opening(Slimes, 1, (c, m) =>
        {
            if (BuffSystem.Get(c.State.PlayerBuffs, BuffId.Strength) >= 3)
            {
                foreach (var e in c.State.Enemies)
                {
                    e.Hp = Math.Min(e.MaxHp, e.Hp + 20);
                }
            }
        }));

        Assert.Contains(failures, f => f.Relation == "monotone");
    }

    [Fact]
    public void InvariantsSeeAStepThatLeavesAMessBehind()
    {
        var failures = Run("invariants", Opening(Slimes, 1, (c, m) => c.State.AutoPlayQueue.Add(c.State.Hand.Count > 0 ? c.State.Hand[0] : default)));

        Assert.Contains(failures, f => f.Relation == "invariant:at_rest");
    }
}
