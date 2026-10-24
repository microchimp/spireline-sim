using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Xunit;
using E = Sts2Emulator.Core.CombatFactory.ActOneEncounter;

namespace Sts2Emulator.Tests;

/// <summary>
/// A weak encounter and its normal twin are two encounters, so they are two ids.
/// </summary>
/// <remarks>
/// <para>
/// The game decides weak or normal by the slot an encounter holds in the act's list: the
/// first <c>NumberOfWeakEncounters</c> come from the weak pool, the rest from the normal
/// one, and <c>RoomSet.NextNormalEncounter</c> hands them out in order. Nothing counts
/// fights. Corpse Slugs and Seapunk were the two pairs the emulator folded into one id
/// each, and picked the roster from a count of the rooms already fought -- normal fights
/// PLUS elites. So a weak Corpse Slugs or Seapunk that came after an elite (the third
/// fight of an act with an elite before it already counted as the fourth) arrived with
/// the normal roster. 190 of the Spire Codex replays stopped on exactly that.
/// </para>
/// <para>
/// The pairs, from the act pools of the decompiled game: Nibbits, Slimes (Overgrowth);
/// Corpse Slugs, Seapunk (Underdocks); Bowlbugs, Exoskeletons (Hive); Scrolls of Biting
/// (Glory). Tunneler has a Normal model too, but no act deals it.
/// </para>
/// </remarks>
public class WeakAndNormalEncounterTests
{
    /// <summary>
    /// Each pair as (weak id, normal id, monsters in the weak fight, monsters in the
    /// normal one), the counts read off each model's <c>GenerateMonsters</c>.
    /// </summary>
    public static IEnumerable<object[]> Pairs()
    {
        (E, E, int, int)[] pairs =
        [
            (E.NibbitsWeak, E.NibbitsNormal, 1, 2),
            (E.SlimesWeak, E.SlimesNormal, 3, 4),
            (E.CorpseSlugs, E.CorpseSlugsNormal, 2, 3),
            (E.Seapunk, E.SeapunkNormal, 1, 2),
            (E.BowlbugsWeak, E.Bowlbugs, 2, 3),
            (E.Exoskeletons, E.ExoskeletonsNormal, 3, 4),
            (E.ScrollsWeak, E.Scrolls, 3, 4),
            (E.Tunneler, E.TunnelerAndChomper, 1, 2),
        ];
        foreach (var (weak, normal, weakMonsters, normalMonsters) in pairs)
        {
            yield return [(int)weak, (int)normal, weakMonsters, normalMonsters];
        }
    }

    private static int Monsters(int encounterId, int seed = 0) =>
        Fight.Encounter((E)encounterId, seed: seed).State.Enemies.Count;

    [Theory]
    [MemberData(nameof(Pairs))]
    public void EachVariantBuildsItsOwnRosterWhateverElseIsGoingOn(
        int weak,
        int normal,
        int weakMonsters,
        int normalMonsters
    )
    {
        Assert.NotEqual(weak, normal);
        // Built from the id alone, whatever the seed: the roster is not rolled between the
        // two, which is what a flag read off the floor made it.
        for (int seed = 0; seed < 4; seed++)
        {
            Assert.Equal(weakMonsters, Monsters(weak, seed));
            Assert.Equal(normalMonsters, Monsters(normal, seed));
        }
    }

    /// <summary>
    /// A slot is weak or normal, never both: no act's weak pool shares an id with its
    /// normal pool. The shared ids were exactly the two pairs that went wrong.
    /// </summary>
    [Theory]
    [InlineData(RunConstants.ActOvergrowth)]
    [InlineData(RunConstants.ActUnderdocks)]
    [InlineData(RunConstants.ActHive)]
    [InlineData(RunConstants.ActGlory)]
    public void NoActDealsTheSameIdAsAWeakAndAsANormalEncounter(int act)
    {
        var weak = GeneratedData.EncounterTags.Pool(act, "Weak");
        var normal = GeneratedData.EncounterTags.Pool(act, "Normal");

        Assert.Empty(weak.Intersect(normal));
    }

    [Fact]
    public void UnderdocksNormalPoolDealsTheNormalSlugsAndSeapunk()
    {
        var normal = GeneratedData.EncounterTags.Pool(RunConstants.ActUnderdocks, "Normal");
        var weak = GeneratedData.EncounterTags.Pool(RunConstants.ActUnderdocks, "Weak");

        Assert.Contains(RunConstants.CorpseSlugsNormalEncounterId, normal);
        Assert.Contains(RunConstants.SeapunkNormalEncounterId, normal);
        Assert.Contains(RunConstants.CorpseSlugsEncounterId, weak);
        Assert.Contains(RunConstants.SeapunkEncounterId, weak);
        Assert.Equal(normal, RunConstants.UnderdocksNormalEncounters.ToArray());
    }

    // --- in a run ------------------------------------------------------------------

    /// <summary>
    /// A run on the map with the act's fight list set to <paramref name="sequence" />,
    /// <paramref name="normalsFought" /> normal fights and <paramref name="elitesFought" />
    /// elites behind it, about to step onto a Monster node.
    /// </summary>
    private static RunEngine AboutToFight(
        int[] sequence,
        int normalsFought,
        int elitesFought,
        string seed = "WEAKNRM001"
    )
    {
        var engine = TestSwitches.Run(new RunEngine());
        engine.Reset(seed, Ascension.DefaultLevel);
        var state = engine.State;
        state.NormalEncounterSequence = sequence;
        state.NormalEncountersVisited = normalsFought;
        state.EliteEncountersVisited = elitesFought;
        state.AwaitingActStartNode = false;
        state.CurrentMapCoord = (RunConstants.MapStartCol, 0);
        state.Phase = RunPhase.Map;
        RunMapGenerator.RefreshMapOptions(state);
        return engine;
    }

    private static CombatState StepIntoTheFight(RunEngine engine)
    {
        int action = Array.IndexOf(engine.State.MapNodeTypes, RunConstants.NodeNormal);
        Assert.True(action >= 0, "the first row offers no Monster node");
        Assert.Equal(0, TestSwitches.Step(engine, action, -1, out _, out _, out _));
        Assert.Equal(RunPhase.Combat, engine.State.Phase);
        return engine.State.ActiveCombat!;
    }

    /// <summary>
    /// The third fight of the act is the last weak one, whatever the player met on the way
    /// -- here an elite from a "?" room, which put the old count at four.
    /// </summary>
    [Theory]
    [InlineData(RunConstants.CorpseSlugsEncounterId, 2)]
    [InlineData(RunConstants.SeapunkEncounterId, 1)]
    public void AWeakFightAfterAnEliteIsStillTheWeakEncounter(int weakId, int monsters)
    {
        var engine = AboutToFight([10, 10, weakId, 10], normalsFought: 2, elitesFought: 1);

        var combat = StepIntoTheFight(engine);

        Assert.Equal(weakId, combat.EncounterId);
        Assert.Equal(monsters, combat.Enemies.Count);
    }

    /// <summary>
    /// And the roster follows the id the other way: a normal slot is normal even where a
    /// count of fights would have called it weak.
    /// </summary>
    [Theory]
    [InlineData(RunConstants.CorpseSlugsNormalEncounterId, 3)]
    [InlineData(RunConstants.SeapunkNormalEncounterId, 2)]
    public void ANormalFightIsTheNormalEncounterWhateverIsBehindIt(int normalId, int monsters)
    {
        var engine = AboutToFight([normalId, 10, 10, 10], normalsFought: 0, elitesFought: 0);

        var combat = StepIntoTheFight(engine);

        Assert.Equal(normalId, combat.EncounterId);
        Assert.Equal(monsters, combat.Enemies.Count);
    }

    /// <summary>
    /// The slugs roll their opening moves off the encounter's own stream, whose seed
    /// carries the model's entry id (CORPSE_SLUGS_WEAK or CORPSE_SLUGS_NORMAL). It was
    /// chosen by the same floor-count flag, so a normal fight could draw from the weak
    /// entry's stream. Here the run's fight must open the way the NORMAL entry's stream
    /// says, on a floor where the two streams disagree.
    /// </summary>
    [Fact]
    public void TheSlugsOpenOnTheStreamOfTheirOwnEncounter()
    {
        var engine = AboutToFight(
            [RunConstants.CorpseSlugsNormalEncounterId, 10, 10, 10],
            normalsFought: 0,
            elitesFought: 0
        );
        int runSeed = (int)engine.State.Rng.Seed;
        int StreamFor(int floor, int encounterId) =>
            EncounterRng.SeedFor(runSeed, floor, encounterId)!.Value;
        var normalOpenings = (int floor) =>
            Openings(StreamFor(floor, RunConstants.CorpseSlugsNormalEncounterId));
        var weakOpenings = (int floor) =>
            Openings(StreamFor(floor, RunConstants.CorpseSlugsEncounterId));
        int floor = Enumerable
            .Range(2, 60)
            .First(f => !normalOpenings(f).SequenceEqual(weakOpenings(f)));
        // The stream is keyed on the floor the fight is entered on: one more than here.
        engine.State.Floor = floor - 1;

        var combat = StepIntoTheFight(engine);

        Assert.Equal(
            normalOpenings(floor),
            combat.Enemies.Select(slug => (slug.CurrentIntent.Type, slug.CurrentIntent.Magnitude))
        );
    }

    /// <summary>The slugs' opening intents with the encounter stream seeded as given.</summary>
    private static IEnumerable<(IntentType, int)> Openings(int encounterSeed) =>
        Fight
            .EncounterWithStream((int)E.CorpseSlugsNormal, encounterSeed)
            .State.Enemies.Select(slug => (slug.CurrentIntent.Type, slug.CurrentIntent.Magnitude));

    // --- the native boundary ---------------------------------------------------------

    private const int Slugs = RunConstants.CorpseSlugsEncounterId;
    private const int SlugsNormal = RunConstants.CorpseSlugsNormalEncounterId;
    private const int Seapunk = RunConstants.SeapunkEncounterId;
    private const int SeapunkNormal = RunConstants.SeapunkNormalEncounterId;

    /// <summary>
    /// Callers across the C ABI still name Corpse Slugs and Seapunk by the one id they
    /// shared and say which fight with the weak context ([0,3) weak, -1 normal). The edge
    /// reads that once; a caller that names the normal id gets it whatever it says.
    /// </summary>
    [Theory]
    [InlineData(Slugs, 0, Slugs)]
    [InlineData(Slugs, 2, Slugs)]
    [InlineData(Slugs, 3, SlugsNormal)]
    [InlineData(Slugs, -1, SlugsNormal)]
    [InlineData(Seapunk, 1, Seapunk)]
    [InlineData(Seapunk, -1, SeapunkNormal)]
    [InlineData(SlugsNormal, 0, SlugsNormal)]
    [InlineData(SeapunkNormal, 1, SeapunkNormal)]
    [InlineData(RunConstants.SlimesWeakEncounterId, -1, RunConstants.SlimesWeakEncounterId)]
    public void TheWeakContextIsReadOnceAtTheNativeEdge(int encounterId, int context, int expected)
    {
        Assert.Equal(expected, WeakContext.Resolve(encounterId, context));
    }
}

/// <summary>
/// <c>CorpseSlugsNormal</c>: three Corpse Slugs, the encounter the emulator could only
/// build by being told it was not weak.
/// </summary>
public class CorpseSlugsNormalTests
{
    private static Fight At(int ascension = 10, int seed = 0) =>
        Fight.Encounter(E.CorpseSlugsNormal, ascension, seed);

    [Fact]
    public void ItIsThreeCorpseSlugs()
    {
        Assert.Equal([KE.CorpseSlug, KE.CorpseSlug, KE.CorpseSlug], At().EnemyDefIds);
    }

    [Fact]
    public void TheWeakEncounterIsTwo()
    {
        var weak = Fight.Encounter(E.CorpseSlugs);

        Assert.Equal([KE.CorpseSlug, KE.CorpseSlug], weak.EnemyDefIds);
    }

    /// <summary>
    /// <c>EnsureCorpseSlugsStartWithDifferentMoves</c> deals consecutive openings from one
    /// roll, so three slugs open on three different moves.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void TheThreeOpenOnThreeDifferentMoves(int streamSeed)
    {
        var openings = Fight
            .EncounterWithStream((int)E.CorpseSlugsNormal, streamSeed)
            .State.Enemies.Select(slug => (slug.CurrentIntent.Type, slug.CurrentIntent.Magnitude));

        Assert.Equal(3, openings.Distinct().Count());
    }

    /// <summary>Ravenous is the slug's own power, 5 at A9 and up and 4 below.</summary>
    [Theory]
    [InlineData(8, 4)]
    [InlineData(10, 5)]
    public void EachSlugIsRavenousForTheAscensionsAmount(int ascension, int amount)
    {
        Assert.All(
            At(ascension).State.Enemies,
            slug => Assert.Equal(amount, BuffSystem.Get(slug.Buffs, BuffId.Ravenous))
        );
    }
}
