using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The engine side of the referee's setup fights (referee/driver/Setup.cs): the developers'
/// console <c>fight</c> entry and <c>power</c> with no applier, built from paths the engine
/// already has.
/// </summary>
public class ConsoleSetupTests
{
    private const string Seed = "84G7XUWQD6";

    /// <summary>A fresh run past Neow, standing on the map.</summary>
    private static RunEngine AtMap(string seed = Seed)
    {
        var e = new RunEngine();
        e.Reset(seed, 10);
        for (int guard = 0; guard < 10 && e.State.Phase != RunPhase.Map; guard++)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Map, e.State.Phase);
        return e;
    }

    private static int FirstOption(RunEngine e, int nodeType) =>
        Enumerable.Range(0, RunConstants.MapChoices).First(i => e.State.MapNodeTypes[i] == nodeType);

    [Fact]
    public void TheMapsOwnEncounterThroughTheDebugEntryIsThePlainMapStep()
    {
        var plain = AtMap();
        var debug = AtMap();
        int option = FirstOption(plain, RunConstants.NodeNormal);
        int encounter = plain.State.MapChoices[option];

        Assert.Equal(0, plain.Step(option, -1, out _, out _, out _));
        // The first normal room of an act deals from the weak pool.
        Assert.Equal(0, debug.EnterEncounterDebug(option, encounter, RunConstants.NodeNormal));

        CombatState a = plain.State.ActiveCombat!, b = debug.State.ActiveCombat!;
        Assert.Equal(a.Enemies.Select(x => (x.DefId, x.Hp, x.MaxHp)), b.Enemies.Select(x => (x.DefId, x.Hp, x.MaxHp)));
        Assert.Equal(a.Hand.Select(c => c.DefId), b.Hand.Select(c => c.DefId));
        Assert.Equal(a.DrawPile.Select(c => c.DefId), b.DrawPile.Select(c => c.DefId));
        Assert.Equal(plain.State.Floor, debug.State.Floor);
        Assert.Equal(plain.State.NormalEncountersVisited, debug.State.NormalEncountersVisited);
        Assert.Equal(plain.State.Rng.Shuffle.CallCount, debug.State.Rng.Shuffle.CallCount);
        Assert.Equal(plain.State.Rng.Niche.CallCount, debug.State.Rng.Niche.CallCount);
    }

    [Fact]
    public void AnEliteEnteredFromAMonsterPointIsAnEliteRoom()
    {
        var e = AtMap();
        int option = FirstOption(e, RunConstants.NodeNormal);

        Assert.Equal(0, e.EnterEncounterDebug(option, (int)CombatFactory.ActOneEncounter.Byrdonis, RunConstants.NodeElite));

        Assert.Equal(RunPhase.Combat, e.State.Phase);
        Assert.Equal(RunConstants.NodeElite, e.State.LastResolvedRoomType);
        Assert.Equal(1, e.State.EliteEncountersVisited);
        Assert.Equal(0, e.State.NormalEncountersVisited);
        Assert.True(e.State.ActiveCombat!.IsEliteRoom);
    }

    [Theory]
    [InlineData(RunConstants.CorpseSlugsEncounterId, 2)]
    [InlineData(RunConstants.CorpseSlugsNormalEncounterId, 3)]
    [InlineData(RunConstants.SeapunkEncounterId, 1)]
    [InlineData(RunConstants.SeapunkNormalEncounterId, 2)]
    public void TheIdAloneDecidesTheRoster(int encounterId, int monsters)
    {
        // The setup fight names the encounter by its place in the game's act list, and a weak
        // fight and its normal twin are two places and two ids. Rooms already fought (here
        // more than any act's weak share, elites among them) are not consulted: the debug
        // entry used to carry a weak flag of its own because the id could not say.
        var e = AtMap();
        e.State.NormalEncountersVisited = 5;
        e.State.EliteEncountersVisited = 2;
        int option = FirstOption(e, RunConstants.NodeNormal);

        Assert.Equal(0, e.EnterEncounterDebug(option, encounterId, RunConstants.NodeNormal));

        Assert.Equal(encounterId, e.State.ActiveCombat!.EncounterId);
        Assert.Equal(monsters, e.State.ActiveCombat!.Enemies.Count);
    }

    [Fact]
    public void OnlyACombatRoomOnTheMapIsAccepted()
    {
        var e = AtMap();
        int option = FirstOption(e, RunConstants.NodeNormal);

        Assert.Equal(-1, e.EnterEncounterDebug(option, 1, RunConstants.NodeRest));
        Assert.Equal(RunPhase.Map, e.State.Phase);

        e.EnterEncounterDebug(option, 1, RunConstants.NodeNormal);
        Assert.Equal(-1, e.EnterEncounterDebug(option, 1, RunConstants.NodeNormal));
    }

    [Fact]
    public void ArtifactEatsADebuffWithNoApplier()
    {
        var fight = Fight.Hand().Enemy(hp: 100, buffs: new BuffState(BuffId.Artifact, 1));

        Assert.True(ConsoleSetup.ApplyPowerWithoutApplier(fight.State, 0, BuffId.Vulnerable, 2));

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Vulnerable));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Artifact));
    }

    [Fact]
    public void AStrengthLossIsADebuffToo()
    {
        var guarded = Fight.Hand().Enemy(hp: 100, buffs: new BuffState(BuffId.Artifact, 2));
        var bare = Fight.Hand().Enemy(hp: 100);

        ConsoleSetup.ApplyPowerWithoutApplier(guarded.State, 0, BuffId.Strength, -2);
        ConsoleSetup.ApplyPowerWithoutApplier(bare.State, 0, BuffId.Strength, -2);

        Assert.Equal(0, guarded.EnemyBuffAmount(BuffId.Strength));
        Assert.Equal(1, guarded.EnemyBuffAmount(BuffId.Artifact));
        Assert.Equal(-2, bare.EnemyBuffAmount(BuffId.Strength));
    }

    [Fact]
    public void RuinedHelmetDoublesOnlyThePlayersFirstStrength()
    {
        var fight = Fight.Hand().Enemy(hp: 100);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.RuinedHelmet));

        ConsoleSetup.ApplyPowerWithoutApplier(fight.State, -1, BuffId.Strength, 3);
        Assert.Equal(6, fight.PlayerBuffAmount(BuffId.Strength));

        ConsoleSetup.ApplyPowerWithoutApplier(fight.State, -1, BuffId.Strength, 3);
        Assert.Equal(9, fight.PlayerBuffAmount(BuffId.Strength));
    }

    [Fact]
    public void ADeadOrMissingEnemyTakesNothing()
    {
        var fight = Fight.Hand().Enemy(hp: 100);
        fight.Enemy0.Hp = 0;

        Assert.False(ConsoleSetup.ApplyPowerWithoutApplier(fight.State, 0, BuffId.Weak, 1));
        Assert.False(ConsoleSetup.ApplyPowerWithoutApplier(fight.State, 3, BuffId.Weak, 1));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Weak));
    }
}
