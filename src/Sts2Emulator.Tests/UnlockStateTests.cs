using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Tests.Invariants;
using Xunit;
using Encounter = Sts2Emulator.Core.CombatFactory.ActOneEncounter;
using Epochs = Sts2Emulator.Core.UnlockState.Epochs;
using static Sts2Emulator.Core.Run.RunConstants;

namespace Sts2Emulator.Tests;

/// <summary>
/// The profile a run is generated for (<see cref="UnlockState" />), read where the game reads its
/// <c>UnlockState</c>: Neow's <c>IsAllowedAtNeow</c> (and Neow's Bones, which asks the same),
/// <c>ActModel.GetRandomList</c>, <c>ActModel.GenerateRooms</c>' event epochs and
/// <c>GetUnlockedAncients</c>, <c>RunManager.GenerateRooms</c>' shared ancients,
/// <c>ApplyDiscoveryOrderModifications</c>' bosses, and every walk over the unlocked characters.
/// Each test also says what the profile does NOT move: which streams keep their draw counts.
/// </summary>
public class UnlockStateTests
{
    static readonly UnlockState NoDefect = UnlockState.All.WithLockedEpochs(Epochs.Defect1);

    static readonly UnlockState IroncladOnly = UnlockState.All.WithLockedEpochs(
        Epochs.Silent1,
        Epochs.Regent1,
        Epochs.Necrobinder1,
        Epochs.Defect1
    );

    static IEnumerable<string> Seeds(int count) =>
        Enumerable.Range(0, count).Select(i => $"UNLOCK{i:D4}");

    static RunState Run(string seed, UnlockState? unlocks = null, int ascension = 10)
    {
        var engine = new RunEngine();
        engine.Reset(seed, ascension, PlayerCharacter.Ironclad, unlocks);
        return engine.State;
    }

    static int Id(Encounter encounter) => (int)encounter;

    /// <summary>Two acts' rooms field by field (a record compares its arrays by reference).</summary>
    static void SameRooms(ActRooms expected, ActRooms actual) =>
        Assert.Empty(DeepState.Diff(expected, actual));

    [Fact]
    public void TheMatureProfileIsTheDefaultRun()
    {
        foreach (string seed in Seeds(5))
        {
            Assert.Empty(DeepState.Diff(Run(seed), Run(seed, UnlockState.All)));
        }

        Assert.True(UnlockState.All.IsAll);
        Assert.Equal(Enum.GetValues<PlayerCharacter>(), UnlockState.All.Characters);
    }

    /// <summary>
    /// `Kaleidoscope.IsAllowedAtNeow` wants every character unlocked, and Neow drops a refused
    /// blessing after the pairs are added and BEFORE the shuffle: one element fewer is a
    /// different shuffle, not the old two minus Kaleidoscope. The curse, drawn first, stays.
    /// </summary>
    [Fact]
    public void NeowDealsFromAListWithoutKaleidoscopeWhenACharacterIsLocked()
    {
        int offered = 0;
        foreach (string seed in Seeds(300))
        {
            var mature = Run(seed);
            var young = Run(seed, NoDefect);
            Assert.DoesNotContain(RunConstants.RelicKaleidoscope, young.NeowOptions);
            Assert.Equal(mature.NeowOptions[2], young.NeowOptions[2]);
            Assert.Equal(GameRule(young), young.NeowOptions);
            if (mature.NeowOptions.Contains(RunConstants.RelicKaleidoscope))
            {
                offered++;
                Assert.NotEqual(mature.NeowOptions, young.NeowOptions);
            }
        }

        Assert.True(offered > 0, "no seed offered Kaleidoscope on the mature profile");

        // `Neow.GenerateInitialOptions`, step by step, with Kaleidoscope refused.
        static int[] GameRule(RunState s)
        {
            var rng = s.Rng.NeowRng();
            int[] curses = RunConstants.NeowCurseOptions.ToArray();
            int cursed = curses[rng.NextInt(curses.Length)];
            var positive = RunConstants.NeowPositiveOptions.ToArray().ToList();
            positive.Remove(cursed switch
            {
                RunConstants.RelicCursedPearl => RunConstants.RelicGoldenPearl,
                RunConstants.RelicHeftyTablet => RunConstants.RelicArcaneScroll,
                RunConstants.RelicLeafyPoultice => RunConstants.RelicNewLeaf,
                RunConstants.RelicPrecariousShears => RunConstants.RelicPreciseScissors,
                _ => 0,
            });
            if (cursed != RunConstants.RelicLargeCapsule)
            {
                positive.Add(rng.NextBool() ? RelicLavaRock : RelicSmallCapsule);
            }

            positive.Add(rng.NextBool() ? RelicNutritiousOyster : RelicStoneHumidifier);
            positive.Add(rng.NextBool() ? RelicNeowsTalisman : RelicPomander);
            positive.Remove(RunConstants.RelicKaleidoscope);
            rng.Shuffle(positive);
            return [positive[0], positive[1], cursed];
        }
    }

    /// <summary>
    /// Neow's Bones offers two of `Neow.AllPossibleOptions` that pass `IsAllowedAtNeow`, shuffled
    /// on the Rewards stream: without Kaleidoscope the list is one shorter and the shuffle one
    /// draw cheaper, which every later reward reads.
    /// </summary>
    [Fact]
    public void NeowsBonesAsksNeowsRuleAndDrawsOneRewardFewer()
    {
        foreach (string seed in Seeds(40))
        {
            var mature = Run(seed);
            var young = Run(seed, NoDefect);
            RunNonCombatEffects.ApplyRelicPickup(mature, RunConstants.RelicNeowsBones);
            RunNonCombatEffects.ApplyRelicPickup(young, RunConstants.RelicNeowsBones);

            Assert.Equal(mature.PlayerRng.Rewards.CallCount - 1, young.PlayerRng.Rewards.CallCount);
            var offered = young.PendingBonusRelicRewards.Append(young.RelicReward);
            Assert.DoesNotContain(RunConstants.RelicKaleidoscope, offered);
        }
    }

    /// <summary>
    /// `ApplyDiscoveryOrderModifications`: the first boss of the act's BossDiscoveryOrder the
    /// profile has not fought replaces the one drawn. The draw is spent anyway, so UpFront ends
    /// where the mature profile's does -- and DoubleBoss's second boss leaves the FORCED one out.
    /// </summary>
    [Fact]
    public void AnUnseenBossReplacesTheDrawnOneAndDrawsNothing()
    {
        var kin = UnlockState.All.WithUnseenEncounters(Id(Encounter.TheKin));
        var vantomFirst = kin.WithUnseenEncounters(Id(Encounter.Vantom));
        var crab = UnlockState.All.WithUnseenEncounters(Id(Encounter.KaiserCrab));
        var queen = UnlockState.All.WithUnseenEncounters(Id(Encounter.Queen));
        int overgrowth = 0;
        foreach (string seed in Seeds(40))
        {
            var mature = Run(seed);
            var forcedKin = Run(seed, kin);
            Assert.Equal(mature.Rng.UpFront.CallCount, forcedKin.Rng.UpFront.CallCount);
            if (mature.Acts[0].Act == RunConstants.ActOvergrowth)
            {
                overgrowth++;
                Assert.Equal(Id(Encounter.TheKin), forcedKin.Acts[0].BossEncounterId);
                var vantom = Run(seed, vantomFirst).Acts[0];
                Assert.Equal(Id(Encounter.Vantom), vantom.BossEncounterId);
            }
            else
            {
                SameRooms(mature.Acts[0], forcedKin.Acts[0]);
            }

            Assert.Equal(Id(Encounter.KaiserCrab), Run(seed, crab).Acts[1].BossEncounterId);
            var glory = Run(seed, queen).Acts[2];
            Assert.Equal(Id(Encounter.Queen), glory.BossEncounterId);
            Assert.NotEqual(glory.BossEncounterId, glory.SecondBossEncounterId);
            Assert.NotEqual(0, glory.SecondBossEncounterId);
        }

        Assert.InRange(overgrowth, 1, 39);
    }

    /// <summary>
    /// An event epoch not revealed takes its event out of the act's list before the shuffle
    /// (Colorful Philosophers is Hive's, Trash Heap Underdocks', Reflections Glory's): the act's
    /// events lack it, the acts generated before it are untouched, and its own act draws one
    /// UpFront value fewer from there on.
    /// </summary>
    [Fact]
    public void ALockedEventLeavesItsActsShuffle()
    {
        var noPhilosophers = UnlockState.All.WithLockedEpochs(Epochs.Event3);
        var noTrashHeap = UnlockState.All.WithLockedEpochs(Epochs.Event1);
        var noReflections = UnlockState.All.WithLockedEpochs(Epochs.Event2);
        int underdocks = 0;
        foreach (string seed in Seeds(40))
        {
            var mature = Run(seed);
            var hive = Run(seed, noPhilosophers);
            SameRooms(mature.Acts[0], hive.Acts[0]);
            Assert.DoesNotContain(RunConstants.EventColorfulPhilosophers, hive.Acts[1].Events);
            Assert.Contains(RunConstants.EventColorfulPhilosophers, mature.Acts[1].Events);
            Assert.NotEqual(mature.Acts[1].NormalEncounters, hive.Acts[1].NormalEncounters);

            var glory = Run(seed, noReflections);
            SameRooms(mature.Acts[1], glory.Acts[1]);
            Assert.DoesNotContain(RunConstants.EventReflections, glory.Acts[2].Events);

            var trash = Run(seed, noTrashHeap);
            Assert.DoesNotContain(RunConstants.EventTrashHeap, trash.Acts[0].Events);
            if (mature.Acts[0].Act == RunConstants.ActUnderdocks)
            {
                underdocks++;
                Assert.NotEqual(mature.Acts[0].NormalEncounters, trash.Acts[0].NormalEncounters);
            }
            else
            {
                // Trash Heap is no Overgrowth event: nothing moves.
                Assert.Empty(DeepState.Diff(mature, trash, 6, ".Unlocks"));
            }
        }

        Assert.InRange(underdocks, 1, 39);
    }

    /// <summary>
    /// Hive offers Orobas only once its epoch is revealed (`Hive.GetUnlockedAncients`), and Darv
    /// is a shared ancient only once his is (`UnlockState.SharedAncients`). Each act still spends
    /// its subset-size and ancient draws, so UpFront ends where it did.
    /// </summary>
    [Fact]
    public void OrobasAndDarvWaitForTheirEpochs()
    {
        var noOrobas = UnlockState.All.WithLockedEpochs(Epochs.Orobas);
        var noDarv = UnlockState.All.WithLockedEpochs(Epochs.Darv);
        bool sawOrobas = false, sawDarv = false;
        foreach (string seed in Seeds(80))
        {
            var mature = Run(seed);
            sawOrobas |= mature.Acts.Any(a => a.Ancient == RunConstants.AncientOrobas);
            sawDarv |= mature.Acts.Any(a => a.Ancient == RunConstants.AncientDarv);
            var orobas = Run(seed, noOrobas);
            var darv = Run(seed, noDarv);
            Assert.DoesNotContain(orobas.Acts, a => a.Ancient == RunConstants.AncientOrobas);
            Assert.DoesNotContain(darv.Acts, a => a.Ancient == RunConstants.AncientDarv);
            Assert.Equal(mature.Rng.UpFront.CallCount, orobas.Rng.UpFront.CallCount);
            Assert.Equal(mature.Rng.UpFront.CallCount, darv.Rng.UpFront.CallCount);
            Assert.Equal(
                mature.Acts.Select(a => a.NormalEncounters),
                orobas.Acts.Select(a => a.NormalEncounters)
            );
        }

        Assert.True(sawOrobas && sawDarv);
    }

    /// <summary>
    /// `ActModel.GetRandomList`: Underdocks is a candidate only once its epoch is revealed, and an
    /// unlocked alternate act never visited is taken without a roll.
    /// </summary>
    [Fact]
    public void UnderdocksIsRolledWhenUnlockedAndTakenWhileUndiscovered()
    {
        var locked = UnlockState.All.WithLockedEpochs(Epochs.Underdocks);
        var undiscovered = UnlockState.All.WithUndiscoveredActs(RunConstants.ActUnderdocks);
        var rolled = new HashSet<int>();
        foreach (string seed in Seeds(30))
        {
            rolled.Add(Run(seed).Acts[0].Act);
            Assert.Equal(RunConstants.ActOvergrowth, Run(seed, locked).Acts[0].Act);
            Assert.Equal(RunConstants.ActUnderdocks, Run(seed, undiscovered).Acts[0].Act);
        }

        Assert.Equal(new[] { ActOvergrowth, ActUnderdocks }, rolled.Order());
    }

    /// <summary>
    /// Every walk over `UnlockState.Characters`: Splash (its own pool when it is the only one),
    /// Prismatic Gem's widened rewards, Colorful Philosophers' gate and pools, Orobas's Sea Glass
    /// (a `NextItem` over nobody draws nothing and brands the player's own).
    /// </summary>
    [Fact]
    public void TheOtherCharactersAreTheProfilesOwn()
    {
        Assert.Equal([PlayerCharacter.Ironclad], IroncladOnly.Characters);
        Assert.DoesNotContain(PlayerCharacter.Defect, NoDefect.Characters);

        int[] defect = CharacterPools.Cards(PlayerCharacter.Defect).ToArray();
        var splash = CardEffects.SplashPool(new CombatState { Unlocks = NoDefect });
        Assert.NotEmpty(splash);
        Assert.DoesNotContain(splash, defect.Contains);
        Assert.Equal(
            CardEffects.SplashPool(new CombatState()).Where(id => !defect.Contains(id)),
            splash
        );
        var own = CardEffects.SplashPool(new CombatState { Unlocks = IroncladOnly });
        Assert.NotEmpty(own);
        int[] ironclad = CharacterPools.Cards(PlayerCharacter.Ironclad).ToArray();
        Assert.All(own, id => Assert.Contains(id, ironclad));

        var gem = RunRewardGenerator.CardRewardPool(PlayerCharacter.Ironclad, false, true, NoDefect);
        Assert.DoesNotContain(gem, defect.Contains);
        var everyPool = RunRewardGenerator.CardRewardPool(PlayerCharacter.Ironclad, false, true);
        Assert.Contains(everyPool, defect.Contains);

        var young = Run("UNLOCK0000", IroncladOnly);
        var mature = Run("UNLOCK0000");
        Assert.False(RunNonCombatEffects.IsEventAllowedForRun(young, EventColorfulPhilosophers));
        Assert.True(RunNonCombatEffects.IsEventAllowedForRun(mature, EventColorfulPhilosophers));
        var three = Run("UNLOCK0000", NoDefect);
        Assert.Equal(3, RunNonCombatEffects.ColorfulPhilosopherPoolOrder(three).Length);

        RunNonCombatEffects.GenerateAncientOptions(young, RunConstants.AncientOrobas);
        Assert.Equal(-1, young.SeaGlassCharacter);
        Assert.Equal(ironclad, RunConstants.SeaGlassPool(young).ToArray());
        RunNonCombatEffects.GenerateAncientOptions(three, RunConstants.AncientOrobas);
        Assert.InRange(three.SeaGlassCharacter, 0, 2);
    }

    [Theory]
    [InlineData("NEOW_EPOCH")]
    [InlineData("RELIC1_EPOCH")]
    [InlineData("IRONCLAD2_EPOCH")]
    public void AnEpochTheEngineDoesNotModelIsRefused(string epoch) =>
        Assert.Throws<ArgumentException>(() => UnlockState.All.WithLockedEpochs(epoch));
}
