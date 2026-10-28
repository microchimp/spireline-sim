using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// D32: the run's action mask has versions. Version 1 (the default) is the mask as it was; version 2 adds the potion
/// discard on the card reward, deck-select, bundle-select and Crystal Sphere screens, where the belt stayed closed, and
/// nothing else. `NPotionPopup` enables its discard button behind `Owner.CanRemovePotions` alone; only the Future of
/// Potions, Ranwid the Elder and the Stone of All Time clear that flag, so version 2 keeps the belt shut inside them.
/// Use is not opened. <see cref="RunMaskGoldenTests" /> pins version 1 to the values it had before.
/// </summary>
public class RunMaskVersionTests
{
    private static readonly RunPhase[] ClosedBeltScreens =
    [
        RunPhase.CardReward,
        RunPhase.TransformSelect,
        RunPhase.BundleSelect,
        RunPhase.CrystalSphere,
    ];

    private static readonly RunPhase[] OpenBeltScreens =
    [
        RunPhase.Map,
        RunPhase.Rest,
        RunPhase.Shop,
        RunPhase.Treasure,
        RunPhase.RelicReward,
        RunPhase.Event,
        RunPhase.Ancient,
    ];

    public static TheoryData<RunPhase> Closed => Rows(ClosedBeltScreens);

    public static TheoryData<RunPhase> Open => Rows(OpenBeltScreens);

    private static TheoryData<RunPhase> Rows(RunPhase[] phases)
    {
        var data = new TheoryData<RunPhase>();
        foreach (var phase in phases)
        {
            data.Add(phase);
        }

        return data;
    }

    private static RunEngine V(int version, RunPhase phase, params string[] belt)
    {
        var engine = RunMaskGoldenTests.At(phase, belt);
        engine.MaskVersion = version;
        return engine;
    }

    [Theory]
    [MemberData(nameof(Closed))]
    public void VersionTwoAddsTheDiscardSlotsOfTheHeldPotionsToTheWholeMaskAndNothingElse(RunPhase phase)
    {
        string[] belt = ["BlockPotion", "BloodPotion", "FruitJuice"];
        var v1 = RunMaskGoldenTests.Mask(V(1, phase, belt));
        var v2 = RunMaskGoldenTests.Mask(V(2, phase, belt));

        var expected = (int[])v1.Clone();
        for (int slot = 0; slot < belt.Length; slot++)
        {
            expected[RunConstants.DiscardPotionAction + slot] = 1;
        }

        Assert.Equal(expected, v2);
        Assert.All(v1.Skip(RunConstants.UsePotionAction), value => Assert.Equal(0, value));
        // Use is not opened: not even for the AnyTime potions (Blood Potion, Fruit Juice) held in v2.
        Assert.All(
            v2.Skip(RunConstants.UsePotionAction).Take(RunConstants.PotionSlotCapacity),
            value => Assert.Equal(0, value)
        );
    }

    [Theory]
    [MemberData(nameof(Closed))]
    public void VersionTwoOffersNothingExtraOnAnEmptyBelt(RunPhase phase)
    {
        Assert.Equal(RunMaskGoldenTests.Mask(V(1, phase)), RunMaskGoldenTests.Mask(V(2, phase)));
    }

    [Theory]
    [MemberData(nameof(Open))]
    public void VersionTwoChangesNothingOnTheScreensWhoseBeltWasAlreadyOpen(RunPhase phase)
    {
        string[] belt = ["BlockPotion", "BloodPotion", "FruitJuice"];
        Assert.Equal(RunMaskGoldenTests.Mask(V(1, phase, belt)), RunMaskGoldenTests.Mask(V(2, phase, belt)));
    }

    [Theory]
    [MemberData(nameof(Closed))]
    public void VersionOneStillRefusesTheDiscardAndVersionTwoTakesIt(RunPhase phase)
    {
        var one = V(1, phase, "BlockPotion", "BloodPotion");
        Assert.Equal(-1, one.Step(RunConstants.DiscardPotionAction, -1, out _, out _, out _));
        Assert.NotEqual(0, one.State.PotionSlots[0]);
        Assert.Contains(
            "belt is closed",
            RunPotionUse.Refusal(one.State, RunConstants.DiscardPotionAction, one.MaskVersion)!
        );

        var two = V(2, phase, "BlockPotion", "BloodPotion");
        int hp = two.State.PlayerHp;
        int gold = two.State.Gold;
        int deck = two.State.Deck.Count;
        Assert.Equal(0, two.Step(RunConstants.DiscardPotionAction, -1, out _, out _, out _));
        Assert.Equal(0, two.State.PotionSlots[0]);
        Assert.Equal(RunNonCombatEffects.NamedPotion("BloodPotion"), two.State.PotionSlots[1]);
        Assert.Equal((hp, gold, deck, phase), (two.State.PlayerHp, two.State.Gold, two.State.Deck.Count, two.State.Phase));
        Assert.Equal(-1, two.Step(RunConstants.DiscardPotionAction, -1, out _, out _, out _)); // the slot is empty now
        Assert.Null(RunPotionUse.Refusal(two.State, RunConstants.DiscardPotionAction + 1, 2));
    }

    [Theory]
    [MemberData(nameof(Closed))]
    public void AUseIsStillRefusedOnTheseScreensInBothVersions(RunPhase phase)
    {
        foreach (int version in new[] { 1, 2 })
        {
            var engine = V(version, phase, "BloodPotion");
            engine.State.PlayerHp = 10;
            Assert.Equal(-1, engine.Step(RunConstants.UsePotionAction, -1, out _, out _, out _));
            Assert.Equal(10, engine.State.PlayerHp);
            Assert.NotEqual(0, engine.State.PotionSlots[0]);
            Assert.NotNull(RunPotionUse.Refusal(engine.State, RunConstants.UsePotionAction, version));
        }
    }

    [Theory]
    [InlineData(RunPhase.CardReward, RunConstants.EventTheFutureOfPotions)]
    [InlineData(RunPhase.TransformSelect, RunConstants.EventStoneOfAllTime)]
    [InlineData(RunPhase.TransformSelect, RunConstants.EventRanwidTheElder)]
    [InlineData(RunPhase.CardReward, RunConstants.EventRanwidTheElder)]
    public void TheEventsThatHoldTheBeltKeepItShutOnTheirOwnCardScreensInVersionTwo(RunPhase phase, int eventId)
    {
        var engine = V(2, phase, "BlockPotion");
        engine.State.EventId = eventId;
        Assert.Equal(0, RunMaskGoldenTests.Mask(engine)[RunConstants.DiscardPotionAction]);
        Assert.Equal(-1, engine.Step(RunConstants.DiscardPotionAction, -1, out _, out _, out _));
        Assert.Contains("belt is closed", RunPotionUse.Refusal(engine.State, RunConstants.DiscardPotionAction, 2)!);
    }

    // ---- version 3 (D34): the use of the AnyTime potions on the four overlay screens

    private static int Slot(string potion) => RunNonCombatEffects.NamedPotion(potion);

    [Theory]
    [MemberData(nameof(Closed))]
    public void VersionThreeAddsExactlyTheUseSlotsOfTheAnyTimePotionsToVersionTwosMask(RunPhase phase)
    {
        string[] belt = ["BlockPotion", "BloodPotion", "FruitJuice"];
        var v2 = RunMaskGoldenTests.Mask(V(2, phase, belt));
        var v3 = RunMaskGoldenTests.Mask(V(3, phase, belt));
        var expected = (int[])v2.Clone();
        expected[RunConstants.UsePotionAction + 1] = 1; // Blood Potion
        expected[RunConstants.UsePotionAction + 2] = 1; // Fruit Juice; the Block Potion is CombatOnly
        Assert.Equal(expected, v3);

        // Entropic Brew is AnyTime too; the Foul Potion wants the merchant's room, a Fire Potion is CombatOnly
        string[] other = ["EntropicBrew", "FoulPotion", "FirePotion"];
        var o2 = RunMaskGoldenTests.Mask(V(2, phase, other));
        var o3 = RunMaskGoldenTests.Mask(V(3, phase, other));
        var expectedOther = (int[])o2.Clone();
        expectedOther[RunConstants.UsePotionAction] = 1;
        Assert.Equal(expectedOther, o3);
    }

    [Theory]
    [MemberData(nameof(Closed))]
    public void VersionThreeOffersNothingExtraOnAnEmptyBeltAndKeepsTheOtherVersionsAsTheyWere(RunPhase phase)
    {
        Assert.Equal(RunMaskGoldenTests.Mask(V(2, phase)), RunMaskGoldenTests.Mask(V(3, phase)));
        Assert.Equal(RunMaskGoldenTests.Mask(V(1, phase)), RunMaskGoldenTests.Mask(V(3, phase)));
        // versions 1 and 2 still list no use on these screens even for a belt of AnyTime potions
        foreach (int version in new[] { 1, 2 })
        {
            var mask = RunMaskGoldenTests.Mask(V(version, phase, "BloodPotion", "FruitJuice", "EntropicBrew"));
            Assert.All(mask.Skip(RunConstants.UsePotionAction).Take(RunConstants.PotionSlotCapacity), v => Assert.Equal(0, v));
        }
    }

    [Theory]
    [MemberData(nameof(Open))]
    public void VersionThreeChangesNothingOnTheScreensWhoseBeltWasAlreadyOpen(RunPhase phase)
    {
        string[] belt = ["BlockPotion", "BloodPotion", "FruitJuice"];
        Assert.Equal(RunMaskGoldenTests.Mask(V(2, phase, belt)), RunMaskGoldenTests.Mask(V(3, phase, belt)));
        Assert.Equal(RunMaskGoldenTests.Mask(V(1, phase, belt)), RunMaskGoldenTests.Mask(V(3, phase, belt)));
    }

    [Theory]
    [MemberData(nameof(Closed))]
    public void ABloodPotionUsedOnTheseScreensHealsAndLeavesTheBeltAndNothingElseChanges(RunPhase phase)
    {
        var engine = V(3, phase, "BlockPotion", "BloodPotion");
        engine.State.PlayerMaxHp = 80;
        engine.State.PlayerHp = 30;
        int gold = engine.State.Gold;
        int deck = engine.State.Deck.Count;
        Assert.Null(RunPotionUse.Refusal(engine.State, RunConstants.UsePotionAction + 1, 3));
        Assert.Equal(0, engine.Step(RunConstants.UsePotionAction + 1, -1, out _, out _, out _));
        Assert.Equal(46, engine.State.PlayerHp); // 20% of the max hp (BloodPotion.OnUse)
        Assert.Equal(0, engine.State.PotionSlots[1]);
        Assert.Equal(Slot("BlockPotion"), engine.State.PotionSlots[0]);
        Assert.Equal((gold, deck, phase), (engine.State.Gold, engine.State.Deck.Count, engine.State.Phase));
        Assert.Equal(-1, engine.Step(RunConstants.UsePotionAction + 1, -1, out _, out _, out _)); // the slot is empty now
        Assert.Contains("empty", RunPotionUse.Refusal(engine.State, RunConstants.UsePotionAction + 1, 3)!);
    }

    [Theory]
    [MemberData(nameof(Closed))]
    public void AFruitJuiceAddsMaxHpAndAnEntropicBrewFillsTheBeltFromTheSameStreamAsAtAShop(RunPhase phase)
    {
        var juice = V(3, phase, "FruitJuice");
        int max = juice.State.PlayerMaxHp;
        Assert.Equal(0, juice.Step(RunConstants.UsePotionAction, -1, out _, out _, out _));
        Assert.Equal(max + 5, juice.State.PlayerMaxHp);
        Assert.Equal(0, juice.State.PotionSlots[0]);

        // The brew empties its own slot first and rolls a potion for every open slot: the same belt as at a shop (version 1).
        var brew = V(3, phase, "EntropicBrew", "BlockPotion");
        Assert.Equal(0, brew.Step(RunConstants.UsePotionAction, -1, out _, out _, out _));
        var atShop = RunMaskGoldenTests.At(RunPhase.Shop, "EntropicBrew", "BlockPotion");
        Assert.Equal(0, atShop.Step(RunConstants.UsePotionAction, -1, out _, out _, out _));
        Assert.Equal(atShop.State.PotionSlots.Take(3), brew.State.PotionSlots.Take(3));
        Assert.Equal(Slot("BlockPotion"), brew.State.PotionSlots[1]);
        Assert.All(brew.State.PotionSlots.Take(3), id => Assert.NotEqual(0, id));
        Assert.Equal(phase, brew.State.Phase);
    }

    [Theory]
    [MemberData(nameof(Closed))]
    public void ACombatOnlyPotionAndTheFoulPotionAreRefusedWithTheGamesReasonAndAUseUnderVersionTwoIsStillRefused(
        RunPhase phase
    )
    {
        var engine = V(3, phase, "BlockPotion", "FoulPotion", "BloodPotion");
        engine.State.Gold = 10;
        engine.State.PlayerHp = 10;
        Assert.Equal(-1, engine.Step(RunConstants.UsePotionAction, -1, out _, out _, out _));
        Assert.Contains("Usage", RunPotionUse.Refusal(engine.State, RunConstants.UsePotionAction, 3)!);
        Assert.Equal(-1, engine.Step(RunConstants.UsePotionAction + 1, -1, out _, out _, out _));
        Assert.Contains("merchant's room", RunPotionUse.Refusal(engine.State, RunConstants.UsePotionAction + 1, 3)!);
        Assert.Equal((10, 10), (engine.State.Gold, engine.State.PlayerHp));
        Assert.Equal(
            (Slot("BlockPotion"), Slot("FoulPotion"), Slot("BloodPotion")),
            (engine.State.PotionSlots[0], engine.State.PotionSlots[1], engine.State.PotionSlots[2])
        );

        var two = V(2, phase, "BloodPotion");
        two.State.PlayerHp = 10;
        Assert.Equal(-1, two.Step(RunConstants.UsePotionAction, -1, out _, out _, out _));
        Assert.Contains("thrown away", RunPotionUse.Refusal(two.State, RunConstants.UsePotionAction, 2)!);
        Assert.Equal(10, two.State.PlayerHp);
    }

    [Theory]
    [InlineData(RunPhase.CardReward, RunConstants.EventTheFutureOfPotions)]
    [InlineData(RunPhase.TransformSelect, RunConstants.EventStoneOfAllTime)]
    [InlineData(RunPhase.TransformSelect, RunConstants.EventRanwidTheElder)]
    [InlineData(RunPhase.CardReward, RunConstants.EventRanwidTheElder)]
    public void TheEventsThatHoldTheBeltKeepTheUseShutToo(RunPhase phase, int eventId)
    {
        var engine = V(3, phase, "BloodPotion");
        engine.State.EventId = eventId;
        engine.State.PlayerHp = 10;
        Assert.Equal(0, RunMaskGoldenTests.Mask(engine)[RunConstants.UsePotionAction]);
        Assert.Equal(-1, engine.Step(RunConstants.UsePotionAction, -1, out _, out _, out _));
        Assert.Equal(10, engine.State.PlayerHp);
        Assert.Contains("belt is closed", RunPotionUse.Refusal(engine.State, RunConstants.UsePotionAction, 3)!);
    }

    [Fact]
    public void ACombatAndAFinishedRunStayClosedForTheUseInVersionThree()
    {
        foreach (var phase in new[] { RunPhase.Combat, RunPhase.Complete })
        {
            var engine = V(3, phase, "BloodPotion");
            Assert.False(RunPotionUse.CanUse(engine.State, 0, 3));
            Assert.False(RunPotionUse.CanDiscard(engine.State, 0, 3));
        }
    }

    [Fact]
    public void ACombatAndAFinishedRunStayClosedInVersionTwo()
    {
        foreach (var phase in new[] { RunPhase.Combat, RunPhase.Complete })
        {
            var engine = V(2, phase, "BlockPotion");
            Assert.False(RunPotionUse.CanDiscard(engine.State, 0, 2));
        }
    }

    [Fact]
    public void TheVersionIsASettingOfTheHandleThatDefaultsToOneSurvivesResetAndIsCopiedByClone()
    {
        var engine = new RunEngine();
        Assert.Equal(1, engine.MaskVersion);
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.MaskVersion = 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.MaskVersion = 5);
        engine.MaskVersion = 4;
        Assert.Equal(4, engine.MaskVersion);
        engine.MaskVersion = 3;
        Assert.Equal(3, engine.MaskVersion);
        engine.MaskVersion = 2;
        engine.Reset("ABCDEF");
        Assert.Equal(2, engine.MaskVersion);
        Assert.Equal(2, engine.Clone().MaskVersion);
        Assert.Equal(2, engine.Clone(7).MaskVersion);
        Assert.Equal(1, new RunEngine().Clone().MaskVersion);
    }

    [Fact]
    public unsafe void TheNativeExportSetsTheVersionRefusesOthersAndTheReasonFollowsIt()
    {
        int handle = RunNativeExports.Sts2Run_Create();
        try
        {
            Assert.Equal(-1, RunNativeExports.Sts2Run_SetMaskVersion(255, 2));
            Assert.Equal(-2, RunNativeExports.Sts2Run_SetMaskVersion(handle, 0));
            Assert.Equal(-2, RunNativeExports.Sts2Run_SetMaskVersion(handle, 5));
            Assert.Equal(0, RunNativeExports.Sts2Run_SetMaskVersion(handle, 4));
            Assert.Equal(0, RunNativeExports.Sts2Run_SetMaskVersion(handle, 1));
            Assert.True(RunNativeExports.TryGetRun(handle, out var run));
            Assert.Equal(1, run.MaskVersion);
            run.Reset("ABCDEF");
            run.State.Phase = RunPhase.CardReward;
            Array.Clear(run.State.PotionSlots);
            run.State.PotionSlots[0] = RunNonCombatEffects.NamedPotion("BlockPotion");
            var buf = new byte[256];
            fixed (byte* p = buf)
            {
                int n = RunNativeExports.Sts2Run_PotionActionReason(handle, RunConstants.DiscardPotionAction, p, buf.Length);
                Assert.True(n > 0);
                Assert.Contains("belt is closed", System.Text.Encoding.UTF8.GetString(buf, 0, n));

                Assert.Equal(0, RunNativeExports.Sts2Run_SetMaskVersion(handle, 2));
                Assert.Equal(
                    0,
                    RunNativeExports.Sts2Run_PotionActionReason(handle, RunConstants.DiscardPotionAction, p, buf.Length)
                );
            }

            var mask = new int[RunConstants.MaxActions];
            run.WriteActionMask(mask);
            Assert.NotEqual(0, mask[RunConstants.DiscardPotionAction]);
            Assert.Equal(32, RunNativeExports.RUN_NATIVE_API_VERSION);

            // version 3: the reason for a use there follows the version too
            run.State.PotionSlots[1] = RunNonCombatEffects.NamedPotion("BloodPotion");
            fixed (byte* p = buf)
            {
                Assert.Contains(
                    "thrown away",
                    System.Text.Encoding.UTF8.GetString(
                        buf,
                        0,
                        RunNativeExports.Sts2Run_PotionActionReason(handle, RunConstants.UsePotionAction + 1, p, buf.Length)
                    )
                );
                Assert.Equal(0, RunNativeExports.Sts2Run_SetMaskVersion(handle, 3));
                Assert.Equal(
                    0,
                    RunNativeExports.Sts2Run_PotionActionReason(handle, RunConstants.UsePotionAction + 1, p, buf.Length)
                );
            }

            run.WriteActionMask(mask);
            Assert.NotEqual(0, mask[RunConstants.UsePotionAction + 1]);
            Assert.Equal(0, mask[RunConstants.UsePotionAction]); // the Block Potion is combat-only
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }
}
