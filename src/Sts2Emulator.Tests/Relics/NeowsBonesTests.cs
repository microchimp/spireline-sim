using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Neow's Bones: "gain 2 random Neow Relics, add 1 random Curse to your Deck", read off
/// MegaCrit.Sts2.Core.Models.Relics/NeowsBones.cs.
///
/// <para>
/// <c>AfterObtained</c> shuffles <c>GetValidRelics</c> on <c>PlayerRng.Rewards</c>, takes
/// two, and OFFERS them — a <c>RewardsSet</c> with <c>WithSkippingDisallowed</c>, so both
/// are claimed — and only then adds the curse. The emulator used to take two independent
/// <c>Rng.UpFront.NextItem</c> draws and apply them on the spot: the wrong stream, a draw
/// that can hand out the same relic twice where a shuffle-and-take cannot, no screen at
/// all, and a candidate list of only the positives.
/// </para>
/// </summary>
public class NeowsBonesTests
{
    private static RunEngine AtNeowsBones(string seed)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        engine.State.Phase = RunPhase.Ancient;
        engine.State.NeowOptions[0] = RunConstants.RelicNeowsBones;
        return engine;
    }

    private static int Claim(RunEngine engine) => engine.Step(0, -1, out _, out _, out _);

    [Fact]
    public void ItOffersTwoRelicsOnAScreenRatherThanGrantingThem()
    {
        var engine = AtNeowsBones("J09SPL8Y3V");

        engine.Step(0, -1, out _, out _, out _);

        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
        Assert.NotEqual(0, engine.State.RelicReward);
        Assert.Single(engine.State.PendingBonusRelicRewards);
        // Only Burning Blood and the Bones themselves — neither offered relic is held yet.
        Assert.Equal(2, engine.State.Relics.Count);
    }

    /// <summary>
    /// A live capture of this seed answers the screen twice and comes away with Winged
    /// Boots and then Silken Tress, in that order (`J09SPL8Y3V`).
    /// </summary>
    [Fact]
    public void TheOfferMatchesTheCapturedRun()
    {
        var engine = AtNeowsBones("J09SPL8Y3V");

        engine.Step(0, -1, out _, out _, out _);

        Assert.Equal(RunConstants.RelicWingedBoots, engine.State.RelicReward);
        Assert.Equal(RunConstants.RelicSilkenTress, engine.State.PendingBonusRelicRewards[0]);
    }

    /// <summary>
    /// A relic claimed from a reward screen is obtained through <c>RelicCmd.Obtain</c>,
    /// which runs its <c>AfterObtained</c> — Silken Tress zeroed the captured run's gold
    /// the moment it was taken.
    /// </summary>
    [Fact]
    public void ClaimingARelicRunsItsPickupEffect()
    {
        var engine = AtNeowsBones("J09SPL8Y3V");
        engine.Step(0, -1, out _, out _, out _);
        Assert.Equal(99, engine.State.Gold);

        Claim(engine);
        Claim(engine);

        Assert.Contains(engine.State.Relics, relic => relic.DefId == RunConstants.RelicSilkenTress);
        Assert.Equal(0, engine.State.Gold);
    }

    /// <summary>
    /// The curse comes after the screen, not with the relics: `AfterObtained` awaits the
    /// Offer() and adds it on the line below.
    /// </summary>
    [Fact]
    public void TheCurseArrivesOnlyOnceBothRelicsAreClaimed()
    {
        var engine = AtNeowsBones("J09SPL8Y3V");
        int deckBefore = engine.State.Deck.Count;
        engine.Step(0, -1, out _, out _, out _);

        Claim(engine);
        Assert.Equal(deckBefore, engine.State.Deck.Count);
        Assert.True(engine.State.PendingNeowsBonesCurse);

        Claim(engine);
        Assert.Equal(deckBefore + 1, engine.State.Deck.Count);
        Assert.False(engine.State.PendingNeowsBonesCurse);
    }

    /// <summary>
    /// And the claim that empties the screen returns to Neow itself, which stays up for
    /// one more Proceed the way it does after every other blessing.
    /// </summary>
    [Fact]
    public void TheLastClaimReturnsToNeowRatherThanTheMap()
    {
        var engine = AtNeowsBones("J09SPL8Y3V");
        engine.Step(0, -1, out _, out _, out _);

        Claim(engine);
        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);

        Claim(engine);
        Assert.Equal(RunPhase.Ancient, engine.State.Phase);

        engine.Step(0, -1, out _, out _, out _);
        Assert.Equal(RunPhase.Map, engine.State.Phase);
    }

    /// <summary>
    /// A shuffle-and-take cannot repeat itself, where two independent draws can. Swept
    /// over many seeds because a duplicate is a one-in-twenty-six coincidence per run.
    /// </summary>
    [Fact]
    public void ItNeverOffersTheSameRelicTwice()
    {
        foreach (
            string seed in new[] { "J09SPL8Y3V", "ABCDEF", "0", "AAB", "QS2GYXRKWN", "WK1DEGZD8P" }
        )
        {
            var engine = AtNeowsBones(seed);
            engine.Step(0, -1, out _, out _, out _);

            Assert.NotEqual(engine.State.RelicReward, engine.State.PendingBonusRelicRewards[0]);
            Assert.NotEqual(RunConstants.RelicNeowsBones, engine.State.RelicReward);
            Assert.NotEqual(RunConstants.RelicNeowsBones, engine.State.PendingBonusRelicRewards[0]);
        }
    }
}

/// <summary>
/// A relic claimed on Neow's Bones' rewards screen can open a grid of its own (Lead
/// Paperweight's colourless pair). The grid closes onto the rewards screen, which still owes
/// the second relic, and only the claim that empties it returns to Neow -- night03 seed
/// 672CRJ8TJ9 in the game takes Lead Paperweight, then Nutritious Oyster, then the curse.
/// </summary>
public class NeowsBonesNestedGridTests
{
    private static RunEngine AtBonesWithPaperweightFirst()
    {
        var engine = new RunEngine();
        engine.Reset("672CRJ8TJ9", 10);
        engine.State.Phase = RunPhase.Ancient;
        engine.State.NeowOptions[0] = RunConstants.RelicNeowsBones;
        engine.Step(0, -1, out _, out _, out _);
        return engine;
    }

    [Fact]
    public void TheGridClosesOntoTheRewardsScreenThatStillOwesARelic()
    {
        var engine = AtBonesWithPaperweightFirst();
        Assert.Equal(RunConstants.RelicLeadPaperweight, engine.State.RelicReward);
        Assert.Equal(RunConstants.RelicNutritiousOyster, engine.State.PendingBonusRelicRewards[0]);

        engine.Step(0, -1, out _, out _, out _); // claim the Paperweight: its grid opens
        Assert.Equal(RunPhase.TransformSelect, engine.State.Phase);
        engine.Step(0, -1, out _, out _, out _); // take the first colourless card

        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
        Assert.Equal(RunConstants.RelicNutritiousOyster, engine.State.RelicReward);
        Assert.True(engine.State.PendingNeowsBonesCurse);
    }

    [Fact]
    public void OnlyTheClaimThatEmptiesTheScreenReturnsToNeowAndAddsTheCurse()
    {
        var engine = AtBonesWithPaperweightFirst();
        engine.Step(0, -1, out _, out _, out _);
        engine.Step(0, -1, out _, out _, out _);
        int deckBefore = engine.State.Deck.Count;

        engine.Step(0, -1, out _, out _, out _); // the Oyster

        Assert.Equal(RunPhase.Ancient, engine.State.Phase);
        Assert.False(engine.State.PendingNeowsBonesCurse);
        Assert.Equal(deckBefore + 1, engine.State.Deck.Count);
    }
}

/// <summary>
/// A relic claimed on Neow's Bones' rewards screen whose pickup opens a screen of its own (a
/// deck selection, Scroll Boxes' bundles) is answered there first; only then is the rewards
/// screen back with the relic still owed, and Neow's Bones' curse comes once none is. The copy
/// dropped the selection and the bundle and walked on (nightly 2026-10-29: Gbc74665482 and
/// Gb64bb714fe, the game on a card selection or a bundle right after Neow's Bones, the copy on
/// the map). Swept over seeds for an offer holding such a relic.
/// </summary>
public class NeowsBonesClaimScreenTests
{
    private static RunEngine AtTheBonesScreen(string seed)
    {
        var engine = new RunEngine();
        engine.Reset(seed, 10);
        engine.State.Phase = RunPhase.Ancient;
        engine.State.NeowOptions[0] = RunConstants.RelicNeowsBones;
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));
        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
        return engine;
    }

    private static bool OnAClaimScreen(RunEngine engine) =>
        engine.State.Phase is RunPhase.TransformSelect or RunPhase.BundleSelect;

    /// <summary>A deck selection or the bundles: the screens the copy dropped (a pre-rolled grid it already opened).</summary>
    private static bool OnADroppedKindOfScreen(RunEngine engine) =>
        engine.State.Phase == RunPhase.BundleSelect
        || (engine.State.Phase == RunPhase.TransformSelect && engine.State.PendingOfferCards.Length == 0);

    /// <summary>Answers the selection or the bundle with its first legal choice.</summary>
    private static void Answer(RunEngine engine)
    {
        var mask = new int[RunConstants.MaxActions];
        for (int i = 0; i < 40 && OnAClaimScreen(engine); i++)
        {
            if (engine.State.Phase == RunPhase.BundleSelect && engine.State.SelectedBundle >= 0)
            {
                Assert.Equal(0, engine.Step(RunConstants.BundleConfirmAction, -1, out _, out _, out _));
                continue;
            }

            engine.WriteActionMask(mask);
            int action = Array.FindIndex(mask, legal => legal != 0);
            Assert.True(action >= 0 && action < RunConstants.UsePotionAction, $"no screen action in {engine.State.Phase}");
            Assert.Equal(0, engine.Step(action, -1, out _, out _, out _));
        }

        Assert.False(OnAClaimScreen(engine));
    }

    [Fact]
    public void AClaimsOwnScreenComesFirstThenTheRestOfTheOffer()
    {
        int found = 0;
        for (int n = 0; n < 400 && found < 6; n++)
        {
            var engine = AtTheBonesScreen($"BONES{n}");
            int second = engine.State.PendingBonusRelicRewards[0];
            int deckBefore = engine.State.Deck.Count;

            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _)); // the first relic
            if (OnAClaimScreen(engine))
            {
                found += OnADroppedKindOfScreen(engine) ? 1 : 0;
                Assert.True(engine.State.PendingNeowsBonesCurse);
                Answer(engine);
                Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
                Assert.Equal(second, engine.State.RelicReward);
                Assert.True(engine.State.PendingNeowsBonesCurse);
                continue;
            }

            Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
            int deckAfterFirst = engine.State.Deck.Count;
            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _)); // the second relic
            if (OnAClaimScreen(engine))
            {
                found += OnADroppedKindOfScreen(engine) ? 1 : 0;
                // The curse waits for the screen: the selection never offers it.
                Assert.True(engine.State.PendingNeowsBonesCurse);
                Answer(engine);
                Assert.Equal(RunPhase.Ancient, engine.State.Phase);
                Assert.False(engine.State.PendingNeowsBonesCurse);
            }
        }

        Assert.True(found >= 3, $"only {found} offers held a relic whose pickup opens a selection or the bundles");
    }
}
