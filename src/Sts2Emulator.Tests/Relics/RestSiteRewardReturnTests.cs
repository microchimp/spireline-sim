using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// A rest-site option that opens a screen of its own is awaited by the rest site: the game's
/// heal awaits its rewards (<c>HealRestSiteOption.ExecuteRestSiteHeal</c> offers Dream
/// Catcher's card and Tiny Mailbox's potions through <c>RewardsCmd.OfferCustom</c>), and when
/// that screen closes the rest site's own screen is back, on its result page. The copy walked
/// on to the map from the reward screen and left the rest site's result flag set, so the next
/// rest site of the run took its first action as that stale result page and was left without a
/// choice (nightly 2026-10-29, group G1b02fc9e96: the game at a rest site, the copy already in
/// the fight after it; the copy's own arm passed 22 rest sites without a decision, every one
/// after an earlier heal that paid a Dream Catcher card).
/// </summary>
public class RestSiteRewardReturnTests
{
    private static RunEngine WalkInto(RunEngine engine, int nodeType)
    {
        Assert.Equal(RunPhase.Map, engine.State.Phase);
        int option = Array.FindIndex(engine.State.MapNodeTypes, type => type != RunConstants.NodeNone);
        Assert.True(option >= 0);
        engine.State.MapNodeTypes[option] = nodeType;
        Assert.Equal(0, engine.Step(option, -1, out _, out _, out _));
        return engine;
    }

    private static RunEngine AtARestSiteWith(string relic)
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        RunNonCombatEffects.ApplyRelicPickup(engine.State, RunNonCombatEffects.NamedRelic(relic));
        engine.State.Phase = RunPhase.Map;
        WalkInto(engine, RunConstants.NodeRest);
        Assert.Equal(RunPhase.Rest, engine.State.Phase);
        engine.State.PlayerHp = 1;
        return engine;
    }

    private static void LeaveTheResultPage(RunEngine engine)
    {
        Assert.Equal(RunPhase.Rest, engine.State.Phase);
        Assert.True(engine.State.RestResultPending);
        Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));
        Assert.Equal(RunPhase.Map, engine.State.Phase);
    }

    /// <summary>The next rest site asks again: the heal is on its screen and stepping it heals.</summary>
    private static void TheNextRestSiteOffersItsOptions(RunEngine engine)
    {
        WalkInto(engine, RunConstants.NodeRest);
        Assert.Equal(RunPhase.Rest, engine.State.Phase);
        Assert.False(engine.State.RestResultPending);
        engine.State.PlayerHp = 1;
        int map = engine.State.Floor;

        Assert.Equal(0, engine.Step(RunConstants.RestHealAction, -1, out _, out _, out _));

        Assert.True(engine.State.PlayerHp > 1, "the second rest site's heal was taken as a leave");
        Assert.Equal(map, engine.State.Floor);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void DreamCatchersCardClosesBackOntoTheRestSite(bool pick)
    {
        var engine = AtARestSiteWith("DreamCatcher");

        Assert.Equal(0, engine.Step(RunConstants.RestHealAction, -1, out _, out _, out _));
        Assert.Equal(RunPhase.CardReward, engine.State.Phase);
        Assert.Equal(0, engine.Step(pick ? 0 : RunConstants.RewardSkipAction, -1, out _, out _, out _));

        LeaveTheResultPage(engine);
        TheNextRestSiteOffersItsOptions(engine);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TinyMailboxsPotionsCloseBackOntoTheRestSite(bool claim)
    {
        var engine = AtARestSiteWith("TinyMailbox");
        Array.Clear(engine.State.PotionSlots);

        Assert.Equal(0, engine.Step(RunConstants.RestHealAction, -1, out _, out _, out _));
        Assert.Equal(RunPhase.RelicReward, engine.State.Phase);
        if (claim)
        {
            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));
            Assert.Equal(0, engine.Step(0, -1, out _, out _, out _));
        }

        if (engine.State.Phase == RunPhase.RelicReward)
        {
            Assert.Equal(0, engine.Step(RunConstants.RewardSkipAction, -1, out _, out _, out _));
        }

        LeaveTheResultPage(engine);
        TheNextRestSiteOffersItsOptions(engine);
    }

    /// <summary>
    /// Under Miniature Tent the smith's card screen closes onto the options left, the smith spent
    /// (2026-10-29: a Tent run that smithed was walked out of the rest site the game kept open).
    /// </summary>
    [Fact]
    public void UnderTheTentTheSmithsScreenClosesOntoTheOptionsLeft()
    {
        var engine = AtARestSiteWith("MiniatureTent");

        Assert.Equal(0, engine.Step(RunConstants.RestUpgradeAction, -1, out _, out _, out _));
        Assert.Equal(RunPhase.TransformSelect, engine.State.Phase);
        int card = engine.State.Deck.FindIndex(RunConstants.IsRunCardUpgradable);
        Assert.Equal(0, engine.Step(card, -1, out _, out _, out _));

        Assert.Equal(RunPhase.Rest, engine.State.Phase);
        Assert.False(engine.State.RestResultPending);
        var mask = new int[RunConstants.MaxActions];
        engine.WriteActionMask(mask);
        Assert.NotEqual(0, mask[RunConstants.RestHealAction]);
        Assert.Equal(0, mask[RunConstants.RestUpgradeAction]);
    }

    /// <summary>Without the Tent the smith ends on the result page, and the next rest site asks again.</summary>
    [Fact]
    public void WithoutTheTentTheSmithEndsOnItsResultPage()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        engine.State.Phase = RunPhase.Map;
        WalkInto(engine, RunConstants.NodeRest);

        Assert.Equal(0, engine.Step(RunConstants.RestUpgradeAction, -1, out _, out _, out _));
        int card = engine.State.Deck.FindIndex(RunConstants.IsRunCardUpgradable);
        Assert.Equal(0, engine.Step(card, -1, out _, out _, out _));

        LeaveTheResultPage(engine);
        TheNextRestSiteOffersItsOptions(engine);
    }

    /// <summary>A rest with nothing to offer goes to its result page and on, as before.</summary>
    [Fact]
    public void APlainHealStillEndsOnItsResultPage()
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        engine.State.Phase = RunPhase.Map;
        WalkInto(engine, RunConstants.NodeRest);
        engine.State.PlayerHp = 1;

        Assert.Equal(0, engine.Step(RunConstants.RestHealAction, -1, out _, out _, out _));

        LeaveTheResultPage(engine);
        TheNextRestSiteOffersItsOptions(engine);
    }
}
