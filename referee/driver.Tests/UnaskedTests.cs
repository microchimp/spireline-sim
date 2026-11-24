using System.Text.Json.Nodes;
using Spireline.Referee;
using Xunit;

namespace Spireline.Referee.Tests;

// The lone fight actions the driver takes without asking (UnaskedLog): what the lockstep is told with the next asked decision.
public class UnaskedTests
{
    [Fact]
    public void ALoneEndTurnInAFightIsListedWithItsKind()
    {
        var log = new UnaskedLog();
        log.Took("combat", new JsonArray("end"));
        Assert.Equal("[[\"combat\",[\"end\"]]]", log.Drain().ToJsonString());
    }

    [Fact]
    public void TheListGoesOutOnceAndStartsEmptyAgain()
    {
        // Two turns passed without a question (nothing playable twice), then a lone card on a fight's card screen: all three,
        // oldest first, with the next asked decision; the decision after it is told of none.
        var log = new UnaskedLog();
        log.Took("combat", new JsonArray("end"));
        log.Took("combat", new JsonArray("end"));
        log.Took("select_cards_combat", new JsonArray("STRIKE_IRONCLAD", 0));
        Assert.Equal(3, log.Count);
        Assert.Equal("[[\"combat\",[\"end\"]],[\"combat\",[\"end\"]],[\"select_cards_combat\",[\"STRIKE_IRONCLAD\",0]]]",
            log.Drain().ToJsonString());
        Assert.Equal("[]", log.Drain().ToJsonString());
    }

    [Fact]
    public void ADecisionOutsideAFightIsNotListed()
    {
        // A lone map node, a lone event option, a rewards screen with one item: the copy's own walk mirrors those.
        var log = new UnaskedLog();
        foreach (var kind in new[] { "map", "event", "reward", "card_reward", "select_cards", "rest", "shop" })
            log.Took(kind, new JsonArray("x"));
        log.Took("card_reward_combat", new JsonArray("ANGER", 1));
        Assert.Equal("[[\"card_reward_combat\",[\"ANGER\",1]]]", log.Drain().ToJsonString());
    }

    [Fact]
    public void TheLabelIsACopyAndMayBeMissing()
    {
        // The label still belongs to the decision's own action list: the log keeps a copy, so a later change to the list
        // (or the list going out in a message of its own) leaves the log as it was.
        var actions = new JsonArray(new JsonArray("end"));
        var log = new UnaskedLog();
        log.Took("combat", actions[0]);
        actions[0]!.AsArray().Add("changed");
        log.Took("combat", null);  // the action list could not be built: the step is still counted
        Assert.Equal("[[\"combat\",[\"end\"]],[\"combat\",null]]", log.Drain().ToJsonString());
    }

    [Fact]
    public void ANewRunStartsWithNothing()
    {
        var log = new UnaskedLog();
        log.Took("combat", new JsonArray("end"));
        log.Clear();
        Assert.Equal(0, log.Count);
        Assert.Equal("[]", log.Drain().ToJsonString());
    }

    [Fact]
    public void TheFightKindsAreTheLockstepsCombatKinds()
    {
        // referee/lockstep.py COMBAT_KINDS = ("combat", "select_cards_combat", "card_reward_combat")
        Assert.True(UnaskedLog.FightKind("combat"));
        Assert.True(UnaskedLog.FightKind("select_cards_combat"));
        Assert.True(UnaskedLog.FightKind("card_reward_combat"));
        Assert.False(UnaskedLog.FightKind("select_cards"));
        Assert.False(UnaskedLog.FightKind(null));
    }
}
