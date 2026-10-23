// The stale re-anchor perturbation (EmuDump reanchor --stale) must leave card identity alone: it rebuilds the
// cards as plain cards (cost changes, bonus damage and the like are lost) and copies the counters of the combat's
// start, but a card's uid is not a value the game could have drifted from. A uid lost or the uid counter moved back
// would give two cards of the rest of the fight the same uid.
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Xunit;

public class ReanchorStaleUidTests
{
    static List<int> Uids(CombatState s) => CardUids.LiveCards(s).Select(c => c.UidOf()).ToList();

    static (CombatState Start, CombatState Now) AFightWithCardsMadeAfterTheStart()
    {
        var start = CombatFactory.NewCombat(21);
        var now = start.Clone();
        now.Hand.Add(now.Fresh(new CardInstance(now.DrawPile[0].DefId, false)));
        now.Hand.Add(now.Fresh(new CardInstance(now.DrawPile[1].DefId, false)));
        now.Hand[0] = now.Hand[0] with { CostForCombat = 0, BonusDamage = 3 };  // the kind of change "cards" undoes
        return (start, now);
    }

    [Theory]
    [InlineData("all")]
    [InlineData("cards")]
    [InlineData("counters")]
    public void StalePerturbationKeepsEveryCardUidAndTheCounter(string group)
    {
        var (start, now) = AFightWithCardsMadeAfterTheStart();
        var uids = Uids(now);
        int counter = now.LastCardUid;
        Assert.True(counter > start.LastCardUid);

        OldReanchor.GoStale(now, start, new HashSet<string> { group });

        Assert.Equal(uids, Uids(now));
        Assert.Equal(counter, now.LastCardUid);
        Assert.True(CardUidInvariant.Holds(now));
        int next = now.NewCardUid();
        Assert.DoesNotContain(next, uids);
    }

    [Fact]
    public void StaleCardsAreStillPlainCards()
    {
        var (start, now) = AFightWithCardsMadeAfterTheStart();

        OldReanchor.GoStale(now, start, new HashSet<string> { "cards" });

        Assert.All(CardUids.LiveCards(now), c => Assert.Equal(new CardInstance(c.DefId, c.Upgraded), c));
        Assert.Equal(int.MinValue, now.Hand[0].CostForCombat);  // unset
    }
}
