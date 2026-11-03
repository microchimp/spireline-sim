using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// What ends Pillage's draw (`do Draw while the card is an Attack and the hand is not full`) when
/// the piles hold nothing but Attacks: the hand limit, or a draw that finds both piles empty, or the
/// fight ending. Hellraiser takes the drawn Strikes out of the hand again (it plays them to the
/// discard), so with Hellraiser on and Strikes in the piles the hand fills only once the Strikes kill
/// -- or once Hellraiser stops playing them. It stops (`HellraiserPower.AfterCardDrawnEarly`) after
/// nine in a turn while every hittable enemy shows infinite HP: a Hardened Shell that has taken its
/// fill this turn, a Waterfall Giant waiting to blow. The emulator had no such count and drew for
/// ever in exactly those two fights (the three seeds of 2026-11-02: one Skulking Colony, two Waterfall
/// Giants, each a Pillage with Hellraiser on and only Attacks left to draw).
/// </summary>
public class HellraiserInfiniteHpTests
{
    private const int GiantWaitingToBlowHp = 999_999_999;

    private static int HellraiserCount(Fight fight) =>
        fight.State.PlayerBuffs.Single(buff => buff.Id == BuffId.Hellraiser).Data;

    /// <summary>Pillage and five Defends; eight Strikes to draw and four in the discard.</summary>
    private static Fight PillageIntoStrikes() =>
        Fight
            .Hand([Card(IC.Pillage), .. Pile(IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad)])
            .Energy(3)
            .Draw([.. Enumerable.Repeat(IC.StrikeIronclad, 8).Select(id => Card(id))])
            .Discard([.. Enumerable.Repeat(IC.StrikeIronclad, 4).Select(id => Card(id))]);

    /// <summary>
    /// The Skulking Colony's case. Its shell has taken 20 of 20 this turn, so no Strike takes HP
    /// off it. Pillage draws eight Strikes Hellraiser plays, the shuffle of those and the four,
    /// one more played -- the ninth -- and then five Strikes that stay, which fill the hand.
    /// </summary>
    [Fact]
    public void PillageWithHellraiserEndsAgainstAShellThatHasTakenItsFill()
    {
        var fight = PillageIntoStrikes()
            .Enemy(hp: 100, buffs: [new BuffState(BuffId.HardenedShell, 20)])
            .PlayerBuff(BuffId.Hellraiser, 1);
        fight.Enemy0.HardenedShellDamageThisTurn = 20;

        fight.Play(0, 0);

        Assert.Equal(100, fight.Enemy0.Hp);
        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal(5, fight.State.Hand.Count(card => card.DefId == IC.StrikeIronclad));
        Assert.Equal(6, fight.State.DrawPile.Count);
        // Nine played, five kept: each one drawn is counted.
        Assert.Equal(14, HellraiserCount(fight));
        Assert.Equal(1, fight.State.DiscardPile.Count(card => card.DefId == IC.StrikeIronclad));
    }

    /// <summary>
    /// The Waterfall Giant's case. Pillage's own hit kills it, Steam Eruption stands it back up at
    /// 999,999,999 HP to wait to blow, and from then on its HP shows infinite: nine Strikes are
    /// played into it, and the rest stay in hand until the hand is full.
    /// </summary>
    [Fact]
    public void PillageWithHellraiserEndsAgainstAGiantWaitingToBlow()
    {
        var fight = PillageIntoStrikes()
            .Enemy(hp: 6, maxHp: 250, defId: KE.WaterfallGiant, buffs: [new BuffState(BuffId.SteamEruption, 20)])
            .PlayerBuff(BuffId.Hellraiser, 1);

        fight.Play(0, 0);

        Assert.Equal(GiantWaitingToBlowHp - 9 * 6, fight.Enemy0.Hp);
        Assert.True(EnemyAI.ShowsInfiniteHp(fight.Enemy0));
        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal(5, fight.State.Hand.Count(card => card.DefId == IC.StrikeIronclad));
        Assert.Equal(14, HellraiserCount(fight));
    }

    /// <summary>
    /// One enemy that can still lose HP is enough: every Strike is played (the second enemy takes
    /// them, or the shell takes nothing), and the count starts again from zero.
    /// </summary>
    [Fact]
    public void HellraiserPlaysEveryStrikeWhileAnEnemyCanLoseHp()
    {
        var fight = Fight
            .Hand([Card(IC.Pillage), .. Pile(IC.DefendIronclad, IC.DefendIronclad)])
            .Energy(3)
            .Draw([.. Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad)])
            .Discard()
            .Enemy(hp: 100, buffs: [new BuffState(BuffId.HardenedShell, 20)])
            .Enemy(hp: 200)
            .PlayerBuff(BuffId.Hellraiser, 1, data: 9);
        fight.Enemy0.HardenedShellDamageThisTurn = 20;

        fight.Play(0, 0);

        Assert.Equal([IC.DefendIronclad, IC.DefendIronclad, IC.DefendIronclad], Fight.Ids(fight.State.Hand));
        Assert.Equal(3, fight.State.DiscardPile.Count(card => card.DefId == IC.StrikeIronclad));
        Assert.Equal(0, HellraiserCount(fight));
    }

    /// <summary>`HellraiserPower.AfterSideTurnEnd`: the player's turn end clears the count.</summary>
    [Fact]
    public void HellraiserCountStartsAgainWhenThePlayersTurnEnds()
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Draw()
            .Discard()
            .Enemy(hp: 100, buffs: [new BuffState(BuffId.HardenedShell, 20)])
            .PlayerBuff(BuffId.Hellraiser, 1, data: 9);
        fight.Enemy0.HardenedShellDamageThisTurn = 20;

        fight.EndTurn();

        Assert.Equal(0, HellraiserCount(fight));
    }

    /// <summary>
    /// The draw itself, with no Hellraiser: only Attacks left, fewer than the hand has room for.
    /// Pillage draws them all, the discard's through a shuffle, and stops when a draw finds both
    /// piles empty (`CardPileCmd.Draw` returns nothing).
    /// </summary>
    [Fact]
    public void PillageStopsWhenThePilesRunOutOfAttacks()
    {
        var fight = Fight
            .Hand(Card(IC.Pillage))
            .Energy(3)
            .Draw([.. Pile(IC.StrikeIronclad, IC.Bash)])
            .Discard([.. Pile(IC.Anger)])
            .Enemy(hp: 100);

        fight.Play(0, 0);

        Assert.Equal([IC.StrikeIronclad, IC.Bash, IC.Anger], Fight.Ids(fight.State.Hand));
        Assert.Empty(fight.State.DrawPile);
        Assert.Equal([IC.Pillage], Fight.Ids(fight.State.DiscardPile));
    }
}
