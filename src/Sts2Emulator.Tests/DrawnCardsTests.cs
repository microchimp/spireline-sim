using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// Pillage, Escape Plan and Scrape read `CardPileCmd.Draw`'s result: the cards that draw took off
/// the pile. The emulator read the end of the hand, which is also whatever else reached the hand
/// during the draw -- Stratagem's pick, answered in the shuffle the draw ran into -- and misses a
/// card that was drawn and did not stay (a Hellraiser Strike, played from the draw).
/// </summary>
public class DrawnCardsTests
{
    private static List<int> Ids(List<CardInstance> pile) => [.. pile.Select(c => c.DefId)];

    [Fact]
    public void EscapePlanWhoseDrawFoundAFullHandGainsNothingForThePick()
    {
        // Nine in hand with Escape Plan, an empty pile: the shuffle asks, the pick (a Defend)
        // fills the hand to ten, and the draw takes nothing. The emulator judged the pick.
        var fight = Fight
            .Hand([Card(SI.EscapePlan), .. Pile(IC.Bash, IC.Bash, IC.Bash, IC.Bash, IC.Bash, IC.Bash, IC.Bash, IC.Bash, IC.Bash)])
            .Energy(3)
            .Draw()
            .Discard([.. Pile(IC.DefendIronclad, IC.Bash)])
            .Enemy(hp: 500)
            .PlayerBuff(BuffId.StratagemPower, 1);

        fight.Play();
        Assert.Equal(CL.Stratagem, fight.Pending!.SourceCardDefId);
        fight.Choose(Ids(fight.State.DrawPile).IndexOf(IC.DefendIronclad));

        Assert.Null(fight.Pending);
        Assert.Equal(10, fight.State.Hand.Count);
        Assert.Equal(0, fight.State.PlayerBlock);
    }

    [Fact]
    public void ScrapeDiscardsWhatItDrewNotThePick()
    {
        // Scrape draws four: the Defend, then the shuffle asks and Bash is picked, then the two
        // Angers. It discards the costly ones it DREW (the Defend); the pick stays.
        var fight = Fight
            .Hand(Card(GeneratedData.Cards.FindId("Scrape")!.Value))
            .Energy(3)
            .Draw(Card(IC.DefendIronclad))
            .Discard([.. Pile(IC.Bash, IC.Anger, IC.Anger)])
            .Enemy(hp: 500)
            .PlayerBuff(BuffId.StratagemPower, 1);

        fight.Play();
        Assert.Equal(CL.Stratagem, fight.Pending!.SourceCardDefId);
        fight.Choose(Ids(fight.State.DrawPile).IndexOf(IC.Bash));

        Assert.Null(fight.Pending);
        Assert.Equal([IC.Bash, IC.Anger, IC.Anger], Ids(fight.State.Hand));
        Assert.Contains(fight.State.DiscardPile, c => c.DefId == IC.DefendIronclad);
        Assert.DoesNotContain(fight.State.DiscardPile, c => c.DefId == IC.Bash);
    }

    [Fact]
    public void PillageKeepsDrawingPastAStrikeHellraiserPlays()
    {
        // `Draw` returns the Strike although Hellraiser plays it from the draw: an Attack, so
        // Pillage draws on, through Anger, to the Defend. The emulator saw no new card in hand
        // and stopped at the Strike.
        var fight = Fight
            .Hand(Card(IC.Pillage))
            .Energy(3)
            .Draw([.. Pile(IC.StrikeIronclad, IC.Anger, IC.DefendIronclad, IC.Bash)])
            .Enemy(hp: 500)
            .PlayerBuff(BuffId.Hellraiser, 1);

        fight.Play();

        Assert.Equal([IC.Anger, IC.DefendIronclad], Ids(fight.State.Hand));
        Assert.Equal([IC.Bash], Ids(fight.State.DrawPile));
    }
}
