using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.GeneratedData;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.Audit;

/// <summary>Static audit pilot, potions (and the shared helpers they ride on).</summary>
public class StaticAuditPotionTests
{
    private const int DistilledChaos = 15;

    /// <summary>
    /// CardPileCmd.AutoPlayFromDrawPile takes ALL `count` cards off the top first (moving each to
    /// the Play pile) and only then plays them, so a draw made by the first card cannot pick up
    /// the second. Pommel Strike draws the card BELOW the three taken.
    /// </summary>
    [Fact]
    public void DistilledChaosTakesAllThreeBeforePlayingAny()
    {
        var fight = Fight
            .Hand()
            .Draw(
                Card(IC.PommelStrike),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.Bash)
            )
            .Enemy(hp: 200);

        PotionEffects.Apply(DistilledChaos, fight.State, new Random(0));

        // Pommel Strike, Defend, Defend were played; Pommel's draw took the fourth card.
        Assert.Equal([IC.Bash], Fight.Ids(fight.State.Hand));
    }

    /// <summary>
    /// ShuffleIfNecessary runs before every take: a draw pile that runs dry pulls the discard
    /// pile back in and the potion still plays three cards.
    /// </summary>
    [Fact]
    public void DistilledChaosReshufflesWhenTheDrawPileRunsDry()
    {
        var fight = Fight
            .Hand()
            .Draw(Card(IC.DefendIronclad))
            .Discard(
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad)
            )
            .Enemy(hp: 200);
        fight.State.ShuffleRng = new CountingRandom(0);

        PotionEffects.Apply(DistilledChaos, fight.State, new Random(0));

        Assert.Equal(15, fight.State.PlayerBlock);
    }

    /// <summary>
    /// Havoc is `AutoPlayFromDrawPile(1, Top, forceExhaust: true)`, so an EMPTY draw pile with a
    /// full discard shuffles first and plays the top card. The emulator's Havoc case does nothing
    /// when the draw pile is empty.
    /// </summary>
    [Fact]
    public void HavocShufflesAnEmptyDrawPileFirst()
    {
        var fight = Fight
            .Hand(Card(IC.Havoc))
            .Draw()
            .Discard(Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .Energy(3)
            .Enemy(hp: 200);
        fight.State.ShuffleRng = new CountingRandom(0);

        fight.Play();

        Assert.Single(fight.State.ExhaustPile);
        Assert.Equal(5, fight.State.PlayerBlock);
    }

    /// <summary>
    /// ShouldDie (Fairy in a Bottle) is asked of EVERY listener before ShouldDieLate (Lizard
    /// Tail) is asked of any, so the potion goes first whatever else is owned, and Lizard Tail
    /// stays unused. The emulator revives with Lizard Tail inside the HP-change hook and only
    /// asks the potion at the end of the step.
    /// </summary>
    [AuditFact("Fairy in a Bottle is spent before Lizard Tail; the engine fix (audit/static-pilot 3184f17) is not merged, only the pilot's tools and tests")]
    public void FairyInABottleIsSpentBeforeLizardTail()
    {
        var fight = Fight.WithRelics(RelicEffects.LizardTail).Enemy(hp: 200).PlayerHp(1, maxHp: 80);
        fight.State.PotionSlots[0] = Potions.FindId("FairyInABottle")!.Value;

        CardEffects.LoseHp(fight.State, 5);

        Assert.Equal(24, fight.State.PlayerHp);
        Assert.Equal(0, fight.State.PotionSlots[0]);
        // Lizard Tail is untouched, so it can still save the run later.
        CardEffects.LoseHp(fight.State, 100);
        Assert.Equal(40, fight.State.PlayerHp);
    }

    /// <summary>
    /// The prevented death happens AT the lethal hit (CreatureCmd.Kill, ShouldDie false,
    /// AfterPreventingDeath heals), so the next hit of the same attack lands on the revived
    /// player: 24 - 5. The emulator revives at the end of the step, after every hit.
    /// </summary>
    [AuditFact("Fairy in a Bottle revives at the lethal hit, before the next one; the engine fix (audit/static-pilot 3184f17) is not merged, only the pilot's tools and tests")]
    public void FairyInABottleRevivesBeforeTheNextHitLands()
    {
        var fight = Fight.Hand().Enemy(hp: 200).PlayerHp(1, maxHp: 80);
        fight.State.PotionSlots[0] = Potions.FindId("FairyInABottle")!.Value;

        CardEffects.LoseHp(fight.State, 5);
        CardEffects.LoseHp(fight.State, 5);

        Assert.Equal(19, fight.State.PlayerHp);
    }

    /// <summary>
    /// Ashwater is ONE multi-select (CardSelectorPrefs 0..max over the hand as it was when the
    /// screen opened) and the exhausts run afterwards, so a card Dark Embrace draws for the first
    /// exhaust is never a candidate. The emulator exhausts per pick and reopens the screen over
    /// the CURRENT hand, drawn card included.
    /// </summary>
    [AuditFact("Ashwater's per-pick screen offers cards drawn by Dark Embrace mid-selection")]
    public void AshwaterNeverOffersACardDrawnDuringItsOwnSelection()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .Draw(Card(IC.Bash), Card(IC.Bash))
            .Enemy(hp: 200)
            .PlayerBuff(BuffId.DarkEmbrace, 1);

        PotionEffects.Apply(1, fight.State, new Random(0));
        fight.Choose(0);

        // Two of the original three are left to pick from; the drawn Bash is not one of them.
        Assert.Equal(2, fight.Pending!.Candidates.Count);
    }
}
