using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CardModel.OnPlayWrapper` plays a card `playCount` times (`GeneratePlayCount`: Duplication,
/// Burst, Echo Form, One-Two Punch, Spiral, a Replay), and each time is a whole play:
/// `Hook.BeforeCardPlayed`, `CombatHistory.CardPlayStarted`, `OnPlay`, the enchantment's
/// `OnPlay`, `CardPlayFinished`, `Hook.AfterCardPlayed`. The card reaches its result pile once,
/// after the last. Only Echo Form (`IsFirstInSeries`), Void Form and Black Hole
/// (`IsLastInSeries`) and Lethality (`CurrentPlayIndex`) tell the repeats apart.
/// </summary>
public class ReplayedPlayTests
{
    private static CardInstance Card(int id, bool upgraded = false) => new(id, upgraded);

    private static Fight Duplicated(params CardInstance[] hand) =>
        Fight.Hand(hand).Enemy(hp: 100).PlayerBuff(BuffId.Duplication, 1).Energy(10);

    /// <summary>`RagePower.AfterCardPlayed` runs after each play of a Duplicated Strike.</summary>
    [Fact]
    public void RagePaysForEveryPlayOfADuplicatedAttack()
    {
        var fight = Duplicated(Card(IC.StrikeIronclad)).PlayerBuff(BuffId.Rage, 3);

        fight.Play(0);

        Assert.Equal(100 - 2 * 6, fight.State.Enemies[0].Hp);
        Assert.Equal(2 * 3, fight.State.PlayerBlock);
    }

    /// <summary>
    /// `Kunai.AfterCardPlayed` counts each play: a Duplicated Strike and one more Strike are
    /// three attacks.
    /// </summary>
    [Fact]
    public void KunaiCountsBothPlaysOfADuplicatedAttack()
    {
        var fight = Duplicated(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad));
        fight.State.Relics.Add(new RelicInstance(RelicEffects.Kunai));

        fight.Play(0);
        fight.Play(0);

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Dexterity));
    }

    /// <summary>
    /// `PenNib.BeforeCardPlayed` counts each play and doubles the tenth: at eight, the first
    /// play of a Duplicated Strike is the ninth attack and its repeat the tenth.
    /// </summary>
    [Fact]
    public void PenNibDoublesTheRepeatThatIsTheTenthAttack()
    {
        var fight = Duplicated(Card(IC.StrikeIronclad));
        fight.State.Relics.Add(new RelicInstance(RelicEffects.PenNib, 8));

        fight.Play(0);

        Assert.Equal(100 - 6 - 12, fight.State.Enemies[0].Hp);
        Assert.Equal(0, fight.State.Relics.Single(r => r.DefId == RelicEffects.PenNib).Counter);
    }

    /// <summary>
    /// `JugglingPower.AfterCardPlayed` copies the third attack: a Duplicated Strike is the
    /// first two, so the next Strike is copied.
    /// </summary>
    [Fact]
    public void JugglingCopiesTheThirdPlayCountingRepeats()
    {
        var fight = Duplicated(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad))
            .PlayerBuff(BuffId.Juggling, 1);

        fight.Play(0);
        fight.Play(0);

        Assert.Single(fight.State.Hand, card => card.DefId == IC.StrikeIronclad);
    }

    /// <summary>
    /// `VelvetChoker.ShouldPlay` refuses a card once six plays have finished this turn, and a
    /// Duplicated Defend is two of them.
    /// </summary>
    [Fact]
    public void VelvetChokerCountsTheRepeat()
    {
        var fight = Duplicated([.. Enumerable.Repeat(Card(IC.DefendIronclad), 6)]);
        fight.State.Relics.Add(new RelicInstance(RelicEffects.VelvetChoker));

        for (int i = 0; i < 5; i++)
        {
            fight.Play(0);
        }

        fight.Play(0);

        Assert.Single(fight.State.Hand);
        Assert.Equal(6 * 5, fight.State.PlayerBlock);
    }

    /// <summary>
    /// `AfterimagePower` records its amount in `BeforeCardPlayed` and pays it in
    /// `AfterCardPlayed`, once for each play.
    /// </summary>
    [Fact]
    public void AfterimagePaysForEveryPlay()
    {
        var fight = Duplicated(Card(IC.DefendIronclad)).PlayerBuff(BuffId.Afterimage, 1);

        fight.Play(0);

        Assert.Equal(2 * (5 + 1), fight.State.PlayerBlock);
    }

    /// <summary>
    /// The enchantment's `OnPlay` is part of each play: Corrupted costs 2 HP every time.
    /// </summary>
    [Fact]
    public void CorruptedCostsHpOnEveryPlay()
    {
        var fight = Duplicated(
            Card(IC.StrikeIronclad) with
            {
                Enchantment = Enchantment.Corrupted,
                EnchantAmount = 1,
            }
        );
        int hp = fight.State.PlayerHp;

        fight.Play(0);

        Assert.Equal(hp - 2 * 2, fight.State.PlayerHp);
    }

    /// <summary>
    /// The repeat plays the card as the first play left it: `Sown.OnPlay` set it Disabled, so
    /// it pays its energy once.
    /// </summary>
    [Fact]
    public void SownPaysOnceAcrossTheRepeats()
    {
        var fight = Duplicated(
            Card(IC.StrikeIronclad) with
            {
                Enchantment = Enchantment.Sown,
                EnchantAmount = 1,
            }
        );

        fight.Play(0);

        Assert.Equal(10 - 1 + 1, fight.State.Energy);
        Assert.True(fight.State.DiscardPile.Single().EnchantSpent);
    }

    /// <summary>
    /// Rampage raises its own damage in `OnPlay`, so its repeat hits for the raised amount, and
    /// the copy reaches the discard pile with both raises.
    /// </summary>
    [Fact]
    public void RampagesRepeatHitsForTheRaisedDamage()
    {
        var fight = Duplicated(Card(IC.Rampage));

        fight.Play(0);

        Assert.Equal(100 - 9 - 14, fight.State.Enemies[0].Hp);
        Assert.Equal(10, fight.State.DiscardPile.Single().BonusDamage);
    }

    /// <summary>
    /// `EchoFormPower` counts only the `IsFirstInSeries` plays this turn: with two stacks, the
    /// second card is doubled too, though the first card already started two plays.
    /// </summary>
    [Fact]
    public void EchoFormCountsCardsNotRepeats()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad))
            .Enemy(hp: 100)
            .PlayerBuff(BuffId.EchoForm, 2)
            .Energy(10);

        fight.Play(0);
        fight.Play(0);

        Assert.Equal(100 - 4 * 6, fight.State.Enemies[0].Hp);
    }

    /// <summary>
    /// Every repetition's `CardPlay.Target` is `OnPlayWrapper`'s `target`, the creature, not
    /// its slot: Panache's burst after the first play kills the enemy before it, whose corpse
    /// leaves the roster, and the repeat still hits the creature aimed at (the middle one, or
    /// the last, whose slot is then past the end).
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void TheRepeatHitsTheCreatureAimedAtAfterAnEarlierEnemyDies(int aimedAt)
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Enemy(hp: 5)
            .Enemy(hp: 50)
            .Enemy(hp: 50)
            .PlayerBuff(BuffId.Duplication, 1)
            .PlayerBuff(BuffId.PanachePower, 10, data: 4)
            .Energy(10);
        var target = fight.State.Enemies[aimedAt];
        var other = fight.State.Enemies[3 - aimedAt];

        fight.Play(0, target: aimedAt);

        Assert.Equal(2, fight.State.Enemies.Count);
        Assert.Equal(50 - 6 - 10 - 6, target.Hp);
        Assert.Equal(50 - 10, other.Hp);
    }

    /// <summary>
    /// `VoidFormPower.AfterCardPlayed` counts a play only when it `IsLastInSeries`: a
    /// Duplicated card is one of its free cards, so the next one is free too.
    /// </summary>
    [Fact]
    public void VoidFormCountsADuplicatedCardOnce()
    {
        var fight = Duplicated(Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .PlayerBuff(BuffId.VoidForm, 2);

        fight.Play(0);
        fight.Play(0);

        Assert.Equal(10, fight.State.Energy);
    }
}
