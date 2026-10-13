using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CardPileCmd.AutoPlayFromDrawPile(count, Top, forceExhaust)`, which Havoc (1, forced),
/// Cascade, Mayhem, Distilled Chaos and I Am Invincible all call: take ALL the cards first,
/// shuffling the discard in whenever the draw pile runs dry, then `CardCmd.AutoPlay` each in
/// order. Referee night03 6GWSWCM3F3 (Vigor left after a Havoc-played attack); night04
/// 499FGPTCUV (a Havoc-played Panache exhausted) and UA0TKYQC82 (a Havoc-played Demonic
/// Shield played with no ally).
/// </summary>
public class AutoPlayDrawPileTests
{
    private const int DistilledChaos = 15;
    private const int DemonicShield = 142;
    private const int Panache = 342;
    private const int Cascade = 546;

    private static CardInstance Card(int id) => new(id, false);

    private static CountingRandom TargetStream(Fight fight, int seed = 7)
    {
        var targets = new CountingRandom(seed);
        fight.State.TargetRng = targets;
        return targets;
    }

    /// <summary>The seed whose first two-way target roll lands on the SECOND enemy.</summary>
    private static int SeedRollingTheSecondEnemy() =>
        Enumerable.Range(1, 100).First(seed => new CountingRandom(seed).Next(2) == 1);

    public static IEnumerable<object[]> Sources() =>
        [["Havoc"], ["Cascade"], ["DistilledChaos"], ["Mayhem"]];

    private static void Fire(Fight fight, string source)
    {
        switch (source)
        {
            case "Havoc":
                fight.State.Hand = [Card(IC.Havoc)];
                fight.Energy(3);
                fight.Play(0);
                break;
            case "Cascade":
                fight.State.Hand = [Card(Cascade)];
                fight.Energy(1);
                fight.Play(0);
                break;
            case "DistilledChaos":
                fight.State.PotionSlots[0] = DistilledChaos;
                fight.Potion(0);
                break;
            case "Mayhem":
                fight.PlayerBuff(BuffId.MayhemPower, 1);
                fight.EndTurn();
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(source));
        }
    }

    [Theory]
    [MemberData(nameof(Sources))]
    public void AnAutoPlayedAnyEnemySkillRollsOnceAndAimsAtTheRolledEnemy(string source)
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Enemy(hp: 60)
            .Enemy(hp: 60)
            // Mayhem plays AFTER the turn's five-card draw, so the Taunt sits behind them.
            .Draw(
                [
                    .. Enumerable.Repeat(Card(IC.DefendIronclad), source == "Mayhem" ? 5 : 0),
                    Card(IC.Taunt),
                    .. Enumerable.Repeat(Card(IC.DefendIronclad), 7),
                ]
            );
        var targets = TargetStream(fight, SeedRollingTheSecondEnemy());

        Fire(fight, source);

        Assert.Equal(1, targets.CallCount);
        Assert.Equal(0, BuffSystem.Get(fight.State.Enemies[0].Buffs, BuffId.Vulnerable));
        Assert.True(BuffSystem.Get(fight.State.Enemies[1].Buffs, BuffId.Vulnerable) > 0);
    }

    [Theory]
    [InlineData(IC.AscendersBane)]
    [InlineData(ST.Wound)]
    public void AHavocPlayedUnplayableCardIsExhaustedWithoutBeingPlayed(int unplayable)
    {
        var fight = Fight.Hand(Card(IC.Havoc)).Enemy(hp: 60).Draw(Card(unplayable), Card(IC.StrikeIronclad));
        var targets = TargetStream(fight);
        fight.Energy(3);

        fight.Play(0);

        Assert.Equal(0, targets.CallCount);
        Assert.Contains(fight.State.ExhaustPile, card => card.DefId == unplayable);
        Assert.Equal(1, fight.State.CardPlaysThisTurn);
    }

    [Theory]
    [InlineData("Cascade")]
    [InlineData("DistilledChaos")]
    public void AnAutoPlayedUnplayableCardGoesToTheDiscardWithoutBeingPlayed(string source)
    {
        var fight = Fight.Hand(Card(IC.DefendIronclad)).Enemy(hp: 60).Draw(Card(ST.Wound));
        TargetStream(fight);

        Fire(fight, source);

        Assert.Contains(fight.State.DiscardPile, card => card.DefId == ST.Wound);
        // Cascade's own play is the only one counted.
        Assert.Equal(source == "Cascade" ? 1 : 0, fight.State.CardPlaysThisTurn);
    }

    /// <summary>
    /// 499FGPTCUV: `GetResultPileTypeForCardPlay` sends a Power to no pile before it reads
    /// `ExhaustOnNextPlay`, so the Panache Havoc plays is not in the exhaust pile.
    /// </summary>
    [Fact]
    public void AHavocPlayedPowerLeavesCombatRatherThanBeingExhausted()
    {
        var fight = Fight.Hand(Card(IC.Havoc)).Enemy(hp: 60).Draw(Card(Panache), Card(IC.StrikeIronclad));
        fight.Energy(3);

        fight.Play(0);

        Assert.True(fight.PlayerBuffAmount(BuffId.PanachePower) > 0);
        Assert.DoesNotContain(fight.State.ExhaustPile, card => card.DefId == Panache);
        Assert.DoesNotContain(fight.State.DiscardPile, card => card.DefId == Panache);
    }

    /// <summary>
    /// UA0TKYQC82: Demonic Shield is AnyAlly; a solo run has no ally to roll, so the card is
    /// filed away -- into the exhaust pile, Havoc having forced it -- and costs no HP.
    /// </summary>
    [Fact]
    public void AHavocPlayedAllyCardIsExhaustedUnplayed()
    {
        var fight = Fight.Hand(Card(IC.Havoc)).Enemy(hp: 60).Draw(Card(DemonicShield), Card(IC.StrikeIronclad));
        var targets = TargetStream(fight);
        fight.Energy(3);
        fight.State.PlayerBlock = 5;
        int hp = fight.State.PlayerHp;

        fight.Play(0);

        Assert.Equal(0, targets.CallCount);
        Assert.Equal(hp, fight.State.PlayerHp);
        Assert.Equal(5, fight.State.PlayerBlock);
        Assert.Contains(fight.State.ExhaustPile, card => card.DefId == DemonicShield);
    }

    /// <summary>
    /// An attack Havoc plays is an attack played: `VigorPower` is spent on it (6GWSWCM3F3).
    /// </summary>
    [Fact]
    public void AHavocPlayedAttackSpendsVigor()
    {
        var fight = Fight.Hand(Card(IC.Havoc)).Enemy(hp: 60).Draw(Card(IC.StrikeIronclad));
        TargetStream(fight);
        fight.PlayerBuff(BuffId.Vigor, 4);
        fight.Energy(3);

        fight.Play(0);

        Assert.Equal(60 - (6 + 4), fight.State.Enemies[0].Hp);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Vigor));
    }

    [Fact]
    public void HavocShufflesTheDiscardPileInWhenTheDrawPileIsEmpty()
    {
        var fight = Fight.Hand(Card(IC.Havoc)).Enemy(hp: 60).Draw().Discard(Card(IC.StrikeIronclad));
        TargetStream(fight);
        fight.Energy(3);

        fight.Play(0);

        Assert.Equal(60 - 6, fight.State.Enemies[0].Hp);
        Assert.Contains(fight.State.ExhaustPile, card => card.DefId == IC.StrikeIronclad);
        Assert.DoesNotContain(fight.State.DiscardPile, card => card.DefId == IC.StrikeIronclad);
    }

    [Fact]
    public void DistilledChaosShufflesTheDiscardPileInWhenTheDrawPileIsEmpty()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad))
            .Enemy(hp: 60)
            .Draw()
            .Discard(Card(IC.DefendIronclad), Card(IC.DefendIronclad), Card(IC.DefendIronclad));
        TargetStream(fight);

        Fire(fight, "DistilledChaos");

        Assert.Equal(15, fight.State.PlayerBlock);
    }

    /// <summary>
    /// All three cards are taken BEFORE any plays, so what Battle Trance draws comes from
    /// behind them.
    /// </summary>
    [Fact]
    public void DistilledChaosTakesAllThreeCardsBeforePlayingAny()
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Enemy(hp: 60)
            .Draw(
                Card(IC.BattleTrance),
                Card(IC.DefendIronclad),
                Card(IC.DefendIronclad),
                Card(IC.StrikeIronclad),
                Card(IC.StrikeIronclad),
                Card(IC.StrikeIronclad)
            );
        TargetStream(fight);

        Fire(fight, "DistilledChaos");

        Assert.Equal(
            [IC.DefendIronclad, IC.StrikeIronclad, IC.StrikeIronclad, IC.StrikeIronclad],
            Fight.Ids(fight.State.Hand)
        );
        Assert.Equal(10, fight.State.PlayerBlock);
        Assert.Empty(fight.State.DrawPile);
    }

    [Fact]
    public void AnAutoPlayRunStopsOnceTheFightIsOver()
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Enemy(hp: 6)
            .Draw(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad), Card(IC.StrikeIronclad));
        TargetStream(fight);

        Fire(fight, "DistilledChaos");

        Assert.Equal(1, fight.State.CardPlaysThisTurn);
    }
}
