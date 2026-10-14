using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// <c>SkittishPower.AfterAttack</c> runs once per <c>AttackCommand</c>, after every hit of it has
/// landed -- not inside the first hit. The block it hands out therefore never protects the
/// rest of the attack that earned it: Twin Strike on a fresh gardener takes both hits as HP.
///
/// <para>
/// It also reads the command's FIRST result against the gardener
/// (<c>Results.SelectMany(...).FirstOrDefault(r => r.Receiver == Owner)</c>) and asks that
/// result for <c>UnblockedDamage != 0</c>. A first hit its own block soaks entirely sets
/// nothing off, however hard the hits after it land.
/// </para>
/// </summary>
public class SkittishAttackCommandTests
{
    private const int Skittish = 7;

    private static Fight GardenerWith(int block, params CardInstance[] hand)
    {
        var fight = Fight.Hand(hand).Energy(9);
        return fight.Enemy(hp: 40, block: block, buffs: new BuffState(BuffId.Skittish, Skittish));
    }

    private static Fight TwoGardeners(params CardInstance[] hand) =>
        Fight
            .Hand(hand)
            .Energy(9)
            .Enemy(hp: 40, buffs: new BuffState(BuffId.Skittish, Skittish))
            .Enemy(hp: 40, buffs: new BuffState(BuffId.Skittish, Skittish));

    private static bool Spent(EnemyState enemy) =>
        BuffSystem.Get(enemy.Buffs, BuffId.SkittishSpent) > 0;

    /// <summary>The mechanism: every hit of a multi-hit attack lands before the block.</summary>
    [Fact]
    public void TwinStrikeTakesBothHitsBeforeTheFlinch()
    {
        var fight = GardenerWith(0, Card(IC.TwinStrike));

        fight.Play(0, target: 0);

        Assert.Equal(40 - 10, fight.Enemy0.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
        Assert.True(Spent(fight.Enemy0));
    }

    /// <summary>
    /// Siblings through the same helper: a single target hit repeatedly, a random target
    /// re-rolled each hit, and every enemy hit repeatedly.
    /// </summary>
    [Theory]
    [InlineData(IC.SwordBoomerang, 3 * 3)]
    [InlineData(IC.Conflagration, 2 * 4)]
    public void EveryHitOfTheAttackLandsBeforeTheFlinch(int cardId, int totalDamage)
    {
        var fight = GardenerWith(0, Card(cardId));

        fight.Play(0, target: 0);

        Assert.Equal(40 - totalDamage, fight.Enemy0.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
        Assert.True(Spent(fight.Enemy0));
    }

    [Fact]
    public void WhirlwindHitsEveryGardenerForEveryHitBeforeEitherFlinches()
    {
        var fight = TwoGardeners(Card(IC.Whirlwind)).Energy(3);

        fight.Play(0);

        Assert.Equal(40 - 15, fight.Enemy0.Hp);
        Assert.Equal(40 - 15, fight.Enemy1.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
        Assert.Equal(Skittish, fight.Enemy1.Block);
    }

    [Fact]
    public void FiendFireTakesEveryHitBeforeTheFlinch()
    {
        var fight = GardenerWith(
            0,
            Card(IC.FiendFire),
            Card(IC.StrikeIronclad),
            Card(IC.DefendIronclad),
            Card(IC.DefendIronclad)
        );

        fight.Play(0, target: 0);

        // Three cards exhausted, seven each.
        Assert.Equal(40 - 21, fight.Enemy0.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
    }

    /// <summary>
    /// Ricochet rolls its own target for every hit in a loop of its own, rather than through the
    /// multi-hit helpers; it is still one command.
    /// </summary>
    [Fact]
    public void RicochetTakesEveryHitBeforeTheFlinch()
    {
        var fight = GardenerWith(0, Card(SI.Ricochet));

        fight.Play(0);

        Assert.Equal(40 - 3 * 4, fight.Enemy0.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
    }

    /// <summary>
    /// Rattle swings Osty once more for every swing before it this turn, as one
    /// <c>WithHitCount</c> attack -- the gardener's block comes after the last of them.
    /// </summary>
    [Fact]
    public void RattleTakesEveryOstySwingBeforeTheFlinch()
    {
        const int Rattle = 382;
        var fight = GardenerWith(0, Card(Rattle));
        fight.State.OstyHp = 1;
        fight.State.OstyMaxHp = 1;
        fight.State.OstyAttacksThisTurn = 1;

        fight.Play(0, target: 0);

        // Two swings of seven.
        Assert.Equal(40 - 14, fight.Enemy0.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
        Assert.True(Spent(fight.Enemy0));
    }

    /// <summary>
    /// Echoing Slash is an <c>AttackContext</c>: its volleys are hits of ONE command, so a gardener
    /// flinches once they are all thrown. The first volley kills the small enemy beside it and
    /// earns a second, which still finds the gardener unshielded.
    /// </summary>
    [Fact]
    public void EchoingSlashFlinchesAfterEveryVolley()
    {
        var fight = Fight
            .Hand(Card(SI.EchoingSlash))
            .Energy(9)
            .Enemy(hp: 40, buffs: new BuffState(BuffId.Skittish, Skittish))
            .Enemy(hp: 5);

        fight.Play(0);

        Assert.Single(fight.State.Enemies);
        Assert.Equal(40 - 20, fight.Enemy0.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
        Assert.True(Spent(fight.Enemy0));
    }

    /// <summary>
    /// Omnislice's splash is a result of the same command (<c>AttackContext.AddHit</c>), with
    /// <c>Unpowered | Move</c> props -- <c>Move</c> is all Skittish asks. A gardener that is
    /// only splashed flinches.
    /// </summary>
    [Fact]
    public void ASplashedGardenerFlinchesToo()
    {
        var fight = TwoGardeners(Card(CL.Omnislice));

        fight.Play(0, target: 0);

        Assert.Equal(40 - 8, fight.Enemy0.Hp);
        Assert.Equal(40 - 8, fight.Enemy1.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
        Assert.Equal(Skittish, fight.Enemy1.Block);
    }

    private const int MadScience = 292;

    private static CardInstance Violent() =>
        new CardInstance(MadScience, false) with
        {
            TinkerType = CardType.Attack,
            TinkerRider = TinkerRider.Violence,
        };

    /// <summary>
    /// Mad Science with Violence is NOT one three-hit command: <c>ExecuteAttack</c> loops three
    /// times over a one-hit <c>DamageCmd.Attack</c>. The gardener flinches after the FIRST of
    /// them, so the second lands on that block (12 against 7: 5 as HP) and the third on none.
    /// </summary>
    [Fact]
    public void MadScienceWithViolenceIsThreeCommandsSoTheSecondHitsTheFlinch()
    {
        var fight = GardenerWith(0, Violent());

        fight.Play(0, target: 0);

        Assert.Equal(40 - 12 - 5 - 12, fight.Enemy0.Hp);
        Assert.Equal(0, fight.Enemy0.Block);
        Assert.True(Spent(fight.Enemy0));
    }

    /// <summary>
    /// Each command reads its OWN first result: a first attack the block soaks whole sets nothing
    /// off, the second attack gets past it and does, and the third pays the price.
    /// </summary>
    [Fact]
    public void MadScienceWithViolenceReadsTheFirstResultOfEachCommand()
    {
        var fight = GardenerWith(12, Violent());

        fight.Play(0, target: 0);

        // Command 1: soaked whole, no flinch. Command 2: 12 as HP, flinch. Command 3: 5 as HP.
        Assert.Equal(40 - 12 - 5, fight.Enemy0.Hp);
        Assert.Equal(0, fight.Enemy0.Block);
        Assert.True(Spent(fight.Enemy0));
    }

    /// <summary>
    /// Without Violence it is one attack of one hit, a command of its own like any other
    /// single-hit card.
    /// </summary>
    [Fact]
    public void MadScienceWithoutViolenceIsOneCommandOfOneHit()
    {
        var fight = GardenerWith(
            0,
            new CardInstance(MadScience, false) with { TinkerType = CardType.Attack }
        );

        fight.Play(0, target: 0);

        Assert.Equal(40 - 12, fight.Enemy0.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
    }

    /// <summary>
    /// The rule reads the FIRST result: a first hit that its block soaks entirely sets
    /// nothing off, and the unblocked second hit does not make up for it.
    /// </summary>
    [Fact]
    public void AFirstHitTheBlockSoaksWholeSetsNothingOffForTheRestOfTheAttack()
    {
        var fight = GardenerWith(5, Card(IC.TwinStrike));

        fight.Play(0, target: 0);

        // 5 absorbed by the block, then 5 as HP.
        Assert.Equal(40 - 5, fight.Enemy0.Hp);
        Assert.Equal(0, fight.Enemy0.Block);
        Assert.False(Spent(fight.Enemy0));
    }

    [Fact]
    public void AFirstHitThatGetsPastTheBlockSetsItOffAfterTheAttack()
    {
        var fight = GardenerWith(3, Card(IC.TwinStrike));

        fight.Play(0, target: 0);

        // 3 absorbed, 2 as HP, then the second hit's 5.
        Assert.Equal(40 - 2 - 5, fight.Enemy0.Hp);
        Assert.Equal(Skittish, fight.Enemy0.Block);
        Assert.True(Spent(fight.Enemy0));
    }

    /// <summary>
    /// Once per turn still holds across commands: the first attack's flinch is the block the
    /// second attack has to chew through, and it adds no more.
    /// </summary>
    [Fact]
    public void ASecondAttackInTheTurnSpendsTheBlockTheFirstGave()
    {
        var fight = GardenerWith(0, Card(IC.TwinStrike), Card(IC.TwinStrike));

        fight.Play(0, target: 0);
        fight.Play(0, target: 0);

        // 7 block absorbs the first hit and 2 of the next: only 3 of it reaches HP.
        Assert.Equal(40 - 10 - 3, fight.Enemy0.Hp);
        Assert.Equal(0, fight.Enemy0.Block);
    }
}
