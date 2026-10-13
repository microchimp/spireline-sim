using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CardCmd.AutoPlay` rolls `Rng.CombatTargets.NextItem(HittableEnemies)` for a card whose
/// TargetType is `AnyEnemy` and that was handed no target -- whichever of the auto-play
/// sources (Stampede, Hellraiser, Mayhem, Havoc) is playing it.
/// </summary>
/// <remarks>
/// Stampede's path (`AutoPlayCardFromHand`) applied the card's effects with whatever target
/// the last player action had left, so the stream never moved: night04's 0MSJJVRJTD,
/// 3KZEF6T5ZZ, KVH8CJ1A4R, 6REDVUTFD0 and LFCAX0UX0G all differ from the game by exactly one
/// CombatTargets draw at the first end of turn. The queued path rolled for every ATTACK, which
/// is the wrong test twice over: an attack that hits everyone does not roll, and a skill that
/// aims at an enemy does.
/// </remarks>
public class AutoPlayRolledTargetIsUsedTests
{
    private static CardInstance Card(int id) => new(id, false);

    /// <summary>A seed whose first two-way draw picks the SECOND enemy, so "first enemy" cannot pass by luck.</summary>
    private static int SeedPickingTheSecondOfTwo()
    {
        for (int seed = 1; ; seed++)
        {
            if (new CountingRandom(seed).Next(2) == 1)
            {
                return seed;
            }
        }
    }

    private static Fight TwoEnemies(params CardInstance[] hand)
    {
        var fight = Fight.Hand(hand).Enemy(hp: 100).Enemy(hp: 100).Draw(Card(IC.DefendIronclad));
        fight.State.TargetRng = new CountingRandom(SeedPickingTheSecondOfTwo());
        return fight;
    }

    [Fact]
    public void StampedeRollsItsTargetOnTheCombatTargetsStream()
    {
        var fight = TwoEnemies(Card(IC.StrikeIronclad)).PlayerBuff(BuffId.Stampede, 1);

        fight.EndTurn();

        Assert.Equal(1, fight.State.TargetRng!.CallCount);
        Assert.Equal(100, fight.Enemy0.Hp);
        Assert.Equal(94, fight.Enemy1.Hp);
    }

    [Fact]
    public void AnAttackThatHitsEveryoneRollsNothing()
    {
        var fight = TwoEnemies(Card(IC.Thunderclap)).PlayerBuff(BuffId.Stampede, 1);

        fight.EndTurn();

        Assert.Equal(0, fight.State.TargetRng!.CallCount);
        Assert.Equal(96, fight.Enemy0.Hp);
        Assert.Equal(96, fight.Enemy1.Hp);
    }

    [Fact]
    public void ASkillThatAimsAtAnEnemyRollsWhenAutoPlayed()
    {
        var fight = TwoEnemies().Draw(Card(CL.DarkShackles));

        CombatEngine.AutoPlayFromDrawPileTop(fight.State, 1, new Random(0));

        Assert.Equal(1, fight.State.TargetRng!.CallCount);
        // Dark Shackles takes Strength off its target: the second enemy, as drawn.
        Assert.Empty(fight.Enemy0.Buffs);
        Assert.NotEmpty(fight.Enemy1.Buffs);
    }
}
