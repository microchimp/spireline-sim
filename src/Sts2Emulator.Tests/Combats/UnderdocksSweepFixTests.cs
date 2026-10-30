using System;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Three defects the underdocks sweep turned up, each reproducing on BOTH seeds.
/// </summary>
public class UnderdocksSweepFixTests
{
    /// <summary>
    /// `PlatingPower.AfterSideTurnStart` decrements for its owner whatever else is true of
    /// it. The Matriarch was excluded by def id, with no reason recorded and none in the
    /// game, so her armour never thinned: the live game held PLATING 11 at the third turn
    /// where the emulator held 12.
    /// </summary>
    [Fact]
    public void TheMatriarchsPlatingDecays()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.LagavulinMatriarch);
        int opening = fight.EnemyBuffAmount(BuffId.Plating);

        fight.EndTurn();
        fight.EndTurn();

        Assert.True(
            fight.EnemyBuffAmount(BuffId.Plating) < opening,
            "her Plating should thin the way every other plated enemy's does"
        );
    }

    /// <summary>
    /// The first enemy phase is exempt -- `AfterSideTurnStart` skips round one for enemies
    /// -- so she ends her opening turn on the full amount.
    /// </summary>
    [Fact]
    public void TheFirstTurnIsStillExempt()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.LagavulinMatriarch);

        fight.EndTurn();

        Assert.Equal(12, fight.EnemyBuffAmount(BuffId.Plating));
    }

    /// <summary>
    /// Her enemy turn end runs AsleepPower's very-early listener before PlatingPower's early one. On her last sleeping
    /// turn (Asleep 1) the first takes the Plating off, so the second has nothing to grant and she wakes into the
    /// player's fourth turn with no block. The copy granted the Plating block first (10 at that turn) and only then
    /// took the power off, so the fourth turn's first 10 damage hit her block (night 2026-10-29, group Gb25e996655:
    /// 15 Matriarch records, the copy's boss 6 to 10 HP higher).
    /// </summary>
    [Fact]
    public void SheWakesWithNoPlatingBlock()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.LagavulinMatriarch);

        fight.EndTurn();
        fight.EndTurn();
        // The control: her second sleeping turn ends on the decayed Plating's block.
        Assert.Equal(11, fight.EnemyBuffAmount(BuffId.Plating));
        Assert.Equal(11, fight.Enemy0.Block);

        fight.EndTurn();

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Asleep));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Plating));
        Assert.Equal(0, fight.Enemy0.Block);
    }

    /// <summary>
    /// `SmoggyPower` has NO self-removal: it stands for the rest of the fight. What
    /// `AfterSideTurnEnd` clears is the SMOG AFFLICTION on the cards, which is the
    /// emulator's one-Skill-a-turn flag.
    /// </summary>
    [Fact]
    public void SmoggySurvivesTheTurnBoundary()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Smoggy, 1);

        fight.EndTurn();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Smoggy));
    }

    /// <summary>...and the per-turn allowance resets, so a Skill is playable again.</summary>
    [Fact]
    public void TheOneSkillAllowanceResetsEachTurn()
    {
        var fight = Fight.Hand();
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Smoggy, 1);
        fight.State.SkillPlayedWhileSmoggy = true;

        fight.EndTurn();

        Assert.False(fight.State.SkillPlayedWhileSmoggy);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Smoggy));
    }

    /// <summary>
    /// `FatGremlin.FLEE_MOVE` carries an EscapeIntent that loops on itself forever. The
    /// emulator announced it as `Buff 0`; the live readout has no magnitude at all.
    /// </summary>
    [Fact]
    public void TheFatGremlinAnnouncesAnEscapeRatherThanABuff()
    {
        var fat = new EnemyState
        {
            DefId = KE.FatGremlin,
            Hp = 15,
            MaxHp = 15,
            MoveIndex = 1,
        };

        EnemyAI.ChooseIntents([fat], turn: 2, new Random(0), aiRng: null, ascension: 8);

        Assert.Equal(IntentType.Unknown, fat.CurrentIntent.Type);
        Assert.NotEqual(IntentType.Buff, fat.CurrentIntent.Type);
    }

    /// <summary>And it still actually leaves, taking what it stole with it.</summary>
    [Fact]
    public void ItStillFlees()
    {
        var fight = Fight.Hand();
        var fat = new EnemyState
        {
            DefId = KE.FatGremlin,
            Hp = 15,
            MaxHp = 15,
            MoveIndex = 1,
            CurrentIntent = new Intent(IntentType.Unknown, 0),
        };
        fight.State.Enemies.Clear();
        fight.State.Enemies.Add(fat);

        EnemyAI.ExecuteIntent(fat, fight.State, new Random(0));

        Assert.True(fat.Escaped);
        Assert.True(fight.State.FatGremlinEscaped);
    }

    /// <summary>The turn it ARRIVES on is a stun, not a flee.</summary>
    [Fact]
    public void ItDoesNotFleeOnTheTurnItArrives()
    {
        var fight = Fight.Hand();
        var fat = new EnemyState
        {
            DefId = KE.FatGremlin,
            Hp = 15,
            MaxHp = 15,
            MoveIndex = 0,
            CurrentIntent = new Intent(IntentType.Unknown, 0),
        };
        fight.State.Enemies.Clear();
        fight.State.Enemies.Add(fat);

        EnemyAI.ExecuteIntent(fat, fight.State, new Random(0));

        Assert.False(fat.Escaped);
    }
}
