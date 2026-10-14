using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

public class InfernoTests
{
    [Fact]
    public void TriggersWhenPlayerLosesHpOnPlayerTurn()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerHp = 50;
        state.Hand = [new CardInstance(IC.Inferno, false), new CardInstance(IC.Hemokinesis, false)];
        state.Energy = 2;
        state.Enemies =
        [
            new EnemyState
            {
                DefId = 16,
                Hp = 100,
                MaxHp = 100,
                Buffs = [],
            },
            new EnemyState
            {
                DefId = 16,
                Hp = 100,
                MaxHp = 100,
                Buffs = [],
            },
        ];

        CombatEngine.Step(state, 0, new Random(0));
        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(48, state.PlayerHp);
        Assert.Equal(6, BuffSystem.Get(state.PlayerBuffs, BuffId.Inferno));
        Assert.Equal(1, BuffSystem.Get(state.PlayerBuffs, BuffId.InfernoSelfDamage));
        Assert.Equal(79, state.Enemies[0].Hp);
        Assert.Equal(94, state.Enemies[1].Hp);
    }

    [Fact]
    public void DamagesPlayerAndEnemiesAtStartOfPlayerTurn()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.PlayerHp = 50;
        state.Hand = [new CardInstance(IC.Inferno, true)];
        state.DrawPile.Clear();
        state.DiscardPile.Clear();
        state.Energy = 1;
        state.Enemies =
        [
            new EnemyState
            {
                DefId = 16,
                Hp = 100,
                MaxHp = 100,
                CurrentIntent = new Intent(IntentType.Defend, 0),
                Buffs = [],
            },
            new EnemyState
            {
                DefId = 16,
                Hp = 100,
                MaxHp = 100,
                CurrentIntent = new Intent(IntentType.Defend, 0),
                Buffs = [],
            },
        ];

        CombatEngine.Step(state, 0, new Random(0));
        CombatEngine.Step(state, 0, new Random(0));

        Assert.Equal(49, state.PlayerHp);
        Assert.Equal(9, BuffSystem.Get(state.PlayerBuffs, BuffId.Inferno));
        Assert.Equal(91, state.Enemies[0].Hp);
        Assert.Equal(91, state.Enemies[1].Hp);
    }
}

/// <summary>
/// `InfernoPower.AfterPlayerTurnStart` is in the powers' AfterPlayerTurnStart pass, after the hand
/// draw (`CombatManager.SetupPlayerTurn`). The emulator burned at the top of the turn start, so a
/// burst that killed the last enemy came before the draw: the draw found the fight ending and drew
/// and shuffled nothing, where the game draws (and shuffles) and then burns. Seen on 30 recorded
/// fights at their last end of turn (Shuffle 27 against the game's 36, 7FT4BN9ULD).
/// </summary>
public class InfernoTurnStartOrderTests
{
    [Fact]
    public void TheBurstThatEndsTheFightComesAfterTheHandDraw()
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Energy(3)
            .Draw()
            .Discard([.. Pile(IC.StrikeIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Bash, IC.Anger, IC.DefendIronclad)])
            .Enemy(hp: 3, intent: new Intent(IntentType.Defend, 0))
            .PlayerBuff(BuffId.Inferno, 6)
            .PlayerBuff(BuffId.InfernoSelfDamage, 1);
        var shuffle = new CountingRandom(5);
        fight.State.ShuffleRng = shuffle;

        var result = fight.EndTurn();

        Assert.True(result.Terminal && result.PlayerWon);
        Assert.True(shuffle.CallCount > 0, "the hand draw shuffled before the burst ended the fight");
        Assert.Equal(5, fight.State.Hand.Count);
    }

    [Fact]
    public void ScreensAndBurnsRunInThePowerListsOrder()
    {
        // Tyranny applied before Inferno: its exhaust screen comes first, and Inferno's
        // self-damage only after the pick.
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Energy(3)
            .Draw([.. Pile(IC.Bash, IC.DefendIronclad, IC.StrikeIronclad, IC.DefendIronclad, IC.Anger, IC.Inflame, IC.TwinStrike)])
            .Enemy(hp: 500, intent: new Intent(IntentType.Defend, 0))
            .PlayerBuff(BuffId.Tyranny, 1)
            .PlayerBuff(BuffId.Inferno, 6)
            .PlayerBuff(BuffId.InfernoSelfDamage, 1);
        int hp = fight.State.PlayerHp;

        fight.EndTurn();

        Assert.Equal(CardSelectionKind.ExhaustFromHandRepeated, fight.Pending!.Kind);
        Assert.Equal(hp, fight.State.PlayerHp);
        fight.Choose(0);
        Assert.Equal(hp - 1, fight.State.PlayerHp);
        Assert.Equal(500 - 6, fight.Enemy0.Hp);
    }
}
