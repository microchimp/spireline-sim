using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CreatureCmd.Stun` sets the creature's next move to a STUNNED move there and then
/// (`SetMoveImmediate`): a stun is a move, not a power, so a second stun before the creature
/// acts replaces the first and it sits out one turn.
/// </summary>
public class StunTests
{
    private static CombatState Slugs(params int[] hps)
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand.Clear();
        state.DrawPile.Clear();
        state.DiscardPile.Clear();
        state.ExhaustPile.Clear();
        state.Enemies = [];
        foreach (int hp in hps)
        {
            state.Enemies.Add(
                new EnemyState
                {
                    DefId = KE.CorpseSlug,
                    Hp = hp,
                    MaxHp = 25,
                    CurrentIntent = new Intent(IntentType.Attack, 8),
                    Buffs = [new BuffState(BuffId.Ravenous, 5)],
                }
            );
        }

        return state;
    }

    /// <summary>
    /// Two slugs killed in one turn: the survivor's Ravenous runs twice, Strength 5 each time
    /// and a stun each time, and the second stun replaces the first. It sits out the next
    /// enemy turn only. The copy counted the stuns, so its survivor sat out two turns while
    /// the game's hit the player with the Strength both deaths gave it.
    /// </summary>
    [Fact]
    public void TwoAlliesDeadInOneTurnStunTheSurvivorOnce()
    {
        var state = Slugs(1, 1, 25);
        state.Hand.Add(new CardInstance(IC.StrikeIronclad, false));
        state.Hand.Add(new CardInstance(IC.StrikeIronclad, false));
        state.Energy = 3;
        var survivor = state.Enemies[2];

        CombatEngine.Step(state, 0, new Random(0), 0);
        CombatEngine.Step(state, 0, new Random(0), 0);

        Assert.Single(state.Enemies, e => e.Hp > 0);
        Assert.Equal(10, BuffSystem.Get(survivor.Buffs, BuffId.Strength));
        Assert.Equal(1, BuffSystem.Get(survivor.Buffs, BuffId.Stunned));

        int hp = state.PlayerHp;
        EnemyAI.ExecuteIntent(survivor, state, new Random(0));

        Assert.Equal(hp, state.PlayerHp);
        Assert.Equal(0, BuffSystem.Get(survivor.Buffs, BuffId.Stunned));
    }

    /// <summary>
    /// Whistle calls `CreatureCmd.Stun` on its target: on an enemy already stunned it changes
    /// nothing, and it is no power, so the stun is not counted twice.
    /// </summary>
    [Fact]
    public void WhistleOnAStunnedEnemyLeavesOneStun()
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand.Clear();
        state.Enemies =
        [
            new EnemyState
            {
                DefId = KE.CorpseSlug,
                Hp = 200,
                MaxHp = 200,
                CurrentIntent = new Intent(IntentType.Attack, 8),
                Buffs = [new BuffState(BuffId.Stunned, 1)],
            },
        ];
        state.Hand.Add(new CardInstance(GeneratedData.Cards.FindId("Whistle")!.Value, false));
        state.Energy = 3;

        CombatEngine.Step(state, 0, new Random(0), 0);

        Assert.Equal(1, BuffSystem.Get(state.Enemies[0].Buffs, BuffId.Stunned));
    }
}
