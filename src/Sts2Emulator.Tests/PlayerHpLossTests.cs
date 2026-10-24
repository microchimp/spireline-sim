using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// One HP-loss step for every way the player loses HP (C4-F01). `CreatureCmd.Damage` runs
/// `Hook.ModifyHpLost` on every damage instance: AfterOsty (IntangiblePower, then the
/// relics in the order held: TungstenRod, BeatingRemnant) and AfterOstyLate (BufferPower),
/// and BeatingRemnant.AfterDamageReceived tallies what landed. The emulator applied them
/// per path: an enemy's attack ignored Buffer, a card's self-damage ignored the relics and
/// Intangible, and the relics always ran rod first.
/// </summary>
public class PlayerHpLossTests
{
    private static Fight WithRelicsHeld(Fight fight, params int[] relicIds)
    {
        foreach (int id in relicIds)
        {
            fight.State.Relics.Add(new RelicInstance(id));
        }

        return fight;
    }

    /// <summary>
    /// Corpus MM7V9M80NV step 4: Bloodletting with Tungsten Rod took the game from 64 to 62
    /// HP; the self-damage path skipped the rod and took 3.
    /// </summary>
    [Fact]
    public void BloodlettingWithTungstenRodLosesTwo()
    {
        var fight = WithRelicsHeld(
            Fight.Hand(Card(IC.Bloodletting)).Energy(0).PlayerHp(64).Enemy(hp: 40),
            RelicEffects.TungstenRod
        );

        fight.Play();

        Assert.Equal(62, fight.State.PlayerHp);
    }

    /// <summary>
    /// Corpus WN659AKM5D: a Waterfall Giant's hit took Buffer from 2 to 1 and no HP in the
    /// game; the emulator's attack path never asked Buffer.
    /// </summary>
    [Fact]
    public void BufferStopsAnEnemyHit()
    {
        var fight = Fight.Hand().PlayerHp(80).PlayerBuff(BuffId.Buffer, 2).Enemy(hp: 100);
        fight.State.PlayerBlock = 0;

        EnemyAI.DealAttackDamageForTests(fight.Enemy0, fight.State, 12);

        Assert.Equal(80, fight.State.PlayerHp);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Buffer));
    }

    /// <summary>
    /// Buffer is spent only on HP that would have come off: `Hook.ModifyHpLost` lists a
    /// modifier only when it moved the amount, and the rod has already taken 1 to 0.
    /// </summary>
    [Fact]
    public void TheRodTakingTheLossToNothingLeavesBufferAlone()
    {
        var fight = WithRelicsHeld(
            Fight.Hand().PlayerHp(50).PlayerBuff(BuffId.Buffer, 1).Enemy(hp: 100),
            RelicEffects.TungstenRod
        );

        CardEffects.LoseHp(fight.State, 1);

        Assert.Equal(50, fight.State.PlayerHp);
        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.Buffer));
    }

    /// <summary>Self-damage is capped by Beating Remnant and counts toward its 20.</summary>
    [Fact]
    public void SelfDamageIsCappedAndCountedByBeatingRemnant()
    {
        var fight = WithRelicsHeld(
            Fight.Hand().PlayerHp(500, 500).Enemy(hp: 100),
            RelicEffects.BeatingRemnant
        );
        fight.State.PlayerBlock = 0;

        CardEffects.LoseHp(fight.State, 15);
        Assert.Equal(485, fight.State.PlayerHp);

        CardEffects.DealDamageToPlayer(fight.State, 15);
        Assert.Equal(480, fight.State.PlayerHp);

        CardEffects.LoseHp(fight.State, 15);
        Assert.Equal(480, fight.State.PlayerHp);
    }

    /// <summary>`IntangiblePower.ModifyHpLostAfterOsty` reads no props: a self-hit is 1.</summary>
    [Fact]
    public void IntangibleCapsSelfDamageAtOne()
    {
        var fight = Fight.Hand().PlayerHp(50).PlayerBuff(BuffId.Intangible, 1).Enemy(hp: 100);

        CardEffects.LoseHp(fight.State, 6);

        Assert.Equal(49, fight.State.PlayerHp);
    }

    /// <summary>
    /// `IterateHookListeners` walks the relics in the order held. With 15 already taken this
    /// turn, a 10: rod first is 9 then the cap's 5; the cap first is 5 then the rod's 4.
    /// </summary>
    [Theory]
    [InlineData(true, 5)]
    [InlineData(false, 4)]
    public void TheRelicsApplyInTheOrderHeld(bool rodFirst, int lost)
    {
        int[] order = rodFirst
            ? [RelicEffects.TungstenRod, RelicEffects.BeatingRemnant]
            : [RelicEffects.BeatingRemnant, RelicEffects.TungstenRod];
        var fight = WithRelicsHeld(Fight.Hand().PlayerHp(500, 500).Enemy(hp: 100), order);
        fight.State.PlayerBlock = 0;
        fight.State.UnblockedDamageThisTurn = 15;

        CardEffects.DealDamageToPlayer(fight.State, 10);

        Assert.Equal(500 - lost, fight.State.PlayerHp);
    }

    /// <summary>
    /// `ReflectPower` pays `result.BlockedDamage`: what block took, 5 of a 12, not the 12
    /// less the HP the rod left to lose.
    /// </summary>
    [Fact]
    public void ReflectReturnsWhatBlockTookNotWhatTheRodSaved()
    {
        var fight = WithRelicsHeld(
            Fight.Hand().PlayerHp(80).PlayerBuff(BuffId.Reflect, 1).Enemy(hp: 100),
            RelicEffects.TungstenRod
        );
        fight.State.PlayerBlock = 5;

        EnemyAI.DealAttackDamageForTests(fight.Enemy0, fight.State, 12);

        Assert.Equal(74, fight.State.PlayerHp);
        Assert.Equal(95, fight.Enemy0.Hp);
    }

    /// <summary>
    /// `CreatureCmd.LoseMaxHp` deals the excess through the same damage, then
    /// `SetMaxHpInternal` clamps: the HP the rod kept off does not stay over the cap.
    /// </summary>
    [Fact]
    public void LosingMaxHpWithTheRodStillEndsAtTheNewMaximum()
    {
        var fight = WithRelicsHeld(
            Fight.Hand().PlayerHp(50, 50).Enemy(hp: 100),
            RelicEffects.TungstenRod
        );

        CardEffects.LoseMaxHp(fight.State, 5);

        Assert.Equal(45, fight.State.PlayerMaxHp);
        Assert.Equal(45, fight.State.PlayerHp);
    }
}

/// <summary>
/// Out of combat the same `CreatureCmd.Damage` charges every event's and relic's HP cost,
/// and `TungstenRod.ModifyHpLostAfterOsty` has no in-combat guard: each loss is 1 less.
/// The run engine subtracted the raw number at every site (Spire Codex: Slippery Bridge's
/// Hold On with the rod).
/// </summary>
public class RunHpLossTests
{
    [Fact]
    public void SlipperyBridgeHoldOnWithTheRodCostsOneLess()
    {
        var engine = TestSwitches.Run(new RunEngine());
        engine.Reset("ABCDEF");
        engine.State.Relics.Add(new RelicInstance(RelicEffects.TungstenRod));
        engine.State.PlayerHp = 50;
        RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventSlipperyBridge);

        Assert.Equal(0, engine.Step(1, -1, out _, out _, out _));
        Assert.Equal(48, engine.State.PlayerHp);
        Assert.Equal(0, engine.Step(1, -1, out _, out _, out _));
        Assert.Equal(45, engine.State.PlayerHp);
    }

    [Fact]
    public void AbyssalBathsImmerseWithTheRodCostsOneLess()
    {
        var state = new RunState
        {
            PlayerHp = 50,
            PlayerMaxHp = 80,
            Relics = [new RelicInstance(RelicEffects.TungstenRod)],
        };

        RunNonCombatEffects.Immerse(state);

        Assert.Equal(50, state.PlayerHp);
    }

    /// <summary>Beating Remnant asks `CombatManager.IsInProgress`: no cap out of combat.</summary>
    [Fact]
    public void BeatingRemnantDoesNotCapAnEventsCost()
    {
        var state = new RunState
        {
            PlayerHp = 50,
            PlayerMaxHp = 80,
            Relics = [new RelicInstance(RelicEffects.BeatingRemnant)],
        };

        RunNonCombatEffects.LoseHp(state, 30);

        Assert.Equal(20, state.PlayerHp);
    }
}
