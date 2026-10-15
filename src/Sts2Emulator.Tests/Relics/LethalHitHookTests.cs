using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `CreatureCmd.Damage` hands a result whose target it killed to `Kill` instead of
/// `Hook.AfterDamageReceived` (`WasTargetKilled` and `IsDead`, read before the death is
/// prevented). A hit that empties the player's HP therefore arms Self-Forming Clay and
/// Centennial Puzzle on no one, even when Lizard Tail then refuses the death and heals. The
/// recorded fight YR5HZ37K98 (Clay and Lizard Tail) paid 3 next-turn block after such a hit
/// where the emulator paid 6; the re-anchor used to hide it by dropping the game's Self-Forming
/// Clay power (the emulator's BlockNextTurn) before every step.
/// </summary>
public class LethalHitHookTests
{
    /// <summary>One lethal hit from a Toadpole's single-hit attack onto a player at 1 HP.</summary>
    private static Fight LethalHit(params int[] relics)
    {
        var fight = Fight.Hand().Enemy(hp: 100);
        foreach (int relic in relics)
        {
            fight.State.Relics.Add(new RelicInstance(relic));
        }

        fight.State.PlayerHp = 1;
        fight.State.PlayerMaxHp = 80;
        fight.State.Enemies[0].DefId = KE.Toadpole;
        fight.State.Enemies[0].MoveIndex = 1;
        fight.State.Enemies[0].CurrentIntent = new Intent(IntentType.Attack, 3, Hits: 1);
        return fight;
    }

    [Fact]
    public void ALethalHitLizardTailRefusesArmsNoSelfFormingClay()
    {
        var fight = LethalHit(RelicEffects.LizardTail, RelicEffects.SelfFormingClay);

        fight.EndTurn();

        Assert.Equal(40, fight.State.PlayerHp);  // revived by the hit that killed
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.BlockNextTurn));
        Assert.Equal(0, fight.State.PlayerBlock);
    }

    [Fact]
    public void ALethalHitLizardTailRefusesDrawsNothingAndKeepsTheCentennialPuzzle()
    {
        var plain = LethalHit(RelicEffects.LizardTail);
        var withPuzzle = LethalHit(RelicEffects.LizardTail, RelicEffects.CentennialPuzzle);

        plain.EndTurn();
        withPuzzle.EndTurn();

        Assert.Equal(plain.State.Hand.Count, withPuzzle.State.Hand.Count);
        // not spent: the next hit that lands on the revived player still draws its three
        Assert.Equal(0, withPuzzle.State.Relics.Single(r => r.DefId == RelicEffects.CentennialPuzzle).Counter);
    }

    [Fact]
    public void TheHitsAfterTheRevivalArmTheRelicAsAlways()
    {
        // The first of three hits kills, Lizard Tail revives, the other two land on a living player.
        var fight = LethalHit(RelicEffects.LizardTail, RelicEffects.SelfFormingClay);
        fight.State.Enemies[0].CurrentIntent = new Intent(IntentType.Attack, 3, Hits: 3);

        fight.EndTurn();

        Assert.Equal(6, fight.State.PlayerBlock);  // two hits armed it, not three
    }

    [Fact]
    public void ASurvivedHitStillArmsIt()
    {
        var fight = LethalHit(RelicEffects.SelfFormingClay);
        fight.State.PlayerHp = 50;

        fight.EndTurn();

        Assert.Equal(3, fight.State.PlayerBlock);
    }

    [Fact]
    public void SelfDamageThatKillsArmsNothingEitherRoute()
    {
        // The self-damage route (Bloodletting and the like) made this cut already; pinned beside the enemy route.
        var state = Fight.Hand().State;
        state.Relics.Add(new RelicInstance(RelicEffects.LizardTail));
        state.Relics.Add(new RelicInstance(RelicEffects.SelfFormingClay));
        state.PlayerHp = 2;

        CardEffects.DealDamageToPlayer(state, 5);

        Assert.Equal(0, BuffSystem.Get(state.PlayerBuffs, BuffId.BlockNextTurn));
    }
}
