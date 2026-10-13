using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Punch Off's fight ("I Can Take Them") is a monster room with two extra rewards,
/// <c>[RelicReward, PotionReward]</c> (<c>EnterCombatWithoutExitingEvent(list, ...)</c>), appended to the
/// room's own gold / potion roll / cards. The emulator paid only the room's own, so the relic and the
/// potion never reached the screen (Spire Codex Ironclad A10 runs: 104 stopped on the missing relic).
/// </summary>
public class PunchOffFightRewardsTests
{
    private static RunEngine Fight(int encounterOverride)
    {
        var engine = new RunEngine();
        engine.Reset("ABCDEF");
        RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventPunchOff);
        Assert.Equal(0, engine.Step(1, -1, out _, out _, out _)); // I Can Take Them
        Assert.Equal(0, engine.Step(0, -1, out _, out _, out _)); // Fight
        Assert.Equal(RunPhase.Combat, engine.State.Phase);
        if (encounterOverride != 0)
        {
            engine.State.ActiveCombat!.EncounterId = encounterOverride; // the control: same fight, no extras
        }

        Assert.Equal(0, engine.DebugResolveCombat());
        return engine;
    }

    [Fact]
    public void TheFightOffersARelicAndAPotionBesideItsOwnRewards()
    {
        var control = Fight(encounterOverride: 1);
        var engine = Fight(encounterOverride: 0);
        var s = engine.State;

        Assert.Equal(RunPhase.RelicReward, s.Phase);
        Assert.Equal(0, control.State.RelicReward);
        Assert.NotEqual(0, s.RelicReward);
        Assert.NotEqual(0, s.RewardPotion);
        // The room's own rewards are the control's, drawn first; then the relic's rarity roll and the
        // potion's two draws (rarity, pick) -- and the relic comes off the front of the queue.
        Assert.Equal(control.State.RewardCards[0], s.RewardCards[0]);
        Assert.Equal(control.State.PlayerRng.Rewards.CallCount + 3, s.PlayerRng.Rewards.CallCount);
    }

    [Fact]
    public void ARolledPotionAndTheExtraOneShareTheScreen()
    {
        // Seeds differ in whether the room's own odds rolled a potion; find one that did.
        for (int n = 0; n < 60; n++)
        {
            var engine = new RunEngine();
            engine.Reset("SEED" + n.ToString("D6"));
            RunNonCombatEffects.BeginEvent(engine.State, RunConstants.EventPunchOff);
            engine.Step(1, -1, out _, out _, out _);
            engine.Step(0, -1, out _, out _, out _);
            engine.DebugResolveCombat();
            if (engine.State.PendingPotionRewards.Count == 1)
            {
                Assert.NotEqual(0, engine.State.RewardPotion);
                Assert.NotEqual(0, engine.State.PendingPotionRewards[0]);
                return;
            }
        }

        Assert.Fail("no seed rolled a room potion");
    }
}
