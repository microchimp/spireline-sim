using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.GeneratedData;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Which held potions are legal moves in a fight. The game's fight loop and `NPotionPopup` offer a potion only
/// when it is not queued, passes `PassesCustomUsabilityCheck` and has `Usage` CombatOnly or AnyTime; an Automatic
/// potion (Fairy in a Bottle) has no use button. The emulator listed every modelled potion, so a belt holding a
/// Fairy made the legal list `[end, potion]` where the game offered End Turn alone.
/// </summary>
public class PotionLegalityTests
{
    private static int Id(string name) => Potions.FindId(name) ?? throw new InvalidOperationException(name);

    private static CombatState FightHolding(params string[] belt)
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Hand = [];
        state.MaxPotionSlots = 3;
        state.PotionSlots = new int[3];
        for (int i = 0; i < belt.Length; i++)
        {
            state.PotionSlots[i] = belt[i] == "-" ? 0 : Id(belt[i]);
        }

        return state;
    }

    [Fact]
    public void FairyInABottle_IsNotALegalCombatAction()
    {
        var state = FightHolding("FairyInABottle");

        // With an empty hand the only legal action is End Turn, as the game lists it.
        Assert.Equal(new[] { state.Hand.Count }, CombatEngine.ValidActions(state));
    }

    [Fact]
    public void StrengthPotion_IsALegalCombatAction()
    {
        var state = FightHolding("StrengthPotion");

        Assert.Equal(new[] { state.Hand.Count, state.Hand.Count + 1 }, CombatEngine.ValidActions(state));
    }

    [Fact]
    public void OnlyTheFairysSlotIsDropped_WhenItSitsBesideOtherPotions()
    {
        var state = FightHolding("FairyInABottle", "StrengthPotion", "FoulPotion");
        int endTurn = state.Hand.Count;

        // Slot 0 (Fairy) is gone; slot 1 (CombatOnly) and slot 2 (Foul Potion: AnyTime, and its custom check
        // passes while a combat is in progress) stay.
        Assert.Equal(new[] { endTurn, endTurn + 2, endTurn + 3 }, CombatEngine.ValidActions(state));
    }

    [Fact]
    public void ManualUseOfAnAutomaticPotion_IsRefusedAndKeepsThePotion()
    {
        var state = FightHolding("FairyInABottle");
        int fairy = state.PotionSlots[0];

        var result = CombatEngine.Step(state, state.Hand.Count + 1, new Random(0));

        Assert.Equal(StepResult.Invalid, result);
        Assert.Equal(fairy, state.PotionSlots[0]);
    }

    [Fact]
    public void RunMask_AgreesWithValidActions_InAFightHoldingAFairy()
    {
        var combat = FightHolding("FairyInABottle", "StrengthPotion", "BlockPotion");
        var engine = new RunEngine();
        engine.Reset("0");
        engine.State.Phase = RunPhase.Combat;
        engine.State.ActiveCombat = combat;
        var mask = new int[RunConstants.MaxActions];

        engine.WriteActionMask(mask);

        int[] fromMask = [.. Enumerable.Range(0, mask.Length).Where(i => mask[i] != 0)];
        int[] legal = CombatEngine.ValidActions(combat);
        Assert.Equal(legal, fromMask);
        Assert.DoesNotContain(combat.Hand.Count + 1, fromMask); // the Fairy's slot
        Assert.Contains(combat.Hand.Count + 2, fromMask); // Strength Potion
        Assert.Contains(combat.Hand.Count + 3, fromMask); // Block Potion
    }
}
