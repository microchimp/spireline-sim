namespace Sts2Emulator.Core.Run;

/// <summary>
/// Using a potion away from a fight: the four potions whose `Usage` is `PotionUsage.AnyTime`
/// (BloodPotion, EntropicBrew, FoulPotion, FruitJuice). Every other potion is `CombatOnly` or
/// `Automatic`, so the belt's "use" button is dead outside a fight for it.
/// </summary>
/// <remarks>
/// The run engine had no such move at all, so a belt could only lose a potion by dropping it for
/// room. Real histories drink Blood Potions at rest sites, shops and events, empty Entropic Brews
/// at shops, and throw Foul Potions at the merchant for 100 gold each (the Fake Merchant's throw
/// was the one use modelled). `NPotionPopup` offers the button whenever the owner's
/// `CanRemovePotions` holds -- The Future of Potions, Ranwid the Elder and the Stone of All Time
/// clear it until they finish -- and the potion's own `PassesCustomUsabilityCheck` passes:
/// `FoulPotion` wants the merchant's room with its inventory shut.
/// </remarks>
public static class RunPotionUse
{
    /// <summary>The slot a use action names, or -1 when the action is not a use.</summary>
    public static int SlotOf(int action) =>
        action >= RunConstants.UsePotionAction
        && action < RunConstants.UsePotionAction + RunConstants.PotionSlotCapacity
            ? action - RunConstants.UsePotionAction
            : -1;

    /// <summary>
    /// Whether the run is on a screen where the belt can be used: not inside a fight (the
    /// combat engine's own move), not inside a card-select or divination modal, and not in an
    /// event that holds the belt (`Owner.CanRemovePotions = false`).
    /// </summary>
    public static bool BeltIsOpen(RunState state)
    {
        switch (state.Phase)
        {
            case RunPhase.Map:
            case RunPhase.Rest:
            case RunPhase.Shop:
            case RunPhase.RelicReward:
            case RunPhase.Treasure:
            case RunPhase.Ancient:
            case RunPhase.FakeMerchant:
                return true;
            case RunPhase.Event:
                return state.EventId
                    is not (
                        RunConstants.EventTheFutureOfPotions
                        or RunConstants.EventRanwidTheElder
                        or RunConstants.EventStoneOfAllTime
                    );
            default:
                return false;
        }
    }

    /// <summary>Whether the potion in <paramref name="slot" /> can be used right now.</summary>
    public static bool CanUse(RunState state, int slot)
    {
        if ((uint)slot >= (uint)state.PotionSlots.Length || state.PotionSlots[slot] == 0)
        {
            return false;
        }

        if (!BeltIsOpen(state))
        {
            return false;
        }

        int potion = state.PotionSlots[slot];
        if (potion == RunNonCombatEffects.NamedPotion("FoulPotion"))
        {
            // `FoulPotion.PassesCustomUsabilityCheck`: a MerchantRoom whose inventory is shut.
            // The Fake Merchant's stall has its own throw (`FakeMerchantThrowAction`), which
            // starts the fight.
            return state.Phase == RunPhase.Shop;
        }

        return potion == RunNonCombatEffects.NamedPotion("BloodPotion")
            || potion == RunNonCombatEffects.NamedPotion("FruitJuice")
            || potion == RunNonCombatEffects.NamedPotion("EntropicBrew");
    }

    /// <summary>
    /// `PotionModel.OnUseWrapper`: the potion leaves the belt FIRST (`RemoveBeforeUse`), then
    /// its `OnUse` runs. Returns false when the slot cannot be used.
    /// </summary>
    public static bool Use(RunState state, int slot)
    {
        if (!CanUse(state, slot))
        {
            return false;
        }

        int potion = state.PotionSlots[slot];
        state.PotionSlots[slot] = 0;

        if (potion == RunNonCombatEffects.NamedPotion("BloodPotion"))
        {
            // BloodPotion.OnUse: Heal(MaxHp * HealPercent(20) / 100).
            int heal = (int)(state.PlayerMaxHp * 20m / 100m);
            state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + heal);
        }
        else if (potion == RunNonCombatEffects.NamedPotion("FruitJuice"))
        {
            // FruitJuice.OnUse: GainMaxHp(MaxHpVar(5)).
            RunNonCombatEffects.GainMaxHp(state, 5);
        }
        else if (potion == RunNonCombatEffects.NamedPotion("EntropicBrew"))
        {
            // EntropicBrew.OnUse: `while (Owner.HasOpenPotionSlots)` roll
            // `PotionFactory.CreateRandomPotionOutOfCombat(Owner, Rng.CombatPotionGeneration)`
            // and `PotionCmd.TryToProcure` it, breaking when the procure fails (Sozu, or a
            // belt that filled up). The brew's own slot is already open.
            while (state.UsablePotionSlots().Any(held => held == 0))
            {
                int rolled = RunRewardGenerator.NextPotion(state, state.Rng.CombatPotionGeneration);
                if (!RunRewardGenerator.AddPotion(state, rolled))
                {
                    break;
                }
            }
        }
        else
        {
            // FoulPotion.OnUse at a MerchantRoom: PlayerCmd.GainGold(GoldVar(100)).
            RunNonCombatEffects.GainGold(state, 100);
        }

        return true;
    }
}
