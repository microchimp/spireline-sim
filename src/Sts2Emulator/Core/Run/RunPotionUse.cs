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

    /// <summary>The slot a discard action names, or -1 when the action is not a discard.</summary>
    public static int DiscardSlotOf(int action) =>
        action >= RunConstants.DiscardPotionAction
        && action < RunConstants.DiscardPotionAction + RunConstants.PotionSlotCapacity
            ? action - RunConstants.DiscardPotionAction
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
                return !EventHoldsBelt(state.EventId);
            default:
                return false;
        }
    }

    /// <summary>`Owner.CanRemovePotions = false` from `BeforeEventStarted` to `OnEventFinished`: the only three events that clear it.</summary>
    private static bool EventHoldsBelt(int eventId) =>
        eventId
            is RunConstants.EventTheFutureOfPotions
                or RunConstants.EventRanwidTheElder
                or RunConstants.EventStoneOfAllTime;

    /// <summary>
    /// Whether a potion can be thrown away on this screen under the given mask version. Version 1: wherever
    /// <see cref="BeltIsOpen" />. Version 2 adds the screens that sit on the overlay stack over the run -- the card
    /// reward, a deck selection (<see cref="RunPhase.TransformSelect" />), the bundle choice and the Crystal Sphere --
    /// unless one of the three events that hold the belt is running (the Future of Potions' card reward and the Stone
    /// of All Time's enchant selection are screens of those events). `NPotionPopup` enables its discard button behind
    /// `Owner.CanRemovePotions` alone; its `InACardSelectScreen` test only gates the use button of a `CombatOnly`
    /// potion, and the popup closes itself on an overlay change only for that usage. A use is NOT opened here: the
    /// owner approved the discard only (D32); version 3 opens
    /// the use of the AnyTime potions on the same screens (<see cref="UseBeltIsOpen" />, D34).
    /// </summary>
    public static bool DiscardBeltIsOpen(RunState state, int maskVersion = RunConstants.MaskVersion1)
    {
        if (BeltIsOpen(state))
        {
            return true;
        }

        return maskVersion >= RunConstants.MaskVersion2 && OnAnOverlayScreen(state);
    }

    /// <summary>
    /// Whether a potion can be used on this screen under the given mask version (D34). Version 1 and 2: wherever
    /// <see cref="BeltIsOpen" />. Version 3 adds the same four overlay screens as <see cref="DiscardBeltIsOpen" />, on
    /// the same conditions: `NPotionPopup` enables the use button of an `AnyTime` potion with no overlay test (its
    /// `InACardSelectScreen` only gates a `CombatOnly` potion in a fight), behind `Owner.CanRemovePotions` and the
    /// potion's own `PassesCustomUsabilityCheck`. Which potions pass is <see cref="CanUse" />'s question.
    /// </summary>
    public static bool UseBeltIsOpen(RunState state, int maskVersion = RunConstants.MaskVersion1)
    {
        if (BeltIsOpen(state))
        {
            return true;
        }

        return maskVersion >= RunConstants.MaskVersion3 && OnAnOverlayScreen(state);
    }

    /// <summary>
    /// The screens that sit on the overlay stack over the run -- the card reward, a deck selection
    /// (<see cref="RunPhase.TransformSelect" />), the bundle choice and the Crystal Sphere -- unless one of the three
    /// events that hold the belt (`CanRemovePotions` false) is running.
    /// </summary>
    private static bool OnAnOverlayScreen(RunState state) =>
        state.Phase
            is RunPhase.CardReward
                or RunPhase.TransformSelect
                or RunPhase.BundleSelect
                or RunPhase.CrystalSphere
        && !EventHoldsBelt(state.EventId);

    /// <summary>
    /// Whether the potion in <paramref name="slot" /> can be used right now. Away from a fight the game offers the use
    /// button only for a `PotionUsage.AnyTime` potion (a targeted potion -- `TargetType.AnyEnemy` and the like -- is
    /// `CombatOnly` and has no button outside a fight). The four AnyTime potions are Blood Potion (`TargetType.AnyPlayer`),
    /// Fruit Juice (`AnyPlayer`), Entropic Brew (`Self`) and Foul Potion, whose `PassesCustomUsabilityCheck` wants the
    /// merchant's room; none of the four overlay screens of mask version 3 is a shop, so Foul Potion stays unusable there
    /// (a deck selection the shop's removal service opened is the one overlay inside a MerchantRoom; the run state does
    /// not tell it from an event's, so that throw is not offered).
    /// </summary>
    public static bool CanUse(RunState state, int slot, int maskVersion = RunConstants.MaskVersion1)
    {
        if ((uint)slot >= (uint)state.PotionSlots.Length || state.PotionSlots[slot] == 0)
        {
            return false;
        }

        if (!UseBeltIsOpen(state, maskVersion))
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

        return Effects.PotionEffects.UsageOf(potion) == PotionUsage.AnyTime;
    }

    /// <summary>
    /// Whether the potion in <paramref name="slot" /> can be thrown away right now: any held potion, on every screen
    /// where the belt is open (`NPotionPopup` offers the discard button for every `Usage`, behind the same
    /// `CanRemovePotions` check as the use button).
    /// </summary>
    public static bool CanDiscard(RunState state, int slot, int maskVersion = RunConstants.MaskVersion1) =>
        (uint)slot < (uint)state.PotionSlots.Length
        && state.PotionSlots[slot] != 0
        && DiscardBeltIsOpen(state, maskVersion);

    /// <summary>
    /// `DiscardPotionGameAction` -> `Player.DiscardPotionInternal`: the potion leaves the belt and nothing else
    /// happens. Returns false when the slot cannot be discarded.
    /// </summary>
    public static bool Discard(RunState state, int slot, int maskVersion = RunConstants.MaskVersion1)
    {
        if (!CanDiscard(state, slot, maskVersion))
        {
            return false;
        }

        state.PotionSlots[slot] = 0;
        return true;
    }

    /// <summary>
    /// Why a use or discard action is refused right now, or null when it is legal (or not a potion action at all).
    /// The reasons are the game's own rules: an empty slot, a belt a fight or a modal holds, a potion whose `Usage`
    /// is not `AnyTime`, and the Foul Potion's merchant-room check.
    /// </summary>
    public static string? Refusal(RunState state, int action, int maskVersion = RunConstants.MaskVersion1)
    {
        int useSlot = SlotOf(action);
        int discardSlot = DiscardSlotOf(action);
        int slot = useSlot >= 0 ? useSlot : discardSlot;
        if (slot < 0)
        {
            return null;
        }

        if ((uint)slot >= (uint)state.PotionSlots.Length || state.PotionSlots[slot] == 0)
        {
            return "the belt slot is empty";
        }

        if (useSlot >= 0 ? !UseBeltIsOpen(state, maskVersion) : !DiscardBeltIsOpen(state, maskVersion))
        {
            return useSlot >= 0 && DiscardBeltIsOpen(state, maskVersion)
                ? "a potion can only be thrown away on this screen, not used (mask version 2 opens the discard alone)"
                : "the belt is closed on this screen (a fight, a card choice, or an event that holds the belt)";
        }

        if (useSlot < 0 || CanUse(state, slot, maskVersion))
        {
            return null;
        }

        if (state.PotionSlots[slot] == RunNonCombatEffects.NamedPotion("FoulPotion"))
        {
            return "the Foul Potion can be used only in the merchant's room";
        }

        return "the potion's Usage is not AnyTime: the game offers no use button for it outside a fight";
    }

    /// <summary>
    /// `PotionModel.OnUseWrapper`: the potion leaves the belt FIRST (`RemoveBeforeUse`), then
    /// its `OnUse` runs. Returns false when the slot cannot be used.
    /// </summary>
    public static bool Use(RunState state, int slot, int maskVersion = RunConstants.MaskVersion1)
    {
        if (!CanUse(state, slot, maskVersion))
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
