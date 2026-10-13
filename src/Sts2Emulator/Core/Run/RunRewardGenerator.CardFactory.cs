using Sts2Emulator.Core.Rng;

namespace Sts2Emulator.Core.Run;

/// <summary>
/// The game's <c>CardCreationFlags</c>, same values. They say which of <c>CardFactory</c>'s
/// steps and relic hooks a card creation takes part in.
/// </summary>
[Flags]
public enum CardCreationFlags
{
    None = 0,
    NoRarityModification = 1,
    NoUpgradeRoll = 2,
    NoHookUpgrades = 4,
    NoModifyHooks = 8,
    NoCardPoolModifications = 0x10,
    NoCardModelModifications = 0x20,
    ForceRarityOddsChange = 0x40,

    /// <summary><c>CardReward</c> adds this to its own options, whatever the event passed.</summary>
    IsCardReward = 0x80,
}

public static partial class RunRewardGenerator
{
    /// <summary>
    /// <c>CardFactory.CreateForReward(player, count, options)</c>: the one door the game's events
    /// and relics use to make a set of cards to choose from, with the relic hooks on it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Per card: a rarity roll (none on <c>Uniform</c> odds, which an event gets by filtering its
    /// pool to one rarity: <paramref name="uniformRarity" />), the pick, and an upgrade roll unless
    /// <c>NoUpgradeRoll</c>. Then, unless <c>NoModifyHooks</c>, <c>Hook.TryModifyCardRewardOptions</c>
    /// runs over the whole set: Silver Crucible upgrades every option of a <c>CardReward</c>
    /// (<c>IsCardReward</c>; three uses, one per set) and the egg relics upgrade their type
    /// (<c>NoHookUpgrades</c> turns them off) -- neither reads anything else, so an event's own
    /// grid gets the eggs but not the Crucible. Wing Charm enchants one option picked off
    /// <c>Rng.Niche</c>; Glitter's Glam and Fresnel Lens' Nimble land when the card is taken.
    /// Lava Lamp only upgrades in a combat room.
    /// </para>
    /// <para>
    /// A creation that skips this and just rolls cards drops the Crucible's upgrades, the eggs'
    /// upgrades and Wing Charm's Niche draw (Brain Leech's Rip with a Crucible: 36 real runs).
    /// </para>
    /// </remarks>
    /// <param name="pool">
    /// What the options' card pools leave after <c>ModifyCardRewardCreationOptions</c>;
    /// <see cref="OwnRewardPool" /> builds it for the player's own pool.
    /// </param>
    /// <param name="rngOverride"><c>CardCreationOptions.RngOverride</c> (the Crystal Sphere's).</param>
    internal static CardOffer CreateForReward(
        RunState state,
        int count,
        ReadOnlySpan<int> pool,
        CardCreationFlags flags,
        CardRarity? uniformRarity = null,
        GameRng? rngOverride = null
    )
    {
        var rng = rngOverride ?? state.PlayerRng.Rewards;
        var poolCards = pool.ToArray();
        var blacklist = new List<int>();
        var cards = new int[count];
        var upgraded = new bool[count];
        for (int i = 0; i < count; i++)
        {
            // RegularEncounter odds from a non-encounter source: the flat base odds, the running
            // rare offset neither read nor grown (CardFactory.RollForRarity).
            int rarity = uniformRarity switch
            {
                CardRarity.Common => RarityCommon,
                CardRarity.Uncommon => RarityUncommon,
                CardRarity.Rare => RarityRare,
                _ => RollCardRarity(
                    state,
                    RegularEncounterCardOdds(state),
                    mutateOffset: false,
                    state.PlayerRng.Rewards,
                    useOffset: false
                ),
            };
            cards[i] = ChooseCardWithRarity(poolCards, rarity, blacklist, rng);
            blacklist.Add(cards[i]);
            if (!flags.HasFlag(CardCreationFlags.NoUpgradeRoll))
            {
                upgraded[i] = RollCardUpgrade(state, cards[i], rng);
            }
        }

        return ApplyRewardOptionHooks(state, new CardOffer(cards, upgraded), flags);
    }

    /// <summary>
    /// The player's own card pool as <c>ModifyCardRewardCreationOptions</c> leaves it: Prismatic
    /// Gem and Dingy Rug widen it for a <c>CardReward</c> unless <c>NoCardPoolModifications</c>.
    /// </summary>
    internal static int[] OwnRewardPool(RunState state, CardCreationFlags flags) =>
        !flags.HasFlag(CardCreationFlags.NoCardPoolModifications)
        && flags.HasFlag(CardCreationFlags.IsCardReward)
            ? CardRewardPool(
                state.Character,
                Effects.RelicEffects.AddsColourlessToCardRewards(state),
                Effects.RelicEffects.WidensCardRewardsToEveryPool(state)
            )
            : OwnCardPool(state).ToArray();

    /// <summary>
    /// <c>Hook.TryModifyCardRewardOptions</c> and its Late pass over a finished set, then
    /// <c>AfterModifyingCardRewardOptions</c>.
    /// </summary>
    private static CardOffer ApplyRewardOptionHooks(
        RunState state,
        CardOffer offer,
        CardCreationFlags flags
    )
    {
        if (flags.HasFlag(CardCreationFlags.NoModifyHooks))
        {
            return offer;
        }

        var upgraded = (bool[])offer.Upgraded.Clone();
        // SilverCrucible.TryModifyCardRewardOptionsLate: IsCardReward, and a use left; the use is
        // counted for the set (AfterModifyingCardRewardOptions) whether or not anything upgraded.
        bool crucible =
            flags.HasFlag(CardCreationFlags.IsCardReward)
            && ConsumeSilverCrucibleCardRewardUpgrade(state);
        bool eggs = !flags.HasFlag(CardCreationFlags.NoHookUpgrades);
        for (int i = 0; i < offer.Cards.Length; i++)
        {
            var fresh = new CardInstance(offer.Cards[i], Upgraded: false);
            if (upgraded[i] || !RunConstants.IsRunCardUpgradable(fresh))
            {
                continue;
            }

            upgraded[i] =
                crucible || (eggs && RunNonCombatEffects.UpgradedByEggs(state, fresh).Upgraded);
        }

        // WingCharm.TryModifyCardRewardOptionsLate: one option that can take Swift, off Niche.
        int enchant = -1;
        if (Effects.RelicEffects.EnchantsACardReward(state))
        {
            var eligible = new List<int>();
            for (int i = 0; i < offer.Cards.Length; i++)
            {
                if (
                    Enchantments.CanEnchant(
                        new CardInstance(offer.Cards[i], upgraded[i]),
                        Enchantment.Swift
                    )
                )
                {
                    eligible.Add(i);
                }
            }

            if (eligible.Count > 0)
            {
                enchant = eligible[state.Rng.Niche.NextInt(eligible.Count)];
            }
        }

        return offer with { Upgraded = upgraded, EnchantIndex = enchant };
    }

    /// <summary>
    /// A card taken from a factory-made set, as the hooks left it: Wing Charm's Swift on its
    /// option, Glitter's Glam on every other option that can take it. The deck's own add
    /// (<see cref="RunNonCombatEffects.AddCardToDeck" />) then adds the eggs and Fresnel Lens.
    /// </summary>
    internal static CardInstance TakeFromRewardSet(
        RunState state,
        int cardId,
        bool upgraded,
        bool isSwiftOption
    )
    {
        var card = new CardInstance(cardId, upgraded);
        var enchantment = isSwiftOption
            ? Enchantment.Swift
            : Effects.RelicEffects.GlitterEnchantmentFor(state, card);
        return enchantment == Enchantment.None
            ? card
            : card with
            {
                Enchantment = enchantment,
                EnchantAmount = 1,
            };
    }

    /// <summary>One card of a factory-made set, straight into the deck.</summary>
    internal static void AddRewardCardToDeck(RunState state, CardOffer offer, int index) =>
        RunNonCombatEffects.AddCardToDeck(
            state,
            TakeFromRewardSet(
                state,
                offer.Cards[index],
                offer.Upgraded[index],
                offer.EnchantIndex == index
            )
        );

    /// <summary>A "choose N of these" grid, made by <see cref="CreateForReward" />.</summary>
    internal static void OpenRewardGrid(RunState state, CardOffer offer, int picks, bool hooked)
    {
        state.PendingOfferCards = offer.Cards;
        state.PendingOfferUpgraded = offer.Upgraded;
        state.PendingOfferSwiftIndex = offer.EnchantIndex;
        state.PendingOfferHooked = hooked;
        state.PendingOfferPicks = picks;
        state.Phase = RunPhase.TransformSelect;
    }
}
