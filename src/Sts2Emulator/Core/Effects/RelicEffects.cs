namespace Sts2Emulator.Core.Effects;

using Run;

public static class RelicEffects
{
    public const int Akabeko = 1;
    public const int Anchor = 4;
    public const int ArtOfWar = 7;
    public const int BagOfMarbles = 9;
    public const int BagOfPreparation = 10;
    public const int BlessedAntler = 21;
    public const int BigMushroom = 16;
    public const int BiiigHug = 17;
    public const int PaelsLegion = 181;

    // The unmodelled pass. Every one of these had an entry in `Relics.g.cs` and no id
    // constant, which means the game could hand it over and the emulator would do nothing
    // at all -- a relic that is in the run's list, shows in the observation, and is inert.
    public const int VeryHotCocoa = 283;
    public const int FencingManual = 86;
    public const int RunicCapacitor = 226;
    public const int SymbioticVirus = 257;
    public const int TwistedFunnel = 275;
    public const int OrangeDough = 171;
    public const int BigHat = 15;
    public const int PowerCell = 203;
    public const int Brimstone = 34;
    public const int NinjaScroll = 165;
    public const int FuneraryMask = 94;
    public const int ToughBandages = 269;
    public const int Tingsha = 264;
    public const int CharonsAshes = 45;
    public const int HelicalDart = 112;
    public const int PaperKrane = 187;
    public const int PaperPhrog = 188;
    public const int UndyingSigil = 277;
    public const int VitruvianMinion = 285;
    public const int SneckoSkull = 244;
    public const int RuinedHelmet = 225;
    public const int Regalite = 216;
    public const int LunarPastry = 143;
    public const int GoldPlatedCables = 106;
    public const int BoneFlute = 24;
    public const int BookRepairKnife = 28;
    public const int Bookmark = 26;
    public const int DemonTongue = 59;
    public const int EmotionChip = 73;
    public const int MiniRegent = 155;
    public const int GalacticDust = 96;
    public const int Metronome = 152;
    public const int LoomingFruit = 138;
    public const int FresnelLens = 92;

    // The fourteen an Act 1 run can be handed and that still did nothing, found when
    // `--reachable` turned out to mean "not in the event pool" rather than "reachable".
    public const int BoneTea = 25;
    public const int DarkstonePeriapt = 55;
    public const int DreamCatcher = 67;
    public const int EmberTea = 72;
    public const int HandDrill = 109;
    public const int HistoryCourse = 113;
    public const int LastingCandy = 130;
    public const int MawBank = 146;
    public const int RazorTooth = 213;
    public const int SparklingRouge = 246;
    public const int SwordOfJade = 255;
    public const int SwordOfStone = 256;
    public const int TeaOfDiscourtesy = 259;
    public const int TheBoot = 261;

    // The nineteen behind events the emulator ALREADY runs: the event arm hands the relic
    // over and the relic does nothing. Fake Merchant's whole stall was built two commits
    // before any of the nine it sells did anything.
    public const int FakeAnchor = 76;
    public const int FakeBloodVial = 77;
    public const int FakeHappyFlower = 78;
    public const int FakeLeesWaffle = 79;
    public const int FakeMango = 80;
    public const int FakeMerchantsRug = 81;
    public const int FakeOrichalcum = 82;
    public const int FakeSneckoEye = 83;
    public const int FakeStrikeDummy = 84;
    public const int FakeVenerableTeaSet = 85;
    public const int BingBong = 18;
    public const int DaughterOfTheWind = 57;
    public const int MrStruggles = 157;
    public const int ForgottenSoul = 90;
    public const int LostWisp = 141;
    public const int PollinousCore = 200;
    public const int RoyalPoison = 223;
    public const int WongoCustomerAppreciationBadge = 294;
    public const int WongosMysteryTicket = 295;
    public const int Byrdpip = 38;

    // Darv's eight -- Tier B of PLAN.md §7. Deck-warpers, all Ancient rarity.
    public const int Astrolabe = 8;
    public const int BlackStar = 20;
    public const int CallingBell = 39;
    public const int DustyTome = 69;
    public const int EmptyCage = 74;
    public const int PandorasBox = 185;
    public const int RunicPyramid = 227;
    public const int SneckoEye = 243;
    public const int PrismaticGem = 208;

    /// <summary>Pollinous Core's `DynamicVar("Turns", 4m)` and `CardsVar(2)`.</summary>
    public const int PollinousCoreTurns = 4;

    /// <summary>Fake Happy Flower's `DynamicVar("Turns", 5m)` -- the real one is three.</summary>
    public const int FakeHappyFlowerTurns = 5;

    /// <summary>Wongo's Mystery Ticket: five combats, then three relics.</summary>
    public const int WongosTicketCombats = 5;

    public const int WongosTicketRelics = 3;

    /// <summary>
    /// Relics that declare no behaviour at all -- an empty `RelicModel` body. Named so the
    /// audit can tell "nobody wrote this" from "there is nothing to write", which are the
    /// same thing from the outside and opposite things to a reader.
    /// </summary>
    public static readonly int[] NoEffectRelics =
    [
        FakeMerchantsRug,
        WongoCustomerAppreciationBadge,
    ];

    /// <summary>Bone Tea's `Combats` var: one fight, then it is used up.</summary>
    public const int BoneTeaCombats = 1;

    /// <summary>Ember Tea's: five fights of Strength 2.</summary>
    public const int EmberTeaCombats = 5;

    /// <summary>Tea of Discourtesy's: one fight, and its `DazedCount` of two.</summary>
    public const int TeaOfDiscourtesyCombats = 1;

    public const int TeaOfDiscourtesyDazed = 2;

    /// <summary>Sword of Stone's `DynamicVar("Elites", 5m)` -- five, not three.</summary>
    public const int SwordOfStoneElites = 5;
    public const int BloodVial = 23;
    public const int BoomingConch = 29;
    public const int BronzeScales = 35;
    public const int CaptainsWheel = 41;
    public const int CentennialPuzzle = 43;
    public const int CloakClasp = 51;
    public const int Ectoplasm = 70;
    public const int DataDisk = 56;
    public const int FestivePopper = 87;
    public const int Gorget = 107;
    public const int HappyFlower = 110;
    public const int HornCleat = 114;
    public const int IvoryTile = 119;
    public const int Kunai = 126;
    public const int Kusarigama = 127;
    public const int GremlinHorn = 108;
    public const int Lantern = 128;
    public const int LetterOpener = 136;
    public const int LizardTail = 137;
    public const int MealTicket = 147;
    public const int MummifiedHand = 158;
    public const int Nunchaku = 166;
    public const int OddlySmoothStone = 169;
    public const int Orichalcum = 172;
    public const int OrnamentalFan = 173;
    public const int ParryingShield = 189;
    public const int Pendulum = 191;
    public const int Pocketwatch = 199;
    public const int PhilosophersStone = 196;
    public const int Permafrost = 193;
    public const int RedMask = 214;
    public const int RedSkull = 215;
    public const int RegalPillow = 217;
    public const int ScreamingFlagon = 230;
    public const int SelfFormingClay = 234;
    public const int Shuriken = 237;
    public const int Sozu = 245;
    public const int SpikedGauntlets = 247;
    public const int StoneCracker = 249;
    public const int TinyMailbox = 265;
    public const int StoneCalendar = 248;
    public const int Vajra = 279;
    public const int TuningFork = 274;
    public const int VelvetChoker = 281;

    // The shared pool's commons and uncommons, added with the batch that modelled them.
    public const int BookOfFiveRings = 27;
    public const int BowlerHat = 31;
    public const int Candelabra = 40;
    public const int EternalFeather = 75;
    public const int JossPaper = 122;
    public const int LuckyFysh = 142;
    public const int MercuryHourglass = 151;
    public const int MiniatureCannon = 153;
    public const int PenNib = 192;
    public const int PetrifiedToad = 194;
    public const int PotionBelt = 202;
    public const int ReptileTrinket = 218;
    public const int RippleBasin = 222;
    public const int StrikeDummy = 253;
    public const int Vambrace = 280;
    public const int VenerableTeaSet = 282;
    public const int AmethystAubergine = 3;

    // The shared pool's rares.
    public const int BeatingRemnant = 11;
    public const int Girya = 100;
    public const int GamblingChip = 97;

    // The four Starter relics, one per character. Every run of that character holds one.
    public const int BoundPhylactery = 30;
    public const int CrackedCore = 52;
    public const int DivineRight = 64;
    public const int RingOfTheSnake = 221;

    // The shop pool.
    public const int BeltBuckle = 14;
    public const int Cauldron = 42;
    public const int DingyRug = 61;
    public const int DragonFruit = 66;
    public const int DollysMirror = 65;
    public const int LavaLamp = 131;
    public const int Orrery = 174;
    public const int WingCharm = 292;
    public const int GnarledHammer = 103;
    public const int Kifuda = 125;
    public const int PunchDagger = 210;
    public const int RoyalStamp = 224;
    public const int Bread = 32;

    /// <summary>`Bread`'s `EnergyVar("LoseEnergy", 2)`, charged on turn one.</summary>
    public const int BreadTurnOneEnergyLoss = 2;
    public const int BurningSticks = 37;
    public const int ChemicalX = 46;
    public const int GhostSeed = 99;
    public const int MiniatureTent = 154;
    public const int MysticLighter = 160;
    public const int RingingTriangle = 219;
    public const int SlingOfCourage = 241;
    public const int TheAbacus = 260;
    public const int Toolbox = 267;
    public const int UnsettlingLamp = 278;
    public const int Shovel = 236;
    public const int Bellows = 13;
    public const int Chandelier = 44;
    public const int GamePiece = 98;
    public const int IceCream = 115;
    public const int IntimidatingHelmet = 117;
    public const int PrayerWheel = 204;
    public const int RainbowRing = 212;
    public const int SturdyClamp = 254;
    public const int TheCourier = 262;
    public const int TungstenRod = 273;
    public const int UnceasingTop = 276;
    public const int VexingPuzzlebox = 284;
    public const int WhiteStar = 291;
    public const int JuzuBracelet = 123;
    public const int Pantograph = 186;
    public const int Planisphere = 198;

    // The Ancient/Starter pass. Ten more that `Relics.g.cs` knew about and that had no id
    // constant, so the run could hand them over and the emulator would do nothing.
    public const int Sai = 228;
    public const int DivineDestiny = 63;
    public const int SealOfGold = 233;
    public const int PaelsBlood = 175;
    public const int RingOfTheDrake = 220;
    public const int BloodSoakedRose = 22;
    public const int JeweledMask = 120;
    public const int DelicateFrond = 58;
    public const int SignetRing = 238;
    public const int Driftwood = 68;
    public const int ThrowingAxe = 263;
    public const int Crossbow = 53;
    public const int PaelsTears = 182;
    public const int RadiantPearl = 211;
    public const int InfusedCore = 116;
    public const int PhylacteryUnbound = 197;
    public const int ToastyMittens = 266;
    public const int PaelsFlesh = 178;
    public const int WhisperingEarring = 289;
    public const int TanxsWhistle = 258;
    public const int JewelryBox = 121;
    public const int IronClub = 118;
    public const int WarHammer = 286;
    public const int Fiddle = 88;

    /// <summary>`IronClub`'s CardsVar(4): every fourth card played draws one.</summary>
    private const int IronClubCards = 4;

    /// <summary>`SealOfGold`'s GoldVar(5) -- the toll it takes for its energy.</summary>
    private const int SealOfGoldPrice = 5;

    /// <summary>`RingOfTheDrake`'s DynamicVar("Turns", 3): the last turn it still pays.</summary>
    private const int RingOfTheDrakeTurns = 3;

    /// <summary>The token potion Petrified Toad procures before every combat.</summary>
    private const int PotionShapedRock = 45;

    /// <summary>`DiamondDiadem`: two cards or fewer this turn and the next enemy turn hits at half.</summary>
    public const int DiamondDiadem = 60;

    /// <summary>`BrilliantScarf`: the FIFTH card each turn is free.</summary>
    public const int BrilliantScarf = 33;

    /// <summary>`MusicBox`: the first Attack each turn is copied into hand, Ethereal.</summary>
    public const int MusicBox = 159;

    /// <summary>`ChoicesParadox`: five generated cards on turn one, one of them kept, Retained.</summary>
    public const int ChoicesParadox = 47;

    /// <summary>Diamond Diadem's `DynamicVar("CardThreshold", 2m)`.</summary>
    private const int DiamondDiademCardThreshold = 2;

    /// <summary>Brilliant Scarf's `CardsVar(5)`: the card AFTER four is the free one.</summary>
    private const int BrilliantScarfCards = 5;

    /// <summary>Choices Paradox's `CardsVar(5)`: five offered, one kept.</summary>
    private const int ChoicesParadoxCards = 5;

    /// <summary>
    /// The relics whose ModifyMaxEnergy adds an EnergyVar(1). Every one of them is the same
    /// +1; what separates them is the price, which each pays through a different hook.
    /// </summary>
    private static readonly int[] MaxEnergyRelics =
    [
        Ectoplasm,
        Sozu,
        SpikedGauntlets,
        VelvetChoker,
        PhilosophersStone,
        BlessedAntler,
        // `PrismaticGem.ModifyMaxEnergy`: EnergyVar(1). Its OTHER half widens the card
        // reward pool -- see `WidensCardRewardsToEveryPool`.
        PrismaticGem,
        // `BloodSoakedRose.ModifyMaxEnergy`: EnergyVar(1), and its price is an Enthralled
        // shuffled into the deck on pickup -- a run-level cost for a combat-level gain,
        // which is why only this half lives here.
        BloodSoakedRose,
        // `WhisperingEarring.ModifyMaxEnergy`: EnergyVar(1). Its OTHER half auto-plays up
        // to thirteen cards on turn one -- modelled since E486, in
        // `CombatEngine.RunWhisperingEarringAutoPlay`. This entry is the ENERGY only, the
        // same as Prismatic Gem's and Blood-Soaked Rose's above.
        WhisperingEarring,
    ];

    /// <summary>The game's DynamicVar("Cards", 6) on Velvet Choker.</summary>
    private const int VelvetChokerCardLimit = 6;

    /// <summary>
    /// Velvet Choker's ShouldPlay: once six cards have been played this turn, nothing else
    /// can be -- auto-played cards included (`CombatEngine.BeginAutoPlay`).
    /// </summary>
    public static bool BlocksFurtherCardPlays(CombatState state) =>
        (HasRelic(state, VelvetChoker) && state.CardPlaysThisTurn >= VelvetChokerCardLimit)
        // `Normality.ShouldPlay` is false once THREE cards have been played this turn, and
        // only while the curse is in HAND -- so drawing it late in a turn can stop the
        // turn dead, and shuffling it away costs nothing. Velvet Choker's rule with the
        // clock on a card instead of a relic, which is why they share this door. Its clock
        // is `CardPlaysStarted` (the play in progress counts), the Choker's AfterCardPlayed.
        || (
            state.Hand.Any(card => card.DefId == ST.Normality)
            && state.CardPlaysStartedThisTurn >= NormalityCardLimit
        );

    /// <summary>Normality's `_numOfCardsPerTurn`.</summary>
    private const int NormalityCardLimit = 3;

    /// <summary>
    /// Spiked Gauntlets' TryModifyEnergyCostInCombat: Powers cost one more. Returned as an
    /// addend so the caller keeps owning the rest of the cost rules.
    /// </summary>
    public static int ExtraEnergyCost(CombatState state, CardDef def) =>
        def.Type == CardType.Power && HasRelic(state, SpikedGauntlets) ? 1 : 0;

    /// <summary>
    /// `Glitter.TryModifyCardRewardOptionsLate`: every card reward the Glam enchantment can
    /// take gets it, at amount 1. Not one option chosen off a stream like Wing Charm's --
    /// ALL of them, which is why this is asked per-card and draws nothing.
    /// </summary>
    /// <remarks>
    /// `SilkenTress.TryModifyCardRewardOptionsLate` does the same to the FIRST card reward
    /// built after it is obtained, then sets `IsUsed` (`AfterModifyingCardRewardOptions`).
    /// Its Counter carries that: 0 unused, 1 the screen on offer is its one, 2 spent
    /// (<see cref="Run.RunRewardGenerator.PopulateCardReward"/> and the card screen's answer).
    /// </remarks>
    public static Enchantment GlitterEnchantmentFor(Run.RunState state, CardInstance card) =>
        state.Relics.Any(relic => relic.DefId == Glitter || (relic.DefId == SilkenTress && relic.Counter == 1))
        && Enchantments.CanEnchant(card, Enchantment.Glam)
            ? Enchantment.Glam
            : Enchantment.None;

    /// <summary>`SilkenTress`: loses all gold on pickup; the next card reward arrives Glam-enchanted.</summary>
    public const int SilkenTress = Run.RunConstants.RelicSilkenTress;

    /// <summary>A card reward screen is being built: Silken Tress claims it if still unused.</summary>
    internal static void MarkSilkenTressScreen(Run.RunState state)
    {
        // Only the first screen built after pickup: a second build (a reroll, the next reward)
        // finds IsUsed already set.
        int i = state.Relics.FindIndex(relic => relic.DefId == SilkenTress);
        if (i >= 0 && state.Relics[i].Counter < 2)
        {
            state.Relics[i] = state.Relics[i] with { Counter = state.Relics[i].Counter + 1 };
        }
    }

    /// <summary>The card screen Silken Tress enchanted has been answered: it is used up.</summary>
    internal static void SpendSilkenTress(Run.RunState state)
    {
        int i = state.Relics.FindIndex(relic => relic.DefId == SilkenTress);
        if (i >= 0 && state.Relics[i].Counter == 1)
        {
            state.Relics[i] = state.Relics[i] with { Counter = 2 };
        }
    }

    /// <summary>`Glitter`: every eligible card reward arrives Glam-enchanted.</summary>
    public const int Glitter = 102;

    /// <summary>`PaelsEye`: one wasted turn a combat is refunded as an extra turn.</summary>
    public const int PaelsEye = 177;

    /// <summary>
    /// `PaelsEye.ShouldTakeExtraTurn`: not used this combat, and NO cards played this turn.
    /// Consumes the relic for the rest of the fight when it fires -- `AfterTakingExtraTurn`
    /// sets `UsedThisCombat`, which rides the relic's Counter here.
    /// </summary>
    /// <remarks>
    /// `WasOwnerPartOfLastPlayerTurn` is the multiplayer half: it exists so a player who
    /// was auto-ended out of a turn they never took does not bank it. Solo, the owner is
    /// part of every player turn, so it is always true.
    ///
    /// `AnyCardsPlayedThisTurn` returns TRUE on turn one for a run holding Whispering
    /// Earring, whether or not the earring played anything -- the game refuses to let the
    /// two relics combine rather than counting the auto-plays.
    ///
    /// The hand is exhausted by `BeforeSideTurnEndEarly`, which is a separate hook and
    /// fires whether or not the extra turn is granted -- but its guard is the same, so in
    /// practice the two go together. See `ExhaustHandForPaelsEye`.
    /// </remarks>
    public static bool TakeExtraTurnForPaelsEye(CombatState state)
    {
        if (!PaelsEyeWouldFire(state))
        {
            return false;
        }

        SetCounter(state, PaelsEye, 1);
        return true;
    }

    private static bool PaelsEyeWouldFire(CombatState state) =>
        HasRelic(state, PaelsEye)
        && GetCounter(state, PaelsEye) == 0
        && state.CardsPlayedThisTurn == 0
        && !(state.Turn == 0 && HasRelic(state, WhisperingEarring));

    /// <summary>
    /// `PaelsEye.BeforeSideTurnEndEarly`: the hand is exhausted -- the price of the extra
    /// turn, paid before the end-of-turn effects the hand might otherwise survive.
    /// </summary>
    public static bool ShouldExhaustHandForPaelsEye(CombatState state) => PaelsEyeWouldFire(state);

    /// <summary>
    /// `MusicBox.BeforeCardPlayed` latches the first ATTACK of the turn and its
    /// `AfterCardPlayed` puts a CLONE of it in hand, Ethereal. Both halves collapse to
    /// one check here because the emulator plays a card in one step: the latch exists in
    /// the game only to survive the await between them.
    /// </summary>
    /// <remarks>
    /// A CLONE, so the copy carries the original's upgrade and enchantment -- and its own
    /// Ethereal grant on top, which is per-copy rather than on the definition.
    /// </remarks>
    public static void ApplyMusicBox(CombatState state, CardInstance card, CardDef def)
    {
        if (!HasRelic(state, MusicBox) || state.MusicBoxUsedThisTurn || def.Type != CardType.Attack)
        {
            return;
        }

        state.MusicBoxUsedThisTurn = true;
        state.Hand.Add(state.Fresh(card) with { EtherealForCombat = true });
    }

    /// <summary>
    /// `BrilliantScarf.ShouldModifyCost`: <c>CardsPlayedThisTurn == Cards.BaseValue - 1</c>,
    /// so with CardsVar(5) it is the card played when four are already down -- the FIFTH.
    /// Both `TryModifyEnergyCostInCombatLate` and `TryModifyStarCost` return zero, so the
    /// card is free in both resources rather than half free.
    /// </summary>
    public static bool IsFreeUnderBrilliantScarf(CombatState state) =>
        HasRelic(state, BrilliantScarf) && state.CardsPlayedThisTurn == BrilliantScarfCards - 1;

    /// <summary>Ectoplasm's ModifyGoldGained returns 0m: the owner gains no gold, ever.</summary>
    public static int ModifyGoldGained(IEnumerable<RelicInstance> relics, int amount)
    {
        var held = relics as IReadOnlyCollection<RelicInstance> ?? relics.ToList();
        if (held.Any(relic => relic.DefId == Ectoplasm))
        {
            return 0;
        }

        // `BowlerHat.ModifyGoldGained` multiplies by a `DynamicVar("GoldIncrease", 1.25m)`.
        // A decimal multiply, so 15 gold becomes 18 and not 19 -- the truncation is the
        // game's, not a rounding choice made here.
        if (held.Any(relic => relic.DefId == BowlerHat))
        {
            return (int)(amount * 1.25m);
        }

        return amount;
    }

    /// <summary>
    /// `DragonFruit.AfterGoldGained`: 1 max HP, through `CreatureCmd.GainMaxHp` -- so it
    /// HEALS 1 as well as raising the cap.
    ///
    /// Once per gain EVENT, not per gold: a hundred from a boss is the same +1 as fifteen
    /// from a mushroom. And `PlayerCmd.GainGold` returns on `!(amount > 0m)` BEFORE
    /// reaching the hook, which is why both chokepoints drop out on a zeroed amount --
    /// Ectoplasm suppresses this relic entirely rather than merely taking the gold.
    /// </summary>
    internal static void ApplyAfterGoldGained(Run.RunState state)
    {
        if (Has(state.Relics, DragonFruit))
        {
            Run.RunNonCombatEffects.GainMaxHp(state, 1);
        }
    }

    /// <inheritdoc cref="ApplyAfterGoldGained(Run.RunState)" />
    internal static void ApplyAfterGoldGained(CombatState state)
    {
        if (HasRelic(state, DragonFruit))
        {
            CardEffects.GainMaxHp(state, 1);
        }
    }

    /// <summary>
    /// `BoundPhylactery.AfterEnergyResetLate`: Osty is re-summoned at the start of every
    /// turn EXCEPT the first, whose summon `BeforeCombatStart` already did.
    ///
    /// The game picks `AfterEnergyResetLate` deliberately and says why: anything that
    /// asks whether Osty exists -- Friendship is the one it names -- has to run BEFORE
    /// the summon, or the relic answers its own question. On a living Osty the summon is
    /// `GainMaxHp`, so turn after turn it is +1 rather than a fresh pet.
    /// </summary>
    internal static void ApplyBoundPhylacteryTurnStart(CombatState state, int turnNumber)
    {
        if (turnNumber != 1 && HasRelic(state, BoundPhylactery))
        {
            CardEffects.SummonOsty(state, 1);
        }
    }

    /// <summary>
    /// Relics whose combat counter means "spent for the rest of the run" rather than
    /// something a fresh combat resets. The run carries these in RunState.UsedUpRelics.
    /// </summary>
    private static readonly int[] OncePerRunRelics = [LizardTail];

    /// <summary>Marks relics the run has already spent, so a new combat cannot reuse them.</summary>
    public static void RestoreUsedUpRelics(CombatState state, IEnumerable<int> usedUp)
    {
        foreach (int relicId in usedUp)
        {
            SetCounter(state, relicId, 1);
        }
    }

    /// <summary>Reports which one-per-run relics this combat spent.</summary>
    public static void CollectUsedUpRelics(CombatState state, List<int> usedUp)
    {
        foreach (int relicId in OncePerRunRelics)
        {
            int index = state.Relics.FindIndex(relic => relic.DefId == relicId);
            if (index >= 0 && state.Relics[index].Counter > 0 && !usedUp.Contains(relicId))
            {
                usedUp.Add(relicId);
            }
        }
    }

    /// <summary>
    /// Runs the relics of <paramref name="hooks"/> in the order the player holds them, as the
    /// game's hook dispatch does: one pass over the held relics, each relic's action at most once.
    /// </summary>
    private static void RunHookInHoldingOrder(CombatState state, Dictionary<int, Action> hooks)
    {
        foreach (int id in state.Relics.Select(relic => relic.DefId).ToList())
        {
            if (hooks.TryGetValue(id, out var action))
            {
                action();
            }
        }
    }

    /// <summary>
    /// Everything the game does for turn one ahead of the opening draw, in its phase order:
    /// `AfterRoomEntered`, `BeforeCombatStart`, `BeforeCombatStartLate`, the side's
    /// `BeforeSideTurnStart`, then the energy reset with `AfterEnergyReset`. Each phase is one pass
    /// over the held relics. The post-draw half (`AfterSideTurnStart`) is
    /// <see cref="ApplyFirstTurnAfterSideTurnStart" />.
    /// </summary>
    public static void ApplyBeforeOpeningHand(CombatState state, Random rng)
    {
        // Phase 1, `AfterRoomEntered`. The powers they apply land in the player's power list in
        // this order.
        RunHookInHoldingOrder(state, new Dictionary<int, Action>
        {
            // `StoneCracker`: CardsVar(2) of the upgradable cards, `StableShuffle` of
            // `Rng.CombatCardSelection` over them, so the roll does not depend on pile order.
            [StoneCracker] = () =>
            {
                var shuffled = CardEffects.StableShuffle(
                    state.DrawPile,
                    Enumerable
                        .Range(0, state.DrawPile.Count)
                        .Where(i => CardEffects.IsUpgradable(state.DrawPile[i])),
                    CardEffects.CardSelectionRngFor(state, rng)
                );
                CardEffects.UpgradeCards(state.DrawPile, shuffled.Take(2));
            },
            // Elite rooms only: the relic's hook fires before the combat exists, so the room
            // type is handed over rather than looked up -- `CombatFactory.Reset` sets
            // `IsEliteRoom` from the room before it runs this phase.
            [SlingOfCourage] = () =>
            {
                if (state.IsEliteRoom)
                {
                    GainPlayerStrength(state, 2);
                }
            },
            // Strength equal to the lifts spent on the relic instance's counter.
            [Girya] = () =>
            {
                int lifts = state.Relics.FirstOrDefault(relic => relic.DefId == Girya).Counter;
                if (lifts > 0)
                {
                    GainPlayerStrength(state, lifts);
                }
            },
            // Stars live on the PlayerCombatState: three at the top of every fight.
            [DivineRight] = () => CardEffects.GainStars(state, 3),
            // The counter IS the once-per-combat flag, cleared here and not at turn start.
            [ThrowingAxe] = () => SetCounter(state, ThrowingAxe, 0),
            // Strength 1 on every living opponent (a mid-combat summon is the game's
            // AfterCreatureAddedToCombat, handled where enemies spawn).
            [PhilosophersStone] = () => ApplyToLivingEnemies(state, BuffId.Strength),
            [EmberTea] = () =>
            {
                if (SpendTeaCombat(state, EmberTea))
                {
                    GainPlayerStrength(state, 2);
                }
            },
            [SwordOfJade] = () => GainPlayerStrength(state, 3),
            [Vajra] = () => GainPlayerStrength(state, 1),
            [OddlySmoothStone] = () => BuffSystem.Apply(state.PlayerBuffs, BuffId.Dexterity, 1),
            [DataDisk] = () => BuffSystem.Apply(state.PlayerBuffs, BuffId.Focus, 1),
            [Gorget] = () => BuffSystem.Apply(state.PlayerBuffs, BuffId.Plating, 4),
            [BronzeScales] = () => BuffSystem.Apply(state.PlayerBuffs, BuffId.Thorns, 3),
        });

        // Phase 2, `BeforeCombatStart`.
        RunHookInHoldingOrder(state, new Dictionary<int, Action>
        {
            // BlockVar(10, Unpowered): Dexterity does not raise it. The fake is 4.
            [Anchor] = () => CardEffects.GainUnpoweredBlock(state, 10, rng),
            [FakeAnchor] = () => CardEffects.GainUnpoweredBlock(state, 4, rng),
            // ConfusedPower at 1, `PowerStackType.Single`, so holding both is still one; up before
            // the opening hand, whose costs are the first to be re-rolled.
            [SneckoEye] = () => BuffSystem.Apply(state.PlayerBuffs, BuffId.Confused, 1),
            [FakeSneckoEye] = () =>
            {
                if (!HasRelic(state, SneckoEye))
                {
                    BuffSystem.Apply(state.PlayerBuffs, BuffId.Confused, 1);
                }
            },
            // Two Dazed into the DRAW pile at random positions, for one combat.
            [TeaOfDiscourtesy] = () =>
            {
                if (SpendTeaCombat(state, TeaOfDiscourtesy))
                {
                    CardEffects.AddCardToDrawPileRandomly(
                        state,
                        ST.Dazed,
                        TeaOfDiscourtesyDazed,
                        state.ShuffleRng ?? rng
                    );
                }
            },
            [PhylacteryUnbound] = () => CardEffects.SummonOsty(state, 5),
            [BoundPhylactery] = () => CardEffects.SummonOsty(state, 1),
        });
        RefreshBeltBuckle(state);

        // Phase 3, `BeforeCombatStartLate`: the first free potion slot, and nothing when the belt
        // is full or Sozu's ShouldProcurePotion says no.
        RunHookInHoldingOrder(state, new Dictionary<int, Action>
        {
            [PetrifiedToad] = () => PotionEffects.TryToProcure(state, PotionShapedRock),
        });

        // Phase 4, the side's `BeforeSideTurnStart`, turn one: the guards on `TurnNumber <= 1`
        // make these combat-start openers.
        RunHookInHoldingOrder(state, new Dictionary<int, Action>
        {
            [CrackedCore] = () => CardEffects.ChannelOrb(state, OrbType.Lightning, rng),
            [BagOfMarbles] = () => ApplyToLivingEnemies(state, BuffId.Vulnerable),
            [RedMask] = () => ApplyToLivingEnemies(state, BuffId.Weak),
            [TwistedFunnel] = () => CardEffects.ApplyPoisonToAllEnemies(state, 4, rng),
            // CardsVar(2) ZERO-COST cards out of the draw pile and into hand, ahead of the
            // opening draw, which then draws five more.
            [PowerCell] = () => CardEffects.MoveZeroCostDrawCardsToHandForPowerCell(state, 2, rng),
            // `PollinousCore.BeforeSideTurnStart` counts the turn, turn one included: it has no
            // guard, and `StartCombatInternal` runs the same `StartTurn` as every later turn. The
            // count is the relic's saved `TurnsSeen`, which no fight resets.
            [PollinousCore] = () => TickPollinousCore(state),
        });

        // Phase 5, the energy reset: ModifyMaxEnergy is read whenever the game asks for the
        // maximum, so the bonus reaches MaxEnergy and the energy already handed out for turn one.
        // Not through GainEnergy: it is the reset catching up, which NoEnergyGain does not touch.
        int extraEnergy = state.Relics.Count(relic => MaxEnergyRelics.Contains(relic.DefId));
        // `PumpkinCandle.ModifyMaxEnergy` only while lit: KindleCount rides the relic's Counter.
        extraEnergy += state.Relics.Count(relic =>
            relic.DefId == Run.RunConstants.RelicPumpkinCandle && relic.Counter > 0
        );
        state.MaxEnergy += extraEnergy;
        state.Energy += extraEnergy;

        // `AfterEnergyReset`: the first reset of the combat after a REST SITE. The armed state is a
        // counter on the relic, set when the run rests. The fake is the same rule at 1 energy.
        RunHookInHoldingOrder(state, new Dictionary<int, Action>
        {
            [VenerableTeaSet] = () =>
            {
                if (SpendRestSiteCharge(state, VenerableTeaSet))
                {
                    CardEffects.GainEnergy(state, 2);
                }
            },
            [FakeVenerableTeaSet] = () =>
            {
                if (SpendRestSiteCharge(state, FakeVenerableTeaSet))
                {
                    CardEffects.GainEnergy(state, 1);
                }
            },
        });

        // The turn-one `BeforeHandDraw` relics (Blessed Antler, Ninja Scroll, Toolbox and the
        // rest) are not here: they are turn one's visit to `ApplyBeforeHandDraw`, which the
        // turn start makes in the order the player holds them.
    }

    private static void ApplyToLivingEnemies(CombatState state, BuffId buff)
    {
        foreach (var enemy in state.Enemies.Where(enemy => enemy.Hp > 0))
        {
            BuffSystem.Apply(enemy.Buffs, buff, 1);
        }
    }

    /// <summary>
    /// Turn one's `AfterSideTurnStart` relics with a `TurnNumber <= 1` guard, which the game runs
    /// after the opening draw and `AfterPlayerTurnStart`: Bread, Divine Destiny, Infused Core,
    /// Very Hot Cocoa, Fencing Manual, Runic Capacitor, Symbiotic Virus, Orange Dough, Big Hat,
    /// Bone Tea, Booming Conch (its energy), Lantern and Akabeko.
    /// </summary>
    internal static void ApplyFirstTurnAfterSideTurnStart(CombatState state, Random rng)
    {
        RunHookInHoldingOrder(state, new Dictionary<int, Action>
        {
            // `PlayerCmd.LoseEnergy(2)` after the reset, clamped at zero: the price of its +1.
            [Lantern] = () => CardEffects.GainEnergy(state, 1),
            [Akabeko] = () => BuffSystem.Apply(state.PlayerBuffs, BuffId.Vigor, 8),
            [Bread] = () => state.Energy = Math.Max(0, state.Energy - BreadTurnOneEnergyLoss),
            [DivineDestiny] = () => CardEffects.GainStars(state, 6),
            [InfusedCore] = () =>
            {
                for (int i = 0; i < 3; i++)
                {
                    CardEffects.ChannelOrb(state, OrbType.Lightning, rng);
                }
            },
            [VeryHotCocoa] = () => CardEffects.GainEnergy(state, 4),
            [FencingManual] = () => CardEffects.Forge(state, 10),
            [RunicCapacitor] = () => state.OrbCapacity += 3,
            [SymbioticVirus] = () => CardEffects.ChannelOrb(state, OrbType.Dark, rng),
            [OrangeDough] = () => CardEffects.AddColorlessCardsToHand(state, 2, rng),
            // Distinct ETHEREAL cards from the player's own pool; skipped when the pool has none.
            [BigHat] = () => CardEffects.AddDistinctEtherealCardsToHand(state, 2, rng),
            // UPGRADE EVERY CARD IN HAND, one combat only.
            [BoneTea] = () =>
            {
                if (SpendTeaCombat(state, BoneTea))
                {
                    CardEffects.UpgradeAllCards(state.Hand);
                }
            },
            [BoomingConch] = () =>
            {
                if (state.IsEliteRoom)
                {
                    CardEffects.GainEnergy(state, 1);
                }
            },
        });
    }

    /// <summary>
    /// Both halves of turn one's combat start in one call (the pre-draw phases, then the
    /// `AfterSideTurnStart` ones). The engine runs them apart, around the opening draw.
    /// </summary>
    public static void ApplyCombatStart(CombatState state, Random rng)
    {
        ApplyBeforeOpeningHand(state, rng);
        ApplyFirstTurnAfterSideTurnStart(state, rng);
    }

    /// <summary>
    /// Hook.AfterCreatureAddedToCombat: an enemy that joins a combat in progress gets the
    /// same treatment as one that started it. Returns the enemy so a spawn site can wrap
    /// its own construction call rather than remembering to follow it with this.
    /// </summary>
    public static EnemyState Spawned(CombatState state, EnemyState enemy)
    {
        if (HasRelic(state, PhilosophersStone))
        {
            BuffSystem.Apply(enemy.Buffs, BuffId.Strength, 1);
        }

        // `FurCoat.AfterCreatureAddedToCombat`: in a marked room the coat bites whatever
        // joins, exactly as `BeforeCombatStart` bit the roster it opened on. Reinforcements
        // are the reason the relic needs both hooks -- a Decimillipede that reattaches, or
        // an egg that hatches, would otherwise arrive at full health in a fight where
        // nothing else did.
        BiteForFurCoat(state, enemy);

        return enemy;
    }

    /// <summary>
    /// `CreatureCmd.SetCurrentHp(creature, 1m)` on a hittable enemy, in a Fur Coat room.
    /// </summary>
    /// <remarks>
    /// `HittableEnemies` is the filter on both of the coat's hooks, so a creature that is
    /// not a target yet -- a reviving corpse, a Steam Eruption remnant -- is passed over
    /// rather than pinned at one hp while it is out of reach.
    /// </remarks>
    public static void BiteForFurCoat(CombatState state, EnemyState enemy)
    {
        if (state.FurCoatBites && EnemyAI.IsTargetable(enemy))
        {
            if (Core2.Kernel.If(Core2.Switch.F08, state) is { } core2)
            {
                Core2.CreatureCmd.SetCurrentHp(core2, Core2.OldCreature.Of(enemy), 1);
            }
            else
            {
                enemy.Hp = 1;
            }
        }
    }

    public static void ApplyStartOfPlayerTurn(CombatState state, Random? rng = null)
    {
        int turnNumber = state.Turn + 1;

        // `MusicBox.BeforeSideTurnStart` clears its used flag -- the box is once a TURN,
        // not once a combat.
        // `ChainsOfBindingPower.BeforeSideTurnEnd` clears the played latch AND strips Bound
        // off every card the player owns -- the affliction lasts the turn it was dealt in,
        // so the next turn's draws are bound afresh rather than inheriting it.
        state.BoundCardPlayedThisTurn = false;
        state.BoundCardsThisTurn = 0;
        CardEffects.ClearBoundCards(state);

        state.MusicBoxUsedThisTurn = false;

        // Every "N cards played this turn" relic counts from zero again; the game resets
        // them in BeforeSideTurnStart, which is this same boundary.
        foreach (int relicId in PerTurnCounters)
        {
            SetCounter(state, relicId, 0);
        }

        // Sai, Pael's Tears, Phylactery Unbound, Crossbow, Seal of Gold, Brimstone and the
        // flowers are `AfterSideTurnStart` relics: `ApplyAfterSideTurnStart`, after the turn
        // start's kills.

        // `MiniRegent` and `DemonTongue` both clear a once-per-turn flag in
        // `BeforeSideTurnStart`; the counter IS that flag.
        SetCounter(state, MiniRegent, 0);
        SetCounter(state, DemonTongue, 0);

        // Both are BlockVar(..., ValueProp.Unpowered): Dexterity does not raise them.
        if (turnNumber == 2 && HasRelic(state, HornCleat))
        {
            CardEffects.GainUnpoweredBlock(state, 14, rng);
        }

        if (turnNumber == 3 && HasRelic(state, CaptainsWheel))
        {
            CardEffects.GainUnpoweredBlock(state, 18, rng);
        }

        // Both of these ask what the player did LAST turn, and the engine clears its
        // per-turn tallies after this hook — so this is the one moment they still hold it.
        if (turnNumber > 1 && HasRelic(state, ArtOfWar) && state.AttackCardsPlayedThisTurn == 0)
        {
            // AfterEnergyReset: the energy arrives on a turn following one with no Attack.
            CardEffects.GainEnergy(state, 1);
        }

        // Pocketwatch's ModifyHandDraw runs after the tallies are cleared, so the verdict
        // is taken here and parked on the relic for the draw to read.
        SetCounter(
            state,
            Pocketwatch,
            turnNumber > 1 && state.CardPlaysThisTurn <= PocketwatchCardThreshold ? 1 : 0
        );
    }

    /// <summary>Pocketwatch's DynamicVar("CardThreshold", 3m) and CardsVar(3).</summary>
    private const int PocketwatchCardThreshold = 3;

    /// <summary>
    /// The cards the hand draw of turn <paramref name="turnNumber"/> owes to relics on top of the
    /// five: the sum of the relics' `ModifyHandDraw` (Fiddle's is the Late pass), ONE function for
    /// every turn, turn one included.
    /// </summary>
    /// <remarks>
    /// The game has no separate first-turn draw: `SetupPlayerTurn` asks every relic on turn ten
    /// what it asked on turn one, and each relic carries its own turn guard. This was two lists,
    /// `ExtraOpeningHandDraw` for turn one and `ExtraHandDraw` for the rest, and a relic that pays
    /// on any turn had to be written into both: Pollinous Core was written into one, so it never
    /// paid on a fight's first turn and, since its count was also ticked only from turn two, the
    /// copy's count ran one turn behind at every fight (259 of the 342 Act 2 and 3 hand records
    /// of 2026-11-04). The guards, as the game's relics write them:
    /// <list type="bullet">
    /// <item>Snecko Eye +2, Pael's Blood +1, Fiddle +2: every turn.</item>
    /// <item>Ring of the Drake +2 while the turn number is 3 or less.</item>
    /// <item>Pocketwatch +3 from turn two, after a turn of three cards or fewer (its counter
    /// holds that verdict, set at the turn start).</item>
    /// <item>Pollinous Core +2 on whichever turn its count stands at four.</item>
    /// <item>Ring of the Snake +2, Bag of Preparation +2, Booming Conch +2 (an elite ROOM) and
    /// Big Mushroom -2: turn one only. Booming Conch asks the room, so a Hive or Glory elite
    /// pays like an Overgrowth one; its energy is `ApplyFirstTurnAfterSideTurnStart`.</item>
    /// </list>
    /// The opening hand's size feeds `ApplyTurnOneDrawPileReorder`, which is what decides how many
    /// Innate cards the hand is guaranteed to hold.
    /// </remarks>
    public static int HandDrawBonus(CombatState state, int turnNumber) =>
        (turnNumber > 1 && state.Relics.Any(relic => relic.DefId == Pocketwatch && relic.Counter > 0) ? 3 : 0)
        // `SneckoEye.ModifyHandDraw`: CardsVar(2), every turn and unconditionally. The two
        // cards are what pays for the Confused; Fake Snecko Eye gives the Confused and no
        // cards at all.
        + (HasRelic(state, SneckoEye) ? 2 : 0)
        // `PaelsBlood.ModifyHandDraw`: CardsVar(1), unconditionally.
        + (HasRelic(state, PaelsBlood) ? 1 : 0)
        // `Fiddle.ModifyHandDrawLate`: CardsVar(2), unconditionally. Its PRICE is `ShouldDraw`,
        // which refuses every draw that is not a hand draw while it is your turn -- see
        // `BlocksExtraDraws`. A bigger hand, and nothing else may add to it.
        + (HasRelic(state, Fiddle) ? 2 : 0)
        // `RingOfTheDrake.ModifyHandDraw`: CardsVar(2) while `TurnNumber <= Turns`, where
        // Turns is 3. The comparison is on the turn number the draw is FOR.
        + (HasRelic(state, RingOfTheDrake) && turnNumber <= RingOfTheDrakeTurns ? 2 : 0)
        // `PollinousCore.ModifyHandDraw`: CardsVar(2) on the turn its counter reaches
        // `Turns` (4); `AfterModifyingHandDraw` resets it to zero.
        + (PollinousCorePays(state) ? 2 : 0)
        // `RingOfTheSnake.ModifyHandDraw` and `BagOfPreparation.ModifyHandDraw`: CardsVar(2)
        // each, returned unchanged when `TurnNumber > 1`.
        + (turnNumber <= 1 && HasRelic(state, RingOfTheSnake) ? 2 : 0)
        + (turnNumber <= 1 && HasRelic(state, BagOfPreparation) ? 2 : 0)
        // `BoomingConch.ModifyHandDraw`: CardsVar(2) on turn one, in an Elite ROOM.
        + (turnNumber <= 1 && HasRelic(state, BoomingConch) && state.IsEliteRoom ? 2 : 0)
        // Big Mushroom's `ModifyHandDraw` SUBTRACTS its `CardsVar(2)` on turn one: the
        // opening hand is three, and that is the price of its twenty max HP.
        - (turnNumber <= 1 && HasRelic(state, BigMushroom) ? 2 : 0);

    /// <summary>Whether Pollinous Core's count stands at its fourth turn, so this draw pays.</summary>
    private static bool PollinousCorePays(CombatState state) =>
        state.Relics.Any(relic => relic.DefId == PollinousCore && relic.Counter == PollinousCoreTurns);

    /// <summary>
    /// `Hook.AfterModifyingHandDraw` for the relics whose `ModifyHandDraw` changed the count:
    /// Pollinous Core's `TurnsSeen` goes back to zero the moment it has paid, so the count the
    /// next turn adds one to is the game's, not one that waits at four for the next tick. Called
    /// with the count settled and before the cards are drawn, on every turn.
    /// </summary>
    internal static void AfterModifyingHandDraw(CombatState state)
    {
        int index = state.Relics.FindIndex(relic =>
            relic.DefId == PollinousCore && relic.Counter == PollinousCoreTurns
        );
        if (index >= 0)
        {
            state.Relics[index] = state.Relics[index] with { Counter = 0 };
        }
    }

    /// <summary>
    /// `Fiddle.ShouldDraw` returns FALSE for its owner on its OWN side turn whenever the
    /// draw is not the hand draw -- so nothing a card or a power does can add to the hand
    /// while it is your turn. Draws on the ENEMY's turn still land, which is the exception
    /// the game writes out (`player.Creature.Side != CombatState.CurrentSide`).
    ///
    /// That refusal is the whole price of its two extra cards, and modelling only the two
    /// would make Fiddle a strictly better Bag of Preparation.
    /// </summary>
    internal static bool BlocksExtraDraws(CombatState state) =>
        HasRelic(state, Fiddle) && state.PlayerTurn;

    /// <summary>
    /// `InfusedCore.ModifyOrbValue`: +1 to the player's LIGHTNING orbs, passive and evoke
    /// alike. Its other half -- three Lightning on turn one -- is in `ApplyCombatStart`.
    /// </summary>
    internal static int LightningOrbBonus(CombatState state) =>
        HasRelic(state, InfusedCore) ? 1 : 0;

    /// <summary>
    /// `Hook.BeforeHandDraw` for the relics, in the order the player holds them, from the relic
    /// at <paramref name="from"/> on. Returns the index of the relic after one whose screen is
    /// now up -- the turn start waits there, as the game awaits the screen -- or -1 when every
    /// relic has answered.
    /// </summary>
    /// <remarks>
    /// `CombatState.IterateHookListeners` walks each creature's powers and then its relics in
    /// list order, and `Hook.BeforeHandDraw` awaits each listener before the next. So two of
    /// these that put cards in the hand or the pile do it in the player's relic order: Ninja
    /// Scroll's Shivs before or after Toolbox's pick, Blessed Antler's Dazed before or after
    /// Toasty Mittens takes the top card. Every turn runs this -- turn one from combat setup,
    /// the rest from `EndTurn` -- and every relic below checks the turn itself, as the game's do.
    /// </remarks>
    internal static int ApplyBeforeHandDraw(CombatState state, int turnNumber, Random rng, int from = 0)
    {
        for (int i = from; i < state.Relics.Count; i++)
        {
            switch (state.Relics[i].DefId)
            {
                // `BlessedAntler`, TurnNumber == 1: three Dazed into the draw pile at
                // CardPilePosition.Random, which CardPileCmd resolves off Rng.Shuffle.
                case BlessedAntler when turnNumber == 1:
                    CardEffects.AddCardToDrawPileRandomly(state, ST.Dazed, 3, state.ShuffleRng ?? rng);
                    break;

                // `FuneraryMask`, TurnNumber == 1: CardsVar(3) Souls into the DRAW pile at
                // CardPilePosition.Random, one insert point per card off Rng.Shuffle -- Blessed
                // Antler's shape with a card the player wants.
                case FuneraryMask when turnNumber == 1:
                    CardEffects.AddCardToDrawPileRandomly(state, 446, 3, state.ShuffleRng ?? rng);
                    break;

                // `JeweledMask`, TurnNumber <= 1: one POWER from the draw pile, chosen off
                // `Rng.CombatCardSelection`, into HAND and free for the turn. It MOVES a card
                // rather than making one, so the draw pile is one shorter for the reorder. The
                // guard is `list.Count != 0` around a `NextItem`, so a draw pile holding exactly
                // one power still spends a roll on picking it.
                case JeweledMask when turnNumber <= 1:
                {
                    var powers = Enumerable
                        .Range(0, state.DrawPile.Count)
                        .Where(k => GeneratedData.Cards.Get(state.DrawPile[k].DefId).Type == CardType.Power)
                        .ToList();
                    if (powers.Count > 0)
                    {
                        var selection = CardEffects.CardSelectionRngFor(state, rng);
                        int chosen = powers[selection.Next(powers.Count)];
                        var card = state.RemoveFromDrawPileAt(chosen);
                        state.Hand.Add(card with { FreeThisTurn = true });
                    }

                    break;
                }

                // `NinjaScroll`, TurnNumber <= 1: `DynamicVar("Shivs", 3m)` into HAND.
                case NinjaScroll when turnNumber <= 1:
                    CardEffects.AddGeneratedCardsToHand(state, 430, 3);
                    break;

                // `RadiantPearl`, TurnNumber == 1: CardsVar(1) Luminesce into hand.
                case RadiantPearl when turnNumber == 1:
                    CardEffects.AddGeneratedCardsToHand(state, 289, 1);
                    break;

                // `ToastyMittens`, every turn: EXHAUST the top card of the draw pile, then
                // Strength 1. On turn one it takes the first NON-Innate card and falls back to
                // the first if every card is Innate. The Strength is applied whether or not a
                // card was found, because the game's `PowerCmd.Apply` sits outside the null
                // check. `ShuffleIfNecessary` comes first: with the draw pile spent the mittens
                // exhaust the top of the RESHUFFLED discard.
                case ToastyMittens:
                {
                    CardEffects.ShuffleIfNecessary(state, rng);
                    if (state.DrawPile.Count > 0)
                    {
                        int index = turnNumber == 1 ? state.DrawPile.FindIndex(card => !card.IsInnate()) : 0;
                        if (index < 0)
                        {
                            index = 0;
                        }

                        var doomed = state.RemoveFromDrawPileAt(index);
                        CardEffects.ExhaustCard(state, doomed, rng: rng);
                    }

                    GainPlayerStrength(state, 1);
                    break;
                }

                // `Toolbox`, TurnNumber == 1: `CardSelectCmd.FromChooseACardScreen` over three
                // distinct colourless cards, and the pick goes to HAND -- before the hand draw,
                // which waits for it.
                case Toolbox when turnNumber == 1:
                    CardEffects.OpenToolboxSelection(state, rng);
                    if (state.PendingSelection is not null)
                    {
                        return i + 1;
                    }

                    break;
            }
        }

        return -1;
    }

    /// <summary>
    /// `ThrowingAxe.ModifyCardPlayCount`: `playCount + 1` for the FIRST card played in a
    /// combat, and `AfterModifyingCardPlayCount` spends it. Once per fight, any card.
    ///
    /// Reads AND consumes, because the game's two hooks fire together: the count is
    /// settled when the play is SET UP, so the axe is spent at setup too. That is the same
    /// reason Burst does not double itself -- see the comment at the call site.
    /// </summary>
    internal static bool ConsumeThrowingAxePlay(CombatState state)
    {
        if (!HasRelic(state, ThrowingAxe) || GetCounter(state, ThrowingAxe) != 0)
        {
            return false;
        }

        SetCounter(state, ThrowingAxe, 1);
        return true;
    }

    /// <summary>
    /// `PollinousCore.BeforeSideTurnStart`: the saved turn count goes up by one at every player-side
    /// turn start, turn one of a fight included. Turn one's is in `ApplyBeforeOpeningHand`, later
    /// turns' in `CombatEngine.EndTurn`. The reset to zero after it has paid is
    /// <see cref="AfterModifyingHandDraw" />, the game's own place for it.
    /// </summary>
    internal static void TickPollinousCore(CombatState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == PollinousCore);
        if (index < 0)
        {
            return;
        }

        int seen = state.Relics[index].Counter;
        // A count left standing at four (a state from before the reset moved to the draw) is the
        // payment already made, so the next turn is the first of the new cycle.
        state.Relics[index] = state.Relics[index] with
        {
            Counter = seen >= PollinousCoreTurns ? 1 : seen + 1,
        };
    }

    /// <summary>
    /// The relics that pay out every Nth card of a type played in one turn. The game holds
    /// this on the relic (Kunai's AttacksPlayedThisTurn and friends), so the counter lives
    /// on the relic instance here too rather than on a shared per-turn tally.
    /// </summary>
    private static readonly int[] PerTurnCounters =
    [
        Kunai,
        Kusarigama,
        Shuriken,
        OrnamentalFan,
        LetterOpener,
    ];

    public static void ApplyAfterCardPlayed(
        CombatState state,
        CardDef def,
        Random? rng = null,
        int energySpent = 0
    )
    {
        // Ivory Tile reads CardPlay.Resources.EnergyValue — what was actually spent, so an
        // X-cost card or a cost reduction changes the answer — and ignores card type.
        if (HasRelic(state, IvoryTile) && energySpent >= 3)
        {
            CardEffects.GainEnergy(state, 1);
        }

        // `DaughterOfTheWind.AfterCardPlayed`: BlockVar(1m, Unpowered) per ATTACK played.
        if (def.Type == CardType.Attack && HasRelic(state, DaughterOfTheWind))
        {
            CardEffects.GainUnpoweredBlock(state, 1, rng);
        }

        // `IronClub.AfterCardPlayed`: every FOURTH card played draws one. The count is a
        // `[SavedProperty]` with no reset anywhere -- not in `AfterCombatEnd`, not at
        // combat start -- so it runs the length of the RUN and a fight can inherit a club
        // three cards from paying out.
        if (HasRelic(state, IronClub))
        {
            int played = GetCounter(state, IronClub) + 1;
            SetCounter(state, IronClub, played % IronClubCards);
            if (played % IronClubCards == 0)
            {
                CardEffects.DrawCards(state, 1, DrawRng(state, rng));
            }
        }

        // `LostWisp.AfterCardPlayed`: 8 unpowered damage to every hittable enemy whenever
        // a POWER is played. The wisp is a Power-deck relic and does nothing else.
        if (def.Type == CardType.Power && HasRelic(state, LostWisp))
        {
            CardEffects.DealUnpoweredDamageToAll(state, 8);
        }

        // `HelicalDart.AfterCardPlayed`: a card tagged Shiv applies
        // `HelicalDartPower(Dexterity 1)`. The TAG, not the Shiv id -- Knife Trap carries
        // it too, so it pays on the trap as well as on every Shiv the trap replays.
        if (HasRelic(state, HelicalDart) && def.ShivTag)
        {
            BuffSystem.Apply(state.PlayerBuffs, BuffId.Dexterity, 1);
        }

        switch (def.Type)
        {
            case CardType.Attack:
                if (CountTowards(state, Shuriken, period: 3))
                {
                    GainPlayerStrength(state, 1);
                }

                if (CountTowards(state, Kunai, period: 3))
                {
                    BuffSystem.Apply(state.PlayerBuffs, BuffId.Dexterity, 1);
                }

                if (CountTowards(state, OrnamentalFan, period: 3))
                {
                    // BlockVar(4m, ValueProp.Unpowered).
                    CardEffects.GainUnpoweredBlock(state, 4, rng);
                }

                if (CountTowards(state, Kusarigama, period: 3))
                {
                    // DamageVar(6m, Unpowered) at Rng.CombatTargets.NextItem(HittableEnemies).
                    var target = CardEffects.RandomLivingEnemy(state, rng);
                    if (target != null)
                    {
                        CardEffects.DealUnpoweredDamageToEnemy(state, target, 6);
                    }
                }

                // Nunchaku counts attacks for the whole combat, not the turn, so its
                // counter is deliberately absent from PerTurnCounters.
                if (CountTowards(state, Nunchaku, period: 10))
                {
                    CardEffects.GainEnergy(state, 1);
                }

                break;

            case CardType.Skill:
                if (CountTowards(state, LetterOpener, period: 3))
                {
                    // DamageVar(5m, ValueProp.Unpowered), to every hittable enemy.
                    CardEffects.DealUnpoweredDamageToAll(state, 5);
                }

                // Tuning Fork keeps its tally across turns (SkillsPlayed is a
                // SavedProperty), unlike Letter Opener two lines up.
                if (CountTowards(state, TuningFork, period: 10))
                {
                    CardEffects.GainUnpoweredBlock(state, 7, rng);
                }

                break;

            case CardType.Power:
                if (FiresOncePerCombat(state, Permafrost))
                {
                    // BlockVar(7m, ValueProp.Unpowered), on the first Power only.
                    CardEffects.GainUnpoweredBlock(state, 7, rng);
                }

                MakeRandomHandCardFree(state, rng);
                break;
        }
    }

    /// <summary>
    /// game: SelfFormingClay.AfterDamageReceived
    /// Extracted from RelicEffects.ApplyAfterUnblockedDamageReceived; the caller keeps the condition.
    /// </summary>
    internal static void SelfFormingClay_AfterDamageReceived(CombatState state)
    {
        // SelfFormingClayPower: 3 unpowered block after the next block clear.
        BuffSystem.Apply(state.PlayerBuffs, BuffId.BlockNextTurn, 3);
    }

    /// <summary>
    /// game: CentennialPuzzle.AfterDamageReceived
    /// Extracted from RelicEffects.ApplyAfterUnblockedDamageReceived; the caller keeps the condition.
    /// </summary>
    internal static void CentennialPuzzle_AfterDamageReceived(CombatState state, Random? rng)
    {
        CardEffects.DrawCards(state, 3, DrawRng(state, rng));
    }

    /// <summary>
    /// game: DemonTongue.AfterDamageReceived
    /// Extracted from RelicEffects.ApplyAfterUnblockedDamageReceived; the caller keeps the condition.
    /// </summary>
    internal static void DemonTongue_AfterDamageReceived(CombatState state, int unblocked)
    {
        SetCounter(state, DemonTongue, 1);
        CardEffects.HealPlayer(state, unblocked);
    }

    /// <summary>
    /// game: LavaLamp.AfterDamageReceived: TookDamageThisCombat set (the combat's latch, which the run copies at the
    /// combat's end). The caller keeps the condition (the owner hit, unblocked damage, not Unblockable); the old damage
    /// sites set the field inline, and Core2's handler calls this.
    /// </summary>
    internal static void LavaLamp_AfterDamageReceived(CombatState state)
    {
        state.TookUnblockedDamage = true;
    }

    /// <summary>
    /// Fires when the player takes unblocked damage. The game hangs both of these off
    /// Hook.AfterDamageReceived, which runs per damage instance; here the caller is the
    /// enemy attack path, so self-damage from a card does not trigger them.
    /// </summary>
    /// <remarks>
    /// `CreatureCmd.Damage` sends a result whose target it killed to `Kill` instead of
    /// `Hook.AfterDamageReceived` (`WasTargetKilled && IsDead`, read before the death is
    /// prevented), so a hit that empties the player's HP arms none of these even when Lizard
    /// Tail or a Fairy in a Bottle then refuses the death; the combat history entry below is
    /// written earlier in the same command and stays. `CardEffects.AfterPlayerDamageReceived`
    /// makes the same cut for the self-damage route.
    /// </remarks>
    /// <param name="unblocked">How much of the hit actually landed, for Demon Tongue.</param>
    public static void ApplyAfterUnblockedDamageReceived(
        CombatState state,
        Random? rng = null,
        int unblocked = 0
    )
    {
        bool killed = state.PlayerHp <= 0;
        if (!killed && HasRelic(state, SelfFormingClay))
        {
            SelfFormingClay_AfterDamageReceived(state);
        }

        if (!killed && FiresOncePerCombat(state, CentennialPuzzle))
        {
            CentennialPuzzle_AfterDamageReceived(state, rng);
        }

        // The combat history's `DamageReceivedEntry`, as Emotion Chip reads it: ANY unblocked
        // hit, whichever side dealt it, stamped with the player turn it landed in.
        RecordPlayerHit(state);

        // `DemonTongue.AfterDamageReceived`: the FIRST unblocked hit the player takes on
        // their own turn is healed straight back, once per turn.
        // `CombatState.CurrentSide == Owner.Creature.Side` -- the player's OWN side turn,
        // which `state.PlayerTurn` is. So it heals self-inflicted damage (Blood Wall,
        // Offering, Hemokinesis, a card that hits its owner) and never an enemy attack.
        if (
            !killed
            && state.PlayerTurn
            && unblocked > 0
            && HasRelic(state, DemonTongue)
            && GetCounter(state, DemonTongue) == 0
        )
        {
            DemonTongue_AfterDamageReceived(state, unblocked);
        }
    }

    /// <summary>
    /// Mummified Hand: after a Power, a random card in hand costs nothing this turn. The
    /// game's preference order is cards with a printed cost that still cost something,
    /// then anything that still costs something, then anything printed-costed, then
    /// anything at all — so a hand of free cards still picks a card.
    /// </summary>
    private static void MakeRandomHandCardFree(CombatState state, Random? rng)
    {
        if (!HasRelic(state, MummifiedHand) || state.Hand.Count == 0)
        {
            return;
        }

        var all = Enumerable.Range(0, state.Hand.Count).ToList();
        var printedCosted = all.Where(i => CostsAnything(state.Hand[i])).ToList();
        var stillCosts = all.Where(i =>
                CombatEngine.EffectiveCost(state.Hand[i], state) > 0 || HasStarCost(state.Hand[i])
            )
            .ToList();

        var candidates = printedCosted.Intersect(stillCosts).ToList();
        if (candidates.Count == 0)
        {
            candidates = stillCosts;
        }

        if (candidates.Count == 0)
        {
            candidates = printedCosted;
        }

        if (candidates.Count == 0)
        {
            candidates = all;
        }

        var selectionRng = state.CardSelectionRng ?? rng;
        int handIndex = candidates[selectionRng?.Next(candidates.Count) ?? 0];
        state.Hand[handIndex] = state.Hand[handIndex] with { FreeThisTurn = true };
    }

    private static int PrintedCost(CardInstance card)
    {
        var def = GeneratedData.Cards.Get(card.DefId);
        int cost = card.CostForCombat == int.MinValue ? def.Cost : card.CostForCombat;
        return card.Upgraded ? cost + def.UpgradeCost : cost;
    }

    /// <summary>
    /// `card.BaseStarCost > 0` -- the Regent's second resource, which Mummified Hand's
    /// filters count alongside energy.
    /// </summary>
    private static bool HasStarCost(CardInstance card)
    {
        var def = GeneratedData.Cards.Get(card.DefId);
        return def.HasStarCostX || def.StarCost > 0;
    }

    /// <summary>
    /// The game's `EnergyCost.GetWithModifiers(None) > 0 || BaseStarCost > 0`: a card that
    /// is PRINTED with a price in either resource.
    /// </summary>
    /// <remarks>
    /// The star half was missing, and it is not a Regent-shaped detail so much as a
    /// Regent-shaped BLIND SPOT: most of the character's cards cost 0 energy and several
    /// stars, so Mummified Hand read a whole deck as free and fell through to its
    /// last-resort "anything at all" branch. Both filters are affected, so the pick came
    /// from the wrong pool AND the card-selection stream was drawn against the wrong size.
    /// </remarks>
    internal static bool CostsAnything(CardInstance card) =>
        PrintedCost(card) > 0 || HasStarCost(card);

    /// <summary>
    /// Advances a relic's counter and reports whether this was the Nth tick. Absent relic
    /// counts as "no" without touching the list.
    /// </summary>
    private static bool CountTowards(CombatState state, int relicId, int period)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == relicId);
        if (index < 0)
        {
            return false;
        }

        int seen = (state.Relics[index].Counter + 1) % period;
        state.Relics[index] = state.Relics[index] with { Counter = seen };
        return seen == 0;
    }

    /// <summary>
    /// A once-per-combat relic, held as a counter because that is the only per-relic state
    /// there is. Combat setup builds fresh RelicInstances, so the flag clears itself.
    /// </summary>
    internal static bool FiresOncePerCombat(CombatState state, int relicId)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == relicId);
        if (index < 0 || state.Relics[index].Counter != 0)
        {
            return false;
        }

        state.Relics[index] = state.Relics[index] with { Counter = 1 };
        return true;
    }

    /// <summary>This relic's counter, or 0 when the run does not hold it.</summary>
    /// <summary>
    /// Stamps an unblocked hit on the player with the player turn it lands in (<see
    /// cref="CombatState.PlayerHitTurn" />).
    /// </summary>
    private static void RecordPlayerHit(CombatState state)
    {
        int turn = state.Turn + 1;
        if (turn != state.PlayerHitTurn)
        {
            state.PlayerHitTurnBefore = state.PlayerHitTurn;
            state.PlayerHitTurn = turn;
        }
    }

    /// <summary>
    /// Whether the player took unblocked damage during player turn <paramref name="turn" />
    /// (the game's TurnNumber): `DamageReceivedEntry.HappenedLastPlayerTurn` when it is the
    /// current turn less one.
    /// </summary>
    internal static bool PlayerHitInTurn(CombatState state, int turn) =>
        turn > 0 && (state.PlayerHitTurn == turn || state.PlayerHitTurnBefore == turn);

    private static int GetCounter(CombatState state, int relicId)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == relicId);
        return index >= 0 ? state.Relics[index].Counter : 0;
    }

    private static void SetCounter(CombatState state, int relicId, int counter)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == relicId);
        if (index >= 0)
        {
            state.Relics[index] = state.Relics[index] with { Counter = counter };
        }
    }

    /// <summary>
    /// The only randomness a relic-driven draw can need is a reshuffle, and that always
    /// reads the run's shuffle stream. The last fallback is only reachable on a state built
    /// without one.
    /// </summary>
    private static Random DrawRng(CombatState state, Random? rng) =>
        state.ShuffleRng ?? rng ?? new Random(0);

    /// <summary>
    /// Whether Lizard Tail would still refuse the player's death: held and not spent (its
    /// counter is set once it fires, and carried across the run).
    /// </summary>
    internal static bool LizardTailUnspent(CombatState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == LizardTail);
        return index >= 0 && state.Relics[index].Counter == 0;
    }

    /// <summary>
    /// game: LizardTail.AfterPreventingDeath (the emulator's heal, in ShouldDieLate's place)
    /// Extracted from RelicEffects.ApplyAfterPlayerHpChanged; the caller keeps the condition.
    /// </summary>
    internal static void LizardTail_AfterPreventingDeath(CombatState state)
    {
        // Bare on purpose: this write is INSIDE the hook, and routing it through
        // HealPlayer would re-enter. Red Skull is re-read a few lines down anyway,
        // against the revived total.
        if (Core2.Kernel.If(Core2.Switch.F08, state) is { } core2)
        {
            // game: LizardTail.AfterPreventingDeath heals (CreatureCmd.Heal) by max(1, max HP x 50%), from 0 HP.
            Core2.CreatureCmd.Heal(core2, Core2.OldCreature.Player, Math.Max(1m, state.PlayerMaxHp * 0.5m));
            return;
        }

        state.PlayerHp = Math.Max(1, state.PlayerMaxHp / 2);
    }

    /// <summary>
    /// game: RedSkull.AfterCurrentHpChanged
    /// Extracted from RelicEffects.ApplyAfterPlayerHpChanged; the caller keeps the condition.
    /// </summary>
    internal static void RedSkull_AfterCurrentHpChanged(CombatState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == RedSkull);
        if (index < 0)
        {
            return;
        }

        bool shouldBeActive = state.PlayerHp <= state.PlayerMaxHp / 2;
        bool isActive = state.Relics[index].Counter > 0;

        if (shouldBeActive == isActive)
        {
            return;
        }

        GainPlayerStrength(state, shouldBeActive ? 3 : -3);
        state.Relics[index] = state.Relics[index] with { Counter = shouldBeActive ? 1 : 0 };
    }

    public static void ApplyAfterPlayerHpChanged(CombatState state)
    {
        // LizardTail.ShouldDieLate refuses the death once per run, then heals HealVar(50m)
        // percent of max HP. The relic is spent, not removed, so the counter records it.
        if (state.PlayerHp <= 0 && FiresOncePerCombat(state, LizardTail))
        {
            LizardTail_AfterPreventingDeath(state);
        }

        RedSkull_AfterCurrentHpChanged(state);
    }

    /// <param name="heldNoBlock">
    /// Whether the player held no block when the turn ended, read before any turn-end hook
    /// granted some (Orichalcum's `BeforeSideTurnEndVeryEarly`); by default, read here.
    /// </param>
    public static void ApplyEndOfPlayerTurn(
        CombatState state,
        Random? rng = null,
        bool? heldNoBlock = null
    )
    {
        bool noBlock = heldNoBlock ?? state.PlayerBlock == 0;
        // F08+F10 on: Lunar Pastry and Parrying Shield (AfterSideTurnEnd) run in the game's `Hook.AfterTurnEnd`, after
        // `FlushPlayerHand` (ApplyAfterEndOfPlayerTurnShared); here, before the Ethereal exhaust and the discard, a card
        // their turn-end kill draws (Gremlin Horn) was flushed.
        bool afterFlush = Core2.Kernel.If(Core2.Switch.F08, state) is not null;

        // `LunarPastry.AfterSideTurnEnd`: StarsVar(1) whenever the player's side turn
        // ends. Through `GainStars` rather than `Stars +=`, so Black Hole sees it.
        if (!afterFlush && HasRelic(state, LunarPastry))
        {
            LunarPastryAfterSideTurnEnd(state);
        }

        if (HasRelic(state, Orichalcum) && noBlock)
        {
            // BlockVar(6m, ValueProp.Unpowered) — flat, whatever the player's Dexterity.
            CardEffects.GainUnpoweredBlock(state, 6, rng);
        }

        // The fake pays 3 where Orichalcum pays 6, on the same "ended the turn with no
        // block" test, read at the same early point -- so holding both pays both.
        if (HasRelic(state, FakeOrichalcum) && noBlock)
        {
            CardEffects.GainUnpoweredBlock(state, 3, rng);
        }

        if (HasRelic(state, CloakClasp) && state.Hand.Count > 0)
        {
            // BlockVar(1m, Unpowered) per card left in hand.
            CardEffects.GainUnpoweredBlock(state, state.Hand.Count, rng);
        }

        if (HasRelic(state, ScreamingFlagon) && state.Hand.Count == 0)
        {
            CardEffects.DealUnpoweredDamageToAll(state, 20);
        }

        if (HasRelic(state, StoneCalendar) && state.Turn + 1 == StoneCalendarDamageTurn)
        {
            CardEffects.DealUnpoweredDamageToAll(state, 52);
        }

        // AfterSideTurnEnd rather than Before, so it sees the block the three above added.
        if (!afterFlush && HasRelic(state, ParryingShield))
        {
            ParryingShieldAfterSideTurnEnd(state, rng);
        }
    }

    /// <summary>game: LunarPastry.AfterSideTurnEnd, the player's side.</summary>
    private static void LunarPastryAfterSideTurnEnd(CombatState state) => CardEffects.GainStars(state, 1);

    /// <summary>game: ParryingShield.AfterSideTurnEnd, the player's side: block 10 or more hits a random enemy for 6.</summary>
    private static void ParryingShieldAfterSideTurnEnd(CombatState state, Random? rng)
    {
        if (state.PlayerBlock < 10)
        {
            return;
        }

        var target = CardEffects.RandomLivingEnemy(state, rng);
        if (target != null)
        {
            CardEffects.DealUnpoweredDamageToEnemy(state, target, 6);
        }
    }

    /// <summary>Stone Calendar's DynamicVar("DamageTurn", 7m).</summary>
    private const int StoneCalendarDamageTurn = 7;

    /// <summary>Meal Ticket's HealVar(15m), on entering a merchant room.</summary>
    public const int MealTicketHeal = 15;

    /// <summary>Regal Pillow's ModifyRestSiteHealAmount: HealVar(15m) on top.</summary>
    public const int RegalPillowRestHeal = 15;

    public static bool Has(IEnumerable<RelicInstance> relics, int relicId) =>
        relics.Any(relic => relic.DefId == relicId);

    private static bool HasRelic(CombatState state, int relicId) =>
        state.Relics.Any(relic => relic.DefId == relicId);

    /// <summary>
    /// game: MiniatureCannon.ModifyDamageAdditive
    /// Extracted from RelicEffects.CardDamageBonus; the caller keeps the condition.
    /// </summary>
    internal static int MiniatureCannon_ModifyDamageAdditive() => 3;

    /// <summary>
    /// game: StrikeDummy.ModifyDamageAdditive
    /// Extracted from RelicEffects.CardDamageBonus; the caller keeps the condition.
    /// </summary>
    internal static int StrikeDummy_ModifyDamageAdditive() => 3;

    /// <summary>
    /// game: FakeStrikeDummy.ModifyDamageAdditive
    /// Extracted from RelicEffects.CardDamageBonus; the caller keeps the condition.
    /// </summary>
    internal static int FakeStrikeDummy_ModifyDamageAdditive() => 1;

    /// <summary>
    /// `ModifyDamageAdditive` from relics that read the CARD. Both are powered-attack only
    /// and both pay a flat 3.
    /// </summary>
    internal static int CardDamageBonus(CombatState state, CardDef def, bool upgraded)
    {
        if (def.Type != CardType.Attack)
        {
            return 0;
        }

        int bonus = 0;
        // MiniatureCannon: `if (!cardSource.IsUpgraded) return 0m;`
        if (upgraded && HasRelic(state, MiniatureCannon))
        {
            bonus += MiniatureCannon_ModifyDamageAdditive();
        }

        // StrikeDummy: `cardSource.Tags.Contains(CardTag.Strike)`. This read the card's
        // NAME behind a comment saying tags were not extracted -- they are now, and the
        // name agreed with the tag on all 22 cards, so this is the same answer written so
        // it stays right. Fake Strike Dummy is the same rule at 1 instead of 3.
        if (def.StrikeTag)
        {
            if (HasRelic(state, StrikeDummy))
            {
                bonus += StrikeDummy_ModifyDamageAdditive();
            }

            if (HasRelic(state, FakeStrikeDummy))
            {
                bonus += FakeStrikeDummy_ModifyDamageAdditive();
            }
        }

        return bonus;
    }

    /// <summary>
    /// game: VitruvianMinion.ModifyDamageMultiplicative
    /// Extracted from RelicEffects.CardDamageMultiplier; the caller keeps the condition.
    /// </summary>
    internal static int VitruvianMinion_ModifyDamageMultiplicative() => 2;

    /// <summary>
    /// game: PenNib.ModifyDamageMultiplicative
    /// Extracted from RelicEffects.CardDamageMultiplier; the caller keeps the condition.
    /// </summary>
    internal static int PenNib_ModifyDamageMultiplicative() => 2;

    /// <summary>
    /// `PenNib.ModifyDamageMultiplicative`: every TENTH Attack the owner plays is doubled.
    /// The counter is incremented in `BeforeCardPlayed`, so the tenth card sees a counter
    /// that has just wrapped to zero.
    /// </summary>
    internal static int CardDamageMultiplier(CombatState state, CardDef def)
    {
        // `VitruvianMinion.ModifyDamageMultiplicative` returns a flat 2 for a card tagged
        // Minion, whatever its TYPE -- so it is read before the Attack gate rather than
        // inside it. Its block half is in `ModifyBlockMultiplicative`, the same rule at
        // the other door: the three Minion cards are Minion Strike, Minion Dive Bomb and
        // Minion Sacrifice, and the relic doubles whichever half each one pays.
        int minion =
            def.MinionTag && HasRelic(state, VitruvianMinion)
                ? VitruvianMinion_ModifyDamageMultiplicative()
                : 1;

        if (def.Type != CardType.Attack)
        {
            return minion;
        }

        int index = state.Relics.FindIndex(relic => relic.DefId == PenNib);
        return minion
            * (
                index >= 0 && state.Relics[index].Counter == 0 && state.PenNibArmed
                    ? PenNib_ModifyDamageMultiplicative()
                    : 1
            );
    }

    /// <summary>Vitruvian Minion's block half: 2x the block of a Minion-tagged card.</summary>
    public static int CardBlockMultiplier(CombatState state, CardDef def) =>
        def.MinionTag && HasRelic(state, VitruvianMinion) ? 2 : 1;

    /// <summary>
    /// `UndyingSigil.ModifyDamageMultiplicative`: a powered attack aimed at the OWNER by
    /// an attacker whose current HP is at or below its own Doom lands at half.
    /// </summary>
    /// <remarks>
    /// The relic's own doc comment says it "doesn't actually _do_ anything" and that Doom
    /// checks for it -- but that is about the SECOND thing it does, moving enemy Doom to
    /// the start of the enemy turn so they die before they attack. `ModifyDamageMultiplicative`
    /// right below that comment is a real halving, and it is the reason the relic is worth
    /// buying: an enemy that is about to die to Doom is also hitting you for half.
    /// </remarks>
    public static float IncomingDamageMultiplierFromDoom(CombatState state, EnemyState attacker)
    {
        if (!HasRelic(state, UndyingSigil))
        {
            return 1f;
        }

        int doom = BuffSystem.Get(attacker.Buffs, BuffId.Doom);
        return doom > 0 && attacker.Hp <= doom ? 0.5f : 1f;
    }

    /// <summary>
    /// `PenNib.BeforeCardPlayed` -- the count rises before the card resolves, and wraps at
    /// ten. Called from the play path rather than from AfterCardPlayed, because the card
    /// being played is the one that gets doubled.
    /// </summary>
    internal static void BeforeCardPlayedRelics(CombatState state, CardDef def)
    {
        if (def.Type != CardType.Attack)
        {
            state.PenNibArmed = false;
            return;
        }

        int index = state.Relics.FindIndex(relic => relic.DefId == PenNib);
        if (index < 0)
        {
            return;
        }

        int next = (state.Relics[index].Counter + 1) % 10;
        state.Relics[index] = state.Relics[index] with { Counter = next };
        state.PenNibArmed = next == 0;
    }

    /// <summary>
    /// `Candelabra.AfterSideTurnStart`. Mercury Hourglass, which was here too, is an
    /// `AfterPlayerTurnStart` relic and fires with the others in `ApplyAfterPlayerTurnStart`.
    /// </summary>
    internal static void ApplyStartOfPlayerTurnShared(
        CombatState state,
        int turnNumber,
        Random? rng
    )
    {
        // Candelabra is turn TWO only -- `TurnNumber == 2`, not "from turn two".
        if (turnNumber == 2 && HasRelic(state, Candelabra))
        {
            CardEffects.GainEnergy(state, 2);
        }

        _ = rng;
    }

    /// <summary>
    /// `RippleBasin.BeforeSideTurnEnd`: 4 unpowered block if the owner played NO Attack
    /// this turn. The emulator counts attacks per turn already.
    /// </summary>
    /// <summary>`RippleBasin.BeforeSideTurnEnd` — before the hand is flushed.</summary>
    internal static void ApplyBeforeEndOfPlayerTurnShared(CombatState state, Random? rng)
    {
        if (HasRelic(state, RippleBasin) && state.AttackCardsPlayedThisTurn == 0)
        {
            CardEffects.GainUnpoweredBlock(state, 4, rng);
        }

        // `DiamondDiadem.BeforeSideTurnEnd`: two cards or fewer this turn and the owner
        // gains the power. A turn that plays NOTHING qualifies -- the comparison is <=, so
        // the relic pays best when it is doing the least.
        if (
            HasRelic(state, DiamondDiadem)
            // Counted in its AfterCardPlayed, auto-plays included.
            && state.CardPlaysThisTurn <= DiamondDiademCardThreshold
        )
        {
            // `PowerStackType.Single`: it is on or off, so a second grant is not a second
            // stack. Two quiet turns in a row leave the same one power.
            if (!BuffSystem.Has(state.PlayerBuffs, BuffId.DiamondDiadem))
            {
                BuffSystem.Apply(state.PlayerBuffs, BuffId.DiamondDiadem, 1);
            }
        }

        // `PaelsTears.BeforeSideTurnEnd` records whether energy was left over; the
        // turn-start half reads it and pays two. The counter IS that flag, written every
        // turn so a spendthrift turn clears what a thrifty one set.
        if (HasRelic(state, PaelsTears))
        {
            SetCounter(state, PaelsTears, state.Energy > 0 ? 1 : 0);
        }
    }

    /// <summary>
    /// The player's relics on `Hook.AfterSideTurnStart` that run every turn: Sai, Pael's Tears,
    /// Phylactery Unbound, Crossbow, Seal of Gold, Brimstone and the two Happy Flowers.
    /// </summary>
    /// <remarks>
    /// `CombatManager.StartTurn` dispatches `Hook.AfterSideTurnStart` after `SetupPlayerTurn`
    /// (energy, hand draw, `AfterPlayerTurnStart`), and the dispatch is a gated one: a turn start
    /// whose Inferno, Rolling Boulder or Festive Popper kills the last enemy reaches it with the
    /// fight ending, and none of these runs (`CombatEngine.IsOverOrEnding`, which the caller asks).
    /// What outlives the fight: the flowers' turn counts saved on the relic (run referee
    /// CGXZ45F6HL: Happy Flower's TurnsSeen 0 in the game, 1 here, after Inferno ended a
    /// Slumbering Beetle fight at a turn start), Crossbow's roll on the run's
    /// CombatCardGeneration stream and Seal of Gold's gold. They ran before the hand draw, so
    /// Crossbow's card now comes after the hand, and its roll after the BeforeHandDraw rolls
    /// (Call of the Void, Spectrum Shift), as in the game.
    ///
    /// The relics on this hook with a turn guard (Lantern, Akabeko, Big Hat, Candelabra,
    /// Chandelier and the rest of turn one's) stay with their turn's other effects.
    /// </remarks>
    internal static void ApplyAfterSideTurnStart(CombatState state, Random? rng)
    {
        // `Candelabra` (turn two) and `Chandelier` (turn three) are `AfterSideTurnStart` relics:
        // they pay after the hand is drawn, not at the energy reset. `TurnNumber` is exact.
        int turnNumber = state.Turn + 1;
        ApplyStartOfPlayerTurnShared(state, turnNumber, rng);
        ApplyStartOfPlayerTurnRares(state, turnNumber, rng);

        // `Brimstone.AfterSideTurnStart` with NO turn guard: 2 Strength to the player and
        // 1 to every LIVING opponent, every single turn. The enemy half is the point --
        // it is a Shop relic that arms the room as fast as it arms you.
        // `Sai.AfterSideTurnStart` with no turn guard: BlockVar(7m, ValueProp.Unpowered),
        // so Dexterity does not raise it. Anchor's shape without Anchor's one-shot guard.
        if (HasRelic(state, Sai))
        {
            CardEffects.GainUnpoweredBlock(state, 7, rng);
        }

        // `PaelsTears`: EnergyVar(2) at turn start, but only if `BeforeSideTurnEnd` saw
        // energy left over LAST turn. The counter is that flag -- it is written at end of
        // turn and read here, so the relic pays for thrift one turn late.
        if (HasRelic(state, PaelsTears) && GetCounter(state, PaelsTears) > 0)
        {
            CardEffects.GainEnergy(state, 2);
        }

        // `PhylacteryUnbound.AfterSideTurnStart`: SummonVar("StartOfTurn", 2), EVERY turn
        // including the first -- unlike Bound Phylactery, whose turn-start summon skips
        // turn one because `BeforeCombatStart` already did it. This one does both, so
        // turn one is five plus two.
        if (HasRelic(state, PhylacteryUnbound))
        {
            CardEffects.SummonOsty(state, 2);
        }

        // `Crossbow.AfterSideTurnStart` with no turn guard: one ATTACK from the player's
        // own pool into hand, free for the turn, every turn. The game returns early when
        // the pool has no Attack in it, which no character's does -- but the guard is
        // transcribed rather than dropped, because Big Hat's identical guard is the
        // difference between two cards and none for an Ironclad.
        if (HasRelic(state, Crossbow))
        {
            CardEffects.AddFreeAttackCardToHand(state, 1, DrawRng(state, rng));
        }

        // `SealOfGold.AfterSideTurnStart`: energy for gold, every turn, and only while the
        // player can still pay. The guard is `Gold >= GoldVar(5)` BEFORE the spend, so a
        // player on exactly five buys one last turn of it and then stops.
        if (HasRelic(state, SealOfGold) && state.PlayerGold >= SealOfGoldPrice)
        {
            // Through `GainEnergy` rather than a bare add, so NoEnergyGain stops it --
            // the chokepoint test is what caught the raw write. The gold goes
            // either way: the game's two commands are sequential and unconditional, so a
            // turn where the energy is blocked still pays the toll.
            CardEffects.GainEnergy(state, 1);
            state.PlayerGold -= SealOfGoldPrice;
        }

        if (HasRelic(state, Brimstone))
        {
            GainPlayerStrength(state, 2);
            foreach (var enemy in state.Enemies.Where(enemy => enemy.Hp > 0))
            {
                BuffSystem.Apply(enemy.Buffs, BuffId.Strength, 1);
            }
        }

        if (CountTowards(state, HappyFlower, period: 3))
        {
            CardEffects.GainEnergy(state, 1);
        }

        // The fake flower is the same clock at five turns instead of three.
        if (CountTowards(state, FakeHappyFlower, period: FakeHappyFlowerTurns))
        {
            CardEffects.GainEnergy(state, 1);
        }
    }

    /// <summary>
    /// `JossPaper.AfterSideTurnEnd` folds the turn's ETHEREAL exhausts into the count in
    /// one go — they are deliberately not counted as they happen.
    /// </summary>
    /// <remarks>
    /// AFTER the hand flush, unlike Ripple Basin above. The distinction is load-bearing:
    /// cards drawn before the flush are thrown straight back out, so a Joss Paper that
    /// paid out early would appear to do nothing at all.
    ///
    /// `Hook.AfterTurnEnd` dispatches it, and fires nothing once the fight is ending: with
    /// <paramref name="fold" /> false (the caller's gate, asked as the dispatch starts) the
    /// Ethereal exhausts banked before a turn-end kill are dropped, as `JossPaper.AfterCombatEnd`
    /// drops them, and never reach the saved count.
    /// </remarks>
    internal static void ApplyAfterEndOfPlayerTurnShared(CombatState state, Random? rng, bool fold = true)
    {
        if (Core2.Kernel.If(Core2.Switch.F08, state) is not null)
        {
            // F08+F10 on: the relics' AfterSideTurnEnd in the obtain order (game: Hook.AfterTurnEnd), Lunar Pastry and
            // Parrying Shield with Joss Paper; `fold` is the pass's gate, asked once as it starts.
            int exhausts = state.EtherealExhaustsThisTurn;
            for (int i = 0; i < state.Relics.Count && fold; i++)
            {
                // game: the pass visits no listener of a player whose hooks a kill earlier in it deactivated (Constrict's
                // turn-end damage, in the player's powers before the relics): each relic is asked with the player alive.
                if (Core2.Kernel.If(Core2.Switch.F08, state)?.World is { PlayerHooks: Core2.PlayerHooks.Deactivated })
                {
                    break;
                }

                switch (state.Relics[i].DefId)
                {
                    case JossPaper when exhausts > 0:
                        AddJossPaperExhausts(state, i, exhausts, rng);
                        exhausts = 0;
                        break;
                    case ParryingShield:
                        ParryingShieldAfterSideTurnEnd(state, rng);
                        break;
                    case LunarPastry:
                        LunarPastryAfterSideTurnEnd(state);
                        break;
                }
            }

            state.EtherealExhaustsThisTurn = 0;
            return;
        }

        int index = state.Relics.FindIndex(relic => relic.DefId == JossPaper);
        if (index >= 0 && state.EtherealExhaustsThisTurn > 0 && fold)
        {
            AddJossPaperExhausts(state, index, state.EtherealExhaustsThisTurn, rng);
        }

        state.EtherealExhaustsThisTurn = 0;
    }

    /// <summary>
    /// `HandDrill.AfterDamageGiven`: Vulnerable 2 on an enemy whose block this hit broke.
    /// </summary>
    internal static void ApplyAfterBlockBroken(CombatState state, EnemyState target)
    {
        if (HasRelic(state, HandDrill) && target.Hp > 0)
        {
            BuffSystem.Apply(target.Buffs, BuffId.Vulnerable, 2);
        }
    }

    /// <summary>
    /// `TheBoot.ModifyHpLostAfterOstyLate`: 1..4 becomes 5. Not a bonus -- a FLOOR, so it
    /// does nothing to a hit that already lands for five or more, and nothing at all to a
    /// hit that landed for zero.
    /// </summary>
    /// <remarks>
    /// The var is named `DamageMinimum` and the relic also declares a `DamageThreshold` of
    /// 4 that its code never reads -- the comparison is `amount >= DamageMinimum`, so the
    /// threshold is display text. Transcribing the threshold instead would give the same
    /// answer here and a different one the day either number moves.
    /// </remarks>
    internal static int BootDamageFloor(CombatState state, int hpLost) =>
        HasRelic(state, TheBoot) && hpLost >= 1 && hpLost < BootDamageMinimum
            ? BootDamageMinimum
            : hpLost;

    private const int BootDamageMinimum = 5;

    /// <summary>
    /// `RazorTooth.AfterCardPlayed`: an Attack or Skill the player plays is UPGRADED, if
    /// it can be. Permanently for the combat, on the copy that was played -- so it lands
    /// in the discard pile upgraded and comes back that way.
    /// </summary>
    internal static bool UpgradesPlayedCard(CombatState state, CardDef def) =>
        HasRelic(state, RazorTooth) && def.Type is CardType.Attack or CardType.Skill;

    /// <summary>
    /// `SparklingRouge.AfterBlockCleared` on TURN THREE only: Strength 1 and Dexterity 1,
    /// once, and then never again for the whole combat.
    /// </summary>
    /// <remarks>
    /// `TurnNumber == 3` is exact -- not "from turn three". Block clears at the start of
    /// the player's turn, so this is the moment turn three begins. The hook fires even
    /// when Barricade, Blur or Sturdy Clamp kept the block: `CombatManager` calls it after
    /// `ClearBlock` either way.
    /// </remarks>
    internal static void ApplyAfterBlockCleared(CombatState state, int turnNumber)
    {
        if (turnNumber == 3 && HasRelic(state, SparklingRouge))
        {
            GainPlayerStrength(state, 1);
            BuffSystem.Apply(state.PlayerBuffs, BuffId.Dexterity, 1);
        }
    }

    /// <summary>
    /// Relics the run can hold whose effect is NOT modelled, and why. Declared rather than
    /// left implicit, the way <c>Enchantments.InertInCombat</c> is: an unmodelled relic
    /// that nothing names reads exactly like a modelled one.
    /// </summary>
    /// <remarks>
    /// <b>Lasting Candy</b> ADDS a fourth option to the card-reward screen on every second
    /// combat -- `options.Add(...)`, not a swap. The emulator's screen is three slots
    /// (<c>RunState.RewardCards</c>) and <c>RunConstants.RewardSkipAction</c> is 3, so a
    /// fourth card and the skip would be the same action. Widening it is an ACTION-SPACE
    /// change: it renumbers the reward screen for every trained policy and for the Python
    /// bridge, which is not something to slip in beside a relic. Its trigger clock is
    /// modelled and pinned by <c>LastingCandyTests</c> so the remaining work is the screen
    /// alone.
    /// </remarks>
    /// <summary>
    /// EMPTY, and the emptying is the point.
    /// </summary>
    /// <remarks>
    /// Every relic in the game is now modelled. The list is kept rather than deleted
    /// because its VALUE was never the entries -- it was the bar for adding one, and the
    /// membership guard in `ActOneObtainableRelicTests` still enforces it. A relic that
    /// cannot be modelled goes here with a written argument, and the guard turns red until
    /// the argument is in the comment.
    ///
    /// The four that were here all left, and only one of the four reasons survived contact:
    /// Fake Snecko Eye went when Confused arrived; Delicate Frond's blocker had gone STALE
    /// (E485); Whispering Earring's was UNSOUND -- it called a first-in-order rule a policy
    /// (E486); and Lasting Candy and Driftwood were held back by an action-space argument
    /// that was real but overstated (E493). A fourth card-reward option does not renumber
    /// the screen if it is APPENDED, and neither does a reroll.
    ///
    /// Which is the lesson worth keeping: a declared gap is an argument, and three of these
    /// four arguments were wrong in a way that only re-reading them against the source
    /// showed.
    /// </remarks>
    public static readonly int[] UnmodelledInRun = [];

    /// <summary>
    /// `LastingCandy.IsInTriggeringCombat`: `CombatsSeen > 0 &amp;&amp; CombatsSeen % 2 == 0`,
    /// where `CombatsSeen` counts up in `AfterCombatEnd` -- so the reward screen of the
    /// second, fourth and sixth fights is the one that gets the extra Power.
    /// </summary>
    public static bool LastingCandyOffersAPower(Run.RunState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == LastingCandy);
        return index >= 0
            && state.Relics[index].Counter > 0
            && state.Relics[index].Counter % 2 == 0;
    }

    /// <summary>Counts the fight the player has just finished.</summary>
    public static void CountCombatForLastingCandy(Run.RunState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == LastingCandy);
        if (index >= 0)
        {
            state.Relics[index] = state.Relics[index] with
            {
                Counter = state.Relics[index].Counter + 1,
            };
        }
    }

    /// <summary>
    /// Relics whose counter is RUN state and that a COMBAT can move: the three teas
    /// spend a combat each fight, so the remainder has to travel home.
    /// </summary>
    /// <remarks>
    /// A short list rather than "copy every counter back": a counter that is not a
    /// `[SavedProperty]` (Shuriken's three attacks this turn) is per-combat state, and
    /// carrying it would start the next fight part-way through it.
    /// </remarks>
    private static readonly int[] RunCounterRelics =
    [
        BoneTea,
        EmberTea,
        TeaOfDiscourtesy,
        VenerableTeaSet,
        FakeVenerableTeaSet,
        // Iron Club's count is a `[SavedProperty]` that nothing resets, so it has to
        // survive the fight the way the teas' remaining-combats do. Without this the club
        // restarts at zero every combat and pays out strictly less often than the game's.
        IronClub,
        // The same for every other tally the game saves on the relic and never resets at a
        // fight's start or end (their AfterCombatEnd only resets Status, or Joss Paper's
        // EtherealCount): Pen Nib's attacks, Joss Paper's exhausts, Nunchaku's attacks,
        // Tuning Fork's skills, the flowers', Pendulum's and Pollinous Core's turns. They
        // were per-combat here, so the next fight started at zero (run referee: Joss Paper
        // and Pendulum after identical fights).
        PenNib,
        JossPaper,
        Nunchaku,
        TuningFork,
        HappyFlower,
        FakeHappyFlower,
        Pendulum,
        PollinousCore,
    ];

    /// <summary>Writes the combat's remaining counts back onto the run's relics.</summary>
    public static void CarryRunCountersBack(CombatState combat, List<RelicInstance> runRelics)
    {
        foreach (int relicId in RunCounterRelics)
        {
            int inCombat = combat.Relics.FindIndex(relic => relic.DefId == relicId);
            int inRun = runRelics.FindIndex(relic => relic.DefId == relicId);
            if (inCombat >= 0 && inRun >= 0)
            {
                runRelics[inRun] = runRelics[inRun] with
                {
                    Counter = combat.Relics[inCombat].Counter,
                };
            }
        }
    }

    /// <summary>
    /// `WongosMysteryTicket`: five combats finished, and it has not paid yet.
    /// </summary>
    /// <remarks>
    /// `CombatsFinished` counts in `AfterCombatEnd` and `TryModifyRewards` fires on the
    /// screen that follows, so the FIFTH fight's rewards are the ones that carry the three
    /// relics. `GaveRelic` then retires it -- the ticket is a one-off, not a tap.
    /// </remarks>
    public static bool WongosTicketPaysOut(Run.RunState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == WongosMysteryTicket);
        return index >= 0 && state.Relics[index].Counter == WongosTicketCombats;
    }

    /// <summary>Counts a finished combat towards the ticket.</summary>
    public static void CountCombatForWongosTicket(Run.RunState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == WongosMysteryTicket);
        if (index >= 0 && state.Relics[index].Counter <= WongosTicketCombats)
        {
            state.Relics[index] = state.Relics[index] with
            {
                Counter = state.Relics[index].Counter + 1,
            };
        }
    }

    /// <summary>Past the threshold is the spent flag -- `GaveRelic`, held as a number.</summary>
    public static void RetireWongosTicket(Run.RunState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == WongosMysteryTicket);
        if (index >= 0)
        {
            state.Relics[index] = state.Relics[index] with { Counter = WongosTicketCombats + 1 };
        }
    }

    /// <summary>
    /// A tea set armed by the last rest site spends its charge and answers true.
    /// </summary>
    private static bool SpendRestSiteCharge(CombatState state, int relicId)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == relicId);
        if (index < 0 || state.Relics[index].Counter <= 0)
        {
            return false;
        }

        state.Relics[index] = state.Relics[index] with { Counter = 0 };
        return true;
    }

    /// <summary>Arms both tea sets, on entering a rest site.</summary>
    public static void ArmTeaSetsAtRestSite(Run.RunState state)
    {
        foreach (int relicId in new[] { VenerableTeaSet, FakeVenerableTeaSet })
        {
            int index = state.Relics.FindIndex(relic => relic.DefId == relicId);
            if (index >= 0)
            {
                state.Relics[index] = state.Relics[index] with { Counter = 1 };
            }
        }
    }

    /// <summary>
    /// A tea with combats left spends one and answers true. `IsUsedUp` is
    /// `CombatsLeft &lt;= 0`, so a spent tea is inert but still occupies a relic slot.
    /// </summary>
    private static bool SpendTeaCombat(CombatState state, int relicId)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == relicId);
        if (index < 0 || state.Relics[index].Counter <= 0)
        {
            return false;
        }

        state.Relics[index] = state.Relics[index] with
        {
            Counter = state.Relics[index].Counter - 1,
        };
        return true;
    }

    /// <summary>
    /// `PaperKrane.ModifyWeakMultiplier`: -0.15 when the relic's owner is the TARGET of a
    /// powered attack, so a Weak enemy hits them for 0.60 rather than 0.75.
    /// </summary>
    /// <remarks>
    /// It reads the target, not the attacker -- Paper Krane is a defensive relic that
    /// deepens Weak on things hitting YOU, and does nothing to the Weak you apply.
    /// </remarks>
    public static float WeakMultiplierDeltaAgainstPlayer(CombatState state) =>
        HasRelic(state, PaperKrane) ? -0.15f : 0f;

    /// <summary>
    /// `PaperPhrog.ModifyVulnerableMultiplier`: +0.25 when the target is NOT its owner, so
    /// a Vulnerable enemy takes 1.75 rather than 1.5.
    /// </summary>
    /// <remarks>
    /// The mirror of Paper Krane, and note the asymmetry in the source: the Krane checks
    /// `target != Owner -> unchanged` and the Phrog checks `target == Owner -> unchanged`.
    /// One helps only when you are hit, the other only when you are not.
    /// </remarks>
    public static float VulnerableMultiplierDeltaAgainstEnemies(CombatState state) =>
        HasRelic(state, PaperPhrog) ? 0.25f : 0f;

    /// <summary>
    /// `PowerCmd.Apply` of Strength to the PLAYER, which is the one door Ruined Helmet
    /// stands at. Every grant of player Strength goes through here so the doubling is
    /// answered once rather than at twenty-eight call sites.
    /// </summary>
    /// <remarks>
    /// A LOSS passes through untouched: `TryModifyPowerAmountReceived` refuses
    /// `amount <= 0`, so Shockwave's -2 and the end-of-turn unwind of temporary Strength
    /// are not "doubled" into a bigger loss.
    /// </remarks>
    public static void GainPlayerStrength(CombatState state, int amount)
    {
        BuffSystem.Apply(
            state.PlayerBuffs,
            BuffId.Strength,
            DoubleFirstStrengthReceived(state, amount)
        );
    }

    /// <summary>
    /// A `TemporaryStrengthPower` subclass applied to the player at <paramref name="amount"/>
    /// (Flex Potion, Reptile Trinket, Setup Strike, Feeding Frenzy, Coordinate): Strength
    /// now, and `amount` taken back when the player's side turn ends.
    /// </summary>
    /// <remarks>
    /// `PowerCmd.Apply` hands `ModifyPowerAmountReceived` the power being applied, which is
    /// the temporary power and not a `StrengthPower`, so Ruined Helmet passes it by and its
    /// Amount stays at `amount`. The power's `BeforeApplied` then applies `amount` of
    /// `StrengthPower`, and that application is the one the helmet doubles.
    /// `AfterSideTurnEnd` applies -Amount. So a grant that is the combat's first Strength
    /// under an unspent helmet gives twice `amount` and leaves `amount` behind. That
    /// leftover is the game's; taking back the doubled figure would diverge from it.
    /// </remarks>
    internal static void GainTemporaryPlayerStrength(CombatState state, int amount)
    {
        GainPlayerStrength(state, amount);
        BuffSystem.Apply(state.PlayerBuffs, BuffId.TemporaryStrength, amount);
    }

    /// <summary>
    /// `Bookmark.AfterFlush`: ONE of the cards that survived the flush, chosen at random
    /// off `Rng.CombatCardSelection`, costs one less until it is played.
    /// </summary>
    /// <remarks>
    /// The candidate list is the RETAINED cards filtered to `!CostsX && cost > 0` -- a
    /// card already at zero is not a candidate and neither is an X-cost one, so a hand of
    /// free Shivs gets nothing. `AddUntilPlayed(-1)` rides on the copy and survives the
    /// turn, which is why this writes `CostBump` rather than `CostForCombat`.
    ///
    /// The flush is not a discard (see `ApplyAfterCardDiscarded`), so this fires on a
    /// boundary where Tough Bandages and Tingsha deliberately do not.
    /// </remarks>
    internal static void ApplyAfterFlush(CombatState state, Random? rng)
    {
        if (!HasRelic(state, Bookmark))
        {
            return;
        }

        var candidates = Enumerable
            .Range(0, state.Hand.Count)
            .Where(i =>
            {
                var def = GeneratedData.Cards.Get(state.Hand[i].DefId);
                return !def.HasEnergyCostX && CombatEngine.EffectiveCost(state.Hand[i], state) > 0;
            })
            .ToList();
        if (candidates.Count == 0)
        {
            return;
        }

        var selection = CardEffects.CardSelectionRngFor(state, rng);
        int chosen = candidates[selection.Next(candidates.Count)];
        state.Hand[chosen] = state.Hand[chosen] with { CostBump = state.Hand[chosen].CostBump - 1 };
    }

    /// <summary>
    /// `GoldPlatedCables.ModifyOrbPassiveTriggerCounts`: `triggerCount + 1` for the orb at
    /// `OrbQueue.Orbs[0]` and no others.
    /// </summary>
    public static int ExtraFrontOrbPassiveTriggers(CombatState state) =>
        HasRelic(state, GoldPlatedCables) ? 1 : 0;

    /// <summary>
    /// `Metronome.AfterOrbChanneled`: the SEVENTH orb channelled in a combat deals
    /// DamageVar(30m, Unpowered) to every hittable enemy. Exactly the seventh -- the test
    /// is `OrbsChanneled == OrbCount`, not `>=`, so an eighth does nothing and the
    /// counter is reset only by entering a new combat room.
    /// </summary>
    internal static void ApplyAfterOrbChanneled(CombatState state)
    {
        if (!HasRelic(state, Metronome))
        {
            return;
        }

        int channelled = GetCounter(state, Metronome) + 1;
        SetCounter(state, Metronome, channelled);
        if (channelled == MetronomeOrbCount)
        {
            CardEffects.DealUnpoweredDamageToAll(state, MetronomeDamage);
        }
    }

    /// <summary>Metronome's `DynamicVar("OrbCount", 7m)` and `DamageVar(30m, Unpowered)`.</summary>
    private const int MetronomeOrbCount = 7;

    private const int MetronomeDamage = 30;

    /// <summary>
    /// `Regalite.AfterCardGeneratedForCombat`: BlockVar(2m, Unpowered) per card the player
    /// generated. Per CARD -- a generator that makes three pays six.
    /// </summary>
    internal static void ApplyAfterCardGenerated(CombatState state, int count, Random? rng)
    {
        if (count > 0 && HasRelic(state, Regalite))
        {
            CardEffects.GainUnpoweredBlock(state, 2 * count, rng);
        }
    }

    /// <summary>
    /// `BoneFlute.AfterAttack`: BlockVar(2m, Unpowered) whenever the OWNER's Osty attacks.
    /// The guard is on the attacker being an Osty whose `PetOwner` is this player, so it
    /// is the pet's swing rather than the card that told it to swing -- one block per
    /// attack, however many the card ordered.
    /// </summary>
    internal static void ApplyAfterOstyAttack(CombatState state, Random? rng)
    {
        if (HasRelic(state, BoneFlute))
        {
            CardEffects.GainUnpoweredBlock(state, 2, rng);
        }
    }

    /// <summary>
    /// `BookRepairKnife.AfterDiedToDoom`: HealVar(3m) for EACH creature that died to Doom,
    /// counting only those whose death triggers fatal effects -- `Powers.All(
    /// ShouldOwnerDeathTriggerFatal)`, which a Minion and an un-detached Decimillipede
    /// segment fail. Nothing at all when the count is zero.
    /// </summary>
    internal static void ApplyAfterDiedToDoom(CombatState state, int fatalDeaths)
    {
        if (fatalDeaths > 0 && HasRelic(state, BookRepairKnife))
        {
            CardEffects.HealPlayer(state, 3 * fatalDeaths);
        }
    }

    /// <summary>
    /// `SneckoSkull.ModifyPowerAmountGivenAdditive`: one more Poison on every Poison the
    /// OWNER applies. Additive on the amount GIVEN, so it lands once per application
    /// rather than once per stack.
    /// </summary>
    public static int ExtraPoisonGiven(CombatState state) => HasRelic(state, SneckoSkull) ? 1 : 0;

    /// <summary>
    /// `RuinedHelmet.TryModifyPowerAmountReceived`: the FIRST positive Strength the player
    /// receives each combat is doubled, then the relic is spent until the combat ends.
    /// </summary>
    /// <remarks>
    /// It is `amount *= 2` on the amount RECEIVED, so it doubles whatever landed -- a
    /// Brimstone turn one gives 4, and everything after it gives the printed number.
    /// </remarks>
    internal static int DoubleFirstStrengthReceived(CombatState state, int amount)
    {
        if (amount <= 0 || !FiresOncePerCombat(state, RuinedHelmet))
        {
            return amount;
        }

        return amount * 2;
    }

    /// <summary>
    /// `Hook.AfterStarsSpent`, which only paying a card's star COST dispatches.
    /// </summary>
    /// <remarks>
    /// Mini-Regent fires once a TURN; Galactic Dust counts stars across the whole RUN
    /// (`[SavedProperty] StarsSpent`) and pays block every tenth. Two relics, two
    /// different clocks, one hook -- and the dust's counter surviving the combat is the
    /// part a per-combat model would get wrong.
    /// </remarks>
    internal static void ApplyAfterStarsSpent(CombatState state, int amount, Random? rng)
    {
        // `MiniRegent`: PowerVar<StrengthPower>(1m), the first spend each turn.
        if (HasRelic(state, MiniRegent) && GetCounter(state, MiniRegent) == 0)
        {
            SetCounter(state, MiniRegent, 1);
            GainPlayerStrength(state, 1);
        }

        // `GalacticDust`: StarsVar(10) and BlockVar(10m, Unpowered). It adds the spend to
        // its counter and, once the counter reaches ten, pays `floor(StarsSpent / 10) * 10`
        // block and takes the counter modulo ten -- so a single spend of twenty-five pays
        // twenty block at once and carries five over, rather than paying ten and losing
        // the rest.
        if (HasRelic(state, GalacticDust))
        {
            int spent = GetCounter(state, GalacticDust) + amount;
            if (spent >= GalacticDustStarsPerBlock)
            {
                CardEffects.GainUnpoweredBlock(
                    state,
                    spent / GalacticDustStarsPerBlock * GalacticDustBlock,
                    rng
                );
                spent %= GalacticDustStarsPerBlock;
            }

            SetCounter(state, GalacticDust, spent);
        }
    }

    /// <summary>Galactic Dust's `StarsVar(10)` and `BlockVar(10m, Unpowered)`.</summary>
    private const int GalacticDustStarsPerBlock = 10;

    private const int GalacticDustBlock = 10;

    /// <summary>
    /// `Hook.AfterCardDiscarded`, once per card an effect discards.
    /// </summary>
    /// <remarks>
    /// Both relics guard on `Owner.Creature.Side == CombatState.CurrentSide` -- the
    /// discard has to happen during the player's own turn. Every effect-driven discard in
    /// the emulator does, and the END-OF-TURN hand dump is not a discard at all:
    /// `FlushPlayerHand` is a plain `CardPileCmd.Add` followed by `Hook.AfterFlush`, with
    /// no `CardDiscarded` history row and no `AfterCardDiscarded` between them. **A hand
    /// emptied at end of turn pays neither of these relics**, which is most of what a
    /// player would assume they do.
    /// </remarks>
    internal static void ApplyAfterCardDiscarded(CombatState state, Random? rng)
    {
        // `Hook.AfterCardDiscarded` is a gated dispatch: a discard after the kill sets off no
        // Tingsha, whose hit would roll the run's CombatTargets stream over a minion left.
        if (CombatEngine.IsOverOrEnding(state))
        {
            return;
        }

        // `ToughBandages`: BlockVar(3m, Unpowered) -- Dexterity does not raise it.
        if (HasRelic(state, ToughBandages))
        {
            CardEffects.GainUnpoweredBlock(state, 3, rng);
        }

        // `Tingsha`: DamageVar(3m, Unpowered) to `Rng.CombatTargets.NextItem(
        // HittableEnemies)` -- one random enemy per card, re-rolled each time.
        if (HasRelic(state, Tingsha))
        {
            // `state.TargetRng` is the run's CombatTargets stream and takes precedence
            // inside RandomLivingEnemy; the parameter is only the single-combat fallback.
            var target = CardEffects.RandomLivingEnemyFor(state, rng ?? new Random(0));
            if (target is not null)
            {
                CardEffects.DealUnpoweredDamage(state, target, 3);
            }
        }
    }

    /// <summary>
    /// `JossPaper.AfterCardExhausted`: every FIVE cards exhausted draws one. An exhaust    /// <summary>
    /// `JossPaper.AfterCardExhausted`: every FIVE cards exhausted draws one. An exhaust
    /// caused by Ethereal is banked for the end of the turn instead of counting now.
    /// </summary>
    internal static void ApplyAfterCardExhausted(
        CombatState state,
        bool causedByEthereal,
        Random? rng
    )
    {
        // `CharonsAshes.AfterCardExhausted`: DamageVar(3m, Unpowered) to every hittable
        // enemy, per card. No Ethereal exception -- a card exhausted by Ethereal at end of
        // turn pays it too, unlike Joss Paper's banked count below.
        if (HasRelic(state, CharonsAshes))
        {
            CardEffects.DealUnpoweredDamageToAll(state, 3);
        }

        // `ForgottenSoul.AfterCardExhausted`: 1 unpowered damage to ONE random enemy off
        // CombatTargets -- Charon's Ashes' shape at a tenth of the reach.
        if (HasRelic(state, ForgottenSoul))
        {
            var target = CardEffects.RandomLivingEnemyFor(state, rng ?? new Random(0));
            if (target is not null)
            {
                CardEffects.DealUnpoweredDamage(state, target, 1);
            }
        }

        int index = state.Relics.FindIndex(relic => relic.DefId == JossPaper);
        if (index < 0)
        {
            return;
        }

        if (causedByEthereal)
        {
            state.EtherealExhaustsThisTurn++;
            return;
        }

        AddJossPaperExhausts(state, index, 1, rng);
    }

    private static void AddJossPaperExhausts(CombatState state, int index, int count, Random? rng)
    {
        int total = state.Relics[index].Counter + count;
        int draws = total / 5;
        state.Relics[index] = state.Relics[index] with { Counter = total % 5 };
        if (draws > 0 && rng is not null)
        {
            CardEffects.DrawCards(state, draws, rng);
        }
    }

    /// <summary>
    /// `Vambrace.ModifyBlockMultiplicative`: the FIRST card each combat that gains block
    /// gains double -- every gain that card makes -- and `BlockGainedThisCombat` latches
    /// once it finishes (`FinishBlockDoublingCard`).
    /// </summary>
    internal static bool DoublesCardBlock(CombatState state) =>
        HasRelic(state, Vambrace) && !state.VambraceSpent;

    /// <summary>`PaelsLegion`'s `DynamicVar("Turns", 2)`.</summary>
    internal const int PaelsLegionCooldownTurns = 2;

    /// <summary>
    /// `PaelsLegion.ModifyBlockMultiplicative`: the pet doubles a CARD's block whenever
    /// its cooldown is clear, then sits out `Turns` of them. The relic was an id in
    /// `EnemyAI` and a Pael blessing option -- the pet existed and did nothing.
    /// </summary>
    internal static bool PaelsLegionDoublesBlock(CombatState state) =>
        HasRelic(state, PaelsLegion) && state.PaelsLegionCooldown <= 0;

    /// <summary>
    /// `PaelsLegion.AfterSideTurnStart` ticks the cooldown down at the start of each of
    /// its owner's turns.
    /// </summary>
    internal static void TickPaelsLegionCooldown(CombatState state)
    {
        if (state.PaelsLegionCooldown > 0)
        {
            state.PaelsLegionCooldown--;
        }
    }

    /// <summary>
    /// game: TungstenRod.ModifyHpLostAfterOsty
    /// Extracted from RelicEffects.ModifyHpLost; the caller keeps the condition.
    /// </summary>
    internal static int TungstenRod_ModifyHpLostAfterOsty(int hpLoss)
    {
        // TungstenRod: `Math.Max(0m, amount - 1m)`.
        return Math.Max(0, hpLoss - 1);
    }

    /// <summary>
    /// game: BeatingRemnant.ModifyHpLostAfterOsty
    /// Extracted from RelicEffects.ModifyHpLost; the caller keeps the condition.
    /// </summary>
    internal static int BeatingRemnant_ModifyHpLostAfterOsty(CombatState state, int hpLoss)
    {
        // BeatingRemnant: `Math.Min(amount, 20 - DamageReceivedThisTurn)` -- a cap on
        // the TURN's total unblocked damage, not on one hit, and the running total
        // resets at the owner's side-turn start.
        return Math.Max(0, Math.Min(hpLoss, 20 - state.UnblockedDamageThisTurn));
    }

    /// <summary>
    /// `ModifyHpLostAfterOsty` from the two relics that reduce or cap HP loss, run after
    /// block has been taken off and before the HP is (`CardEffects.LosePlayerHp`). In the
    /// order the player holds them, as `IterateHookListeners` walks them: the rod before the
    /// remnant's cap takes one less than the cap, the rod after it one off what the cap left.
    /// </summary>
    internal static int ModifyHpLost(CombatState state, int hpLoss)
    {
        foreach (var relic in state.Relics)
        {
            if (relic.DefId == TungstenRod)
            {
                hpLoss = TungstenRod_ModifyHpLostAfterOsty(hpLoss);
            }
            else if (relic.DefId == BeatingRemnant)
            {
                hpLoss = BeatingRemnant_ModifyHpLostAfterOsty(state, hpLoss);
            }
        }

        return hpLoss;
    }

    /// <summary>
    /// `IceCream.ShouldPlayerResetEnergy` returns FALSE from turn two onwards — so energy
    /// carries over, and the turn's energy is added to it (`AddMaxEnergyToCurrent`) rather
    /// than replacing it. Turn one still resets, which is what puts the starting energy on
    /// the board at all.
    /// </summary>
    internal static bool ShouldResetEnergy(CombatState state, int turnNumber) =>
        turnNumber <= 1 || !HasRelic(state, IceCream);

    /// <summary>
    /// `SturdyClamp.ShouldClearBlock` returns false for its owner, and
    /// `AfterPreventingBlockClear` then trims whatever is over ten. So it is not
    /// Barricade: block survives, but only ten of it.
    /// </summary>
    internal static bool KeepsBlockCappedAtTen(CombatState state) => HasRelic(state, SturdyClamp);

    /// <summary>
    /// `Chandelier`, turn three. Bellows and Vexing Puzzlebox, which were here too, are
    /// `AfterPlayerTurnStart` relics and fire with the others in `ApplyAfterPlayerTurnStart`.
    /// </summary>
    internal static void ApplyStartOfPlayerTurnRares(CombatState state, int turnNumber, Random? rng)
    {
        if (turnNumber == 3 && HasRelic(state, Chandelier))
        {
            CardEffects.GainEnergy(state, 3);
        }

        _ = rng;
    }

    /// <summary>
    /// `Hook.AfterPlayerTurnStart` for the relics: the last step of `CombatManager
    /// .SetupPlayerTurn`, after the hand draw, in the order the player holds them, from the
    /// relic at <paramref name="from"/> on. Returns the index of the relic after one whose
    /// screen is now up -- the turn start waits there -- or -1 when every relic has answered.
    /// </summary>
    /// <remarks>
    /// `IterateHookListeners` walks a creature's powers and then its relics in list order, and
    /// each listener is awaited before the next: a Gambling Chip held before a Vexing Puzzlebox
    /// pitches and redraws before the Puzzlebox card arrives, one held after it may pitch that
    /// card, and Bellows upgrades whatever the relics before it put in the hand. Turn one runs
    /// this from combat setup, every later turn from `EndTurn`.
    /// </remarks>
    internal static int ApplyAfterPlayerTurnStart(CombatState state, int turnNumber, Random rng, int from = 0)
    {
        for (int i = from; i < state.Relics.Count; i++)
        {
            switch (state.Relics[i].DefId)
            {
                // `Bellows`, TurnNumber <= 1: `CardCmd.Upgrade(PileType.Hand...Cards)` -- the
                // opening hand, and only the opening hand.
                case Bellows when turnNumber <= 1:
                    CardEffects.UpgradeAllCards(state.Hand);
                    break;

                // `ChoicesParadox`, TurnNumber == 1: five distinct cards off the character's own
                // pool, each granted Retain, and one of them is kept.
                case ChoicesParadox when turnNumber == 1:
                    CardEffects.OpenChoicesParadoxOffer(state, ChoicesParadoxCards, rng);
                    if (state.PendingSelection is not null)
                    {
                        return i + 1;
                    }

                    break;

                // `EmotionChip`: if the player lost HP during the PREVIOUS player turn, every orb
                // in the queue fires its passive. `LostHpInPreviousTurn` asks the history for a
                // hit stamped TurnNumber - 1, so this turn's own start -- Inferno's and Crimson
                // Mantle's burn, Royal Poison -- counts next turn, not now, and nothing ever
                // counts on turn one.
                case EmotionChip:
                    if (PlayerHitInTurn(state, turnNumber - 1))
                    {
                        CardEffects.TriggerEveryOrbPassive(state, DrawRng(state, rng));
                    }

                    break;

                // `FestivePopper`, TurnNumber == 1: DamageVar(9m, ValueProp.Unpowered).
                case FestivePopper when turnNumber == 1:
                    CardEffects.DealUnpoweredDamageToAll(state, 9);
                    break;

                // `GamblingChip`, TurnNumber <= 1: `FromHandForDiscard` with a minimum of zero,
                // then `DiscardAndDraw` for as many as were pitched. An empty hand has nothing
                // to offer.
                case GamblingChip when turnNumber <= 1:
                    if (CombatEngine.OpenGamblingChipScreen(state))
                    {
                        return i + 1;
                    }

                    break;

                // `MercuryHourglass`, every turn: DamageVar(3m, ValueProp.Unpowered) to every
                // hittable enemy.
                case MercuryHourglass:
                    CardEffects.DealUnpoweredDamageToAll(state, 3);
                    break;

                // `MrStruggles`, every turn: unpowered damage to every hittable enemy equal to
                // the TURN NUMBER.
                case MrStruggles:
                    CardEffects.DealUnpoweredDamageToAll(state, turnNumber);
                    break;

                // `Pendulum`, every turn: `TurnsSeen = (TurnsSeen + 1) % 3`, and at zero a draw
                // of CardsVar(1) -- after the hand draw, so the card is the last one in hand.
                case Pendulum:
                    if (CountTowards(state, Pendulum, period: 3))
                    {
                        CardEffects.DrawCards(state, 1, DrawRng(state, rng));
                    }

                    break;

                // `RoyalPoison`, TurnNumber <= 1: 4 damage to its OWN owner, Unblockable and
                // Unpowered.
                case RoyalPoison when turnNumber <= 1:
                    CardEffects.LoseHp(state, 4);
                    break;

                // `VexingPuzzlebox`, TurnNumber == 1: one card from the character's WHOLE pool
                // (Creative Ai filters to Power and this does not), free this turn.
                case VexingPuzzlebox when turnNumber == 1:
                    CardEffects.AddRandomPoolCardToHand(state, rng);
                    break;
            }
        }

        return -1;
    }

    /// <summary>
    /// `Hook.AfterPlayerTurnStart`'s LATE pass, which runs once every listener's ordinary pass
    /// has: `BloodVial` (HealVar(2)) and `FakeBloodVial` (HealVar(1)), both TurnNumber <= 1.
    /// So Royal Poison's bite lands before either heal, and a full-health opener keeps the heal.
    /// </summary>
    internal static void ApplyAfterPlayerTurnStartLate(CombatState state, int turnNumber)
    {
        for (int i = 0; i < state.Relics.Count; i++)
        {
            switch (state.Relics[i].DefId)
            {
                case BloodVial when turnNumber <= 1:
                    CardEffects.HealPlayer(state, 2);
                    break;

                case FakeBloodVial when turnNumber <= 1:
                    CardEffects.HealPlayer(state, 1);
                    break;
            }
        }
    }

    /// <summary>
    /// The relics whose `AfterCardPlayed` answers each PLAY of a card rather than the card:
    /// `CardModel.OnPlayWrapper` runs `Hook.AfterCardPlayed` once per play, so a card whose
    /// play count Throwing Axe, Echo Form, Burst or Duplication raised pays it again.
    /// </summary>
    /// <remarks>
    /// `GamePiece`: draw after a Power. Called right after each resolution, while the card
    /// is still in play -- as the game's hook is -- so a draw that reshuffles cannot pull
    /// the card being played back in.
    /// </remarks>
    internal static void ApplyAfterEachPlay(CombatState state, CardDef def, Random? rng)
    {
        if (def.Type == CardType.Power && HasRelic(state, GamePiece) && rng is not null)
        {
            CardEffects.DrawCards(state, 1, rng);
        }
    }

    /// <summary>
    /// `RainbowRing` (one Attack, one Skill and one Power in a turn pays Strength and
    /// Dexterity, once).
    /// </summary>
    internal static void ApplyAfterCardPlayedRares(CombatState state, CardDef def, Random? rng)
    {
        if (!HasRelic(state, RainbowRing) || state.RainbowRingPaidThisTurn)
        {
            return;
        }

        // `ActivationCountThisTurn < 1` -- once a turn, and the counts are only advanced
        // while it has not yet paid.
        switch (def.Type)
        {
            case CardType.Attack:
                state.RainbowRingAttacks++;
                break;
            case CardType.Skill:
                state.RainbowRingSkills++;
                break;
            case CardType.Power:
                state.RainbowRingPowers++;
                break;
        }

        if (
            state.RainbowRingAttacks > 0
            && state.RainbowRingSkills > 0
            && state.RainbowRingPowers > 0
        )
        {
            GainPlayerStrength(state, 1);
            BuffSystem.Apply(state.PlayerBuffs, BuffId.Dexterity, 1);
            state.RainbowRingPaidThisTurn = true;
        }
    }

    /// <summary>
    /// `IntimidatingHelmet.BeforeCardPlayed`: 4 unpowered block when the card actually
    /// COST two or more — `cardPlay.Resources.EnergyValue`, so a cost reduction or a free
    /// play stops it paying.
    /// </summary>
    internal static void ApplyBeforeCardPlayedRares(CombatState state, int energySpent, Random? rng)
    {
        if (energySpent >= 2 && HasRelic(state, IntimidatingHelmet))
        {
            CardEffects.GainUnpoweredBlock(state, 4, rng);
        }
    }

    /// <summary>
    /// `UnceasingTop.AfterHandEmptied`: draw a card whenever the hand runs out during the
    /// PLAY phase — not at turn start, and not while a screen is up.
    /// </summary>
    internal static void ApplyAfterHandEmptied(CombatState state, Random? rng)
    {
        if (state.Hand.Count == 0 && HasRelic(state, UnceasingTop) && rng is not null)
        {
            CardEffects.DrawCards(state, 1, rng);
        }
    }

    /// <summary>
    /// `PrayerWheel` and `WhiteStar` each add a whole extra `CardReward` — the Wheel after
    /// a MONSTER room, the Star after an ELITE, and the Star's three come from the BOSS
    /// pool rather than the room's own.
    /// </summary>
    internal static bool AddsExtraCardReward(Run.RunState state, int roomNodeType) =>
        (roomNodeType == Run.RunConstants.NodeNormal && Has(state.Relics, PrayerWheel))
        || (roomNodeType == Run.RunConstants.NodeElite && Has(state.Relics, WhiteStar));

    /// <summary>`TheCourier.ModifyMerchantPrice`: a flat 20% off everything the shop sells.</summary>
    internal static int ModifyMerchantPrice(Run.RunState state, int price) =>
        Has(state.Relics, TheCourier) ? (int)(price * 0.8m) : price;

    /// <summary>
    /// `Hook.AfterRoomEntered`, which the game fires from each room's `Enter()` — Combat,
    /// Merchant, Treasure, Event and RestSite, and nothing else.
    /// </summary>
    /// <remarks>
    /// <paramref name="cameFromUnknown" /> is separate from the room type on purpose.
    /// Planisphere asks about the MAP POINT (`CurrentMapPoint.PointType == Unknown`) and
    /// not about what the point resolved into, so it pays out on whichever room a "?"
    /// turned out to be — including a fight.
    /// </remarks>
    public static void ApplyAfterRoomEntered(
        Run.RunState state,
        bool isRestSite,
        bool cameFromUnknown
    )
    {
        // `EternalFeather.AfterRoomEntered(room is RestSiteRoom)`: HealVar(3) per
        // CardsVar(5) cards in the deck, integer division -- 12 cards is two, not two and
        // a bit.
        // Both tea sets arm on entering a rest site and pay on the next combat's first
        // energy reset.
        if (isRestSite)
        {
            ArmTeaSetsAtRestSite(state);
        }

        if (isRestSite && Has(state.Relics, EternalFeather))
        {
            int heal = state.Deck.Count / 5 * 3;
            // Run-side: `AfterCurrentHpChanged`'s listeners all gate on
            // `CombatManager.IsInProgress`, so a heal between combats dispatches nothing.
            state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + heal);
        }

        if (cameFromUnknown && Has(state.Relics, Planisphere))
        {
            state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + 5);
        }
    }

    /// <summary>
    /// `Bread.ModifyMaxEnergy`: +1 from turn TWO onwards. Turn one is excluded here and
    /// then charged 2 at its side-turn start, which is the card's whole shape — a bad
    /// first turn bought with a better every turn after.
    /// </summary>
    /// <remarks>
    /// `PaelsFlesh.ModifyMaxEnergy` is the same shape with a later gate: `TurnNumber < 3`
    /// returns the amount unchanged, so it pays from turn THREE. It rides this hook rather
    /// than `MaxEnergyRelics` for the reason Bread does -- that list is folded into
    /// `MaxEnergy` once at combat start, and a relic whose answer depends on the turn
    /// cannot be answered once.
    /// </remarks>
    internal static int ModifyMaxEnergy(CombatState state, int maxEnergy, int turnNumber) =>
        maxEnergy
        + (turnNumber > 1 && HasRelic(state, Bread) ? 1 : 0)
        + (turnNumber >= 3 && HasRelic(state, PaelsFlesh) ? 1 : 0);

    /// <summary>
    /// `LavaLamp.TryModifyCardRewardOptionsLate`: every upgradable option on the card
    /// reward screen is upgraded, if the combat took no UNBLOCKED, blockable damage.
    /// </summary>
    /// <remarks>
    /// Its `AfterDamageReceived` ignores `ValueProp.Unblockable` — so a Burn or a curse's
    /// self-damage does not spoil it, and only damage the player could have blocked does.
    /// </remarks>
    public static bool UpgradesCardRewards(Run.RunState state) =>
        Has(state.Relics, LavaLamp) && !state.TookUnblockedDamageThisCombat;

    /// <summary>
    /// `DingyRug.ModifyCardRewardCreationOptions`: the COLOURLESS pool is added to the
    /// pools a card reward rolls from — added, not replaced, so the character's own cards
    /// are still on offer.
    /// </summary>
    public static bool AddsColourlessToCardRewards(Run.RunState state) =>
        Has(state.Relics, DingyRug);

    /// <summary>
    /// `PrismaticGem.ModifyCardRewardCreationOptions`: a card REWARD rolls from every
    /// character's pool at once, not just the player's.
    /// </summary>
    /// <remarks>
    /// Card REWARDS only -- the hook refuses anything without `IsCardReward`, anything
    /// with `NoCardPoolModifications`, anything on a custom pool, and anything already
    /// colourless-only. So a shop, an event's own offer and a combat generator all keep
    /// their pools; this is the reward screen and nothing else.
    ///
    /// The emulator's arm added a random reward card ON PICKUP, which is a different
    /// relic entirely -- and it never ran, because the id constant behind it was 1533 and
    /// no relic has that id.
    /// </remarks>
    public static bool WidensCardRewardsToEveryPool(Run.RunState state) =>
        Has(state.Relics, PrismaticGem);

    /// <summary>
    /// `WingCharm.TryModifyCardRewardOptionsLate`: ONE option on the screen, rolled on
    /// `Rng.Niche` from those that can take it, gains the Swift enchantment.
    /// </summary>
    public static bool EnchantsACardReward(Run.RunState state) => Has(state.Relics, WingCharm);

    /// <summary>
    /// `MysticLighter.ModifyDamageAdditive`: 9 more from a powered attack whose card
    /// carries ANY enchantment — `cardSource?.Enchantment == null` is the only filter.
    /// </summary>
    internal static int EnchantedCardDamageBonus(CombatState state) =>
        HasRelic(state, MysticLighter) ? 9 : 0;

    /// <summary>`ChemicalX.ModifyXValue`: every X-cost card resolves two higher.</summary>
    internal static int ModifyXValue(CombatState state, int x) =>
        HasRelic(state, ChemicalX) ? x + 2 : x;

    /// <summary>
    /// `RingingTriangle.ShouldFlush` returns false on turn ONE, so the opening hand is
    /// retained whole rather than discarded.
    /// </summary>
    internal static bool SkipsHandFlush(CombatState state, int turnNumber) =>
        (turnNumber <= 1 && HasRelic(state, RingingTriangle))
        // `RunicPyramid.ShouldFlush` is false for its owner on EVERY turn, with no clock
        // at all -- the hand is simply never discarded. The Ringing Triangle's turn-one
        // version is the same hook with a guard.
        || HasRelic(state, RunicPyramid);

    /// <summary>
    /// `MiniatureTent.ShouldDisableRemainingRestSiteOptions` returns false — so taking one
    /// rest option leaves the others available instead of ending the visit.
    /// </summary>
    public static bool KeepsRestSiteOpen(Run.RunState state) => Has(state.Relics, MiniatureTent);

    /// <summary>
    /// The relics' `AfterShuffle`, in the order they are held (the hook visits relics in
    /// `player.Relics` order), once the shuffled pile is in place.
    /// </summary>
    internal static void ApplyAfterShuffle(CombatState state, Random? rng)
    {
        // By index: a block gain's hooks may write a relic's counter back into the list.
        for (int i = 0; i < state.Relics.Count; i++)
        {
            int relic = state.Relics[i].DefId;
            // `TheAbacus.AfterShuffle`: 6 unpowered block every time the pile is shuffled.
            if (relic == TheAbacus)
            {
                CardEffects.GainUnpoweredBlock(state, 6, rng);
            }
            // `BiiigHug.AfterShuffle`: a SOOT into the draw pile at a random position, every
            // shuffle, for the whole run. Only its pickup half -- remove four chosen cards --
            // was modelled, which made a hug that costs nothing a hug that only pays.
            else if (relic == BiiigHug)
            {
                CardEffects.AddCardToDrawPileRandomly(state, ST.Soot, 1, rng ?? new Random());
            }
        }
    }

    /// <summary>
    /// `BurningSticks.AfterCardExhausted`: the first SKILL exhausted each combat is copied
    /// back into hand. Once per combat, and Skills only.
    /// </summary>
    internal static void ApplyBurningSticks(CombatState state, CardInstance card)
    {
        if (
            state.BurningSticksUsed
            || !HasRelic(state, BurningSticks)
            || GeneratedData.Cards.Get(card.DefId).Type != CardType.Skill
        )
        {
            return;
        }

        state.BurningSticksUsed = true;
        if (state.Hand.Count < CardEffects.MaxCardsInHand)
        {
            state.Hand.Add(state.Fresh(card) with { FreeThisTurn = false });
        }
    }

    /// <summary>
    /// `BeltBuckle`: Dexterity 2 while the owner holds NO potions, applied and removed as
    /// the belt fills and empties rather than checked once.
    /// </summary>
    /// <remarks>
    /// Re-evaluated at every point the game hooks — combat start, and after a potion is
    /// procured, used or discarded — because the whole design is that it toggles. A
    /// once-at-combat-start reading would give the Dexterity to a player who then drinks
    /// their way out of it, and withhold it from one who empties their belt mid-fight.
    /// </remarks>
    internal static void RefreshBeltBuckle(CombatState state)
    {
        if (!HasRelic(state, BeltBuckle))
        {
            return;
        }

        bool shouldHold = !state.PotionSlots.Any(slot => slot != 0);
        if (shouldHold == state.BeltBuckleApplied)
        {
            return;
        }

        BuffSystem.Apply(state.PlayerBuffs, BuffId.Dexterity, shouldHold ? 2 : -2);
        state.BeltBuckleApplied = shouldHold;
    }

    /// <summary>
    /// `GhostSeed.AfterCardEnteredCombat`: every BASIC Strike or Defend gains Ethereal.
    /// </summary>
    /// <remarks>
    /// The tag stand-in is the same one `Card.IsStrikeOrDefend` uses, and here the caveat
    /// does not bite: the filter is `Rarity == Basic` AND tagged, and among Basic cards
    /// the entry slug and the tag agree for every character.
    /// </remarks>
    internal static bool MakesBasicsEthereal(CombatState state) => HasRelic(state, GhostSeed);

    /// <summary>
    /// `GamblingChip.AfterPlayerTurnStart` on turn one: a discard screen with no upper
    /// bound, and the draw comes after.
    /// </summary>
    /// <summary>
    /// `UnsettlingLamp`: the FIRST card of a combat that lands a debuff on an enemy has
    /// its debuffs doubled — all of them, for that one card.
    /// </summary>
    /// <remarks>
    /// The game latches in `BeforePowerAmountChanged` on the first qualifying application
    /// and unlatches in `AfterCardPlayed` for that same card, so a card applying three
    /// debuffs gets all three doubled and the next card gets none. Modelled with a
    /// per-combat spent flag plus a per-card latch for the same reason.
    ///
    /// The game also excludes debuffs whose source is a TEMPORARY power already doubled
    /// (`HasDoubledTemporaryPowerSource`), which stops the internally-applied Strength of
    /// a TemporaryStrengthPower being doubled twice. The emulator applies those two as one
    /// pair through `ApplyTemporaryStrengthDownTo`, so the double lands once either way.
    /// </remarks>
    internal static int ModifyEnemyDebuffMagnitude(CombatState state, BuffId id, int magnitude)
    {
        if (magnitude == 0 || state.UnsettlingLampSpent || !HasRelic(state, UnsettlingLamp))
        {
            return magnitude;
        }

        // `power.GetTypeForAmount(amount) != PowerType.Debuff` -- the SIGN decides for a
        // power that reads both ways. Strength down is a negative magnitude and is very
        // much a debuff; Weak and Vulnerable are positive ones. Reading only positives
        // doubled Malaise's Weak and left its Strength loss alone, which is half a card.
        bool isDebuff = id == BuffId.Strength ? magnitude < 0 : magnitude > 0;
        if (!isDebuff)
        {
            return magnitude;
        }

        state.UnsettlingLampCard = true;
        return magnitude * 2;
    }

    /// <summary>
    /// `AfterCardPlayed` on the latched card: the doubling is finished for the combat.
    /// </summary>
    internal static void FinishUnsettlingLampCard(CombatState state)
    {
        if (state.UnsettlingLampCard)
        {
            state.UnsettlingLampCard = false;
            state.UnsettlingLampSpent = true;
        }
    }

    /// <summary>
    /// `Vambrace.AfterCardPlayed` and `PaelsLegion.AfterCardPlayed` on the card whose block
    /// they doubled: Vambrace is spent for the combat, and the Legion's cooldown starts.
    /// Until then every gain the card makes is doubled (Second Wind's, one per exhaust).
    /// </summary>
    internal static void FinishBlockDoublingCard(CombatState state)
    {
        if (state.VambraceCard)
        {
            state.VambraceCard = false;
            state.VambraceSpent = true;
        }

        if (state.PaelsLegionCard)
        {
            state.PaelsLegionCard = false;
            state.PaelsLegionCooldown = PaelsLegionCooldownTurns;
        }
    }

    /// <summary>
    /// `Girya.AfterRoomEntered(CombatRoom)`: Strength equal to the number of times it has
    /// been LIFTED, at the start of every combat. Zero lifts is zero Strength, so a Girya
    /// nobody rested with does nothing at all.
    /// </summary>
    public static int GiryaStrength(Run.RunState state)
    {
        int index = state.Relics.FindIndex(relic => relic.DefId == Girya);
        return index < 0 ? 0 : state.Relics[index].Counter;
    }

    /// <summary>
    /// `Pantograph.BeforeCombatStart`: 25 HP, but only when the room is a BOSS. Its
    /// `AfterRoomEntered` sets a display status and nothing else, which is why the heal
    /// lives here rather than in the room hook above.
    /// </summary>
    public static void ApplyBeforeBossCombat(Run.RunState state)
    {
        if (Has(state.Relics, Pantograph))
        {
            state.PlayerHp = Math.Min(state.PlayerMaxHp, state.PlayerHp + 25);
        }
    }

    /// <summary>
    /// `AmethystAubergine.TryModifyRewards`: a GoldVar(15) reward after any COMBAT room,
    /// except the boss of the final act — there is no reward screen to add it to when the
    /// run is over.
    /// </summary>
    /// <remarks>
    /// Returns the RAW 15. The reward's gold is claimed through `RunNonCombatEffects.GainGold`
    /// like any other, and that applies `ModifyGoldGained` -- applying it here too would pay
    /// Bowler Hat twice on this one relic.
    /// </remarks>
    public static int ExtraCombatRewardGold(Run.RunState state, bool isFinalActBoss) =>
        !isFinalActBoss && Has(state.Relics, AmethystAubergine) ? 15 : 0;

    /// <summary>
    /// `JuzuBracelet.ModifyUnknownMapPointRoomTypes`: a "?" can never be a Monster room.
    /// The relic removes the type from the allowed SET before the odds are rolled, so the
    /// probability mass redistributes rather than the roll being re-taken.
    /// </summary>
    public static bool ForbidsUnknownMonsterRooms(Run.RunState state) =>
        Has(state.Relics, JuzuBracelet);

    /// <summary>
    /// `ReptileTrinket.AfterPotionUsed`: 3 Strength, and it is a TemporaryStrengthPower --
    /// handed back at the end of the turn, like Piercing Wail's loss in the other
    /// direction.
    /// </summary>
    internal static void ApplyAfterPotionUsed(CombatState state)
    {
        if (HasRelic(state, ReptileTrinket))
        {
            GainTemporaryPlayerStrength(state, 3);
        }
    }
}
