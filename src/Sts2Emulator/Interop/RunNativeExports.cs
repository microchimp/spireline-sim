using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Interop;

public static class RunNativeExports
{
    // v13: the observation carries the shop's whole board, priced.
    // v15: the map offers a whole row, not four children -- Winged Boots' free travel.
    // v16: state list 17 reports an open card-offer grid.
    // v17: state list 19 names the enemies in the active combat, and 20 says what an open
    // card-select screen is FOR.
    // v18: state list 21 reports where on the map the run is standing.
    // v19: the observation no longer carries the encounter behind a map choice, so the
    // layout is one number shorter and every offset after the map's node types moves.
    // v20: Sts2Run_Reset takes the run's ascension level, which the engine no longer
    // hardcodes at 8.
    // v21: Sts2Run_DebugResolveCombat and Sts2Run_DebugSetPlayer, the two hooks a real
    // run's run layer is replayed through (scripts/replay_real_run.py); state lists 22
    // (shop cards shown upgraded) and 23 (an offer grid's upgrades).
    // v22: Sts2Run_ForkCombat hands the run's fight to a combat handle, faithful or as the
    // fight of Sts2Run_Clone's resampled copy (rl/run_train/DESIGN.md 6.5).
    // v23: Sts2Run_AiStateAssumed (DESIGN.md 1.1's ai_state_assumed for the run's fight).
    // v24: Sts2Run_SetAnchorTables and Sts2Run_ReanchorFight put the fight a run stands in on the
    // referee's record of it (referee/lockstep.py, the re-anchor at a fight's start).
    // v25: Sts2Run_ReanchorFightHidden puts the fight on a record that carries the game's streams and the draw pile in
    // order (its flags say which), and Sts2Run_DebugSetStreams puts the run's streams on the game's counters (the
    // lockstep's re-anchor at a fight start and at a room entry; referee/lockstep.py).
    // v26: Sts2Run_ResetAs starts a run as any of the five characters; the run's action mask carries a discard action
    // per held potion (RunConstants.DiscardPotionAction) beside the use actions; Sts2Run_PotionActionReason says why a
    // use or discard action is refused (referee/lockstep.py maps the referee's potion labels onto them).
    // v27: Sts2Run_SetMaskVersion picks the run's action-mask version (default 1, today's mask; version 2 adds the potion
    // discard actions on the card reward, deck-select, bundle-select and Crystal Sphere screens; RunEngine.MaskVersion).
    // v28: a run made after Sts2_SetEngine("core2") runs the Tier 1 kernel (docs/rebuild/NATIVE_KERNEL.md): Sts2Run_Step
    // and Sts2Run_StartCombat go through it; Sts2Run_Clone, Sts2Run_ForkCombat and the re-anchors refuse such a run
    // (Sts2_GetLastError, the re-anchor's message) until their pieces land. Old runs are unchanged.
    // v29: under core2 Sts2Run_Clone carries the engine's kernel (the copy's fight steps through its gate) and the
    //      re-anchors bind the anchored fight (NATIVE_KERNEL.md pieces 2-3): neither is refused any more.
    // v30: mask version 3 (Sts2Run_SetMaskVersion accepts 1..3): also the use actions of the AnyTime potions (Blood Potion,
    //      Fruit Juice, Entropic Brew) on the card reward, deck-select, bundle-select and Crystal Sphere screens (D34).
    public const int RUN_NATIVE_API_VERSION = 30;
    private static readonly RunEngine?[] _pool = new RunEngine?[256];

    public static int Sts2Run_NativeApiVersion() => RUN_NATIVE_API_VERSION;

    public static int Sts2Run_ObsSize() => RunConstants.RunObsSize;

    public static int Sts2Run_MaxActions() => RunConstants.MaxActions;

    public static int Sts2Run_InfoSize() => RunConstants.RunInfoSize;

    /// <summary>How many numbers <see cref="Sts2Run_ObsLayout"/> writes.</summary>
    public const int RUN_OBS_LAYOUT_SIZE = 13;

    /// <summary>
    /// Where the run observation's variable-length blocks sit, so a consumer does not have
    /// to hard-code offsets that move whenever a block grows:
    ///
    /// <c>[scalars, deck offset, deck slots, ints per card, relic offset, relic slots,
    /// ints per relic, shop offset, shop slots, ints per shop slot, map choices, map node
    /// type offset, Lasting Candy's extra-option offset]</c>
    ///
    /// APPENDED, never inserted -- a consumer reading by index must keep reading the same
    /// thing across builds, which is the whole reason this export exists.
    ///
    /// Offsets are relative to the start of the run block, which itself begins at the
    /// combat observation's own size.
    /// </summary>
    /// <returns>How many numbers were written, or -1 if the buffer is too small.</returns>
    public static unsafe int Sts2Run_ObsLayout(int* buf, int len)
    {
        if (len < RUN_OBS_LAYOUT_SIZE)
        {
            return -1;
        }

        var layout = new Span<int>(buf, len);
        layout[0] = RunConstants.RunScalarObsSize;
        layout[1] = RunConstants.DeckObsOffset;
        layout[2] = RunConstants.MaxObservedDeck;
        layout[3] = RunConstants.DeckSlotSize;
        layout[4] = RunConstants.RelicObsOffset;
        layout[5] = RunConstants.MaxObservedRelics;
        layout[6] = RunConstants.RelicSlotSize;
        layout[7] = RunConstants.ShopObsOffset;
        layout[8] = RunConstants.ShopSlots;
        layout[9] = RunConstants.ShopSlotSize;
        layout[10] = RunConstants.MapChoices;
        layout[11] = RunConstants.MapNodeTypeObsOffset;
        layout[12] = RunConstants.FourthRewardObsOffset;
        return RUN_OBS_LAYOUT_SIZE;
    }

    /// <summary>
    /// The engine new handles are made with (<see cref="NativeEngine" />, docs/rebuild/NATIVE_KERNEL.md section 3): old
    /// unless <c>Sts2_SetEngine</c> said otherwise. A handle keeps the engine it was made with.
    /// </summary>
    internal static NativeEngine Engine { get; private set; } = NativeEngine.Old;

    /// <summary>
    /// <c>Sts2_SetEngine</c>: the engine of the handles made after this call ("old", "core2", "core2:F08,F10"). Returns
    /// null, or why the spec is refused (the engine is then unchanged).
    /// </summary>
    internal static string? SetEngine(string spec)
    {
        try
        {
            Engine = NativeEngine.Parse(spec);
            return null;
        }
        catch (ArgumentException e)
        {
            return e.Message;
        }
    }

    public static int Sts2Run_Create() => CreateWith(Engine);

    /// <summary>A new run handle on <paramref name="engine" /> (the test seam of <see cref="Sts2Run_Create" />).</summary>
    internal static int CreateWith(NativeEngine engine)
    {
        var run = new RunEngine();
        if (engine.Kernel is { } kernel)
        {
            run.Kernel = kernel;  // bound to the run's state now and to every state Reset makes and combat it starts
        }

        for (int i = 0; i < _pool.Length; i++)
        {
            if (_pool[i] is null)
            {
                _pool[i] = run;
                return i;
            }
        }

        return -1;
    }

    /// <summary>
    /// Fork a run into a new handle. <paramref name="resampleHidden" /> non-zero
    /// resamples everything the agent has not been shown -- future rewards, shop stock,
    /// encounter composition, and the unseen part of the draw pile -- off
    /// <paramref name="resampleSeed" />. A clone taken without it is a faithful copy,
    /// which for a tree search is an oracle. See docs/agent-interface.md.
    /// </summary>
    /// <returns>
    /// The new handle, -1 if the source is unknown or the pool is full, or -3 when the run's engine has a kernel switch on
    /// (<c>Sts2_GetLastError</c> says why): <see cref="RunEngine.Clone" /> does not carry the engine's kernel yet. -3 is
    /// <see cref="Sts2Run_ForkCombat" />'s code for the same refusal, so a caller tells it from a bad handle without
    /// reading an error text that may be an older refusal's.
    /// </returns>
    public static unsafe int Sts2Run_Clone(
        int handle,
        int resampleHidden,
        int resampleSeed,
        int* obsBuf
    )
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        var copy = run.Clone(resampleHidden != 0 ? resampleSeed : null);
        for (int i = 0; i < _pool.Length; i++)
        {
            if (_pool[i] is null)
            {
                _pool[i] = copy;
                copy.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
                return i;
            }
        }

        return -1;
    }

    /// <summary>Start a new run on this handle at the given ascension level.</summary>
    /// <returns>
    /// 0, -1 for an unknown handle, or -2 for a level the run engine does not model (see
    /// <see cref="Ascension.MaxRunLevel" />). Checked here rather than left to
    /// <see cref="RunEngine.Reset" />'s exception, which would abort the host process.
    /// </returns>
    public static unsafe int Sts2Run_Reset(
        int handle,
        byte* seedPtr,
        int seedLen,
        int ascension,
        int* obsBuf
    )
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        if (ascension is < 0 or > Ascension.MaxRunLevel)
        {
            return -2;
        }

        string seed = Encoding.UTF8.GetString(new ReadOnlySpan<byte>(seedPtr, seedLen));
        run.Reset(seed, ascension);
        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        return 0;
    }

    /// <summary>
    /// Start a new run on this handle as <paramref name="character" /> (a <see cref="PlayerCharacter" /> index: 0
    /// Ironclad, 1 Silent, 2 Regent, 3 Necrobinder, 4 Defect) at the given ascension level: that character's start
    /// deck, relic and max HP, and its pools in every roll the run makes.
    /// </summary>
    /// <returns>0, -1 for an unknown handle, -2 for an ascension level the engine does not model, -3 for a character index.</returns>
    public static unsafe int Sts2Run_ResetAs(
        int handle,
        byte* seedPtr,
        int seedLen,
        int ascension,
        int character,
        int* obsBuf
    )
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        if (ascension is < 0 or > Ascension.MaxRunLevel)
        {
            return -2;
        }

        if (!Enum.IsDefined(typeof(PlayerCharacter), character))
        {
            return -3;
        }

        string seed = Encoding.UTF8.GetString(new ReadOnlySpan<byte>(seedPtr, seedLen));
        run.Reset(seed, ascension, (PlayerCharacter)character);
        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        return 0;
    }

    /// <summary>
    /// Pick the version of the action mask this run writes (<see cref="RunEngine.MaskVersion" />): 1 is the default and the
    /// mask as it was before the versions; 2 also offers the potion discard on the card reward, deck-select,
    /// bundle-select and Crystal Sphere screens. The setting belongs to the handle: <c>Sts2Run_Reset*</c> keeps it, and
    /// <c>Sts2Run_Clone</c> copies it.
    /// </summary>
    /// <returns>0, -1 for an unknown handle, -2 for a version the mask does not have.</returns>
    public static int Sts2Run_SetMaskVersion(int handle, int version)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        if (version is < RunConstants.MaskVersion1 or > RunConstants.MaxMaskVersion)
        {
            return -2;
        }

        run.MaskVersion = version;
        return 0;
    }

    /// <summary>
    /// Why the use or discard potion <paramref name="action" /> is refused on this run right now (UTF-8, no terminator).
    /// </summary>
    /// <returns>
    /// The bytes written (0 when the action is legal), -1 for an unknown handle, -2 when the action is not a potion
    /// action, -3 when the buffer is too small.
    /// </returns>
    public static unsafe int Sts2Run_PotionActionReason(int handle, int action, byte* buf, int bufLen)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        if (RunPotionUse.SlotOf(action) < 0 && RunPotionUse.DiscardSlotOf(action) < 0)
        {
            return -2;
        }

        byte[] reason = Encoding.UTF8.GetBytes(RunPotionUse.Refusal(run.State, action, run.MaskVersion) ?? "");
        if (reason.Length > bufLen)
        {
            return -3;
        }

        reason.CopyTo(new Span<byte>(buf, bufLen));
        return reason.Length;
    }

    /// <summary>
    /// True when the run's engine has a Tier 1 kernel switch on (TIER1_KERNEL.md section 5; the engine option core2,
    /// docs/rebuild/NATIVE_KERNEL.md). The run's own steps and the combats it starts go through the kernel
    /// (<see cref="RunEngine.Step" />'s entry gate, <c>StartCombatWithDeck</c>'s bind); every export whose combat would
    /// skip the kernel (a clone, a fork, a re-anchor, a combat handle, a snapshot) refuses such a run rather than run it
    /// old, silently, until its piece lands.
    /// </summary>
    internal static bool KernelOn(RunEngine run) => run.Kernel is { AnyOn: true };

    /// <summary>
    /// With a switch on, a fight the run stands in that carries no KernelState: <see cref="RunEngine.Step" /> would throw
    /// (its kernel guard), and a throw across the native boundary aborts the host, so the export refuses first.
    /// </summary>
    private static bool UnboundFight(RunEngine run) =>
        KernelOn(run) && run.State.Phase == RunPhase.Combat && run.State.ActiveCombat is { Kernel: null };

    public static unsafe int Sts2Run_Step(
        int handle,
        int action,
        int targetEnemyIndex,
        int* obsBuf,
        float* rewardOut,
        int* terminalOut,
        int* truncatedOut
    )
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        if (UnboundFight(run))
        {
            NativeExports.Refuse("kernel_on: the fight carries no KernelState (bind it where the combat was built)");
            return -1;
        }

        int status = run.Step(
            action,
            targetEnemyIndex,
            out float reward,
            out bool terminal,
            out bool truncated
        );
        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        *rewardOut = reward;
        *terminalOut = terminal ? 1 : 0;
        *truncatedOut = truncated ? 1 : 0;
        return status;
    }

    public static unsafe int Sts2Run_StartCombat(
        int handle,
        int* deckIds,
        int deckLen,
        int encounterId,
        int* relicIds,
        int relicLen,
        int playerHp,
        int playerMaxHp,
        int* potionIds,
        int potionLen,
        int playerGold,
        int completedCombatRoomsBeforeCurrent,
        int* obsBuf
    )
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        // A switched-on kernel binds the combat StartCombatWithDeck builds, before its setup, as for any combat of the run.
        // The same rule as every combat reset: [0,3) keeps the weak encounter, anything
        // else (a negative count included, which this export used to clamp to 0, the weak
        // fight) is the normal one.
        int status = run.StartCombat(
            new ReadOnlySpan<int>(deckIds, deckLen),
            WeakContext.Resolve(encounterId, completedCombatRoomsBeforeCurrent),
            new ReadOnlySpan<int>(relicIds, relicLen),
            playerHp,
            playerMaxHp,
            new ReadOnlySpan<int>(potionIds, potionLen),
            playerGold
        );
        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        return status;
    }

    public static unsafe int Sts2Run_ActionMask(int handle, int* maskBuf, int maskLen)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        run.WriteActionMask(new Span<int>(maskBuf, maskLen));
        return 0;
    }

    public static unsafe int Sts2Run_GetInfo(int handle, int* infoBuf, int infoLen)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        run.WriteInfo(new Span<int>(infoBuf, infoLen));
        return 0;
    }

    public static unsafe int Sts2Run_GetStateList(int handle, int listId, int* outBuf, int outLen)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        if (outLen < 0)
        {
            return -2;
        }

        Span<int> output = new(outBuf, outLen);
        return listId switch
        {
            0 => WriteCardList(run.State.Deck, output),
            1 => WriteRelicList(run.State.Relics, output),
            // The ACTIVE COMBAT's belt while there is one, exactly as the observation does
            // (`RunEngine`: `activeCombat?.PotionSlots ?? State.PotionSlots`). A combat is
            // handed a COPY of the run's slots and only syncs back when it ends, so reading
            // the run's here reported the belt as it stood when the fight started -- every
            // potion drunk, auto-fired or picked up mid-combat was invisible until the
            // fight was over. Three of the four known run-trace divergences were this one
            // line. See E492.
            2 => WriteIntArray(ActivePotionSlots(run.State), output),
            3 => WriteIntArray(run.State.NeowOptions, output),
            4 => WriteIntArray(run.State.ShopCosts, output),
            5 => WriteBoolArray(run.State.RewardUpgraded, output),
            6 => WriteRewardList(run.State, output),
            7 => WriteMapOptionCoords(run.State, output),
            8 => WriteIntArray(run.State.ShopCards, output),
            9 => WriteIntArray(run.State.ShopRelics, output),
            10 => WriteIntArray(run.State.ShopPotions, output),
            // 11-14: run-generation data, for differential testing against a live
            // save's acts[].rooms (see scripts/verify_run_generation.py).
            11 => WriteIntArray(run.State.NormalEncounterSequence, output),
            12 => WriteIntArray(run.State.EliteEncounterSequence, output),
            13 => WriteIntArray(run.State.EventSequence, output),
            // 14: (act, boss, map node count, second boss). The second boss is A10's
            // DoubleBoss on the last act and 0 everywhere else; it comes last so a reader of
            // the first three is unchanged.
            14 => WriteIntArray(
                [
                    run.State.Act,
                    run.State.BossEncounterId,
                    run.State.MapNodes.Count,
                    run.State.SecondBossEncounterId,
                ],
                output
            ),
            // 15: the whole map as (col, row, nodeType) triples, ordered so the
            // readout is stable. Lets the differential test compare map *structure*
            // against a live save's saved_map.points, not just a node count.
            15 => WriteIntArray(
                [
                    .. run
                        .State.MapNodes.Values.OrderBy(n => n.Row)
                        .ThenBy(n => n.Col)
                        .SelectMany(n => new[] { n.Col, n.Row, n.NodeType }),
                ],
                output
            ),
            // 16: every edge as a (col, row, childCol, childRow) quadruple. Node
            // positions alone do not pin a map — the same 62 dots can be wired
            // differently — and the live save records each point's `children`, so the
            // differential test can check connectivity rather than assume it.
            16 => WriteIntArray(
                [
                    .. run
                        .State.MapNodes.Values.OrderBy(n => n.Row)
                        .ThenBy(n => n.Col)
                        .SelectMany(n =>
                            n.Children.OrderBy(c => c.Row)
                                .ThenBy(c => c.Col)
                                .SelectMany(c => new[] { n.Col, n.Row, c.Col, c.Row })
                        ),
                ],
                output
            ),
            // 17: the cards on an OFFER grid, if one is open. A card-select phase is two
            // different screens wearing one phase -- a grid of cards the run has rolled
            // and is offering (CardSelectCmd.FromChooseACardScreen, which the game calls
            // `card_select` and resolves on the click) versus a selection over the deck
            // (which toggles and needs a confirm). A replay has to tell them apart to know
            // whether an answer needs a confirm after it, and guessing from a screen
            // message is not the same as asking the run.
            17 => WriteIntArray(run.State.PendingOfferCards, output),
            // 18: Scroll Boxes' bundles, flat -- bundle 0's three cards then bundle 1's.
            // The screen offers a WHOLE bundle, so an agent needs all six to choose
            // between them, and a replay needs them to map the live screen's indexes.
            18 => WriteIntArray(run.State.BundleOffer, output),
            // 19: the DefId of every enemy in the active combat, in the engine's own enemy
            // order -- the DEAD included, because the enemy list keeps its dead where the
            // game removes them, and slot i here has to name the same creature as enemy
            // slot i of the observation. The observation carries an enemy's hp, block,
            // intent and buffs but never says WHICH enemy it is, so nothing outside the
            // engine could put a name to what it was fighting.
            19 => WriteIntArray(
                run.State.ActiveCombat is { } combat
                    ? [.. combat.Enemies.Select(enemy => enemy.DefId)]
                    : [],
                output
            ),
            // 20: what an open card-select screen is FOR, as
            // (DeckSelection, its argument, whether it is a rest-site upgrade). The
            // card-select phase offers a list of the deck and says nothing about what
            // answering it does -- removal, upgrade, transform and Dolly's Mirror are one
            // screen and four different decisions, and the difference is not recoverable
            // from the cards on offer.
            20 => WriteIntArray(
                [
                    (int)run.State.PendingSelectionKind,
                    run.State.PendingSelectionArg,
                    run.State.PendingRestUpgrade ? 1 : 0,
                ],
                output
            ),
            // 21: where the run is standing, as (col, row). Lists 15 and 16 give the
            // map's nodes and its edges but not the position they are read from, and
            // without it a drawn map has no "you are here" -- which is most of what
            // looking at a map is for. At the start of an act that is the virtual node
            // below row 1, which is a real entry in MapNodes and carries the act's
            // ancient rather than NodeNone.
            21 => WriteIntArray(
                [run.State.CurrentMapCoord.Col, run.State.CurrentMapCoord.Row],
                output
            ),
            // v21: which shop cards the merchant shows upgraded (an egg's doing), and which
            // cards on an open offer grid rolled an upgrade.
            22 => WriteBoolArray(run.State.ShopCardUpgraded, output),
            23 => WriteBoolArray(run.State.PendingOfferUpgraded, output),
            _ => -3,
        };
    }

    public static int Sts2Run_GetPhase(int handle)
    {
        return TryGet(handle, out var run) ? (int)run.State.Phase : -1;
    }

    /// <summary>
    /// Hand a run extra HP, for soaking only. A random or greedy policy dies around
    /// floor six and never exercises the back half of the act, so scripts/soak_act_one.py
    /// uses this to reach the boss. It is a DEBUG hook and not a game rule: anything a
    /// boosted soak turns up has to be reproduced on an untouched run before it counts.
    /// </summary>
    public static unsafe int Sts2Run_DebugSetHp(int handle, int hp, int maxHp, int* obsBuf)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        run.State.PlayerMaxHp = Math.Max(1, maxHp);
        run.State.PlayerHp = Math.Clamp(hp, 1, run.State.PlayerMaxHp);
        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        return 0;
    }

    /// <summary>
    /// <c>CreatureCmd.GainMaxHp</c>: raise the maximum and heal by the same amount.
    /// </summary>
    /// <remarks>
    /// The mirror of the mod's <c>debug_gain_max_hp</c>, and it exists so a BUFFED live
    /// capture can be replayed. Not the same as <c>DebugSetHp</c>, which sets absolutes:
    /// the game's command heals as it raises, so a replay that only moved the maximum
    /// would diverge on HP one step after the buff. It routes through the same
    /// <c>RunNonCombatEffects.GainMaxHp</c> that every relic uses, which is the version
    /// captures have already checked.
    /// </remarks>
    public static unsafe int Sts2Run_DebugGainMaxHp(int handle, int amount, int* obsBuf)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        Core.Run.RunNonCombatEffects.GainMaxHp(run.State, amount);
        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        return 0;
    }

    /// <summary>
    /// <c>RunManager.EnterNextAct</c> on demand — the mod's <c>debug_enter_next_act</c>.
    /// </summary>
    /// <remarks>
    /// Not a shortcut around the rules: it calls the same <c>RunEngine.EnterNextAct</c>
    /// the boss reward does. What it skips is having to WIN act 1 first, which is several
    /// minutes of buffed run per act-2 data point and a boss fight that can lose.
    /// Returns 1 when an act was entered and 0 when the run was already in its last.
    /// </remarks>
    public static unsafe int Sts2Run_DebugEnterNextAct(int handle, int* obsBuf)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        bool entered = run.EnterNextAct();
        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        return entered ? 1 : 0;
    }

    /// <summary>Upgrade every upgradable card in the deck. Debug hook, as above.</summary>
    /// <remarks>
    /// Every mutating export refreshes the observation before returning, and these three
    /// did not — which is invisible for a soak, because it steps immediately afterwards
    /// and the next step rewrites it anyway, and fatal for a replay, whose snapshot for
    /// the buff step IS the buffed state. The deck is read out of the observation buffer
    /// while HP is read from the live info struct, so an unrefreshed buffer showed the
    /// max HP moving and the upgrades not happening at all.
    /// </remarks>
    public static unsafe int Sts2Run_DebugUpgradeDeck(int handle, int* obsBuf)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        for (int i = 0; i < run.State.Deck.Count; i++)
        {
            if (Core.Run.RunConstants.IsRunCardUpgradable(run.State.Deck[i]))
            {
                run.State.Deck[i] = run.State.Deck[i] with { Upgraded = true };
            }
        }

        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        return 0;
    }

    /// <summary>
    /// Win the fight in progress without playing it -- <see cref="RunEngine.DebugResolveCombat" />,
    /// with the fight's reward-relevant results (Heist gold, a Gremlin Merc's escape).
    /// Returns -1 when no fight is in progress.
    /// </summary>
    public static unsafe int Sts2Run_DebugResolveCombat(
        int handle,
        int stolenBackGold,
        int fatGremlinEscaped,
        int mercGoldWasStolen,
        int* obsBuf
    )
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        int status = run.DebugResolveCombat(
            stolenBackGold,
            fatGremlinEscaped != 0,
            mercGoldWasStolen != 0
        );
        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        return status;
    }

    /// <summary>
    /// Set the player's holdings -- <see cref="RunEngine.DebugSetPlayer" />. A negative
    /// scalar or a negative list length leaves that holding alone; cards are four numbers
    /// each (id, upgraded, enchantment, amount).
    /// </summary>
    public static unsafe int Sts2Run_DebugSetPlayer(
        int handle,
        int hp,
        int maxHp,
        int gold,
        int* cards,
        int cardInts,
        int* relics,
        int relicLen,
        int* potions,
        int potionLen,
        int maxPotionSlots,
        int* obsBuf
    )
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        run.DebugSetPlayer(
            hp,
            maxHp,
            gold,
            cardInts < 0 ? null : new ReadOnlySpan<int>(cards, cardInts).ToArray(),
            relicLen < 0 ? null : new ReadOnlySpan<int>(relics, relicLen).ToArray(),
            potionLen < 0 ? null : new ReadOnlySpan<int>(potions, potionLen).ToArray(),
            maxPotionSlots
        );
        run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
        return 0;
    }

    /// <summary>The id tables <see cref="Sts2Run_ReanchorFight" /> reads a record through.</summary>
    private static FightAnchor.Tables? _anchorTables;

    /// <summary>
    /// Hand over, once per process, the game's id tables the fight loader reads a record through:
    /// UTF-8 JSON <c>{"id_map": data/id_map.json, "aliases": tools/fidelity/power_aliases.json}</c>.
    /// The library reads no file, so the caller does. Returns 0, or -1 when the text is not those two objects.
    /// </summary>
    public static unsafe int Sts2Run_SetAnchorTables(byte* json, int len)
    {
        try
        {
            var doc = JsonNode.Parse(new ReadOnlySpan<byte>(json, len))!.AsObject();
            _anchorTables = new FightAnchor.Tables(doc["id_map"]!.AsObject(), doc["aliases"]!.AsObject());
            return 0;
        }
        catch (Exception)
        {
            return -1;
        }
    }

    /// <summary>
    /// Put the fight the run stands in on the record of the game's fight: UTF-8 JSON of the referee's
    /// combat snapshot as an agent sees it (hand, draw pile as a sorted list, discard, exhaust, living
    /// enemies with their powers and intents, player hit points, block, energy and powers, round; no
    /// streams). Written over a copy of the fight and swapped in only when the whole record could be
    /// loaded, so a refused record leaves the run as it was. The combat's own streams, draw order (where
    /// its cards are the record's) and relic and enemy hidden state stay: the record cannot say them.
    /// <paramref name="msgBuf" /> receives a NUL-terminated UTF-8 message: what was set on success, the
    /// reason on a refusal.
    /// </summary>
    /// <returns>
    /// The number of fields that were set (0 or more); -1 no fight in progress or no such run; -2 the
    /// record is not a JSON object; -3 no tables were handed over; -4 the record names something the
    /// emulator has no id for or an enemy it has no counterpart of; -5 another failure.
    /// </returns>
    public static unsafe int Sts2Run_ReanchorFight(
        int handle,
        byte* record,
        int recordLen,
        byte* msgBuf,
        int msgLen,
        int* obsBuf
    ) => ReanchorFightWith(handle, record, recordLen, 0, msgBuf, msgLen, obsBuf);

    /// <summary>The flag of <see cref="Sts2Run_ReanchorFightHidden" />: the record's "rng" holds the game's streams.</summary>
    public const int AnchorStreams = 1;

    /// <summary>The flag of <see cref="Sts2Run_ReanchorFightHidden" />: the record's draw pile is in the game's order.</summary>
    public const int AnchorDrawOrder = 2;

    /// <summary>
    /// <see cref="Sts2Run_ReanchorFight" /> with the game's hidden state. <paramref name="flags" />:
    /// <see cref="AnchorStreams" /> (the record's "rng" is the game's run streams, <c>{name: {seed, counter}}</c>: the
    /// combat's streams are put on them, and so are the run's own, which the end of the fight reads) and
    /// <see cref="AnchorDrawOrder" /> (the record's "draw" is the draw pile top first, as the game holds it, not the
    /// sorted list an agent is shown). Flags 0 is the call above. Only the lockstep's emulator side may call this
    /// (the record is the referee's hidden state), never a search.
    /// </summary>
    public static unsafe int Sts2Run_ReanchorFightHidden(
        int handle,
        byte* record,
        int recordLen,
        int flags,
        byte* msgBuf,
        int msgLen,
        int* obsBuf
    ) => ReanchorFightWith(handle, record, recordLen, flags, msgBuf, msgLen, obsBuf);

    private static unsafe int ReanchorFightWith(
        int handle,
        byte* record,
        int recordLen,
        int flags,
        byte* msgBuf,
        int msgLen,
        int* obsBuf
    )
    {
        string message;
        int status;
        if (!TryGet(handle, out var run))
        {
            message = "no_run";
            status = -1;
        }
        else if (record == null || recordLen < 0)
        {
            // Checked before the span is built: a bad length comes back as a status, never as a throw across the boundary.
            message = "no record";
            status = -2;
        }
        else
        {
            status = ReanchorFight(run, new ReadOnlySpan<byte>(record, recordLen), flags, out message);
            if (status >= 0 && obsBuf != null)
            {
                run.WriteObservation(new Span<int>(obsBuf, RunConstants.RunObsSize));
            }
        }

        if (msgLen > 0)
        {
            var bytes = Encoding.UTF8.GetBytes(message);
            int n = Math.Min(bytes.Length, msgLen - 1);
            new ReadOnlySpan<byte>(bytes, 0, n).CopyTo(new Span<byte>(msgBuf, n));
            msgBuf[n] = 0;
        }

        return status;
    }

    /// <summary><see cref="Sts2Run_ReanchorFight" /> on a run, with the record as bytes.</summary>
    internal static int ReanchorFight(RunEngine run, ReadOnlySpan<byte> record, out string message) =>
        ReanchorFight(run, record, 0, out message);

    /// <summary>The run's stream a game stream name stands for (the property names of the game's RunRngSet and PlayerRngSet).</summary>
    private static GameRng? StreamNamed(RunEngine run, string name) => name switch
    {
        "UpFront" => run.State.Rng.UpFront,
        "Shuffle" => run.State.Rng.Shuffle,
        "UnknownMapPoint" => run.State.Rng.UnknownMapPoint,
        "CombatCardGeneration" => run.State.Rng.CombatCardGeneration,
        "CombatPotionGeneration" => run.State.Rng.CombatPotionGeneration,
        "CombatCardSelection" => run.State.Rng.CombatCardSelection,
        "CombatEnergyCosts" => run.State.Rng.CombatEnergyCosts,
        "CombatTargets" => run.State.Rng.CombatTargets,
        "MonsterAi" => run.State.Rng.MonsterAi,
        "Niche" => run.State.Rng.Niche,
        "CombatOrbGeneration" or "CombatOrbs" => run.State.Rng.CombatOrbs,
        "TreasureRoomRelics" => run.State.Rng.TreasureRoomRelics,
        "Rewards" => run.State.PlayerRng.Rewards,
        "Shops" => run.State.PlayerRng.Shops,
        "Transformations" => run.State.PlayerRng.Transformations,
        _ => null,
    };

    /// <summary>
    /// Why <paramref name="streams" /> cannot be applied, or null. Every entry that names a stream of the run must hold a
    /// seed and a non-negative counter, so a bad payload is refused before anything moves (all or nothing).
    /// </summary>
    internal static string? CheckStreams(RunEngine run, JsonObject streams)
    {
        foreach (var kv in streams)
        {
            if (StreamNamed(run, kv.Key) is null)
            {
                continue;
            }

            if (kv.Value is not JsonObject g
                || g["seed"] is not JsonValue seed || !seed.TryGetValue<long>(out _)
                || g["counter"] is not JsonValue counter || !counter.TryGetValue<int>(out int n) || n < 0)
            {
                return $"bad stream {kv.Key}: needs a seed and a counter of 0 or more";
            }
        }

        return null;
    }

    /// <summary>
    /// Puts the run's streams on the game's: for each <c>{name: {seed, counter}}</c> whose stream here has that seed,
    /// the stream is moved to that counter (backwards too). A name the run has no stream for, or one with another seed
    /// (not the same stream), is left alone. Returns how many streams moved.
    /// </summary>
    internal static int SetStreams(RunEngine run, JsonObject streams)
    {
        if (CheckStreams(run, streams) is { } problem)
        {
            throw new ArgumentException(problem);
        }

        int moved = 0;
        foreach (var kv in streams)
        {
            if (kv.Value is not JsonObject g || StreamNamed(run, kv.Key) is not { } stream)
            {
                continue;
            }

            int seed = unchecked((int)(uint)(long)g["seed"]!);
            int counter = (int)g["counter"]!;
            if (stream.RawSeed != seed || stream.CallCount == counter)
            {
                continue;
            }

            stream.SeekTo(counter);
            moved++;
        }

        return moved;
    }

    /// <summary>
    /// Put the run's streams on the game's -- <see cref="SetStreams" />. UTF-8 JSON <c>{name: {seed, counter}}</c>, as the
    /// referee driver reports a stream set. Returns how many streams moved, -1 no such run, -2 the text is not that object.
    /// </summary>
    public static unsafe int Sts2Run_DebugSetStreams(int handle, byte* json, int len)
    {
        if (!TryGet(handle, out var run))
        {
            return -1;
        }

        if (json == null || len < 0)
        {
            return -2;
        }

        try
        {
            return SetStreams(run, JsonNode.Parse(new ReadOnlySpan<byte>(json, len))!.AsObject());
        }
        catch (Exception)
        {
            return -2;
        }
    }

    internal static int ReanchorFight(RunEngine run, ReadOnlySpan<byte> record, int flags, out string message)
    {
        if (_anchorTables is not { } tables)
        {
            message = "no_tables";
            return -3;
        }

        if (run.State.Phase != RunPhase.Combat || run.State.ActiveCombat is not { } live)
        {
            message = "no_fight";
            return -1;
        }

        JsonObject game;
        try
        {
            game = JsonNode.Parse(record)!.AsObject();
        }
        catch (Exception)
        {
            message = "bad_record";
            return -2;
        }

        try
        {
            var q = live.Clone();
            var patched = new List<string>();
            string? problem = FightAnchor.Anchor(tables, q, game, patched,
                new FightAnchor.Source((flags & AnchorStreams) != 0, (flags & AnchorDrawOrder) != 0));
            if (problem != null)
            {
                message = problem;
                return -4;
            }

            // The enemies' next moves come from the record's intents; with no later turns known, the nearest position that announces them.
            var (moves, unknown) = FightAnchor.AnchorMoves(q, game, patched, _ => []);
            if ((flags & AnchorStreams) != 0 && game["rng"] is JsonObject record_rng && CheckStreams(run, record_rng) is { } bad)
            {
                message = bad;  // refused before the swap: the run is as it was
                return -2;
            }

            run.State.ActiveCombat = q;
            // The anchored fight is the run's and steps through RunEngine.Step's gate (NATIVE_KERNEL.md section 7): it carries
            // the KernelState its clone copied (as EmuDump's re-anchor carries its clone's), bound to this run as the run's
            // fight is (StateCloning.Clone(RunState)), or a fresh one when the fight it was anchored on had none.
            run.BindActiveCombat();
            q.Kernel?.At(q, run.State);
            if ((flags & AnchorStreams) != 0 && game["rng"] is JsonObject streams)
            {
                // The run's own streams stand where the fight began; the end of the fight moves them on to the combat's
                // (forward only), so they go to the game's position too.
                int moved = SetStreams(run, streams);
                if (moved > 0)
                {
                    patched.Add($"run.streams={moved}");
                }
            }

            message = $"moves={moves}" + (unknown == null ? "" : ":" + string.Join(",", unknown)) + ";patched=" + string.Join(",", patched);
            return patched.Count;
        }
        catch (Exception ex)
        {
            message = ex.GetType().Name + ": " + ex.Message;
            return -5;
        }
    }

    public static int Sts2Run_PlayerWon(int handle)
    {
        return TryGet(handle, out var run) && run.State.LastPlayerWon ? 1 : 0;
    }

    public static int Sts2Run_AiStateAssumed(int handle)
    {
        return TryGet(handle, out var run) && run.State.ActiveCombat?.AiStateAssumed == true ? 1 : 0;
    }

    public static int Sts2Run_EncounterId(int handle)
    {
        return TryGet(handle, out var run) ? run.ActiveEncounterId : -1;
    }

    public static int Sts2Run_GetShuffleRngCallCount(int handle)
    {
        return TryGet(handle, out var run) ? run.ActiveShuffleRngCallCount : 0;
    }

    public static int Sts2Run_GetNicheRngCallCount(int handle)
    {
        return TryGet(handle, out var run) ? run.ActiveNicheRngCallCount : 0;
    }

    public static void Sts2Run_Destroy(int handle)
    {
        if ((uint)handle < _pool.Length)
        {
            _pool[handle] = null;
        }
    }

    private static int WriteCardList(IReadOnlyList<CardInstance> cards, Span<int> output)
    {
        int count = Math.Min(cards.Count, output.Length);
        for (int i = 0; i < count; i++)
        {
            output[i] = cards[i].Upgraded ? -cards[i].DefId : cards[i].DefId;
        }

        return cards.Count;
    }

    private static int WriteRelicList(IReadOnlyList<RelicInstance> relics, Span<int> output)
    {
        int count = Math.Min(relics.Count, output.Length);
        for (int i = 0; i < count; i++)
        {
            output[i] = relics[i].DefId;
        }

        return relics.Count;
    }

    private static int WriteRewardList(RunState state, Span<int> output)
    {
        Span<int> rewards = stackalloc int[5];
        rewards[0] = state.RewardGold;
        rewards[1] = state.RewardPotion;
        rewards[2] = state.RelicReward;
        // How many CARD items the screen holds, not whether it holds one. The action mask
        // offers exactly this many after the other items, so the two are one count.
        rewards[3] = RunRewardGenerator.CardItemCount(state);
        // Then the SpecialCardReward rows (a Hopper's stolen card, the Lantern Key), one
        // action each after the card items. Last, so a reader of the first four is unchanged.
        rewards[4] = RunRewardGenerator.SpecialCardItemCount(state);
        int count = Math.Min(rewards.Length, output.Length);
        rewards[..count].CopyTo(output);
        return rewards.Length;
    }

    private static int WriteMapOptionCoords(RunState state, Span<int> output)
    {
        int required = RunConstants.MapChoices * 2;
        int count = Math.Min(required, output.Length);
        for (int i = 0; i < count / 2; i++)
        {
            var coord = state.MapOptionCoords[i];
            output[i * 2] = coord?.Col ?? -1;
            output[i * 2 + 1] = coord?.Row ?? -1;
        }
        return required;
    }

    /// <summary>
    /// The belt as it stands RIGHT NOW: the active combat's while a fight is in progress,
    /// the run's otherwise.
    /// </summary>
    /// <remarks>
    /// A combat is handed a COPY of the run's slots at the start and only syncs back when
    /// it ends, so reading the run's during a fight reported the belt as it stood when the
    /// fight began -- every potion drunk, auto-fired or gained mid-combat was invisible
    /// until the fight was over. Three of the four known run-trace divergences were this:
    /// a Fairy in a Bottle that had already revived the player, a Potion Shaped Rock
    /// already spent, and a Fire Potion the run had picked up. See E492.
    ///
    /// The PHASE guard is not optional and is the whole reason this is a method.
    /// `State.ActiveCombat` is only nulled on `Reset`, so a finished combat lingers on the
    /// run; taking its slots outside `RunPhase.Combat` reports the belt from whatever fight
    /// happened last. Written without the guard first, and it cost twenty-four traces --
    /// `RunEngine`'s observation has always spelled it `State.Phase == RunPhase.Combat ?
    /// State.ActiveCombat : null`, which is exactly this.
    /// </remarks>
    private static int[] ActivePotionSlots(Core.Run.RunState state) =>
        state.Phase == Core.Run.RunPhase.Combat && state.ActiveCombat is { } combat
            ? combat.PotionSlots
            : state.PotionSlots;

    private static int WriteIntArray(int[] values, Span<int> output)
    {
        int count = Math.Min(values.Length, output.Length);
        values.AsSpan(0, count).CopyTo(output);
        return values.Length;
    }

    private static int WriteBoolArray(bool[] values, Span<int> output)
    {
        int count = Math.Min(values.Length, output.Length);
        for (int i = 0; i < count; i++)
        {
            output[i] = values[i] ? 1 : 0;
        }

        return values.Length;
    }

    /// <summary>The run on <paramref name="handle" />, for an export that reads one.</summary>
    internal static bool TryGetRun(
        int handle,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out RunEngine? run
    ) => TryGet(handle, out run);

    private static bool TryGet(
        int handle,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out RunEngine? run
    )
    {
        if ((uint)handle < _pool.Length && _pool[handle] is { } existing)
        {
            run = existing;
            return true;
        }

        run = null;
        return false;
    }
}
