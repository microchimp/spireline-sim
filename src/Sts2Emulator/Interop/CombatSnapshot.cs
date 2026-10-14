using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Interop;

/// <summary>
/// A snapshot document the loader refused, naming the field at fault by its JSON path.
/// </summary>
public sealed class SnapshotException : Exception
{
    public SnapshotException(string path, string message)
        : base($"{path}: {message}")
    {
        Path = path;
    }

    /// <summary>Where in the document the problem is, e.g. <c>enemies[1].buffs[0].id</c>.</summary>
    public string Path { get; }
}

/// <summary>
/// `sts2-emu-snapshot/1`: a combat position as JSON. <see cref="Read" /> builds a FRESH
/// <see cref="CombatState" /> from one, and <see cref="Write" /> writes any combat back out
/// in the same form -- including one <c>CombatFactory.Reset</c> built.
/// </summary>
/// <remarks>
/// <para>
/// The binding schema is spire-coach docs/superpowers/specs/2026-09-24-snapshot-schemas.md
/// §2. Field names are the snake_case of the C# fields they fill, every id is the
/// emulator's int, and an omitted optional field takes the default the C# type declares.
/// Two exceptions: <c>encounter_id</c> omitted or null means UNKNOWN, held as -1 because
/// 0 is the Cultists; and <c>whispering_earring_spent</c> omitted means TRUE, so a loaded
/// turn-one position never replays the earring's auto-plays.
/// </para>
/// <para>
/// Loading skips the whole start-of-combat pipeline -- no roster creation, no intent roll,
/// no shuffle, no opening draw, no relic hooks -- because the position it describes has
/// been through all of that already. Everything is assigned directly, and buffs are
/// PLACED rather than applied: <c>BuffSystem.Apply</c> lets Artifact eat an incoming
/// debuff, and a debuff in the document has already landed.
/// </para>
/// <para>
/// The loader refuses what it cannot honour exactly, naming the field: an unknown id, a
/// missing required field, a value of the wrong type, a field the schema does not have,
/// a count past what the game allows (eleven cards in hand, a seventh enemy, a seventh
/// potion slot), a buff list the engine could not hold (one BuffId twice, or an amount
/// of zero), or a draw-pile candidate it cannot place (below). It never drops or trims
/// anything to make a document fit -- a state that is silently half-built gives
/// confident advice about a fight nobody is in. Buffs past the observation's ten
/// (player) or five (enemy) slots are loaded all the same.
/// </para>
/// <para>
/// The load shuffles the draw pile's unordered middle with its seed, so an index into
/// that region would name whatever card the seed put there. A draw-pile selection names
/// its candidates there by card, in <c>candidate_cards</c>, matched in the document's list
/// and followed through the shuffle; only its candidates at the known ends are indices.
/// The export writes a draw-pile selection the same way, so the offer survives any load
/// seed. See <see cref="ReadCandidates" />.
/// </para>
/// <para>
/// <c>counters</c> carries every per-turn and per-combat field of CombatState that has no
/// place elsewhere in the document. Deliberately left out are the fields idle whenever a
/// step has returned: <c>TargetEnemyIndex</c> (Step's own argument, overwritten on entry),
/// <c>AutoPlaying</c> and
/// <c>ResolvingDefendCard</c> (restored around each play), <c>AutoPlayQueue</c> (drained
/// before Step returns), <c>StrangleBeforePlay</c> (captured afresh at the start of every
/// play) and <c>ExecutedThisEnemyTurn</c> (differential-testing introspection). Nor is
/// <c>PlayerTurn</c>, which is true whenever the player is to act and which the loader
/// sets. The RNG streams are not carried either: the live game's are hidden information,
/// so the caller seeds them from its own load seed. The one exception is the save document
/// (below), which is not this one.
/// </para>
/// <para>
/// A combat stopped on a screen raised in the middle of a step (a card's OnPlay, a potion, the
/// turn end, the turn start) is the one position a step has NOT returned from: the screen stands over
/// the combat as it is at that moment, with the card being played in the play pile, and the rest of
/// the step -- the card's way to its pile and its after-play hooks, the hand draw after Foregone
/// Conclusion's pick, the turn end after Well-Laid Plans' -- runs once the screen is answered.
/// <see cref="Write" /> writes it with the public part of the step it stopped (<c>stopped_step</c>,
/// <see cref="WriteStoppedStep" />): the kind of step, the card or potion in play, the action, the
/// answers so far, the step's start as the player saw it, written as this document is written, and
/// what the player saw the step do before the screen: the cards it took off the draw pile in order
/// (<c>taken</c>: the hand it drew, the card Havoc, Mayhem or Hellraiser auto-played) and the screens
/// it reached (<c>screens</c>). Never the start's true draw order or the streams: those name the
/// future (a stopped Burning Pact's true start names the two cards it draws). So two determinized
/// copies of one stopped combat write the same document. Loaded, the step goes on from the screen as
/// it does in memory: its answer replays the step from the public start in the loader's world, held
/// to <c>taken</c> and <c>screens</c> -- it takes exactly those cards and reaches exactly those
/// screens, or the answer is refused -- and where the replay reaches the last screen the loaded copy
/// is put back whole (<see cref="HookChoiceReplay" />, <see cref="ScreenHiddenState" />), so the
/// counts, the counters and the rest of the turn come out as answering in memory does. A step whose
/// replay does not reach its screen is refused at load (<see cref="Read" />), and so at export
/// (<see cref="Write" />): never written as a document that would load as another position. A
/// document from the live game has no such block (the live converter finishes the play in its
/// document instead).
/// </para>
/// <para>
/// <see cref="WriteSave" /> is the emulator's save document: the same document plus the true world of
/// the stopped step (<c>before_draw</c>, <c>before_streams</c>, <c>answer_indices</c>, <c>hidden</c>):
/// the start's draw order and streams and the hidden state of each answering copy, so that a load
/// replays the step from its true start. It carries the true draw order and the RNG positions, so it
/// is banned outside verification and save code, as <c>Sts2_GetPile</c> and <c>Sts2_RngCallCount</c>
/// are (review condition E12): native <c>Sts2_GetSaveJson</c>, Python <c>native.get_save_json</c>.
/// </para>
/// <para>
/// JsonDocument and Utf8JsonWriter only, never the reflection serializer: the core is
/// NativeAOT.
/// </para>
/// </remarks>
public static class CombatSnapshot
{
    public const string SchemaId = "sts2-emu-snapshot/1";

    /// <summary>The value an unset <c>cost_for_combat</c>/<c>cost_this_turn</c> holds; JSON null.</summary>
    private const int Unset = int.MinValue;

    /// <summary>
    /// Build a combat from a document. <paramref name="seed" /> shuffles the draw pile's
    /// unordered middle; nothing else in the state is random until its streams are seeded.
    /// </summary>
    /// <exception cref="SnapshotException">The document is malformed; the message names the field.</exception>
    public static CombatState Read(ReadOnlySpan<byte> json, int seed, out int turn)
    {
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(json.ToArray());
        }
        catch (JsonException e)
        {
            throw new SnapshotException("$", $"not valid JSON: {e.Message}");
        }

        using (document)
        {
            var state = ReadCombat(document.RootElement, seed);
            turn = state.Turn;
            return state;
        }
    }

    /// <summary>
    /// The combat as a document. Every optional field is written explicitly, so a document
    /// read back in reproduces the state exactly -- the unseen order and the streams aside, which
    /// the loader's seed decides, and a step stopped on a screen, of which it writes the public
    /// part (see the remarks): use it wherever an agent, a search or the live game's side reads
    /// positions.
    /// </summary>
    /// <exception cref="SnapshotException">
    /// The combat is stopped in the middle of a step whose replay from the document's public start
    /// does not reach its screen (<c>CombatEngine.CheckStoppedStep</c>): its document would load as
    /// another position, so none is written. The path is <c>stopped_step</c>.
    /// </exception>
    public static byte[] Write(CombatState state)
    {
        var document = WriteDocument(state, save: false);
        if (state.PendingSelection?.Replay is not null)
        {
            try
            {
                Read(document, ExportCheckSeed, out _);
            }
            catch (SnapshotException e)
            {
                throw new SnapshotException("stopped_step", $"the stopped step cannot be written as a document that loads back as this position ({e.Message})");
            }
        }

        return document;
    }

    /// <summary>
    /// <see cref="Write" />'s document without its export check, for code that prints a state and
    /// never loads it (EmuDump's dumps, the trace): a stopped step it would refuse is written all the
    /// same. Never for a document an agent, a search or a loader reads.
    /// </summary>
    public static byte[] WriteUnchecked(CombatState state) => WriteDocument(state, save: false);

    /// <summary>The load seed <see cref="Write" /> reads its own document back with, to check it loads.</summary>
    private const int ExportCheckSeed = 1;

    /// <summary>
    /// The save document: <see cref="Write" />'s, plus the true world of the step stopped on the
    /// screen (<c>stopped_step</c>): its start's true draw order and streams, and each answer's
    /// true index and hidden state. Banned outside verification and save code (review condition
    /// E12), as the true pile is.
    /// </summary>
    public static byte[] WriteSave(CombatState state) => WriteDocument(state, save: true);

    private static byte[] WriteDocument(CombatState state, bool save)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            WriteCombat(writer, state, save);
        }

        return buffer.WrittenSpan.ToArray();
    }

    // ── counters ─────────────────────────────────────────────────────────────────────

    /// <summary>A CombatState field by reference, so one accessor both reads and writes it.</summary>
    private delegate ref T FieldRef<T>(CombatState state);

    private sealed record IntCounter(string Name, FieldRef<int> Field, bool NullIsUnset = false);

    private sealed record BoolCounter(string Name, FieldRef<bool> Field, bool WhenOmitted = false);

    private static readonly IntCounter[] IntCounters =
    [
        new("stolen_back_gold", s => ref s.StolenBackGold),
        new("royalties_gold", s => ref s.RoyaltiesGold),
        new("paels_legion_cooldown", s => ref s.PaelsLegionCooldown),
        new("afterimage_before_play", s => ref s.AfterimageBeforePlay),
        new("storm_before_play", s => ref s.StormBeforePlay),
        new("subroutine_before_play", s => ref s.SubroutineBeforePlay),
        new("status_cards_drawn_this_turn", s => ref s.StatusCardsDrawnThisTurn),
        new("played_card_cost_for_combat", s => ref s.PlayedCardCostForCombat, NullIsUnset: true),
        new("played_card_bonus_damage", s => ref s.PlayedCardBonusDamage),
        new("played_card_bonus_block", s => ref s.PlayedCardBonusBlock),
        new("energy_spent_this_turn", s => ref s.EnergySpentThisTurn),
        new("ethereal_exhausts_this_turn", s => ref s.EtherealExhaustsThisTurn),
        new("unblocked_damage_this_turn", s => ref s.UnblockedDamageThisTurn),
        new("rainbow_ring_attacks", s => ref s.RainbowRingAttacks),
        new("rainbow_ring_skills", s => ref s.RainbowRingSkills),
        new("rainbow_ring_powers", s => ref s.RainbowRingPowers),
        new("gambling_chip_discarded", s => ref s.GamblingChipDiscarded),
        new("turn_start_held_phase", s => ref s.TurnStartHeldPhase),
        new("turn_start_held_relic", s => ref s.TurnStartHeldRelic),
        new("played_card_cost_bump", s => ref s.PlayedCardCostBump),
        new("cards_played_this_turn", s => ref s.CardsPlayedThisTurn),
        new("card_plays_started_this_turn", s => ref s.CardPlaysStartedThisTurn),
        new("card_play_series_started_this_turn", s => ref s.CardPlaySeriesStartedThisTurn),
        new("attack_cards_played_this_turn", s => ref s.AttackCardsPlayedThisTurn),
        new("attack_or_skill_cards_played_this_turn", s => ref s.AttackOrSkillCardsPlayedThisTurn),
        new("card_plays_this_turn", s => ref s.CardPlaysThisTurn),
        new("cards_played_this_combat", s => ref s.CardsPlayedThisCombat),
        new("cards_drawn_this_combat", s => ref s.CardsDrawnThisCombat),
        new("shivs_played_this_turn", s => ref s.ShivsPlayedThisTurn),
        new("extra_card_rewards", s => ref s.ExtraCardRewards),
        new("block_gains_this_turn", s => ref s.BlockGainsThisTurn),
        new("player_hp_lost_this_turn", s => ref s.PlayerHpLostThisTurn),
        new("cards_exhausted_this_turn", s => ref s.CardsExhaustedThisTurn),
        new("lightning_orbs_channeled_this_combat", s => ref s.LightningOrbsChanneledThisCombat),
        new("ethereal_exhaust_count", s => ref s.EtherealExhaustCount),
        new("ethereal_card_plays_this_combat", s => ref s.EtherealCardPlaysThisCombat),
        new("skill_cards_played_this_turn", s => ref s.SkillCardsPlayedThisTurn),
        new("bound_cards_this_turn", s => ref s.BoundCardsThisTurn),
        new("card_plays_last_turn", s => ref s.CardPlaysLastTurn),
        new("extra_cards_drawn_this_turn", s => ref s.ExtraCardsDrawnThisTurn),
        new("osty_attacks_this_turn", s => ref s.OstyAttacksThisTurn),
        new("stars_gained_this_turn", s => ref s.StarsGainedThisTurn),
        new("played_card_stars_spent", s => ref s.PlayedCardStarsSpent),
        new("void_form_cards_played_this_turn", s => ref s.VoidFormCardsPlayedThisTurn),
        new("cards_generated_this_combat", s => ref s.CardsGeneratedThisCombat),
        new("tender_cards_played_this_turn", s => ref s.TenderCardsPlayedThisTurn),
        new("rupture_owed", s => ref s.RuptureOwed),
        new("unblocked_damage_hit_count", s => ref s.UnblockedDamageHitCount),
        new("player_hit_turn", s => ref s.PlayerHitTurn),
        new("player_hit_turn_before", s => ref s.PlayerHitTurnBefore),
    ];

    private static readonly BoolCounter[] BoolCounters =
    [
        new("fat_gremlin_escaped", s => ref s.FatGremlinEscaped),
        new("battleworn_dummy_ran_out_of_time", s => ref s.BattlewornDummyRanOutOfTime),
        new("merc_gold_was_stolen", s => ref s.MercGoldWasStolen),
        new("pen_nib_armed", s => ref s.PenNibArmed),
        new("vambrace_spent", s => ref s.VambraceSpent),
        new("vambrace_card", s => ref s.VambraceCard),
        new("paels_legion_card", s => ref s.PaelsLegionCard),
        new("rainbow_ring_paid_this_turn", s => ref s.RainbowRingPaidThisTurn),
        new("unsettling_lamp_spent", s => ref s.UnsettlingLampSpent),
        new("unsettling_lamp_card", s => ref s.UnsettlingLampCard),
        new("burning_sticks_used", s => ref s.BurningSticksUsed),
        new("belt_buckle_applied", s => ref s.BeltBuckleApplied),
        new("took_unblocked_damage", s => ref s.TookUnblockedDamage),
        new("played_card_enchant_spent", s => ref s.PlayedCardEnchantSpent),
        new("played_card_enchant_grew", s => ref s.PlayedCardEnchantGrew),
        // A position read off the game on turn one is past the pre-play phase, so an
        // earring the document does not mention has already done its auto-plays.
        new("whispering_earring_spent", s => ref s.WhisperingEarringSpent, WhenOmitted: true),
        new("skill_played_while_smoggy", s => ref s.SkillPlayedWhileSmoggy),
        new("lethal_attack_play", s => ref s.LethalAttackPlay),
        new("bound_card_played_this_turn", s => ref s.BoundCardPlayedThisTurn),
        new("music_box_used_this_turn", s => ref s.MusicBoxUsedThisTurn),
        new("doom_applied_this_turn", s => ref s.DoomAppliedThisTurn),
        new("end_turn_after_play", s => ref s.EndTurnAfterPlay),
    ];

    private const string LastAttackOrSkillThisTurn = "last_attack_or_skill_this_turn";
    private const string LastAttackOrSkillLastTurn = "last_attack_or_skill_last_turn";
    private const string StolenBackCards = "stolen_back_cards";

    // ── the fields each object may hold ──────────────────────────────────────────────

    private static readonly string[] CombatFields =
    [
        "schema",
        "ascension",
        "character",
        "encounter_id",
        "is_elite_combat",
        "is_elite_room",
        "fur_coat_bites",
        "turn",
        "player",
        "relics",
        "potions",
        "piles",
        "incoming",
        "enemies",
        "counters",
        "pending_selection",
        "stopped_step",
    ];

    private static readonly string[] StoppedStepFields =
        ["step", "card", "potion", "action", "target", "answers", "taken", "screens", "peeks", "before", "answer_indices", "before_draw", "before_streams", "hidden"];

    /// <summary>A screen the stopped step reached (<see cref="StepScreen" />).</summary>
    private static readonly string[] ScreenFields = ["kind", "source", "count", "taken", "cards", "generated"];

    /// <summary>An answer the document names by card: a pick from the draw pile.</summary>
    private static readonly string[] AnswerFields = ["card"];

    private static readonly string[] StreamFields = ["seed", "count", "lanes"];

    private static readonly string[] HiddenFields = ["streams", "draw", "candidates"];

    /// <summary>
    /// The streams a stopped step's block names: the combat's own (<c>StateCloning.StreamNames</c>,
    /// in snake case) and the step's.
    /// </summary>
    private static readonly string[] StreamNames =
    [
        .. StateCloning.StreamNames.Select(name => System.Text.RegularExpressions.Regex.Replace(name, "(?<=[a-z])(?=[A-Z])", "_").ToLowerInvariant()),
        "step",
    ];

    private static readonly string[] PlayerFields =
    [
        "hp",
        "max_hp",
        "block",
        "energy",
        "max_energy",
        "gold",
        "stars",
        "osty_hp",
        "osty_max_hp",
        "orb_capacity",
        "base_orb_slots",
        "orbs",
        "buffs",
        "debuffs_at_round_start",
    ];

    private static readonly string[] OrbFields = ["type", "evoke", "passive"];
    private static readonly string[] BuffFields = ["id", "amount", "data"];
    private static readonly string[] RelicFields = ["id", "counter"];
    private static readonly string[] PotionFields = ["max_slots", "slots"];
    private static readonly string[] PileFields = ["hand", "discard", "exhaust", "draw"];
    private static readonly string[] DrawFields = ["known_top", "known_bottom", "unordered"];
    private static readonly string[] IncomingFields = ["return_to_hand", "copies_to_hand"];

    private static readonly string[] EnemyFields =
    [
        "def_id",
        "slot",
        "hp",
        "max_hp",
        "block",
        "buffs",
        "intent",
        "ai",
    ];

    private static readonly string[] IntentFields = ["type", "magnitude", "hits", "carries_damage"];

    private static readonly string[] AiFields =
    [
        "move_index",
        "last_move",
        "last_move_repeats",
        "once_only_move_used",
        "move_history",
        "starts_on_branch",
        "rolls_next_move",
        "curl_up_armed",
        "last_branch",
        "repeat_streak",
        "branch_cooldown",
        "starter_move",
        "off_balance",
        "nemesis_intangible_on",
        "escaped",
        "last_spawned_def_id",
        "stolen_gold",
        "heist_gold",
        "powered_hits_this_turn",
        "hardened_shell_damage_this_turn",
        "stolen_card",
    ];

    private static readonly string[] CardFields =
    [
        "def_id",
        "upgraded",
        "enchantment",
        "enchant_amount",
        "cost_bump",
        "cost_for_combat",
        "cost_this_turn",
        "free_this_turn",
        "free_until_played",
        "retain_this_turn",
        "retain_for_combat",
        "sly_this_turn",
        "sly_for_combat",
        "ethereal_for_combat",
        "played_this_turn",
        "bound",
        "dampened_upgrade",
        "enchant_spent",
        "replay_count",
        "bonus_damage",
        "bonus_block",
        "tinker_type",
        "tinker_rider",
    ];

    private static readonly string[] SelectionFields =
    [
        "kind",
        "candidates",
        "candidate_cards",
        "source_card_def_id",
        "amount",
        "skippable",
        "generated_candidates",
        "generated_upgraded",
        "generated_retained",
        "after_selection_to_hand",
        "end_turn_awaiting_selection",
    ];

    private static readonly string[] CounterFields =
    [
        .. IntCounters.Select(counter => counter.Name),
        .. BoolCounters.Select(counter => counter.Name),
        LastAttackOrSkillThisTurn,
        LastAttackOrSkillLastTurn,
        StolenBackCards,
    ];

    // ── reading ──────────────────────────────────────────────────────────────────────

    private static CombatState ReadCombat(JsonElement element, int seed)
    {
        var root = new Fields(element, "", CombatFields);
        string schema = root.Text("schema");
        if (schema != SchemaId)
        {
            throw new SnapshotException("schema", $"expected \"{SchemaId}\", got \"{schema}\"");
        }

        var state = new CombatState
        {
            AscensionLevel = root.Int("ascension"),
            Character = root.EnumValue("character", PlayerCharacter.Ironclad, "character"),
            Turn = root.Int("turn"),
            PlayerTurn = true,
            EncounterId = ReadEncounterId(root),
        };
        if (state.Turn < 0)
        {
            throw new SnapshotException("turn", $"{state.Turn}; the emulator's turn counts from 0");
        }

        state.IsEliteCombat = root.Bool("is_elite_combat", state.IsEliteCombat);
        state.IsEliteRoom = root.Bool("is_elite_room", state.IsEliteRoom);
        state.FurCoatBites = root.Bool("fur_coat_bites", state.FurCoatBites);

        ReadPlayer(root.Child("player", PlayerFields), state);
        state.Relics = root.Items("relics", ReadRelic);
        ReadPotions(root.OptionalChild("potions", PotionFields), state);
        var unordered = ReadPiles(root.OptionalChild("piles", PileFields), state, seed);

        var incoming = root.OptionalChild("incoming", IncomingFields);
        state.ReturnToHandBeforeDraw = incoming.Items("return_to_hand", ReadCard);
        state.CopiesToHandBeforeDraw = incoming.Items("copies_to_hand", ReadCard);

        state.Enemies = root.RequiredItems(
            "enemies",
            ReadEnemy,
            NativeExports.MAX_ENEMIES,
            "enemies"
        );
        if (state.Enemies.Count == 0)
        {
            throw new SnapshotException("enemies", "no enemies; a combat needs at least one");
        }

        // The one derived field: what each enemy's move declares beside its main intent.
        EnemyAI.UpdateSecondaryIntents(state.Enemies);

        ReadCounters(root.OptionalChild("counters", CounterFields), state);
        ReadPendingSelection(root, state, unordered);
        ReadStoppedStep(root, state, seed);
        return state;
    }

    /// <summary>
    /// A screen raised in the middle of a step (<see cref="HookChoiceReplay" />): the step it stopped,
    /// as <see cref="WriteStoppedStep" /> wrote it, put back on the screen so that its answer runs the
    /// rest of the step. Refused without a screen to put it on, and when its kind or the card or
    /// potion in play is not what its start and action make it. The agent's block (no
    /// <c>before_streams</c>) is a step from a document (<see cref="HookChoiceReplay.FromDocument" />):
    /// its start is read with <paramref name="seed" /> like the rest of the document, and its answers
    /// have no hidden state. The save document's adds the start's true order and streams and each
    /// answer's hidden state and true index.
    /// </summary>
    private static void ReadStoppedStep(Fields root, CombatState state, int seed)
    {
        if (!root.TryGet("stopped_step", out var element) || element.ValueKind == JsonValueKind.Null)
        {
            return;
        }

        var fields = new Fields(element, "stopped_step", StoppedStepFields);
        if (state.PendingSelection is not { } screen)
        {
            throw new SnapshotException("stopped_step", "a stopped step needs the screen it stopped on (pending_selection)");
        }

        if (!fields.TryGet("before", out var beforeElement))
        {
            throw new SnapshotException(fields.At("before"), "missing");
        }

        var before = ReadCombat(beforeElement, seed);
        if (before.PendingSelection?.Replay is not null)
        {
            throw new SnapshotException(fields.At("before"), "the step's start cannot itself be stopped");
        }

        // Each answer is an index, or {"card": ...} for a pick from the draw pile, named by its card.
        var given = fields.Items("answers", ReadAnswer);
        var taken = fields.RequiredItems("taken", ReadCard);
        var screens = fields.RequiredItems("screens", ReadScreen);
        // What each look at the top decided: whether I Am Invincible played itself.
        var peeks = fields.RequiredItems("peeks", AsBool);
        if (screens.Count != given.Count + 1)
        {
            throw new SnapshotException(fields.At("screens"), $"{screens.Count} screens for {given.Count} answers; a stopped step reaches one screen per answer and the one it stopped on");
        }

        if (screens.Select(reached => reached.Taken).Prepend(0).Zip(screens.Select(reached => reached.Taken)).Any(pair => pair.First > pair.Second) || screens[^1].Taken != taken.Count)
        {
            throw new SnapshotException(fields.At("screens"), $"the takes before each screen must run up to the {taken.Count} cards taken");
        }

        if (screens[^1].Differs(screen, taken.Count) is { } other)
        {
            throw new SnapshotException(fields.At("screens"), $"the last screen is not the one pending_selection shows: pending_selection is {other}");
        }
        HookChoiceReplay replay;
        if (!fields.TryGet("before_streams", out _))
        {
            foreach (string saveOnly in (string[])["before_draw", "hidden", "answer_indices"])
            {
                if (fields.TryGet(saveOnly, out _))
                {
                    throw new SnapshotException(fields.At(saveOnly), "a save document's field without the step's streams (before_streams)");
                }
            }

            replay = new HookChoiceReplay
            {
                Before = before,
                RngBefore = null,
                Action = fields.Int("action"),
                TargetEnemyIndex = fields.Int("target"),
                Answers = given,
                Hidden = [.. given.Select(_ => ScreenHiddenState.None)],
                Taken = taken,
                Screens = screens,
                Peeks = peeks,
            };
        }
        else
        {
            var order = fields.Items("before_draw", ReadCard);
            if (!SameCards(order, before.DrawPile))
            {
                throw new SnapshotException(fields.At("before_draw"), "not the cards of before's draw pile");
            }

            before.DrawPile = order;
            var (streams, stepRng) = ReadStreams(fields.Child("before_streams", StreamNames));
            for (int i = 0; i < streams.Length; i++)
            {
                StateCloning.SetStream(before, i, streams[i]);
            }

            var hidden = fields.Items(
                "hidden",
                (item, path) =>
                {
                    var entry = new Fields(item, path, HiddenFields);
                    var (kept, keptStep) = ReadStreams(entry.Child("streams", StreamNames));
                    var candidates = entry.NullableItems("candidates", ReadCandidate);
                    return new ScreenHiddenState(kept, keptStep, [.. entry.Items("draw", ReadCard)], candidates?.ToArray());
                }
            );
            var answers = fields.Items("answer_indices", (item, path) => AsInt(item, path));
            if (hidden.Count != answers.Count || given.Count != answers.Count)
            {
                throw new SnapshotException(
                    fields.At("answer_indices"),
                    $"{answers.Count} answer indices for {given.Count} answers and {hidden.Count} hidden states"
                );
            }

            replay = new HookChoiceReplay
            {
                Before = before,
                RngBefore = stepRng ?? throw new SnapshotException(fields.At("before_streams"), "no step stream"),
                Action = fields.Int("action"),
                TargetEnemyIndex = fields.Int("target"),
                Answers = [.. answers.Zip(given, (index, answer) => answer with { Index = index })],
                Hidden = hidden,
                Taken = taken,
                Screens = screens,
                Peeks = peeks,
            };
        }

        var (step, card, potion) = InPlay(replay, state.Turn);
        string written = fields.Text("step");
        if (written != step)
        {
            throw new SnapshotException(fields.At("step"), $"\"{written}\", but the step's start and action make it \"{step}\"");
        }

        if (fields.NullableCard("card") != card)
        {
            throw new SnapshotException(fields.At("card"), "not the card the action plays from the step's start");
        }

        int? potionWritten =
            fields.TryGet("potion", out var potionElement) && potionElement.ValueKind != JsonValueKind.Null
                ? AsInt(potionElement, fields.At("potion"))
                : null;
        if (potionWritten != potion)
        {
            throw new SnapshotException(fields.At("potion"), "not the potion the action drinks from the step's start");
        }

        state.PendingSelection = screen.Clone(replay);

        // The replay must reach the screen the document shows, the way the step went: else an answer
        // would put the loaded copy back at another screen, or finish another step.
        if (CombatEngine.CheckStoppedStep(state, seed) is { } why)
        {
            throw new SnapshotException("stopped_step", $"the step does not replay to its screen from its start: {why}");
        }
    }

    /// <summary>A screen in a stopped step's block (<see cref="StepScreen" />).</summary>
    private static StepScreen ReadScreen(JsonElement item, string path)
    {
        var fields = new Fields(item, path, ScreenFields);
        int kind = fields.Int("kind");
        if (!Enum.IsDefined((CardSelectionKind)kind) || kind == (int)CardSelectionKind.None)
        {
            throw new SnapshotException(fields.At("kind"), $"unknown selection kind {kind}");
        }

        var cards = fields.Items("cards", ReadCard);
        var generated = fields.Items("generated", (element, at) => AsInt(element, at));
        int count = fields.Int("count");
        int taken = fields.Int("taken");
        if (count < 0 || taken < 0 || cards.Count > count || generated.Count > count)
        {
            throw new SnapshotException(path, $"{count} candidates, {cards.Count} cards and {generated.Count} generated, {taken} taken");
        }

        return new StepScreen((CardSelectionKind)kind, fields.Int("source"), count, taken, [.. cards], [.. generated]);
    }

    /// <summary>An answer in a stopped step's block: an index, or a pick from the draw pile by its card (index -1).</summary>
    private static StepAnswer ReadAnswer(JsonElement item, string path)
    {
        if (item.ValueKind != JsonValueKind.Object)
        {
            return new StepAnswer(AsInt(item, path), null);
        }

        var answer = new Fields(item, path, AnswerFields);
        return answer.NullableCard("card") is { } card
            ? new StepAnswer(-1, card)
            : throw new SnapshotException(answer.At("card"), "missing required field");
    }

    /// <summary>
    /// What kind of step <paramref name="stopped" /> is and what it has in play: a card's play (the
    /// card, in the play pile while its screen is open: in none of the document's piles), a potion
    /// (its id), the turn end before the turn changes (Well-Laid Plans) or the next turn's start after
    /// it (Foregone Conclusion, Tyranny, Entropy, Stratagem's shuffle), or the answer to a screen that
    /// stood at the step's start. <paramref name="turn" /> is the turn on the screen.
    /// </summary>
    private static (string Step, CardInstance? Card, int? Potion) InPlay(HookChoiceReplay stopped, int turn)
    {
        var before = stopped.Before;
        int hand = before.Hand.Count;
        if (before.PendingSelection is not null)
        {
            return ("screen", null, null);
        }

        if (stopped.Action >= 0 && stopped.Action < hand)
        {
            return ("card_play", before.Hand[stopped.Action], null);
        }

        if (stopped.Action == hand)
        {
            return (turn != before.Turn ? "turn_start" : "end_turn", null, null);
        }

        int slot = stopped.Action - hand - 1;
        return ("potion", null, slot >= 0 && slot < before.PotionSlots.Length ? before.PotionSlots[slot] : null);
    }

    private static (CountingRandom?[] Streams, CountingRandom? Step) ReadStreams(Fields fields)
    {
        var streams = new CountingRandom?[StateCloning.StreamCount];
        CountingRandom? step = null;
        for (int i = 0; i < StreamNames.Length; i++)
        {
            if (!fields.TryGet(StreamNames[i], out var value) || value.ValueKind == JsonValueKind.Null)
            {
                continue;
            }

            var stream = new Fields(value, fields.At(StreamNames[i]), StreamFields);
            string lanes = stream.Text("lanes");
            var hex = System.Globalization.NumberStyles.HexNumber;
            if (
                lanes.Length != 64
                || !ulong.TryParse(lanes.AsSpan(0, 16), hex, null, out ulong s0)
                || !ulong.TryParse(lanes.AsSpan(16, 16), hex, null, out ulong s1)
                || !ulong.TryParse(lanes.AsSpan(32, 16), hex, null, out ulong s2)
                || !ulong.TryParse(lanes.AsSpan(48, 16), hex, null, out ulong s3)
            )
            {
                throw new SnapshotException(stream.At("lanes"), "expected 64 hex digits, the generator's four lanes");
            }

            var rng = CountingRandom.FromLanes(stream.Int("seed"), stream.Int("count"), (s0, s1, s2, s3));
            if (i < streams.Length)
            {
                streams[i] = rng;
            }
            else
            {
                step = rng;
            }
        }

        return (streams, step);
    }

    private static bool SameCards(List<CardInstance> left, List<CardInstance> right) =>
        left.Count == right.Count
        && left.GroupBy(card => card).All(group => right.Count(card => card.Equals(group.Key)) == group.Count());

    private static int ReadEncounterId(Fields root)
    {
        if (!root.TryGet("encounter_id", out var value) || value.ValueKind == JsonValueKind.Null)
        {
            return -1;
        }

        int id = AsInt(value, "encounter_id");
        return Enum.IsDefined((CombatFactory.ActOneEncounter)id)
            ? id
            : throw new SnapshotException("encounter_id", $"unknown encounter id {id}");
    }

    private static void ReadPlayer(Fields player, CombatState state)
    {
        state.PlayerHp = player.Int("hp");
        state.PlayerMaxHp = player.Int("max_hp");
        state.PlayerBlock = player.Int("block", state.PlayerBlock);
        state.Energy = player.Int("energy");
        state.MaxEnergy = player.Int("max_energy");
        state.PlayerGold = player.Int("gold", state.PlayerGold);
        state.Stars = player.Int("stars", state.Stars);
        state.OstyHp = player.Int("osty_hp", state.OstyHp);
        state.OstyMaxHp = player.Int("osty_max_hp", state.OstyMaxHp);
        state.OrbCapacity = player.Int("orb_capacity", state.OrbCapacity);
        state.BaseOrbSlots = player.Int("base_orb_slots", state.BaseOrbSlots);
        state.Orbs = player.Items("orbs", ReadOrb);
        state.PlayerBuffs = ReadBuffs(player, "buffs");
        // Null means "what the engine would have snapshotted": the duration debuffs held
        // now, which is what `TickDurationDebuffs` records at every round boundary.
        var atRoundStart = player.NullableItems("debuffs_at_round_start", ReadBuff);
        state.PlayerDebuffsAtRoundStart = atRoundStart is null
            ? BuffSystem.DurationDebuffSnapshot(state.PlayerBuffs)
            : DistinctBuffs(atRoundStart, player.At("debuffs_at_round_start"));
    }

    private static void ReadPotions(Fields potions, CombatState state)
    {
        int maxSlots = potions.Int("max_slots", state.MaxPotionSlots);
        if (maxSlots < 0 || maxSlots > RunConstants.PotionSlotCapacity)
        {
            throw new SnapshotException(
                potions.At("max_slots"),
                $"{maxSlots}; must be between 0 and {RunConstants.PotionSlotCapacity}"
            );
        }

        state.MaxPotionSlots = maxSlots;
        if (!potions.TryGet("slots", out var slots))
        {
            return;
        }

        string path = potions.At("slots");
        RequireArray(slots, path);
        if (slots.GetArrayLength() != state.PotionSlots.Length)
        {
            throw new SnapshotException(
                path,
                $"{slots.GetArrayLength()} entries; the slots are {state.PotionSlots.Length} wide"
            );
        }

        int index = 0;
        foreach (var slot in slots.EnumerateArray())
        {
            string slotPath = Item(path, index);
            int id = AsInt(slot, slotPath);
            state.PotionSlots[index++] = id == 0 ? 0 : PotionId(id, slotPath);
        }
    }

    /// <returns>
    /// The unordered middle as the document lists it, each card with the draw-pile index
    /// the load's shuffle took it to, where <see cref="FindInUnorderedMiddle" /> looks.
    /// </returns>
    private static List<(CardInstance Card, int At)> ReadPiles(
        Fields piles,
        CombatState state,
        int seed
    )
    {
        state.Hand = piles.Items("hand", ReadCard, NativeExports.MAX_HAND, "cards");
        state.DiscardPile = piles.Items("discard", ReadCard);
        state.ExhaustPile = piles.Items("exhaust", ReadCard);

        var draw = piles.OptionalChild("draw", DrawFields);
        var top = draw.Items("known_top", ReadCard);
        var bottom = draw.Items("known_bottom", ReadCard);
        var unordered = draw.Items("unordered", ReadCard);
        // The player knows the middle's contents and not its order, so the load picks an
        // order from its own seed. A resampling clone reshuffles exactly this region. Each
        // card is shuffled with its place in the document's list, which leaves the cards
        // in the order they would take alone and says where each listed card went.
        var listed = unordered.Select((card, i) => (Card: card, Listed: i)).ToList();
        CardEffects.ShufflePile(listed, new CountingRandom(seed));
        state.DrawPile = [.. top, .. listed.Select(entry => entry.Card), .. bottom];
        state.KnownTopCount = top.Count;
        state.KnownBottomCount = bottom.Count;

        var at = new int[unordered.Count];
        for (int i = 0; i < listed.Count; i++)
        {
            at[listed[i].Listed] = top.Count + i;
        }

        return [.. unordered.Select((card, i) => (card, at[i]))];
    }

    private static EnemyState ReadEnemy(JsonElement element, string path)
    {
        var fields = new Fields(element, path, EnemyFields);
        var enemy = new EnemyState();
        enemy.DefId = EnemyId(fields.Int("def_id"), fields.At("def_id"));
        enemy.Slot = fields.Int("slot", enemy.Slot);
        enemy.Hp = fields.Int("hp");
        enemy.MaxHp = fields.Int("max_hp");
        enemy.Block = fields.Int("block", enemy.Block);
        enemy.Buffs = ReadBuffs(fields, "buffs");
        enemy.CurrentIntent = ReadIntent(fields.Child("intent", IntentFields));

        var ai = fields.OptionalChild("ai", AiFields);
        enemy.MoveIndex = ai.Int("move_index", enemy.MoveIndex);
        enemy.LastMove = ai.Int("last_move", enemy.LastMove);
        enemy.LastMoveRepeats = ai.Int("last_move_repeats", enemy.LastMoveRepeats);
        enemy.OnceOnlyMoveUsed = ai.Bool("once_only_move_used", enemy.OnceOnlyMoveUsed);
        enemy.MoveHistory = ai.Items("move_history", AsInt);
        enemy.StartsOnBranch = ai.Bool("starts_on_branch", enemy.StartsOnBranch);
        enemy.RollsNextMove = ai.Bool("rolls_next_move", enemy.RollsNextMove);
        enemy.CurlUpArmed = ai.Bool("curl_up_armed", enemy.CurlUpArmed);
        enemy.LastBranch = ai.Int("last_branch", enemy.LastBranch);
        enemy.RepeatStreak = ai.Int("repeat_streak", enemy.RepeatStreak);
        enemy.BranchCooldown = ai.Int("branch_cooldown", enemy.BranchCooldown);
        enemy.StarterMove = ai.Int("starter_move", enemy.StarterMove);
        enemy.OffBalance = ai.Bool("off_balance", enemy.OffBalance);
        enemy.NemesisIntangibleOn = ai.Bool("nemesis_intangible_on", enemy.NemesisIntangibleOn);
        enemy.Escaped = ai.Bool("escaped", enemy.Escaped);
        int spawned = ai.Int("last_spawned_def_id", enemy.LastSpawnedDefId);
        enemy.LastSpawnedDefId = spawned == 0 ? 0 : EnemyId(spawned, ai.At("last_spawned_def_id"));
        enemy.StolenGold = ai.Int("stolen_gold", enemy.StolenGold);
        enemy.HeistGold = ai.Int("heist_gold", enemy.HeistGold);
        enemy.PoweredHitsThisTurn = ai.Int("powered_hits_this_turn", enemy.PoweredHitsThisTurn);
        enemy.HardenedShellDamageThisTurn = ai.Int(
            "hardened_shell_damage_this_turn",
            enemy.HardenedShellDamageThisTurn
        );
        enemy.StolenCard = ai.NullableCard("stolen_card");
        return enemy;
    }

    private static Intent ReadIntent(Fields fields)
    {
        var intent = new Intent(
            fields.EnumValue<IntentType>("type", nameof(IntentType)),
            fields.Int("magnitude")
        );
        return intent with
        {
            Hits = fields.Int("hits", intent.Hits),
            CarriesDamage = fields.Bool("carries_damage", intent.CarriesDamage),
        };
    }

    private static CardInstance ReadCard(JsonElement element, string path)
    {
        var fields = new Fields(element, path, CardFields);
        var card = new CardInstance(
            CardId(fields.Int("def_id"), fields.At("def_id")),
            fields.Bool("upgraded")
        );
        return card with
        {
            Enchantment = fields.EnumValue("enchantment", card.Enchantment, nameof(Enchantment)),
            EnchantAmount = fields.Int("enchant_amount", card.EnchantAmount),
            CostBump = fields.Int("cost_bump", card.CostBump),
            CostForCombat = fields.IntOrUnset("cost_for_combat"),
            CostThisTurn = fields.IntOrUnset("cost_this_turn"),
            FreeThisTurn = fields.Bool("free_this_turn", card.FreeThisTurn),
            FreeUntilPlayed = fields.Bool("free_until_played", card.FreeUntilPlayed),
            RetainThisTurn = fields.Bool("retain_this_turn", card.RetainThisTurn),
            RetainForCombat = fields.Bool("retain_for_combat", card.RetainForCombat),
            SlyThisTurn = fields.Bool("sly_this_turn", card.SlyThisTurn),
            SlyForCombat = fields.Bool("sly_for_combat", card.SlyForCombat),
            EtherealForCombat = fields.Bool("ethereal_for_combat", card.EtherealForCombat),
            PlayedThisTurn = fields.Bool("played_this_turn", card.PlayedThisTurn),
            Bound = fields.Bool("bound", card.Bound),
            DampenedUpgrade = fields.Bool("dampened_upgrade", card.DampenedUpgrade),
            EnchantSpent = fields.Bool("enchant_spent", card.EnchantSpent),
            ReplayCount = fields.Int("replay_count", card.ReplayCount),
            BonusDamage = fields.Int("bonus_damage", card.BonusDamage),
            BonusBlock = fields.Int("bonus_block", card.BonusBlock),
            TinkerType = fields.EnumValue("tinker_type", card.TinkerType, nameof(CardType)),
            TinkerRider = fields.EnumValue("tinker_rider", card.TinkerRider, nameof(TinkerRider)),
        };
    }

    /// <summary>
    /// A buff list, held to the two rules every list the engine builds obeys: at most one
    /// entry per BuffId (an Instanced power may have one per instance), and none at zero. See
    /// <see cref="DistinctBuffs" /> and <see cref="ReadBuff" />.
    /// </summary>
    private static List<BuffState> ReadBuffs(Fields fields, string name) =>
        DistinctBuffs(fields.Items(name, ReadBuff), fields.At(name));

    /// <summary>
    /// Refuses a list holding one BuffId twice. The engine keeps at most one entry per id
    /// -- `BuffSystem.Apply` merges into the first match and `BuffSystem.Get` reads only
    /// the first -- so a second entry would be played as if absent while the export and
    /// the observation still showed it. Two aliased live powers mapped onto one id is how
    /// a converter would produce one. An Instanced power is the exception: each instance is
    /// an entry (`BuffSystem.IsInstanced`), and the engine walks all of them.
    /// </summary>
    private static List<BuffState> DistinctBuffs(List<BuffState> buffs, string path)
    {
        for (int i = 1; i < buffs.Count; i++)
        {
            for (int first = 0; first < i; first++)
            {
                if (buffs[first].Id == buffs[i].Id && !BuffSystem.IsInstanced(buffs[i].Id))
                {
                    throw new SnapshotException(
                        Field(Item(path, i), "id"),
                        $"duplicate BuffId {(int)buffs[i].Id} (first at [{first}])"
                    );
                }
            }
        }

        return buffs;
    }

    /// <summary>
    /// One buff. A zero amount is refused: `BuffSystem.Apply` and the end-of-turn tick both
    /// remove an entry the moment it reaches zero, so no combat ever holds one. Negative
    /// amounts are real -- Strength below zero, the Shrinker Beetle's permanent Shrink.
    /// </summary>
    private static BuffState ReadBuff(JsonElement element, string path)
    {
        var fields = new Fields(element, path, BuffFields);
        var buff = new BuffState(
            fields.EnumValue<BuffId>("id", nameof(BuffId)),
            fields.Int("amount"),
            fields.Int("data", 0)
        );
        return buff.Magnitude != 0
            ? buff
            : throw new SnapshotException(
                fields.At("amount"),
                "0; the engine never holds a buff at zero"
            );
    }

    private static OrbState ReadOrb(JsonElement element, string path)
    {
        var fields = new Fields(element, path, OrbFields);
        var orb = new OrbState(fields.EnumValue<OrbType>("type", nameof(OrbType)));
        return orb with
        {
            EvokeValue = fields.Int("evoke", orb.EvokeValue),
            PassiveValue = fields.Int("passive", orb.PassiveValue),
        };
    }

    private static RelicInstance ReadRelic(JsonElement element, string path)
    {
        var fields = new Fields(element, path, RelicFields);
        var relic = new RelicInstance(RelicId(fields.Int("id"), fields.At("id")));
        return relic with { Counter = fields.Int("counter", relic.Counter) };
    }

    private static void ReadCounters(Fields counters, CombatState state)
    {
        foreach (var counter in IntCounters)
        {
            counter.Field(state) = counter.NullIsUnset
                ? counters.IntOrUnset(counter.Name)
                : counters.Int(counter.Name, 0);
        }

        foreach (var counter in BoolCounters)
        {
            counter.Field(state) = counters.Bool(counter.Name, counter.WhenOmitted);
        }

        state.LastAttackOrSkillThisTurn = counters.NullableCard(LastAttackOrSkillThisTurn);
        state.LastAttackOrSkillLastTurn = counters.NullableCard(LastAttackOrSkillLastTurn);
        state.StolenBackCards = counters.Items(StolenBackCards, ReadCard);
    }

    private static void ReadPendingSelection(
        Fields root,
        CombatState state,
        List<(CardInstance Card, int At)> unordered
    )
    {
        if (
            !root.TryGet("pending_selection", out var element)
            || element.ValueKind == JsonValueKind.Null
        )
        {
            return;
        }

        var fields = new Fields(element, "pending_selection", SelectionFields);
        var kind = fields.EnumValue<CardSelectionKind>("kind", nameof(CardSelectionKind));
        if (kind == CardSelectionKind.None)
        {
            throw new SnapshotException(
                fields.At("kind"),
                "0 is CardSelectionKind.None; write null when no selection is open"
            );
        }

        state.PendingSelection = new PendingCardSelection
        {
            Kind = kind,
            Candidates = ReadCandidates(fields, kind, state, unordered),
            SourceCardDefId = fields.Int("source_card_def_id"),
            Amount = fields.Int("amount", 0),
            Skippable = fields.Bool("skippable", false),
            GeneratedCandidates = fields.Items(
                "generated_candidates",
                (item, path) => CardId(AsInt(item, path), path)
            ),
            GeneratedUpgraded = fields.Bool("generated_upgraded", false),
            GeneratedRetained = fields.Bool("generated_retained", false),
            AfterSelectionToHand = fields.Items("after_selection_to_hand", ReadCard),
        };
        state.EndTurnAwaitingSelection = fields.Bool("end_turn_awaiting_selection", false);
    }

    /// <summary>
    /// A candidate is an index into the pile (or generated list) the selection draws from.
    /// Every resolution checks the upper bound; a negative one would crash it instead.
    /// </summary>
    private static int ReadCandidate(JsonElement element, string path)
    {
        int index = AsInt(element, path);
        return index >= 0
            ? index
            : throw new SnapshotException(path, $"{index}; a candidate index cannot be negative");
    }

    /// <summary>
    /// The selection's candidates: the indices <c>candidates</c> gives, and for a draw-pile
    /// selection the indices the load's shuffle took its <c>candidate_cards</c> to
    /// (<see cref="FindInUnorderedMiddle" />), all in pile order.
    /// </summary>
    /// <remarks>
    /// An index into that middle names whatever card the load's seed put there, not the one
    /// the document meant, so a draw-pile selection may not give one: its known-top and
    /// known-bottom candidates are indices, in pile order and each once, and the middle's
    /// are cards. Its candidates then have one order whether cards join them or not. Any
    /// other selection indexes a pile the load keeps in order, names nothing by card, and
    /// keeps its candidates in the order given.
    /// </remarks>
    private static List<int> ReadCandidates(
        Fields fields,
        CardSelectionKind kind,
        CombatState state,
        List<(CardInstance Card, int At)> unordered
    )
    {
        var candidates = fields.RequiredItems("candidates", ReadCandidate);
        var cards = fields.Items("candidate_cards", ReadCard);
        if (!kind.ChoosesFromDrawPile())
        {
            return cards.Count == 0
                ? candidates
                : throw new SnapshotException(
                    fields.At("candidate_cards"),
                    $"kind {(int)kind} does not choose from the draw pile; only a draw-pile "
                        + "selection names its candidates by card"
                );
        }

        var (start, end) = UnorderedMiddle(state);
        for (int i = 0; i < candidates.Count; i++)
        {
            string path = Item(fields.At("candidates"), i);
            if (i > 0 && candidates[i] <= candidates[i - 1])
            {
                throw new SnapshotException(
                    path,
                    $"{candidates[i]} is not after {candidates[i - 1]}; a draw-pile selection "
                        + "lists its candidates in pile order, each once"
                );
            }

            if (candidates[i] >= start && candidates[i] < end)
            {
                throw new SnapshotException(
                    path,
                    $"{candidates[i]} is in piles.draw.unordered, whose order the load "
                        + "shuffles; name that candidate in candidate_cards"
                );
            }
        }

        if (cards.Count == 0)
        {
            return candidates;
        }

        var found = FindInUnorderedMiddle(cards, unordered, fields.At("candidate_cards"));
        return [.. candidates.Concat(found).Order()];
    }

    /// <summary>
    /// The draw-pile index each of <paramref name="cards" /> ended up at: matched in the
    /// document's own <c>unordered</c> list, then read off where the load's shuffle took the
    /// card it matched (<see cref="ReadPiles" />). Nothing is moved.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A card takes the first free card of the list, in the document's order, that matches
    /// it field for field. Only then does a card nothing matches take the first free card
    /// with its def_id and upgrade, and only if every such card is alike: otherwise which
    /// card the selection offers would be a guess, so the document is refused instead. A
    /// card with no free match is refused too. The known ends are not searched, because a
    /// card there has an index the document gives.
    /// </para>
    /// <para>
    /// Matching in the list, before the shuffle, is what keeps the loaded order fair. Copies
    /// alike in every field make the same state whichever of them is named, but the named
    /// copy is where the pick will leave its gap. Found in the shuffled pile instead, it was
    /// always the topmost copy, so the draws after the pick leaned away from its kind: with
    /// four Strikes among nine cards and one taken, the next card was a Strike 1/6 of the
    /// time instead of 3/8. Matched in the list, it is one fixed card, and the shuffle puts
    /// it anywhere as readily as any other.
    /// </para>
    /// </remarks>
    private static List<int> FindInUnorderedMiddle(
        List<CardInstance> cards,
        List<(CardInstance Card, int At)> unordered,
        string path
    )
    {
        var taken = new bool[unordered.Count];
        var found = new int[cards.Count];
        Array.Fill(found, -1);

        for (int i = 0; i < cards.Count; i++)
        {
            for (int listed = 0; listed < unordered.Count; listed++)
            {
                if (!taken[listed] && unordered[listed].Card == cards[i])
                {
                    found[i] = unordered[listed].At;
                    taken[listed] = true;
                    break;
                }
            }
        }

        for (int i = 0; i < cards.Count; i++)
        {
            if (found[i] >= 0)
            {
                continue;
            }

            var card = cards[i];
            var alike = Enumerable
                .Range(0, unordered.Count)
                .Where(listed =>
                    !taken[listed]
                    && unordered[listed].Card.DefId == card.DefId
                    && unordered[listed].Card.Upgraded == card.Upgraded
                )
                .ToList();
            if (alike.Count == 0)
            {
                throw new SnapshotException(
                    Item(path, i),
                    $"no card left in piles.draw.unordered matches it (def_id {card.DefId}, "
                        + $"upgraded {(card.Upgraded ? "true" : "false")})"
                );
            }

            if (alike.Select(listed => unordered[listed].Card).Distinct().Count() > 1)
            {
                throw new SnapshotException(
                    Item(path, i),
                    $"matches {alike.Count} cards of piles.draw.unordered by def_id and upgraded "
                        + "that differ in their other fields, and none field for field"
                );
            }

            found[i] = unordered[alike[0]].At;
            taken[alike[0]] = true;
        }

        return [.. found];
    }

    /// <summary>
    /// Where the unordered middle of the draw pile sits: past the known top and short of
    /// the known bottom, each held inside the pile, as [Start, End).
    /// </summary>
    private static (int Start, int End) UnorderedMiddle(CombatState state)
    {
        int count = state.DrawPile.Count;
        int top = Math.Clamp(state.KnownTopCount, 0, count);
        int bottom = Math.Clamp(state.KnownBottomCount, 0, count - top);
        return (top, count - bottom);
    }

    // ── ids ──────────────────────────────────────────────────────────────────────────

    private static int CardId(int id, string path)
    {
        try
        {
            GeneratedData.Cards.Get(id);
            return id;
        }
        catch (ArgumentException)
        {
            throw new SnapshotException(path, $"unknown card id {id}");
        }
    }

    private static int RelicId(int id, string path)
    {
        try
        {
            GeneratedData.Relics.Get(id);
            return id;
        }
        catch (ArgumentException)
        {
            throw new SnapshotException(path, $"unknown relic id {id}");
        }
    }

    private static int PotionId(int id, string path)
    {
        try
        {
            GeneratedData.Potions.Get(id);
            return id;
        }
        catch (ArgumentException)
        {
            throw new SnapshotException(path, $"unknown potion id {id}");
        }
    }

    private static int EnemyId(int id, string path)
    {
        try
        {
            GeneratedData.Enemies.Get(id);
            return id;
        }
        catch (ArgumentException)
        {
            throw new SnapshotException(path, $"unknown enemy id {id}");
        }
    }

    /// <summary>An int as one of the emulator's (int-backed) enums, refusing a value it does not declare.</summary>
    private static TEnum ToEnum<TEnum>(int raw, string path, string typeName)
        where TEnum : struct, Enum
    {
        var value = Unsafe.As<int, TEnum>(ref raw);
        return Enum.IsDefined(value)
            ? value
            : throw new SnapshotException(path, $"unknown {typeName} {raw}");
    }

    // ── JSON values ──────────────────────────────────────────────────────────────────

    private static string Field(string path, string name) =>
        path.Length == 0 ? name : $"{path}.{name}";

    private static string Item(string path, int index) => $"{path}[{index}]";

    /// <summary>The root object has no name of its own, so errors about it say "$".</summary>
    private static string Self(string path) => path.Length == 0 ? "$" : path;

    private static string Describe(JsonElement value) =>
        value.ValueKind switch
        {
            JsonValueKind.Object => "an object",
            JsonValueKind.Array => "an array",
            JsonValueKind.String => "a string",
            JsonValueKind.Number => "a number",
            JsonValueKind.True or JsonValueKind.False => "a boolean",
            JsonValueKind.Null => "null",
            _ => "nothing",
        };

    private static int AsInt(JsonElement value, string path)
    {
        if (value.ValueKind != JsonValueKind.Number)
        {
            throw new SnapshotException(path, $"expected an integer, got {Describe(value)}");
        }

        return value.TryGetInt32(out int result)
            ? result
            : throw new SnapshotException(
                path,
                $"expected a 32-bit integer, got {value.GetRawText()}"
            );
    }

    private static bool AsBool(JsonElement value, string path) =>
        value.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => throw new SnapshotException(
                path,
                $"expected true or false, got {Describe(value)}"
            ),
        };

    private static void RequireArray(JsonElement value, string path)
    {
        if (value.ValueKind != JsonValueKind.Array)
        {
            throw new SnapshotException(path, $"expected an array, got {Describe(value)}");
        }
    }

    private static List<T> ReadItems<T>(
        JsonElement array,
        string path,
        Func<JsonElement, string, T> read,
        int max,
        string noun
    )
    {
        RequireArray(array, path);
        int count = array.GetArrayLength();
        if (count > max)
        {
            throw new SnapshotException(path, $"{count} {noun}; at most {max}");
        }

        var items = new List<T>(count);
        int index = 0;
        foreach (var item in array.EnumerateArray())
        {
            items.Add(read(item, Item(path, index++)));
        }

        return items;
    }

    /// <summary>
    /// One object of the document with the path that names it in an error. Refuses any
    /// field it does not know: a misspelt <c>"retain_this_trun": true</c> would otherwise be
    /// a value the loader dropped without a word, which is the same failure as dropping an
    /// unknown id.
    /// </summary>
    private readonly struct Fields
    {
        private readonly JsonElement _object;

        public Fields(JsonElement element, string path, string[] known)
        {
            if (element.ValueKind != JsonValueKind.Object)
            {
                throw new SnapshotException(
                    Self(path),
                    $"expected an object, got {Describe(element)}"
                );
            }

            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in element.EnumerateObject())
            {
                if (Array.IndexOf(known, property.Name) < 0)
                {
                    throw new SnapshotException(Field(path, property.Name), "unknown field");
                }

                if (!seen.Add(property.Name))
                {
                    throw new SnapshotException(Field(path, property.Name), "duplicate field");
                }
            }

            _object = element;
            Path = path;
        }

        /// <summary>An object the document left out: every field in it reads as omitted.</summary>
        private Fields(string path)
        {
            _object = default;
            Path = path;
        }

        public string Path { get; }

        public string At(string name) => Field(Path, name);

        public bool TryGet(string name, out JsonElement value)
        {
            if (
                _object.ValueKind == JsonValueKind.Object
                && _object.TryGetProperty(name, out value)
            )
            {
                return true;
            }

            value = default;
            return false;
        }

        private JsonElement Require(string name) =>
            TryGet(name, out var value)
                ? value
                : throw new SnapshotException(At(name), "missing required field");

        public int Int(string name) => AsInt(Require(name), At(name));

        public int Int(string name, int whenOmitted) =>
            TryGet(name, out var value) ? AsInt(value, At(name)) : whenOmitted;

        /// <summary>A cost override: null (or omitted) is unset, which the engine holds as int.MinValue.</summary>
        public int IntOrUnset(string name)
        {
            if (!TryGet(name, out var value) || value.ValueKind == JsonValueKind.Null)
            {
                return Unset;
            }

            int result = AsInt(value, At(name));
            return result != Unset
                ? result
                : throw new SnapshotException(At(name), $"{Unset} is the unset marker; write null");
        }

        public bool Bool(string name) => AsBool(Require(name), At(name));

        public bool Bool(string name, bool whenOmitted) =>
            TryGet(name, out var value) ? AsBool(value, At(name)) : whenOmitted;

        public string Text(string name)
        {
            var value = Require(name);
            return value.ValueKind == JsonValueKind.String
                ? value.GetString()!
                : throw new SnapshotException(
                    At(name),
                    $"expected a string, got {Describe(value)}"
                );
        }

        public TEnum EnumValue<TEnum>(string name, string typeName)
            where TEnum : struct, Enum => ToEnum<TEnum>(Int(name), At(name), typeName);

        public TEnum EnumValue<TEnum>(string name, TEnum whenOmitted, string typeName)
            where TEnum : struct, Enum =>
            TryGet(name, out var value)
                ? ToEnum<TEnum>(AsInt(value, At(name)), At(name), typeName)
                : whenOmitted;

        public Fields Child(string name, string[] known) => new(Require(name), At(name), known);

        public Fields OptionalChild(string name, string[] known) =>
            TryGet(name, out var value) ? new Fields(value, At(name), known) : new Fields(At(name));

        public List<T> Items<T>(
            string name,
            Func<JsonElement, string, T> read,
            int max = int.MaxValue,
            string noun = "entries"
        ) => TryGet(name, out var value) ? ReadItems(value, At(name), read, max, noun) : [];

        public List<T> RequiredItems<T>(
            string name,
            Func<JsonElement, string, T> read,
            int max = int.MaxValue,
            string noun = "entries"
        ) => ReadItems(Require(name), At(name), read, max, noun);

        public List<T>? NullableItems<T>(string name, Func<JsonElement, string, T> read) =>
            TryGet(name, out var value) && value.ValueKind != JsonValueKind.Null
                ? ReadItems(value, At(name), read, int.MaxValue, "entries")
                : null;

        public CardInstance? NullableCard(string name) =>
            TryGet(name, out var value) && value.ValueKind != JsonValueKind.Null
                ? ReadCard(value, At(name))
                : null;
    }

    // ── writing ──────────────────────────────────────────────────────────────────────

    private static void WriteCombat(Utf8JsonWriter w, CombatState state, bool save)
    {
        w.WriteStartObject();
        w.WriteString("schema", SchemaId);
        w.WriteNumber("ascension", state.AscensionLevel);
        if (state.Character != PlayerCharacter.Ironclad)
        {
            w.WriteNumber("character", (int)state.Character);
        }

        if (state.EncounterId >= 0)
        {
            w.WriteNumber("encounter_id", state.EncounterId);
        }
        else
        {
            w.WriteNull("encounter_id");
        }

        w.WriteBoolean("is_elite_combat", state.IsEliteCombat);
        w.WriteBoolean("is_elite_room", state.IsEliteRoom);
        w.WriteBoolean("fur_coat_bites", state.FurCoatBites);
        w.WriteNumber("turn", state.Turn);

        w.WriteStartObject("player");
        w.WriteNumber("hp", state.PlayerHp);
        w.WriteNumber("max_hp", state.PlayerMaxHp);
        w.WriteNumber("block", state.PlayerBlock);
        w.WriteNumber("energy", state.Energy);
        w.WriteNumber("max_energy", state.MaxEnergy);
        w.WriteNumber("gold", state.PlayerGold);
        w.WriteNumber("stars", state.Stars);
        w.WriteNumber("osty_hp", state.OstyHp);
        w.WriteNumber("osty_max_hp", state.OstyMaxHp);
        w.WriteNumber("orb_capacity", state.OrbCapacity);
        w.WriteNumber("base_orb_slots", state.BaseOrbSlots);
        WriteItems(w, "orbs", state.Orbs, WriteOrb);
        WriteItems(w, "buffs", state.PlayerBuffs, WriteBuff);
        WriteItems(w, "debuffs_at_round_start", state.PlayerDebuffsAtRoundStart, WriteBuff);
        w.WriteEndObject();

        WriteItems(w, "relics", state.Relics, WriteRelic);

        w.WriteStartObject("potions");
        w.WriteNumber("max_slots", state.MaxPotionSlots);
        WriteItems(w, "slots", state.PotionSlots, (writer, id) => writer.WriteNumberValue(id));
        w.WriteEndObject();

        w.WriteStartObject("piles");
        WriteItems(w, "hand", state.Hand, WriteCard);
        // A card held out of its result pile while its selection screen is open belongs in
        // that pile in a document, which has no play pile: written there, it is not lost.
        WriteItems(
            w,
            "discard",
            state.HeldPlayedCard is { } toDiscard && !state.HeldPlayedCardExhausts
                ? state.DiscardPile.Append(toDiscard)
                : state.DiscardPile,
            WriteCard
        );
        WriteItems(
            w,
            "exhaust",
            state.HeldPlayedCard is { } toExhaust && state.HeldPlayedCardExhausts
                ? state.ExhaustPile.Append(toExhaust)
                : state.ExhaustPile,
            WriteCard
        );
        WriteDrawPile(w, state);
        w.WriteEndObject();

        w.WriteStartObject("incoming");
        WriteItems(w, "return_to_hand", state.ReturnToHandBeforeDraw, WriteCard);
        WriteItems(w, "copies_to_hand", state.CopiesToHandBeforeDraw, WriteCard);
        w.WriteEndObject();

        WriteItems(w, "enemies", state.Enemies, WriteEnemy);
        WriteCounters(w, state);
        WritePendingSelection(w, state);
        WriteStoppedStep(w, state, save);

        w.WriteEndObject();
    }

    /// <summary>
    /// A screen raised in the middle of a step: the step it stopped (<see cref="HookChoiceReplay" />),
    /// so that a loaded copy's answer runs the rest of it -- the card goes on to its pile and its
    /// after-play hooks run, the hand is drawn after a turn-start pick, the turn ends after Well-Laid
    /// Plans' -- as answering in memory does. Omitted when no step is stopped.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every document writes the public part (<see cref="Write" />): the kind of step
    /// (<c>step</c>: card_play, potion, end_turn, turn_start, screen), the card in play (<c>card</c>,
    /// in the play pile: in none of the document's piles) or the potion, the action and target, the
    /// answers given so far (a pick from the draw pile as <c>{"card": ...}</c>, since its index
    /// points into the true order), the cards the step took off the draw pile before the screen in
    /// the order it took them (<c>taken</c>), each screen it reached (<c>screens</c>: kind, source
    /// card, candidate count, the takes before it, the candidates it showed by card -- a draw-pile
    /// screen's as a sorted multiset, since their places are the true order -- or by generated id),
    /// and the step's start (<c>before</c>) as the player saw it: the
    /// position the action was taken from, written as this document is written, its unseen cards a
    /// sorted multiset and no stream. That start is history the player has seen; the replay of a
    /// loaded one runs in the loader's world and reaches the screen's place in the step, where the
    /// loaded copy is put back whole (<see cref="ScreenHiddenState" />). Two determinized copies of
    /// one stopped combat write the same block, and so do two combats that look the same and started
    /// the step from positions that look the same.
    /// </para>
    /// <para>
    /// The save document (<see cref="WriteSave" />) adds the true world: the start's draw order
    /// (<c>before_draw</c>) and streams (<c>before_streams</c>), each answer's true index
    /// (<c>answer_indices</c>) and the hidden state of the copy that gave it (<c>hidden</c>). A reader
    /// of those reads the future, so they are banned outside verification and save code. A step read
    /// from the agent's document has no true world, and its save is the public part alone.
    /// </para>
    /// </remarks>
    private static void WriteStoppedStep(Utf8JsonWriter w, CombatState state, bool save)
    {
        if (state.PendingSelection?.Replay is not { } stopped)
        {
            return;
        }

        var (step, card, potion) = InPlay(stopped, state.Turn);
        w.WriteStartObject("stopped_step");
        w.WriteString("step", step);
        if (card is { } played)
        {
            w.WritePropertyName("card");
            WriteCard(w, played);
        }
        else
        {
            w.WriteNull("card");
        }

        if (potion is { } drunk)
        {
            w.WriteNumber("potion", drunk);
        }
        else
        {
            w.WriteNull("potion");
        }

        w.WriteNumber("action", stopped.Action);
        w.WriteNumber("target", stopped.TargetEnemyIndex);
        w.WriteStartArray("answers");
        foreach (var answer in stopped.Answers)
        {
            if (answer.Pick is { } pick)
            {
                w.WriteStartObject();
                w.WritePropertyName("card");
                WriteCard(w, pick);
                w.WriteEndObject();
            }
            else
            {
                w.WriteNumberValue(answer.Index);
            }
        }

        w.WriteEndArray();
        WriteItems(w, "taken", stopped.Taken, WriteCard);
        WriteItems(
            w,
            "screens",
            stopped.Screens,
            (writer, reached) =>
            {
                writer.WriteStartObject();
                writer.WriteNumber("kind", (int)reached.Kind);
                writer.WriteNumber("source", reached.Source);
                writer.WriteNumber("count", reached.Count);
                writer.WriteNumber("taken", reached.Taken);
                // A screen over the draw pile offered cards, not places: in pile order they would
                // name the true order (Foregone Conclusion's whole pile), so they are the multiset
                // the pile's middle is. A hand's or a discard pile's order is the player's to see.
                WriteItems<CardInstance>(writer, "cards", reached.Kind.ChoosesFromDrawPile() ? Sorted(reached.Cards) : reached.Cards, WriteCard);
                WriteItems(writer, "generated", reached.Generated, (inner, id) => inner.WriteNumberValue(id));
                writer.WriteEndObject();
            }
        );
        WriteItems(w, "peeks", stopped.Peeks, (writer, played) => writer.WriteBooleanValue(played));
        w.WritePropertyName("before");
        WriteCombat(w, stopped.Before, save: false);
        if (save && !stopped.FromDocument)
        {
            WriteItems(w, "answer_indices", stopped.Answers, (writer, answer) => writer.WriteNumberValue(answer.Index));
            WriteItems(w, "before_draw", stopped.Before.DrawPile, WriteCard);
            w.WritePropertyName("before_streams");
            WriteStreams(
                w,
                [.. Enumerable.Range(0, StateCloning.StreamCount).Select(i => StateCloning.Stream(stopped.Before, i))],
                stopped.RngBefore
            );
            WriteItems(
                w,
                "hidden",
                stopped.Hidden,
                (writer, kept) =>
                {
                    writer.WriteStartObject();
                    writer.WritePropertyName("streams");
                    WriteStreams(writer, kept.Streams, kept.StepRng);
                    WriteItems(writer, "draw", kept.DrawPile, WriteCard);
                    if (kept.Candidates is { } candidates)
                    {
                        WriteItems(writer, "candidates", candidates, (inner, index) => inner.WriteNumberValue(index));
                    }
                    else
                    {
                        writer.WriteNull("candidates");
                    }

                    writer.WriteEndObject();
                }
            );
        }

        w.WriteEndObject();
    }

    private static void WriteStreams(Utf8JsonWriter w, CountingRandom?[] streams, CountingRandom? step)
    {
        w.WriteStartObject();
        for (int i = 0; i < StreamNames.Length; i++)
        {
            var rng = i < streams.Length ? streams[i] : step;
            if (rng is null)
            {
                w.WriteNull(StreamNames[i]);
                continue;
            }

            var (s0, s1, s2, s3) = rng.Lanes;
            w.WriteStartObject(StreamNames[i]);
            w.WriteNumber("seed", rng.Seed);
            w.WriteNumber("count", rng.CallCount);
            w.WriteString("lanes", $"{s0:x16}{s1:x16}{s2:x16}{s3:x16}");
            w.WriteEndObject();
        }

        w.WriteEndObject();
    }

    /// <summary>
    /// The draw pile in three parts: the known top and bottom in pile order, and the middle
    /// the player knows only as a multiset, <see cref="Sorted" />.
    /// </summary>
    private static void WriteDrawPile(Utf8JsonWriter w, CombatState state)
    {
        var pile = state.DrawPile;
        var (start, end) = UnorderedMiddle(state);

        w.WriteStartObject("draw");
        WriteItems(w, "known_top", pile.Take(start), WriteCard);
        WriteItems(w, "known_bottom", pile.Skip(end), WriteCard);
        WriteItems(w, "unordered", Sorted(pile.GetRange(start, end - start)), WriteCard);
        w.WriteEndObject();
    }

    /// <summary>
    /// Cards the document holds as a multiset, SORTED -- by def id, then upgraded, then by
    /// every other field the card carries -- so two states holding the same cards in
    /// different orders write the same document.
    /// </summary>
    private static List<CardInstance> Sorted(IEnumerable<CardInstance> cards)
    {
        var keyed = cards.Select(card => (Card: card, Json: CardJson(card))).ToList();
        keyed.Sort(
            static (left, right) =>
            {
                int order = left.Card.DefId.CompareTo(right.Card.DefId);
                if (order == 0)
                {
                    order = left.Card.Upgraded.CompareTo(right.Card.Upgraded);
                }

                return order != 0 ? order : left.Json.AsSpan().SequenceCompareTo(right.Json);
            }
        );
        return [.. keyed.Select(entry => entry.Card)];
    }

    /// <summary>
    /// A selection's candidates as the document holds them. A draw-pile selection's
    /// candidates in the unordered middle are cards, <see cref="Sorted" /> as that middle
    /// is, because the document gives the middle no order an index could point into; every
    /// other candidate is an index, in the order the selection keeps it.
    /// </summary>
    private static (List<int> Indices, List<CardInstance> Cards) DocumentCandidates(
        CombatState state,
        PendingCardSelection selection
    )
    {
        if (!selection.Kind.ChoosesFromDrawPile())
        {
            return (selection.Candidates, []);
        }

        var (start, end) = UnorderedMiddle(state);
        bool InMiddle(int index) => index >= start && index < end;
        return (
            [.. selection.Candidates.Where(index => !InMiddle(index))],
            Sorted(selection.Candidates.Where(InMiddle).Select(index => state.DrawPile[index]))
        );
    }

    private static byte[] CardJson(CardInstance card)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            WriteCard(writer, card);
        }

        return buffer.WrittenSpan.ToArray();
    }

    private static void WriteEnemy(Utf8JsonWriter w, EnemyState enemy)
    {
        w.WriteStartObject();
        w.WriteNumber("def_id", enemy.DefId);
        w.WriteNumber("slot", enemy.Slot);
        w.WriteNumber("hp", enemy.Hp);
        w.WriteNumber("max_hp", enemy.MaxHp);
        w.WriteNumber("block", enemy.Block);
        WriteItems(w, "buffs", enemy.Buffs, WriteBuff);

        w.WriteStartObject("intent");
        w.WriteNumber("type", (int)enemy.CurrentIntent.Type);
        w.WriteNumber("magnitude", enemy.CurrentIntent.Magnitude);
        w.WriteNumber("hits", enemy.CurrentIntent.Hits);
        w.WriteBoolean("carries_damage", enemy.CurrentIntent.CarriesDamage);
        w.WriteEndObject();

        w.WriteStartObject("ai");
        w.WriteNumber("move_index", enemy.MoveIndex);
        w.WriteNumber("last_move", enemy.LastMove);
        w.WriteNumber("last_move_repeats", enemy.LastMoveRepeats);
        w.WriteBoolean("once_only_move_used", enemy.OnceOnlyMoveUsed);
        WriteItems(
            w,
            "move_history",
            enemy.MoveHistory,
            (writer, move) => writer.WriteNumberValue(move)
        );
        w.WriteBoolean("starts_on_branch", enemy.StartsOnBranch);
        w.WriteBoolean("rolls_next_move", enemy.RollsNextMove);
        w.WriteBoolean("curl_up_armed", enemy.CurlUpArmed);
        w.WriteNumber("last_branch", enemy.LastBranch);
        w.WriteNumber("repeat_streak", enemy.RepeatStreak);
        w.WriteNumber("branch_cooldown", enemy.BranchCooldown);
        w.WriteNumber("starter_move", enemy.StarterMove);
        w.WriteBoolean("off_balance", enemy.OffBalance);
        w.WriteBoolean("nemesis_intangible_on", enemy.NemesisIntangibleOn);
        w.WriteBoolean("escaped", enemy.Escaped);
        w.WriteNumber("last_spawned_def_id", enemy.LastSpawnedDefId);
        w.WriteNumber("stolen_gold", enemy.StolenGold);
        w.WriteNumber("heist_gold", enemy.HeistGold);
        w.WriteNumber("powered_hits_this_turn", enemy.PoweredHitsThisTurn);
        w.WriteNumber("hardened_shell_damage_this_turn", enemy.HardenedShellDamageThisTurn);
        WriteNullableCard(w, "stolen_card", enemy.StolenCard);
        w.WriteEndObject();

        w.WriteEndObject();
    }

    private static void WriteCounters(Utf8JsonWriter w, CombatState state)
    {
        w.WriteStartObject("counters");
        foreach (var counter in IntCounters)
        {
            int value = counter.Field(state);
            if (counter.NullIsUnset && value == Unset)
            {
                w.WriteNull(counter.Name);
            }
            else
            {
                w.WriteNumber(counter.Name, value);
            }
        }

        foreach (var counter in BoolCounters)
        {
            w.WriteBoolean(counter.Name, counter.Field(state));
        }

        WriteNullableCard(w, LastAttackOrSkillThisTurn, state.LastAttackOrSkillThisTurn);
        WriteNullableCard(w, LastAttackOrSkillLastTurn, state.LastAttackOrSkillLastTurn);
        WriteItems(w, StolenBackCards, state.StolenBackCards, WriteCard);
        w.WriteEndObject();
    }

    private static void WritePendingSelection(Utf8JsonWriter w, CombatState state)
    {
        if (state.PendingSelection is not { } selection)
        {
            w.WriteNull("pending_selection");
            return;
        }

        w.WriteStartObject("pending_selection");
        w.WriteNumber("kind", (int)selection.Kind);
        var (indices, cards) = DocumentCandidates(state, selection);
        WriteItems(w, "candidates", indices, (writer, index) => writer.WriteNumberValue(index));
        WriteItems(w, "candidate_cards", cards, WriteCard);
        w.WriteNumber("source_card_def_id", selection.SourceCardDefId);
        w.WriteNumber("amount", selection.Amount);
        w.WriteBoolean("skippable", selection.Skippable);
        WriteItems(
            w,
            "generated_candidates",
            selection.GeneratedCandidates,
            (writer, id) => writer.WriteNumberValue(id)
        );
        w.WriteBoolean("generated_upgraded", selection.GeneratedUpgraded);
        w.WriteBoolean("generated_retained", selection.GeneratedRetained);
        WriteItems(w, "after_selection_to_hand", selection.AfterSelectionToHand, WriteCard);
        w.WriteBoolean("end_turn_awaiting_selection", state.EndTurnAwaitingSelection);
        w.WriteEndObject();
    }

    private static void WriteCard(Utf8JsonWriter w, CardInstance card)
    {
        w.WriteStartObject();
        w.WriteNumber("def_id", card.DefId);
        w.WriteBoolean("upgraded", card.Upgraded);
        w.WriteNumber("enchantment", (int)card.Enchantment);
        w.WriteNumber("enchant_amount", card.EnchantAmount);
        w.WriteNumber("cost_bump", card.CostBump);
        WriteIntOrUnset(w, "cost_for_combat", card.CostForCombat);
        WriteIntOrUnset(w, "cost_this_turn", card.CostThisTurn);
        w.WriteBoolean("free_this_turn", card.FreeThisTurn);
        w.WriteBoolean("free_until_played", card.FreeUntilPlayed);
        w.WriteBoolean("retain_this_turn", card.RetainThisTurn);
        w.WriteBoolean("retain_for_combat", card.RetainForCombat);
        w.WriteBoolean("sly_this_turn", card.SlyThisTurn);
        w.WriteBoolean("sly_for_combat", card.SlyForCombat);
        w.WriteBoolean("ethereal_for_combat", card.EtherealForCombat);
        w.WriteBoolean("played_this_turn", card.PlayedThisTurn);
        w.WriteBoolean("bound", card.Bound);
        w.WriteBoolean("dampened_upgrade", card.DampenedUpgrade);
        w.WriteBoolean("enchant_spent", card.EnchantSpent);
        w.WriteNumber("replay_count", card.ReplayCount);
        w.WriteNumber("bonus_damage", card.BonusDamage);
        w.WriteNumber("bonus_block", card.BonusBlock);
        w.WriteNumber("tinker_type", (int)card.TinkerType);
        w.WriteNumber("tinker_rider", (int)card.TinkerRider);
        w.WriteEndObject();
    }

    private static void WriteBuff(Utf8JsonWriter w, BuffState buff)
    {
        w.WriteStartObject();
        w.WriteNumber("id", (int)buff.Id);
        w.WriteNumber("amount", buff.Magnitude);
        if (buff.Data != 0)
        {
            w.WriteNumber("data", buff.Data);
        }

        w.WriteEndObject();
    }

    private static void WriteOrb(Utf8JsonWriter w, OrbState orb)
    {
        w.WriteStartObject();
        w.WriteNumber("type", (int)orb.Type);
        w.WriteNumber("evoke", orb.EvokeValue);
        w.WriteNumber("passive", orb.PassiveValue);
        w.WriteEndObject();
    }

    private static void WriteRelic(Utf8JsonWriter w, RelicInstance relic)
    {
        w.WriteStartObject();
        w.WriteNumber("id", relic.DefId);
        w.WriteNumber("counter", relic.Counter);
        w.WriteEndObject();
    }

    private static void WriteIntOrUnset(Utf8JsonWriter w, string name, int value)
    {
        if (value == Unset)
        {
            w.WriteNull(name);
        }
        else
        {
            w.WriteNumber(name, value);
        }
    }

    private static void WriteNullableCard(Utf8JsonWriter w, string name, CardInstance? card)
    {
        if (card is { } value)
        {
            w.WritePropertyName(name);
            WriteCard(w, value);
        }
        else
        {
            w.WriteNull(name);
        }
    }

    private static void WriteItems<T>(
        Utf8JsonWriter w,
        string name,
        IEnumerable<T> items,
        Action<Utf8JsonWriter, T> write
    )
    {
        w.WriteStartArray(name);
        foreach (var item in items)
        {
            write(w, item);
        }

        w.WriteEndArray();
    }
}
