// What the old engine's state cannot represent of the game's listener lists (docs/rebuild/TIER1_KERNEL.md section 3).
// OldState.cs names a gap where it meets one; Listeners.cs reports it for a pass only when the pass has a handler that
// could stand for the missing listener. The kernel never guesses a listener the old state does not hold.
namespace Sts2Emulator.Core2;

/// <summary>Why a listener of the game's list has no faithful entry in the old state.</summary>
public enum NotModelledKind : byte
{
    /// <summary>
    /// A BuffId that holds several game powers summed (tools/fidelity/power_aliases.json "folds": every
    /// TemporaryStrengthPower subclass is one TemporaryStrength buff, SelfFormingClayPower joins BlockNextTurn). The
    /// game has one power instance per application, each a listener in application order; the old state has one
    /// number, so neither the instances, their models nor their order are known. Left out of the pass.
    /// </summary>
    FoldedPower,

    /// <summary>Osty is in the combat, but the old state holds no powers for it (only its HP).</summary>
    OstyPowers,

    /// <summary>
    /// An affliction other than Bound: the old state holds Bound per card (<c>CardInstance.Bound</c>) and no other
    /// affliction, so a card's Entangled, Galvanized, Hexed, Ringing, Smog or Tainted listener is unknown.
    /// </summary>
    Affliction,

    /// <summary>
    /// Cards queued to auto-play (<c>CombatState.AutoPlayQueue</c>): live cards whose pile, and so whose place in the
    /// game's list, the old state does not hold. Left out of the pass (still members by uid).
    /// </summary>
    AutoPlayQueue,

    /// <summary>A run+combat or run pass over a combat with no run state: the run deck's cards are unknown.</summary>
    DeckWithoutRun,

    /// <summary>
    /// The played card held while its screen is open (<c>HeldPlayedCard</c>) while a card of equal value already ends
    /// its result pile. Either the held card is stale (filed by a snapshot that never restores the field) or it is a
    /// twin of the filed one; the old state cannot tell, so the Play pile is left empty.
    /// </summary>
    HeldCardFiled,
}

/// <summary>A listener the old state cannot represent, as OldState meets it.</summary>
/// <param name="Why">The kind of gap.</param>
/// <param name="Kind">The kind of listener missing.</param>
/// <param name="Creature">The creature it would sit on (game order), or -1.</param>
/// <param name="ModelId">The listener's model when known (then reported only when that model has a handler), else 0
/// (reported when any listener of <paramref name="Kind" /> has one).</param>
/// <param name="What">A short note for logs (identifiers only).</param>
public readonly record struct OldGap(NotModelledKind Why, ListenerKind Kind, int Creature, int ModelId, string What);

/// <summary>One gap reported for one pass: the pass's serial (per <see cref="OldListeners" />), its hook virtual and the
/// gap.</summary>
public readonly record struct NotModelled(int Pass, HookId Hook, OldGap Gap);

/// <summary>
/// Represented despite a different shape (no NotModelled):
/// <list type="bullet">
/// <item>Melted relics: the old engine removes a melted wax relic from the list; the game keeps it and skips it in
/// every iterator (<c>!IsMelted</c>), so the order of the rest is the same.</item>
/// <item>The Play pile: the old engine has none. The card in flight is <see cref="OldWorld.InFlight" /> (kernel play
/// sites set it for the length of a play) or, while its screen is open, <c>HeldPlayedCard</c>.</item>
/// <item>A game power the old engine keeps in several BuffIds (InfernoSelfDamage beside Inferno): visited once, at its
/// first entry.</item>
/// <item>A creature dead but not yet removed (the window inside <c>CreatureCmd.Kill</c>): the caller marks it in
/// <see cref="OldWorld.Dying" />; otherwise a dead enemy that would be removed is out of the combat.</item>
/// <item>Modifiers, badges and the multiplayer scaling model: the old state has none; the caller supplies them on
/// <see cref="OldWorld" /> (none in a standard single-player run).</item>
/// </list>
/// </summary>
public static class NotModelledNotes;
