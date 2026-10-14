namespace Sts2Emulator.Core;

/// <summary>
/// The profile a run is generated for: the game's <c>UnlockState</c> (the epochs revealed and
/// the encounters seen) and the one profile fact generation reads beside it,
/// <c>ProgressState.DiscoveredActs</c>. A run's seed does not decide these; the player's save
/// does, and a run history does not record it.
/// </summary>
/// <remarks>
/// <para>
/// Held as what the profile LACKS, so <see cref="All" /> -- the game's <c>UnlockState.all</c>:
/// every epoch, every encounter, every act discovered -- is the empty one, and is what the
/// emulator generates for unless told otherwise. Every site below reads it the way the game
/// reads its own, with the same draws, so a different profile moves a stream only where the
/// game's does.
/// </para>
/// <para>
/// Read by: <c>ActModel.GetRandomList</c> (an alternate act is drawn only if unlocked, and
/// taken without a draw while undiscovered), <c>RunManager.GenerateRooms</c>
/// (<c>SharedAncients</c>), <c>ActModel.GenerateRooms</c> (the event epochs,
/// <c>GetUnlockedAncients</c>), <c>ActModel.ApplyDiscoveryOrderModifications</c> (the first
/// unseen boss of the act's <c>BossDiscoveryOrder</c> replaces the one drawn),
/// <c>RelicModel.IsAllowedAtNeow</c> (Kaleidoscope), and every walk over
/// <c>UnlockState.Characters</c> / <c>CharacterCardPools</c> (Kaleidoscope's rewards, Splash,
/// Prismatic Gem, Orobas's Sea Glass, Colorful Philosophers).
/// </para>
/// <para>
/// Not modelled, so refused: an epoch that only filters content -- <c>FilterThroughEpochs</c>
/// over the card pools, <c>GetUnlockedRelics</c>, <c>GetUnlockedPotions</c> -- and Neow's own
/// epoch (a profile without it starts its runs without Neow). Nor is
/// <c>UnlockState.NumberOfRuns</c>: its one value that changes generation, 0, is the profile's
/// first run, whose room order, "?" rooms and rewards follow a script of their own
/// (<c>Overgrowth.ApplyActDiscoveryOrderModifications</c>, <c>UnknownMapPointOdds.Roll</c>,
/// <c>RewardsSet.TryGenerateTutorialRewards</c>).
/// </para>
/// </remarks>
public sealed class UnlockState
{
    /// <summary>The epoch ids (the game's <c>EpochModel.Id</c>) generation reads here.</summary>
    public static class Epochs
    {
        public const string Silent1 = "SILENT1_EPOCH";
        public const string Regent1 = "REGENT1_EPOCH";
        public const string Necrobinder1 = "NECROBINDER1_EPOCH";
        public const string Defect1 = "DEFECT1_EPOCH";
        public const string Neow = "NEOW_EPOCH";
        public const string Darv = "DARV_EPOCH";
        public const string Orobas = "OROBAS_EPOCH";
        public const string Underdocks = "UNDERDOCKS_EPOCH";
        public const string Event1 = "EVENT1_EPOCH";
        public const string Event2 = "EVENT2_EPOCH";
        public const string Event3 = "EVENT3_EPOCH";

        /// <summary>Every epoch a profile may lock here; any other is refused.</summary>
        public static readonly IReadOnlySet<string> Modelled = new HashSet<string>
        {
            Silent1,
            Regent1,
            Necrobinder1,
            Defect1,
            Darv,
            Orobas,
            Underdocks,
            Event1,
            Event2,
            Event3,
        };
    }

    public static UnlockState All { get; } = new();

    private readonly HashSet<string> _lockedEpochs;
    private readonly HashSet<int> _unseenEncounters;
    private readonly HashSet<int> _undiscoveredActs;

    /// <param name="lockedEpochs">Epochs not revealed, of <see cref="Epochs.Modelled" />.</param>
    /// <param name="unseenEncounters">Encounters (emulator ids) the profile never fought.</param>
    /// <param name="undiscoveredActs">Acts (emulator ids) not in <c>DiscoveredActs</c>.</param>
    /// <exception cref="ArgumentException">An epoch the model does not read.</exception>
    public UnlockState(
        IEnumerable<string>? lockedEpochs = null,
        IEnumerable<int>? unseenEncounters = null,
        IEnumerable<int>? undiscoveredActs = null
    )
    {
        _lockedEpochs = [.. lockedEpochs ?? []];
        _unseenEncounters = [.. unseenEncounters ?? []];
        _undiscoveredActs = [.. undiscoveredActs ?? []];
        string? unmodelled = _lockedEpochs.FirstOrDefault(e => !Epochs.Modelled.Contains(e));
        if (unmodelled is not null)
        {
            throw new ArgumentException(
                $"{unmodelled}: the run engine does not model what this epoch locks.",
                nameof(lockedEpochs)
            );
        }

        // The timeline reveals the characters in a chain: Silent1 leads to Regent1, Regent1
        // to Necrobinder1, Necrobinder1 to Defect1 (each epoch's own list of the ones it
        // unlocks), so the characters a profile has are a prefix of AllCharacters.
        Characters =
        [
            .. Enum.GetValues<PlayerCharacter>()
                .Where(c => CharacterEpoch(c) is not { } epoch || IsEpochRevealed(epoch)),
        ];
    }

    public IReadOnlySet<string> LockedEpochs => _lockedEpochs;

    public IReadOnlySet<int> UnseenEncounters => _unseenEncounters;

    public IReadOnlySet<int> UndiscoveredActs => _undiscoveredActs;

    /// <summary>The profile is the game's <c>UnlockState.all</c>.</summary>
    public bool IsAll =>
        _lockedEpochs.Count == 0 && _unseenEncounters.Count == 0 && _undiscoveredActs.Count == 0;

    /// <summary>
    /// <c>UnlockState.Characters</c>: <c>ModelDb.AllCharacters</c> minus each character whose
    /// first epoch is not revealed, in that order. The Ironclad has no such epoch.
    /// </summary>
    public IReadOnlyList<PlayerCharacter> Characters { get; }

    public bool IsEpochRevealed(string epochId) => !_lockedEpochs.Contains(epochId);

    public bool HasSeenEncounter(int encounterId) => !_unseenEncounters.Contains(encounterId);

    public bool IsActDiscovered(int act) => !_undiscoveredActs.Contains(act);

    /// <summary>
    /// <c>UnlockState.CharacterCardPools</c> minus the player's own, in AllCharacters order
    /// (Splash's <c>list.Remove(Owner.Character.CardPool)</c>, Orobas's
    /// <c>Where(c => c.Id != Owner.Character.Id)</c>).
    /// </summary>
    public IEnumerable<PlayerCharacter> Others(PlayerCharacter own) =>
        Characters.Where(c => c != own);

    /// <summary>The character's first epoch, which <c>UnlockState.Characters</c> reads.</summary>
    public static string? CharacterEpoch(PlayerCharacter character) =>
        character switch
        {
            PlayerCharacter.Silent => Epochs.Silent1,
            PlayerCharacter.Regent => Epochs.Regent1,
            PlayerCharacter.Necrobinder => Epochs.Necrobinder1,
            PlayerCharacter.Defect => Epochs.Defect1,
            _ => null,
        };

    /// <summary>A copy with these epochs locked as well.</summary>
    public UnlockState WithLockedEpochs(params string[] epochs) =>
        new([.. _lockedEpochs, .. epochs], _unseenEncounters, _undiscoveredActs);

    /// <summary>A copy with these encounters unseen as well.</summary>
    public UnlockState WithUnseenEncounters(params int[] encounters) =>
        new(_lockedEpochs, [.. _unseenEncounters, .. encounters], _undiscoveredActs);

    /// <summary>A copy with these acts undiscovered as well.</summary>
    public UnlockState WithUndiscoveredActs(params int[] acts) =>
        new(_lockedEpochs, _unseenEncounters, [.. _undiscoveredActs, .. acts]);

    /// <summary>What the profile lacks, for a report: empty for <see cref="All" />.</summary>
    public override string ToString()
    {
        var parts = new List<string>();
        if (_lockedEpochs.Count > 0)
        {
            parts.Add("locked " + string.Join(",", _lockedEpochs.Order(StringComparer.Ordinal)));
        }

        if (_unseenEncounters.Count > 0)
        {
            parts.Add("unseen " + string.Join(",", _unseenEncounters.Order()));
        }

        if (_undiscoveredActs.Count > 0)
        {
            parts.Add("undiscovered " + string.Join(",", _undiscoveredActs.Order()));
        }

        return string.Join("; ", parts);
    }
}
