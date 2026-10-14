// The profile a recorded run was generated for, fitted to what its history shows. The seed does
// not decide a run alone: the player's profile (Sts2Emulator.Core.UnlockState) decides which acts
// can be rolled, which events are in an act's shuffle, which ancients an act can open on, which
// boss an act is forced to, and whether Neow may offer Kaleidoscope -- and a run history does not
// record the profile. Each of these leaves a mark the history does show (the acts, the fights in
// order, the ancients, Neow's three options), so the walker generates the run's start under the
// mature profile first and, only where a mark disagrees, under the profile changes that explain
// it, keeping a change only when it makes every mark of its kind agree.
using System.Text.Json;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Epochs = Sts2Emulator.Core.UnlockState.Epochs;

namespace EmuDump.Codex;

public static class ProfileFit
{
    /// <summary>
    /// Fights a history must show before a locked event epoch is fitted from them. A locked
    /// event moves every later UpFront draw, so a wrong guess still matches a fight or two by
    /// chance; it does not match three in order.
    /// </summary>
    public const int MinFightsForEventEpochs = 3;

    /// <summary>The marks a history leaves of its run's start, in the emulator's names.</summary>
    public sealed record Marks(
        string[] Acts,
        string[] Neow,
        string?[] Ancients,
        List<string>[] Normals,
        List<string>[] Elites,
        List<string>[] Bosses
    )
    {
        public int Fights => Normals.Sum(l => l.Count) + Elites.Sum(l => l.Count) + Bosses.Sum(l => l.Count);

        public static Marks Of(JsonElement rec)
        {
            string[] acts = [.. rec.GetProperty("acts").EnumerateArray().Select(a => Names.Bare(a.GetString()))];
            var history = rec.GetProperty("map_point_history").EnumerateArray().ToList();
            int n = history.Count;
            var marks = new Marks(acts, [], new string?[n], New(n), New(n), New(n));
            string[] neow = [];
            for (int act = 0; act < n; act++)
            {
                bool first = true;
                foreach (var point in history[act].EnumerateArray())
                {
                    string kind = point.GetProperty("map_point_type").GetString() ?? "";
                    var room = point.TryGetProperty("rooms", out var rs) && rs.GetArrayLength() > 0 ? rs[0] : default;
                    string model = room.ValueKind == JsonValueKind.Object && room.TryGetProperty("model_id", out var m) ? m.GetString() ?? "" : "";
                    if (first && kind == "ancient")
                    {
                        marks.Ancients[act] = Names.Bare(model);
                        if (act == 0 && point.TryGetProperty("player_stats", out var ps) && ps.GetArrayLength() > 0
                            && ps[0].TryGetProperty("ancient_choice", out var choice) && choice.ValueKind == JsonValueKind.Array)
                        {
                            neow = [.. choice.EnumerateArray().Select(Walker.AncientOption)];
                        }
                    }

                    first = false;
                    string type = room.ValueKind == JsonValueKind.Object && room.TryGetProperty("room_type", out var t) ? t.GetString() ?? "" : "";
                    if (kind is not ("monster" or "elite" or "boss" or "unknown") || model.Length == 0)
                    {
                        continue;
                    }

                    var list = type switch
                    {
                        "monster" => marks.Normals[act],
                        "elite" => marks.Elites[act],
                        "boss" => marks.Bosses[act],
                        _ => null,
                    };
                    list?.Add(Names.EncounterModelToEmulator(model));
                }
            }

            return marks with { Neow = neow };
        }

        static List<string>[] New(int n) => [.. Enumerable.Range(0, n).Select(_ => new List<string>())];
    }

    /// <summary>The profile the run's marks fit, or <see cref="UnlockState.All" /> when they all agree with it.</summary>
    /// <remarks>Leaves <paramref name="engine" /> reset to the run under the returned profile.</remarks>
    public static UnlockState Fit(JsonElement rec, RunEngine engine, string seed, int ascension, PlayerCharacter character)
    {
        var marks = Marks.Of(rec);
        RunState Generate(UnlockState profile)
        {
            engine.Reset(seed, ascension, character, profile);
            return engine.State;
        }

        var profile = UnlockState.All;
        var run = Generate(profile);

        // The act roll has a stream of its own, so the acts are fitted alone: an alternate act the
        // profile has not unlocked is never rolled, and one it has unlocked but never visited is
        // taken without a roll.
        if (!ActsAgree(marks, run))
        {
            foreach (var candidate in new[]
                     {
                         profile.WithLockedEpochs(Epochs.Underdocks),
                         profile.WithUndiscoveredActs(RunConstants.ActUnderdocks),
                     })
            {
                if (ActsAgree(marks, Generate(candidate)))
                {
                    profile = candidate;
                    break;
                }
            }

            run = Generate(profile);
        }

        // The fights: a locked event takes one draw out of its act's shuffle, which moves every
        // fight after it; a boss the profile has never met replaces the one drawn and moves
        // nothing. Fewest locked epochs first, and the bosses fitted inside each try.
        if (!FightsAgree(marks, run))
        {
            foreach (string[] events in EventEpochSubsets())
            {
                if (events.Length > 0 && marks.Fights < MinFightsForEventEpochs)
                {
                    break;
                }

                var candidate = profile.WithLockedEpochs(events);
                var tried = Generate(candidate);
                var unseen = UnseenBosses(marks, tried);
                if (unseen.Length > 0)
                {
                    candidate = candidate.WithUnseenEncounters(unseen);
                    tried = Generate(candidate);
                }

                if (FightsAgree(marks, tried))
                {
                    profile = candidate;
                    break;
                }
            }

            run = Generate(profile);
        }

        // The later acts' ancients: Hive offers Orobas only once its epoch is revealed, and Darv
        // joins the shared ancients with his. Neither moves a draw, only what it picks.
        if (!AncientsAgree(marks, run))
        {
            foreach (var candidate in new[]
                     {
                         profile.WithLockedEpochs(Epochs.Orobas),
                         profile.WithLockedEpochs(Epochs.Darv),
                         profile.WithLockedEpochs(Epochs.Orobas, Epochs.Darv),
                     })
            {
                if (AncientsAgree(marks, Generate(candidate)))
                {
                    profile = candidate;
                    break;
                }
            }

            run = Generate(profile);
        }

        // Neow: Kaleidoscope wants every character unlocked. The characters unlock in a chain
        // ending with the Defect, so a profile short of one lacks the Defect at least; which
        // others it lacks Neow does not show.
        if (!NeowAgrees(marks, run) && character != PlayerCharacter.Defect)
        {
            var candidate = profile.WithLockedEpochs(Epochs.Defect1);
            if (NeowAgrees(marks, Generate(candidate)))
            {
                profile = candidate;
            }

            Generate(profile);
        }

        return profile;
    }

    /// <summary>What the profile lacks, for the report, with unseen encounters by name.</summary>
    public static string Describe(UnlockState profile)
    {
        var parts = new List<string>();
        if (profile.LockedEpochs.Count > 0)
        {
            parts.Add("locked " + string.Join(",", profile.LockedEpochs.Order(StringComparer.Ordinal)));
        }

        if (profile.UnseenEncounters.Count > 0)
        {
            parts.Add("unseen " + string.Join(",", profile.UnseenEncounters.Order().Select(Names.EncounterName)));
        }

        if (profile.UndiscoveredActs.Count > 0)
        {
            parts.Add("undiscovered " + string.Join(",", profile.UndiscoveredActs.Order().Select(Names.Act)));
        }

        return string.Join("; ", parts);
    }

    /// <summary>Every subset of the three event epochs, fewest first.</summary>
    static IEnumerable<string[]> EventEpochSubsets()
    {
        string[] all = [Epochs.Event1, Epochs.Event2, Epochs.Event3];
        return Enumerable.Range(0, 1 << all.Length)
            .Select(mask => all.Where((_, i) => (mask & (1 << i)) != 0).ToArray())
            .OrderBy(s => s.Length);
    }

    /// <summary>
    /// The recorded boss of each act that the run did not draw but its BossDiscoveryOrder holds:
    /// the profile has not met it, so it is forced.
    /// </summary>
    static int[] UnseenBosses(Marks marks, RunState run)
    {
        var unseen = new List<int>();
        for (int act = 0; act < Math.Min(marks.Bosses.Length, run.Acts.Count); act++)
        {
            if (marks.Bosses[act].Count == 0 || marks.Bosses[act][0] == Names.EncounterName(run.Acts[act].BossEncounterId))
            {
                continue;
            }

            int[] order = RunConstants.BossDiscoveryOrder(run.Acts[act].Act);
            int at = Array.FindIndex(order, id => Names.EncounterName(id) == marks.Bosses[act][0]);
            if (at >= 0)
            {
                unseen.Add(order[at]);
            }
        }

        return [.. unseen];
    }

    public static bool ActsAgree(Marks marks, RunState run) =>
        marks.Acts.SequenceEqual(run.Acts.Select(a => Names.Act(a.Act)));

    /// <summary>Every recorded fight is the next one of its kind the run generated, act by act.</summary>
    public static bool FightsAgree(Marks marks, RunState run)
    {
        for (int act = 0; act < marks.Normals.Length; act++)
        {
            if (act >= run.Acts.Count)
            {
                return false;
            }

            var rooms = run.Acts[act];
            int[] bosses = rooms.SecondBossEncounterId != 0 ? [rooms.BossEncounterId, rooms.SecondBossEncounterId] : [rooms.BossEncounterId];
            if (!Prefix(marks.Normals[act], rooms.NormalEncounters) || !Prefix(marks.Elites[act], rooms.EliteEncounters)
                || !Prefix(marks.Bosses[act], bosses))
            {
                return false;
            }
        }

        return true;
    }

    public static bool AncientsAgree(Marks marks, RunState run)
    {
        for (int act = 1; act < marks.Ancients.Length; act++)
        {
            if (marks.Ancients[act] is { } ancient && (act >= run.Acts.Count || run.Acts[act].Ancient != ancient))
            {
                return false;
            }
        }

        return true;
    }

    public static bool NeowAgrees(Marks marks, RunState run) =>
        marks.Neow.Length == 0 || marks.Neow.SequenceEqual(run.NeowOptions.Where(x => x != 0).Select(Names.Relic));

    static bool Prefix(List<string> recorded, int[] generated) =>
        recorded.Count <= generated.Length && recorded.Select((name, i) => name == Names.EncounterName(generated[i])).All(x => x);
}
