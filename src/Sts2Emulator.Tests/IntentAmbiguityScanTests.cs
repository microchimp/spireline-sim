using System.Runtime.CompilerServices;
using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Spireline DESIGN.md 7.6, L8 (current move behind a non-identifying intent): a committed scan of
/// every monster and position where more than one move consistent with the observed move history
/// shows the same displayed intent (category, and damage times hits where the screen shows them).
/// <para>
/// The scan plays every encounter the emulator can open, at A0 and A10, with the enemies' real AI
/// and no player damage. At each position a roster reaches (the AI fields of every creature in it
/// are the position: the moves taken so far), the AI pass is run again from that position with 64
/// fresh streams, and the outcomes of one creature are grouped by the intent it shows. A group
/// holding more than one outcome (the AI fields the pass writes: the move taken, the branch it
/// came from) is a position where the displayed intent does not identify the move. Output is ids
/// and counts only, so it can be committed. <c>tools/leaks/intent_ambiguity_scan.json</c> is the
/// committed result; the test fails when a fresh scan differs from it.
/// </para>
/// <para>
/// Regenerate: <c>SPIRELINE_WRITE_L8_SCAN=1 dotnet test --filter IntentAmbiguityScan</c>.
/// </para>
/// </summary>
public class IntentAmbiguityScanTests
{
    private const string Seed = "84G7XUWQD6";
    private const int Draws = 64;
    private const int Rollouts = 24;
    private const int Turns = 10;
    private static readonly int[] Ascensions = [0, 10];

    private static string RepoFile([CallerFilePath] string here = "") =>
        Path.GetFullPath(Path.Combine(Path.GetDirectoryName(here)!, "..", "..", "tools", "leaks", "intent_ambiguity_scan.json"));

    private sealed class DefTally
    {
        public int Positions;
        public int AmbiguousPositions;
        public int RolledPositions;
        public readonly SortedSet<int> AmbiguousAtMoveIndex = [];
        public readonly SortedSet<int> Encounters = [];
        public readonly List<string> Examples = [];
        public int MaxMovesBehindOneIntent = 1;
    }

    private static string Show(Intent a) =>
        a.Type == IntentType.Attack || a.CarriesDamage ? $"{a.Type}/{a.Magnitude}x{a.Hits}" : $"{a.Type}";

    private static string ShownBy(EnemyState e) =>
        Show(e.CurrentIntent) + (e.SecondaryIntent is { } s ? "+" + Show(s) : "");

    /// <summary>Everything a pass writes for one creature: which move it took, and from which branch.</summary>
    private static string Outcome(EnemyState e) => $"{e.MoveIndex}/{e.LastMove}/{e.LastBranch}";

    private static string PositionKey(IReadOnlyList<EnemyState> roster) =>
        string.Join(
            ";",
            roster.Select(e =>
                $"{e.DefId}:{(e.Hp > 0 ? 1 : 0)}:{e.MoveIndex}:{e.LastMove}:{e.LastBranch}:{e.RepeatStreak}:{e.LastMoveRepeats}:{(e.OnceOnlyMoveUsed ? 1 : 0)}:{e.BranchCooldown}:{(e.RollsNextMove ? 1 : 0)}:{e.OffBalance}:{string.Join(',', e.MoveHistory)}"
            )
        );

    private static CombatState? Open(int encounter, int ascension)
    {
        var e = new RunEngine();
        e.Reset(Seed, ascension);
        for (int guard = 0; guard < 10 && e.State.Phase != RunPhase.Map; guard++)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        if (e.State.Phase != RunPhase.Map)
        {
            return null;
        }

        int option = Enumerable.Range(0, RunConstants.MapChoices).First(i => e.State.MapNodeTypes[i] == RunConstants.NodeNormal);
        try
        {
            if (e.EnterEncounterDebug(option, encounter, RunConstants.NodeNormal) < 0)
            {
                return null;
            }
        }
        catch (Exception)
        {
            return null;
        }

        return e.State.ActiveCombat;
    }

    private static SortedDictionary<int, DefTally> Scan(out int encountersOpened, out int rolloutErrors)
    {
        var tally = new SortedDictionary<int, DefTally>();
        var seenPositions = new HashSet<string>();
        encountersOpened = 0;
        rolloutErrors = 0;
        foreach (int ascension in Ascensions)
        {
            var seenRosters = new HashSet<string>();
            for (int encounter = 1; encounter <= 130; encounter++)
            {
                var opened = Open(encounter, ascension);
                if (opened is null || opened.Enemies.Count == 0)
                {
                    continue;
                }

                var start = opened.LastIntentRoll?.Before is { } before
                    ? before.Select(x => x.Clone()).ToList()
                    : opened.Enemies.Select(x => x.Clone()).ToList();
                if (!seenRosters.Add(string.Join(",", start.Select(x => x.DefId))))
                {
                    continue; // the same roster under another id (a weak fight and its normal twin)
                }

                encountersOpened++;
                for (int rollout = 0; rollout < Rollouts; rollout++)
                {
                    var combat = opened.Clone();
                    combat.Enemies.Clear();
                    foreach (var enemy in start)
                    {
                        combat.Enemies.Add(enemy.Clone());
                    }

                    var rng = new Random(encounter * 100003 + ascension * 1009 + rollout);
                    try
                    {
                        for (int turn = 0; turn < Turns && combat.Enemies.Any(x => x.Hp > 0); turn++)
                        {
                            Visit(tally, seenPositions, combat, encounter, ascension, rng);
                            EnemyAI.ChooseIntents(combat.Enemies, turn, rng, rng, ascension);
                            combat.PlayerHp = combat.PlayerMaxHp = 9999;
                            combat.PlayerBlock = 0;
                            foreach (var enemy in combat.Enemies.ToList().Where(x => x.Hp > 0))
                            {
                                EnemyAI.ExecuteIntent(enemy, combat, rng);
                            }
                        }
                    }
                    catch (Exception)
                    {
                        rolloutErrors++;
                    }
                }
            }
        }

        return tally;
    }

    /// <summary>One position: run the pass again from it with fresh streams and group by what each creature shows.</summary>
    private static void Visit(
        SortedDictionary<int, DefTally> tally,
        HashSet<string> seen,
        CombatState combat,
        int encounter,
        int ascension,
        Random rng
    )
    {
        var live = combat.Enemies;
        string key = $"{encounter}/{ascension}/{PositionKey(live)}";
        if (!seen.Add(key))
        {
            return;
        }

        var outcomes = new List<(string Shown, string Outcome)>[live.Count];
        for (int i = 0; i < live.Count; i++)
        {
            outcomes[i] = [];
        }

        for (int draw = 0; draw < Draws; draw++)
        {
            var particle = live.Select(x => x.Clone()).ToList();
            var stream = new Random(rng.Next());
            EnemyAI.ChooseIntents(particle, 0, stream, stream, ascension);
            for (int i = 0; i < live.Count; i++)
            {
                outcomes[i].Add((ShownBy(particle[i]), Outcome(particle[i])));
            }
        }

        for (int i = 0; i < live.Count; i++)
        {
            if (live[i].Hp <= 0)
            {
                continue;
            }

            if (!tally.TryGetValue(live[i].DefId, out var t))
            {
                tally[live[i].DefId] = t = new DefTally();
            }

            t.Positions++;
            if (outcomes[i].Select(o => o.Outcome).Distinct().Count() > 1)
            {
                t.RolledPositions++;
            }

            t.Encounters.Add(encounter);
            var worst = outcomes[i]
                .GroupBy(o => o.Shown)
                .Select(g => (Shown: g.Key, Moves: g.GroupBy(o => o.Outcome).OrderBy(m => m.Key, StringComparer.Ordinal).ToList()))
                .Where(g => g.Moves.Count > 1)
                .OrderByDescending(g => g.Moves.Count)
                .ToList();
            if (worst.Count == 0)
            {
                continue;
            }

            t.AmbiguousPositions++;
            t.AmbiguousAtMoveIndex.Add(live[i].MoveIndex);
            t.MaxMovesBehindOneIntent = Math.Max(t.MaxMovesBehindOneIntent, worst[0].Moves.Count);
            if (t.Examples.Count < 3)
            {
                t.Examples.Add(
                    $"enc {encounter} A{ascension} move_index {live[i].MoveIndex}: {worst[0].Shown} <- "
                        + string.Join(", ", worst[0].Moves.Select(m => $"{m.Key} x{m.Count()}"))
                );
            }
        }
    }

    private static string Render(SortedDictionary<int, DefTally> tally, int encountersOpened, int rolloutErrors)
    {
        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append("  \"what\": \"L8 scan: positions where one displayed intent (category, damage x hits) is shown by more than one move consistent with the move history. Ids and counts only.\",\n");
        sb.Append($"  \"method\": \"every encounter id 1..130 that opens, rosters deduplicated; ascensions {string.Join(",", Ascensions)}; {Rollouts} rollouts x {Turns} turns with the real AI and no player damage; at each new roster position the AI pass is re-run {Draws} times with fresh streams; outcome = move_index/last_move/last_branch after the pass\",\n");
        sb.Append($"  \"encounters_opened\": {encountersOpened},\n");
        sb.Append($"  \"rollout_errors\": {rolloutErrors},\n");
        sb.Append($"  \"monsters_scanned\": {tally.Count},\n");
        sb.Append($"  \"monsters_with_ambiguous_positions\": {tally.Count(kv => kv.Value.AmbiguousPositions > 0)},\n");
        sb.Append("  \"monsters\": {\n");
        var rows = tally.Select(kv =>
        {
            var t = kv.Value;
            var examples = string.Join(", ", t.Examples.Select(e => "\"" + e + "\""));
            return $"    \"{kv.Key}\": {{\"positions\": {t.Positions}, \"rolled_positions\": {t.RolledPositions}, \"ambiguous_positions\": {t.AmbiguousPositions}, \"displayed_intent_identifies_move\": {(t.AmbiguousPositions == 0 ? "true" : "false")}, \"max_moves_behind_one_intent\": {t.MaxMovesBehindOneIntent}, \"ambiguous_at_move_index\": [{string.Join(",", t.AmbiguousAtMoveIndex)}], \"encounters\": [{string.Join(",", t.Encounters)}], \"examples\": [{examples}]}}";
        });
        sb.Append(string.Join(",\n", rows));
        sb.Append("\n  }\n}\n");
        return sb.ToString();
    }

    [Fact]
    public void TheCommittedScanIsWhatTheEmulatorShowsNow()
    {
        var tally = Scan(out int opened, out int errors);
        string fresh = Render(tally, opened, errors);
        string path = RepoFile();
        if (Environment.GetEnvironmentVariable("SPIRELINE_WRITE_L8_SCAN") == "1")
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, fresh);
        }

        // Positive control: the scan does see rolls (otherwise an empty ambiguity list would prove nothing).
        Assert.True(tally.Values.Count(t => t.RolledPositions > 0) >= 10, "the scan saw almost no rolled move");
        Assert.True(File.Exists(path), "tools/leaks/intent_ambiguity_scan.json is not committed");
        Assert.Equal(File.ReadAllText(path).ReplaceLineEndings("\n"), fresh);
    }
    /// <summary>
    /// The scan above finds no position where two moves share an intent, so L8's "more than one move id
    /// in posterior proportions" is shown on a table of its own: moves 0 and 1 both announce an attack
    /// for 7 (weights 0.2 and 0.4), move 2 a defend (0.4). The player sees "attack 7"; the posterior
    /// over the move behind it is 1 : 2.
    /// </summary>
    [Fact]
    public void TheSamplerVariesTheMoveBehindASharedIntentInPosteriorProportion()
    {
        static void Pass(List<EnemyState> roster, Random rng)
        {
            foreach (var e in roster.Where(x => x.Hp > 0))
            {
                double r = rng.NextDouble();
                int move = r < 0.2 ? 0 : r < 0.6 ? 1 : 2;
                e.LastMove = move;
                e.MoveIndex = move;
                e.MoveHistory.Add(move);
                e.CurrentIntent = move == 2 ? new Intent(IntentType.Defend, 8) : new Intent(IntentType.Attack, 7);
            }
        }

        var combat = CombatFactory.NewCombat(seed: 0);
        combat.Enemies.Clear();
        combat.Enemies.Add(new EnemyState { DefId = KE.Mawler, Hp = 50, MaxHp = 50 });
        var before = combat.Enemies.Select(e => e.Clone()).ToArray();
        int truth = -1;
        for (int s = 0; truth != 0; s++)
        {
            // The move the fight really took: 0, one of the two behind "attack 7".
            combat.Enemies.Clear();
            combat.Enemies.AddRange(before.Select(e => e.Clone()));
            Pass(combat.Enemies, new Random(s));
            truth = combat.Enemies[0].MoveIndex;
        }

        combat.LastIntentRoll = new IntentRoll(before, [.. combat.Enemies.Select(e => e.Clone())], 0);
        var seen = new Dictionary<int, int>();
        for (int k = 0; k < 200; k++)
        {
            var fork = combat.Clone();
            Assert.True(IntentParticles.Resample(fork, new Random(k), (roster, rng) => Pass(roster, rng)));
            Assert.False(fork.AiStateAssumed);
            Assert.Equal(new Intent(IntentType.Attack, 7), fork.Enemies[0].CurrentIntent);
            seen[fork.Enemies[0].MoveIndex] = seen.GetValueOrDefault(fork.Enemies[0].MoveIndex) + 1;
        }

        Assert.Equal([0, 1], seen.Keys.OrderBy(m => m));
        // Posterior P(move 1 | attack 7) = 2/3: 133 of 200, standard deviation 6.7.
        Assert.InRange(seen[1], 106, 160);
        Assert.Equal(200, seen[0] + seen[1]);
    }
}
