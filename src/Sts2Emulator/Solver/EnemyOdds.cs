namespace Sts2Emulator.Solver;

using Sts2Emulator.Core;

/// <summary>
/// The game's odds for enemies' next moves (docs/rebuild/FIGHT_SOLVER.md section 4). The shown
/// intent is state, not chance: it is fixed and performed as shown. The next move is rolled in the
/// intent pass (<see cref="EnemyAI.ChooseIntents" />, one <c>SelectIntent</c> per living enemy in
/// roster order, on the MonsterAi stream), and every random call that pass makes is a chance node:
/// a uniform <c>Next(n)</c> over the eligible moves, or a weighted branch through
/// <see cref="Core.Rng.Roll" /> at weight / sum. The odds are found by running the pass on copies with
/// the chance hook and branching each call, so they are the emulator's own move rules and nothing
/// written beside them.
/// </summary>
/// <remarks>
/// The game's <c>RandomBranchState</c> weighs each branch by its base weight (or the owner's weight
/// function) times an eligibility factor from the machine's state log (zero for a use-only-once move
/// already used, for a cannot-repeat or repeat-limited move at its limit, for a move inside its
/// cooldown); <c>ConditionalBranchState</c> and a move's follow-up roll nothing. The emulator writes
/// those tables into <c>SelectIntent</c> per enemy; EnemyOddsTests holds the helper to the game's
/// tables for the enemies it names and to the engine's sampled frequencies.
/// </remarks>
public static class EnemyOdds
{
    /// <summary>The MonsterAi stream's index in <c>StateCloning.StreamNames</c>.</summary>
    public static readonly int AiStream = Array.IndexOf(StateCloning.StreamNames, nameof(CombatState.AiRng));

    /// <summary>
    /// The intent pass on <paramref name="roster" /> as it stands at the roll (what
    /// <see cref="EnemyAI.ChooseIntents" /> is handed): every joint outcome, as the rolled roster
    /// copy, with its probability. The roster itself is not touched.
    /// </summary>
    public static List<ChanceLeaf<List<EnemyState>>> Pass(
        IReadOnlyList<EnemyState> roster,
        int turn,
        int ascension,
        int maxWidth = 64,
        int maxLeaves = 100_000
    ) =>
        ChanceTree.Enumerate(
            source =>
            {
                var copy = roster.Select(e => e.Clone()).ToList();
                var ai = new CountingRandom(0);
                ai.SetChance(source, AiStream);
                EnemyAI.ChooseIntents(copy, turn, ai, ai, ascension);
                return copy;
            },
            maxWidth,
            maxLeaves
        );

    /// <summary>
    /// One enemy's next move, the marginal of <see cref="Pass" />: its rolled copy's move, grouped by
    /// <paramref name="move" /> (by default the announced intents and the move state the rules keep),
    /// with the probabilities summed.
    /// </summary>
    public static List<(TKey Move, double P)> Next<TKey>(
        IReadOnlyList<EnemyState> roster,
        int enemyIndex,
        int turn,
        int ascension,
        Func<EnemyState, TKey> move
    )
        where TKey : notnull
    {
        var odds = new Dictionary<TKey, double>();
        foreach (var leaf in Pass(roster, turn, ascension))
        {
            var key = move(leaf.Result[enemyIndex]);
            odds[key] = odds.GetValueOrDefault(key) + leaf.P;
        }

        return [.. odds.Select(kv => (kv.Key, kv.Value))];
    }

    /// <summary>The move an enemy rolled, as <see cref="Next{TKey}" /> groups it by default.</summary>
    public static MoveKey MoveOf(EnemyState enemy) =>
        new(enemy.CurrentIntent, enemy.SecondaryIntent, enemy.LastMove, enemy.LastBranch);

    /// <inheritdoc cref="Next{TKey}" />
    public static List<(MoveKey Move, double P)> Next(IReadOnlyList<EnemyState> roster, int enemyIndex, int turn, int ascension) =>
        Next(roster, enemyIndex, turn, ascension, MoveOf);

    /// <summary>
    /// From a decision state with its shown intents fixed: <paramref name="step" /> (the end of the
    /// turn) run on copies with only the MonsterAi stream branched, every other stream drawing as it
    /// would. Each leaf is the stepped copy, so its enemies hold the move they rolled at the game's odds,
    /// after everything the enemy turn did to the inputs of their move rules.
    /// </summary>
    public static List<ChanceLeaf<CombatState>> AfterStep(
        CombatState state,
        CountingRandom rng,
        Action<CombatState, CountingRandom> step,
        int maxWidth = 64,
        int maxLeaves = 100_000
    )
    {
        if (state.AiRng is not CountingRandom)
        {
            throw new InvalidOperationException("the intent pass rolls on the step's own stream: its odds cannot be told from the step's other draws");
        }

        return ChanceTree.Enumerate(
            source =>
            {
                var copy = state.Clone();
                var r = rng.Copy();
                ChanceHook.Attach(copy, r, source);
                step(copy, r);
                ChanceHook.Attach(copy, r, null);
                return copy;
            },
            maxWidth,
            maxLeaves,
            streams: [AiStream],
            passOthers: true
        );
    }
}

/// <summary>An enemy's rolled move: what it announces and the move state its rules keep.</summary>
public readonly record struct MoveKey(Intent Intent, Intent? Secondary, int LastMove, int LastBranch);
