using System.Text;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Sts2Emulator.Tests.Invariants;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// A copy determinized while it stands on a screen raised mid-step: the answer replays the step on
/// the streams and the draw order the step started with, so that it reaches the same screen, and
/// at that screen -- before the answer is resolved -- the copy's own hidden state goes back on
/// (<see cref="ScreenHiddenState" />). The pick and everything after it play in the copy's world.
/// </summary>
/// <remarks>
/// The rejected round (02443058) replayed the whole step on the true streams and order, and redid
/// the determinization only once the step was over (b3af1d35): Burning Pact's two draws were the
/// true next two cards in every fork, so a search choosing what to exhaust saw what it would draw.
/// </remarks>
public class DeterminizedStopTests
{
    // ── the reviewer's probe, through the native exports a search uses ───────────────────

    private static readonly int[] EightUnknown =
    [
        IC.Anger, IC.DefendIronclad, IC.Inflame, IC.TwinStrike,
        IC.ShrugItOff, IC.PommelStrike, IC.IronWave, IC.Thunderclap,
    ];

    private static string BurningPactDocument()
    {
        static string Card(int id) => $$"""{"def_id": {{id}}, "upgraded": false}""";
        return $$"""
            {
              "schema": "sts2-emu-snapshot/1",
              "ascension": 0,
              "turn": 1,
              "player": {"hp": 70, "max_hp": 80, "energy": 3, "max_energy": 3},
              "piles": {
                "hand": [{{Card(IC.BurningPact)}}, {{Card(IC.StrikeIronclad)}}, {{Card(IC.DefendIronclad)}}, {{Card(IC.Bash)}}],
                "draw": {"unordered": [{{string.Join(", ", EightUnknown.Select(Card))}}]}
              },
              "enemies": [{"def_id": {{KE.Chomper}}, "hp": 400, "max_hp": 400, "intent": {"type": 0, "magnitude": 0} }]
            }
            """;
    }

    /// <summary>
    /// Burning Pact stopped on its screen over an eight-card draw pile nobody has seen; five forks
    /// resampled with seeds 100 to 104 (`Sts2_Clone`: Clone(seed) and ReshuffleUnknownDrawOrder),
    /// each answered with the same exhaust. Each fork draws the top two of ITS OWN pile as it stood
    /// on the screen, and the five draw five different pairs, as on main, where the screen stood
    /// over the finished play. The rejected round drew the true pair in all five.
    /// </summary>
    [Fact]
    public void FiveResampledForksOfBurningPactsScreenDrawFromTheirOwnPiles()
    {
        int handle = Native.Load(BurningPactDocument(), seed: 1);
        try
        {
            Native.Step(handle, 0);
            Assert.Equal(CardSelectionKind.ExhaustFromHandThenDraw, (CardSelectionKind)Native.PendingKind(handle));
            int[] truePile = Native.Pile(handle, 0);

            var pairs = new List<string>();
            foreach (int seed in new[] { 100, 101, 102, 103, 104 })
            {
                int fork = Native.Clone(handle, resample: true, seed);
                try
                {
                    int[] pile = Native.Pile(fork, 0);
                    Assert.Equal(truePile.Order(), pile.Order());
                    int[] hand = Native.Pile(fork, 1);

                    Native.Step(fork, 0);

                    int[] drawn = Added(hand, Native.Pile(fork, 1));
                    Assert.Equal(pile[..2].Order(), drawn.Order());
                    Assert.Equal(pile[2..], Native.Pile(fork, 0));
                    pairs.Add(string.Join(",", pile[..2]));
                }
                finally
                {
                    Native.Destroy(fork);
                }
            }

            Assert.Equal(5, pairs.Distinct().Count());
        }
        finally
        {
            Native.Destroy(handle);
        }
    }

    // ── every kind of screen with something hidden after the pick ───────────────────────

    /// <summary>A position that stops on a screen, and whether its draw order is still hidden there.</summary>
    private sealed record Scenario(
        string Name,
        Func<Fight> Build,
        Action<Fight, Random> Stop,
        bool DrawOrderHiddenAtTheStop
    );

    private static readonly Scenario[] Scenarios =
    [
        // A card screen with draws after the pick.
        new(
            "Burning Pact: exhaust, then draw two",
            () => Fight.Hand(Card(IC.BurningPact), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash))
                .Energy(3).Draw([.. Pile(EightUnknown)]).Enemy(hp: 400),
            (f, rng) => CombatEngine.Step(f.State, 0, rng),
            DrawOrderHiddenAtTheStop: true
        ),
        // A random target after the pick: the exhaust feeds Feel No Pain's block to Juggernaut.
        new(
            "Burning Pact into Feel No Pain and Juggernaut, two enemies",
            () => Fight.Hand(Card(IC.BurningPact), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash))
                .Energy(3).Draw([.. Pile(EightUnknown)]).Enemy(hp: 400).Enemy(hp: 400)
                .PlayerBuff(BuffId.FeelNoPain, 3).PlayerBuff(BuffId.Juggernaut, 5),
            (f, rng) => CombatEngine.Step(f.State, 0, rng),
            DrawOrderHiddenAtTheStop: true
        ),
        // A potion screen with draws after the picks.
        new(
            "Gambler's Brew: discard, then draw as many",
            () =>
            {
                var f = Fight.Hand(Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash), Card(IC.Anger))
                    .Energy(3).Draw([.. Pile(EightUnknown)]).Enemy(hp: 400);
                f.State.PotionSlots[0] = 30;
                return f;
            },
            (f, rng) => CombatEngine.Step(f.State, f.State.Hand.Count + 1, rng),
            DrawOrderHiddenAtTheStop: true
        ),
        // Auto-plays after the pick, each rolling its target.
        new(
            "Decisions, Decisions: Taunt auto-played thrice at random enemies",
            () =>
            {
                var f = Fight.Hand(Card(IC.Taunt), Card(129), Card(IC.DefendIronclad))
                    .Enemy(hp: 60).Enemy(hp: 60)
                    .Draw([.. Enumerable.Repeat(Card(IC.StrikeIronclad), 6)]);
                f.State.Stars = 9;
                return f;
            },
            (f, rng) => CombatEngine.Step(f.State, 1, rng),
            DrawOrderHiddenAtTheStop: false
        ),
        // A hook screen inside a draw's shuffle, and the draw goes on after the pick.
        new(
            "Stratagem: the shuffle asks, the hand draw goes on",
            () => Fight.Hand(Card(IC.StrikeIronclad)).Energy(3).Enemy(hp: 400).Draw()
                .Discard([.. Pile(EightUnknown)]).PlayerBuff(BuffId.StratagemPower, 1),
            (f, rng) => CombatEngine.Step(f.State, f.State.Hand.Count, rng),
            DrawOrderHiddenAtTheStop: true
        ),
        // A turn-start screen with a roll after the pick.
        new(
            "Entropy: the pick is transformed on the card-selection stream",
            () => Fight.Hand().Energy(3).Draw([.. Pile(EightUnknown)]).Enemy(hp: 400)
                .PlayerBuff(BuffId.EntropyPower, 1),
            (f, rng) => CombatEngine.Step(f.State, f.State.Hand.Count, rng),
            DrawOrderHiddenAtTheStop: false
        ),
        new(
            "Tyranny into Feel No Pain and Juggernaut, two enemies",
            () => Fight.Hand(Card(IC.StrikeIronclad)).Energy(3).Draw([.. Pile(EightUnknown)])
                .Enemy(hp: 400).Enemy(hp: 400)
                .PlayerBuff(BuffId.Tyranny, 1).PlayerBuff(BuffId.FeelNoPain, 3).PlayerBuff(BuffId.Juggernaut, 5),
            (f, rng) => CombatEngine.Step(f.State, f.State.Hand.Count, rng),
            DrawOrderHiddenAtTheStop: false
        ),
    ];

    public static TheoryData<string> EveryScenario()
    {
        var data = new TheoryData<string>();
        foreach (var scenario in Scenarios)
        {
            data.Add(scenario.Name);
        }

        return data;
    }

    private static readonly string[] Hidden =
    [
        ".ShuffleRng", ".TargetRng", ".CardSelectionRng", ".EnergyCostRng", ".CardGenerationRng",
        ".PotionGenerationRng", ".OrbGenerationRng", ".AiRng", ".NicheHpRng", ".DrawPile[",
        ".PendingSelection",
    ];

    /// <summary>
    /// Two worlds that look the same on the screen and differ in what is hidden -- every stream,
    /// and the draw order where nothing has drawn from it yet -- each forked onto the SAME hidden
    /// state and answered the same way, end the same: nothing after the pick reads the world the
    /// fork came from. Before the fix each fork went on in its own world.
    /// </summary>
    [Theory]
    [MemberData(nameof(EveryScenario))]
    public void TwoWorldsForkedOntoTheSameHiddenStateAnswerTheSame(string name)
    {
        var scenario = Scenarios.Single(s => s.Name == name);
        var a = scenario.Build();
        var b = scenario.Build();
        b.State.ReseedStreams(555);
        if (scenario.DrawOrderHiddenAtTheStop)
        {
            b.State.DrawPile.Reverse();
        }

        var rngA = new CountingRandom(1);
        var rngB = new CountingRandom(2);
        scenario.Stop(a, rngA);
        scenario.Stop(b, rngB);
        Assert.NotNull(a.Pending?.Replay);
        Assert.NotNull(b.Pending?.Replay);
        // The same screen over the same visible position.
        Assert.Empty(DeepState.Diff(a.State, b.State, skip: Hidden));
        Assert.Equal(a.Pending!.Candidates, b.Pending!.Candidates);
        Assert.Equal(a.State.DrawPile.Select(c => c.ToString()).Order(), b.State.DrawPile.Select(c => c.ToString()).Order());

        var forkA = a.State.Clone();
        var forkB = b.State.Clone();
        var world = a.State.Clone(rngSeed: 4242);
        world.ReshuffleUnknownDrawOrder(new CountingRandom(9));
        foreach (var fork in new[] { forkA, forkB })
        {
            for (int i = 0; i < StateCloning.StreamCount; i++)
            {
                StateCloning.SetStream(fork, i, StateCloning.Stream(world, i)?.Copy());
            }

            fork.DrawPile = [.. world.DrawPile];
        }

        var forkRng = rngA.Reseeded(4242)!;
        var forkRngA = forkRng.Copy();
        var forkRngB = forkRng.Copy();
        Answer(forkA, forkRngA);
        Answer(forkB, forkRngB);

        Assert.Empty(DeepState.Diff(forkA, forkB));
        Assert.Equal(forkRngA.CallCount, forkRngB.CallCount);
    }

    /// <summary>Answers every screen until none is open: the first card, then decline where it may.</summary>
    private static void Answer(CombatState state, Random rng)
    {
        bool first = true;
        for (int guard = 0; guard < 16 && state.PendingSelection is { } open; guard++)
        {
            int answer = !first && open.Skippable ? open.Candidates.Count : 0;
            first = false;
            CombatEngine.Step(state, answer, rng);
        }

        Assert.Null(state.PendingSelection);
    }

    // ── the copy nobody determinized, and the run's own forks ──────────────────────────

    /// <summary>
    /// A resampled copy answered keeps its resampled streams -- the seeds a resample gives them,
    /// at the original's positions -- and its twin plays out the same; the original keeps its own.
    /// </summary>
    [Fact]
    public void AResampledStopAnsweredGoesOnOnTheResampledStreams()
    {
        var fight = Scenarios[0].Build();
        var rng = new CountingRandom(11);
        Scenarios[0].Stop(fight, rng);
        int trueSeed = fight.State.TargetRng!.Seed;
        var resampled = fight.State.Clone(rngSeed: 99);
        int resampledSeed = resampled.TargetRng!.Seed;
        Assert.NotEqual(trueSeed, resampledSeed);
        var twin = fight.State.Clone(rngSeed: 99);

        CombatEngine.Step(resampled, 0, rng.Clone());
        CombatEngine.Step(twin, 0, rng.Clone());
        CombatEngine.Step(fight.State, 0, rng);

        Assert.Equal(fight.State.ExhaustPile, resampled.ExhaustPile);
        Assert.Equal(resampledSeed, resampled.TargetRng!.Seed);
        Assert.Equal(fight.State.TargetRng!.CallCount, resampled.TargetRng.CallCount);
        Assert.Equal(trueSeed, fight.State.TargetRng.Seed);
        Assert.Empty(DeepState.Diff(twin, resampled));
    }

    /// <summary>
    /// <c>RunEngine.Clone(resampleSeed)</c> and <c>ForkActiveCombat(resampleSeed)</c> determinize
    /// the run's fight their own way (the resampled run's streams, a reshuffle on the run's
    /// resample stream) and note nothing on the stopped step: the answer still draws from the
    /// fork's pile, because what goes back on at the screen is whatever the fork holds.
    /// </summary>
    [Fact]
    public void TheRunsResampledClonesAndForksDrawFromTheirOwnPilesToo()
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN");
        Assert.Equal(0, engine.StartCombat([.. RunConstants.StarterDeckIds], RunConstants.SlitheringStranglerEncounterId, [], 64, 80, [], 99));
        var combat = engine.State.ActiveCombat!;
        combat.Hand = [Card(IC.BurningPact), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash)];
        combat.DrawPile = [.. Pile(EightUnknown)];
        combat.KnownTopCount = 0;
        combat.KnownBottomCount = 0;
        combat.Energy = 3;
        CombatEngine.Step(combat, 0, engine.State.ActiveCombatRng!);
        Assert.NotNull(combat.PendingSelection?.Replay);

        var pairs = new List<string>();
        for (int seed = 1; seed <= 4; seed++)
        {
            var clone = engine.Clone(seed);
            pairs.Add(DrawsTheTopTwo(clone.State.ActiveCombat!, clone.State.ActiveCombatRng!));
            var (fork, forkRng) = engine.ForkActiveCombat(seed)!.Value;
            Assert.Equal(pairs[^1], DrawsTheTopTwo(fork, forkRng));
        }

        Assert.True(pairs.Distinct().Count() > 1, "four resampled run clones drew the same pair");
    }

    private static string DrawsTheTopTwo(CombatState combat, CountingRandom rng)
    {
        var pile = combat.DrawPile.Select(c => c.DefId).ToArray();
        var hand = combat.Hand.Select(c => c.DefId).ToList();
        CombatEngine.Step(combat, 0, rng);
        var drawn = Added(hand, combat.Hand.Select(c => c.DefId)).Order().ToArray();
        Assert.Equal(pile[..2].Order(), drawn);
        return string.Join(",", pile[..2]);
    }

    /// <summary>The cards in <paramref name="after" /> that <paramref name="before" /> did not hold, counted as a multiset.</summary>
    private static int[] Added(IEnumerable<int> before, IEnumerable<int> after)
    {
        var left = before.ToList();
        var added = new List<int>();
        foreach (int card in after)
        {
            if (!left.Remove(card))
            {
                added.Add(card);
            }
        }

        return [.. added];
    }

    private static unsafe class Native
    {
        public static int Load(string json, int seed)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(json);
            var observation = new int[NativeExports.OBS_SIZE];
            delegate* unmanaged<byte*, int, int, int*, int> load = &NativeExports.LoadCombatSnapshot;
            fixed (byte* text = bytes)
            fixed (int* obs = observation)
            {
                int handle = load(text, bytes.Length, seed, obs);
                Assert.True(handle >= 0, $"Sts2_LoadCombatSnapshot returned {handle}");
                return handle;
            }
        }

        public static void Step(int handle, int action)
        {
            delegate* unmanaged<int, int, int*, float*, int> step = &NativeExports.Sts2_Step;
            var observation = new int[NativeExports.OBS_SIZE];
            float reward;
            fixed (int* obs = observation)
            {
                step(handle, action, obs, &reward);
            }
        }

        public static int Clone(int handle, bool resample, int seed)
        {
            delegate* unmanaged<int, int, int, int*, int> clone = &NativeExports.Sts2_Clone;
            var observation = new int[NativeExports.OBS_SIZE];
            fixed (int* obs = observation)
            {
                int copy = clone(handle, resample ? 1 : 0, seed, obs);
                Assert.True(copy >= 0, $"Sts2_Clone returned {copy}");
                return copy;
            }
        }

        public static void Destroy(int handle)
        {
            delegate* unmanaged<int, void> destroy = &NativeExports.Sts2_Destroy;
            destroy(handle);
        }

        /// <summary>Def ids of pile 0 (draw, top first) or 1 (hand).</summary>
        public static int[] Pile(int handle, int pile)
        {
            delegate* unmanaged<int, int, int*, int, int> getPile = &NativeExports.Sts2_GetPile;
            var pairs = new int[2 * 64];
            int count;
            fixed (int* buffer = pairs)
            {
                count = getPile(handle, pile, buffer, 64);
            }

            return Enumerable.Range(0, count).Select(i => pairs[2 * i]).ToArray();
        }

        public static int PendingKind(int handle)
        {
            delegate* unmanaged<int, byte*, int, int> getStateJson = &NativeExports.GetStateJson;
            int needed = getStateJson(handle, null, 0);
            var buffer = new byte[needed];
            fixed (byte* text = buffer)
            {
                getStateJson(handle, text, needed);
            }

            return JsonNode.Parse(Encoding.UTF8.GetString(buffer))!["pending_selection"]!["kind"]!.GetValue<int>();
        }
    }
}
