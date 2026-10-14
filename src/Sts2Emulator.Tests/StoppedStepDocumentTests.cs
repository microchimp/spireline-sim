using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Interop;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// The agent's document (<c>CombatSnapshot.Write</c>, native <c>Sts2_GetStateJson</c>) of a combat
/// stopped on a screen raised in the middle of a step, read back and answered: the step goes on from
/// the screen exactly as answering in memory does. Round 4's document wrote the screen alone, so the
/// card being played was in no pile and lost for good (17 cards, not 18), the play was never counted,
/// Brand's Strength never came, Foregone Conclusion's hand was never drawn (1 card, not 6) and
/// Well-Laid Plans' turn never ended. The document now carries the public part of the step (the kind
/// of step, the card or potion in play, the answers, the start as the player saw it, the cards the
/// step took off the draw pile and the screens it reached) and never the true order or the streams.
/// </summary>
/// <remarks>
/// Round 5's document lost the stops whose way to the screen ran through the hidden order (review,
/// ZReviewR5Doc, seeds 1-30): Havoc's and Mayhem's Burning Pact were played off the top of a pile in
/// the loader's order, so the replay played another card and the pick was dropped (24 and 26 of 30
/// wrong, a card duplicated); Hellraiser's hand draw drew other Strikes, so the Seeker Strike was lost
/// or doubled and visible damage undone (28, 8 and 22 of 30). The replay is now held to the cards the
/// step took and the screens it reached, and each case below loads with thirty seeds.
/// </remarks>
public class StoppedStepDocumentTests
{
    private static readonly int[] Twelve =
    [
        IC.Anger, IC.DefendIronclad, IC.Inflame, IC.TwinStrike,
        IC.ShrugItOff, IC.PommelStrike, IC.IronWave, IC.Thunderclap,
        IC.Bash, IC.StrikeIronclad, IC.Armaments, IC.Headbutt,
    ];

    /// <summary>
    /// The reviewer's ten screens (round 4, ZReviewR4Doc): five card plays, four turn screens, a potion;
    /// and round 5's stops whose way to the screen runs through the hidden order: Havoc's and Mayhem's
    /// Burning Pact (with Entropy answered first, too), Hellraiser's Seeker Strike, and Entropy's
    /// screen after a Hellraiser hand draw that drew the Seeker Strike (its pick answered) or did not;
    /// and Entropy's after a turn end where I Am Invincible, on top of the true pile, played itself;
    /// and round 6's (review, ZReviewR6.B_): Well-Laid Plans' and Foregone Conclusion's screens after a
    /// turn end where I Am Invincible, second from the top, did not play itself and nothing was taken
    /// before the screen, and Foregone Conclusion's second pick after its first took I Am Invincible.
    /// </summary>
    public static TheoryData<string> Screens() =>
        new()
        {
            "BurningPact", "TrueGrit+", "Headbutt", "Armaments", "Brand", "ForegoneConclusion", "Tyranny", "Entropy", "WellLaidPlans", "GamblersBrew",
            "HavocBurningPact", "MayhemBurningPact", "MayhemEntropyBurningPact", "HellraiserSeekerStrike", "HellraiserEntropyStrikeDrawn", "HellraiserEntropyStrikeUndrawn",
            "IAmInvincibleEntropy",
            "WellLaidPlansIAmInvincibleSecond", "ForegoneConclusionIAmInvincibleSecond", "ForegoneConclusionPicksIAmInvincible",
        };

    private const int IAmInvincible = 257;

    /// <summary>The seeds every case is loaded with.</summary>
    private static readonly int[] Seeds = [.. Enumerable.Range(1, 30)];

    /// <summary>A combat stopped on <paramref name="screen" />, and the step stream it stopped on.</summary>
    internal static (Fight Fight, CountingRandom Rng, string Step) Stopped(string screen)
    {
        Fight fight;
        CountingRandom rng;
        string step;
        switch (screen)
        {
            case "BurningPact" or "TrueGrit+" or "Headbutt" or "Armaments" or "Brand":
            {
                var (card, upgraded) = screen switch
                {
                    "BurningPact" => (IC.BurningPact, false),
                    "TrueGrit+" => (IC.TrueGrit, true),
                    "Headbutt" => (IC.Headbutt, false),
                    "Armaments" => (IC.Armaments, false),
                    _ => (IC.Brand, false),
                };
                fight = Fight
                    .Hand(Card(card, upgraded), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash))
                    .Energy(3)
                    .Draw([.. Pile(Twelve)])
                    .Discard(Card(IC.Inflame), Card(IC.Anger))
                    .Enemy(hp: 400);
                rng = new CountingRandom(77);
                CombatEngine.Step(fight.State, 0, rng);
                step = "card_play";
                break;
            }

            case "HavocBurningPact":
                // Havoc plays the top of the draw pile: the true order's top is Burning Pact.
                fight = Fight
                    .Hand(Card(IC.Havoc), Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash))
                    .Energy(3)
                    .Draw([.. Pile(IC.BurningPact, IC.Anger, IC.DefendIronclad, IC.Inflame, IC.ShrugItOff, IC.IronWave, IC.Thunderclap)])
                    .Enemy(hp: 400);
                rng = new CountingRandom(9);
                CombatEngine.Step(fight.State, 0, rng);
                step = "card_play";
                break;

            case "MayhemBurningPact" or "MayhemEntropyBurningPact":
                // The turn start draws five and Mayhem plays the sixth, Burning Pact.
                fight = Fight
                    .Hand(Card(IC.DefendIronclad))
                    .Energy(3)
                    .Draw([.. Pile(IC.DefendIronclad, IC.Bash, IC.Anger, IC.Inflame, IC.ShrugItOff, IC.BurningPact, IC.IronWave, IC.Thunderclap, IC.DefendIronclad, IC.ShrugItOff)])
                    .Enemy(hp: 400)
                    .PlayerBuff(BuffId.MayhemPower, 1);
                if (screen == "MayhemEntropyBurningPact")
                {
                    fight.PlayerBuff(BuffId.EntropyPower, 1);
                }

                rng = new CountingRandom(9);
                CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
                if (screen == "MayhemEntropyBurningPact")
                {
                    Assert.Equal(CardSelectionKind.TransformInHand, fight.Pending!.Kind);
                    CombatEngine.Step(fight.State, 0, rng);
                }

                Assert.Equal(IC.BurningPact, fight.Pending!.SourceCardDefId);
                step = "turn_start";
                break;

            case "HellraiserSeekerStrike" or "HellraiserEntropyStrikeDrawn" or "HellraiserEntropyStrikeUndrawn":
            {
                // Hellraiser plays a Strike the hand draw draws, there and then: the Seeker Strike
                // on top asks inside the draw; at the bottom it is never drawn.
                int[] rest = [IC.DefendIronclad, IC.Bash, IC.Anger, IC.Inflame, IC.ShrugItOff, IC.IronWave, IC.Thunderclap, IC.Armaments];
                int[] pile = screen == "HellraiserEntropyStrikeUndrawn" ? [.. rest, CL.SeekerStrike] : [CL.SeekerStrike, .. rest];
                fight = Fight
                    .Hand(Card(IC.DefendIronclad))
                    .Energy(3)
                    .Draw([.. Pile(pile)])
                    .Enemy(hp: 400)
                    .PlayerBuff(BuffId.Hellraiser, 1);
                if (screen != "HellraiserSeekerStrike")
                {
                    fight.PlayerBuff(BuffId.EntropyPower, 1);
                }

                rng = new CountingRandom(9);
                CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
                if (screen != "HellraiserEntropyStrikeUndrawn")
                {
                    Assert.Equal(CL.SeekerStrike, fight.Pending!.SourceCardDefId);
                }

                if (screen == "HellraiserEntropyStrikeDrawn")
                {
                    CombatEngine.Step(fight.State, 0, rng);
                }

                if (screen != "HellraiserSeekerStrike")
                {
                    Assert.Equal(CardSelectionKind.TransformInHand, fight.Pending!.Kind);
                }

                step = "turn_start";
                break;
            }

            case "IAmInvincibleEntropy":
                // I Am Invincible plays itself when it is on top as the play phase ends: whether
                // it does is read off the hidden order, and its block meets the enemy's hit.
                fight = Fight
                    .Hand(Card(IC.DefendIronclad))
                    .Energy(3)
                    .Draw([Card(IAmInvincible), .. Pile(IC.DefendIronclad, IC.Bash, IC.Anger, IC.Inflame, IC.ShrugItOff, IC.IronWave, IC.Thunderclap, IC.Armaments)])
                    .Enemy(hp: 400)
                    .PlayerBuff(BuffId.EntropyPower, 1);
                rng = new CountingRandom(9);
                CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
                Assert.Equal(CardSelectionKind.TransformInHand, fight.Pending!.Kind);
                Assert.Contains(fight.State.ExhaustPile.Concat(fight.State.DiscardPile), card => card.DefId == IAmInvincible);
                step = "turn_start";
                break;

            case "WellLaidPlansIAmInvincibleSecond" or "ForegoneConclusionIAmInvincibleSecond" or "ForegoneConclusionPicksIAmInvincible":
            {
                // I Am Invincible second from the top: it does not play itself as the play phase
                // ends. Nothing is taken before Well-Laid Plans' screen or Foregone Conclusion's, and
                // Foregone Conclusion's pick of it is taken from the middle: the loader's order must
                // not play it.
                bool picks = screen == "ForegoneConclusionPicksIAmInvincible";
                fight = Fight
                    .Hand(Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
                    .Energy(3)
                    .Draw([Card(Twelve[0]), Card(IAmInvincible), .. Pile(Twelve[1..])])
                    .Enemy(hp: 400)
                    .PlayerBuff(screen.StartsWith("WellLaidPlans") ? BuffId.WellLaidPlans : BuffId.ForegoneConclusion, picks ? 2 : 1);
                rng = new CountingRandom(9);
                CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
                Assert.Contains(fight.State.DrawPile, card => card.DefId == IAmInvincible);
                Assert.Equal([false], fight.Pending!.Replay!.Peeks);
                if (picks)
                {
                    var open = fight.Pending!;
                    Assert.Equal(CardSelectionKind.DrawPileToHand, open.Kind);
                    int offering = Enumerable.Range(0, open.Candidates.Count).First(c => fight.State.DrawPile[open.Candidates[c]].DefId == IAmInvincible);
                    CombatEngine.Step(fight.State, offering, rng);
                    Assert.Equal(CardSelectionKind.DrawPileToHand, fight.Pending!.Kind);
                    Assert.Contains(fight.State.Hand, card => card.DefId == IAmInvincible);
                    Assert.Equal(IAmInvincible, Assert.Single(fight.Pending.Replay!.Taken).DefId);
                }
                else
                {
                    Assert.Empty(fight.Pending!.Replay!.Taken);
                }

                step = screen.StartsWith("WellLaidPlans") ? "end_turn" : "turn_start";
                break;
            }

            case "GamblersBrew":
                fight = Fight
                    .Hand(Card(IC.StrikeIronclad), Card(IC.DefendIronclad), Card(IC.Bash), Card(IC.Anger))
                    .Energy(3)
                    .Draw([.. Pile(Twelve)])
                    .Enemy(hp: 400);
                fight.State.PotionSlots[0] = 30;
                rng = new CountingRandom(3);
                CombatEngine.Step(fight.State, fight.State.Hand.Count + 1, rng);
                step = "potion";
                break;

            default:
            {
                var buff = screen switch
                {
                    "ForegoneConclusion" => BuffId.ForegoneConclusion,
                    "Tyranny" => BuffId.Tyranny,
                    "Entropy" => BuffId.EntropyPower,
                    _ => BuffId.WellLaidPlans,
                };
                fight = Fight
                    .Hand(Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
                    .Energy(3)
                    .Draw([.. Pile(Twelve)])
                    .Enemy(hp: 400)
                    .PlayerBuff(buff, 1);
                rng = new CountingRandom(9);
                CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
                step = screen == "WellLaidPlans" ? "end_turn" : "turn_start";
                break;
            }
        }

        Assert.NotNull(fight.Pending?.Replay);
        return (fight, rng, step);
    }

    /// <summary>Answers the screen with its first candidate, and any screen after it with its decline (or first).</summary>
    internal static void Answer(CombatState state, Random rng)
    {
        CombatEngine.Step(state, 0, rng);
        for (int guard = 0; guard < 8 && state.PendingSelection is { } more; guard++)
        {
            CombatEngine.Step(state, more.Skippable ? more.Candidates.Count : 0, rng);
        }
    }

    internal static int Cards(CombatState s) => s.Hand.Count + s.DrawPile.Count + s.DiscardPile.Count + s.ExhaustPile.Count;

    /// <summary>What the reviewer's probe compared, and the turn.</summary>
    internal static string Shape(CombatState s) =>
        $"cards {Cards(s)} hand {s.Hand.Count} draw {s.DrawPile.Count} discard {s.DiscardPile.Count} exhaust {s.ExhaustPile.Count} "
        + $"strength {BuffSystem.Get(s.PlayerBuffs, BuffId.Strength)} played {s.CardsPlayedThisCombat}/{s.CardsPlayedThisTurn} "
        + $"energy {s.Energy} turn {s.Turn} hp {s.PlayerHp} enemy {s.Enemies[0].Hp} screen {s.PendingSelection?.Kind.ToString() ?? "-"}";

    private static CombatState Answered(CombatState state, CountingRandom rng)
    {
        var copy = state.Clone();
        Answer(copy, rng.Copy());
        return copy;
    }

    /// <summary>
    /// The reviewer's probe: Write, Read with each of thirty seeds, answer on a stream of its own,
    /// against answering the same screen in memory. The unseen order and the streams are the
    /// loader's, so the cards drawn after the screen may differ; the card count, the counts, the
    /// counters, Brand's Strength, the hand Foregone Conclusion draws, Well-Laid Plans' next turn,
    /// Havoc's and Mayhem's Burning Pact and Hellraiser's Strikes may not.
    /// </summary>
    [Theory]
    [MemberData(nameof(Screens))]
    public void TheAgentsDocumentOfAStoppedStepFinishesItAsAnsweringInMemoryDoes(string screen)
    {
        var (fight, rng, step) = Stopped(screen);
        var inMemory = Answered(fight.State, rng);
        byte[] document = CombatSnapshot.Write(fight.State);
        Assert.Contains($"\"step\":\"{step}\"", Encoding.UTF8.GetString(document));

        var wrong = new List<string>();
        foreach (int seed in Seeds)
        {
            var loaded = CombatSnapshot.Read(document, seed, out _);
            Assert.True(loaded.PendingSelection!.Replay!.FromDocument);
            Answer(loaded, new CountingRandom(seed));
            if (Shape(loaded) != Shape(inMemory) || loaded.PendingSelection is not null)
            {
                wrong.Add($"seed {seed}: {Shape(loaded)}");
            }
        }

        Assert.True(wrong.Count == 0, $"in memory {Shape(inMemory)}\n" + string.Join("\n", wrong));
        // No card is made or lost by these steps: the step's start holds them all, in its piles.
        Assert.Equal(Cards(fight.State.PendingSelection!.Replay!.Before), Cards(inMemory));
    }

    /// <summary>
    /// The same round trip with the loaded copy handed the original's hidden state (its streams, its
    /// draw order and the screen's candidates over it): the answer comes out as the in-memory answer,
    /// byte for byte. Nothing of the step is lost on the way through the document.
    /// </summary>
    [Theory]
    [MemberData(nameof(Screens))]
    public void HandedTheTrueWorldTheLoadedDocumentAnswersExactlyAsInMemory(string screen)
    {
        var (fight, rng, _) = Stopped(screen);
        var inMemory = Answered(fight.State, rng);

        var loaded = CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 5, out _);
        HandOver(fight.State, loaded);
        Answer(loaded, rng.Copy());

        Assert.Equal(Encoding.UTF8.GetString(CombatSnapshot.Write(inMemory)), Encoding.UTF8.GetString(CombatSnapshot.Write(loaded)));
    }

    /// <summary>A search fork of a loaded stop answers in its own world and finishes the step the same way.</summary>
    [Theory]
    [MemberData(nameof(Screens))]
    public void AForkOfALoadedStopFinishesTheStepToo(string screen)
    {
        var (fight, rng, _) = Stopped(screen);
        var inMemory = Answered(fight.State, rng);
        var loaded = CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 5, out _);
        ShareStreams(from: fight.State, to: loaded);

        foreach (int seed in new[] { 11, 12 })
        {
            var fork = loaded.Clone(seed);
            fork.ReshuffleUnknownDrawOrder(new CountingRandom(seed));
            Assert.Equal(CombatSnapshot.Write(loaded), CombatSnapshot.Write(fork));
            Answer(fork, new CountingRandom(seed));
            Assert.Equal(Shape(inMemory), Shape(fork));
        }
    }

    /// <summary>
    /// Forks resampled differently at a stopped step still write one agent's document, and the block
    /// carries no true world: no true order, no stream, no answering copy's hidden state, and the
    /// start's unseen cards as the sorted multiset the rest of the document writes.
    /// </summary>
    [Theory]
    [MemberData(nameof(Screens))]
    public void ForksResampledDifferentlyWriteOneDocumentWithNoTrueWorldInIt(string screen)
    {
        var (fight, _, _) = Stopped(screen);
        byte[] original = CombatSnapshot.Write(fight.State);
        string text = Encoding.UTF8.GetString(original);

        foreach (string field in new[] { "before_draw", "before_streams", "answer_indices", "\"hidden\"", "lanes" })
        {
            Assert.DoesNotContain(field, text);
        }

        foreach (int seed in new[] { 4242, 7, 999, 31337 })
        {
            var fork = fight.State.Clone(seed);
            fork.ReshuffleUnknownDrawOrder(new CountingRandom(seed));
            Assert.Equal(original, CombatSnapshot.Write(fork));
        }

        Assert.Contains("before_streams", Encoding.UTF8.GetString(CombatSnapshot.WriteSave(fight.State)));
    }

    /// <summary>
    /// A loaded stop written again is the same document, and so is its save document: a step from a
    /// document has no true world to add.
    /// </summary>
    [Theory]
    [MemberData(nameof(Screens))]
    public void ALoadedStopWritesBackTheSameDocument(string screen)
    {
        var (fight, _, _) = Stopped(screen);
        byte[] written = CombatSnapshot.Write(fight.State);

        var loaded = CombatSnapshot.Read(written, seed: 5, out _);

        Assert.Equal(written, CombatSnapshot.Write(loaded));
        Assert.Equal(written, CombatSnapshot.WriteSave(loaded));
    }

    /// <summary>
    /// A full deck holding I Am Invincible in the draw pile at turn one, Foregone Conclusion and Mayhem
    /// on: the first end turn stops on Foregone Conclusion's screen. The export checks its step on seed
    /// 1 alone, so every order of one deck got the same answer, and round 6 refused to write 11 of 12
    /// (review, ZReviewR6.F_). Every order now writes its document, and it loads and answers.
    /// </summary>
    [Fact]
    public void AFullDeckTurnOnePileHoldingIAmInvincibleIsWrittenAtEveryOrder()
    {
        int[] deck =
        [
            IC.Hellraiser, IC.BattleTrance, IC.PommelStrike, CL.SeekerStrike, CL.SeekerStrike, IC.StrikeIronclad, IC.TwinStrike,
            IC.BurningPact, IC.Armaments, IC.ShrugItOff, IC.DefendIronclad, IC.DefendIronclad, IC.Havoc, IAmInvincible,
        ];
        var refused = new List<string>();
        int played = 0;
        foreach (int seed in Seeds)
        {
            var order = new Random(seed);
            var fight = Fight
                .Hand()
                .Energy(3)
                .Draw([.. deck.OrderBy(_ => order.Next()).Select(id => new CardInstance(id, false))])
                .Enemy(hp: 120)
                .Enemy(hp: 90)
                .Seed(seed)
                .PlayerBuff(BuffId.ForegoneConclusion, 1)
                .PlayerBuff(BuffId.MayhemPower, 1);
            var rng = new CountingRandom(seed);
            CombatEngine.Step(fight.State, 0, rng);
            Assert.Equal(CardSelectionKind.DrawPileToHand, fight.Pending!.Kind);
            played += fight.Pending.Replay!.Peeks.Count(peek => peek);
            try
            {
                byte[] document = CombatSnapshot.Write(fight.State);
                var loaded = CombatSnapshot.Read(document, seed: 1, out _);
                Answer(loaded, new CountingRandom(seed));
            }
            catch (Exception e) when (e is SnapshotException or StoppedStepMismatch)
            {
                refused.Add($"order {seed}: {e.Message}");
            }
        }

        Assert.True(refused.Count == 0, string.Join('\n', refused));
        Assert.InRange(played, 0, Seeds.Length - 1);
    }

    /// <summary>
    /// Two screens in one turn start (Tyranny's, then Entropy's), the document written on the second:
    /// the first answer is replayed in the loader's world, the second screen is the document's, and the
    /// turn goes on as in memory.
    /// </summary>
    [Fact]
    public void AStepStoppedOnItsSecondScreenGoesOnFromTheDocument()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .Energy(3)
            .Draw([.. Pile(Twelve)])
            .Enemy(hp: 400)
            .PlayerBuff(BuffId.Tyranny, 1)
            .PlayerBuff(BuffId.EntropyPower, 1);
        var rng = new CountingRandom(9);
        CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
        CombatEngine.Step(fight.State, 0, rng);
        Assert.Single(fight.Pending!.Replay!.Answers);

        var inMemory = Answered(fight.State, rng);
        var loaded = CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 5, out _);
        Assert.Single(loaded.PendingSelection!.Replay!.Answers);
        Answer(loaded, new CountingRandom(5));

        Assert.Equal(Shape(inMemory), Shape(loaded));

        var exact = CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 5, out _);
        HandOver(fight.State, exact);
        Answer(exact, rng.Copy());
        Assert.Equal(Encoding.UTF8.GetString(CombatSnapshot.Write(inMemory)), Encoding.UTF8.GetString(CombatSnapshot.Write(exact)));
    }

    /// <summary>
    /// A pick from the draw pile answered before the document was written is named by its card, not by
    /// its index into the true order (Foregone Conclusion+, two picks, written on the second).
    /// </summary>
    [Fact]
    public void AnEarlierDrawPilePickIsWrittenByCard()
    {
        var fight = Fight
            .Hand(Card(IC.StrikeIronclad), Card(IC.DefendIronclad))
            .Energy(3)
            .Draw([.. Pile(Twelve)])
            .Enemy(hp: 400)
            .PlayerBuff(BuffId.ForegoneConclusion, 2);
        var rng = new CountingRandom(9);
        CombatEngine.Step(fight.State, fight.State.Hand.Count, rng);
        Assert.Equal(CardSelectionKind.DrawPileToHand, fight.Pending!.Kind);
        var first = fight.State.DrawPile[fight.Pending.Candidates[3]];
        CombatEngine.Step(fight.State, 3, rng);
        // The screen reopens for the second pick, the first one in the step's answers.
        Assert.Equal(CardSelectionKind.DrawPileToHand, fight.Pending!.Kind);
        Assert.Equal([new StepAnswer(3, first)], fight.Pending.Replay!.Answers);

        string text = Encoding.UTF8.GetString(CombatSnapshot.Write(fight.State));
        Assert.Contains("\"answers\":[{\"card\":{\"def_id\":" + first.DefId + ",", text);

        var inMemory = Answered(fight.State, rng);
        var loaded = CombatSnapshot.Read(CombatSnapshot.Write(fight.State), seed: 5, out _);
        Answer(loaded, new CountingRandom(5));
        Assert.Equal(Shape(inMemory), Shape(loaded));
        Assert.Contains(first, loaded.Hand);
    }

    /// <summary>
    /// The in-place copy a step from a document puts its answering copy back with: the same document as
    /// a clone, and the combat keeps its lists and its enemies (the turn end's baseline holds them).
    /// </summary>
    [Fact]
    public void AnInPlaceCopyKeepsTheListsAndEnemiesItFills()
    {
        var (fight, _, _) = Stopped("Headbutt");
        var into = new CombatState();
        fight.State.CopyTo(into);
        var hand = into.Hand;
        var enemy = into.Enemies[0];
        var buffs = enemy.Buffs;

        var other = Stopped("Brand").Fight.State;
        other.CopyTo(into, inPlace: true);

        Assert.Same(hand, into.Hand);
        Assert.Same(enemy, into.Enemies[0]);
        Assert.Same(buffs, into.Enemies[0].Buffs);
        Assert.Equal(CombatSnapshot.WriteSave(other.Clone()), CombatSnapshot.WriteSave(into));
        Assert.NotSame(other.Hand, into.Hand);
        Assert.NotSame(other.Enemies[0], into.Enemies[0]);
    }

    /// <summary>The loaded copy handed the original's streams, draw order and the screen's candidates over it.</summary>
    private static void HandOver(CombatState from, CombatState to)
    {
        ShareStreams(from, to);
        to.DrawPile = [.. from.DrawPile];
        to.PendingSelection = to.PendingSelection!.Clone(to.PendingSelection.Replay, candidates: [.. from.PendingSelection!.Candidates]);
    }

    private static void ShareStreams(CombatState from, CombatState to)
    {
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            StateCloning.SetStream(to, i, StateCloning.Stream(from, i)?.Copy());
        }
    }
}
