// The setup mirror's replay of a pick the game made on a card selection a relic's pickup opened
// (tools/emudump/Engines/Old/OldSetup.cs AnswerDeckSelection). The fixtures are real referee records (lab batches
// dense-a and dense-b), trimmed to the map items; the game's selector took the first card of each list it
// was shown. CardSelectCmd lists the eligible deck cards in deck order, except a removal, which lists its
// curses first (then deck order).
//
// - Kifuda and Gnarled Hammer ask for zero to three cards, so the game's one pick closes the screen;
//   the emulator's screen wants three and has DONE, which the mirror used to never press: the item was
//   "unsupported: the emulator's selection wants more cards than the game's screen required".
// - A later Punch Dagger or Royal Stamp lists the deck without the card an earlier relic enchanted, so
//   its first pick is the next eligible Strike: the mirror used to match the pick by card id among the
//   cards the earlier (wrong) picks left, and found none: "card 472 the game picked is not selectable".
// - A removal (Biiig Hug, Empty Cage, Precarious Shears, Precise Scissors, Preserved Fog) lists a curse
//   in the deck first, so the game's first pick is the curse, not the deck's first card.
// - A screen whose cards differ from the game's (an earlier item left the emulator's deck elsewhere) is
//   matched by card id, as the mirror always did, and is a gap only when the id cannot tell the copies apart.
using System.Text.Json.Nodes;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

public class SetupPickTests
{
    static JsonObject Fixture(string name) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", name + ".json")))!.AsObject();

    /// <summary>The fixture's map items replayed as EmuDump does; returns the run and the emulator's setup log.</summary>
    static (RunEngine e, List<JsonObject> log) Replay(JsonObject record)
    {
        var outp = new JsonObject();
        var e = OldEngine.EnterRoom(record, outp);
        return (e, outp["setup_log"]!.AsArray().Select(x => x!.AsObject()).ToList());
    }

    static JsonObject LogOf(List<JsonObject> log, JsonObject record, string relic) =>
        log.First(x => (int)x["i"]! >= 0 && (string?)record["setup"]![(int)x["i"]!]!["id"] == relic);

    static void AssertSupported(List<JsonObject> log)
    {
        foreach (var x in log.Where(x => (int)x["i"]! >= 0))
        {
            Assert.True(x["unsupported"] == null, x.ToJsonString());
        }
    }

    [Fact]
    public void KifudaTakesTheOneCardTheGamePickedAndClosesTheScreen()
    {
        var record = Fixture("kifuda");
        var (e, log) = Replay(record);

        var item = LogOf(log, record, "KIFUDA");
        Assert.True((bool)item["ok"]!, item.ToJsonString());
        AssertSupported(log);
        // The game's list started with the deck's first Strike and it took that one card, not three.
        var adroit = e.State.ActiveCombat!.Hand.Concat(e.State.ActiveCombat.DrawPile).Concat(e.State.ActiveCombat.DiscardPile)
            .Count(c => c.Enchantment == Enchantment.Adroit);
        Assert.Equal(1, adroit);
    }

    [Fact]
    public void GnarledHammerTakesTheOneCardTheGamePickedAndALaterPunchDaggerTheNextStrike()
    {
        var record = Fixture("gnarled_hammer_then_punch_dagger");
        var (e, log) = Replay(record);

        Assert.True((bool)LogOf(log, record, "GNARLED_HAMMER")["ok"]!);
        Assert.True((bool)LogOf(log, record, "PUNCH_DAGGER")["ok"]!);
        AssertSupported(log);
        var all = e.State.ActiveCombat!.Hand.Concat(e.State.ActiveCombat.DrawPile).Concat(e.State.ActiveCombat.DiscardPile).ToList();
        Assert.Equal(1, all.Count(c => c.Enchantment == Enchantment.Sharp));
        Assert.Equal(1, all.Count(c => c.Enchantment == Enchantment.Momentum));
        // Two different Strikes: the Hammer's card is out of the Dagger's list.
        Assert.All(all.Where(c => c.Enchantment is Enchantment.Sharp or Enchantment.Momentum),
            c => Assert.Equal(OldIds.CardIdBySnake("STRIKE_IRONCLAD"), c.DefId));
    }

    [Fact]
    public void RoyalStampTakesTheCardAfterTheOneKifudaEnchanted()
    {
        var record = Fixture("kifuda_then_royal_stamp");
        var (e, log) = Replay(record);

        Assert.True((bool)LogOf(log, record, "KIFUDA")["ok"]!);
        Assert.True((bool)LogOf(log, record, "ROYAL_STAMP")["ok"]!);
        AssertSupported(log);
        var all = e.State.ActiveCombat!.Hand.Concat(e.State.ActiveCombat.DrawPile).Concat(e.State.ActiveCombat.DiscardPile).ToList();
        Assert.Equal(1, all.Count(c => c.Enchantment == Enchantment.Adroit));
        var stamped = all.Where(c => c.Enchantment == Enchantment.RoyallyApproved).ToList();
        Assert.Single(stamped);
        Assert.Equal(OldIds.CardIdBySnake("STRIKE_IRONCLAD"), stamped[0].DefId);
    }

    // ---- Screens that list other cards than the game's (records from lab batch dense-b) --------------------
    // Royal Stamp's fight already differs at step 0 (an earlier item left the emulator's deck elsewhere), so the two
    // lists differ; the game's pick is the first Strike, which the id names without doubt. The mirror counted the item
    // as supported before the list check, and must go on doing so. Dolly's Mirror's record differed the same way until
    // the Neow's Bones fix (below).

    [Fact]
    public void RoyalStampOnOtherCardsThanTheGamesStillTakesTheStrikeTheGamePickedByIdAndNotesTheDifference()
    {
        var record = Fixture("royal_stamp_on_other_cards");
        var (e, log) = Replay(record);

        var item = LogOf(log, record, "ROYAL_STAMP");
        Assert.True((bool)item["ok"]!, item.ToJsonString());
        Assert.Null(item["unsupported"]);
        Assert.StartsWith("note: the emulator's selection lists", (string)item["msg"]!);
        AssertSupported(log);
        var all = e.State.ActiveCombat!.Hand.Concat(e.State.ActiveCombat.DrawPile).Concat(e.State.ActiveCombat.DiscardPile).ToList();
        var stamped = all.Where(c => c.Enchantment == Enchantment.RoyallyApproved).ToList();
        Assert.Single(stamped);
        Assert.Equal(OldIds.CardIdBySnake("STRIKE_IRONCLAD"), stamped[0].DefId);
    }

    /// <summary>
    /// This record's Neow pick is Neow's Bones, and the deck the emulator's screen listed differed from the game's because a
    /// relic claimed off the Bones' screen never opened its own selection. Since the claim opens it (fix/f16-gold-screens),
    /// the two screens list the same cards: the game's pick is followed by position and there is nothing to note. The
    /// other-cards path stays covered by Royal Stamp's record above.
    /// </summary>
    [Fact]
    public void DollysMirrorAfterNeowsBonesListsTheGamesCardsAndCopiesTheStrikeTheGamePicked()
    {
        var record = Fixture("dollys_mirror_on_other_cards");
        var (e, log) = Replay(record);

        var item = LogOf(log, record, "DOLLYS_MIRROR");
        Assert.True((bool)item["ok"]!, item.ToJsonString());
        Assert.Null(item["unsupported"]);
        Assert.DoesNotContain("selection lists", (string?)item["msg"] ?? "");
        AssertSupported(log);
        // The starter deck's five Strikes and the copy.
        var all = e.State.ActiveCombat!.Hand.Concat(e.State.ActiveCombat.DrawPile).Concat(e.State.ActiveCombat.DiscardPile).ToList();
        Assert.Equal(6, all.Count(c => c.DefId == OldIds.CardIdBySnake("STRIKE_IRONCLAD")));
    }

    // ---- Every relic whose pickup opens a deck selection -------------------------------------------------

    /// <summary>The relics whose pickup opens a deck selection the mirror answers (RunFollowUp.TransformSelect).</summary>
    public static TheoryData<int> Siblings =>
        new()
        {
            RunConstants.RelicAstrolabe, RunConstants.RelicBeautifulBracelet, RunConstants.RelicBiiigHug, RunConstants.RelicClaws,
            RelicEffects.DollysMirror, RunConstants.RelicElectricShrymp, RunConstants.RelicEmptyCage, RelicEffects.GnarledHammer,
            RelicEffects.Kifuda, RunConstants.RelicNewLeaf, RunConstants.RelicPaelsGrowth, RunConstants.RelicPaelsTooth,
            RunConstants.RelicPomander, RunConstants.RelicPrecariousShears, RunConstants.RelicPreciseScissors,
            RunConstants.RelicPreservedFog, RelicEffects.PunchDagger, RelicEffects.RoyalStamp, RunConstants.RelicTriBoomerang,
            RunConstants.RelicYummyCookie,
        };

    /// <summary>The relics whose pickup opens a removal (FromDeckForRemoval, which lists the curses first).</summary>
    public static TheoryData<int> Removals =>
        new()
        {
            RunConstants.RelicBiiigHug, RunConstants.RelicEmptyCage, RunConstants.RelicPaelsTooth, RunConstants.RelicPrecariousShears,
            RunConstants.RelicPreciseScissors, RunConstants.RelicPreservedFog,
        };

    /// <summary>Sixteen different attacks and skills, so every kind has cards to offer and a pick that lands on the wrong card shows.</summary>
    static readonly string[] SixteenCards =
    {
        "BASH", "SHRUG_IT_OFF", "SEEKER_STRIKE", "TAUNT", "BLUDGEON", "IMPERVIOUS", "THUNDERCLAP", "SHOCKWAVE", "HEADBUTT",
        "DEFEND_IRONCLAD", "TWIN_STRIKE", "OFFERING", "REND", "STRIKE_IRONCLAD", "MANGLE", "HEMOKINESIS",
    };

    static RunEngine AtMap()
    {
        var e = new RunEngine();
        e.Reset("MIRRORPICK1", 10);
        for (int guard = 0; guard < 10 && e.State.Phase != RunPhase.Map; guard++)
        {
            e.Step(0, -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Map, e.State.Phase);
        return e;
    }

    static RunEngine AtMapWithDeck(params string[] cards)
    {
        var e = AtMap();
        e.State.Deck.Clear();
        foreach (string card in cards)
        {
            RunNonCombatEffects.AddCardToDeck(e.State, new CardInstance(OldIds.CardIdBySnake(card), Upgraded: false));
        }

        return e;
    }

    static string[] Names(RunState s) => s.Deck.Select(c => RunSnap.CardEntry(c.DefId)).ToArray();

    /// <summary>The curses these tests put in decks (the game's card type Curse).</summary>
    static readonly HashSet<string> Curses = new() { "DOUBT", "SHAME" };

    static bool IsCurse(RunState s, int deckIndex) => Curses.Contains(RunSnap.CardEntry(s.Deck[deckIndex].DefId));

    /// <summary>
    /// The deck indices of the cards the pending selection offers, in the order the game lists them: deck order, except a
    /// removal (FromDeckForRemoval), whose sorting order puts every Curse first (a stable sort, so deck order within each part).
    /// </summary>
    static int[] GamesList(RunState s)
    {
        var eligible = Enumerable.Range(0, s.Deck.Count).Where(i => RunNonCombatEffects.CanSelectCard(s, i)).ToArray();
        bool removal = s.PendingSelectionKind is DeckSelection.Remove or DeckSelection.RemoveUpgradable or DeckSelection.RemoveTaggedBasic;
        return removal ? eligible.Where(i => IsCurse(s, i)).Concat(eligible.Where(i => !IsCurse(s, i))).ToArray() : eligible;
    }

    /// <summary>The record the referee's selector writes for the screen on show: the game's list, the first n taken.</summary>
    static JsonObject TheGamesChoice(RunEngine e, out int[] listed, out int n)
    {
        var s = e.State;
        listed = GamesList(s);
        int max = s.PendingSelectionCount;
        int min = s.PendingSelectionOptional ? 0 : max;
        // FirstChoiceSelector.Pick: min(list, max(min, min(1, max))).
        n = Math.Min(listed.Length, Math.Max(min, Math.Min(1, max)));
        string Entry(int i) => RunSnap.CardEntry(s.Deck[i].DefId);
        return new JsonObject
        {
            ["kind"] = "cards", ["step"] = -1, ["min"] = min, ["max"] = max,
            ["options"] = new JsonArray(listed.Select(i => (JsonNode)Entry(i)).ToArray()),
            ["picked"] = new JsonArray(Enumerable.Range(0, n).Select(i => (JsonNode)i).ToArray()),
        };
    }

    /// <summary>A card choice as the referee's selector logs it, written out card by card.</summary>
    static JsonObject Choice(int min, int max, string[] options, params int[] picked) =>
        new()
        {
            ["kind"] = "cards", ["step"] = -1, ["min"] = min, ["max"] = max,
            ["options"] = new JsonArray(options.Select(o => (JsonNode)o).ToArray()),
            ["picked"] = new JsonArray(picked.Select(p => (JsonNode)p).ToArray()),
        };

    static void AssertTheGamesFirstCardsAreTheOnesTaken(int relic, string[] cards)
    {
        var e = AtMapWithDeck(cards);
        var s = e.State;
        var before = s.Deck.ToList();
        var followUp = RunNonCombatEffects.ApplyRelicPickup(s, relic, SelectionReturn.Map);
        Assert.Equal(RunFollowUp.TransformSelect, followUp);
        s.Phase = RunPhase.TransformSelect;
        var kind = s.PendingSelectionKind;
        var choice = TheGamesChoice(e, out var listed, out int n);
        Assert.True(n >= 1 && listed.Length > n, "the fixture deck must make the game ask");

        string? gap = OldSetup.AnswerDeckSelection(e, new List<JsonObject> { choice }, out var note);

        Assert.Null(gap);
        Assert.Null(note);  // the two screens listed the same cards
        Assert.NotEqual(RunPhase.TransformSelect, s.Phase);
        // The cards the game took are the first n of its list, and no others were touched.
        var taken = listed.Take(n).ToList();
        if (kind is DeckSelection.Enchant or DeckSelection.Upgrade)
        {
            // Changed in place: exactly the first n of the list differ.
            Assert.Equal(taken.OrderBy(i => i), Enumerable.Range(0, before.Count).Where(i => s.Deck[i] != before[i]));
        }
        else if (kind == DeckSelection.Duplicate)
        {
            Assert.True(s.Deck.Take(before.Count).SequenceEqual(before));
            Assert.True(s.Deck.Skip(before.Count).Take(n).SequenceEqual(taken.Select(i => before[i])));
        }
        else
        {
            // Taken out of the deck (a transform adds its result at the end): the rest keep their order.
            var untouched = Enumerable.Range(0, before.Count).Where(i => !taken.Contains(i)).Select(i => before[i]).ToList();
            Assert.True(s.Deck.Take(untouched.Count).SequenceEqual(untouched), "the cards the game left alone moved or changed");
        }
    }

    [Theory]
    [MemberData(nameof(Siblings))]
    public void EveryRelicWhoseSelectionTheGameClosedWithItsPicksReplaysThem(int relic) =>
        AssertTheGamesFirstCardsAreTheOnesTaken(relic, SixteenCards);

    [Theory]
    [MemberData(nameof(Removals))]
    public void ARemovalTheGameListedCursesFirstTakesTheCursesBeforeTheDecksFirstCards(int relic)
    {
        // Two curses among the sixteen, neither first: the game lists them first, in deck order, whatever n it takes.
        var cards = SixteenCards.ToList();
        cards.Insert(5, "DOUBT");
        cards.Insert(10, "SHAME");
        AssertTheGamesFirstCardsAreTheOnesTaken(relic, cards.ToArray());
    }

    [Fact]
    public void EmptyCageRemovesTheCurseAndThenTheFirstCardOfTheDeck()
    {
        var e = AtMapWithDeck("BASH", "SHRUG_IT_OFF", "TAUNT", "BLUDGEON", "HEADBUTT", "DOUBT");
        var s = e.State;
        Assert.Equal(RunFollowUp.TransformSelect, RunNonCombatEffects.ApplyRelicPickup(s, RunConstants.RelicEmptyCage, SelectionReturn.Map));
        s.Phase = RunPhase.TransformSelect;
        Assert.Equal(2, s.PendingSelectionCount);
        // The game's list, as its selector logged it: the curse first, then the deck; it took the first two.
        var choice = Choice(2, 2, new[] { "DOUBT", "BASH", "SHRUG_IT_OFF", "TAUNT", "BLUDGEON", "HEADBUTT" }, 0, 1);

        string? gap = OldSetup.AnswerDeckSelection(e, new List<JsonObject> { choice });

        Assert.Null(gap);
        Assert.Equal(new[] { "SHRUG_IT_OFF", "TAUNT", "BLUDGEON", "HEADBUTT" }, Names(s));
    }

    [Fact]
    public void PreciseScissorsRemovesTheCurseNotTheDecksFirstCard()
    {
        var e = AtMapWithDeck("BASH", "SHRUG_IT_OFF", "TAUNT", "BLUDGEON", "HEADBUTT", "DOUBT");
        var s = e.State;
        Assert.Equal(RunFollowUp.TransformSelect, RunNonCombatEffects.ApplyRelicPickup(s, RunConstants.RelicPreciseScissors, SelectionReturn.Map));
        s.Phase = RunPhase.TransformSelect;
        Assert.Equal(1, s.PendingSelectionCount);
        var choice = Choice(1, 1, new[] { "DOUBT", "BASH", "SHRUG_IT_OFF", "TAUNT", "BLUDGEON", "HEADBUTT" }, 0);

        string? gap = OldSetup.AnswerDeckSelection(e, new List<JsonObject> { choice });

        Assert.Null(gap);
        Assert.Equal(new[] { "BASH", "SHRUG_IT_OFF", "TAUNT", "BLUDGEON", "HEADBUTT" }, Names(s));
    }

    [Fact]
    public void DollysMirrorListsItsCurseInDeckOrderNotFirst()
    {
        // FromDeckGeneric has no sorting order for the copy (only a removal sorts the curses first).
        var e = AtMapWithDeck("BASH", "SHRUG_IT_OFF", "DOUBT", "TAUNT");
        var s = e.State;
        Assert.Equal(RunFollowUp.TransformSelect, RunNonCombatEffects.ApplyRelicPickup(s, RelicEffects.DollysMirror, SelectionReturn.Map));
        s.Phase = RunPhase.TransformSelect;
        var choice = Choice(1, 1, new[] { "BASH", "SHRUG_IT_OFF", "DOUBT", "TAUNT" }, 2);  // the game's list is the deck's order

        string? gap = OldSetup.AnswerDeckSelection(e, new List<JsonObject> { choice }, out var note);

        Assert.Null(gap);
        Assert.Null(note);
        Assert.Equal(new[] { "BASH", "SHRUG_IT_OFF", "DOUBT", "TAUNT", "DOUBT" }, Names(s));
    }

    // ---- Screens that disagree --------------------------------------------------------------------------

    [Fact]
    public void AScreenListingOtherCardsThanTheGamesIsMatchedByCardIdAndTheDifferenceIsNoted()
    {
        var e = AtMapWithDeck(SixteenCards);
        var s = e.State;
        Assert.Equal(RunFollowUp.TransformSelect, RunNonCombatEffects.ApplyRelicPickup(s, RunConstants.RelicPomander, SelectionReturn.Map));
        s.Phase = RunPhase.TransformSelect;
        var choice = TheGamesChoice(e, out var listed, out _);
        choice["options"]!.AsArray().RemoveAt(5);  // the game's screen did not have a card the emulator's has; it took the first

        string? gap = OldSetup.AnswerDeckSelection(e, new List<JsonObject> { choice }, out var note);

        Assert.Null(gap);
        Assert.NotNull(note);
        Assert.Contains($"lists {listed.Length} cards and the game's screen {listed.Length - 1}", note);
        Assert.Contains("matched by card id", note);
        Assert.NotEqual(RunPhase.TransformSelect, s.Phase);
        Assert.True(s.Deck[0].Upgraded);  // BASH, which the game's pick names
        Assert.Equal(1, s.Deck.Count(c => c.Upgraded));
    }

    [Fact]
    public void ACardTheGamePickedThatTheEmulatorsScreenDoesNotHaveIsAMirrorGap()
    {
        var e = AtMap();
        var s = e.State;
        Assert.Equal(RunFollowUp.TransformSelect, RunNonCombatEffects.ApplyRelicPickup(s, RunConstants.RelicPomander, SelectionReturn.Map));
        s.Phase = RunPhase.TransformSelect;
        var choice = TheGamesChoice(e, out _, out _);
        choice["options"]![0] = "ANGER";  // the game listed, and took, a card the emulator's deck does not have

        string? gap = OldSetup.AnswerDeckSelection(e, new List<JsonObject> { choice });

        Assert.NotNull(gap);
        Assert.NotEqual(RunPhase.TransformSelect, s.Phase);  // the run still walks on, so the fight can be replayed
    }

    [Fact]
    public void ACardTheGamePickedByIdThatTheEmulatorHoldsInDifferentCopiesOnADifferentListIsAMirrorGap()
    {
        // Dolly's Mirror offers every card; the emulator holds the card as BASH+ and as BASH, the game's record
        // says only "BASH", and the lists differ, so which copy the game took cannot be told.
        var e = AtMapWithDeck("TAUNT", "STRIKE_IRONCLAD");
        var s = e.State;
        RunNonCombatEffects.AddCardToDeck(s, new CardInstance(OldIds.CardIdBySnake("BASH"), Upgraded: true));
        RunNonCombatEffects.AddCardToDeck(s, new CardInstance(OldIds.CardIdBySnake("BASH"), Upgraded: false));
        Assert.Equal(RunFollowUp.TransformSelect, RunNonCombatEffects.ApplyRelicPickup(s, RelicEffects.DollysMirror, SelectionReturn.Map));
        s.Phase = RunPhase.TransformSelect;
        var choice = Choice(1, 1, new[] { "BASH", "TAUNT" }, 0);

        string? gap = OldSetup.AnswerDeckSelection(e, new List<JsonObject> { choice }, out var note);

        Assert.NotNull(gap);
        Assert.Contains("ambiguous", gap);
        Assert.Null(note);  // a gap carries its own reason
        Assert.NotEqual(RunPhase.TransformSelect, s.Phase);
    }

    [Fact]
    public void ACardTheGamePickedByIdThatTheEmulatorHoldsInIdenticalCopiesOnADifferentListIsNoGap()
    {
        var e = AtMapWithDeck("TAUNT", "BASH", "BASH");
        var s = e.State;
        Assert.Equal(RunFollowUp.TransformSelect, RunNonCombatEffects.ApplyRelicPickup(s, RelicEffects.DollysMirror, SelectionReturn.Map));
        s.Phase = RunPhase.TransformSelect;
        var choice = Choice(1, 1, new[] { "BASH", "TAUNT" }, 0);

        string? gap = OldSetup.AnswerDeckSelection(e, new List<JsonObject> { choice }, out var note);

        Assert.Null(gap);
        Assert.NotNull(note);
        Assert.Equal(3, s.Deck.Count(c => c.DefId == OldIds.CardIdBySnake("BASH")));
    }

    [Fact]
    public void AnOptionalScreenTheGameNeverShowedIsClosedWithDoneAndNoted()
    {
        var e = AtMap();
        var s = e.State;
        Assert.Equal(RunFollowUp.TransformSelect, RunNonCombatEffects.ApplyRelicPickup(s, RelicEffects.Kifuda, SelectionReturn.Map));
        s.Phase = RunPhase.TransformSelect;

        string? gap = OldSetup.AnswerDeckSelection(e, new List<JsonObject>(), out var note);

        // The game took nothing and so does the emulator: the state is the game's, and the item is no gap.
        Assert.Null(gap);
        Assert.NotNull(note);
        Assert.Contains("did not", note);
        Assert.NotEqual(RunPhase.TransformSelect, s.Phase);
        Assert.DoesNotContain(s.Deck, c => c.Enchantment == Enchantment.Adroit);
    }
}
