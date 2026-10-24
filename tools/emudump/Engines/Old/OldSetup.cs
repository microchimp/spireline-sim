// The old engine's side of the setup mirror (referee/driver/Setup.cs): what each kind of setup item does to the run
// engine, behind the IEngine members Setup.cs calls, and the id, belt and counter helpers the old engine's dumps
// use. The items, their order and their log lines are Setup.cs's (engine-neutral); a wrong mirror shows at step 0.
// An item the emulator cannot build comes back "unsupported": a mirror gap, not an emulator divergence.
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using EmuDump.Engine;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;

namespace EmuDump.Engines.Old;

static class OldSetup
{
    static string Snake(string name) => Regex.Replace(name, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

    /// <summary>Emulator relic id for a game relic entry, or null.</summary>
    internal static int? RelicId(string entry)
    {
        foreach (var r in Sts2Emulator.GeneratedData.Relics.All)
        {
            if (r.Entry == entry) return r.Id;
        }

        return null;
    }

    static Dictionary<string, int>? _potions;
    static Dictionary<int, string>? _potionNames;

    /// <summary>Emulator potion id for a game potion entry (FIRE_POTION), or null; through data/id_map.json.</summary>
    internal static int? PotionId(string entry)
    {
        LoadPotions();
        return _potions!.TryGetValue(entry, out int v) ? v : null;
    }

    static void LoadPotions()
    {
        if (_potions != null) return;
        var map = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))!["potions"]!.AsObject();
        _potions = map.ToDictionary(kv => Snake(kv.Key), kv => (int)kv.Value!);
        _potionNames = _potions.ToDictionary(kv => kv.Value, kv => kv.Key);
    }

    /// <summary>Game potion entry of an emulator potion id.</summary>
    internal static string PotionEntry(int id)
    {
        LoadPotions();
        return _potionNames!.TryGetValue(id, out var n) ? n : id.ToString();
    }

    /// <summary>The potion belt as the game's snapshot writes it: one entry per slot, null when empty.</summary>
    internal static JsonArray Belt(int[] slots, int max) =>
        new(slots.Take(Math.Min(max, slots.Length)).Select(p => p == 0 ? null : (JsonNode)PotionEntry(p)).ToArray());

    /// <summary>The run's RNG streams' call counts under the game's stream names (the game's snapshot "rng").</summary>
    internal static JsonObject RunRng(RunState s)
    {
        var o = new JsonObject();
        foreach (var p in s.Rng.GetType().GetProperties().Where(p => p.PropertyType == typeof(GameRng)))
        {
            o[p.Name == "CombatOrbs" ? "CombatOrbGeneration" : p.Name] = ((GameRng)p.GetValue(s.Rng)!).CallCount;
        }

        return o;
    }

    /// <summary>Game relic entry (BURNING_BLOOD) of an emulator relic id.</summary>
    internal static string RelicEntry(int id)
    {
        foreach (var r in Sts2Emulator.GeneratedData.Relics.All)
        {
            if (r.Id == id) return r.Entry;
        }

        return id.ToString();
    }

    internal static JsonObject Counters(RunState s)
    {
        var o = RunRng(s);

        var pl = new JsonObject();
        foreach (var p in s.PlayerRng.GetType().GetProperties().Where(p => p.PropertyType == typeof(GameRng)))
        {
            pl[p.Name] = ((GameRng)p.GetValue(s.PlayerRng)!).CallCount;
        }

        return new JsonObject { ["rng"] = o, ["player_rng"] = pl };
    }

    static SetupResult Ok(string? note = null) => new(true, note);

    static SetupResult No(string reason) => new(false, reason);

    /// <summary>IEngine.ObtainRelic: the relic's pickup on the map, then the selection it opens answered with the game's picks.</summary>
    internal static SetupResult Relic(RunEngine e, string id, List<JsonObject> choices)
    {
        var s = e.State;
        if (s.ActiveCombat != null) return No("unsupported: relic in a fight");
        int relic = RelicId(id) ?? throw new KeyNotFoundException("no emulator relic " + id);
        var followUp = RunNonCombatEffects.ApplyRelicPickup(s, relic, SelectionReturn.Map);
        if (followUp == RunFollowUp.TransformSelect)
        {
            s.Phase = RunPhase.TransformSelect;
        }
        else if (followUp != RunFollowUp.None)
        {
            return No("unsupported: relic follow-up " + followUp);
        }

        string? gap = AnswerDeckSelection(e, choices, out var note);
        return gap == null ? Ok(note == null ? null : "note: " + note) : No("unsupported: " + gap);
    }

    /// <summary>IEngine.ProcurePotion: the first free slot of the belt, in a fight or on the map.</summary>
    internal static SetupResult Potion(RunEngine e, string id)
    {
        var s = e.State;
        var c = s.ActiveCombat;
        int potion = PotionId(id) ?? throw new KeyNotFoundException("no emulator potion " + id);
        bool ok = c != null ? AddToCombatBelt(c, potion) : RunRewardGenerator.AddPotion(s, potion);
        return new(ok, ok ? null : "refused");
    }

    /// <summary>IEngine.AddCard: the deck on the map, a combat pile in a fight.</summary>
    internal static SetupResult Card(RunEngine e, string id, bool upgraded, string pile, bool onTop)
    {
        var s = e.State;
        var c = s.ActiveCombat;
        var card = new CardInstance(OldIds.CardIdBySnake(id), Upgraded: upgraded);
        if (c == null)
        {
            if (pile != "deck") return No("unsupported: card into " + pile + " on the map");
            RunNonCombatEffects.AddCardToDeck(s, card);
            return Ok();
        }

        // In a fight, the referee's CardPileCmd.Add of a card made in the combat's scope: the hand and the
        // discard pile take it at the end, the draw pile at the bottom or, with "pos": "top", on top (the
        // next card drawn). No hook runs for it here; a card that reacts to entering a pile is a mirror gap.
        card = c.Fresh(card);  // a card made in the combat's scope enters it with a fresh uid
        switch (pile)
        {
            case "hand":
                c.Hand.Add(card);
                return Ok();
            case "discard":
                c.DiscardPile.Add(card);
                return Ok();
            case "draw":
                if (onTop) c.TopDeck(card);
                else c.BottomDeck(card);
                return Ok();
            default:
                return No("unsupported: card into " + pile + " in a fight");
        }
    }

    /// <summary>IEngine.ApplyPower.</summary>
    internal static SetupResult Power(RunEngine e, string id, int targetSlot, int amount)
    {
        var c = e.State.ActiveCombat;
        if (c == null) return No("unsupported: power outside a fight");
        var buff = OldAnchor.BuffFor(id) ?? throw new KeyNotFoundException("no emulator buff for " + id);
        bool ok = ConsoleSetup.ApplyPowerWithoutApplier(c, targetSlot, buff, amount);
        return new(ok, ok ? null : "no living target");
    }

    /// <summary>IEngine.GainEnergy.</summary>
    internal static SetupResult Energy(RunEngine e, int amount)
    {
        var c = e.State.ActiveCombat;
        if (c == null) return No("unsupported: energy outside a fight");
        Sts2Emulator.Core.Effects.CardEffects.GainEnergy(c, amount);
        return Ok();
    }

    /// <summary>IEngine.SetHp: CreatureCmd.SetMaxHp, then SetCurrentHp, on the map.</summary>
    internal static SetupResult Hp(RunEngine e, int? maxHp, int? currentHp)
    {
        var s = e.State;
        if (s.ActiveCombat != null) return No("hp: map only");
        int max = maxHp ?? s.PlayerMaxHp;
        int cur = Math.Min(currentHp ?? s.PlayerHp, max);
        if (max < 1 || cur < 1) return No("hp below 1");
        s.PlayerMaxHp = max;
        s.PlayerHp = cur;
        return Ok();
    }

    /// <summary>IEngine.SetBlock: the player's block field written as it stands (no hook, no block modifier runs).</summary>
    internal static SetupResult Block(RunEngine e, int amount)
    {
        var c = e.State.ActiveCombat;
        if (c == null) return No("unsupported: block outside a fight");
        if (amount < 0) return No($"block {amount} below 0");
        c.PlayerBlock = amount;
        return Ok();
    }

    /// <summary>
    /// IEngine.SetEnemyHp: the enemy's max HP and HP fields written as they stand (no hook runs, the intent already rolled stays).
    /// The slot is an index into the combat's enemies, dead ones included, as ConsoleSetup.ApplyPowerWithoutApplier counts it.
    /// </summary>
    internal static SetupResult EnemyHp(RunEngine e, int slot, int? maxHp, int? currentHp)
    {
        var c = e.State.ActiveCombat;
        if (c == null) return No("unsupported: enemy_hp outside a fight");
        if (LivingEnemy(c, slot) is not { } en) return No($"no living enemy in slot {slot}");
        int max = maxHp ?? en.MaxHp;
        if (max < 1) return No($"enemy {slot}: max HP {max} below 1");
        if (currentHp is { } cur)
        {
            if (cur < 1) return No($"enemy {slot}: current HP {cur} below 1");
            if (cur > max) return No($"enemy {slot}: current HP {cur} above max HP {max}");
        }

        en.MaxHp = max;
        en.Hp = Math.Min(currentHp ?? en.Hp, max);
        return Ok();
    }

    /// <summary>IEngine.SetEnemyBlock: the enemy's block field written as it stands (no hook runs).</summary>
    internal static SetupResult EnemyBlock(RunEngine e, int slot, int amount)
    {
        var c = e.State.ActiveCombat;
        if (c == null) return No("unsupported: enemy_block outside a fight");
        if (LivingEnemy(c, slot) is not { } en) return No($"no living enemy in slot {slot}");
        if (amount < 0) return No($"enemy {slot}: block {amount} below 0");
        en.Block = amount;
        return Ok();
    }

    /// <summary>The living enemy in a slot, or null when the slot is out of range or its enemy is dead.</summary>
    static EnemyState? LivingEnemy(CombatState c, int slot) =>
        slot >= 0 && slot < c.Enemies.Count && c.Enemies[slot].Hp > 0 ? c.Enemies[slot] : null;

    /// <summary>IEngine.RefsetItem: RefsetSetup's items (relics_set, deck_set, potions_set, player_load, act), or null for any other.</summary>
    internal static SetupResult? Refset(RunEngine e, JsonObject item, List<JsonObject> choices) =>
        RefsetSetup.Item(e, item, choices) is { } r ? new SetupResult(r.ok, r.msg) : null;

    /// <summary>PotionCmd.TryToProcure in a fight: Sozu refuses, else the first free slot.</summary>
    static bool AddToCombatBelt(CombatState c, int potion)
    {
        if (c.Relics.Any(r => r.DefId == Sts2Emulator.Core.Effects.RelicEffects.Sozu)) return false;
        for (int i = 0; i < Math.Min(c.MaxPotionSlots, c.PotionSlots.Length); i++)
        {
            if (c.PotionSlots[i] != 0) continue;
            c.PotionSlots[i] = potion;
            return true;
        }

        return false;
    }

    /// <summary>A screen an item opened, answered with the first option as the referee's selector does.</summary>
    static void AnswerScreens(RunEngine e)
    {
        var s = e.State;
        int guard = 0;
        while (s.Phase != RunPhase.Map && s.Phase != RunPhase.Combat && guard++ < 10)
        {
            e.Step(0, -1, out _, out _, out _);
            if (s.Phase == RunPhase.BundleSelect)
            {
                e.Step(RunConstants.BundleConfirmAction, -1, out _, out _, out _);
            }
        }
    }

    /// <summary>
    /// A selection a relic's pickup opened, answered with the picks the game's selector made. The game's
    /// selector is shown the eligible cards in the order CardSelectCmd lists them (deck order for upgrades,
    /// transforms, enchantments and Dolly's Mirror; a removal sorts every Curse first, then deck order, see
    /// <see cref="InGameOrder"/>) and takes the first n, so the game's pick is a position in that list. When
    /// the emulator's screen lists the same cards in that order, the pick is the card at that position,
    /// followed through the removals and transforms earlier picks made. When the lists differ (the emulator's
    /// run already holds another deck than the game's) the pick is matched by card id, as it always was, and
    /// the difference comes back in <paramref name="note"/>; that is no gap as long as the id names one kind
    /// of card (the copies the emulator offers do not differ in upgrade or enchantment). The game closes a
    /// screen that asked for zero to N cards with the picks it made, so an optional emulator screen is closed
    /// with DONE once the game's picks run out, and one the game never showed is closed with DONE and a note.
    /// Returns a reason when the two screens cannot be told apart (the emulator wants more picks than the
    /// game's screen required, a pick is not selectable, or an id names different copies on lists that
    /// differ); the run is then walked on with the first selectable cards so the fight can still be
    /// replayed, and the item counts as a mirror gap.
    /// </summary>
    internal static string? AnswerDeckSelection(RunEngine e, List<JsonObject> choices) => AnswerDeckSelection(e, choices, out _);

    /// <inheritdoc cref="AnswerDeckSelection(RunEngine, List{JsonObject})"/>
    /// <param name="e">The run, in the selection's phase.</param>
    /// <param name="choices">The game's card choices tagged with the item.</param>
    /// <param name="note">What differed between the two screens without making the item a gap, or null.</param>
    internal static string? AnswerDeckSelection(RunEngine e, List<JsonObject> choices, out string? note)
    {
        var s = e.State;
        string? gap = null;
        note = null;
        var mask = new int[4096];
        // A grid of offered cards is indexed by the offer, any other selection by the deck.
        bool Grid() => s.PendingOfferCards.Length > 0;
        int IdAt(int i) => Grid() ? Math.Abs(s.PendingOfferCards[i]) : s.Deck[i].DefId;
        int[] Selectable()
        {
            e.WriteActionMask(mask);
            int n = Grid() ? s.PendingOfferCards.Length : s.Deck.Count;
            return Enumerable.Range(0, n).Where(i => mask[i] != 0).ToArray();
        }

        var want = new Queue<int>();
        foreach (var c in choices)
        {
            var opts = c["options"]!.AsArray().Select(x => (string)x!).ToList();
            foreach (var p in c["picked"]!.AsArray()) want.Enqueue(OldIds.CardIdBySnake(opts[(int)p!]));
        }

        // One game screen over a deck selection: follow each pick as a position among the cards listed.
        List<int>? at = null;      // the deck index each listed card has now (-1: taken)
        Queue<int>? picked = null; // the listed positions the game took, in its order
        bool differs = false;      // the two screens list other cards: the picks are matched by card id
        if (choices.Count == 1 && s.Phase == RunPhase.TransformSelect && !Grid() && s.PendingSelectionKind != DeckSelection.None)
        {
            var listed = InGameOrder(s, Selectable());
            var theirs = choices[0]["options"]!.AsArray().Select(x => OldIds.CardIdBySnake((string)x!)).ToList();
            var mine = listed.Select(IdAt).ToList();
            if (mine.SequenceEqual(theirs))
            {
                at = listed.ToList();
                picked = new Queue<int>(choices[0]["picked"]!.AsArray().Select(x => (int)x!));
            }
            else
            {
                differs = true;
                int common = Math.Min(mine.Count, theirs.Count);
                int pos = Enumerable.Range(0, common).Where(k => mine[k] != theirs[k]).DefaultIfEmpty(common).First();
                string At(List<int> ids) => pos < ids.Count ? RunSnap.CardEntry(ids[pos]) : "nothing";
                note = $"the emulator's selection lists {mine.Count} cards and the game's screen {theirs.Count}, first different at position {pos} ({At(mine)} against {At(theirs)}); the picks were matched by card id";
            }
        }

        int guard = 0;
        while (s.Phase == RunPhase.TransformSelect && guard++ < 64)
        {
            var ok = Selectable();
            if (ok.Length == 0) { gap ??= "selection with nothing selectable"; break; }
            int pick;
            int listedPos = -1;
            bool optional = Grid() ? s.PendingOfferOptional : s.PendingSelectionOptional;
            if (picked != null && picked.Count > 0)
            {
                listedPos = picked.Dequeue();
                pick = listedPos < at!.Count ? at[listedPos] : -1;
                if (!ok.Contains(pick)) { gap ??= $"the card at position {listedPos} the game picked is not selectable here"; pick = ok[0]; listedPos = -1; }
                want.Dequeue();
            }
            else if (picked == null && want.Count > 0)
            {
                int id = want.Dequeue();
                var named = ok.Where(i => IdAt(i) == id).ToList();
                pick = named.Count > 0 ? named[0] : -1;
                if (pick < 0) { gap ??= $"card {id} the game picked is not selectable here"; pick = ok[0]; }
                // The game's record names the card by id only: copies that differ (upgrade, enchantment) cannot be told apart.
                else if (differs && named.Any(i => s.Deck[i] != s.Deck[pick])) gap ??= $"card {id} the game picked is ambiguous among the emulator's copies, and {note}";
            }
            else if (optional)
            {
                pick = Grid() ? s.PendingOfferCards.Length : s.Deck.Count;  // DONE: the game took no more
                if (choices.Count == 0) note ??= "the emulator offered a selection the game did not; closed with DONE";
            }
            else
            {
                if (choices.Count > 0) gap ??= "the emulator's selection wants more cards than the game's screen required";
                pick = ok[0];
            }

            bool removes = s.PendingSelectionKind is not (DeckSelection.Enchant or DeckSelection.Upgrade or DeckSelection.Duplicate);
            e.Step(pick, -1, out _, out _, out _);
            if (at != null && listedPos >= 0)
            {
                // A removal or transform takes the card out of its place (a transform adds the new card at the end).
                int taken = at[listedPos];
                at[listedPos] = -1;
                if (removes) for (int k = 0; k < at.Count; k++) if (at[k] > taken) at[k]--;
            }
        }

        if (want.Count > 0) gap ??= $"the game picked {want.Count} more cards than the emulator's selection took";
        AnswerScreens(e);
        if (gap != null) note = null;  // the item is a gap: its reason says it all
        return gap;
    }

    /// <summary>
    /// The deck indices of the cards a selection offers, in the order the game's selector is shown them.
    /// CardSelectCmd lists the eligible cards in deck order for every selection except a removal:
    /// FromDeckForRemoval hands FromDeckGeneric a sorting order that puts every Curse first and the rest after
    /// it in deck order (the sort is stable, so the curses keep their deck order too). The removal kinds are
    /// Remove (Biiig Hug, Empty Cage, Precarious Shears, Precise Scissors, Preserved Fog), RemoveUpgradable
    /// (Pael's Tooth) and RemoveTaggedBasic (Amalgamator); the last two offer no curse, but they go through
    /// the same sort. Upgrade (FromDeckForUpgrade), the transforms (FromDeckForTransformation), Enchant
    /// (FromDeckForEnchantment sorts its cards by deck index) and Duplicate (FromDeckGeneric, no sorting
    /// order) list in deck order, which is the order <paramref name="byDeck"/> comes in.
    /// </summary>
    internal static int[] InGameOrder(RunState s, int[] byDeck)
    {
        if (s.PendingSelectionKind is not (DeckSelection.Remove or DeckSelection.RemoveUpgradable or DeckSelection.RemoveTaggedBasic))
        {
            return byDeck;
        }

        bool Curse(int i) => Sts2Emulator.GeneratedData.Cards.Get(s.Deck[i].DefId).Type == CardType.Curse;
        return byDeck.Where(Curse).Concat(byDeck.Where(i => !Curse(i))).ToArray();
    }

    /// <summary>
    /// For the scenario generator (IEngine.SetupInfo): the follow-up each relic's pickup opens on the map (None and
    /// TransformSelect are mirrored; any other is a mirror gap), probed on a fresh run past Neow, and the power keys
    /// (<paramref name="keys"/>: IEngine.PowerKeys).
    /// </summary>
    internal static JsonObject Info(JsonObject keys)
    {
        var relics = new JsonObject();
        foreach (var r in Sts2Emulator.GeneratedData.Relics.All)
        {
            var e = new RunEngine();
            e.Reset("SETUPINFO1", 10);
            AnswerScreens(e);
            string followUp;
            try
            {
                followUp = RunNonCombatEffects.ApplyRelicPickup(e.State, r.Id, SelectionReturn.Map).ToString();
            }
            catch (Exception ex)
            {
                followUp = "error: " + ex.GetType().Name;
            }

            relics[r.Entry] = followUp;
        }

        return new JsonObject { ["relics"] = relics, ["buff_keys"] = keys["buff_keys"]!.DeepClone(), ["internal_keys"] = keys["internal_keys"]!.DeepClone() };
    }

    /// <summary>
    /// IEngine.EnterEncounter, the fight item: the same map option the referee walked onto, meeting the encounter the
    /// game entered, found by its place in the game's act lists (act, kind, index) and taken from the emulator's own
    /// copy of that list (tools/fidelity/table_check.py checks the two agree). Returns the encounter's name.
    /// </summary>
    internal static string EnterEncounter(RunEngine e, int option, string actName, string kind, int index, string gameId)
    {
        int act = Tables.ActIds.First(a => a.Name == actName).Act;
        var list = Tables.EncounterList(act, kind);
        if (index >= list.Length) throw new InvalidOperationException($"{actName} {kind} has {list.Length} encounters, the game's {gameId} is at {index}");
        int encounter = list[index];
        int node = kind switch { "elite" => RunConstants.NodeElite, "boss" => RunConstants.NodeBoss, _ => RunConstants.NodeNormal };
        if (e.EnterEncounterDebug(option, encounter, node) != 0)
            throw new InvalidOperationException("EnterEncounterDebug refused option " + option);
        return ((CombatFactory.ActOneEncounter)encounter).ToString();
    }
}
