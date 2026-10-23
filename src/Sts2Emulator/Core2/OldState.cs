// The kernel's one view of the old engine's state (docs/rebuild/TIER1_KERNEL.md section 3): the game's listener lists
// enumerated over CombatState and RunState. No other kernel file knows the old representation. Game code is cited by
// class and method only.
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace Sts2Emulator.Core2;

/// <summary>What one BuffId is in the game's power list.</summary>
public enum PowerMapKind : byte
{
    /// <summary>One game power (by its own name, or through power_aliases.json "emu_rename").</summary>
    Game,
    /// <summary>A piece of a game power the engine keeps in several buffs (InfernoSelfDamage of Inferno).</summary>
    Piece,
    /// <summary>Several game powers summed into one buff (power_aliases.json "folds"): NotModelled.</summary>
    Folded,
    /// <summary>Engine bookkeeping that is no game power (a stun, a counter, a marker): never a listener. Only the BuffIds
    /// <c>OldModelIds</c> names in its bookkeeping list are this; no rule falls through to it.</summary>
    EngineOnly,

    /// <summary>A BuffId no rule maps to a game power and the bookkeeping list does not name: it may stand for a game
    /// power listener the mapping has not identified, so a pass with any power handler reports NotModelled.</summary>
    NotModelled,
}

/// <summary>A BuffId's game power: the kind of mapping, the power's id in data/id_map.json "powers" (0 when none) and
/// its key (UPPER_SNAKE without _POWER).</summary>
public readonly record struct PowerMapping(PowerMapKind Kind, int ModelId, string GameKey);

/// <summary>
/// The model ids the kernel keys listeners by: the values of data/id_map.json. For cards, relics, potions and monsters
/// those are the old engine's def ids; powers have their own numbering there, mapped from BuffId by the rule of
/// tools/emudump OldAnchor (UPPER_SNAKE name without _POWER; "emu_rename" first; a fold target is Folded; a buff whose
/// name starts with a game power's name and "_" is a piece of that power; a BuffId named in the bookkeeping list is
/// engine-only; anything else is NotModelled, never silently engine-only). A fold target or member of power_aliases.json
/// is Folded.
/// Orbs and enchantments use the old enums, afflictions <see cref="GameAffliction" />.
/// </summary>
public sealed class OldModelIds
{
    // The BuffIds that are engine bookkeeping with no game power of their own (no class of that name in the game's
    // powers; the comments of BuffState.cs say what each counts). A BuffId is engine-only only by being named here.
    private static readonly HashSet<BuffId> Bookkeeping =
    [
        BuffId.Stunned, BuffId.SummonCooldown, BuffId.TerrorQueued, BuffId.BackupCount, BuffId.Reviving,
        BuffId.FranticEscapePlayedCountUnused,
    ];

    // A game power the engine keeps under another buff that power_aliases.json "emu_rename" does not list yet:
    // AccuracyPower is the ShivDamage buff (sts2_gym.live.powers BUFF_ALIASES). Kept here, not in that file, so
    // fight_compare and Reanchor, which read it, are unchanged in this round.
    private static readonly Dictionary<string, string> RenamesNotInAliases = new() { ["SHIV_DAMAGE"] = "ACCURACY" };
    private readonly Dictionary<BuffId, PowerMapping> _powers = new();
    private readonly Dictionary<int, string> _powerNames = new();

    /// <param name="idMap">data/id_map.json parsed.</param>
    /// <param name="aliases">tools/fidelity/power_aliases.json parsed.</param>
    public OldModelIds(JsonObject idMap, JsonObject aliases)
    {
        var game = new Dictionary<string, int>();
        foreach (var (name, id) in idMap["powers"]!.AsObject())
        {
            game[PowerKey(name)] = (int)id!;
            _powerNames[(int)id!] = name;
        }

        var rename = aliases["emu_rename"]!.AsObject().ToDictionary(kv => kv.Key, kv => (string)kv.Value!);
        foreach (var (emu, gameKey) in RenamesNotInAliases)
        {
            rename.TryAdd(emu, gameKey);
        }

        var foldTargets = aliases["folds"]!.AsObject().Select(kv => kv.Key).ToHashSet();
        var foldMembers = aliases["folds"]!.AsObject().SelectMany(kv => kv.Value!.AsArray().Select(x => (string)x!))
            .ToHashSet();
        OstyModelId = (int)idMap["enemies"]!["Osty"]!;
        foreach (var id in Enum.GetValues<BuffId>())
        {
            string key = PowerKey(id.ToString());
            PowerMapping m;
            if (Bookkeeping.Contains(id))
            {
                m = new(PowerMapKind.EngineOnly, 0, key);
            }
            else if (foldTargets.Contains(key) || foldMembers.Contains(key))
            {
                m = new(PowerMapKind.Folded, game.GetValueOrDefault(key), key);
            }
            else if (game.TryGetValue(rename.GetValueOrDefault(key, key), out int model))
            {
                m = new(PowerMapKind.Game, model, rename.GetValueOrDefault(key, key));
            }
            else if (game.Keys.Where(p => key.StartsWith(p + "_", StringComparison.Ordinal))
                         .OrderByDescending(p => p.Length).FirstOrDefault() is { } owner)
            {
                m = new(PowerMapKind.Piece, game[owner], owner);
            }
            else
            {
                m = new(PowerMapKind.NotModelled, 0, key);
            }

            _powers[id] = m;
        }
    }

    /// <summary>The tables read from data/id_map.json and tools/fidelity/power_aliases.json.</summary>
    public static OldModelIds Load(string idMapPath, string aliasesPath) =>
        new(JsonNode.Parse(File.ReadAllText(idMapPath))!.AsObject(),
            JsonNode.Parse(File.ReadAllText(aliasesPath))!.AsObject());

    /// <summary>Osty's monster id (data/id_map.json "enemies").</summary>
    public int OstyModelId { get; }

    public PowerMapping Power(BuffId id) => _powers.TryGetValue(id, out var m) ? m : new(PowerMapKind.NotModelled, 0, "");

    /// <summary>The game's power class name of a power model id, or null.</summary>
    public string? PowerName(int modelId) => _powerNames.GetValueOrDefault(modelId);

    internal static string PowerKey(string name)
    {
        string n = Regex.Replace(name, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();
        return n.EndsWith("_POWER", StringComparison.Ordinal) ? n[..^"_POWER".Length] : n;
    }
}

/// <summary>
/// The old engine's state as the kernel reads it, plus what only the kernel's own sites know: the card in flight, the
/// combat setup, creatures between death and removal, and the run models the old state has none of.
/// </summary>
public sealed class OldWorld(CombatState? combat, RunState? run, OldModelIds ids)
{
    public CombatState? Combat { get; set; } = combat;
    public RunState? Run { get; set; } = run;
    public OldModelIds Ids { get; } = ids;

    private readonly HashSet<EnemyState> _dying = new(ReferenceEqualityComparer.Instance);

    // The marks below are what only the kernel's own sites know. Each has a scoped setter (a K3 obligation: the site
    // that opens the window sets it, with a test); a mark nobody sets leaves the default, which drops or adds
    // listeners silently unless the state shows the mark is missing (Listeners.cs reports those as NotModelled).

    /// <summary>The card being played: the game's Play pile, set by <see cref="Flying" /> for the length of a play (the
    /// old engine holds the card in a local while its effects run). Not detectable when missing.</summary>
    public CardInstance? InFlight { get; private set; }

    /// <summary>Combat setup (the game's IsStarting): the guarded iterator's exemption, set by <see cref="Setup" />.
    /// Detectable when missing: a guarded pass whose guard is set before turn 1 reports NotModelled.</summary>
    public bool InSetup { get; private set; }

    /// <summary>The player's hooks as the kill sequence declared them (see <see cref="Core2.PlayerHooks" />). Detectable
    /// when missing: the player down with no declaration reports NotModelled.</summary>
    public PlayerHooks PlayerHooks { get; private set; }

    /// <summary>Enemies dead but not yet removed (inside CreatureCmd.Kill, before the removal): still in the combat. Set
    /// by <see cref="MarkDying" />. Not detectable when missing: a dead enemy left in the roster looks the same after
    /// its removal.</summary>
    public bool IsDying(EnemyState enemy) => _dying.Contains(enemy);

    /// <summary>Marks <paramref name="card" /> as the Play pile's card until the scope is disposed.</summary>
    public IDisposable Flying(CardInstance card)
    {
        var before = InFlight;
        InFlight = card;
        return new Scope(() => InFlight = before);
    }

    /// <summary>Marks the combat as starting (IsStarting) until the scope is disposed.</summary>
    public IDisposable Setup()
    {
        bool before = InSetup;
        InSetup = true;
        return new Scope(() => InSetup = before);
    }

    /// <summary>Marks <paramref name="enemy" /> as dead but not yet removed until the scope is disposed.</summary>
    public IDisposable MarkDying(EnemyState enemy)
    {
        bool added = _dying.Add(enemy);
        return new Scope(() =>
        {
            if (added)
            {
                _dying.Remove(enemy);
            }
        });
    }

    /// <summary>Declares the player's hooks deactivated (CreatureCmd.Kill's DeactivateHooks has run) until the scope is
    /// disposed.</summary>
    public IDisposable DeactivatePlayerHooks() => SetHooks(PlayerHooks.Deactivated);

    /// <summary>Declares the player's hooks still active until the scope is disposed: the death passes of the player's
    /// own kill (death prevention, AfterDeath) run before DeactivateHooks, with the player at 0 HP.</summary>
    public IDisposable KeepPlayerHooks() => SetHooks(PlayerHooks.Active);

    private Scope SetHooks(PlayerHooks to)
    {
        var before = PlayerHooks;
        PlayerHooks = to;
        return new Scope(() => PlayerHooks = before);
    }

    private sealed class Scope(Action undo) : IDisposable
    {
        private Action? _undo = undo;

        public void Dispose()
        {
            _undo?.Invoke();
            _undo = null;
        }
    }

    /// <summary>The combat's modifiers, badges and multiplayer scaling model by model id (the old state has none).</summary>
    public List<int> Modifiers { get; } = [];
    public List<int> Badges { get; } = [];
    public int? Scaling { get; set; }
}

/// <summary>What the kill sequence declared of the player's hooks. Undeclared counts as active (the game deactivates a
/// creature's hooks only after its death-prevention hooks and AfterDeath), but with the player down it is reported.</summary>
public enum PlayerHooks : byte { Undeclared, Active, Deactivated }

/// <summary>Where a card listener sits: the game's piles in their order, and the run deck.</summary>
public enum OldPile : sbyte { None = -1, Hand, Draw, Discard, Exhaust, Play, Deck }

/// <summary>
/// One entry of a listener list over the old state: who (kind, model id, instance), the creature it sits on (game
/// order: the player 0, then pets, then enemies; -1 for modifiers and run models), its index within its kind (power:
/// application order, relic: obtain order, potion: slot, orb: queue position, card: position in its pile, deck card:
/// deck index), its pile, the buff that stands for a power and the enemy object of an enemy's power or monster.
/// </summary>
public readonly record struct OldListener(ListenerRef Who, int Creature, int Index, OldPile Pile, BuffId Buff,
    EnemyState? Enemy)
{
    public ListenerKind Kind => (ListenerKind)Who.Kind;
}

/// <summary>The game's listener lists over the old state, and its membership rule (CombatState.Contains).</summary>
public static class OldState
{
    /// <summary>A power's instance field: its creature and the instance among that creature's powers of its model.</summary>
    public static int PowerInstance(int creature, int instance) => (creature << 8) | instance;

    /// <summary>A run deck card's instance field: negative, so it never meets a combat card's uid.</summary>
    public static int DeckInstance(int deckIndex) => -(deckIndex + 1);

    /// <summary>
    /// game: CombatState.IterateHookListeners. Per creature (allies then enemies in roster order) its powers in
    /// application order; a creature without a player then lists its monster; the player, when active for hooks, its
    /// relics in obtain order, potions by slot, orbs in queue order and the cards of Hand, Draw, Discard, Exhaust and
    /// Play, each followed by its affliction and enchantment. Then modifiers, badges and the scaling model.
    /// </summary>
    public static void Combat(OldWorld w, List<OldListener> into, List<OldGap> gaps)
    {
        var s = w.Combat ?? throw new InvalidOperationException("no combat");
        int c = 0;
        PlayerGaps(w, gaps);
        Powers(w, s.PlayerBuffs, c, null, into, gaps);
        if (PlayerActive(w))
        {
            for (int i = 0; i < s.Relics.Count; i++)
            {
                into.Add(new(new((int)ListenerKind.Relic, s.Relics[i].DefId, i), c, i, OldPile.None, default, null));
            }

            Potions(s.PotionSlots, c, into);
            for (int i = 0; i < s.Orbs.Count; i++)
            {
                into.Add(new(new((int)ListenerKind.Orb, (int)s.Orbs[i].Type, i), c, i, OldPile.None, default, null));
            }

            Piles(w, s, c, into, gaps);
        }

        c++;
        if (s.OstyMaxHp > 0)
        {
            // A pet: its powers (the old state holds none), then, having no player, its monster.
            gaps.Add(new(NotModelledKind.OstyPowers, ListenerKind.Power, c, 0, "Osty"));
            into.Add(new(new((int)ListenerKind.Monster, w.Ids.OstyModelId, c), c, 0, OldPile.None, default, null));
            c++;
        }

        foreach (var enemy in s.Enemies)
        {
            if (!EnemyInCombat(w, enemy))
            {
                continue;
            }

            Powers(w, enemy.Buffs, c, enemy, into, gaps);
            into.Add(new(new((int)ListenerKind.Monster, enemy.DefId, c), c, 0, OldPile.None, default, enemy));
            c++;
        }

        RunModels(w, into);
    }

    /// <summary>
    /// game: RunState.IterateHookListeners, its first part: the run deck's cards, each followed by its enchantment,
    /// while the player is active. With no combat (<paramref name="withRunItems" />) the run's relics (melted ones are
    /// gone from the old list) and potions, then modifiers, badges and the scaling model follow.
    /// </summary>
    public static void Run(OldWorld w, List<OldListener> into, List<OldGap> gaps, bool withRunItems)
    {
        if (w.Run is not { } r)
        {
            gaps.Add(new(NotModelledKind.DeckWithoutRun, ListenerKind.Card, 0, 0, "deck"));
            gaps.Add(new(NotModelledKind.DeckWithoutRun, ListenerKind.Enchantment, 0, 0, "deck"));
            return;
        }

        PlayerGaps(w, gaps);
        if (!PlayerActive(w))
        {
            return;
        }

        for (int i = 0; i < r.Deck.Count; i++)
        {
            var card = r.Deck[i];
            int inst = DeckInstance(i);
            into.Add(new(new((int)ListenerKind.Card, card.DefId, inst), 0, i, OldPile.Deck, default, null));
            if (card.Enchantment != Enchantment.None)
            {
                into.Add(new(new((int)ListenerKind.Enchantment, (int)card.Enchantment, inst), 0, i, OldPile.Deck,
                    default, null));
            }
        }

        if (!withRunItems)
        {
            return;
        }

        for (int i = 0; i < r.Relics.Count; i++)
        {
            into.Add(new(new((int)ListenerKind.Relic, r.Relics[i].DefId, i), 0, i, OldPile.None, default, null));
        }

        Potions(r.PotionSlots, 0, into);
        RunModels(w, into);
    }

    private static void Powers(OldWorld w, List<BuffState> buffs, int creature, EnemyState? enemy,
        List<OldListener> into, List<OldGap> gaps)
    {
        // One game power instance per (model, k): the k-th entry of any of its buffs; a power the engine keeps in
        // several buffs is visited once, at its first entry.
        var seen = new HashSet<(int, int)>();
        var count = new Dictionary<BuffId, int>();
        for (int i = 0; i < buffs.Count; i++)
        {
            var b = buffs[i];
            var m = w.Ids.Power(b.Id);
            if (m.Kind == PowerMapKind.EngineOnly)
            {
                continue;
            }

            if (m.Kind == PowerMapKind.Folded)
            {
                gaps.Add(new(NotModelledKind.FoldedPower, ListenerKind.Power, creature, 0, m.GameKey));
                continue;
            }

            if (m.Kind == PowerMapKind.NotModelled)
            {
                gaps.Add(new(NotModelledKind.UnclassifiedPower, ListenerKind.Power, creature, 0, m.GameKey));
                continue;
            }

            int k = count.GetValueOrDefault(b.Id);
            count[b.Id] = k + 1;
            if (seen.Add((m.ModelId, k)))
            {
                into.Add(new(new((int)ListenerKind.Power, m.ModelId, PowerInstance(creature, k)), creature, i,
                    OldPile.None, b.Id, enemy));
            }
        }
    }

    private static void Potions(int[] slots, int creature, List<OldListener> into)
    {
        for (int slot = 0; slot < slots.Length; slot++)
        {
            if (slots[slot] != 0)
            {
                into.Add(new(new((int)ListenerKind.Potion, slots[slot], slot), creature, slot, OldPile.None, default,
                    null));
            }
        }
    }

    private static void Piles(OldWorld w, CombatState s, int c, List<OldListener> into, List<OldGap> gaps)
    {
        var play = PlayCard(w, s, gaps);
        int skip = play?.Uid ?? 0;
        Pile(s.Hand, OldPile.Hand);
        Pile(s.DrawPile, OldPile.Draw);
        Pile(s.DiscardPile, OldPile.Discard);
        Pile(s.ExhaustPile, OldPile.Exhaust);
        if (play is { } p)
        {
            Card(p, OldPile.Play, 0);
        }

        if (s.AutoPlayQueue.Count > 0)
        {
            gaps.Add(new(NotModelledKind.AutoPlayQueue, ListenerKind.Card, c, 0, "auto-play queue"));
            gaps.Add(new(NotModelledKind.AutoPlayQueue, ListenerKind.Enchantment, c, 0, "auto-play queue"));
        }

        for (var a = GameAffliction.Entangled; a <= GameAffliction.Tainted; a++)
        {
            gaps.Add(new(NotModelledKind.Affliction, ListenerKind.Affliction, c, (int)a, a.ToString()));
        }

        void Pile(List<CardInstance> pile, OldPile at)
        {
            for (int i = 0; i < pile.Count; i++)
            {
                if (skip == 0 || pile[i].Uid != skip)
                {
                    Card(pile[i], at, i);
                }
            }
        }

        void Card(CardInstance card, OldPile at, int i)
        {
            into.Add(new(new((int)ListenerKind.Card, card.DefId, card.Uid), c, i, at, default, null));
            if (card.Bound)
            {
                into.Add(new(new((int)ListenerKind.Affliction, (int)GameAffliction.Bound, card.Uid), c, i, at, default,
                    null));
            }

            if (card.Enchantment != Enchantment.None)
            {
                into.Add(new(new((int)ListenerKind.Enchantment, (int)card.Enchantment, card.Uid), c, i, at, default,
                    null));
            }
        }
    }

    /// <summary>
    /// The game's Play pile over the old state: the kernel's card in flight; else the card held while the screen its
    /// play opened is open (<c>HeldPlayedCard</c> with a pending selection), unless that card is already filed. The
    /// field can outlive its play (only ReleaseHeldPlayedCard clears it, and a snapshot never restores it), so a held
    /// card whose uid is in a pile is stale and not visited; one whose result pile ends with a card of equal value is
    /// NotModelled (stale or a twin, the old state cannot tell).
    /// </summary>
    private static CardInstance? PlayCard(OldWorld w, CombatState s, List<OldGap> gaps)
    {
        if (w.InFlight is { } flying)
        {
            return flying;
        }

        if (s.HeldPlayedCard is not { } held || s.PendingSelection is null)
        {
            return null;
        }

        if (InPiles(s, held.Uid))
        {
            return null;
        }

        var result = s.HeldPlayedCardExhausts ? s.ExhaustPile : s.DiscardPile;
        if (result.Count > 0 && result[^1] == held)
        {
            gaps.Add(new(NotModelledKind.HeldCardFiled, ListenerKind.Card, 0, held.DefId, "held played card"));
            return null;
        }

        return held;
    }

    private static bool InPiles(CombatState s, int uid) =>
        s.Hand.Exists(x => x.Uid == uid) || s.DrawPile.Exists(x => x.Uid == uid)
        || s.DiscardPile.Exists(x => x.Uid == uid) || s.ExhaustPile.Exists(x => x.Uid == uid);

    private static void RunModels(OldWorld w, List<OldListener> into)
    {
        for (int i = 0; i < w.Modifiers.Count; i++)
        {
            into.Add(new(new((int)ListenerKind.Modifier, w.Modifiers[i], i), -1, i, OldPile.None, default, null));
        }

        for (int i = 0; i < w.Badges.Count; i++)
        {
            into.Add(new(new((int)ListenerKind.Badge, w.Badges[i], i), -1, i, OldPile.None, default, null));
        }

        if (w.Scaling is { } scaling)
        {
            into.Add(new(new((int)ListenerKind.Scaling, scaling, 0), -1, 0, OldPile.None, default, null));
        }
    }

    // ------------------------------------------------------------------------------------------- membership

    /// <summary>
    /// game: CombatState.Contains (and RunState.Contains for deck cards), read when the entry is visited. A power while
    /// its owner creature is in the combat and, for the player's, while the player is active for hooks (a power removed
    /// from its owner is still a member); a relic, potion, card, orb, affliction or enchantment while it is still held
    /// and the player is active; a monster while its creature is in the combat; modifiers, badges and scaling always.
    /// </summary>
    public static bool IsMember(OldWorld w, in OldListener e, PassView? view = null)
    {
        var s = w.Combat;
        var who = e.Who;
        switch (e.Kind)
        {
            case ListenerKind.Power:
            case ListenerKind.Monster:
                return CreatureInCombat(w, e) && (e.Kind == ListenerKind.Monster || e.Creature != 0
                                                  || ActiveAt(w, e, view));
            case ListenerKind.Relic:
                return ActiveAt(w, e, view) && (s?.Relics ?? w.Run?.Relics)?.Exists(r => r.DefId == who.ModelId) == true;
            case ListenerKind.Potion:
            {
                var slots = s?.PotionSlots ?? w.Run?.PotionSlots;
                return ActiveAt(w, e, view) && slots is not null && e.Index < slots.Length && slots[e.Index] == who.ModelId;
            }
            case ListenerKind.Orb:
                // Orbs have no identity in the old state: an orb is a member while one of its type is still held, which
                // is a guess once the queue has changed (an evoke leaves a wrong member, the index names another orb).
                // The pass reports NotModelled at the first orb visited after any change to the list.
                if (view?.Orbs is { } orbs && orbs.Changed(s?.Orbs))
                {
                    view.Note(new(NotModelledKind.OrbsChangedMidPass, ListenerKind.Orb, 0, who.ModelId, "orb queue"));
                }

                return ActiveAt(w, e, view) && s is not null && s.Orbs.Exists(o => (int)o.Type == who.ModelId);
            case ListenerKind.Card:
            case ListenerKind.Affliction:
            case ListenerKind.Enchantment:
            {
                if (!ActiveAt(w, e, view))
                {
                    return false;
                }

                if (e.Pile == OldPile.Deck)
                {
                    // Deck cards are matched by index: a removal shifts the later entries. The pass reports NotModelled at
                    // the first deck entry visited after any change to the deck list.
                    if (view?.Deck is { } deckProbe && deckProbe.Changed(w.Run?.Deck))
                    {
                        view.Note(new(NotModelledKind.DeckChangedMidPass, e.Kind, 0, who.ModelId, "run deck"));
                    }

                    int i = -who.Instance - 1;
                    if (w.Run?.Deck is not { } deck || i >= deck.Count)
                    {
                        return false;
                    }

                    return e.Kind == ListenerKind.Enchantment
                        ? (int)deck[i].Enchantment == who.ModelId
                        : deck[i].DefId == who.ModelId;
                }

                if (FindLive(w, who.Instance) is not { } card)
                {
                    return false;
                }

                return e.Kind switch
                {
                    ListenerKind.Affliction => card.Bound,
                    ListenerKind.Enchantment => (int)card.Enchantment == who.ModelId,
                    _ => true,
                };
            }
            default:
                return true;
        }
    }

    /// <summary>
    /// The player's IsActiveForHooks over the old state: a caller mark, active unless the kill sequence declared its
    /// hooks deactivated. The game deactivates a creature's hooks (CreatureCmd.Kill) only after the death-prevention
    /// hooks and AfterDeath, so the player's own death passes still visit its listeners at 0 HP.
    /// </summary>
    public static bool PlayerActive(OldWorld w) => w.PlayerHooks != PlayerHooks.Deactivated;

    /// <summary>The player is down (death final: IsAboutToLose) and the caller declared nothing about its hooks, so
    /// whether the pass is inside the player's kill (active) or after it (deactivated) is unknown.</summary>
    public static bool PlayerActivityUndeclared(OldWorld w) =>
        w.PlayerHooks == PlayerHooks.Undeclared && w.Combat is { } s && CombatEngine.IsAboutToLose(s);

    private static readonly ListenerKind[] PlayerKinds =
    [
        ListenerKind.Power, ListenerKind.Relic, ListenerKind.Potion, ListenerKind.Orb, ListenerKind.Card,
        ListenerKind.Affliction, ListenerKind.Enchantment,
    ];

    private static void PlayerGaps(OldWorld w, List<OldGap> gaps)
    {
        if (PlayerActivityUndeclared(w))
        {
            foreach (var kind in PlayerKinds)
            {
                gaps.Add(new(NotModelledKind.PlayerHooksUndeclared, kind, 0, 0, "player down, hooks not declared"));
            }
        }
    }

    private static bool ActiveAt(OldWorld w, in OldListener e, PassView? view)
    {
        if (view is not null && PlayerActivityUndeclared(w))
        {
            view.Note(new(NotModelledKind.PlayerHooksUndeclared, e.Kind, e.Creature, e.Who.ModelId,
                "player down, hooks not declared"));
        }

        return PlayerActive(w);
    }

    /// <summary>
    /// An enemy is in the combat while the roster holds it and it is alive, marked dying, or kept after death
    /// (ShouldCreatureBeRemovedFromCombatAfterDeath vetoed: Reattach, Illusion, Adaptable, SteamEruption,
    /// PainfulStabs).
    /// </summary>
    public static bool EnemyInCombat(OldWorld w, EnemyState enemy) =>
        w.Combat is { } s && s.Enemies.Exists(x => ReferenceEquals(x, enemy))
                          && (enemy.Hp > 0 || w.IsDying(enemy) || !EnemyAI.ShouldRemoveOnDeath(enemy));

    private static bool CreatureInCombat(OldWorld w, in OldListener e) =>
        e.Enemy is { } enemy ? EnemyInCombat(w, enemy)
        : e.Creature == 0 ? w.Combat is not null
        : w.Combat is { OstyMaxHp: > 0 };

    /// <summary>The live card with this uid: the card in flight, the piles, the auto-play queue, the held card.</summary>
    public static CardInstance? FindLive(OldWorld w, int uid)
    {
        if (uid <= 0)
        {
            return null;
        }

        if (w.InFlight is { } flying && flying.Uid == uid)
        {
            return flying;
        }

        if (w.Combat is not { } s)
        {
            return null;
        }

        foreach (var card in CardUids.LiveCards(s))
        {
            if (card.Uid == uid)
            {
                return card;
            }
        }

        return null;
    }
}

/// <summary>
/// What one pass keeps for its visit-time checks (Listeners.cs creates it at the pass's start): a change probe over the
/// orb queue and the run deck when the pass lists them, and the sink for the gaps only a visit can see. A gap is
/// noted once per pass and kind.
/// </summary>
public sealed class PassView
{
    private readonly HashSet<NotModelledKind> _noted = [];

    internal ChangeProbe<OrbState>? Orbs { get; set; }
    internal ChangeProbe<CardInstance>? Deck { get; set; }
    internal Action<OldGap>? Gap { get; set; }

    /// <summary>Starts the change probes for the lists this part of the pass lists (OldState is the one file that
    /// names the old representation).</summary>
    internal void Snapshot(OldWorld w, bool orbs, bool deck)
    {
        if (orbs && w.Combat is { } c)
        {
            Orbs = new ChangeProbe<OrbState>(c.Orbs);
        }

        if (deck && w.Run is { } r)
        {
            Deck = new ChangeProbe<CardInstance>(r.Deck);
        }
    }

    internal void Note(OldGap gap)
    {
        if (_noted.Add(gap.Why))
        {
            Gap?.Invoke(gap);
        }
    }
}

/// <summary>
/// Whether a list has been changed since the probe was made, by any mutation (add, remove, set, clear) or by being
/// replaced: a List enumerator taken at the start throws when the list's version has moved, which is public API and
/// survives NativeAOT. Orbs and run deck cards have no identity in the old state, so a change in the middle of a pass
/// leaves their entries unmatchable and the pass NotModelled.
/// </summary>
internal sealed class ChangeProbe<T>(List<T> list)
{
    private List<T>.Enumerator _probe = list.GetEnumerator();

    public bool Changed(List<T>? current)
    {
        if (!ReferenceEquals(current, list))
        {
            return true;
        }

        try
        {
            _probe.MoveNext();
            return false;
        }
        catch (InvalidOperationException)
        {
            return true;
        }
    }
}
