using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

/// <summary>
/// The kernel's listener order over the old state (Core2/OldState.cs, Core2/Listeners.cs), each case from the game's
/// rule: CombatState.IterateHookListeners (the order), CombatState.Contains (membership at visit), RunState's
/// IterateHookListeners (run+combat), Hook.IterateCombatHookListeners (the guard).
/// </summary>
public class ListenerOrderTests
{
    internal static readonly OldModelIds Ids = LoadIds();
    private const HookId H = HookId.AfterCardPlayed;

    private static OldModelIds LoadIds()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "data", "id_map.json")))
        {
            dir = dir.Parent;
        }

        Assert.NotNull(dir);
        return OldModelIds.Load(Path.Combine(dir!.FullName, "data", "id_map.json"),
            Path.Combine(dir.FullName, "tools", "fidelity", "power_aliases.json"));
    }

    private static int P(BuffId id) => Ids.Power(id).ModelId;

    private static EnemyState Enemy(int def, params BuffId[] buffs) =>
        new() { DefId = def, Hp = 40, MaxHp = 40, Buffs = buffs.Select(b => new BuffState(b, 2)).ToList() };

    private static CombatState Combat(params EnemyState[] enemies)
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = enemies.Length > 0 ? enemies.ToList() : [Enemy(16)];
        return s;
    }

    private static CardInstance Card(CombatState s, int def, int enchant = 0, bool bound = false) =>
        s.Fresh(new CardInstance(def, false, Enchantment: (Enchantment)enchant, EnchantAmount: enchant == 0 ? 0 : 1,
            Bound: bound));

    private static string Show(OldListener e) =>
        $"{e.Kind}:{e.Who.ModelId}@{e.Creature}" + (e.Pile is OldPile.None ? "" : $"/{e.Pile}{e.Index}");

    private static List<string> Order(OldWorld w, HookIterator it = HookIterator.CombatDirect) =>
        OldListeners.Enumerate(w, it).Select(Show).ToList();

    /// <summary>Visits a cursor to the end; <paramref name="at" /> runs after each visit with the visit count.</summary>
    private static List<ListenerRef> Visit(HookCursor<Action> cur, Action<int>? at = null)
    {
        var seen = new List<ListenerRef>();
        while (cur.Next(out var l))
        {
            seen.Add(l.Who);
            at?.Invoke(seen.Count);
        }

        return seen;
    }

    private static ListenerRef Relic(int def, int i) => new((int)ListenerKind.Relic, def, i);

    [Fact] // game: CombatState.IterateHookListeners (allies then enemies; powers, then the monster of a non-player)
    public void CreaturesInGameOrderWithPowersBeforeTheirMonster()
    {
        var s = Combat(Enemy(16, BuffId.Weak, BuffId.Strength, BuffId.Vulnerable), Enemy(17, BuffId.Dexterity));
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 1));
        s.OstyMaxHp = s.OstyHp = 10;
        var gaps = new List<OldGap>();
        var got = OldListeners.Enumerate(new OldWorld(s, null, Ids), HookIterator.CombatDirect, gaps).Select(Show);
        Assert.Equal(new[]
        {
            $"Power:{P(BuffId.Strength)}@0", $"Monster:{Ids.OstyModelId}@1",
            $"Power:{P(BuffId.Weak)}@2", $"Power:{P(BuffId.Strength)}@2", $"Power:{P(BuffId.Vulnerable)}@2",
            "Monster:16@2", $"Power:{P(BuffId.Dexterity)}@3", "Monster:17@3",
        }, got);
        Assert.Contains(gaps, g => g.Why == NotModelledKind.OstyPowers && g.Creature == 1);
    }

    [Fact] // game: CombatState.IterateHookListeners (relics in the player's list order, melted skipped)
    public void RelicsInObtainOrderAndItsReverse()
    {
        var s = Combat();
        s.Relics = [new RelicInstance(3), new RelicInstance(1), new RelicInstance(2)];
        var w = new OldWorld(s, null, Ids);
        Assert.Equal(new[] { "Relic:3@0", "Relic:1@0", "Relic:2@0", "Monster:16@1" }, Order(w));
        s.Relics.Reverse();
        Assert.Equal(new[] { "Relic:2@0", "Relic:1@0", "Relic:3@0", "Monster:16@1" }, Order(w));
    }

    [Fact] // game: CombatState.IterateHookListeners (PotionSlots by index, empty slots skipped)
    public void PotionsBySlotWithGaps()
    {
        var s = Combat();
        s.PotionSlots[0] = 2;
        s.PotionSlots[2] = 1;
        var pots = OldListeners.Enumerate(new OldWorld(s, null, Ids), HookIterator.CombatDirect)
            .Where(e => e.Kind == ListenerKind.Potion).Select(e => (e.Who.ModelId, e.Index)).ToList();
        Assert.Equal(new[] { (2, 0), (1, 2) }, pots);
    }

    [Fact] // game: CombatState.IterateHookListeners (orbs in queue order, whatever their types)
    public void OrbsInQueueOrderNotByType()
    {
        var s = Combat();
        s.Orbs = [new OrbState((OrbType)2), new OrbState((OrbType)0), new OrbState((OrbType)1)];
        var orbs = OldListeners.Enumerate(new OldWorld(s, null, Ids), HookIterator.CombatDirect)
            .Where(e => e.Kind == ListenerKind.Orb).Select(e => (e.Who.ModelId, e.Index)).ToList();
        Assert.Equal(new[] { (2, 0), (0, 1), (1, 2) }, orbs);
    }

    [Fact] // game: CombatState.IterateHookListeners (orbs, then AllPiles: Hand, Draw, Discard, Exhaust, Play)
    public void PilesInOrderWithAfflictionAndEnchantmentAfterEachCard()
    {
        var s = Combat();
        s.Relics = [new RelicInstance(1)];
        s.PotionSlots[0] = 2;
        s.Orbs = [new OrbState((OrbType)1), new OrbState((OrbType)2)];
        s.Hand = [Card(s, 10, bound: true), Card(s, 11, enchant: 1)];
        s.DrawPile = [Card(s, 12)];
        s.DiscardPile = [Card(s, 13, enchant: 2, bound: true)];
        s.ExhaustPile = [Card(s, 14)];
        var w = new OldWorld(s, null, Ids);
        using var flying = w.Flying(Card(s, 15));
        Assert.Equal(new[]
        {
            "Relic:1@0", "Potion:2@0", "Orb:1@0", "Orb:2@0",
            "Card:10@0/Hand0", "Affliction:1@0/Hand0", "Card:11@0/Hand1", "Enchantment:1@0/Hand1",
            "Card:12@0/Draw0", "Card:13@0/Discard0", "Affliction:1@0/Discard0", "Enchantment:2@0/Discard0",
            "Card:14@0/Exhaust0", "Card:15@0/Play0", "Monster:16@1",
        }, Order(w));
        // A card listener's instance is its K2b uid, the CardHandle.
        var hand = OldListeners.Enumerate(w, HookIterator.CombatDirect).First(e => e.Kind == ListenerKind.Card);
        Assert.Equal(s.Hand[0].Uid, hand.Who.Instance);
    }

    [Fact] // game: CombatState.IterateHookListeners (the player's powers, then its relics, potions, orbs and cards)
    public void ThePlayersPowersComeBeforeItsRelicsPotionsOrbsAndCards()
    {
        var s = Combat();
        s.PlayerBuffs = [new BuffState(BuffId.Strength, 1), new BuffState(BuffId.Dexterity, 1)];
        s.Relics = [new RelicInstance(1)];
        s.PotionSlots[0] = 2;
        s.Orbs = [new OrbState((OrbType)1)];
        s.Hand = [Card(s, 10)];
        Assert.Equal(new[]
        {
            $"Power:{P(BuffId.Strength)}@0", $"Power:{P(BuffId.Dexterity)}@0", "Relic:1@0", "Potion:2@0", "Orb:1@0",
            "Card:10@0/Hand0", "Monster:16@1",
        }, Order(new OldWorld(s, null, Ids)));
    }

    [Fact] // game: CombatState.IterateHookListeners (Modifiers, BadgeModels, MultiplayerScalingModel after the creatures)
    public void ModifiersBadgesAndScalingLast()
    {
        var s = Combat(Enemy(16, BuffId.Weak));
        var w = new OldWorld(s, null, Ids) { Scaling = 7 };
        w.Modifiers.Add(5);
        w.Badges.Add(6);
        Assert.Equal(new[] { $"Power:{P(BuffId.Weak)}@1", "Monster:16@1", "Modifier:5@-1", "Badge:6@-1", "Scaling:7@-1" },
            Order(w));
    }

    [Fact] // game: RunState.IterateHookListeners (deck cards and enchantments first; the child combat's list after)
    public void RunCombatListsTheDeckFirstAndBuildsTheCombatPartWhenTheDeckEnds()
    {
        var s = Combat();
        s.Relics = [new RelicInstance(1)];
        var run = new RunState();
        run.Deck = [new CardInstance(20, false), new CardInstance(21, false, Enchantment: (Enchantment)1)];
        var lis = new OldListeners(new OldWorld(s, run, Ids), EveryListener.Instance);
        var cur = lis.Cursor<Action>(HookIterator.RunCombat, HookId.ModifyDamageAdditive);
        var seen = Visit(cur, n =>
        {
            if (n == 1)
            {
                s.Relics.Add(new RelicInstance(2)); // before the deck part ends: in the late-built combat part
            }
        });
        Assert.Equal(new ListenerRef[]
        {
            new((int)ListenerKind.Card, 20, -1), new((int)ListenerKind.Card, 21, -2),
            new((int)ListenerKind.Enchantment, 1, -2), Relic(1, 0), Relic(2, 1), new((int)ListenerKind.Monster, 16, 1),
        }, seen);
        Assert.Equal(5, cur.Index);
    }

    [Fact] // game: CombatState.Contains (a power while its owner creature is in the combat)
    public void APowerRemovedMidPassIsStillVisited()
    {
        var s = Combat();
        s.PlayerBuffs = [new BuffState(BuffId.Strength, 1), new BuffState(BuffId.Dexterity, 1)];
        var cur = new OldListeners(new OldWorld(s, null, Ids), EveryListener.Instance).Cursor<Action>(HookIterator.Combat, H);
        var seen = Visit(cur, n =>
        {
            if (n == 1)
            {
                s.PlayerBuffs.RemoveAt(1);
            }
        });
        Assert.Equal(P(BuffId.Dexterity), seen[1].ModelId);
    }

    [Fact] // game: CombatState.Contains (a monster or power while its creature's CombatState is set)
    public void ACreatureRemovedMidPassIsNotVisited()
    {
        var a = Enemy(16, BuffId.Weak);
        var b = Enemy(17, BuffId.Vulnerable);
        var s = Combat(a, b);
        s.PlayerBuffs.Add(new BuffState(BuffId.Strength, 1));
        var cur = new OldListeners(new OldWorld(s, null, Ids), EveryListener.Instance).Cursor<Action>(HookIterator.CombatDirect, H);
        var seen = Visit(cur, n =>
        {
            if (n == 1)
            {
                s.Enemies.Remove(b);
                a.Hp = 0; // dead with no power keeping it: removed
            }
        });
        Assert.Single(seen);

        // Dead but not yet removed (inside the kill): still in the combat.
        a.Hp = 0;
        s.Enemies = [a];
        var w = new OldWorld(s, null, Ids);
        using (w.MarkDying(a))
        {
            Assert.Contains("Monster:16@1", Order(w));
        }

        Assert.DoesNotContain("Monster:16@1", Order(w)); // the kill's mark ended: removed
    }

    [Fact] // game: CombatState.Contains (IsActiveForHooks read when each entry is visited)
    public void ADeactivatedPlayersListenersAreSkippedFromThatEntryOn()
    {
        var e = Enemy(16, BuffId.Weak);
        var s = Combat(e);
        s.Relics = [new RelicInstance(1), new RelicInstance(2)];
        s.Hand = [Card(s, 10)];
        var w = new OldWorld(s, null, Ids);
        var cur = new OldListeners(w, EveryListener.Instance).Cursor<Action>(HookIterator.CombatDirect, H);
        IDisposable? down = null;
        var seen = Visit(cur, n =>
        {
            if (n == 1)
            {
                down = w.DeactivatePlayerHooks(); // CreatureCmd.Kill's DeactivateHooks, after the death passes
            }
        });
        down?.Dispose();
        Assert.Equal(new ListenerRef[]
        {
            Relic(1, 0), new((int)ListenerKind.Power, P(BuffId.Weak), OldState.PowerInstance(1, 0)),
            new((int)ListenerKind.Monster, 16, 1),
        }, seen);
    }

    [Fact] // game: CombatState.IterateHookListeners (the list is built before the first listener runs)
    public void AListenerAddedMidPassIsNotVisited()
    {
        var e = Enemy(16);
        var s = Combat(e);
        s.Relics = [new RelicInstance(1)];
        var cur = new OldListeners(new OldWorld(s, null, Ids), EveryListener.Instance).Cursor<Action>(HookIterator.CombatDirect, H);
        var seen = Visit(cur, n =>
        {
            if (n == 1)
            {
                s.Relics.Add(new RelicInstance(2));
                s.Hand.Add(Card(s, 10));
                e.Buffs.Add(new BuffState(BuffId.Weak, 1));
            }
        });
        Assert.Equal(new ListenerRef[] { Relic(1, 0), new((int)ListenerKind.Monster, 16, 1) }, seen);
    }

    [Fact] // game: CombatState.Contains (a card while it has not been removed from the combat's state)
    public void ACardMovedToAnotherPileMidPassIsVisitedOnceByUid()
    {
        var s = Combat();
        s.Hand = [Card(s, 10), Card(s, 11)];
        var moved = s.Hand[1];
        var gone = Card(s, 12);
        s.DiscardPile = [gone];
        var cur = new OldListeners(new OldWorld(s, null, Ids), EveryListener.Instance).Cursor<Action>(HookIterator.CombatDirect, H);
        var seen = Visit(cur, n =>
        {
            if (n == 1)
            {
                s.Hand.RemoveAt(1);
                s.ExhaustPile.Add(moved);
                s.DiscardPile.Clear(); // removed from the combat
            }
        });
        Assert.Equal(new[] { s.Hand[0].Uid, moved.Uid },
            seen.Where(x => x.Kind == (int)ListenerKind.Card).Select(x => x.Instance));
    }

    [Fact] // game: Hook.IterateCombatHookListeners (IsOverOrEnding asked once, when the pass starts; not in setup)
    public void TheGuardIsReadOnceAtPassStart()
    {
        var e = Enemy(16);
        var s = Combat(e);
        s.Relics = [new RelicInstance(1), new RelicInstance(2)];
        Assert.False(CombatEngine.IsOverOrEnding(s));
        var w = new OldWorld(s, null, Ids);
        var lis = new OldListeners(w, EveryListener.Instance);
        var seen = Visit(lis.Cursor<Action>(HookIterator.Combat, H), n =>
        {
            if (n == 1)
            {
                e.Hp = 0;
            }
        });
        Assert.Equal(new[] { Relic(1, 0), Relic(2, 1) }, seen); // the pass runs on; the dead enemy is gone
        Assert.True(CombatEngine.IsOverOrEnding(s));
        Assert.Empty(Visit(lis.Cursor<Action>(HookIterator.Combat, H)));
        Assert.Equal(2, Visit(lis.Cursor<Action>(HookIterator.CombatDirect, H)).Count);
        using (w.Setup())
        {
            Assert.Equal(2, Visit(lis.Cursor<Action>(HookIterator.Combat, H)).Count);
        }
    }

    [Fact] // game: CombatState.IterateHookListeners (the Play pile holds the card while its OnPlay runs)
    public void AStaleHeldPlayedCardIsNotASecondPlayPileVisit()
    {
        var s = Combat();
        var held = Card(s, 10);
        s.HeldPlayedCard = held;
        s.PendingSelection = new PendingCardSelection { Kind = CardSelectionKind.None, Candidates = [], SourceCardDefId = 10 };
        var w = new OldWorld(s, null, Ids);
        Assert.Contains("Card:10@0/Play0", Order(w)); // held while its screen is open: the Play pile

        s.DiscardPile = [held]; // filed, the field left set (only ReleaseHeldPlayedCard clears it)
        Assert.Equal(new[] { "Card:10@0/Discard0" }, Order(w).Where(x => x.StartsWith("Card:")));

        s.DiscardPile = [held.WithUid(s.NewCardUid())]; // filed by a snapshot read: a new uid, the same card
        var gaps = new List<OldGap>();
        var cards = OldListeners.Enumerate(w, HookIterator.CombatDirect, gaps).Where(x => x.Kind == ListenerKind.Card);
        Assert.Equal(new[] { OldPile.Discard }, cards.Select(x => x.Pile));
        Assert.Contains(gaps, g => g.Why == NotModelledKind.HeldCardFiled);

        s.DiscardPile = [];
        s.PendingSelection = null; // no screen: the play is over, the field is stale
        Assert.DoesNotContain(Order(w), x => x.StartsWith("Card:"));
    }

    [Fact] // the BuffId to game power rule (OldModelIds): engine-only out, folds NotModelled, split powers once
    public void PowersMapToGamePowersVisitedOnceAndFoldsAreNotModelled()
    {
        var s = Combat();
        s.PlayerBuffs =
        [
            new BuffState(BuffId.InfernoSelfDamage, 3), new BuffState(BuffId.Stunned, 1),
            new BuffState(BuffId.TemporaryStrength, 2), new BuffState(BuffId.Inferno, 3),
            new BuffState(BuffId.NextTurnEnergy, 1),
        ];
        Assert.Equal(PowerMapKind.Piece, Ids.Power(BuffId.InfernoSelfDamage).Kind);
        Assert.Equal(P(BuffId.Inferno), Ids.Power(BuffId.InfernoSelfDamage).ModelId);
        Assert.Equal("EnergyNextTurnPower", Ids.PowerName(P(BuffId.NextTurnEnergy)));
        Assert.Equal("AccuracyPower", Ids.PowerName(P(BuffId.ShivDamage)));
        Assert.Equal(PowerMapKind.EngineOnly, Ids.Power(BuffId.Stunned).Kind);
        Assert.Equal(PowerMapKind.Folded, Ids.Power(BuffId.TemporaryDexterity).Kind);
        var lis = new OldListeners(new OldWorld(s, null, Ids), EveryListener.Instance);
        var seen = Visit(lis.Cursor<Action>(HookIterator.CombatDirect, H));
        Assert.Equal(new[] { P(BuffId.Inferno), P(BuffId.NextTurnEnergy) },
            seen.Where(x => x.Kind == (int)ListenerKind.Power).Select(x => x.ModelId));
        Assert.Contains(lis.LastPass, n => n.Gap.Why == NotModelledKind.FoldedPower && n.Hook == H);
    }

    [Fact] // NotModelled is reported only for a pass that has a handler the gap could stand for
    public void GapsAreReportedPerPassOnlyWhereAHandlerNeedsThem()
    {
        var s = Combat();
        s.PlayerBuffs = [new BuffState(BuffId.TemporaryStrength, 2)];
        s.Relics = [new RelicInstance(1)];
        var relicsOnly = new OnlyKind(ListenerKind.Relic);
        var lis = new OldListeners(new OldWorld(s, null, Ids), relicsOnly);
        var reported = new List<NotModelled>();
        lis.Report = reported.Add;
        Assert.Equal(new[] { Relic(1, 0) }, Visit(lis.Cursor<Action>(HookIterator.CombatDirect, H)));
        Assert.Empty(reported);

        var all = new OldListeners(new OldWorld(s, null, Ids), EveryListener.Instance) { Report = reported.Add };
        Visit(all.Cursor<Action>(HookIterator.RunCombat, H)); // no run state: the deck is unknown
        Assert.Contains(reported, n => n.Gap.Why == NotModelledKind.DeckWithoutRun && n.Pass == 1);
        Assert.Contains(reported, n => n.Gap.Why == NotModelledKind.FoldedPower);
    }

    [Fact] // game: CreatureCmd.Kill (death prevention and AfterDeath run before DeactivateHooks)
    public void ThePlayersOwnDeathPassVisitsItsListenersAndAnUndeclaredOneIsNotModelled()
    {
        var s = Combat(Enemy(16, BuffId.Weak));
        s.PlayerHp = 0; // down with no Fairy in a Bottle and no Lizard Tail: the death is final
        s.PlayerBuffs = [new BuffState(BuffId.Strength, 1)];
        s.Relics = [new RelicInstance(1)];
        s.Hand = [Card(s, 10)];
        Assert.True(CombatEngine.IsAboutToLose(s));
        var w = new OldWorld(s, null, Ids);
        var lis = new OldListeners(w, EveryListener.Instance);
        var reported = new List<NotModelled>();
        lis.Report = reported.Add;
        var strength = new ListenerRef((int)ListenerKind.Power, P(BuffId.Strength), OldState.PowerInstance(0, 0));
        var weak = new ListenerRef((int)ListenerKind.Power, P(BuffId.Weak), OldState.PowerInstance(1, 0));
        var card = new ListenerRef((int)ListenerKind.Card, 10, s.Hand[0].Uid);
        var monster = new ListenerRef((int)ListenerKind.Monster, 16, 1);

        using (w.KeepPlayerHooks()) // the kill's own passes: the player's hooks are still on, at 0 HP
        {
            Assert.Equal(new[] { strength, Relic(1, 0), card, weak, monster },
                Visit(lis.Cursor<Action>(HookIterator.CombatDirect, H)));
            Assert.DoesNotContain(reported, n => n.Gap.Why == NotModelledKind.PlayerHooksUndeclared);
        }

        using (w.DeactivatePlayerHooks()) // after DeactivateHooks
        {
            Assert.Equal(new[] { weak, monster }, Visit(lis.Cursor<Action>(HookIterator.CombatDirect, H)));
            Assert.DoesNotContain(reported, n => n.Gap.Why == NotModelledKind.PlayerHooksUndeclared);
        }

        // Nothing declared with the player down: inside the kill or after it, the state cannot say. (A fresh world: a
        // deactivation is final.)
        w = new OldWorld(s, null, Ids);
        lis = new OldListeners(w, EveryListener.Instance) { Report = reported.Add };
        Visit(lis.Cursor<Action>(HookIterator.CombatDirect, H));
        Assert.Contains(reported, n => n.Gap.Why == NotModelledKind.PlayerHooksUndeclared && n.Gap.Kind == ListenerKind.Relic);
        Assert.Contains(reported, n => n.Gap.Why == NotModelledKind.PlayerHooksUndeclared && n.Gap.Kind == ListenerKind.Power);
        s.PlayerHp = 50; // alive: nothing to declare
        reported.Clear();
        Visit(lis.Cursor<Action>(HookIterator.CombatDirect, H));
        Assert.DoesNotContain(reported, n => n.Gap.Why == NotModelledKind.PlayerHooksUndeclared);
    }

    [Fact] // game: Hook.IterateCombatHookListeners (the guard's IsStarting exemption) with the setup mark missing
    public void AGuardSetBeforeTurnOneWithoutTheSetupMarkIsNotModelled()
    {
        var e = Enemy(16);
        var s = Combat(e);
        s.Relics = [new RelicInstance(1)];
        e.Hp = 0; // nothing left to fight yet (the roster is still being filled): the guard is set
        var w = new OldWorld(s, null, Ids);
        var lis = new OldListeners(w, EveryListener.Instance);
        var reported = new List<NotModelled>();
        lis.Report = reported.Add;
        Assert.Empty(Visit(lis.Cursor<Action>(HookIterator.Combat, H)));
        Assert.Contains(reported, n => n.Gap.Why == NotModelledKind.SetupUndeclared);

        reported.Clear();
        using (w.Setup())
        {
            Assert.Equal(new[] { Relic(1, 0) }, Visit(lis.Cursor<Action>(HookIterator.Combat, H)));
        }

        Assert.DoesNotContain(reported, n => n.Gap.Why == NotModelledKind.SetupUndeclared);
        s.Turn = 1; // a combat that really is over: the guard's empty pass is the game's
        Assert.Empty(Visit(lis.Cursor<Action>(HookIterator.Combat, H)));
        Assert.DoesNotContain(reported, n => n.Gap.Why == NotModelledKind.SetupUndeclared);
    }

    [Fact] // the marks' scoped setters: the previous value is back when the scope ends, nesting included
    public void ScopedMarksRestoreTheirPreviousValue()
    {
        var s = Combat();
        var w = new OldWorld(s, null, Ids);
        var a = Card(s, 10);
        var b = Card(s, 11);
        var enemy = s.Enemies[0];
        using (w.Flying(a))
        {
            using (w.Flying(b))
            {
                Assert.Equal(b, w.InFlight);
            }

            Assert.Equal(a, w.InFlight);
        }

        Assert.Null(w.InFlight);
        using (w.Setup())
        {
            w.Setup().Dispose();
            Assert.True(w.InSetup);
        }

        Assert.False(w.InSetup);
        using (w.MarkDying(enemy))
        {
            w.MarkDying(enemy).Dispose();
            Assert.True(w.IsDying(enemy));
        }

        Assert.False(w.IsDying(enemy));
        using (w.KeepPlayerHooks())
        {
            using (w.DeactivatePlayerHooks())
            {
                Assert.Equal(PlayerHooks.Deactivated, w.PlayerHooks);
            }

            // Absorbing: neither the inner scope's end nor the outer's restores an earlier state.
            Assert.Equal(PlayerHooks.Deactivated, w.PlayerHooks);
        }

        Assert.Equal(PlayerHooks.Deactivated, w.PlayerHooks);
        w.KeepPlayerHooks().Dispose();
        Assert.Equal(PlayerHooks.Deactivated, w.PlayerHooks);
    }

    private static List<NotModelled> OrbPass(Action<CombatState>? change)
    {
        var s = Combat();
        s.Relics = [new RelicInstance(1)];
        s.Orbs = [new OrbState(OrbType.Lightning), new OrbState(OrbType.Lightning)];
        var lis = new OldListeners(new OldWorld(s, null, Ids), EveryListener.Instance);
        var reported = new List<NotModelled>();
        lis.Report = reported.Add;
        Visit(lis.Cursor<Action>(HookIterator.CombatDirect, H), n =>
        {
            if (n == 1)
            {
                change?.Invoke(s); // after the relic, before the orbs
            }
        });
        return reported;
    }

    [Fact] // orbs have no identity in the old state: any change to the queue after the snapshot is NotModelled
    public void AnOrbQueueChangedMidPassIsNotModelled()
    {
        static bool Orbs(List<NotModelled> r) => r.Exists(n => n.Gap.Why == NotModelledKind.OrbsChangedMidPass);

        Assert.False(Orbs(OrbPass(null)));
        Assert.True(Orbs(OrbPass(s => s.Orbs.RemoveAt(0)))); // an evoke
        Assert.True(Orbs(OrbPass(s => { s.Orbs.RemoveAt(0); s.Orbs.Add(new OrbState(OrbType.Lightning)); }))); // equal in type
        Assert.True(Orbs(OrbPass(s => s.Orbs.Add(new OrbState(OrbType.Frost)))));
        Assert.True(Orbs(OrbPass(s => s.Orbs[1] = s.Orbs[1] with { PassiveValue = 3 })));
        Assert.True(Orbs(OrbPass(s => s.Orbs = [new OrbState(OrbType.Lightning), new OrbState(OrbType.Lightning)])));
        Assert.Single(OrbPass(s => s.Orbs.RemoveAt(0)), n => n.Gap.Why == NotModelledKind.OrbsChangedMidPass);
    }

    private static List<NotModelled> DeckPass(Action<RunState>? change)
    {
        var s = Combat();
        var run = new RunState();
        run.Deck = [new CardInstance(20, false), new CardInstance(21, false), new CardInstance(22, false)];
        var lis = new OldListeners(new OldWorld(s, run, Ids), EveryListener.Instance);
        var reported = new List<NotModelled>();
        lis.Report = reported.Add;
        Visit(lis.Cursor<Action>(HookIterator.RunCombat, H), n =>
        {
            if (n == 1)
            {
                change?.Invoke(run); // after the first deck card, before the later ones
            }
        });
        return reported;
    }

    [Fact] // deck cards are matched by index: any change to the deck after the snapshot is NotModelled
    public void ARunDeckChangedMidPassIsNotModelled()
    {
        static bool Deck(List<NotModelled> r) => r.Exists(n => n.Gap.Why == NotModelledKind.DeckChangedMidPass);

        Assert.False(Deck(DeckPass(null)));
        Assert.True(Deck(DeckPass(r => r.Deck.RemoveAt(0)))); // later entries shift
        Assert.True(Deck(DeckPass(r => r.Deck.RemoveAt(1))));
        Assert.True(Deck(DeckPass(r => r.Deck[1] = new CardInstance(21, true))));
        Assert.True(Deck(DeckPass(r => r.Deck = [new CardInstance(20, false)])));
        Assert.Single(DeckPass(r => r.Deck.RemoveAt(0)), n => n.Gap.Why == NotModelledKind.DeckChangedMidPass);
    }

    private sealed class OnlyKind(ListenerKind kind) : IHandlerSource
    {
        public Delegate? Find(HookId hook, ListenerKind k, int modelId) => k == kind ? EveryListener.Noop : null;

        public bool AnyOf(HookId hook, ListenerKind k) => k == kind;
    }
}
