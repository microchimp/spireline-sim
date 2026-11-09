using System.Reflection;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Sts2Emulator.GeneratedData;
using Xunit;

namespace Sts2Emulator.Tests.Core2;

/// <summary>
/// The engine-speed changes (docs/rebuild/NATIVE_KERNEL.md section 15): each is held to the form it replaced, which these tests keep as the
/// reference. A change that cannot alter a game shows here as two forms that agree on every input tried; the fight exam's
/// 1,600 games (tools/gym_engine_move_check.py) are the proof on whole fights.
/// </summary>
public class EngineSpeedTests
{
    private static readonly OldModelIds Ids = ListenerOrderTests.Ids;

    // ------------------------------------------------------------------------------------------ indexed definition lookup

    private static T[] TableOf<T>(Type table) =>
        (T[])table.GetField("_all", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!;

    /// <summary>The ids to try: every id in the table, the ones around them, zero, negatives and past the dense range.</summary>
    private static IEnumerable<int> Probes(IEnumerable<int> ids)
    {
        var set = new SortedSet<int> { -1, 0, int.MinValue, int.MaxValue, 16383, 16384, 16385, 100000 };
        foreach (int id in ids)
        {
            set.Add(id);
            set.Add(id - 1);
            set.Add(id + 1);
        }

        return set;
    }

    [Fact]
    public void CardsGetIsTheFirstMatchOfTheScanForEveryId()
    {
        var all = TableOf<CardDef>(typeof(Cards));
        foreach (int id in Probes(all.Select(c => c.Id)))
        {
            var first = Array.Find(all, c => c.Id == id);
            if (first.Id != 0)
            {
                Assert.Equal(first, Cards.Get(id));
                Assert.Equal(first, Cards.Ref(id));
            }
            else
            {
                Assert.Throws<ArgumentException>(() => Cards.Get(id));
                Assert.Throws<ArgumentException>(() => Cards.Ref(id));
            }
        }
    }

    [Fact]
    public void TheOtherTablesGetIsTheFirstMatchOfTheScanForEveryId()
    {
        var enemies = TableOf<EnemyDef>(typeof(Enemies));
        foreach (int id in Probes(enemies.Select(e => e.Id)))
        {
            var first = Array.Find(enemies, e => e.Id == id);
            if (first.Id > 0)
                Assert.Equal(first, Enemies.Get(id));
            else
                Assert.Throws<ArgumentException>(() => Enemies.Get(id));
        }

        var potions = TableOf<PotionDef>(typeof(Potions));
        foreach (int id in Probes(potions.Select(p => p.Id)))
        {
            var first = Array.Find(potions, p => p.Id == id);
            if (first.Id > 0)
                Assert.Equal(first, Potions.Get(id));
            else
                Assert.Throws<ArgumentException>(() => Potions.Get(id));
        }

        var powers = TableOf<PowerDef>(typeof(Powers));
        foreach (int id in Probes(powers.Select(p => p.Id)))
        {
            var first = Array.Find(powers, p => p.Id == id);
            if (first.Id > 0)
                Assert.Equal(first, Powers.Get(id));
            else
                Assert.Throws<ArgumentException>(() => Powers.Get(id));
        }

        var relics = TableOf<RelicDef>(typeof(Relics));
        foreach (int id in Probes(relics.Select(r => r.Id)))
        {
            var first = Array.Find(relics, r => r.Id == id);
            Assert.Equal(first.Id > 0, Relics.TryGet(id, out var got));
            Assert.Equal(first, got);
            if (first.Id > 0)
                Assert.Equal(first, Relics.Get(id));
            else
                Assert.Throws<ArgumentException>(() => Relics.Get(id));
        }
    }

    [Fact]
    public void ARepeatedIdAnswersWithTheFirstDefinition()
    {
        var table = new[] { new CardDef(5, "A", 1, 0, 0, 0, 0, 0, CardType.Skill, CardRarity.Basic), new CardDef(5, "B", 1, 0, 0, 0, 0, 0, CardType.Skill, CardRarity.Basic),
            new CardDef(40000, "Far", 1, 0, 0, 0, 0, 0, CardType.Skill, CardRarity.Basic), new CardDef(40000, "Far2", 1, 0, 0, 0, 0, 0, CardType.Skill, CardRarity.Basic) };
        var index = DefIndex.Of(table, c => c.Id);
        Assert.Equal(0, index.First(5));
        Assert.Equal(2, index.First(40000));
        Assert.Equal(-1, index.First(6));
        Assert.Equal(-1, index.First(-5));
        Assert.Equal(-1, index.First(int.MaxValue));
    }

    // ------------------------------------------------------------------------------------------ the card-uid safety net

    private static List<int> Reference_Classify_Walk(CombatState state, out int fixedCount)
    {
        // The form Normalise had: rewrite every live card through Classify, a fresh uid for each breach.
        var seen = new HashSet<int>();
        int fixes = 0;
        int counter = state.LastCardUid;
        CardUids.Rewrite(state, card =>
        {
            int uid = card.UidOf();
            bool ok = uid != 0 && uid <= counter && seen.Add(uid);
            if (ok) return card;
            fixes++;
            var fresh = state.Fresh(card);
            seen.Add(fresh.UidOf());
            return fresh;
        });
        fixedCount = fixes;
        return CardUids.LiveCards(state).Select(c => c.UidOf()).ToList();
    }

    private static CombatState RandomCards(Random r, bool breaks)
    {
        var s = new CombatState();
        int uid = 0;
        CardInstance Make()
        {
            var c = new CardInstance(r.Next(1, 40), r.Next(2) == 0);
            return c.WithUid(++uid);
        }

        foreach (var pile in new[] { s.Hand, s.DrawPile, s.DiscardPile, s.ExhaustPile, s.AutoPlayQueue })
            for (int i = r.Next(0, 9); i > 0; i--)
                pile.Add(Make());
        if (r.Next(3) == 0)
            s.HeldPlayedCard = Make();
        s.LastCardUid = uid;
        if (breaks)
        {
            var piles = new[] { s.Hand, s.DrawPile, s.DiscardPile, s.ExhaustPile, s.AutoPlayQueue }.Where(p => p.Count > 0).ToList();
            if (piles.Count > 0)
            {
                var pile = piles[r.Next(piles.Count)];
                int at = r.Next(pile.Count);
                int kind = r.Next(4);
                pile[at] = pile[at].WithUid(kind switch { 0 => 0, 1 => uid + 1 + r.Next(5), 2 => pile[(at + 1) % pile.Count].UidOf(), _ => -3 });
            }
        }

        return s;
    }

    [Fact]
    public void NormaliseAgreesWithTheRewriteFormOnIntactAndBrokenCombats()
    {
        var r = new Random(20261109);
        for (int round = 0; round < 600; round++)
        {
            bool breaks = round % 3 != 0;
            int seed = r.Next();
            var a = RandomCards(new Random(seed), breaks);
            var b = RandomCards(new Random(seed), breaks);
            var expected = Reference_Classify_Walk(b, out int expectedFixes);
            int got = CardUidInvariant.Normalise(a);
            Assert.Equal(expectedFixes, got);
            Assert.Equal(expected, CardUids.LiveCards(a).Select(c => c.UidOf()).ToList());
            Assert.Equal(b.LastCardUid, a.LastCardUid);
            Assert.Equal(CardUids.LiveCards(b).ToList(), CardUids.LiveCards(a).ToList());
        }
    }

    [Fact]
    public void NormaliseLeavesAnIntactCombatAlone()
    {
        var s = RandomCards(new Random(7), breaks: false);
        var before = CardUids.LiveCards(s).ToList();
        long acting = CardUidInvariant.ActingCalls;
        Assert.Equal(0, CardUidInvariant.Normalise(s));
        Assert.Equal(before, CardUids.LiveCards(s).ToList());
        Assert.Equal(acting, CardUidInvariant.ActingCalls);
    }

    [Fact]
    public void NormaliseTakesTheGeneralPathForACounterPastTheBitSet()
    {
        var s = new CombatState();
        s.Hand.Add(new CardInstance(3, false).WithUid(1));
        s.Hand.Add(new CardInstance(3, false).WithUid(1));
        s.LastCardUid = 100000;
        Assert.Equal(1, CardUidInvariant.Normalise(s));
        Assert.Equal(100001, s.Hand[1].UidOf());
    }

    // ------------------------------------------------------------------------------------------ kind-masked listener lists

    private static CombatState RichCombat()
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = [Enemy(16, BuffId.Weak, BuffId.Strength, BuffId.Vulnerable, BuffId.Strength), Enemy(17, BuffId.Dexterity), Enemy(18)];
        s.Enemies[2].Hp = 0;
        s.PlayerBuffs = [new BuffState(BuffId.Strength, 1), new BuffState(BuffId.Dexterity, 1), new BuffState(BuffId.Weak, 2)];
        s.OstyMaxHp = s.OstyHp = 10;
        s.Relics = [new RelicInstance(1), new RelicInstance(2), new RelicInstance(3)];
        s.PotionSlots[0] = 2;
        s.Orbs = [new OrbState((OrbType)1), new OrbState((OrbType)2)];
        s.Hand = [Card(s, 20), Card(s, 21, enchant: 1), Card(s, 22, bound: true)];
        s.DrawPile = [Card(s, 23), Card(s, 24)];
        s.DiscardPile = [Card(s, 25, enchant: 2)];
        s.ExhaustPile = [Card(s, 26)];
        s.AutoPlayQueue = [Card(s, 27)];
        return s;
    }

    private static EnemyState Enemy(int def, params BuffId[] buffs) =>
        new() { DefId = def, Hp = 40, MaxHp = 40, Buffs = buffs.Select(b => new BuffState(b, 2)).ToList() };

    private static CardInstance Card(CombatState s, int def, int enchant = 0, bool bound = false) =>
        s.Fresh(new CardInstance(def, false, Enchantment: (Enchantment)enchant, EnchantAmount: enchant == 0 ? 0 : 1, Bound: bound));

    private static string Show(OldListener e) =>
        $"{e.Kind}:{e.Who.ModelId}:{e.Who.Instance}@{e.Creature}/{e.Index}/{e.Pile}/{e.Buff}/{e.Enemy?.DefId}";

    private static string Show(OldGap g) => $"{g.Why}:{g.Kind}:{g.Creature}:{g.ModelId}:{g.What}";

    private static IEnumerable<int> KindSets()
    {
        int all = (1 << Enum.GetValues<ListenerKind>().Length) - 1;
        for (int mask = 0; mask <= all; mask++)
            yield return mask;
    }

    [Fact]
    public void AMaskedCombatListingIsTheFullListingWithTheOtherKindsLeftOut()
    {
        var s = RichCombat();
        var w = new OldWorld(s, null, Ids) { Scaling = 7 };
        w.Modifiers.Add(5);
        w.Badges.Add(6);
        var fullL = new List<OldListener>();
        var fullG = new List<OldGap>();
        OldState.Combat(w, fullL, fullG);
        Assert.Contains(fullL, e => e.Kind == ListenerKind.Orb);
        Assert.Contains(fullL, e => e.Kind == ListenerKind.Enchantment);
        foreach (int kinds in KindSets())
        {
            var l = new List<OldListener>();
            var g = new List<OldGap>();
            OldState.Combat(w, l, g, kinds);
            Assert.Equal(fullL.Where(e => OldState.Wants(kinds, e.Kind)).Select(Show), l.Where(e => OldState.Wants(kinds, e.Kind)).Select(Show));
            Assert.Equal(fullG.Where(x => OldState.Wants(kinds, x.Kind)).Select(Show), g.Where(x => OldState.Wants(kinds, x.Kind)).Select(Show));
            Assert.All(g, x => Assert.True(fullG.Select(Show).Contains(Show(x))));
        }
    }

    [Fact]
    public void AMaskedRunListingIsTheFullListingWithTheOtherKindsLeftOut()
    {
        var s = RichCombat();
        var run = new RunState { Deck = [new CardInstance(20, false), new CardInstance(21, false, Enchantment: (Enchantment)1)] };
        run.Relics = [new RelicInstance(4)];
        var w = new OldWorld(s, run, Ids) { Scaling = 7 };
        w.Modifiers.Add(5);
        foreach (bool items in new[] { false, true })
        {
            var fullL = new List<OldListener>();
            var fullG = new List<OldGap>();
            OldState.Run(w, fullL, fullG, items);
            foreach (int kinds in KindSets())
            {
                var l = new List<OldListener>();
                var g = new List<OldGap>();
                OldState.Run(w, l, g, items, kinds);
                Assert.Equal(fullL.Where(e => OldState.Wants(kinds, e.Kind)).Select(Show), l.Where(e => OldState.Wants(kinds, e.Kind)).Select(Show));
                Assert.Equal(fullG.Where(x => OldState.Wants(kinds, x.Kind)).Select(Show), g.Where(x => OldState.Wants(kinds, x.Kind)).Select(Show));
            }
        }
    }

    /// <summary>The form of <c>OldState.Powers</c> the scan replaced: a set of (model, k) and a count by BuffId.</summary>
    private static List<string> ReferencePowers(OldWorld w, List<BuffState> buffs, int creature)
    {
        var seen = new HashSet<(int, int)>();
        var count = new Dictionary<BuffId, int>();
        var got = new List<string>();
        for (int i = 0; i < buffs.Count; i++)
        {
            var b = buffs[i];
            var m = w.Ids.Power(b.Id);
            if (m.Kind is PowerMapKind.EngineOnly or PowerMapKind.Folded or PowerMapKind.NotModelled)
                continue;
            int k = count.GetValueOrDefault(b.Id);
            count[b.Id] = k + 1;
            if (seen.Add((m.ModelId, k)))
                got.Add($"{m.ModelId}:{OldState.PowerInstance(creature, k)}:{i}:{b.Id}");
        }

        return got;
    }

    [Fact]
    public void TheScanFormOfPowersListsWhatTheSetAndCountFormListed()
    {
        var r = new Random(11);
        var ids = Enum.GetValues<BuffId>();
        for (int round = 0; round < 400; round++)
        {
            var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
            // few distinct ids, so repeats are common; some draws from the whole enum, so folded, piece and bookkeeping ones appear
            var pool = Enumerable.Range(0, r.Next(1, 8)).Select(_ => ids[r.Next(ids.Length)]).ToArray();
            int n = r.Next(0, 30);
            s.PlayerBuffs = Enumerable.Range(0, n).Select(_ => new BuffState(r.Next(4) == 0 ? ids[r.Next(ids.Length)] : pool[r.Next(pool.Length)], r.Next(1, 5))).ToList();
            s.Enemies = [Enemy(16)];
            var w = new OldWorld(s, null, Ids);
            var l = new List<OldListener>();
            OldState.Combat(w, l, new List<OldGap>(), 1 << (int)ListenerKind.Power);
            var mine = l.Where(e => e.Creature == 0).Select(e => $"{e.Who.ModelId}:{e.Who.Instance}:{e.Index}:{e.Buff}").ToList();
            Assert.Equal(ReferencePowers(w, s.PlayerBuffs, 0), mine);
        }
    }

    [Fact]
    public void PowerByBuffIdIsTheDictionaryItReplaced()
    {
        var table = (Dictionary<BuffId, PowerMapping>)typeof(OldModelIds).GetField("_powers", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(Ids)!;
        foreach (var id in Enum.GetValues<BuffId>())
            Assert.Equal(table[id], Ids.Power(id));
        Assert.Equal(new PowerMapping(PowerMapKind.NotModelled, 0, ""), Ids.Power((BuffId)9999));
        Assert.Equal(new PowerMapping(PowerMapKind.NotModelled, 0, ""), Ids.Power((BuffId)(-1)));
    }

    // ------------------------------------------------------------------------------------------ the handler index

    private static (HandlerTable Table, List<(HookId, ListenerKind, int)> Added) SomeHandlers()
    {
        var table = new HandlerTable(Ids, _ => true);
        var map = JsonNode.Parse(File.ReadAllText(Path.Combine(RepoRoot(), "data", "id_map.json")))!.AsObject();
        var added = new List<(HookId, ListenerKind, int)>();
        var hooks = new[] { HookId.AfterCardPlayed, HookId.ModifyDamageAdditive, HookId.AfterDamageReceived };
        int n = 0;
        foreach (var (kind, section) in new[] { (ListenerKind.Relic, "relics"), (ListenerKind.Power, "powers"), (ListenerKind.Card, "cards") })
        {
            foreach (var (name, id) in map[section]!.AsObject().Take(6))
            {
                var hook = hooks[n++ % hooks.Length];
                if (table.Model(kind, name).On(hook, EveryListener.Noop) is { } reg)
                    added.Add((hook, kind, (int)id!));
            }
        }

        return (table, added);
    }

    private static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "data", "id_map.json")))
            dir = dir.Parent;
        return dir!.FullName;
    }

    [Fact]
    public void TheHandlerIndexAnswersAsTheKeyedDictionaryDid()
    {
        var (table, added) = SomeHandlers();
        Assert.NotEmpty(added);
        var byKey = added.ToHashSet();
        foreach (var hook in new[] { HookId.AfterCardPlayed, HookId.ModifyDamageAdditive, HookId.AfterDamageReceived, HookId.AfterCardDrawn })
        {
            int expectedKinds = 0;
            foreach (var kind in Enum.GetValues<ListenerKind>())
            {
                bool any = added.Any(a => a.Item1 == hook && a.Item2 == kind);
                Assert.Equal(any, table.AnyOf(hook, kind));
                if (any)
                    expectedKinds |= 1 << (int)kind;
                for (int model = 0; model < 700; model++)
                    Assert.Equal(byKey.Contains((hook, kind, model)), table.Find(hook, kind, model) is not null);
            }

            Assert.Equal(expectedKinds, table.KindsOf(hook));
            Assert.Equal(expectedKinds != 0, table.HasAny(hook));
        }
    }

    // ------------------------------------------------------------------------------------------ passes with many listeners

    [Fact]
    public void APassKeepingManyListenersVisitsThemInOrderAndChecksMembership()
    {
        var s = new CombatState { PlayerHp = 50, PlayerMaxHp = 50 };
        s.Enemies = [Enemy(16)];
        for (int i = 0; i < 40; i++)
            s.Hand.Add(Card(s, 20 + i % 5));
        s.Relics = Enumerable.Range(1, 12).Select(i => new RelicInstance(i)).ToList();
        var w = new OldWorld(s, null, Ids);
        var expected = OldListeners.Enumerate(w, HookIterator.Combat).Select(e => e.Who).ToList();
        var cur = new OldListeners(w, EveryListener.Instance).Cursor<Action>(HookIterator.Combat, HookId.AfterCardPlayed);
        var seen = new List<ListenerRef>();
        while (cur.Next(out var l))
        {
            seen.Add(l.Who);
            if (seen.Count == 3)
            {
                s.Hand.RemoveAt(10); // the card is not found when its turn comes: it is skipped, the rest keep their order
            }
        }

        Assert.True(expected.Count > 40);
        var skipped = expected.Single(x => !seen.Contains(x));
        Assert.Equal(expected.Where(x => x != skipped), seen);
    }

    // ------------------------------------------------------------------------------------------ the action history

    [Fact]
    public void ACopyOfTheQueueSharesTheFinishedRecordsAndClonesTheLiveOnes()
    {
        var q = new ActionQueueSet();
        q.CombatStarted();
        var done = q.Enqueue(new GameActionRec(GameActionKind.PlayCard, GameActionType.CombatPlayPhaseOnly) { Action = 1 });
        q.Start(done);
        q.Finish(done);
        var waiting = q.Enqueue(new GameActionRec(GameActionKind.EndTurn, GameActionType.CombatPlayPhaseOnly) { Action = 2 });
        var copy = q.Copy();

        Assert.Equal(q.All.Select(a => (a.Id, a.Kind, a.State, a.Action)), copy.All.Select(a => (a.Id, a.Kind, a.State, a.Action)));
        Assert.Same(done, copy.All[0]); // finished and in no queue: never written again
        Assert.NotSame(waiting, copy.All[1]); // waiting in a queue: the copy's own
        Assert.Same(copy.All[1], copy.QueueOf()[0]); // and the queue holds the copy's record
        copy.Start(copy.All[1]);
        Assert.Equal(GameActionState.WaitingForExecution, waiting.State); // the source is untouched
        Assert.Equal(GameActionState.Executing, copy.All[1].State);
        Assert.Single(q.QueueOf());
        Assert.Equal(q.NextId, copy.NextId);
    }
}
