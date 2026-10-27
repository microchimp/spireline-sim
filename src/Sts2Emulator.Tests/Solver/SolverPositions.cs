using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core2;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;

namespace Sts2Emulator.Tests.Solver;

/// <summary>
/// The harness's small positions, built from tests/fixtures/solver/positions.jsonl (written by tools/solver/harness.py):
/// a corpus fixture's deck, relics and potions or self-play's default deck, seeded random play into the fight, the cut
/// (enemy HP capped, draw and discard piles trimmed), the steps after it (to open screens and mid-turn positions), the
/// card played last to stand on its screen. A recipe may name its fight (<c>encounter</c>), the corpus file whose deck
/// it takes (<c>fixture</c>) and a buff the player waits for, ending turns, before the cut (<c>until</c>: <c>player:Tender</c>,
/// <c>enemy:Thorns</c>).
/// </summary>
internal static class SolverPositions
{
    public sealed record Recipe(int Id, string Source, string Engine, int Seed, int Setup, int Steps, int EnemyHp, int DrawKeep, int DiscardKeep, string[] Mix, int After, int Horizon, string Play, string Encounter, string Until, string Fixture);

    public sealed record Position(Recipe Recipe, CombatState State, CountingRandom Rng);

    public static List<Recipe> Recipes()
    {
        var path = Path.Combine(KeyFieldCoverageTests.Repo(), "tests", "fixtures", "solver", "positions.jsonl");
        var opts = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
        return [.. File.ReadAllLines(path).Where(l => l.Length > 0).Select(l => JsonSerializer.Deserialize<Recipe>(l, opts)!)];
    }

    public static Kernel? KernelFor(string engine) =>
        engine == "core2" ? new Kernel(KernelOptions.FromFamilies(["F08", "F10"]), () => Sts2Emulator.Tests.Core2.ListenerOrderTests.Ids, Sts2Emulator.Core2.Models.Registry.All) : null;

    /// <summary>A fight on <paramref name="engine" /> with every stream a CountingRandom, its pre-play phase resolved.</summary>
    public static (CombatState State, CountingRandom Rng) Fight(string engine, int seed, int[] deck, int[] relics, int[] potions, int hp = 80, int? encounter = null)
    {
        var s = new CombatState();
        KernelFor(engine)?.Bind(s);
        var rng = new CountingRandom(seed * 101);
        CombatFactory.Reset(s, rng, deck, encounter, relics, hp, 80, potions, 0);
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            if (StateCloning.Stream(s, i) is not CountingRandom)
            {
                StateCloning.SetStream(s, i, new CountingRandom(unchecked(seed * 131 + i)));
            }
        }

        CombatEngine.ResolvePrePlayPhase(s, rng);
        return (s, rng);
    }

    public static Position? Build(Recipe r, List<(string Name, int[] Deck, int[] Relics, int[] Potions)> corpus)
    {
        int[] mix = [.. r.Mix.Select(Cards.FindId).Where(x => x is not null).Select(x => x!.Value)];
        int[] deck, relics = [], potions = [];
        if (r.Fixture.Length > 0)
        {
            var c = corpus.Find(x => x.Name == r.Fixture);
            if (c.Deck is null)
            {
                return null;
            }

            (deck, relics, potions) = (c.Deck, c.Relics, c.Potions);
        }
        else if (r.Source == "corpus" && corpus.Count > 0)
        {
            var c = corpus[r.Setup % corpus.Count];
            (deck, relics, potions) = (c.Deck, c.Relics, c.Potions);
        }
        else
        {
            deck = TestDeck.StarterDeckIds.ToArray();
        }

        try
        {
            int? encounter = r.Encounter.Length > 0 ? (int)Enum.Parse<CombatFactory.ActOneEncounter>(r.Encounter) : null;
            var (s, rng) = Fight(r.Engine, r.Seed, [.. deck, .. mix], relics, potions, encounter: encounter);
            var policy = new Random(r.Seed * 7919);
            for (int i = 0; i < r.Steps; i++)
            {
                if (!Play(s, rng, policy, endTurn: true))
                {
                    return null;
                }
            }

            for (int i = 0; r.Until.Length > 0 && !Reached(s, r.Until); i++)
            {
                if (i >= 40 || !Pass(s, rng, policy))
                {
                    return null;
                }
            }

            if (s.PendingSelection is null)
            {
                Cut(s, r, mix);
            }

            for (int i = 0; i < r.After; i++)
            {
                if (!Play(s, rng, policy, endTurn: false))
                {
                    return null;
                }
            }

            if (r.Play.Length > 0 && s.PendingSelection is null && Cards.FindId(r.Play) is int play)
            {
                int at = s.Hand.FindIndex(c => Math.Abs(c.DefId) == play);
                var m = Actions.Legal(s, Pruning.None).Where(x => at >= 0 && x.Action == at).Take(1).ToList();
                if (m.Count > 0 && Successors.Step(s, rng, m[0]).Terminal)
                {
                    return null;
                }
            }

            return new Position(r, s, rng);
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return null;  // a setup the engine does not take: no position from it
        }
    }

    /// <summary>The player ends the turn (a screen in the way is answered at random): the enemies act, nothing of theirs is killed on the way.</summary>
    static bool Pass(CombatState s, CountingRandom rng, Random policy)
    {
        var moves = Actions.Legal(s, Pruning.None);
        var m = s.PendingSelection is null ? moves.First(x => Actions.EndsTurn(s, x)) : moves[policy.Next(moves.Count)];
        return !Successors.Step(s, rng, m).Terminal;
    }

    /// <summary>The player (<c>player:Name</c>) or a living enemy (<c>enemy:Name</c>) holds the buff, on the player's turn with no screen open.</summary>
    static bool Reached(CombatState s, string until)
    {
        var parts = until.Split(':');
        var id = Enum.Parse<BuffId>(parts[1]);
        return s.PendingSelection is null && (parts[0] == "player"
            ? BuffSystem.Get(s.PlayerBuffs, id) > 0
            : s.Enemies.Exists(e => e.Hp > 0 && BuffSystem.Get(e.Buffs, id) > 0));
    }

    /// <summary>One seeded random legal move (end turn only when allowed or alone); false when the fight ended.</summary>
    static bool Play(CombatState s, CountingRandom rng, Random policy, bool endTurn)
    {
        var moves = Actions.Legal(s, Pruning.None);
        if (!endTurn && s.PendingSelection is null && moves.Count > 1)
        {
            moves = [.. moves.Where(m => !Actions.EndsTurn(s, m))];
        }

        var m = moves[policy.Next(moves.Count)];
        return !Successors.Step(s, rng, m).Terminal;
    }

    /// <summary>
    /// The cut: the mixed-in cards brought to the hand (so their screens, draws and rolls are met), living enemies' HP
    /// capped, the draw pile's tail and the discard trimmed (cards a card_ref points at kept).
    /// </summary>
    static void Cut(CombatState s, Recipe r, int[] mix)
    {
        foreach (int id in mix)
        {
            if (s.Hand.Count >= 10 || s.Hand.Exists(c => Math.Abs(c.DefId) == id))
            {
                continue;
            }

            foreach (var pile in new[] { s.DrawPile, s.DiscardPile })
            {
                int at = pile.FindIndex(c => Math.Abs(c.DefId) == id);
                if (at >= 0)
                {
                    s.Hand.Add(pile[at]);
                    pile.RemoveAt(at);
                    break;
                }
            }
        }

        s.KnownTopCount = 0;
        s.KnownBottomCount = 0;
        foreach (var e in s.Enemies)
        {
            if (e.Hp > 0)
            {
                e.Hp = Math.Min(e.Hp, r.EnemyHp);
            }
        }

        var pointed = s.Enemies.Select(e => e.CurlUpCardUid).Where(u => u != 0).ToHashSet();
        Trim(s.DrawPile, r.DrawKeep, pointed);
        Trim(s.DiscardPile, r.DiscardKeep, pointed);
        s.KnownTopCount = Math.Clamp(s.KnownTopCount, 0, s.DrawPile.Count);
        s.KnownBottomCount = Math.Clamp(s.KnownBottomCount, 0, s.DrawPile.Count - s.KnownTopCount);
    }

    static void Trim(List<CardInstance> pile, int keep, HashSet<int> pointed)
    {
        for (int i = pile.Count - 1; i >= 0 && pile.Count > keep; i--)
        {
            if (!pointed.Contains(pile[i].Uid))
            {
                pile.RemoveAt(i);
            }
        }
    }

    /// <summary>The corpus's setups: each combat fixture's deck, relics and potions, by the game's ids.</summary>
    public static List<(string Name, int[] Deck, int[] Relics, int[] Potions)> Corpus()
    {
        var list = new List<(string, int[], int[], int[])>();
        var dir = Path.Combine(KeyFieldCoverageTests.Repo(), "tests", "fixtures", "combat");
        foreach (var f in Directory.GetFiles(dir, "*.json").OrderBy(x => x, StringComparer.Ordinal))
        {
            JsonNode? p;
            try
            {
                p = JsonNode.Parse(File.ReadAllText(f))?["player"];
            }
            catch (Exception)
            {
                continue;
            }

            if (p is null)
            {
                continue;
            }

            int[] Ids(string key, Func<string, int?> find) =>
                [.. ((p[key] as JsonArray) ?? []).Select(x => x is JsonObject o ? (string?)o["id"] : null).Where(x => x is not null).Select(x => find(Pascal(x!))).Where(x => x is not null).Select(x => x!.Value)];
            var deck = Ids("deck", Cards.FindId);
            if (deck.Length >= 5)
            {
                list.Add((Path.GetFileName(f), deck, Ids("relics", Relics.FindId), Ids("potions", Potions.FindId)));
            }
        }

        return list;
    }

    static string Pascal(string id) => string.Concat(id.Split('_', StringSplitOptions.RemoveEmptyEntries).Select(w => w[..1] + w[1..].ToLowerInvariant()));
}
