// The draw observers (docs/rebuild/FIGHT_SOLVER.md section 17): the solver's table of tests, DrawObservers.All, holds
// exactly the observers tools/solver/draw_observers.json names (scan_draw_observers.py writes it from the engine's draw
// loop, its draw-result readers and the AfterCardDrawn listeners, each entry classed with why; --check fails on an
// unclassed one), and each observer's test switches the macro draw off.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.Solver;

public class DrawObserverTests
{
    [Fact]
    public void TheTableHoldsTheRegistrysObservers()
    {
        var doc = JsonNode.Parse(File.ReadAllText(Path.Combine(KeyFieldCoverageTests.Repo(), "tools", "solver", "draw_observers.json")))!;
        var named = doc["observers"]!.AsArray().Select(x => (string)x!).OrderBy(x => x, StringComparer.Ordinal).ToList();
        Assert.Equal(named, DrawObservers.All.Select(o => o.Name).OrderBy(x => x, StringComparer.Ordinal).ToList());
        foreach (var (key, e) in doc["entries"]!.AsObject())
        {
            string cls = (string)e!["class"]!;
            Assert.True(cls is "observer" or "safe", key);
            Assert.True(cls == "safe" || named.Contains((string)e["observer"]!), key);
        }
    }

    [Theory]
    [InlineData("Hellraiser", "Hellraiser")]
    [InlineData("ChainsOfBinding", "ChainsOfBinding")]
    [InlineData("Speedster", "Speedster")]
    [InlineData("Iteration", "Iteration")]
    [InlineData("Pagestorm", "Pagestorm")]
    [InlineData("StratagemPower", "Stratagem")]
    public void APowerObserverSwitchesTheMacroDrawOff(string buff, string observer)
    {
        var s = Plain();
        foreach (var e in s.Enemies)
        {
            e.Hp = 15;  // within Speedster's reach
        }

        Assert.Null(DrawObservers.Off(s));
        BuffSystem.Apply(s.PlayerBuffs, Enum.Parse<BuffId>(buff), 2);
        Assert.Equal("observer:" + observer, DrawObservers.Off(s));
    }

    [Theory]
    [InlineData("Pillage", "observer:Pillage")]
    [InlineData("Scrape", "observer:Scrape")]
    [InlineData("CalculatedGamble", "hand-ordered")]
    public void ACardObserverOrAnOrderedHandSwitchesItOff(string card, string why)
    {
        var s = Plain();
        s.DiscardPile.Add(s.Fresh(new CardInstance(Cards.FindId(card)!.Value, false)));
        Assert.Equal(why, DrawObservers.Off(s));
    }

    [Fact]
    public void SpeedsterIsAnObserverOnlyWhereItsHitsCanKill()
    {
        var s = Plain();
        BuffSystem.Apply(s.PlayerBuffs, BuffId.Speedster, 1);
        foreach (var e in s.Enemies)
        {
            e.Hp = 200;
        }

        Assert.Null(DrawObservers.Off(s));
        s.Enemies[0].Hp = 10;
        Assert.Equal("observer:Speedster", DrawObservers.Off(s));
    }

    static CombatState Plain() =>
        Fight.Hand(new CardInstance(IC.DefendIronclad, false)).Energy(3).Draw(new CardInstance(IC.StrikeIronclad, false), new CardInstance(IC.DefendIronclad, false)).Discard().State;
}
