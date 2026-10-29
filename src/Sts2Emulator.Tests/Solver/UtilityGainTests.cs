// The scanned utility-gain registry (docs/rebuild/FIGHT_SOLVER.md section 19): the C# table held to utility_gains.json,
// every name resolved, a holder anywhere in the combat turning the bound off, and the tight bound's arithmetic.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Solver;
using Xunit;

namespace Sts2Emulator.Tests.Solver;

public class UtilityGainTests
{
    static JsonNode Registry() =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(KeyFieldCoverageTests.Repo(), "tools", "solver", "utility_gains.json")))!;

    [Fact]
    public void TheHoldersAreTheScannedRegistrysAndNameSomething()
    {
        var holders = Registry()["holders"]!.AsArray().Select(h => (string)h!).ToHashSet();
        var table = UtilityGains.CardNames.Select(n => "card:" + n)
            .Concat(UtilityGains.RelicNames.Select(n => "relic:" + n))
            .Concat(UtilityGains.PotionNames.Select(n => "potion:" + n))
            .Concat(UtilityGains.BuffNames.Select(n => "buff:" + n))
            .Concat(UtilityGains.EnemyNames.Select(n => "enemy:" + n)).ToHashSet();
        Assert.True(holders.SetEquals(table), "UtilityGains.g.cs differs from utility_gains.json: run tools/solver/scan_utility_gains.py");
        Assert.Empty(UtilityGains.Unresolved());
        var ends = Registry()["end"]!.AsArray().Select(e => ((string)e!["holder"]!, (int)e["constant"]!, (string)e["applied"]! == "before-terminal")).ToHashSet();
        Assert.True(ends.SetEquals(UtilityGains.EndHeals.Select(e => ("relic:" + e.Relic, e.Constant, e.BeforeTerminal))));
    }

    [Fact]
    public void AHolderAnywhereTurnsTheBoundOffAndTheTightBoundIsL0OfTheHpAndPotions()
    {
        var (s, _) = SolverPositions.Fight("old", 11, [.. TestDeck.StarterDeckIds], [], [], hp: 50);
        Assert.Null(UtilityGains.Off(s));
        Assert.Equal(Utility.L0(s), UtilityGains.Tight(s));
        Assert.Equal(Utility.L0(s), UtilityGains.Upper(s));
        int feed = Cards.FindId("Feed")!.Value;
        var withFeed = s.Clone();
        withFeed.DiscardPile.Add(withFeed.Fresh(new CardInstance(feed, false)));
        Assert.Equal("card:Feed", UtilityGains.Off(withFeed));
        Assert.Equal(Utility.Max, UtilityGains.Upper(withFeed));
        var withRelic = s.Clone();
        withRelic.Relics.Add(new RelicInstance(Relics.FindId("LizardTail")!.Value));
        Assert.Equal("relic:LizardTail", UtilityGains.Off(withRelic));
        var withRegen = s.Clone();
        BuffSystem.Apply(withRegen.PlayerBuffs, BuffId.Regen, 3);
        Assert.Equal("buff:Regen", UtilityGains.Off(withRegen));
        var withPotion = s.Clone();
        withPotion.PotionSlots[0] = Potions.FindId("FirePotion")!.Value;
        Assert.Null(UtilityGains.Off(withPotion));
        Assert.Equal(Utility.L0(withPotion), UtilityGains.Tight(withPotion));
        Assert.Equal(Utility.L0(s), UtilityGains.Tight(withPotion, -1));
        withPotion.PotionSlots[1] = Potions.FindId("BloodPotion")!.Value;
        Assert.Equal("potion:BloodPotion", UtilityGains.Off(withPotion));
        // Burning Blood heals after the terminal is scored: present, the bound stays on and adds nothing.
        var withBlood = s.Clone();
        withBlood.Relics.Add(new RelicInstance(Relics.FindId("BurningBlood")!.Value));
        Assert.Null(UtilityGains.Off(withBlood));
        Assert.Equal(0, UtilityGains.EndConstant(withBlood));
    }
}
