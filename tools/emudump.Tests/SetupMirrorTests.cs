// The setup mirror (tools/emudump/Setup.cs) against a synthetic referee record: cards the referee's setup put
// into a combat pile land where CardPileCmd.Add puts them (the end of the hand and the discard pile, the top or
// the bottom of the draw pile), and a combat pile named before the fight is a mirror gap, never a silent skip.
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

public class SetupMirrorTests
{
    const string Seed = "MIRRORTST1";

    static JsonObject Card(string id, string pile, string? pos = null)
    {
        var o = new JsonObject { ["cmd"] = "card", ["id"] = id, ["pile"] = pile };
        if (pos != null) o["pos"] = pos;
        return o;
    }

    static JsonObject FightItem() => new() { ["cmd"] = "fight", ["id"] = "NIBBITS_WEAK" };

    /// <summary>A record as the referee writes one: Neow's first option, then the first map option, meeting Nibbits.</summary>
    static JsonObject Record(JsonArray setup)
    {
        var e = new RunEngine();
        e.Reset(Seed, 10);
        e.Step(0, -1, out _, out _, out _);
        int guard = 0;
        while (e.State.Phase != RunPhase.Map && guard++ < 10)
        {
            e.Step(0, -1, out _, out _, out _);
            if (e.State.Phase == RunPhase.BundleSelect) e.Step(RunConstants.BundleConfirmAction, -1, out _, out _, out _);
        }

        var room = e.State.MapOptionCoords.First(x => x != null)!.Value;
        return new JsonObject
        {
            ["seed"] = Seed, ["ascension"] = 10, ["neow_pick"] = 0, ["room"] = new JsonArray(room.Col, room.Row),
            ["setup"] = setup,
            // Where the game lists the encounter (Overgrowth's weak list, second), as the referee records it.
            ["fight"] = new JsonObject { ["id"] = "NIBBITS_WEAK", ["act"] = "OVERGROWTH", ["kind"] = "weak", ["index"] = 1 },
        };
    }

    [Fact]
    public void CardsGoIntoTheCombatPileAndPlaceTheRefereeNamed()
    {
        var setup = new JsonArray(FightItem(), Card("HAVOC", "Hand"), Card("INFLAME", "Draw", "top"), Card("ANGER", "Draw"),
            Card("BASH", "Discard"));
        var outp = new JsonObject();
        var c = Fight.EnterRoom(Record(setup), outp).State.ActiveCombat!;

        Assert.Equal(Fight.CardIdBySnake("HAVOC"), c.Hand[^1].DefId);
        Assert.Equal(Fight.CardIdBySnake("INFLAME"), c.DrawPile[0].DefId);
        Assert.Equal(Fight.CardIdBySnake("ANGER"), c.DrawPile[^1].DefId);
        Assert.Equal(Fight.CardIdBySnake("BASH"), c.DiscardPile[^1].DefId);
        // The player put them there, so both ends of the draw pile are known.
        Assert.True(c.KnownTopCount >= 1 && c.KnownBottomCount >= 1);
        var items = outp["setup_log"]!.AsArray().Where(x => (int)x!["i"]! >= 1).ToList();
        Assert.Equal(4, items.Count);
        Assert.All(items, x => Assert.True((bool)x!["ok"]! && x["unsupported"] == null, x!.ToJsonString()));
    }

    [Fact]
    public void ACombatPileBeforeTheFightIsAMirrorGap()
    {
        var setup = new JsonArray(Card("HAVOC", "Hand"), FightItem());
        var outp = new JsonObject();
        Fight.EnterRoom(Record(setup), outp);
        var item = outp["setup_log"]!.AsArray().First(x => (int)x!["i"]! == 0)!;
        Assert.False((bool)item["ok"]!);
        Assert.True((bool?)item["unsupported"] == true);
    }
}
