// OldAnchor.PatchBuffs writes the game's power list over the emulator's buffs through the comparer's alias
// table (tools/fidelity/power_aliases.json). A game power the emulator holds in another buff must be written
// as that buff, and the buff must not be taken away because the game has no power of its name: Self-Forming
// Clay's power is the emulator's BlockNextTurn, so after a Bloodletting under the relic the write-back removed
// the next-turn block the emulator had just earned.
using System.Text.Json.Nodes;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Xunit;

public class ReanchorPowersTests
{
    static JsonArray Game(params (string Name, int Amount)[] powers) =>
        new(powers.Select(p => (JsonNode)new JsonArray(p.Name, p.Amount)).ToArray());

    static (bool Patched, List<BuffState> Result) Patch(List<BuffState> cur, JsonArray game)
    {
        bool patched = OldAnchor.PatchBuffs(OldReanchor.Tables, cur, game, out var result, new JsonArray(), "player.powers");
        return (patched, result);
    }

    [Fact]
    public void SelfFormingClayIsTheEmulatorsBlockNextTurnAndIsLeftAlone()
    {
        var cur = new List<BuffState> { new(BuffId.BlockNextTurn, 3) };
        var (patched, result) = Patch(cur, Game(("SELF_FORMING_CLAY_POWER", 3)));

        Assert.False(patched);  // it said nothing the emulator lacked; it used to be dropped as "no buff" and the buff removed
        Assert.Equal([new BuffState(BuffId.BlockNextTurn, 3)], result);
    }

    [Fact]
    public void SelfFormingClayIsWrittenAsBlockNextTurn()
    {
        var (patched, result) = Patch([], Game(("SELF_FORMING_CLAY_POWER", 3)));

        Assert.True(patched);
        Assert.Equal([new BuffState(BuffId.BlockNextTurn, 3)], result);
    }

    [Fact]
    public void ClayBesideTheGamesOwnBlockNextTurnIsOneSummedBuff()
    {
        // Dodge and Roll's BlockNextTurnPower (5) and a Clay hit (3): the game holds two powers, the emulator one buff of 8.
        var game = Game(("BLOCK_NEXT_TURN_POWER", 5), ("SELF_FORMING_CLAY_POWER", 3));
        Assert.False(Patch([new BuffState(BuffId.BlockNextTurn, 8)], game).Patched);

        var (patched, result) = Patch([], game);
        Assert.True(patched);
        Assert.Equal([new BuffState(BuffId.BlockNextTurn, 8)], result);  // not two entries of one buff
    }

    [Fact]
    public void ABlockNextTurnAloneIsWrittenAsItWas()
    {
        var (patched, result) = Patch([], Game(("BLOCK_NEXT_TURN_POWER", 5)));

        Assert.True(patched);
        Assert.Equal([new BuffState(BuffId.BlockNextTurn, 5)], result);
    }

    [Fact]
    public void TemporaryStrengthSubclassesStillFoldIntoOneBuff()
    {
        var (patched, result) = Patch([], Game(("MANGLE_POWER", 10), ("DARK_SHACKLES_POWER", 9), ("STRENGTH_POWER", -9)));

        Assert.True(patched);
        Assert.Equal(2, result.Count);
        Assert.Contains(new BuffState(BuffId.TemporaryStrength, 19), result);
        Assert.Contains(new BuffState(BuffId.Strength, -9), result);
    }
}
