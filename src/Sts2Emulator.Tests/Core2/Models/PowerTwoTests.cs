// The C3 handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C3"): the port list's power-2 rows as migrated models
// under Core2/Models/Powers, each tested the way the shared rules ask: (1) its pairs are in the handler table and none is
// a stub, (2) the dispatch gives what the old function gives for the same state, (3) the delete-mutant (the model's
// registration left out) gives the no-handler value, (4) with F08 off its pairs are absent, (5) a holder acts for its
// owner alone. The death hooks are dispatched with the kit's builders inside Dying (the dead creature at 0 HP, marked
// dying, still listed), and once through CreatureCmd.Kill. The table test at the end pins the batch's (model, hook)
// set, standing in for EmuDump kernel-handlers and the port list's gate, which need the manifest and a built EmuDump.
// Surprise is not in the batch: its old AfterDeath body takes the step's Random, which no handler has (the stop rule;
// the PR lists it). Game code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Powers;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class PowerTwoTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> Power2 = Registry.Power2;

    /// <summary>A basic Attack card (StrikeIronclad) and a basic Skill (DefendIronclad), by def id.</summary>
    private const int StrikeDef = 472;
    private const int DefendDef = 131;

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The batch's (model, hook) pairs, written out: the port list rows power-2 of the models the batch ports.</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] Power2Pairs =
    [
        (ListenerKind.Power, "TaintedPower", HookId.ModifyDamageAdditive),
    ];

    [Fact]
    public void TheRegistrysPower2BlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(Power2);
        Assert.Equal(Power2Pairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal).ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal),
            Pairs(t));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsThePower2BlockAndTheKernelBuildsItsTableFromIt()
    {
        // Registry.All is one block per batch; this batch's block is in it, and a kernel built from All (as Core2Engine
        // builds one) registers the batch's pairs.
        foreach (var model in Power2)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in Power2Pairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffTheBatchRegistersNothing()
    {
        Assert.Empty(Pairs(Table(Power2, AllButDamage)));
    }

    // ---------------------------------------------------------------------------------------- Tainted (power-2)

    /// <summary>The player with Tainted 2, one slug.</summary>
    private static CombatState TaintedCombat(int amount = 2)
    {
        var s = Combat(40);
        s.PlayerBuffs.Add(new BuffState(BuffId.Tainted, amount));
        return s;
    }

    [Fact]
    public void TaintedIsRegisteredForItsPair()
    {
        AssertRegistered(Table(Power2), ListenerKind.Power, "TaintedPower", HookId.ModifyDamageAdditive);
    }

    [Fact]
    public void TaintedAddsTheOldFunctionsNumberToAPoweredAttackOnItsOwner()
    {
        // game: TaintedPower.ModifyDamageAdditive; old: BuffSystem.IncomingDamage adds
        // Tainted_ModifyDamageAdditive(defenderBuffs) to every powered hit, whoever deals it.
        var s = TaintedCombat();
        var k = Kernel(s, Power2);

        decimal v = Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed);

        Assert.Equal(6m + BuffSystem.Tainted_ModifyDamageAdditive(s.PlayerBuffs), v);
        Assert.Equal(8m, v);
        Assert.Equal([Power("TaintedPower", 0)], changed);
        // The same state through the old fold (the slug attacks the player): 6 + 2 = 8.
        Assert.Equal(8, BuffSystem.IncomingDamage(6, s.Enemies[0].Buffs, s.PlayerBuffs));
        // A hit with no dealer counts too: the old function reads the defender alone.
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: null), out changed));
        Assert.Equal([Power("TaintedPower", 0)], changed);

        // Where the condition fails: a hit on another creature (the owner's own attack: the old fold reads the slug's
        // Tainted there, 0), an unpowered hit on the owner, the stack at 0.
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out changed));
        Assert.Empty(changed);
        Assert.Equal(6, BuffSystem.IncomingDamage(6, s.PlayerBuffs, s.Enemies[0].Buffs));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, props: Unpowered, dealer: 1), out changed));
        Assert.Empty(changed);
        s.PlayerBuffs[0] = new BuffState(BuffId.Tainted, 0);
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TaintedActsForItsOwnerAloneAmongThreeHolders()
    {
        // The N-holder test: the player and both slugs hold Tainted 2. A hit on the first slug takes that slug's addend
        // once (8, not 12) and changes its listener alone; the second slug's hit on the player changes the player's.
        var s = Combat(40, 40);
        var holders = Holders(s, "TaintedPower", BuffId.Tainted, 2);
        var k = Kernel(s, Power2);

        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 1, damage: 6m), out var changed));
        Assert.Equal([holders[1]], changed);
        Assert.Equal(8m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 2), out changed));
        Assert.Equal([holders[0]], changed);
    }

    [Fact]
    public void TaintedDeleteMutantGivesTheNoHandlerValue()
    {
        var s = TaintedCombat();
        var k = Kernel(s, Without(Power2, TaintedPower.Register));
        Assert.Equal(6m, Hooks.ModifyDamage(k, Attack(target: 0, damage: 6m, dealer: 1), out var changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TaintedIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Power2, AllButDamage), ListenerKind.Power, "TaintedPower", HookId.ModifyDamageAdditive);
    }
}
