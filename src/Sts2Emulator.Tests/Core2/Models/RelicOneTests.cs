// The C6 handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3, "C6"): the port list's relic-1 rows as migrated models
// under Core2/Models/Relics, each tested the way the shared rules ask: (1) its pairs are in the handler table and none is
// a stub, (2) the dispatch gives what the old function gives for the same state, (3) the delete-mutant (the model's
// registration left out) gives the no-handler value, (4) with F08 off its pairs are absent, (5) the holder rule: a relic
// has one holder, the player, and no second copy in a combat, so the two-holder setup of rule 5 has no relic case; what it
// asks, that a handler acts for its owner alone, is pinned per model by the dispatches about another creature (Tungsten
// Rod on an enemy's or Osty's loss, Gremlin Horn on a death on the owner's side) and by the two relics of one old function
// each landing once (Strike Dummy with Fake Strike Dummy). The table test at the top pins the batch's (model, hook) set,
// standing in for EmuDump kernel-handlers and the port list's gate, which need the manifest and a built EmuDump. Game
// code is cited by class and method only.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core2;
using Sts2Emulator.Core2.Models;
using Sts2Emulator.Core2.Models.Relics;
using Xunit;
using static Sts2Emulator.Tests.Core2.Models.ModelTestKit;

namespace Sts2Emulator.Tests.Core2.Models;

public class RelicOneTests
{
    private static readonly IReadOnlyList<Action<HandlerTable>> Relic1 = Registry.Relic1;

    // ---------------------------------------------------------------------------------------- the batch's table

    /// <summary>The batch's (model, hook) pairs, written out: the port list's relic-1 rows the briefs mark CLOUD.</summary>
    private static readonly (ListenerKind Kind, string Model, HookId Hook)[] Relic1Pairs =
    [
        (ListenerKind.Relic, "TungstenRod", HookId.AfterModifyingHpLostAfterOsty),
        (ListenerKind.Relic, "TungstenRod", HookId.ModifyHpLostAfterOsty),
    ];

    [Fact]
    public void TheRegistrysRelic1BlockHoldsExactlyTheBatchsPairsAndNoStub()
    {
        var t = Table(Relic1);
        Assert.Equal(Relic1Pairs.OrderBy(p => p.Kind).ThenBy(p => p.Model, StringComparer.Ordinal).ThenBy(p => p.Hook.ToString(), StringComparer.Ordinal),
            Pairs(t));
        Assert.All(t.Registered(), r => Assert.False(r.Stub, $"{r.Kind} {r.GameId} {r.Hook} is a stub"));
    }

    [Fact]
    public void TheRegistrysAllHoldsTheRelic1BlockAndTheKernelBuildsItsTableFromIt()
    {
        // Registry.All is one block per batch; this batch's block is in it, and a kernel built from All (as Core2Engine
        // builds one) registers the batch's pairs.
        foreach (var model in Relic1)
            Assert.Contains(model, Registry.All);
        var all = Pairs(Table(Registry.All));
        foreach (var pair in Relic1Pairs)
            Assert.Contains(pair, all);
    }

    [Fact]
    public void WithTheDamageFamilyOffTheBatchRegistersNothing()
    {
        Assert.Empty(Pairs(Table(Relic1, AllButDamage)));
    }

    // ---------------------------------------------------------------------------------------- the cards the tests play

    /// <summary>An Attack with the Strike tag (StrikeIronclad).</summary>
    private const int StrikeDef = 472;

    /// <summary>An Attack without it (Bash).</summary>
    private const int BashDef = 30;

    /// <summary>A Skill (DefendIronclad).</summary>
    private const int DefendDef = 131;

    private static CardDef Def(int id) => GeneratedData.Cards.Get(id);

    // ---------------------------------------------------------------------------------------- Tungsten Rod (relic-1)

    /// <summary>The player holding Tungsten Rod, one slug.</summary>
    private static (CombatState S, ListenerRef Relic) TungstenRodCombat(int enemyHp = 40)
    {
        var s = Combat(enemyHp);
        return (s, Hold(s, RelicEffects.TungstenRod));
    }

    [Fact]
    public void TungstenRodIsRegisteredForItsTwoPairs()
    {
        AssertRegistered(Table(Relic1), ListenerKind.Relic, "TungstenRod", HookId.ModifyHpLostAfterOsty,
            HookId.AfterModifyingHpLostAfterOsty);
    }

    [Fact]
    public void TungstenRodTakesOneOffTheOwnersHpLossInTheAfterOstyPass()
    {
        // game: TungstenRod.ModifyHpLostAfterOsty; old: CardEffects.LosePlayerHp runs RelicEffects.ModifyHpLost, which
        // applies TungstenRod_ModifyHpLostAfterOsty to the player's loss while the relic is held, whatever brought the loss.
        var (s, relic) = TungstenRodCombat();
        var k = Kernel(s, Relic1);

        decimal v = Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 1), out var changed);

        Assert.Equal(RelicEffects.TungstenRod_ModifyHpLostAfterOsty(3), v);
        Assert.Equal(2m, v);
        Assert.Equal([relic], changed);
        // The same state through the old fold: 3 becomes 2.
        Assert.Equal(2, RelicEffects.ModifyHpLost(s, 3));

        // Every route: an unpowered loss (a Burn, Poison, Thorns), a loss with no dealer, a card's own cost.
        Assert.Equal(2m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, props: Unpowered, dealer: 1), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(2m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: null), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(2m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, props: Unpowered, dealer: 0), out changed));
        Assert.Equal([relic], changed);

        // The floor at zero: one lost becomes nothing; nothing lost stays nothing and counts as no change.
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 1m, dealer: 1), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 0m, dealer: 1), out changed));
        Assert.Empty(changed);

        // Decimals: the running value moves by one and keeps its fraction (3.5 to 2.5, truncated to 2 at the loss; 1.5
        // to 0.5, truncated to 0, which the trunc-ne criterion counts as a change from 1); a loss under 1 is floored to 0
        // (the old engine's cast had made it 0 already) and, truncating to 0 either way, counts as no change.
        Assert.Equal(2.5m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3.5m, dealer: 1), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(0.5m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 1.5m, dealer: 1), out changed));
        Assert.Equal([relic], changed);
        Assert.Equal(0m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 0.5m, dealer: 1), out changed));
        Assert.Empty(changed);

        // Where the condition fails: the before-Osty phase (another pass), a loss of an enemy (the player's hit on the
        // slug), the relic not held.
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 1, phase: HpLossHookPhase.BeforeOsty), out changed));
        Assert.Empty(changed);
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 3m, dealer: 0), out changed));
        Assert.Empty(changed);
        s.Relics.Clear();
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 1), out changed));
        Assert.Empty(changed);
    }

    [Fact]
    public void TungstenRodActsForItsOwnerAndNotForOstyOrAnEnemy()
    {
        // The per-owner condition (CLOUD_BRIEFS section 3, rule 5, for a relic: one holder): a loss of Osty (creature 1
        // once it is in the combat) or of the slug (creature 2) is another creature's, and the rod leaves it.
        var (s, relic) = TungstenRodCombat();
        WithOsty(s);
        var k = Kernel(s, Relic1);

        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 1, 3m, dealer: 2), out var changed));
        Assert.Empty(changed);
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 2, 3m, dealer: 0), out changed));
        Assert.Empty(changed);
        Assert.Equal(2m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 2), out changed));
        Assert.Equal([relic], changed);
    }

    [Fact]
    public void TungstenRodsReductionLandsThroughTheDamageCommandAfterBlock()
    {
        // game: CreatureCmd.Damage runs Hook.ModifyHpLost after the block absorption and the pet redirect, then
        // LoseHpInternal with the one cast. The slug hits the player: 3 on no block, 2 lost. Block 2 against 5: 3 unblocked,
        // 2 lost. Block 5 against 3: fully blocked, nothing lost. 1 on no block: nothing lost. 3.5: 2.5 after the rod,
        // 2 lost at the cast. 0.5: nothing. An unpowered 3 (a Burn's): 2 lost.
        var (s, _) = TungstenRodCombat();
        var slug = OldCreature.Of(s.Enemies[0]);
        var k = Kernel(s, Relic1);

        Assert.Equal(2, Assert.Single(Hit(k, OldCreature.Player, 3m, dealer: slug)).UnblockedDamage);
        Assert.Equal(48, s.PlayerHp);
        s.PlayerBlock = 2;
        Assert.Equal(2, Assert.Single(Hit(k, OldCreature.Player, 5m, dealer: slug)).UnblockedDamage);
        Assert.Equal(46, s.PlayerHp);
        s.PlayerBlock = 5;
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Player, 3m, dealer: slug)).UnblockedDamage);
        Assert.Equal(46, s.PlayerHp);
        s.PlayerBlock = 0;
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Player, 1m, dealer: slug)).UnblockedDamage);
        Assert.Equal(46, s.PlayerHp);
        Assert.Equal(2, Assert.Single(Hit(k, OldCreature.Player, 3.5m, dealer: slug)).UnblockedDamage);
        Assert.Equal(44, s.PlayerHp);
        Assert.Equal(0, Assert.Single(Hit(k, OldCreature.Player, 0.5m, dealer: slug)).UnblockedDamage);
        Assert.Equal(44, s.PlayerHp);
        Assert.Equal(2, Assert.Single(Hit(k, OldCreature.Player, 3m, dealer: slug, props: Unpowered)).UnblockedDamage);
        Assert.Equal(42, s.PlayerHp);

        // The old route for the first hit: CardEffects.LosePlayerHp takes 2 off 50 and reports 2.
        var (oracle, _) = TungstenRodCombat();
        Assert.Equal(2, CardEffects.LosePlayerHp(oracle, 3));
        Assert.Equal(48, oracle.PlayerHp);
    }

    [Fact]
    public void TungstenRodsAfterModifyingHpLostAfterOstyDoesNothingToTheState()
    {
        // game: TungstenRod.AfterModifyingHpLostAfterOsty, reached by the filter-set pass when the relic modified the loss.
        // The old engine has no body at this point; the handler returns Done and the state is as it was.
        // NO DELETE-MUTANT for this pair (CLOUD_BRIEFS section 3, rule 3, does not apply): the game's override only flashes
        // the relic (presentation, no combat state), so with the handler deleted the state is the same and no assertion on
        // it could fail. The pair is pinned instead by TungstenRodIsRegisteredForItsTwoPairs and the table test. A mutant
        // is not faked here.
        var (s, relic) = TungstenRodCombat();
        var k = Kernel(s, Relic1);
        var status = Hooks.AfterModifyingHpLostAfterOsty(k, new AfterModifyingHpLostAfterOstyArgs([relic]));
        Assert.False(status.IsPaused);
        Assert.Equal(50, s.PlayerHp);
        Assert.Equal(40, s.Enemies[0].Hp);
        Assert.Equal([new RelicInstance(RelicEffects.TungstenRod)], s.Relics);
    }

    [Fact]
    public void TungstenRodDeleteMutantGivesTheNoHandlerValue()
    {
        // The fold pair (ModifyHpLostAfterOsty) only: the after-modifying pair has no mutant, see above.
        var (s, _) = TungstenRodCombat();
        var k = Kernel(s, Without(Relic1, TungstenRod.Register));
        Assert.Equal(3m, Hooks.ModifyHpLost(k, HpLoss(target: 0, 3m, dealer: 1), out var changed));
        Assert.Empty(changed);
        Assert.Equal(3, Assert.Single(Hit(k, OldCreature.Player, 3m, dealer: OldCreature.Of(s.Enemies[0]))).UnblockedDamage);
        Assert.Equal(47, s.PlayerHp);
    }

    [Fact]
    public void TungstenRodIsAbsentWithTheDamageFamilyOff()
    {
        AssertAbsent(Table(Relic1, AllButDamage), ListenerKind.Relic, "TungstenRod", HookId.ModifyHpLostAfterOsty,
            HookId.AfterModifyingHpLostAfterOsty);
    }

}
