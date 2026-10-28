using Sts2Emulator.Core;
using Sts2Emulator.Interop;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The two debug hooks a real run's RUN LAYER is replayed through
/// (scripts/replay_real_run.py): win the fight in progress without playing it, and put the
/// player's holdings back where the real run had them.
/// </summary>
/// <remarks>
/// A real session's fights are measured on their own (the coach's live checks); the run
/// replay only needs to come out of each one where the game did, so every offering after
/// it -- rewards, shops, events, the map -- is compared against the game's own. Neither
/// hook is a game rule, and neither may leave the run anywhere a real fight could not.
/// </remarks>
public class ReplayDebugHookTests
{
    private const int MadScience = 292;

    private static RunEngine AtFirstCombat(string seed)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        for (int i = 0; i < 16 && engine.State.Phase != RunPhase.Map; i++)
        {
            engine.Step(0, -1, out _, out _, out _);
        }

        Assert.Equal(RunPhase.Map, engine.State.Phase);
        int action = Array.FindIndex(
            engine.State.MapNodeTypes,
            nodeType => nodeType == RunConstants.NodeNormal
        );
        Assert.True(action >= 0);
        Assert.Equal(0, engine.Step(action, -1, out _, out _, out _));
        Assert.Equal(RunPhase.Combat, engine.State.Phase);
        return engine;
    }

    /// <summary>
    /// A resolved fight lands on the same reward screen a won one does: the after-combat
    /// sync, then <c>GenerateCombatRewards</c> from the very same stream positions.
    /// </summary>
    [Theory]
    [InlineData("QS2GYXRKWN")]
    [InlineData("QEY9JXP3KR")]
    public void AResolvedFightOffersWhatTheWonFightWould(string seed)
    {
        var resolved = AtFirstCombat(seed);
        var reference = resolved.Clone();

        Assert.Equal(0, resolved.DebugResolveCombat());

        reference.SyncAfterCombatForTest();
        RunRewardGenerator.GenerateCombatRewards(reference.State);

        Assert.Equal(RunPhase.RelicReward, resolved.State.Phase);
        Assert.True(resolved.State.LastPlayerWon);
        Assert.Equal(reference.State.RewardGold, resolved.State.RewardGold);
        Assert.Equal(reference.State.RewardPotion, resolved.State.RewardPotion);
        Assert.Equal(reference.State.RelicReward, resolved.State.RelicReward);
        Assert.Equal(reference.State.RewardCards, resolved.State.RewardCards);
        Assert.Equal(reference.State.RewardUpgraded, resolved.State.RewardUpgraded);
    }

    /// <summary>
    /// <c>HeistPower.BeforeDeath</c>'s <c>GoldReward(Amount, wasGoldStolenBack: true)</c> is
    /// its own row beside the fight's gold, claimed separately; its fixed amount still spends
    /// one Rewards draw (StolenLootRewardTests).
    /// </summary>
    [Fact]
    public void AResolvedFightHandsBackTheHeistGoldItWasGiven()
    {
        var engine = AtFirstCombat("QS2GYXRKWN");

        engine.DebugResolveCombat(stolenBackGold: 80);

        Assert.Contains(80, engine.State.PendingGoldRewards);
    }

    /// <summary>
    /// <c>GremlinMercNormal.CalculateGoldProportion</c> is 0 when a gremlin escaped with
    /// the loot, and <c>RewardsSet</c> only rolls a <c>GoldReward</c> when the proportion
    /// is above zero -- so the escape costs the fight its gold AND its draw.
    /// </summary>
    [Fact]
    public void AMercFightWhoseGremlinEscapedWithTheLootRollsNoGold()
    {
        var escaped = AtFirstCombat("QS2GYXRKWN");
        escaped.State.ActiveCombat!.EncounterId = RunConstants.GremlinMercEncounterId;
        var paid = escaped.Clone();

        escaped.DebugResolveCombat(fatGremlinEscaped: true, mercGoldWasStolen: true);
        paid.DebugResolveCombat();

        Assert.Equal(0, escaped.State.RewardGold);
        Assert.NotEqual(0, paid.State.RewardGold);
        Assert.Equal(
            paid.State.PlayerRng.Rewards.CallCount - 1,
            escaped.State.PlayerRng.Rewards.CallCount
        );
    }

    [Fact]
    public void ResolvingOutsideAFightIsRefusedAndChangesNothing()
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN");
        var phase = engine.State.Phase;

        Assert.Equal(-1, engine.DebugResolveCombat());
        Assert.Equal(phase, engine.State.Phase);
    }

    /// <summary>
    /// The holdings are SET, not earned: no pickup effect runs, so a relic that grants
    /// max HP on pickup does not grant it twice when the real run's totals already
    /// include it. A relic the run keeps keeps its counter.
    /// </summary>
    [Fact]
    public void SettingThePlayerReplacesItsHoldingsWithoutPickupEffects()
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN");
        int kept = engine.State.Relics[0].DefId;
        engine.State.Relics[0] = engine.State.Relics[0] with { Counter = 7 };
        const int strawberry = RunConstants.RelicStrawberry;

        engine.DebugSetPlayer(
            hp: 41,
            maxHp: 77,
            gold: 123,
            cards: [IC.Bash, 1, 0, 0, IC.StrikeIronclad, 0, 0, 0],
            relics: [kept, strawberry],
            potions: [0, 3],
            maxPotionSlots: 3
        );

        Assert.Equal(41, engine.State.PlayerHp);
        Assert.Equal(77, engine.State.PlayerMaxHp);
        Assert.Equal(123, engine.State.Gold);
        Assert.Equal(
            new[] { (IC.Bash, true), (IC.StrikeIronclad, false) },
            engine.State.Deck.Select(card => (card.DefId, card.Upgraded))
        );
        Assert.Equal(new[] { kept, strawberry }, engine.State.Relics.Select(r => r.DefId));
        Assert.Equal(7, engine.State.Relics[0].Counter);
        Assert.Equal(3, engine.State.MaxPotionSlots);
        Assert.Equal(new[] { 0, 3, 0 }, engine.State.PotionSlots.Take(3));
    }

    /// <summary>A negative scalar or a negative-length list leaves that holding alone.</summary>
    [Fact]
    public void SettingThePlayerLeavesUnnamedHoldingsAlone()
    {
        var engine = new RunEngine();
        engine.Reset("QS2GYXRKWN");
        var deck = engine.State.Deck.ToArray();
        var relics = engine.State.Relics.ToArray();
        int maxHp = engine.State.PlayerMaxHp;

        engine.DebugSetPlayer(
            hp: 10,
            maxHp: -1,
            gold: -1,
            cards: null,
            relics: null,
            potions: null
        );

        Assert.Equal(10, engine.State.PlayerHp);
        Assert.Equal(maxHp, engine.State.PlayerMaxHp);
        Assert.Equal(deck, engine.State.Deck);
        Assert.Equal(relics, engine.State.Relics);
    }

    // ---- D11F2: the game's per-card and per-relic state, as the lockstep's re-anchor passes it (run API 32).

    /// <summary>
    /// Sts2Run_DebugSetPlayerEx carries a card's enchantment and amount (the four numbers a card always had) and Mad
    /// Science's Tinker Time type and rider (new, the game's enum values: Skill is 2); the original export, which has no
    /// such list, still sets the deck and leaves the tinker fields at None.
    /// </summary>
    [Fact]
    public unsafe void TheExportCarriesAnEnchantmentAndMadSciencesTinkerFields()
    {
        int handle = RunNativeExports.Sts2Run_Create();
        try
        {
            Assert.True(RunNativeExports.TryGetRun(handle, out var run));
            run.Reset("QS2GYXRKWN");
            var obs = new int[RunConstants.RunObsSize];
            int[] cards = [IC.Bash, 1, (int)Enchantment.Swift, 2, MadScience, 0, 0, 0, IC.StrikeIronclad, 0, 0, 0];
            int[] tinker = [-1, 0, 2, 5]; // Bash: no type; Mad Science: Skill, rider 5 (Wisdom); Strike: past the list's end
            fixed (int* c = cards, t = tinker, o = obs)
            {
                Assert.Equal(
                    0,
                    RunNativeExports.Sts2Run_DebugSetPlayerEx(handle, -1, -1, -1, c, cards.Length, null, -1, null, -1, -1, t, tinker.Length, null, -1, o)
                );
            }

            var deck = run.State.Deck;
            Assert.Equal((Enchantment.Swift, 2), (deck[0].Enchantment, deck[0].EnchantAmount));
            Assert.Equal((CardType.None, TinkerRider.None), (deck[0].TinkerType, deck[0].TinkerRider));
            Assert.Equal((CardType.Skill, TinkerRider.Wisdom), (deck[1].TinkerType, deck[1].TinkerRider));
            Assert.Equal((CardType.None, TinkerRider.None), (deck[2].TinkerType, deck[2].TinkerRider));

            fixed (int* c = cards, o = obs)
            {
                Assert.Equal(0, RunNativeExports.Sts2Run_DebugSetPlayer(handle, -1, -1, -1, c, cards.Length, null, -1, null, -1, -1, o));
            }

            Assert.Equal((Enchantment.Swift, 2), (run.State.Deck[0].Enchantment, run.State.Deck[0].EnchantAmount));
            Assert.Equal((CardType.None, TinkerRider.None), (run.State.Deck[1].TinkerType, run.State.Deck[1].TinkerRider));
        }
        finally
        {
            RunNativeExports.Sts2Run_Destroy(handle);
        }
    }

    /// <summary>A counter set reads back on the run's relic and on the open fight's copy of it; a relic not held is skipped.</summary>
    [Fact]
    public void ARelicCounterIsSetOnTheRunAndTheOpenFight()
    {
        var engine = AtFirstCombat("QS2GYXRKWN");
        engine.DebugSetPlayer(-1, -1, -1, null, null, null, relicCounters: [RelicEffects.Pendulum, 2]);
        Assert.DoesNotContain(engine.State.Relics, relic => relic.DefId == RelicEffects.Pendulum);

        int kept = engine.State.Relics[0].DefId;
        engine.DebugSetPlayer(-1, -1, -1, null, null, null, relicCounters: [kept, 5]);
        Assert.Equal(5, engine.State.Relics[0].Counter);
        Assert.Equal(5, engine.State.ActiveCombat!.Relics.First(relic => relic.DefId == kept).Counter);
    }

    private static RunEngine IntoFirstFightWithPendulum(string seed, int? counter)
    {
        var engine = new RunEngine();
        engine.Reset(seed);
        for (int i = 0; i < 16 && engine.State.Phase != RunPhase.Map; i++)
        {
            engine.Step(0, -1, out _, out _, out _);
        }

        if (counter is int set)
        {
            engine.DebugSetPlayer(
                -1,
                -1,
                -1,
                null,
                [.. engine.State.Relics.Select(relic => relic.DefId), RelicEffects.Pendulum],
                null,
                relicCounters: [RelicEffects.Pendulum, set]
            );
        }

        int action = Array.FindIndex(engine.State.MapNodeTypes, nodeType => nodeType == RunConstants.NodeNormal);
        Assert.Equal(0, engine.Step(action, -1, out _, out _, out _));
        Assert.Equal(RunPhase.Combat, engine.State.Phase);
        return engine;
    }

    private static int EndTurnAndCountHand(RunEngine engine)
    {
        engine.Step(engine.State.ActiveCombat!.Hand.Count, -1, out _, out _, out _);
        return engine.State.ActiveCombat!.Hand.Count;
    }

    /// <summary>
    /// Pendulum draws one card every third turn. Set to 0 on the map it ticks on turns one and two and pays on the third;
    /// set to 2 it pays on the first. Held without the set it starts at 0, as the plain run does.
    /// </summary>
    [Fact]
    public void APendulumCounterSetOnTheMapFiresOnTheRightTurn()
    {
        var plain = IntoFirstFightWithPendulum("QS2GYXRKWN", null);
        var fromZero = IntoFirstFightWithPendulum("QS2GYXRKWN", 0);
        var fromTwo = IntoFirstFightWithPendulum("QS2GYXRKWN", 2);

        int baseHand = plain.State.ActiveCombat!.Hand.Count;
        Assert.Equal(baseHand, fromZero.State.ActiveCombat!.Hand.Count);
        Assert.Equal(baseHand + 1, fromTwo.State.ActiveCombat!.Hand.Count);

        Assert.Equal(EndTurnAndCountHand(plain), EndTurnAndCountHand(fromZero)); // turn two: still ticking
        int thirdPlain = EndTurnAndCountHand(plain);
        Assert.Equal(thirdPlain + 1, EndTurnAndCountHand(fromZero)); // turn three: the draw
    }
}
