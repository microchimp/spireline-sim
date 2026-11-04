using System.Collections.Generic;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The Scythe and Genetic Algorithm grow for the whole run: their play grows the deck's
/// card as well as the fight's, so the next fight opens with the grown card. The copy kept
/// the growth inside the fight, and a Scythe played in an earlier fight hit 4 (5 upgraded)
/// short per earlier play in every later one (the nights' Scythe records: the game's enemy
/// 4 to 6 lower after a Scythe).
/// </summary>
public class PlayGrowthReachesTheDeckTests
{
    private const int TheScythe = 501;

    private static RunEngine RunWith(params CardInstance[] deck)
    {
        var engine = new RunEngine();
        engine.Reset("ACT2TEST01");
        engine.State.Deck.Clear();
        engine.State.Deck.AddRange(deck);
        return engine;
    }

    [Fact]
    public void AScythePlayedInOneFightOpensTheNextOneStronger()
    {
        var engine = RunWith(new CardInstance(TheScythe, false));
        var fight = Fight.Hand(engine.State.Deck[0]).Energy(9).Enemy(hp: 500);
        fight.Play();
        engine.State.ActiveCombat = fight.State;

        engine.SyncAfterCombatForTest();

        Assert.Equal(4, engine.State.Deck[0].BonusDamage);
        var next = Fight.Hand(engine.State.Deck[0]).Energy(9).Enemy(hp: 500);
        next.Play();
        Assert.Equal(500 - 17, next.Enemy0.Hp);
    }

    [Fact]
    public void AnUpgradedScytheCarriesFive()
    {
        var engine = RunWith(new CardInstance(TheScythe, true));
        var fight = Fight.Hand(engine.State.Deck[0]).Energy(9).Enemy(hp: 500);
        fight.Play();
        engine.State.ActiveCombat = fight.State;

        engine.SyncAfterCombatForTest();

        Assert.Equal(5, engine.State.Deck[0].BonusDamage);
    }

    /// <summary>Two Scythes, one played: the deck's grown card is one of them, the other stays.</summary>
    [Fact]
    public void OnlyThePlayedCopyGrows()
    {
        var engine = RunWith(new CardInstance(TheScythe, false), new CardInstance(TheScythe, false));
        var fight = Fight.Hand(engine.State.Deck[0]).Draw(engine.State.Deck[1]).Energy(9).Enemy(hp: 500);
        fight.Play();
        engine.State.ActiveCombat = fight.State;

        engine.SyncAfterCombatForTest();

        Assert.Equal([4, 0], engine.State.Deck.Select(c => c.BonusDamage).OrderByDescending(v => v));
    }

    [Fact]
    public void GeneticAlgorithmCarriesItsBlock()
    {
        int geneticAlgorithm = GeneratedData.Cards.FindId("GeneticAlgorithm")!.Value;
        var engine = RunWith(new CardInstance(geneticAlgorithm, false));
        var fight = Fight.Hand(engine.State.Deck[0]).Energy(9).Enemy(hp: 500);
        fight.Play();
        engine.State.ActiveCombat = fight.State;

        engine.SyncAfterCombatForTest();

        Assert.Equal(3, engine.State.Deck[0].BonusBlock);
    }

    /// <summary>A Rampage's growth is the fight's alone: the deck card is left as it was.</summary>
    [Fact]
    public void RampageDoesNotCarry()
    {
        int rampage = GeneratedData.Cards.FindId("Rampage")!.Value;
        var engine = RunWith(new CardInstance(rampage, false));
        var fight = Fight.Hand(engine.State.Deck[0]).Energy(9).Enemy(hp: 500);
        fight.Play();
        engine.State.ActiveCombat = fight.State;

        engine.SyncAfterCombatForTest();

        Assert.Equal(0, engine.State.Deck[0].BonusDamage);
    }
}
