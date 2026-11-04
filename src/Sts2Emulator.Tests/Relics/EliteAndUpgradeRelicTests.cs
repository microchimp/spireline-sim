using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

/// <summary>
/// Booming Conch (CardsVar(2) + EnergyVar(1), elite rooms only), Stone Cracker
/// (CardsVar(2) upgraded off the draw pile) and Venerable Tea Set (EnergyVar(2)), all
/// read off MegaCrit.Sts2.Core.Models.Relics.
/// </summary>
public class EliteAndUpgradeRelicTests
{
    [Fact]
    public void BoomingConchDoesNothingOutsideAnEliteFight()
    {
        var plain = Fight.WithRelics();
        var withConch = Fight.WithRelics(RelicEffects.BoomingConch);

        Assert.False(withConch.State.IsEliteRoom);
        Assert.Equal(plain.State.Energy, withConch.State.Energy);
        Assert.Equal(plain.State.Hand.Count, withConch.State.Hand.Count);
    }

    /// <summary>
    /// It fires in an elite room, and every act-1 elite has to read as one.
    /// </summary>
    /// <remarks>
    /// The elite test used to be a RANGE over the encounter enum, <c>BygoneEffigy</c> to
    /// <c>WaterfallGiant</c> — true when WaterfallGiant was the last name declared, and
    /// quietly false once Architect and SkulkingColony were appended after it. A live
    /// capture opened a Skulking Colony with seven cards and four energy where the
    /// emulator had five and three (`L9R346P3YD`).
    /// </remarks>
    [Fact]
    public void BoomingConchFiresInEveryActOneElite()
    {
        CombatFactory.ActOneEncounter[] elites =
        [
            CombatFactory.ActOneEncounter.BygoneEffigy,
            CombatFactory.ActOneEncounter.Byrdonis,
            CombatFactory.ActOneEncounter.PhrogParasite,
            CombatFactory.ActOneEncounter.PhantasmalGardeners,
            CombatFactory.ActOneEncounter.SkulkingColony,
            CombatFactory.ActOneEncounter.TerrorEel,
        ];

        foreach (var encounter in elites)
        {
            var plain = Fight.Encounter(encounter);
            var withConch = Fight.Encounter(
                encounter,
                Ascension.DefaultLevel,
                0,
                RelicEffects.BoomingConch
            );

            Assert.True(withConch.State.IsEliteRoom, $"{encounter} did not read as elite");
            Assert.Equal(plain.State.Energy + 1, withConch.State.Energy);
            Assert.Equal(plain.State.Hand.Count + 2, withConch.State.Hand.Count);
        }
    }

    /// <summary>
    /// A boss is not an elite. BoomingConch asks for <c>RoomType.Elite</c> and a boss room
    /// is <c>RoomType.Boss</c>; the old range swept every boss in with the elites.
    /// </summary>
    [Fact]
    public void BoomingConchDoesNothingInABossFight()
    {
        CombatFactory.ActOneEncounter[] bosses =
        [
            CombatFactory.ActOneEncounter.TheKin,
            CombatFactory.ActOneEncounter.WaterfallGiant,
            CombatFactory.ActOneEncounter.CeremonialBeast,
        ];

        foreach (var encounter in bosses)
        {
            var fight = Fight.Encounter(
                encounter,
                Ascension.DefaultLevel,
                0,
                RelicEffects.BoomingConch
            );

            Assert.False(fight.State.IsEliteRoom, $"{encounter} read as an elite");
        }
    }

    private static int Strength(CombatState state) =>
        BuffSystem.Get(state.PlayerBuffs, BuffId.Strength);

    /// <summary>
    /// Booming Conch (two cards, one energy) and Sling of Courage (two Strength) ask the ROOM, and a
    /// room is an elite room in every act: the direct combat environment, which has only an
    /// encounter to go by, knows all twelve elites. The elite test was a list of the six Act 1
    /// ones, so no Hive or Glory elite ever paid (40 hand records of three nights, Entomancer,
    /// Infested Prisms and Decimillipede).
    /// </summary>
    [Fact]
    public void BoomingConchAndSlingOfCourageFireInEveryEliteOfEveryAct()
    {
        // Overgrowth, Underdocks, Hive and Glory, by encounter id.
        int[] elites = [62, 68, 65, 72, 86, 67, 69, 63, 64, 70, 71, 66];

        foreach (int encounter in elites)
        {
            var plain = Fight.Encounter(encounter);
            var relics = Fight.Encounter(encounter, RelicEffects.BoomingConch, RelicEffects.SlingOfCourage);

            Assert.True(relics.State.IsEliteRoom, $"encounter {encounter} did not read as an elite room");
            Assert.Equal(plain.State.Hand.Count + 2, relics.State.Hand.Count);
            Assert.Equal(plain.State.Energy + 1, relics.State.Energy);
            Assert.Equal(Strength(plain.State) + 2, Strength(relics.State));
        }
    }

    /// <summary>
    /// A run knows its room, and the combat-start hooks and the opening draw ask it while the reset
    /// runs: the room has to be there before they do. It was set after the reset, so Sling of
    /// Courage never fired in any run elite, Act 1 included.
    /// </summary>
    [Theory]
    [InlineData(63)] // Entomancer, Act 2
    [InlineData(68)] // Byrdonis, Act 1
    public void ARunEliteRoomPaysBothRelics(int encounterId)
    {
        var plain = RunAt(RunConstants.NodeElite, encounterId, []);
        var relics = RunAt(
            RunConstants.NodeElite,
            encounterId,
            [RelicEffects.BoomingConch, RelicEffects.SlingOfCourage]
        );

        Assert.True(relics.IsEliteRoom);
        Assert.Equal(plain.Hand.Count + 2, relics.Hand.Count);
        Assert.Equal(plain.Energy + 1, relics.Energy);
        Assert.Equal(Strength(plain) + 2, Strength(relics));
    }

    /// <summary>The ROOM decides, not the encounter: an elite's roster met in a hallway room pays nothing.</summary>
    [Fact]
    public void AnEliteRosterInAHallwayRoomPaysNothing()
    {
        var plain = RunAt(RunConstants.NodeNormal, 63, []);
        var relics = RunAt(
            RunConstants.NodeNormal,
            63,
            [RelicEffects.BoomingConch, RelicEffects.SlingOfCourage]
        );

        Assert.False(relics.IsEliteRoom);
        Assert.Equal(plain.Hand.Count, relics.Hand.Count);
        Assert.Equal(plain.Energy, relics.Energy);
        Assert.Equal(Strength(plain), Strength(relics));
    }

    private static CombatState RunAt(int roomType, int encounterId, int[] relics)
    {
        var engine = new RunEngine();
        engine.Reset("NXV45HW43K");
        engine.State.LastResolvedRoomType = roomType;
        Assert.Equal(
            0,
            engine.StartCombat([.. RunConstants.StarterDeckIds], encounterId, relics, 80, 80, [], 99)
        );
        return engine.State.ActiveCombat!;
    }

    [Fact]
    public void StoneCrackerUpgradesTheFirstTwoUpgradableCardsInTheDrawPile()
    {
        var fight = Fight.WithRelics(RelicEffects.StoneCracker);

        // The starter deck's Ascender's Bane cannot be upgraded; two of the rest are.
        Assert.Equal(2, fight.State.DrawPile.Concat(fight.State.Hand).Count(card => card.Upgraded));
    }

    /// <summary>
    /// The relic picks with <c>Where(IsUpgradable).StableShuffle(Rng.CombatCardSelection)
    /// .Take(2)</c> (StoneCracker.cs, game 0.107.1), not the first two in pile order: this
    /// pinned the latter, and the referee fight night04 3JFLX5V4BH (Coordinate upgraded in
    /// the game, Defend here) showed it wrong. See StoneCrackerTests for the order-independence
    /// and the draw count; this keeps the pile-order case honest, in that the pile's order is
    /// not what decides.
    /// </summary>
    [Fact]
    public void StoneCrackerRollsWhichTwoOffTheSelectionStream()
    {
        var selection = new CountingRandom(0);
        var state = new CombatState
        {
            Relics = [new RelicInstance(RelicEffects.StoneCracker, 0)],
            CardSelectionRng = selection,
        };
        state.DrawPile =
        [
            Card(IC.StrikeIronclad),
            Card(IC.DefendIronclad),
            Card(IC.Bash),
            Card(IC.Anger),
        ];

        RelicEffects.ApplyBeforeOpeningHand(state, new Random(0));

        Assert.Equal(2, state.DrawPile.Count(card => card.Upgraded));
        Assert.Equal(3, selection.CallCount);
    }

    // Venerable Tea Set's two tests used to live here and drove a synthetic
    // `VenerableTeaSetActive` relic id that NOTHING IN THE RUN EVER ADDED, so they proved
    // the seam worked while the relic was inert for the whole run (E393). The armed state
    // is a counter on the relic now, and the tests live in `VenerableTeaSetTests` beside
    // the rest site that arms it.
}
