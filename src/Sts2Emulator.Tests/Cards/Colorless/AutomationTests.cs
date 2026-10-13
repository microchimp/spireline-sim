using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// 1-cost Power. MegaCrit.Sts2.Core.Models.Cards/Automation.cs applies AutomationPower at EnergyVar(1) — energy after every ten cards drawn; OnUpgrade only makes it cheaper.
public class AutomationTests
{
    [Fact]
    public void AppliesOne()
    {
        var fight = Fight.Hand(Card(CL.Automation)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.AutomationPower));
    }

    [Fact]
    public void UpgradedCostsZeroRatherThanApplyingMore()
    {
        var fight = Fight.Hand(Card(CL.Automation, upgraded: true)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Equal(1, fight.PlayerBuffAmount(BuffId.AutomationPower));
    }

    [Fact]
    public void LeavesPlayLikeAnyPower()
    {
        var fight = Fight.Hand(Card(CL.Automation)).Energy(3).Enemy(hp: 40);

        fight.Play();

        Assert.Empty(fight.State.Hand);
        Assert.Empty(fight.State.DiscardPile);
    }

    /// <summary>
    /// TEN cards drawn, then the energy. `AutomationPower.DisplayAmount` counts DOWN from
    /// ten to the next payout while the power's own amount is the energy it grants, so a
    /// capture cannot compare the number -- the live readout is the complement of the
    /// instance's `Data` (the cards drawn towards its ten), which is not its amount at
    /// all. That is why `AUTOMATION_POWER` is presence-only in the capture generator,
    /// and why the count belongs here instead.
    /// </summary>
    [Fact]
    public void TheEnergyArrivesOnTheTenthCardDrawn()
    {
        var fight = Fight.Hand(Card(CL.Automation)).Energy(3).Enemy(hp: 40);
        fight.Play();

        fight.State.DrawPile.Clear();
        for (int i = 0; i < 20; i++)
        {
            fight.State.DrawPile.Add(new CardInstance(IC.StrikeIronclad, false));
        }

        fight.State.Hand.Clear();
        int before = fight.State.Energy;
        CardEffects.DrawCards(fight.State, 9, new System.Random(0));
        Assert.Equal(before, fight.State.Energy);

        fight.State.Hand.Clear();
        CardEffects.DrawCards(fight.State, 1, new System.Random(0));

        Assert.Equal(before + 1, fight.State.Energy);
        Assert.Equal(0, Automations(fight).Single().Data);
    }

    private static List<BuffState> Automations(Fight fight) =>
        fight.State.PlayerBuffs.Where(b => b.Id == BuffId.AutomationPower).ToList();

    private static void DrawFromAFullPile(Fight fight, int cards)
    {
        fight.State.DrawPile.Clear();
        for (int i = 0; i < 40; i++)
        {
            fight.State.DrawPile.Add(new CardInstance(IC.StrikeIronclad, false));
        }

        int held = fight.State.Hand.Count;
        CardEffects.DrawCards(fight.State, cards, new System.Random(0));
        fight.State.Hand.RemoveRange(held, fight.State.Hand.Count - held);
    }

    /// <summary>
    /// `AutomationPower` is Instanced: a second play is a power of its own that counts its own
    /// ten from the moment it lands, not a bigger payout on the first one's count. After seven
    /// draws with one Automation and then eight with two, the first has seen fifteen (a payout
    /// at ten, five over) and the second eight.
    /// </summary>
    [Fact]
    public void ASecondAutomationCountsItsOwnTenFromWhenItWasPlayed()
    {
        var fight = Fight.Hand(Card(CL.Automation), Card(CL.Automation)).Energy(2).Enemy(hp: 40);
        fight.Play();
        DrawFromAFullPile(fight, 7);
        fight.Play();

        Assert.Equal(
            [new BuffState(BuffId.AutomationPower, 1, 7), new BuffState(BuffId.AutomationPower, 1, 0)],
            Automations(fight)
        );

        int energy = fight.State.Energy;
        DrawFromAFullPile(fight, 3);

        // Only the first reaches ten: one energy, not two.
        Assert.Equal(energy + 1, fight.State.Energy);
        Assert.Equal(
            [new BuffState(BuffId.AutomationPower, 1, 0), new BuffState(BuffId.AutomationPower, 1, 3)],
            Automations(fight)
        );
    }

    /// <summary>Each instance pays its own Amount, so the two payouts are separate gains.</summary>
    [Fact]
    public void EveryInstancePaysItsOwnEnergyWhenItsTenAreDrawn()
    {
        var fight = Fight.Hand().Energy(0).Enemy(hp: 40);
        fight.PlayerBuff(BuffId.AutomationPower, 1);
        fight.PlayerBuff(BuffId.AutomationPower, 2);

        DrawFromAFullPile(fight, 10);

        Assert.Equal(3, fight.State.Energy);
        Assert.All(Automations(fight), a => Assert.Equal(0, a.Data));
    }
}
