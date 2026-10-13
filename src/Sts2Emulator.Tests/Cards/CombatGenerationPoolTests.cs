using System;
using System.Collections.Generic;
using System.Linq;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests;

// CardFactory.GetDistinctForCombat and GetForCombat both run FilterForPlayerCount, which in a
// solo run removes every MultiplayerOnly card before the shuffle or the NextItem roll. Every
// in-combat generator (Discovery, Jackpot, Stoke, Infernal Blade, Splash, Hello World, Call
// of the Void, Choices Paradox, the Colorless potions and relics) goes through that factory,
// so the emulator's one shared pool, CardEffects.CombatGenerationPool, has to drop them too.
public class CombatGenerationPoolTests
{
    private static CombatState IroncladCombat() => Fight.Hand().Energy(3).Enemy(hp: 40).State;

    [Fact]
    public void TheIroncladPoolHasNoMultiplayerOnlyCard()
    {
        var pool = CardEffects.OwnerCharacterGenerationPool(IroncladCombat());

        Assert.NotEmpty(pool);
        Assert.DoesNotContain(pool, id => GeneratedData.Cards.Get(id).MultiplayerOnly);
        // The two the night03 Discovery and Stoke fights tripped over.
        Assert.DoesNotContain(IC.Tank, pool);
        Assert.DoesNotContain(IC.DemonicShield, pool);
    }

    [Fact]
    public void EveryTypedPoolHasNoMultiplayerOnlyCard()
    {
        var state = IroncladCombat();
        foreach (var type in new[] { CardType.Attack, CardType.Skill, CardType.Power })
        {
            Assert.DoesNotContain(
                CardEffects.OwnerCharacterGenerationPool(state, type),
                id => GeneratedData.Cards.Get(id).MultiplayerOnly
            );
        }

        Assert.DoesNotContain(
            CardEffects.OwnerCharacterGenerationPool(state, CardType.Attack), // Infernal Blade's pool
            id => GeneratedData.Cards.Get(id).MultiplayerOnly
        );
        Assert.DoesNotContain(
            CardEffects.SplashPool(state),
            id => GeneratedData.Cards.Get(id).MultiplayerOnly
        );
    }

    [Fact]
    public void TheColorlessOfferHasNoMultiplayerOnlyCard()
    {
        var fight = Fight.Hand().Energy(3).Enemy(hp: 40);
        for (int seed = 0; seed < 40; seed++)
        {
            fight.State.CardGenerationRng = new CountingRandom(seed);

            var offered = CardEffects.DistinctColorlessForCombat(
                fight.State,
                count: 100,
                new Random(0),
                excludeDefId: CL.JackOfAllTrades
            );

            // Jack of All Trades' own pool: 49 cards in a solo run.
            Assert.Equal(49, offered.Count);
            Assert.DoesNotContain(offered, id => GeneratedData.Cards.Get(id).MultiplayerOnly);
        }
    }
}
