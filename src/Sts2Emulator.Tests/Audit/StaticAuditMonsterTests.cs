using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.Audit;

/// <summary>Static audit pilot, monsters and encounter moves.</summary>
public class StaticAuditMonsterTests
{
    /// <summary>
    /// Mawler's machine starts on CLAW_MOVE: MultiAttackIntent(ClawDamage, 2) -- two hits of 5 at
    /// A10, not one hit of 10.
    /// </summary>
    [Fact]
    public void MawlerOpensOnTwoClaws()
    {
        var fight = Fight.Encounter(CombatFactory.ActOneEncounter.Mawler, ascension: 10);

        var intent = fight.Enemy0.CurrentIntent;

        Assert.Equal(IntentType.Attack, intent.Type);
        Assert.Equal(5, intent.Magnitude);
        Assert.Equal(2, intent.Hits);
    }
}
