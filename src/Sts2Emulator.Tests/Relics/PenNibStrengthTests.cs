using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Pen Nib on the tenth Attack with Strength on the player (lockstep record FTV3HZZPFE #135, docs/kits/PenNibStrength.md).
/// game: Hook.ModifyDamage folds every additive term before the multiplicative ones -- StrengthPower.ModifyDamageAdditive
/// adds the owner's Strength to the card's base, then PenNib.ModifyDamageMultiplicative doubles that total -- so a Strike
/// (6) at Strength 2 deals (6 + 2) * 2 = 16 on the tenth Attack and 12 at Strength 0. The relic's counter rises in
/// PenNib.BeforeCardPlayed, so a card played at counter 9 sees it wrapped to 0 and armed.
/// Each case runs with the F08+F10 switch off (CombatEngine.Step, the old path) and on (a Core2 kernel bound to the combat
/// as TestSwitches.Step does, whatever STS2_TEST_SWITCHES says), so the pair of paths is pinned in one class.
/// </summary>
public class PenNibStrengthTests
{
    private static readonly Lazy<Kernel> OnKernel = new(() => new Kernel(
        new KernelOptions([Sts2Emulator.Core2.Switch.F08, Sts2Emulator.Core2.Switch.F10]),
        () => Core2.ListenerOrderTests.Ids,
        Sts2Emulator.Core2.Models.Registry.All));

    /// <summary>Pen Nib at nine Attacks played, the given Strength, a Strike in hand, one 50 HP enemy with no block or powers.</summary>
    private static CombatState TenthAttack(int strength)
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Relics.Add(new RelicInstance(RelicEffects.PenNib, 9));
        state.PenNibArmed = false;
        if (strength != 0)
            BuffSystem.Apply(state.PlayerBuffs, BuffId.Strength, strength);
        state.Hand.Clear();
        state.Hand.Add(new CardInstance(IC.StrikeIronclad, false));
        state.Enemies =
        [
            new EnemyState
            {
                DefId = 21,
                Hp = 50,
                MaxHp = 50,
                Block = 0,
                CurrentIntent = new Intent(IntentType.Attack, 5),
                Buffs = [],
            },
        ];
        state.Energy = 3;
        return state;
    }

    /// <summary>Plays the hand's one card at the one enemy: the old step, or the kernel-bound step with the switch on.</summary>
    private static void Play(CombatState state, bool switchOn)
    {
        var rng = new Random(0);
        if (!switchOn)
        {
            CombatEngine.Step(state, 0, rng);
            return;
        }

        var engine = new RunEngine { Kernel = OnKernel.Value };
        CardUids.NumberNew(state);
        OnKernel.Value.Attach(state, engine.State);
        engine.StepCombat(state, 0, rng);
    }

    [Theory]
    [InlineData(false, 2, 34)]
    [InlineData(true, 2, 34)]
    [InlineData(false, 0, 38)]
    [InlineData(true, 0, 38)]
    public void StrengthIsAddedBeforePenNibDoubles(bool switchOn, int strength, int enemyHpAfter)
    {
        var state = TenthAttack(strength);

        Play(state, switchOn);

        Assert.Equal(switchOn, state.Kernel is not null);
        Assert.Equal(0, state.Relics.Single(r => r.DefId == RelicEffects.PenNib).Counter);
        Assert.Equal(enemyHpAfter, state.Enemies[0].Hp);
    }
}
