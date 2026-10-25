using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Core2;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// The card-level additive terms under a card multiplier (docs/kits/DamageOrderClass.md, follow-up of PenNibStrengthTests).
/// game: Hook.ModifyDamage folds EVERY additive listener -- the card's own terms (AccuracyPower for Shivs, Claw's counter, the
/// card model's bonus damage, CalcifyPower on an Osty attack) as well as the creature's Strength and Vigor -- before the first
/// multiplicative one (PenNib, GigantificationPower, VitruvianMinion), so a multiplier doubles or triples the sum.
/// Each case runs with the F08+F10 switch off (CombatEngine.Step, the old path) and on (a Core2 kernel bound to the combat as
/// PenNibStrengthTests does), one 50 HP enemy with no block or powers, and pins the enemy's HP after the play.
/// </summary>
public class DamageOrderClassTests
{
    private const int Claw = 89;
    private const int BoneShards = 53;

    private static readonly Lazy<Kernel> OnKernel = new(() => new Kernel(
        new KernelOptions([Sts2Emulator.Core2.Switch.F08, Sts2Emulator.Core2.Switch.F10]),
        () => Core2.ListenerOrderTests.Ids,
        Sts2Emulator.Core2.Models.Registry.All));

    /// <summary>Pen Nib at <paramref name="attacksPlayed" /> Attacks played, the given card in hand, one 50 HP enemy.</summary>
    private static CombatState Setup(CardInstance card, int attacksPlayed = 9)
    {
        var state = CombatFactory.NewCombat(seed: 0);
        state.Relics.Add(new RelicInstance(RelicEffects.PenNib, attacksPlayed));
        state.PenNibArmed = false;
        state.Hand.Clear();
        state.Hand.Add(card);
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

    /// <summary>
    /// AccuracyPower.ModifyDamageAdditive on a Shiv (printed 4) at Accuracy 4, on a Pen Nib tenth Attack:
    /// (4 + 4) * 2 = 16, not 4 * 2 + 4 = 12.
    /// </summary>
    [Theory]
    [InlineData(false, 34)]
    [InlineData(true, 34)]
    public void AccuracyIsAddedBeforePenNibDoublesAShiv(bool switchOn, int enemyHpAfter)
    {
        var state = Setup(new CardInstance(SI.Shiv, false));
        BuffSystem.Apply(state.PlayerBuffs, BuffId.ShivDamage, 4);

        Play(state, switchOn);

        Assert.Equal(switchOn, state.Kernel is not null);
        Assert.Equal(enemyHpAfter, state.Enemies[0].Hp);
    }

    /// <summary>Claw's accumulated damage (6) on a Gigantified Claw (printed 3): (3 + 6) * 3 = 27, not 3 * 3 + 6 = 15.</summary>
    [Theory]
    [InlineData(false, 23)]
    [InlineData(true, 23)]
    public void ClawDamageIsAddedBeforeGigantificationTriples(bool switchOn, int enemyHpAfter)
    {
        var state = Setup(new CardInstance(Claw, false), attacksPlayed: 0);
        BuffSystem.Apply(state.PlayerBuffs, BuffId.ClawDamage, 6);
        BuffSystem.Apply(state.PlayerBuffs, BuffId.Gigantification, 1);

        Play(state, switchOn);

        Assert.Equal(enemyHpAfter, state.Enemies[0].Hp);
    }

    /// <summary>
    /// The card model's bonus damage (a Rampage copy grown by 5; no Minion card carries bonus damage, so the multiplier is
    /// Pen Nib's tenth Attack) on the printed 9: (9 + 5) * 2 = 28, not 9 * 2 + 5 = 23.
    /// </summary>
    [Theory]
    [InlineData(false, 22)]
    [InlineData(true, 22)]
    public void BonusDamageIsAddedBeforePenNibDoubles(bool switchOn, int enemyHpAfter)
    {
        var state = Setup(new CardInstance(IC.Rampage, false) with { BonusDamage = 5 });

        Play(state, switchOn);

        Assert.Equal(enemyHpAfter, state.Enemies[0].Hp);
    }

    /// <summary>
    /// CalcifyPower.ModifyDamageAdditive (4) on an Osty attack that wraps its Dmg (Bone Shards, printed 9) on a Pen Nib
    /// tenth Attack: (9 + 4) * 2 = 26, not 9 * 2 + 4 = 22.
    /// </summary>
    [Theory]
    [InlineData(false, 24)]
    [InlineData(true, 24)]
    public void CalcifyIsAddedBeforePenNibDoublesAnOstyAttack(bool switchOn, int enemyHpAfter)
    {
        var state = Setup(new CardInstance(BoneShards, false));
        CardEffects.SummonOsty(state, 10);
        BuffSystem.Apply(state.PlayerBuffs, BuffId.Calcify, 4);

        Play(state, switchOn);

        Assert.Equal(enemyHpAfter, state.Enemies[0].Hp);
    }
}
