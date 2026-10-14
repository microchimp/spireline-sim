using Sts2Emulator.Core;
using Sts2Emulator.Core.Effects;
using Xunit;
using static Sts2Emulator.Tests.TestDeck;

namespace Sts2Emulator.Tests.Audit;

/// <summary>
/// Static audit pilot, relics and the damage/block/power pipelines they ride on. Each test names
/// the game class it was read against (short identifiers only) and states the game's behaviour.
/// </summary>
public class StaticAuditRelicTests
{
    /// <summary>
    /// RuinedHelmet.TryModifyPowerAmountReceived doubles the first POSITIVE Strength the owner
    /// receives through PowerCmd.Apply / PowerCmd.ModifyAmount (silent or not). TenderPower's
    /// AfterSideTurnEnd hands the turn's Strength back through PowerCmd.Apply, so with an unspent
    /// helmet the return is doubled. The emulator writes it raw (CombatEngine, Tender return).
    /// The two -1 applications per card are not doubled (amount is not positive).
    /// </summary>
    [AuditFact("Tender's Strength return goes through Ruined Helmet; the engine fix (audit/static-pilot eb3c35f) is not merged, only the pilot's tools and tests")]
    public void TenderReturnIsDoubledByAnUnspentRuinedHelmet()
    {
        var fight = Fight
            .WithRelics(RelicEffects.RuinedHelmet)
            .Enemy(hp: 200)
            .PlayerBuff(BuffId.Tender, 1)
            .Draw(Card(IC.StrikeIronclad), Card(IC.StrikeIronclad), Card(IC.StrikeIronclad))
            .Energy(3);
        fight.State.Hand = [Card(IC.StrikeIronclad), Card(IC.StrikeIronclad)];

        fight.Play();
        fight.Play();
        Assert.Equal(-2, fight.PlayerBuffAmount(BuffId.Strength));

        fight.EndTurn();

        // +2 returned, doubled to +4 by the helmet: -2 + 4.
        Assert.Equal(2, fight.PlayerBuffAmount(BuffId.Strength));
    }

    /// <summary>
    /// ShrinkPower.AfterDeath removes the power when its applier dies, whatever killed it: a
    /// beetle killed by Charon's Ashes (unpowered damage) takes the Shrink with it.
    /// </summary>
    [Fact]
    public void ShrinkEndsWhenTheBeetleDiesToCharonsAshes()
    {
        var fight = Fight
            .WithRelics(RelicEffects.CharonsAshes)
            .Enemy(hp: 2, defId: KE.ShrinkerBeetle)
            .Enemy(hp: 100);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Shrink, -1);

        CardEffects.ExhaustCard(fight.State, Card(IC.StrikeIronclad), rng: new Random(0));

        Assert.True(fight.State.Enemies[0].Hp <= 0);
        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Shrink));
    }

    /// <summary>
    /// The same ShrinkPower.AfterDeath, reached by poison: the tick kills the beetle at the start
    /// of the enemy turn and the Shrink goes with it.
    /// </summary>
    [Fact]
    public void ShrinkEndsWhenThePoisonKillsTheBeetle()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 3, defId: KE.ShrinkerBeetle, buffs: new BuffState(BuffId.Poison, 5, 0))
            .Enemy(hp: 100);
        BuffSystem.Apply(fight.State.PlayerBuffs, BuffId.Shrink, -1);

        fight.EndTurn();

        Assert.Equal(0, fight.PlayerBuffAmount(BuffId.Shrink));
    }

    /// <summary>
    /// HardenedShellPower.ModifyHpLostBeforeOstyLate caps HP lost per turn for EVERY damage
    /// instance (no props check).
    /// </summary>
    [Fact]
    public void HardenedShellCapsUnpoweredDamageToo()
    {
        var fight = Fight.Hand().Enemy(hp: 100, buffs: new BuffState(BuffId.HardenedShell, 20, 0));

        CardEffects.DealUnpoweredDamageToEnemy(fight.State, fight.Enemy0, 50);

        Assert.Equal(80, fight.Enemy0.Hp);
    }

    /// <summary>
    /// AsleepPower.AfterDamageReceived wakes the Lagavulin Matriarch on ANY unblocked damage
    /// (it reads no props): its Plating goes and it is stunned, whether Charon's Ashes, Combust,
    /// Juggernaut or a potion struck her.
    /// </summary>
    [Fact]
    public void UnpoweredDamageWakesTheMatriarch()
    {
        var fight = Fight
            .Hand()
            .Enemy(
                hp: 200,
                defId: KE.LagavulinMatriarch,
                buffs: [new BuffState(BuffId.Asleep, 3, 0), new BuffState(BuffId.Plating, 12, 0)]
            );

        CardEffects.DealUnpoweredDamageToEnemy(fight.State, fight.Enemy0, 3);

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Asleep));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Plating));
    }

    /// <summary>
    /// SlumberPower.AfterDamageReceived (Slumbering Beetle) costs a point of sleep per instance
    /// of unblocked damage and reads no props; at zero it stuns the beetle into its wake-up.
    /// </summary>
    [Fact]
    public void UnpoweredDamageWearsDownASleepingBeetle()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 200, defId: KE.SlumberingBeetle, buffs: new BuffState(BuffId.Slumber, 1, 0));

        CardEffects.DealUnpoweredDamageToEnemy(fight.State, fight.Enemy0, 3);

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Slumber));
        Assert.Equal(1, fight.EnemyBuffAmount(BuffId.Stunned));
    }

    /// <summary>
    /// ShriekPower.AfterDamageReceived (Terror Eel): unblocked damage that leaves it at or below
    /// the threshold stuns it into TERROR, whatever the damage's props.
    /// </summary>
    [Fact]
    public void UnpoweredDamageTripsTheEelsShriek()
    {
        var fight = Fight
            .Hand()
            .Enemy(hp: 100, defId: KE.TerrorEel, buffs: new BuffState(BuffId.Shriek, 60, 0));

        CardEffects.DealUnpoweredDamageToEnemy(fight.State, fight.Enemy0, 50);

        Assert.Equal(1, fight.EnemyBuffAmount(BuffId.Stunned));
        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Shriek));
    }

    /// <summary>
    /// BurrowedPower.AfterBlockBroken (Tunneler) fires for any damage that takes the last block
    /// off: the Tunneler is stunned out of its burrow.
    /// </summary>
    [Fact]
    public void UnpoweredDamageBreaksTheTunnelersBurrow()
    {
        var fight = Fight
            .Hand()
            .Enemy(
                hp: 200,
                block: 10,
                defId: KE.Tunneler,
                buffs: new BuffState(BuffId.Burrowed, 1, 0)
            );

        CardEffects.DealUnpoweredDamageToEnemy(fight.State, fight.Enemy0, 10);

        Assert.Equal(0, fight.EnemyBuffAmount(BuffId.Burrowed));
        Assert.Equal(1, fight.EnemyBuffAmount(BuffId.Stunned));
    }

    /// <summary>
    /// CreatureCmd.Damage: WasBlockBroken = Block &lt;= 0 AND blockedDamage &gt; 0, evaluated after
    /// the absorb. A hit exactly the size of the block breaks it (it is also fully blocked), and
    /// Hand Drill reads WasBlockBroken without asking whether any HP was lost.
    /// </summary>
    [Fact]
    public void HandDrillCountsAHitExactlyTheSizeOfTheBlock()
    {
        var fight = Fight.Encounter(3, RelicEffects.HandDrill).Enemy(hp: 200);
        fight.Enemy0.Block = 5;

        CardEffects.DealPoweredDamage(fight.State, fight.Enemy0, 5);

        Assert.Equal(2, fight.EnemyBuffAmount(BuffId.Vulnerable));
    }

    /// <summary>
    /// PoisonPower deals its tick through CreatureCmd.Damage, so the unblocked HP it takes runs
    /// AfterDamageReceived like any hit: Self-Forming Clay arms its block, Demon Tongue heals,
    /// Centennial Puzzle draws. The tick is `AfterSideTurnStart`, after this turn's block clear,
    /// so the Clay's block comes on the NEXT turn. Not reachable today (nothing in the game
    /// poisons the player); fixed with the other unpowered-damage routes (C1-F02).
    /// </summary>
    [Fact]
    public void PoisonTickArmsSelfFormingClay()
    {
        var fight = Fight
            .WithRelics(RelicEffects.SelfFormingClay)
            .Enemy(hp: 200)
            .PlayerBuff(BuffId.Poison, 3);

        fight.EndTurn();

        Assert.Equal(0, fight.State.PlayerBlock);
        Assert.Equal(3, fight.PlayerBuffAmount(BuffId.BlockNextTurn));
    }

    /// <summary>
    /// Hook.ModifyDamage runs every ADDITIVE modifier (Strength, Vigor, relic bonuses) and only
    /// then the MULTIPLICATIVE ones, and PenNib.ModifyDamageMultiplicative is one of them: the
    /// tenth Attack is (6 + 3 Strength) * 2 = 18. The emulator doubles the card's own number
    /// before Strength is added (CardEffects.Dmg, then BuffSystem.IncomingDamage): 6 * 2 + 3 = 15.
    /// </summary>
    [AuditFact("Pen Nib doubles the card number before Strength is added instead of after")]
    public void PenNibDoublesStrengthToo()
    {
        var fight = Fight.WithRelics(RelicEffects.PenNib).Enemy(hp: 999).PlayerBuff(BuffId.Strength, 3);
        fight.State.Energy = 999;
        fight.State.Hand = [.. Enumerable.Range(0, 10).Select(_ => Card(IC.StrikeIronclad))];

        for (int i = 0; i < 9; i++)
        {
            int before = fight.Enemy0.Hp;
            fight.Play();
            Assert.Equal(before - 9, fight.Enemy0.Hp);
        }

        int last = fight.Enemy0.Hp;
        fight.Play();

        Assert.Equal(last - 18, fight.Enemy0.Hp);
    }

    /// <summary>
    /// UnmovablePower.ModifyBlockMultiplicative returns 1 unless props.IsCardOrMonsterMove()
    /// (the Move flag), and it counts only earlier Move-flagged BlockGainedEntry objects. Feel No
    /// Pain's block is ValueProp.Unpowered, so it is neither doubled nor counted: the first
    /// Defend afterwards is still the doubled one. The emulator doubles every gain.
    /// </summary>
    [Fact]
    public void UnmovableLeavesUnpoweredBlockAlone()
    {
        var fight = Fight
            .Hand(Card(IC.DefendIronclad))
            .Energy(3)
            .Enemy(hp: 100)
            .PlayerBuff(BuffId.UnmovablePower, 1)
            .PlayerBuff(BuffId.FeelNoPain, 3);

        CardEffects.ExhaustCard(fight.State, Card(IC.StrikeIronclad), rng: new Random(0));
        Assert.Equal(3, fight.State.PlayerBlock);

        fight.Play();

        // The Defend is the turn's first Move block: 5 doubled to 10.
        Assert.Equal(13, fight.State.PlayerBlock);
    }

    /// <summary>
    /// DarkEmbracePower.AfterCardExhausted draws for every non-Ethereal exhaust. The auto-resolved
    /// selection path (state.AutoPlaying) called ExhaustCard without the RNG, which skipped the
    /// draw. That path is gone: an auto-played card's screen is awaited and answered like any
    /// other (CombatEngine.AwaitSelection), through the one resolution, which draws.
    /// </summary>
    [Fact]
    public void DarkEmbraceDrawsWhenAnAutoPlayedSelectionExhausts()
    {
        // Havoc plays True Grit+ off the draw pile; its exhaust screen over the lone Strike is
        // answered without asking, and Dark Embrace draws for that exhaust and for Havoc's
        // exhaust of True Grit itself.
        var fight = Fight
            .Hand(Card(IC.Havoc), Card(IC.StrikeIronclad))
            .Energy(3)
            .Draw(Card(IC.TrueGrit, upgraded: true), Card(IC.DefendIronclad), Card(IC.DefendIronclad))
            .Enemy(hp: 100)
            .PlayerBuff(BuffId.DarkEmbrace, 1);

        fight.Play(0);

        Assert.Null(fight.Pending);
        Assert.Contains(fight.State.ExhaustPile, c => c.DefId == IC.StrikeIronclad);
        Assert.Equal([IC.DefendIronclad, IC.DefendIronclad], fight.State.Hand.Select(c => c.DefId));
    }
}
