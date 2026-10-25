// The migrated models (docs/rebuild/TIER1_KERNEL.md section 4): one `internal static class <GameClass>` per game model under
// Core2/Models/{Powers,Relics,Cards,Monsters,Potions}/ (namespace Sts2Emulator.Core2.Models.<Kind>), each with a
// `internal static void Register(HandlerTable t)` that adds its handlers. A model is listed here when its family migrates; its
// handlers register only while that family's switch is on.
//
// One named block per handler batch (docs/rebuild/CLOUD_BRIEFS.md section 3), and `All` spreads the blocks. A batch adds
// its block next to the others and one `.. <Block>` line to `All`; a textual conflict between two batches is resolved by
// keeping every block. The blocks to come, by the names the briefs and the rebuild branches use:
//   local: Monsters (monster-1 and monster-2, rebuild/f08-m1, f08-m2), L1, L2 and L3a (the LOCAL non-monster models,
//          rebuild/f08-l1, f08-l2, f08-l3a);
//   cloud: Power1, Power2, Power3, Power4 (C2-C5) and Relic1 (C6).
using Sts2Emulator.Core2.Models.Cards;
using Sts2Emulator.Core2.Models.Monsters;
using Sts2Emulator.Core2.Models.Potions;
using Sts2Emulator.Core2.Models.Powers;
using Sts2Emulator.Core2.Models.Relics;

namespace Sts2Emulator.Core2.Models;

public static class Registry
{
    /// <summary>C1, the pilot (cloud/f08-pilot): the port list's power-5, relic-2 and card-1 rows.</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Pilot =
    [
        PhantomBladesPower.Register,
        LethalityPower.Register,
        TrackingPower.Register,
        DoubleDamagePower.Register,
        VitruvianMinion.Register,
        TheBoot.Register,
        HandDrill.Register,
        Melancholy.Register,
    ];

    /// <summary>F08/F10 monster-1 and monster-2 (rebuild/f08-m1, f08-m2).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Monsters =
    [
        // F08/F10 monster-1 (docs/rebuild/port-f08-f10.md), first half.
        Ovicopter.Register, SoulNexus.Register, ThievingHopper.Register, CubexConstruct.Register,
        DecimillipedeSegment.Register, SlumberingBeetle.Register, Queen.Register, Vantom.Register, Aeonglass.Register,
        // Second half of monster-1, and monster-2 (R6.m2).
        LagavulinMatriarch.Register, Crusher.Register, Rocket.Register,
        OwlMagistrate.Register, SnappingJaxfruit.Register, KnowledgeDemon.Register, TheInsatiable.Register, SoulFysh.Register,
        KinPriest.Register, TestSubject.Register, WaterfallGiant.Register,
    ];

    /// <summary>Batch L1, the LOCAL non-monster models (rebuild/f08-l1).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> L1 =
    [
        // Batch L1: the LOCAL non-monster models (docs/rebuild/CLOUD_BRIEFS.md section 1).
        PaperCutsPower.Register, TheGambitPower.Register, HardToKillPower.Register, SlowPower.Register,
        MinionPower.Register, ReattachPower.Register, AdaptablePower.Register, SteamEruptionPower.Register,
        PainfulStabsPower.Register, LizardTail.Register, FairyInABottle.Register,
    ];

    /// <summary>Batch L2, the LOCAL models L1 left (rebuild/f08-l2).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> L2 =
    [
        // Batch L2: the player-side AfterDamageReceived listeners Core2's damage command called as one old body.
        InfernoPower.Register, RupturePower.Register, SelfFormingClay.Register, CentennialPuzzle.Register,
        DemonTongue.Register, RedSkull.Register,
        // Batch L2: the AfterDamageReceived tallies Core2's damage command kept, and Curl Up.
        BeatingRemnant.Register, LavaLamp.Register, HardenedShellPower.Register, CurlUpPower.Register,
        // Batch L2: the death pairs and the open pairs.
        ShrinkPower.Register, IllusionPower.Register, ImbalancedPower.Register, VitalSparkPower.Register,
        SlumberPower.Register, HexPower.Register, TangledPower.Register, RingingPower.Register, SandpitPower.Register,
        BurrowedPower.Register, DampenPower.Register, AsleepPower.Register, MeatOnTheBone.Register,
    ];

    /// <summary>R4c, the LOCAL Conqueror (rebuild/f08-r4).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> R4c =
    [
        // R4c: Conqueror, a LOCAL model (Sovereign Blade's one AttackCmd takes its doubling from this hook)
        ConquerorPower.Register,
    ];

    /// <summary>C2 (cloud/f08-power-1): the port list's power-1 rows that are CLOUD (CLOUD_BRIEFS section 1). Thorns and
    /// Flame Barrier are not here: their old bodies are private, the stop rule's case (the branch's PR lists them).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Power1 =
    [
        StrengthPower.Register,
        VulnerablePower.Register,
        WeakPower.Register,
        VigorPower.Register,
        TankPower.Register,
        ColossusPower.Register,
        ShriekPower.Register,
    ];

    /// <summary>C3 (cloud/f08-power-2): the port list's power-2 rows of the models the batch ports.</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Power2 =
    [
        TaintedPower.Register,
        GigantificationPower.Register,
        IntangiblePower.Register,
        SlipperyPower.Register,
        BufferPower.Register,
        PossessSpeedPower.Register,
        PossessStrengthPower.Register,
        SwipePower.Register,
    ];

    /// <summary>C4 (cloud/f08-power-3): the port list's power-3 rows. Not here: Crab Rage and Surrounded, whose AfterDeath
    /// bodies (CombatEngine.TryEnrageCrabPartner, FacePast, TurnPlayerToFaceSurvivor) are private (the stop rule of
    /// CLOUD_BRIEFS section 3); Stock and Asleep, the local batch L2's.</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Power3 =
    [
        FlutterPower.Register,
        PersonalHivePower.Register,
        PlowPower.Register,
        SoarPower.Register,
    ];

    /// <summary>C5 (cloud/f08-power-4): the port list's power-4 rows that are CLOUD (Conqueror is ported locally, round R4c).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Power4 =
    [
        CalcifyPower.Register,
        AccuracyPower.Register,
        SicEmPower.Register,
        ConstrictPower.Register,
        HeistPower.Register,
    ];

    /// <summary>C6 (cloud/f08-relic-1): the port list's relic-1 rows.</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> Relic1 =
    [
        TungstenRod.Register,
        StrikeDummy.Register,
        PenNib.Register,
        GremlinHorn.Register,
        MiniatureCannon.Register,
        MysticLighter.Register,
        FakeStrikeDummy.Register,
    ];

    /// <summary>Batch L3a, the LOCAL models whose old bodies were private or acted for every holder (rebuild/f08-l3a).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> L3a =
    [
        // Batch L3a: the retaliations (re-entries), and the death answers made per owner.
        ThornsPower.Register, FlameBarrierPower.Register, CrabRagePower.Register, SurroundedPower.Register,
        RavenousPower.Register,
    ];

    /// <summary>Batch L3b (rebuild/f08-l3b): the LOCAL models that needed a kernel addition: Surprise and Infested (the
    /// HookCtx.Niche stream), Stock (CreatureCmd.Add) and Hang (the card path's multiply gated off under the switch).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> L3b =
    [
        SurprisePower.Register,
        InfestedPower.Register,
        StockPower.Register,
        HangPower.Register,
    ];

    /// <summary>Batch R7a (rebuild/f08-r7a), the R7 blockers: Die For You, the pet redirect the enemy-attack site reaches
    /// through AttackCmd's CreatureCmd.Damage, as the card path does.</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> R7a =
    [
        DieForYouPower.Register,
    ];

    /// <summary>R7f: the damage folds of models no record reaches (port_list --unreached): Diamond Diadem's and
    /// Undying Sigil's halving.</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> R7f =
    [
        DiamondDiademPower.Register,
        UndyingSigil.Register,
    ];

    // Declare every block above All: static fields initialise in textual order, so a block below All is still null when
    // All spreads it.
    /// <summary>Every migrated model's registration, one block per batch, in no particular order (the table is keyed,
    /// not ordered).</summary>
    public static readonly IReadOnlyList<Action<HandlerTable>> All =
    [
        .. Pilot,
        .. Monsters,
        .. L1,
        .. L2,
        .. R4c,
        .. Power1,
        .. Power2,
        .. Power3,
        .. Power4,
        .. L3b,
        .. Relic1,
        .. L3a,
        .. R7a,
        .. R7f,
    ];
}
