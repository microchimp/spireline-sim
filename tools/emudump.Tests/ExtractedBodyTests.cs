// ExtractedBody.Is (tools/emudump/Engines/Old/ExtractedBody.cs) and the callees rule of tools/fidelity/hook_table.py must judge the
// same functions extracted (docs/rebuild/MIGRATION_F08.md R5: the <Model>_<Hook> bodies). Both suites pin the engine's set against
// the one list below (tests/python/test_hook_table.py: EXTRACTED_FUNCTIONS), so a function the two would judge differently fails one.
using System.Reflection;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Xunit;

public class ExtractedBodyTests
{
    /// <summary>Every extracted body in the engine, as Class.Function in ordinal order: the Python test's list, verbatim.</summary>
    static readonly string[] Pinned =
    [
        "BuffSystem.DoubleDamage_ModifyDamageMultiplicative",
        "BuffSystem.Flutter_ModifyDamageMultiplicative",
        "BuffSystem.Shrink_ModifyDamageMultiplicative",
        "BuffSystem.Soar_ModifyDamageMultiplicative",
        "BuffSystem.Strength_ModifyDamageAdditive",
        "BuffSystem.Surrounded_ModifyDamageMultiplicative",
        "BuffSystem.Tainted_ModifyDamageAdditive",
        "BuffSystem.Tank_ModifyDamageMultiplicative",
        "BuffSystem.Tracking_ModifyDamageMultiplicative",
        "BuffSystem.Vigor_ModifyDamageAdditive",
        "BuffSystem.Vulnerable_ModifyDamageMultiplicative",
        "BuffSystem.Weak_ModifyDamageMultiplicative",
        "CardEffects.Accuracy_ModifyDamageAdditive",
        "CardEffects.BeatingRemnant_AfterDamageReceived",
        "CardEffects.Buffer_AfterModifyingHpLostAfterOsty",
        "CardEffects.Buffer_ModifyHpLostAfterOstyLate",
        "CardEffects.Calcify_ModifyDamageAdditive",
        "CardEffects.CurlUp_AfterDamageReceived",
        "CardEffects.Flutter_AfterDamageReceived",
        "CardEffects.Gigantification_ModifyDamageMultiplicative",
        "CardEffects.HardenedShell_AfterDamageReceived",
        "CardEffects.HardenedShell_ModifyHpLostBeforeOstyLate",
        "CardEffects.Lethality_ModifyDamageMultiplicative",
        "CardEffects.PersonalHive_AfterDamageReceived",
        "CardEffects.PhantomBlades_ModifyDamageAdditive",
        "CardEffects.Rupture_AfterDamageReceived",
        "CardEffects.SicEm_AfterDamageGiven",
        "CardEffects.Slippery_AfterModifyingHpLostAfterOsty",
        "CardEffects.Slippery_ModifyHpLostAfterOsty",
        "CardEffects.Slumber_AfterDamageReceived",
        "CombatEngine.Adaptable_ShouldStopCombatFromEnding",
        "CombatEngine.Constrict_AfterDeath",
        "CombatEngine.Dampen_AfterDeath",
        "CombatEngine.FairyInABottle_AfterPreventingDeath",
        "CombatEngine.GremlinHorn_AfterDeath",
        "CombatEngine.Heist_BeforeDeath",
        "CombatEngine.Hex_AfterDeath",
        "CombatEngine.Infested_AfterDeath",
        "CombatEngine.Infested_ShouldStopCombatFromEnding",
        "CombatEngine.PossessSpeed_AfterDeath",
        "CombatEngine.PossessStrength_AfterDeath",
        "CombatEngine.Ravenous_AfterDeath",
        "CombatEngine.SteamEruption_ShouldStopCombatFromEnding",
        "CombatEngine.Stock_ShouldStopCombatFromEnding",
        "CombatEngine.Surprise_AfterDeath",
        "CombatEngine.Surprise_ShouldStopCombatFromEnding",
        "CombatEngine.Swipe_BeforeDeath",
        "EnemyAI.Adaptable_ShouldCreatureBeRemovedFromCombatAfterDeath",
        "EnemyAI.Colossus_ModifyDamageMultiplicative",
        "EnemyAI.Illusion_ShouldCreatureBeRemovedFromCombatAfterDeath",
        "EnemyAI.Imbalanced_AfterDamageGiven",
        "EnemyAI.PainfulStabs_ShouldCreatureBeRemovedFromCombatAfterDeath",
        "EnemyAI.PaperCuts_AfterDamageGiven",
        "EnemyAI.Reattach_ShouldCreatureBeRemovedFromCombatAfterDeath",
        "EnemyAI.SteamEruption_ShouldCreatureBeRemovedFromCombatAfterDeath",
        "EnemyAI.TheGambit_AfterDamageReceived",
        "RelicEffects.BeatingRemnant_ModifyHpLostAfterOsty",
        "RelicEffects.CentennialPuzzle_AfterDamageReceived",
        "RelicEffects.DemonTongue_AfterDamageReceived",
        "RelicEffects.FakeStrikeDummy_ModifyDamageAdditive",
        "RelicEffects.LizardTail_AfterPreventingDeath",
        "RelicEffects.MiniatureCannon_ModifyDamageAdditive",
        "RelicEffects.PenNib_ModifyDamageMultiplicative",
        "RelicEffects.RedSkull_AfterCurrentHpChanged",
        "RelicEffects.SelfFormingClay_AfterDamageReceived",
        "RelicEffects.StrikeDummy_ModifyDamageAdditive",
        "RelicEffects.TungstenRod_ModifyHpLostAfterOsty",
        "RelicEffects.VitruvianMinion_ModifyDamageMultiplicative",
    ];

    const BindingFlags Declared = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance
        | BindingFlags.DeclaredOnly;

    /// <summary>
    /// The methods ExtractedBody.Is accepts, over every type of the emulator's assembly, as Class.Function. The compiler's own
    /// members are left out, since no source function bears their names: the state-machine and closure classes (named
    /// <c>&lt;Method&gt;d__N</c>; the trace reads a frame in one as its declaring method, which is scanned here by its own
    /// name), lambdas and local functions (<c>&lt;Method&gt;b__N</c>), and explicit interface implementations
    /// (<c>System.Collections.IEnumerator.get_Current</c>, whose dotted name the rule would otherwise accept).
    /// </summary>
    static List<string> InEngine() => typeof(CombatState).Assembly.GetTypes()
        .Where(t => !t.Name.StartsWith('<'))
        .SelectMany(t => t.GetMethods(Declared).Select(m => (Type: t, m.Name)))
        .Where(x => !x.Name.StartsWith('<') && !x.Name.Contains('.') && ExtractedBody.Is(x.Type, x.Name))
        .Select(x => $"{x.Type.Name}.{x.Name}")
        .Distinct()
        .OrderBy(n => n, StringComparer.Ordinal)
        .ToList();

    [Fact]
    public void TheEnginesExtractedFunctionsAreThePinnedList()
    {
        Assert.Equal(Pinned, InEngine());
    }

    [Theory]
    [InlineData("Strength_ModifyDamageAdditive", true)]
    [InlineData("A_B", true)]
    [InlineData("Foo1_Bar", true)]
    [InlineData("Foo_Bar_baz", true)] // the first underscore decides
    [InlineData("Foo_bar_Baz", false)] // a lower-case letter after it
    [InlineData("Foo__Bar", false)]
    [InlineData("_Foo_Bar", false)]
    [InlineData("foo_Bar", false)]
    [InlineData("Foo_", false)]
    [InlineData("Foo_1Bar", false)]
    [InlineData("FooBar", false)]
    [InlineData("get_Foo", false)]
    public void TheNameRuleIsAnUpperCaseInitialThenTheFirstUnderscoreThenAnUpperCaseLetter(string name, bool extracted)
    {
        Assert.Equal(extracted, ExtractedBody.Is(typeof(CombatState), name));
    }

    [Fact]
    public void OnlyTheOldEnginesNamespacesCount()
    {
        var core2 = typeof(CombatState).Assembly.GetTypes().First(t => t.Namespace == "Sts2Emulator.Core2");
        Assert.False(ExtractedBody.Is(core2, "Strength_ModifyDamageAdditive"));
        Assert.False(ExtractedBody.Is(typeof(ExtractedBodyTests), "Strength_ModifyDamageAdditive"));
    }
}
