using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Interop;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Every C ABI entry point that takes an encounter id reads Corpse Slugs' and Seapunk's
/// shared ids the same way, through <see cref="WeakContext.Resolve" />.
/// </summary>
/// <remarks>
/// <para>
/// The two pairs are two encounters each (<c>CorpseSlugsWeak</c> 9 and
/// <c>CorpseSlugsNormal</c> 88, <c>SeapunkWeak</c> 12 and <c>SeapunkNormal</c> 22), but
/// callers across the C ABI still name the weak id and say which fight with a weak
/// context: [0,3) weak, anything else normal. The resets that take no context
/// (<c>Sts2_ResetEncounter</c>, <c>Sts2_ResetWithDeckAndEncounter</c>,
/// <c>Sts2_ResetWithDeckEncounterAndRelics</c>) have always dealt the NORMAL encounter for
/// those ids. The first version of the split read the context only on the resets that take
/// one, so those three -- and the gym's <c>reset_encounter</c> without a total_floor, which
/// is <c>Sts2_ResetEncounter</c> -- quietly switched to the weak roster.
/// </para>
/// <para>
/// Each test here goes through the unmanaged entry point itself, as ctypes does, and reads
/// the fight back through <c>Sts2_EncounterId</c> and <c>Sts2_GetEnemies</c>.
/// </para>
/// </remarks>
public unsafe class WeakContextEntryPointTests
{
    private const int Slugs = RunConstants.CorpseSlugsEncounterId;
    private const int SlugsNormal = RunConstants.CorpseSlugsNormalEncounterId;
    private const int Seapunk = RunConstants.SeapunkEncounterId;
    private const int SeapunkNormal = RunConstants.SeapunkNormalEncounterId;
    private const int Ascension = 10;

    /// <summary>(id named, context, id dealt, monsters) for a reset that takes a context.</summary>
    public static TheoryData<int, int, int, int> WithContext =>
        new()
        {
            { Slugs, WeakContext.None, SlugsNormal, 3 },
            { Slugs, 0, Slugs, 2 },
            { Slugs, 2, Slugs, 2 },
            { Slugs, 3, SlugsNormal, 3 },
            { SlugsNormal, 0, SlugsNormal, 3 },
            { Seapunk, WeakContext.None, SeapunkNormal, 2 },
            { Seapunk, 1, Seapunk, 1 },
            { SeapunkNormal, 1, SeapunkNormal, 2 },
            { SeapunkNormal, WeakContext.None, SeapunkNormal, 2 },
        };

    /// <summary>(id named, id dealt, monsters) for a reset that takes none: the normal fight.</summary>
    public static TheoryData<int, int, int> WithoutContext =>
        new()
        {
            { Slugs, SlugsNormal, 3 },
            { SlugsNormal, SlugsNormal, 3 },
            { Seapunk, SeapunkNormal, 2 },
            { SeapunkNormal, SeapunkNormal, 2 },
        };

    /// <summary>The exports these tests cover, by entry point name.</summary>
    private static readonly string[] Covered =
    [
        "Sts2_ResetEncounter",
        "Sts2_ResetEncounterWeak",
        "Sts2_ResetEncounterAtFloor",
        "Sts2_ResetEncounterAtFloorWithExtraCards",
        "Sts2_ResetWithDeckAndEncounter",
        "Sts2_ResetWithDeckEncounterAndRelics",
        "Sts2_ResetArena",
        "Sts2Run_StartCombat",
    ];

    /// <summary>
    /// A new export that takes an encounter id has to be added here, with its own test: the
    /// regression this file pins was an export nobody had listed.
    /// </summary>
    [Fact]
    public void EveryExportThatTakesAnEncounterIdIsCovered()
    {
        var exports = typeof(NativeExports)
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.GetParameters().Any(p => p.Name == "encounterId"))
            .Select(m => m.GetCustomAttribute<UnmanagedCallersOnlyAttribute>()?.EntryPoint)
            .Where(name => name is not null)
            .Order()
            .ToArray();

        Assert.Equal(Covered.Order().ToArray(), exports);
    }

    [Theory]
    [MemberData(nameof(WithoutContext))]
    public void ResetEncounter(int named, int dealt, int monsters) =>
        AssertFight(h => Native.ResetEncounter(h, named), dealt, monsters);

    [Theory]
    [MemberData(nameof(WithoutContext))]
    public void ResetWithDeckAndEncounter(int named, int dealt, int monsters) =>
        AssertFight(h => Native.ResetWithDeckAndEncounter(h, named), dealt, monsters);

    [Theory]
    [MemberData(nameof(WithoutContext))]
    public void ResetWithDeckEncounterAndRelics(int named, int dealt, int monsters) =>
        AssertFight(h => Native.ResetWithDeckEncounterAndRelics(h, named), dealt, monsters);

    [Theory]
    [MemberData(nameof(WithContext))]
    public void ResetEncounterWeak(int named, int context, int dealt, int monsters) =>
        AssertFight(h => Native.ResetEncounterWeak(h, named, context), dealt, monsters);

    [Theory]
    [MemberData(nameof(WithContext))]
    public void ResetEncounterAtFloor(int named, int context, int dealt, int monsters) =>
        AssertFight(h => Native.ResetEncounterAtFloor(h, named, context), dealt, monsters);

    [Theory]
    [MemberData(nameof(WithContext))]
    public void ResetEncounterAtFloorWithExtraCards(int named, int context, int dealt, int monsters) =>
        AssertFight(h => Native.ResetEncounterAtFloorWithExtraCards(h, named, context), dealt, monsters);

    [Theory]
    [MemberData(nameof(WithContext))]
    public void ResetArena(int named, int context, int dealt, int monsters) =>
        AssertFight(h => Native.ResetArena(h, named, context), dealt, monsters);

    [Theory]
    [MemberData(nameof(WithContext))]
    public void RunStartCombat(int named, int context, int dealt, int monsters)
    {
        int handle = Native.RunCreate();
        try
        {
            Assert.Equal(0, Native.RunStartCombat(handle, named, context));
            Assert.Equal(dealt, Native.RunEncounterId(handle));
            Assert.True(RunNativeExports.TryGetRun(handle, out var run));
            Assert.Equal(monsters, run.State.ActiveCombat!.Enemies.Count);
        }
        finally
        {
            Native.RunDestroy(handle);
        }
    }

    /// <summary>
    /// The rolled first combat names no encounter: it deals from the act-1 weak pools
    /// (<c>CombatFactory.SelectFirstCombatEncounter</c>), so a Corpse Slugs or Seapunk it
    /// rolls is the weak encounter, as the game's weak pool entry is.
    /// </summary>
    [Fact]
    public void TheRolledFirstCombatIsTheWeakEncounter()
    {
        var seen = new HashSet<int>();
        for (int seed = 0; seed < 400 && seen.Count < 2; seed++)
        {
            int handle = Native.Create(seed);
            try
            {
                Native.Reset(handle);
                int id = Native.EncounterId(handle);
                if (id is Slugs or Seapunk)
                {
                    Assert.Equal(id == Slugs ? 2 : 1, Native.Enemies(handle));
                    seen.Add(id);
                }

                Assert.NotEqual(SlugsNormal, id);
                Assert.NotEqual(SeapunkNormal, id);
            }
            finally
            {
                Native.Destroy(handle);
            }
        }

        Assert.Equal([Slugs, Seapunk], seen.Order());
    }

    private static void AssertFight(Action<int> reset, int dealt, int monsters)
    {
        int handle = Native.Create(7);
        try
        {
            reset(handle);
            Assert.Equal(dealt, Native.EncounterId(handle));
            Assert.Equal(monsters, Native.Enemies(handle));
        }
        finally
        {
            Native.Destroy(handle);
        }
    }

    /// <summary>The exports, through the same unmanaged entry points ctypes uses.</summary>
    private static class Native
    {
        private static readonly int[] Deck = RunConstants.StartingDeckIds(Ascension);
        private static readonly int[] NoIds = [0];

        public static int Create(int seed)
        {
            delegate* unmanaged<int, int> f = &NativeExports.Sts2_Create;
            return f(seed);
        }

        public static void Destroy(int handle)
        {
            delegate* unmanaged<int, void> f = &NativeExports.Sts2_Destroy;
            f(handle);
        }

        public static int EncounterId(int handle)
        {
            delegate* unmanaged<int, int> f = &NativeExports.Sts2_EncounterId;
            return f(handle);
        }

        public static int Enemies(int handle)
        {
            delegate* unmanaged<int, int*, int, int> f = &NativeExports.Sts2_GetEnemies;
            var buf = new int[NativeExports.MAX_ENEMIES * 3];
            fixed (int* b = buf)
            {
                return f(handle, b, NativeExports.MAX_ENEMIES);
            }
        }

        public static void Reset(int handle)
        {
            delegate* unmanaged<int, int*, void> f = &NativeExports.Sts2_Reset;
            fixed (int* obs = new int[NativeExports.OBS_SIZE])
            {
                f(handle, obs);
            }
        }

        public static void ResetEncounter(int handle, int encounterId)
        {
            delegate* unmanaged<int, int, int*, void> f = &NativeExports.Sts2_ResetEncounter;
            fixed (int* obs = new int[NativeExports.OBS_SIZE])
            {
                f(handle, encounterId, obs);
            }
        }

        public static void ResetEncounterWeak(int handle, int encounterId, int context)
        {
            delegate* unmanaged<int, int, int, int*, void> f =
                &NativeExports.Sts2_ResetEncounterWeak;
            fixed (int* obs = new int[NativeExports.OBS_SIZE])
            {
                f(handle, encounterId, context, obs);
            }
        }

        public static void ResetEncounterAtFloor(int handle, int encounterId, int context)
        {
            delegate* unmanaged<int, int, int, int, int, int*, void> f =
                &NativeExports.Sts2_ResetEncounterAtFloor;
            fixed (int* obs = new int[NativeExports.OBS_SIZE])
            {
                f(handle, encounterId, context, 1, Ascension, obs);
            }
        }

        public static void ResetEncounterAtFloorWithExtraCards(
            int handle,
            int encounterId,
            int context
        )
        {
            delegate* unmanaged<int, int*, int, int, int, int, int, int*, void> f =
                &NativeExports.Sts2_ResetEncounterAtFloorWithExtraCards;
            int[] extra = [Deck[0]];
            fixed (int* e = extra)
            fixed (int* obs = new int[NativeExports.OBS_SIZE])
            {
                f(handle, e, extra.Length, encounterId, context, 1, Ascension, obs);
            }
        }

        public static void ResetWithDeckAndEncounter(int handle, int encounterId)
        {
            delegate* unmanaged<int, int*, int, int, int*, void> f =
                &NativeExports.Sts2_ResetWithDeckAndEncounter;
            fixed (int* d = Deck)
            fixed (int* obs = new int[NativeExports.OBS_SIZE])
            {
                f(handle, d, Deck.Length, encounterId, obs);
            }
        }

        public static void ResetWithDeckEncounterAndRelics(int handle, int encounterId)
        {
            delegate* unmanaged<int, int*, int, int, int*, int, int*, void> f =
                &NativeExports.Sts2_ResetWithDeckEncounterAndRelics;
            fixed (int* d = Deck)
            fixed (int* r = NoIds)
            fixed (int* obs = new int[NativeExports.OBS_SIZE])
            {
                f(handle, d, Deck.Length, encounterId, r, 0, obs);
            }
        }

        public static void ResetArena(int handle, int encounterId, int context)
        {
            delegate* unmanaged<
                int,
                int*,
                int,
                int*,
                int,
                int*,
                int,
                int,
                int*,
                int,
                int*,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int*,
                void> f = &NativeExports.Sts2_ResetArena;
            fixed (int* d = Deck)
            fixed (int* none = NoIds)
            fixed (int* obs = new int[NativeExports.OBS_SIZE])
            {
                f(
                    handle,
                    d,
                    Deck.Length,
                    none,
                    0,
                    none,
                    0,
                    encounterId,
                    none,
                    0,
                    none,
                    0,
                    64,
                    80,
                    0,
                    -1,
                    Ascension,
                    1,
                    context,
                    obs
                );
            }
        }

        public static int RunCreate()
        {
            delegate* unmanaged<int> create = &NativeExports.Sts2Run_Create;
            delegate* unmanaged<int, byte*, int, int, int*, int> reset =
                &NativeExports.Sts2Run_Reset;
            int handle = create();
            byte[] seed = Encoding.UTF8.GetBytes("WEAKCTX01");
            fixed (byte* s = seed)
            fixed (int* obs = new int[RunConstants.RunObsSize])
            {
                Assert.Equal(0, reset(handle, s, seed.Length, Ascension, obs));
            }

            return handle;
        }

        public static int RunStartCombat(int handle, int encounterId, int context)
        {
            delegate* unmanaged<
                int,
                int*,
                int,
                int,
                int*,
                int,
                int,
                int,
                int*,
                int,
                int,
                int,
                int*,
                int> f = &NativeExports.Sts2Run_StartCombat;
            fixed (int* d = Deck)
            fixed (int* none = NoIds)
            fixed (int* obs = new int[RunConstants.RunObsSize])
            {
                return f(handle, d, Deck.Length, encounterId, none, 0, 64, 80, none, 0, 99, context, obs);
            }
        }

        public static int RunEncounterId(int handle)
        {
            delegate* unmanaged<int, int> f = &NativeExports.Sts2Run_EncounterId;
            return f(handle);
        }

        public static void RunDestroy(int handle)
        {
            delegate* unmanaged<int, void> f = &NativeExports.Sts2Run_Destroy;
            f(handle);
        }
    }
}
