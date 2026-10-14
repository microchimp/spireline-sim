namespace Sts2Emulator.Core;

using Rng;

/// <summary>
/// Wraps the game's deterministic RNG and counts calls so the caller can sync
/// run-level streams after combat-local RNG consumption.
///
/// Backed by <see cref="MegaRandom" /> (Xoshiro256**), the generator the game
/// actually uses. It is a <see cref="Random" /> subclass purely so existing
/// helpers like CardEffects.ShufflePile can take it; the values come from
/// MegaRandom, never from the base class.
/// </summary>
public sealed class CountingRandom : Random
{
    private readonly MegaRandom _rng;

    /// <summary>The stream's raw seed, kept so the stream can be cloned.</summary>
    public int Seed { get; private set; }

    public int CallCount { get; private set; }

    /// <summary>
    /// Seed is the stream's raw seed. It goes through uint so it is zero-extended
    /// into MegaRandom's ulong, matching the game's <c>new MegaRandom(uintSeed)</c>.
    /// </summary>
    public CountingRandom(int seed)
    {
        Seed = seed;
        _rng = new MegaRandom(unchecked((uint)seed));
    }

    /// <summary>
    /// A stream already <paramref name="counter" /> draws in, which is
    /// <c>Rng.FastForwardCounter</c>: it takes one <c>MegaRandom.NextInt()</c> per step,
    /// the same draw a `Next` costs, and the game's own constructor uses it to resume a
    /// saved stream.
    /// </summary>
    /// <remarks>
    /// A rebuilt combat starts every stream at zero while the live run's had advanced, so
    /// a card that ROLLS matched the game's logic and disagreed about the result. The mod
    /// reports seed and counter for each named stream; this is the other half.
    /// </remarks>
    private CountingRandom(int seed, MegaRandom state, int callCount)
    {
        Seed = seed;
        _rng = state;
        CallCount = callCount;
    }

    /// <summary>The generator's lanes, which with <see cref="Seed" /> and <see cref="CallCount" /> are the whole stream.</summary>
    internal (ulong S0, ulong S1, ulong S2, ulong S3) Lanes => _rng.Lanes;

    /// <summary>The stream <see cref="Seed" />, <see cref="CallCount" /> and <see cref="Lanes" /> describe.</summary>
    internal static CountingRandom FromLanes(int seed, int callCount, (ulong S0, ulong S1, ulong S2, ulong S3) lanes) =>
        new(seed, MegaRandom.FromLanes(lanes), callCount);

    /// <summary>An exact copy at the same position, in O(1) (see <see cref="GameRng.Copy" />).</summary>
    public CountingRandom Copy() => new(Seed, _rng.Copy(), CallCount);

    /// <summary>
    /// Moves this stream back to exactly <paramref name="other" />'s position, in place -- for a
    /// caller that holds this object, as a step's caller holds its stream (CombatEngine.Step).
    /// </summary>
    public void RestoreFrom(CountingRandom other)
    {
        _rng.CopyStateFrom(other._rng);
        CallCount = other.CallCount;
    }

    /// <summary>
    /// Makes this stream, in place, exactly <paramref name="other" />: its seed as well as its
    /// position, for a stream that is to go on as a different one while whoever holds this object
    /// keeps holding it -- a step's stream and the combat's at a screen a determinized copy answers
    /// (<c>HiddenAtScreen</c>).
    /// </summary>
    public void Become(CountingRandom other)
    {
        RestoreFrom(other);
        Seed = other.Seed;
    }

    /// <summary>
    /// A combat stream picking up exactly where a run stream stands: the same as
    /// <c>new CountingRandom(stream.RawSeed, stream.CallCount)</c>, in O(1).
    /// </summary>
    public static CountingRandom At(GameRng stream) =>
        new(stream.RawSeed, stream.Generator.Copy(), stream.CallCount);

    public CountingRandom(int seed, int counter)
        : this(seed)
    {
        for (int i = 0; i < counter; i++)
        {
            _rng.NextInt();
        }

        CallCount = counter;
    }

    public override int Next(int maxValue)
    {
        CallCount++;
        CombatTrace.OnRng(this);
        return _rng.Next(maxValue);
    }

    public override int Next()
    {
        CallCount++;
        CombatTrace.OnRng(this);
        return _rng.Next(int.MaxValue);
    }

    public override int Next(int minValue, int maxValue)
    {
        CallCount++;
        CombatTrace.OnRng(this);
        return _rng.Next(minValue, maxValue);
    }

    public override double NextDouble()
    {
        CallCount++;
        CombatTrace.OnRng(this);
        return _rng.NextDouble();
    }
}
