namespace Sts2Emulator.Core;

using Rng;
using Sts2Emulator.Solver;

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
    public CountingRandom Copy() => new(Seed, _rng.Copy(), CallCount) { Chance = Chance, ChanceStream = ChanceStream };

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

    /// <summary>
    /// The chance hook (docs/rebuild/FIGHT_SOLVER.md section 4): null everywhere outside the
    /// solver, and then every call below is the draw it always was, behind one null check. Set on a
    /// solver's copy, each call is put to the source as a <see cref="ChanceCall" /> (this stream,
    /// the call's index, its range or branch weights) and answered by it. A copy of the stream
    /// (<see cref="Copy" />) keeps the source, so a hooked combat's clones stay hooked; moving a
    /// stream (<see cref="RestoreFrom" />, <see cref="Become" />) keeps the receiver's own.
    /// </summary>
    internal IChanceSource? Chance { get; private set; }

    /// <summary>The stream's index in <c>StateCloning.StreamNames</c> while hooked; -1 for a step's own stream.</summary>
    internal int ChanceStream { get; private set; } = -1;

    /// <summary>Hooks this stream to <paramref name="source" /> (null unhooks it) as stream <paramref name="stream" />.</summary>
    internal void SetChance(IChanceSource? source, int stream)
    {
        Chance = source;
        ChanceStream = stream;
    }

    public override int Next(int maxValue)
    {
        if (Chance is not null)
        {
            return Hooked(ChanceKind.Next, 0, maxValue);
        }

        CallCount++;
        CombatTrace.OnRng(this);
        return _rng.Next(maxValue);
    }

    public override int Next()
    {
        if (Chance is not null)
        {
            return Hooked(ChanceKind.Next, 0, int.MaxValue);
        }

        CallCount++;
        CombatTrace.OnRng(this);
        return _rng.Next(int.MaxValue);
    }

    public override int Next(int minValue, int maxValue)
    {
        if (Chance is not null)
        {
            return Hooked(ChanceKind.Next, minValue, maxValue);
        }

        CallCount++;
        CombatTrace.OnRng(this);
        return _rng.Next(minValue, maxValue);
    }

    public override double NextDouble()
    {
        if (Chance is { } chance)
        {
            // A raw float roll has no odds the solver could give it; the source refuses it
            // (`unlisted-roll`) or, observing only, lets the stream answer.
            var call = new ChanceCall(ChanceStream, CallCount, ChanceKind.RawDouble, 0, 0, null, null);
            if (chance.Answer(in call) is not null)
            {
                throw new InvalidOperationException("a chance source cannot answer a raw NextDouble");
            }

            CallCount++;
            CombatTrace.OnRng(this);
            double d = _rng.NextDouble();
            chance.Drawn(in call, -1);
            return d;
        }

        CallCount++;
        CombatTrace.OnRng(this);
        return _rng.NextDouble();
    }

    /// <summary>
    /// A weighted branch (<see cref="Roll.Branch" />): one float draw, mapped to a branch by
    /// <paramref name="decide" />, exactly the inline code it replaced; hooked, the source answers the
    /// branch index from the weights. The stream moves by one draw either way.
    /// </summary>
    internal int Branch(string site, double[] weights, Func<double, int> decide)
    {
        if (Chance is not { } chance)
        {
            return decide(NextDouble());
        }

        var call = new ChanceCall(ChanceStream, CallCount, ChanceKind.Branch, 0, weights.Length, weights, site);
        int? answer = chance.Answer(in call);
        CallCount++;
        CombatTrace.OnRng(this);
        double d = _rng.NextDouble();
        int value = answer ?? decide(d);
        chance.Drawn(in call, value);
        return value;
    }

    /// <summary>A uniform call put to the source; the stream still takes its one draw, so its position is the unhooked one.</summary>
    private int Hooked(ChanceKind kind, int minValue, int maxValue)
    {
        var chance = Chance!;
        var call = new ChanceCall(ChanceStream, CallCount, kind, minValue, maxValue, null, null);
        int? answer = chance.Answer(in call);
        CallCount++;
        CombatTrace.OnRng(this);
        int drawn = minValue == 0 ? _rng.Next(maxValue) : _rng.Next(minValue, maxValue);
        int value = answer ?? drawn;
        chance.Drawn(in call, value);
        return value;
    }
}
