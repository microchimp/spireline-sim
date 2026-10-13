using System.Collections;
using System.Reflection;
using System.Text;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// `RunEngine.Reset` on an engine that has already played starts the same run a new engine
/// does: no counter, pending screen or relic bookkeeping of the earlier run carries over. The
/// native `Sts2Run_Reset` reuses its engine for every episode, so a leak here put each later
/// gym episode on the wrong encounter, event and offer.
/// </summary>
public class RunResetTests
{
    const string FirstSeed = "RESETLEAK1";
    const string NextSeed = "RESETLEAK2";

    /// <summary>A fixed walk: a fight is resolved, anything else takes a legal action by step.</summary>
    static void Play(RunEngine e, int steps)
    {
        var mask = new int[256];
        for (int step = 0; step < steps && e.State.Phase != RunPhase.Complete; step++)
        {
            if (e.State.Phase == RunPhase.Combat)
            {
                e.DebugResolveCombat();
                continue;
            }

            Array.Clear(mask);
            e.WriteActionMask(mask);
            var legal = Enumerable.Range(0, mask.Length).Where(i => mask[i] != 0).ToList();
            if (legal.Count == 0)
            {
                return;
            }

            e.Step(legal[(step * 7 + 3) % legal.Count], -1, out _, out _, out _);
        }
    }

    [Theory]
    [InlineData(PlayerCharacter.Ironclad, 10)]
    [InlineData(PlayerCharacter.Silent, 0)]
    public void Reset_after_a_played_run_is_the_run_a_new_engine_starts(PlayerCharacter character, int ascension)
    {
        var used = new RunEngine();
        used.Reset(FirstSeed, ascension, character);
        Play(used, 400);
        Assert.True(used.State.Floor > 3, "the first run should get past its first floors");

        used.Reset(NextSeed, ascension, character);
        var fresh = new RunEngine();
        fresh.Reset(NextSeed, ascension, character);
        Assert.Equal(Digest(fresh.State), Digest(used.State));

        // And it goes on to the same run, step for step.
        var a = new int[RunConstants.RunObsSize];
        var b = new int[RunConstants.RunObsSize];
        for (int chunk = 0; chunk < 40; chunk++)
        {
            Play(fresh, 10);
            Play(used, 10);
            fresh.WriteObservation(a);
            used.WriteObservation(b);
            Assert.Equal(a, b);
            Assert.Equal(Digest(fresh.State), Digest(used.State));
        }
    }

    /// <summary>Every public field and property of the state, read recursively, as text.</summary>
    static string Digest(object root)
    {
        var sb = new StringBuilder();
        Write(sb, root, "", 0, new HashSet<object>(ReferenceEqualityComparer.Instance));
        return sb.ToString();
    }

    static void Write(StringBuilder sb, object? value, string path, int depth, HashSet<object> seen)
    {
        if (value is null || value is string || value.GetType().IsPrimitive || value.GetType().IsEnum || value is decimal)
        {
            sb.Append(path).Append('=').Append(value?.ToString() ?? "null").Append('\n');
            return;
        }

        if (depth > 6 || (!value.GetType().IsValueType && !seen.Add(value)))
        {
            return;
        }

        if (value is IEnumerable items)
        {
            int i = 0;
            foreach (object? item in items)
            {
                Write(sb, item, $"{path}[{i++}]", depth + 1, seen);
            }

            sb.Append(path).Append(".count=").Append(i).Append('\n');
            return;
        }

        var type = value.GetType();
        foreach (var f in type.GetFields(BindingFlags.Public | BindingFlags.Instance))
        {
            Write(sb, f.GetValue(value), $"{path}.{f.Name}", depth + 1, seen);
        }

        foreach (var p in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (p.GetIndexParameters().Length > 0 || p.GetMethod is null)
            {
                continue;
            }

            object? v;
            try
            {
                v = p.GetValue(value);
            }
            catch (TargetInvocationException)
            {
                continue;
            }

            Write(sb, v, $"{path}.{p.Name}", depth + 1, seen);
        }
    }
}
