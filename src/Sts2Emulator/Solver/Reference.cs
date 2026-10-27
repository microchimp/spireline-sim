namespace Sts2Emulator.Solver;

using Sts2Emulator.Core;

/// <summary>
/// The reference solver (FIGHT_SOLVER.md section 6): plain tree expectimax over full labelled states for small fights.
/// No table, no pruning: every hand index and potion slot is its own move, every targetable enemy its own target, a
/// draw from the unknown middle branches over each card at 1 / size, and equal outcomes are separate children. It
/// shares only the engine step, the roll odds and the utility with <see cref="Expectimax" />, so the equal-value
/// harness holds each pruning to it, to the last bit (both compute exact rationals).
/// </summary>
public static class Reference
{
    public sealed record Result(Rational Value, bool Exact, IReadOnlyList<MoveValue> Moves, long Nodes, long Steps, string? Refused);

    public static Result Solve(CombatState state, CountingRandom rng, int horizon, long nodeCap = 2_000_000, int widthCap = 32)
    {
        var c = new SolveCounters();
        c.Check = () =>
        {
            if (c.Nodes > nodeCap || c.Steps > nodeCap * 4)
            {
                throw new BudgetExceeded($"reference cap {nodeCap}");
            }
        };
        try
        {
            StateKey.Encode(state, KeyMode.Expected);
            var moves = new List<MoveValue>();
            var values = new List<NodeValue>();
            foreach (var m in Actions.Legal(state, Pruning.None))
            {
                var v = MoveValueOf(state, rng, m, horizon, c, widthCap);
                moves.Add(new MoveValue(m, v.Value, v.Exact));
                values.Add(v);
            }

            var b = NodeValue.Max(values, state);
            return new Result(b.Value, b.Exact, moves, c.Nodes, c.Steps, null);
        }
        catch (SolveRefused e)
        {
            return new Result(Rational.Zero, false, [], c.Nodes, c.Steps, e.Reason);
        }
        catch (StateKeyRefusedException e)
        {
            return new Result(Rational.Zero, false, [], c.Nodes, c.Steps, e.Reason);
        }
        catch (BudgetExceeded e)
        {
            return new Result(Rational.Zero, false, [], c.Nodes, c.Steps, "budget: " + e.What);
        }
    }

    static NodeValue Node(CombatState s, CountingRandom rng, int remaining, SolveCounters c, int widthCap)
    {
        c.Nodes++;
        c.Check?.Invoke();

        // The key is not used, but a position the key refuses (a busy kernel queue) is refused here too.
        StateKey.Encode(s, KeyMode.Expected);
        var values = new List<NodeValue>();
        foreach (var m in Actions.Legal(s, Pruning.None))
        {
            values.Add(MoveValueOf(s, rng, m, remaining, c, widthCap));
        }

        return NodeValue.Max(values, s);
    }

    static NodeValue MoveValueOf(CombatState s, CountingRandom rng, Move m, int remaining, SolveCounters c, int widthCap)
    {
        if (Actions.EndsTurn(s, m) && remaining <= 0)
        {
            return NodeValue.Horizon(s);
        }

        var value = Rational.Zero;
        var mass = Rational.Zero;
        var upper = Rational.Zero;
        bool exact = true;
        foreach (var o in Successors.Expand(s, rng, m, byCard: true, widthCap, c))
        {
            var v = o.Result.Terminal ? NodeValue.End(o.State, o.Result.PlayerWon) : Node(o.State, o.Rng, remaining - (o.State.Turn - s.Turn), c, widthCap);
            value += o.P * v.Value;
            mass += o.P * v.Mass;
            upper += o.P * v.Upper;
            exact &= v.Exact;
        }

        return new NodeValue(value, exact, mass, exact ? value : upper);
    }
}
