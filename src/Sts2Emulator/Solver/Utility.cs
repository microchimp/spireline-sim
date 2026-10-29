namespace Sts2Emulator.Solver;

using System.Numerics;
using Sts2Emulator.Core;

/// <summary>
/// An exact rational number. Every probability the rules give is one (1 / width for a uniform roll,
/// weight / sum for a weighted branch, count / size for a draw), and so is the combat score L0, so
/// an expectimax value is one too: the solver computes it exactly, and two solvers that prune
/// differently agree to the last bit or not at all (the equal-value harness, FIGHT_SOLVER.md
/// section 6). <see cref="ToDouble" /> is for reporting.
/// </summary>
public readonly struct Rational : IComparable<Rational>, IEquatable<Rational>
{
    public BigInteger Num { get; }

    public BigInteger Den { get; }

    public static readonly Rational Zero = new(0, 1, true);

    public static readonly Rational One = new(1, 1, true);

    private Rational(BigInteger num, BigInteger den, bool normal)
    {
        Num = num;
        Den = den;
    }

    public Rational(BigInteger num, BigInteger den)
    {
        if (den.IsZero)
        {
            throw new DivideByZeroException();
        }

        if (den.Sign < 0)
        {
            num = -num;
            den = -den;
        }

        var g = BigInteger.GreatestCommonDivisor(num, den);
        if (!g.IsOne && !g.IsZero)
        {
            num /= g;
            den /= g;
        }

        Num = num;
        Den = num.IsZero ? BigInteger.One : den;
    }

    /// <summary>A double's exact value (a double is a dyadic rational).</summary>
    public static Rational FromDouble(double d)
    {
        if (double.IsNaN(d) || double.IsInfinity(d))
        {
            throw new ArgumentOutOfRangeException(nameof(d), d, "not a finite number");
        }

        long bits = BitConverter.DoubleToInt64Bits(d);
        bool negative = bits < 0;
        int exponent = (int)((bits >> 52) & 0x7FF);
        long mantissa = bits & 0xFFFFFFFFFFFFFL;
        if (exponent == 0)
        {
            exponent++;
        }
        else
        {
            mantissa |= 1L << 52;
        }

        exponent -= 1075;
        BigInteger num = mantissa;
        BigInteger den = BigInteger.One;
        if (exponent > 0)
        {
            num <<= exponent;
        }
        else
        {
            den <<= -exponent;
        }

        return new Rational(negative ? -num : num, den);
    }

    public static Rational operator +(Rational a, Rational b)
    {
        int prof = SolverProf.Enter(Phase.Rational);
        var r = a.Den == b.Den ? new Rational(a.Num + b.Num, a.Den) : new Rational(a.Num * b.Den + b.Num * a.Den, a.Den * b.Den);
        SolverProf.Exit(prof);
        return r;
    }

    public static Rational operator *(Rational a, Rational b)
    {
        int prof = SolverProf.Enter(Phase.Rational);
        var r = new Rational(a.Num * b.Num, a.Den * b.Den);
        SolverProf.Exit(prof);
        return r;
    }

    public static Rational operator /(Rational a, Rational b)
    {
        int prof = SolverProf.Enter(Phase.Rational);
        var r = new Rational(a.Num * b.Den, a.Den * b.Num);
        SolverProf.Exit(prof);
        return r;
    }

    public static bool operator <(Rational a, Rational b) => a.CompareTo(b) < 0;

    public static bool operator >(Rational a, Rational b) => a.CompareTo(b) > 0;

    public static bool operator <=(Rational a, Rational b) => a.CompareTo(b) <= 0;

    public static bool operator >=(Rational a, Rational b) => a.CompareTo(b) >= 0;

    public static bool operator ==(Rational a, Rational b) => a.Equals(b);

    public static bool operator !=(Rational a, Rational b) => !a.Equals(b);

    public int CompareTo(Rational other)
    {
        int prof = SolverProf.Enter(Phase.Rational);
        int c = (Num * other.Den).CompareTo(other.Num * Den);
        SolverProf.Exit(prof);
        return c;
    }

    public bool Equals(Rational other) => Num == other.Num && Den == other.Den;

    public override bool Equals(object? obj) => obj is Rational r && Equals(r);

    public override int GetHashCode() => HashCode.Combine(Num, Den);

    public double ToDouble() => (double)Num / (double)Den is var d && !double.IsNaN(d) ? d : Math.Exp(BigInteger.Log(Num) - BigInteger.Log(Den));

    public override string ToString() => Den.IsOne ? Num.ToString() : $"{Num}/{Den}";
}

/// <summary>
/// The solve's utility (FIGHT_SOLVER.md sections 10 and 20). <see cref="L0" /> is the value scale of labels, regret and
/// grades. <see cref="Lethal" /> and <see cref="Survival" /> are the S5 root check's two questions, each a utility whose
/// horizon leaf has a known value, so a solve at a fixed horizon is exact for it: a win scores 1, a death 0, and a horizon
/// leaf 0 for <see cref="Lethal" /> (the chance to win before the horizon's last end turn) and 1 for
/// <see cref="Survival" /> (the chance to be alive, or to have won, at the first decision past the horizon's last turn
/// with no screen pending). 1 and 2 are kept for SearchHp and L0Priced (S5).
/// </summary>
public enum UtilityKind
{
    L0 = 0,
    Lethal = 3,
    Survival = 4,
}

/// <summary>
/// The value scale (FIGHT_SOLVER.md section 10): D4's combat score L0, 0 if dead, else
/// 0.5 + 0.5 * min(1, (HP + 10 * potions) / max HP), maximised in expectation. It scores a fight's end
/// and, as the placeholder until the network comes in S5, a horizon leaf (the position as it stands).
/// </summary>
public static class Utility
{
    /// <summary>The value of a dead player.</summary>
    public static readonly Rational Loss = Rational.Zero;

    /// <summary>The greatest value L0 can take: a sound bound for any position cut at the horizon.</summary>
    public static readonly Rational Max = Rational.One;

    public static int Potions(CombatState s)
    {
        int n = 0;
        int slots = Math.Min(s.PotionSlots.Length, s.MaxPotionSlots);
        for (int i = 0; i < slots; i++)
        {
            if (s.PotionSlots[i] != 0)
            {
                n++;
            }
        }

        return n;
    }

    /// <summary>L0 of a living player as the position stands.</summary>
    public static Rational L0(CombatState s)
    {
        if (s.PlayerHp <= 0)
        {
            return Loss;
        }

        int max = Math.Max(1, s.PlayerMaxHp);
        long score = Math.Min((long)s.PlayerHp + 10L * Potions(s), max);
        return new Rational(max + score, 2L * max);
    }

    /// <summary>A fight's end: the loss value, or L0 of the winner.</summary>
    public static Rational Terminal(CombatState s, bool won) => won ? L0(s) : Loss;

    /// <summary>A fight's end under <paramref name="u" />: L0's terminal score, or 1 for a win and 0 for a death.</summary>
    public static Rational Terminal(UtilityKind u, CombatState s, bool won) => u == UtilityKind.L0 ? Terminal(s, won) : won ? Rational.One : Loss;

    /// <summary>A horizon leaf's value under <paramref name="u" />: L0 of the position, 0 (lethal) or 1 (survival).</summary>
    public static Rational Leaf(UtilityKind u, CombatState s) => u switch
    {
        UtilityKind.Lethal => Rational.Zero,
        UtilityKind.Survival => Rational.One,
        _ => L0(s),
    };

    /// <summary>
    /// Whether a decision node is the survival utility's horizon leaf: a living player's decision with no screen pending,
    /// on a turn past the horizon's last (<paramref name="remaining" /> at or below 0). Ending the turn there is a horizon
    /// leaf of value 1, the utility's maximum, so the node's value is 1 whatever its other moves.
    /// </summary>
    public static bool SurvivalLeaf(UtilityKind u, CombatState s, int remaining) =>
        u == UtilityKind.Survival && remaining <= 0 && s.PendingSelection is null && s.PlayerHp > 0;
}
