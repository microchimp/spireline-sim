using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;

namespace Sts2Emulator.Tests.Invariants;

/// <summary>One action and the enemy it is aimed at (-1: the engine's default).</summary>
public readonly record struct Move(int Action, int Target)
{
    public override string ToString() => Target >= 0 ? $"{Action}->{Target}" : $"{Action}";
}

/// <summary>A fight in progress: the combat and the stream its steps draw from.</summary>
public class Combat
{
    public Combat(CombatState state, CountingRandom rng)
    {
        State = state;
        Rng = rng;
    }

    public CombatState State { get; }

    public CountingRandom Rng { get; }

    public bool Over { get; protected set; }

    public bool Won { get; protected set; }

    /// <summary>Everything the next step can change, as one object for <see cref="DeepState"/>.</summary>
    public (CombatState State, CountingRandom Rng) Whole => (State, Rng);

    /// <summary>A copy that shares nothing with this one, at the same position.</summary>
    public virtual Combat Clone() => new(State.Clone(), Rng.Clone()) { Over = Over, Won = Won };

    public virtual StepResult Step(Move m)
    {
        var r = m.Target >= 0
            ? CombatEngine.Step(State, m.Action, Rng, m.Target)
            : CombatEngine.Step(State, m.Action, Rng);
        Over |= r.Terminal;
        Won |= r.Terminal && r.PlayerWon;
        return r;
    }

    /// <summary>The name of the card an action plays, "END_TURN", "POTION" or "SELECT".</summary>
    public string Describe(Move m)
    {
        if (State.PendingSelection is not null)
        {
            return "SELECT";
        }

        if (m.Action < State.Hand.Count)
        {
            return Cards.Get(State.Hand[m.Action].DefId).Entry;
        }

        return m.Action == State.Hand.Count ? "END_TURN" : "POTION";
    }
}

/// <summary>Seeded uniform play over the legal actions: the chooser every relation shares.</summary>
public static class RandomPlay
{
    /// <summary>One legal action chosen by <paramref name="chooser"/>, or null when there is none.</summary>
    public static Move? Pick(CombatState s, Random chooser, bool potions = false)
    {
        var valid = CombatEngine.ValidActions(s).AsEnumerable();
        if (!potions && s.PendingSelection is null)
        {
            valid = valid.Where(a => a <= s.Hand.Count);
        }

        int[] actions = valid.ToArray();
        if (actions.Length == 0)
        {
            return null;
        }

        int action = actions[chooser.Next(actions.Length)];
        return new Move(action, TargetFor(s, action, chooser));
    }

    /// <summary>A living enemy at random for a card that is aimed; -1 for everything else.</summary>
    public static int TargetFor(CombatState s, int action, Random chooser)
    {
        if (s.PendingSelection is not null || action >= s.Hand.Count)
        {
            return -1;
        }

        if (Cards.Get(s.Hand[action].DefId).Target != CardTarget.AnyEnemy)
        {
            return -1;
        }

        var alive = Enumerable.Range(0, s.Enemies.Count).Where(i => s.Enemies[i].Hp > 0).ToArray();
        return alive.Length == 0 ? -1 : alive[chooser.Next(alive.Length)];
    }

    /// <summary>Plays up to <paramref name="steps"/> chosen actions; returns the moves made.</summary>
    public static List<Move> Play(Combat c, Random chooser, int steps)
    {
        var made = new List<Move>();
        while (made.Count < steps && !c.Over && Pick(c.State, chooser) is { } m)
        {
            c.Step(m);
            made.Add(m);
        }

        return made;
    }
}
