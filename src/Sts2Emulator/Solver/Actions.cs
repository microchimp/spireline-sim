namespace Sts2Emulator.Solver;

using Sts2Emulator.Core;
using Sts2Emulator.GeneratedData;

/// <summary>One decision: an engine action id (<c>CombatEngine.ValidActions</c>) with its target (-1 for none).</summary>
public readonly record struct Move(int Action, int Target)
{
    public override string ToString() => Target < 0 ? $"{Action}" : $"{Action}@{Target}";
}

/// <summary>The lossless prunings (FIGHT_SOLVER.md section 5), each one switchable for the equal-value harness.</summary>
[Flags]
public enum Pruning
{
    None = 0,

    /// <summary>Values memoised by the state key: commuting plays reach one node, and equal outcomes of a chance node are one child.</summary>
    Transposition = 1,

    /// <summary>Identical cards are one: one play per distinct (card value, target), one potion per distinct (potion, target).</summary>
    MultisetHand = 2,

    /// <summary>Enemies equal but for their slot take one target (<c>StateKey.EnemyClasses</c>).</summary>
    InterchangeableEnemies = 4,

    /// <summary>The table outlives a solve: the next decision of the fight looks its position up first.</summary>
    SubtreeReuse = 8,

    /// <summary>A draw from the unknown middle branches over its distinct values at count / size, not over each card.</summary>
    DrawByValue = 16,

    /// <summary>
    /// A run of plain draws from the unknown middle is one chance node over the sub-multisets drawn, hypergeometric
    /// (FIGHT_SOLVER.md section 17): off where the hand is kept in order or a draw observer is present. Needs <see cref="DrawByValue" />.
    /// </summary>
    MacroDraw = 32,

    /// <summary>
    /// The tight utility bound (FIGHT_SOLVER.md section 19): where no heal, max-HP change or potion gain is reachable in the
    /// combat (the scanned registry, <see cref="UtilityGains" />), a horizon leaf's and a wide roll's bound is L0 of the
    /// position's HP and potions, not L0's maximum, 1. Values are unchanged; bounds, and so exactness, are tighter.
    /// </summary>
    UtilityBound = 64,

    /// <summary>At a max node below the root, an exact move whose value reaches the node's bound closes the node: its other moves are not expanded.</summary>
    LethalClosure = 128,

    /// <summary>At a max node below the root, a move whose bound is at or below the best exact value found there is cut (not searched).</summary>
    BranchBound = 256,

    /// <summary>At a max node below the root, likely closers first: kills, other attacks, other plays, end turn, potions last. Node counts only.</summary>
    CloserFirst = 512,

    All = Transposition | MultisetHand | InterchangeableEnemies | SubtreeReuse | DrawByValue | MacroDraw | UtilityBound | LethalClosure | BranchBound | CloserFirst,

    /// <summary>Every pruning but the bounds round's (the engine round's <c>All</c>).</summary>
    NoBounds = All & ~(UtilityBound | LethalClosure | BranchBound | CloserFirst),
}

/// <summary>
/// The legal decisions of a position (FIGHT_SOLVER.md section 3): a card play per living, targetable enemy
/// for an aimed card, a potion likewise, end turn, or an open screen's candidates and its decline.
/// </summary>
public static class Actions
{
    /// <summary>The enemies an aimed play can take: alive and targetable (not reviving).</summary>
    public static List<int> Targets(CombatState s)
    {
        var t = new List<int>();
        for (int i = 0; i < s.Enemies.Count; i++)
        {
            var e = s.Enemies[i];
            if (e.Hp > 0 && EnemyAI.IsTargetable(e))
            {
                t.Add(i);
            }
        }

        return t;
    }

    public static bool CardAims(CombatState s, CardInstance c) =>
        Sts2Emulator.Core.Effects.CardEffects.TargetTypeOf(s, c, Cards.Get(Math.Abs(c.DefId))) == CardTarget.AnyEnemy;

    /// <summary>
    /// Every legal move in action order. With <see cref="Pruning.MultisetHand" /> a card whose key value
    /// equals an earlier hand card's (and a potion equal to an earlier slot's) is left out where the key
    /// holds that pile as a multiset; with <see cref="Pruning.InterchangeableEnemies" /> only the first
    /// enemy of each class is a target. A potion whose slot's bit is clear in <paramref name="potionMask" />
    /// is not used (the potion curve's subsets; -1 allows every slot).
    /// </summary>
    public static List<Move> Legal(CombatState s, Pruning prune, int potionMask = -1)
    {
        int[] valid = CombatEngine.ValidActions(s);
        var moves = new List<Move>(valid.Length + 4);
        if (s.PendingSelection is not null)
        {
            foreach (int a in valid)
            {
                moves.Add(new Move(a, -1));
            }

            return moves;
        }

        var targets = Targets(s);
        if ((prune & Pruning.InterchangeableEnemies) != 0 && targets.Count > 1)
        {
            int[] classes = StateKey.EnemyClasses(s);
            targets = [.. targets.Where(i => classes[i] == i || !targets.Contains(classes[i]))];
        }

        bool handSet = (prune & Pruning.MultisetHand) != 0 && !StateKey.HandOrdered(s);
        bool potionSet = (prune & Pruning.MultisetHand) != 0 && !StateKey.PotionOrdered(s);
        var seenCards = new List<byte[]>();
        var seenPotions = new HashSet<int>();
        int hand = s.Hand.Count;
        foreach (int a in valid)
        {
            if (a < hand)
            {
                var card = s.Hand[a];
                if (handSet)
                {
                    var v = DrawSteer.CardValue(s, card);
                    if (seenCards.Exists(x => x.AsSpan().SequenceEqual(v)))
                    {
                        continue;
                    }

                    seenCards.Add(v);
                }

                AddAimed(moves, a, CardAims(s, card), targets);
            }
            else if (a == hand)
            {
                moves.Add(new Move(a, -1));
            }
            else
            {
                int slot = a - hand - 1;
                int potion = s.PotionSlots[slot];
                if (slot < 31 && (potionMask & (1 << slot)) == 0)
                {
                    continue;
                }

                if (potionSet && !seenPotions.Add(potion))
                {
                    continue;
                }

                AddAimed(moves, a, Sts2Emulator.Core.Effects.PotionEffects.NeedsEnemyTarget(potion), targets);
            }
        }

        return moves;
    }

    /// <summary>
    /// An aimed action is one move per target and none when there is no target to aim at (<see cref="CombatEngine.ValidActions" />
    /// no longer lists such an action; it was once kept here as a move at target -1, a play the game refuses).
    /// </summary>
    static void AddAimed(List<Move> moves, int action, bool aims, List<int> targets)
    {
        if (!aims)
        {
            moves.Add(new Move(action, -1));
            return;
        }

        foreach (int t in targets)
        {
            moves.Add(new Move(action, t));
        }
    }

    /// <summary>Whether the move ends the player's turn (end turn with no screen open).</summary>
    public static bool EndsTurn(CombatState s, Move m) => s.PendingSelection is null && m.Action == s.Hand.Count;

    /// <summary>A potion's use (an action past end turn, no screen open).</summary>
    public static bool IsPotion(CombatState s, Move m) => s.PendingSelection is null && m.Action > s.Hand.Count;

    /// <summary>The change to L0's potion count a potion's use makes: its slot is emptied before its effect (<c>UsePotion</c>).</summary>
    public static int PotionsUsed(CombatState s, Move m)
    {
        int slot = m.Action - s.Hand.Count - 1;
        return slot >= 0 && slot < Math.Min(s.PotionSlots.Length, s.MaxPotionSlots) && s.PotionSlots[slot] != 0 ? -1 : 0;
    }
}
