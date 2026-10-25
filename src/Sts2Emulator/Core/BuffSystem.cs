namespace Sts2Emulator.Core;

public static class BuffSystem
{
    /// <summary>
    /// The powers whose game class is <c>PowerInstanceType.Instanced</c>: every application is
    /// an instance of its own, in <c>Creature._powers</c> order, instead of stacking.
    /// </summary>
    /// <remarks>
    /// <c>PowerCmd.FindExistingInstanceForStacking</c> returns null for the type, so
    /// <c>Apply</c> builds a fresh power with its own <c>Amount</c> and its own DynamicVars
    /// and appends it. The emulator's list is that list: one <see cref="BuffState" /> per
    /// instance, oldest first, <c>Magnitude</c> the Amount and <c>Data</c> the second number.
    /// Every hook the game runs over the instances runs oldest first, so a consumer visits
    /// them in list order (<see cref="InstanceIndex" />).
    ///
    /// Only the ids whose every reader walks the instances are listed. Moving an id here is
    /// what makes <c>Apply</c> append, so a reader that still calls <see cref="Get" /> (first
    /// instance only) must change in the same commit. <c>InstancedPerApplier</c> (Oblivion,
    /// Strangle) is not here: a lone player is the only applier, and one instance per applier
    /// is then the stack.
    ///
    /// The game's other Instanced powers are not listed. Covered, Flanking and Guarded come from
    /// multiplayer-only cards; MagicBomb is not modelled as a buff; Heist and Thievery are
    /// (<see cref="BuffId.Heist" />, <see cref="BuffId.Thievery" />) but with one instance by
    /// construction, as a lone player is the only one a Merc steals from; Swipe, Sandpit and
    /// Withering Presence are put on once per fight by the monster that owns them.
    ///
    /// The counters an instance keeps for itself -- Toric Toughness's block, how many cards an
    /// Automation or a Panache has counted, how much energy an Orbit has seen, how much Strength
    /// a Monologue has handed out -- are its <c>Data</c>, which counts UP from zero so that an
    /// instance built without it (a snapshot of the live game) starts where a new one does.
    /// </remarks>
    public static bool IsInstanced(BuffId id) =>
        id
            is BuffId.TheBombPower
                or BuffId.Knockdown
                or BuffId.TagTeam
                or BuffId.RollingBoulderPower
                or BuffId.Nightmare
                or BuffId.ToricToughness
                or BuffId.AutomationPower
                or BuffId.Orbit
                or BuffId.PanachePower
                or BuffId.Monologue;

    /// <summary>
    /// Index of the <paramref name="k" />-th (from 0, oldest first) instance of an
    /// <see cref="IsInstanced" /> power, or -1. Addressing an instance by its ordinal rather
    /// than its list index survives the list changing under a long effect: damage that kills
    /// can add or drop OTHER entries, but never reorders the instances of one id.
    /// </summary>
    public static int InstanceIndex(List<BuffState> buffs, BuffId id, int k)
    {
        for (int i = 0; i < buffs.Count; i++)
        {
            if (buffs[i].Id == id && k-- == 0)
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>How many instances of an <see cref="IsInstanced" /> power are on the list.</summary>
    public static int InstanceCount(List<BuffState> buffs, BuffId id)
    {
        int count = 0;
        for (int i = 0; i < buffs.Count; i++)
        {
            if (buffs[i].Id == id)
            {
                count++;
            }
        }

        return count;
    }

    /// <param name="data">The instance's second number; only an Instanced power keeps one.</param>
    public static void Apply(List<BuffState> buffs, BuffId id, int magnitude, int data = 0)
    {
        if (magnitude == 0)
        {
            return;
        }

        if (magnitude > 0 && IsDebuff(id))
        {
            if (TryConsumeArtifact(buffs))
            {
                return;
            }
        }

        // `FindExistingInstanceForStacking` finds nothing for an Instanced power. There is no
        // instance to adjust either, so a negative Apply is a caller still reading it as a stack.
        if (IsInstanced(id))
        {
            if (magnitude < 0)
            {
                throw new InvalidOperationException(
                    $"{id} is Instanced: change one instance through InstanceIndex"
                );
            }

            buffs.Add(new BuffState(id, magnitude, data));
            return;
        }

        int idx = IndexOf(buffs, id);
        if (idx >= 0)
        {
            int newVal = buffs[idx].Magnitude + magnitude;
            if (newVal == 0)
            {
                buffs.RemoveAt(idx);
            }
            else
            {
                buffs[idx] = buffs[idx] with { Magnitude = newVal };
            }
        }
        else
        {
            buffs.Add(new BuffState(id, magnitude));
        }
    }

    /// <summary>
    /// Index of <paramref name="id" /> in <paramref name="buffs" />, or -1.
    /// </summary>
    /// <remarks>
    /// A hand-written loop rather than <c>FindIndex(b =&gt; b.Id == id)</c>, because that
    /// lambda CAPTURES `id` — so it allocates a closure and a delegate on every call, and
    /// this is the most-called function in the emulator (242 call sites, several of them
    /// per point of damage). It was ~3KB of garbage per attacking enemy, which is most of
    /// what made the enemy phase the hottest path in a step. Buff lists are a handful of
    /// entries; the scan is cheaper than the allocation was.
    /// </remarks>
    private static int IndexOf(List<BuffState> buffs, BuffId id)
    {
        for (int i = 0; i < buffs.Count; i++)
        {
            if (buffs[i].Id == id)
            {
                return i;
            }
        }

        return -1;
    }

    public static int Get(List<BuffState> buffs, BuffId id)
    {
        int idx = IndexOf(buffs, id);
        return idx >= 0 ? buffs[idx].Magnitude : 0;
    }

    public static bool Has(List<BuffState> buffs, BuffId id) => Get(buffs, id) > 0;

    public static void Remove(List<BuffState> buffs, BuffId id)
    {
        for (int i = buffs.Count - 1; i >= 0; i--)
        {
            if (buffs[i].Id == id)
            {
                buffs.RemoveAt(i);
            }
        }
    }

    /// <summary>
    /// The powers the game declares <c>AllowNegative</c> on a <c>Counter</c> stack. A
    /// NEGATIVE application of one of these is a DEBUFF -- `PowerModel.GetTypeForAmount`
    /// says so outright -- which is what puts it in Artifact's way.
    /// </summary>
    /// <remarks>
    /// Strength, Dexterity, Focus, Shrink and Shriek are the whole list in
    /// `Models.Powers`. Shrink and Shriek are printed Debuffs either way; the first three
    /// are Buffs whose sign decides.
    /// </remarks>
    public static bool AllowsNegative(BuffId id) =>
        id is BuffId.Strength or BuffId.Dexterity or BuffId.Focus or BuffId.Shrink;

    /// <summary>
    /// Whether applying <paramref name="magnitude"/> of <paramref name="id"/> is a DEBUFF
    /// landing -- the question Artifact answers, and not the same as "is this a debuff".
    /// </summary>
    /// <remarks>
    /// Nine points of Strength off an enemy is a debuff; nine points ON is not, and both
    /// are `BuffId.Strength`. `Apply` only asks the positive half of this because most of
    /// its negative callers are BOOKKEEPING -- a temporary power handing back what it
    /// took, which the game does by removing the power rather than by applying a debuff,
    /// and which Artifact must not eat.
    /// </remarks>
    public static bool IsDebuffApplication(BuffId id, int magnitude) =>
        (magnitude > 0 && IsDebuff(id)) || (magnitude < 0 && AllowsNegative(id));

    public static bool TryConsumeArtifact(List<BuffState> buffs)
    {
        int artifact = Get(buffs, BuffId.Artifact);
        if (artifact <= 0)
        {
            return false;
        }

        int artifactIdx = IndexOf(buffs, BuffId.Artifact);
        if (artifact == 1)
        {
            buffs.RemoveAt(artifactIdx);
        }
        else
        {
            buffs[artifactIdx] = buffs[artifactIdx] with { Magnitude = artifact - 1 };
        }

        return true;
    }

    // Called at end of turn for the owning side (tick debuffs down by 1).
    /// <summary>The duration debuffs, as a snapshot to compare a later stack against.</summary>
    public static List<BuffState> DurationDebuffSnapshot(List<BuffState> buffs) =>
        [.. buffs.Where(buff => IsDurationDebuff(buff.Id))];

    private static bool IsDurationDebuff(BuffId id) =>
        id is BuffId.Vulnerable or BuffId.Weak or BuffId.Frail or BuffId.Shrink;

    /// <param name="atRoundStart">
    /// What the owner held when the round began, for the player's one-tick grace: a stack
    /// bigger now than it was then was applied during this round, and the game's
    /// SkipNextDurationTick means this tick passes it over. Omit for enemies, which get
    /// no grace.
    /// </param>
    public static void TickEndOfTurn(List<BuffState> buffs, List<BuffState>? atRoundStart = null)
    {
        for (int i = buffs.Count - 1; i >= 0; i--)
        {
            var b = buffs[i];
            switch (b.Id)
            {
                // SlumberPower.AfterSideTurnEnd, which ticks for its OWNER's side only --
                // the beetle is an enemy, and this runs for the enemies. Its other
                // decrement, on unblocked damage, is in DealDamageToEnemy.
                // IntangiblePower.AfterSideTurnEnd decrements at the end of the ENEMY
                // side turn whoever owns it, which is this moment. Listed apart from the
                // duration DEBUFFS below because it is a Buff: PowerCmd.Apply's
                // skip-a-tick grace is only given to a debuff landing on a player-side
                // creature, so an Intangible the player gains this turn ticks tonight.
                // TaintedPower.AfterSideTurnEnd REMOVES itself outright rather than
                // decrementing, so a round's worth of Skills is paid for once.
                // `ReboundPower.AfterSideTurnEnd` removes itself outright too, so an
                // unspent Rebound does not carry into the next turn -- the card is "the
                // next card THIS turn", not the next card ever.
                case BuffId.Tainted:
                case BuffId.Rebound:
                    buffs.RemoveAt(i);
                    break;

                case BuffId.Intangible:
                    buffs[i] = b with { Magnitude = b.Magnitude - 1 };
                    if (buffs[i].Magnitude <= 0)
                    {
                        buffs.RemoveAt(i);
                    }

                    break;

                case BuffId.Slumber:
                case BuffId.Vulnerable:
                case BuffId.Weak:
                case BuffId.Frail:
                case BuffId.Shrink:
                    if (b.Magnitude < 0)
                    {
                        break; // negative = permanent (e.g. ShrinkerBeetle Shrink)
                    }

                    if (atRoundStart != null && WasAppliedThisRound(b, atRoundStart))
                    {
                        break;
                    }

                    buffs[i] = b with { Magnitude = b.Magnitude - 1 };
                    if (buffs[i].Magnitude <= 0)
                    {
                        buffs.RemoveAt(i);
                    }

                    break;
            }
        }
    }

    /// <summary>
    /// Whether this debuff is new to the player this round, and so owed the one-tick
    /// grace. It is the POWER that carries `SkipNextDurationTick`, not the application:
    /// `PowerCmd.Apply` sets the flag on the model it creates, so a debuff that lands on
    /// a stack the player already had leaves the existing power's flag alone and that
    /// power ticks as usual.
    ///
    /// Live, at A8, the Two-Tailed Rats screech nearly every turn and the player's Frail
    /// reads 1, 1, 0, 1 — each new point cancelled by the same round's tick. Treating any
    /// increase as a skip made it climb instead, which is a point of block a turn.
    /// </summary>
    private static bool WasAppliedThisRound(BuffState buff, List<BuffState> atRoundStart)
    {
        return !atRoundStart.Any(other => other.Id == buff.Id);
    }

    /// <summary>
    /// game: StrengthPower.ModifyDamageAdditive
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static int Strength_ModifyDamageAdditive(List<BuffState> ownerBuffs) =>
        Get(ownerBuffs, BuffId.Strength);

    /// <summary>
    /// game: VigorPower.ModifyDamageAdditive
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static int Vigor_ModifyDamageAdditive(List<BuffState> ownerBuffs) =>
        Get(ownerBuffs, BuffId.Vigor);

    /// <summary>
    /// The additive damage terms IncomingDamage adds on the ATTACKER's side (Strength, then Vigor; Tainted is the
    /// defender's and no card attack's target carries it). CardEffects.DmgFrom hands this to the card's multiplicative
    /// fold so Pen Nib, Vitruvian Minion and Gigantification land after these terms, as Hook.ModifyDamage orders them.
    /// Keep it equal to the attacker-side additions at the top of IncomingDamage.
    /// </summary>
    internal static int AttackerDamageAdditive(List<BuffState> attackerBuffs) =>
        Strength_ModifyDamageAdditive(attackerBuffs) + Vigor_ModifyDamageAdditive(attackerBuffs);

    /// <summary>
    /// game: TaintedPower.ModifyDamageAdditive
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static int Tainted_ModifyDamageAdditive(List<BuffState> targetBuffs) =>
        Get(targetBuffs, BuffId.Tainted);

    /// <summary>
    /// game: WeakPower.ModifyDamageMultiplicative
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static float Weak_ModifyDamageMultiplicative(
        List<BuffState> attackerBuffs,
        float weakDelta
    )
    {
        // `DebilitatePower.ModifyWeakMultiplier` is `amount - (1 - amount)` for its
        // OWNER's attacks, so a debilitated attacker's Weak lands at 0.5 instead of
        // 0.75. The amount is a duration and does not scale the doubling.
        return (Get(attackerBuffs, BuffId.Debilitate) > 0 ? 0.5f : 0.75f) + weakDelta;
    }

    /// <summary>
    /// game: ShrinkPower.ModifyDamageMultiplicative
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static float Shrink_ModifyDamageMultiplicative() => 0.70f;

    /// <summary>
    /// game: FlutterPower.ModifyDamageMultiplicative
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static float Flutter_ModifyDamageMultiplicative() => 0.5f;

    /// <summary>
    /// game: TankPower.ModifyDamageMultiplicative
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static float Tank_ModifyDamageMultiplicative() => 2f;

    /// <summary>
    /// game: VulnerablePower.ModifyDamageMultiplicative
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static float Vulnerable_ModifyDamageMultiplicative(
        List<BuffState> attackerBuffs,
        List<BuffState> defenderBuffs,
        float vulnerableDelta
    )
    {
        float mult = 1.5f + Get(attackerBuffs, BuffId.CrueltyPower) / 100f + vulnerableDelta;
        // `DebilitatePower.ModifyVulnerableMultiplier` is `amount + (amount - 1)` when
        // the target is its owner, which doubles the BONUS rather than the multiplier:
        // 1.5 becomes 2.0, and a Cruelty-raised 1.75 becomes 2.5.
        if (Get(defenderBuffs, BuffId.Debilitate) > 0)
        {
            mult += mult - 1f;
        }

        return mult;
    }

    /// <summary>
    /// game: SurroundedPower.ModifyDamageMultiplicative
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// The multiplier is 1 when the attack is not from behind: the product is unchanged.
    /// </summary>
    internal static float Surrounded_ModifyDamageMultiplicative(
        List<BuffState> attackerBuffs,
        List<BuffState> defenderBuffs
    )
    {
        int facing = Get(defenderBuffs, BuffId.Surrounded);
        bool fromBehind =
            facing == Run.RunConstants.FacingRight
                ? Get(attackerBuffs, BuffId.BackAttackLeft) > 0
                : facing == Run.RunConstants.FacingLeft
                    && Get(attackerBuffs, BuffId.BackAttackRight) > 0;
        return fromBehind ? 1.5f : 1f;
    }

    /// <summary>
    /// game: TrackingPower.ModifyDamageMultiplicative
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static int Tracking_ModifyDamageMultiplicative(int amount) => amount;

    /// <summary>
    /// game: DoubleDamagePower.ModifyDamageMultiplicative
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static int DoubleDamage_ModifyDamageMultiplicative() => 2;

    /// <summary>
    /// game: SoarPower.ModifyDamageMultiplicative
    /// Extracted from BuffSystem.IncomingDamage; the caller keeps the condition.
    /// </summary>
    internal static float Soar_ModifyDamageMultiplicative() => 0.5f;

    /// <param name="weakDelta">
    /// `Hook.ModifyWeakMultiplier`, which Paper Krane answers with `-0.15` when its owner
    /// is the TARGET -- a Weak attacker hits them for 0.60 rather than 0.75.
    /// </param>
    /// <param name="vulnerableDelta">
    /// `Hook.ModifyVulnerableMultiplier`, which Paper Phrog answers with `+0.25` when the
    /// target is NOT its owner -- a Vulnerable enemy takes 1.75 rather than 1.5.
    /// </param>
    /// <remarks>
    /// The two deltas arrive as parameters rather than being read from relics here,
    /// because this function deliberately cannot see a `CombatState` -- it is called with
    /// two buff lists and nothing else, including from `Intent.AnnouncedDamage` where the
    /// caller may not have a combat at all. Each caller that CAN see the relics passes
    /// them, which is also what keeps the announced intent and the landed damage in step.
    /// </remarks>
    public static int IncomingDamage(
        int baseDamage,
        List<BuffState> attackerBuffs,
        List<BuffState> defenderBuffs,
        float cardMultiplier = 1f,
        float weakDelta = 0f,
        float vulnerableDelta = 0f
    )
    {
        // Everything in this function is the POWERED-ATTACK path, which is what lets the
        // two hooks below live here at all. `ValueProp.IsPoweredAttack` is `Move &&
        // !Unpowered` -- attack damage from Attack cards and from enemy creatures
        // attacking -- and the two callers are exactly those: EnemyAI.DealAttackDamage and
        // CardEffects.DealDamageToEnemy. Relic, potion, thorns and poison damage all go
        // through the `Unpowered` helpers instead and never arrive here.
        float dmg = baseDamage;
        dmg += Strength_ModifyDamageAdditive(attackerBuffs);
        dmg += Vigor_ModifyDamageAdditive(attackerBuffs);
        // TaintedPower.ModifyDamageAdditive, which the Infested Prism's Vital Spark stamps
        // on the player for every Skill they play.
        dmg += Tainted_ModifyDamageAdditive(defenderBuffs);
        if (Get(attackerBuffs, BuffId.Weak) > 0)
        {
            dmg *= Weak_ModifyDamageMultiplicative(attackerBuffs, weakDelta);
        }

        if (Get(attackerBuffs, BuffId.Shrink) != 0)
        {
            dmg *= Shrink_ModifyDamageMultiplicative(); // negative = permanent
        }

        // `FlutterPower.ModifyDamageMultiplicative` returns DamageDecrease/100 = 0.5 for
        // its owner, and only for a POWERED attack -- which is the whole of this function,
        // so no extra guard is needed here. Thorns, poison and relic damage go through the
        // Unpowered helpers and never reach it, exactly as the game's prop check intends.
        if (Get(defenderBuffs, BuffId.Flutter) > 0)
        {
            dmg *= Flutter_ModifyDamageMultiplicative();
        }

        // TankPower.ModifyDamageMultiplicative: 2 for a powered attack whose TARGET is its
        // owner. Like Flutter above, the target check is the defender's buff list.
        if (Get(defenderBuffs, BuffId.Tank) > 0)
        {
            dmg *= Tank_ModifyDamageMultiplicative();
        }

        if (Get(defenderBuffs, BuffId.Vulnerable) > 0)
        {
            dmg *= Vulnerable_ModifyDamageMultiplicative(
                attackerBuffs,
                defenderBuffs,
                vulnerableDelta
            );
        }

        // SurroundedPower.ModifyDamageMultiplicative, the Kaiser Crab's: an attack from
        // the half at the player's BACK lands at 1.5x. It belongs here rather than at the
        // point of damage because the game's readout shows the modified number --
        // AttackIntent.GetSingleDamage runs the move through Hook.ModifyDamage first --
        // and a live capture confirms it: the Crusher opens announcing 18 for a base 12.
        dmg *= Surrounded_ModifyDamageMultiplicative(attackerBuffs, defenderBuffs);

        // TrackingPower.ModifyDamageMultiplicative: a powered CARD attack against a target
        // that has Weak is multiplied by the power's own amount, so Tracking 2 is double
        // damage. It reads the DEFENDER's Weak and the ATTACKER's Tracking.
        int tracking = Get(attackerBuffs, BuffId.Tracking);
        if (tracking > 0 && Get(defenderBuffs, BuffId.Weak) > 0)
        {
            dmg *= Tracking_ModifyDamageMultiplicative(tracking);
        }

        // DoubleDamagePower.ModifyDamageMultiplicative returns a flat `2m` for a powered
        // CARD attack by its owner. The amount is a STACK COUNT and not a multiplier --
        // two stacks are still double, and buy a second turn of it rather than quadruple
        // damage.
        //
        // The game also requires `cardSource != null`, which this function cannot see.
        // Player damage that is not from a card is Unpowered in the emulator and does not
        // reach here, so the two agree for every source that exists today -- the same
        // standing caveat Tracking above carries.
        if (Get(attackerBuffs, BuffId.DoubleDamage) > 0)
        {
            dmg *= DoubleDamage_ModifyDamageMultiplicative();
        }

        // SoarPower.ModifyDamageMultiplicative: a flying Owl Magistrate takes half from a
        // POWERED attack, which is the only kind that reaches this function.
        if (Get(defenderBuffs, BuffId.Soar) > 0)
        {
            dmg *= Soar_ModifyDamageMultiplicative();
        }

        // A multiplicative the caller worked out because it depends on the CARD -- today
        // only `LethalityPower`, which pays out on the first Attack card of the turn.
        // Folded in here rather than applied to the result, so it lands before the single
        // `(int)` at the end, which is where the game's own multiplicatives land.
        dmg *= cardMultiplier;

        return CapIncomingDamage(Math.Max(0, (int)dmg), defenderBuffs);
    }

    /// <summary>
    /// <c>IntangiblePower.ModifyDamageCap</c>: damage aimed at its owner is capped at 1.
    /// </summary>
    /// <remarks>
    /// The cap runs inside <c>Hook.ModifyDamage</c> under the <c>Cap</c> flag, and
    /// <c>ModifyDamageHookType.All</c> — which is what almost every back-end call passes —
    /// includes it. So this belongs with the additive and multiplicative modifiers rather
    /// than at the point of damage, and it reaches the READOUT as well as the blow:
    /// <c>AttackIntent.GetSingleDamage</c> runs the move through the same hook, so an
    /// intangible player is told the enemy will hit them for 1, not for thirty.
    ///
    /// Applied per HIT, which is where the game applies it: a two-hit attack against an
    /// intangible creature announces 2 and lands two ones.
    /// </remarks>
    public static int CapIncomingDamage(int damage, List<BuffState> defenderBuffs) =>
        Get(defenderBuffs, BuffId.Intangible) > 0 ? Math.Min(damage, 1) : damage;

    /// <summary>
    /// <c>IntangiblePower.ModifyHpLostAfterOsty</c>: any HP loss of 1 or more becomes 1.
    /// </summary>
    /// <remarks>
    /// The second half of Intangible, and the reason the power carries two hooks: the cap
    /// above governs the damage NUMBER — what block absorbs, what a preview shows — and
    /// this one is the backstop on HP itself. It is what covers HP lost by a route that is
    /// not an attack at all. Poison happens to be capped by the first hook rather than this
    /// one, since <c>PoisonPower</c> runs its own damage through
    /// <c>Hook.ModifyDamage(..., All, ...)</c>, but either way the answer is 1.
    /// </remarks>
    public static int CapHpLoss(int hpLoss, List<BuffState> defenderBuffs) =>
        Get(defenderBuffs, BuffId.Intangible) > 0 ? Math.Min(hpLoss, 1) : hpLoss;

    public static int IncomingBlock(int baseBlock, List<BuffState> buffs, bool isDefend = false) =>
        (int)IncomingBlockExact(baseBlock, buffs, isDefend);

    /// <summary>
    /// The same pipeline without the final truncation. `Hook.ModifyBlock` keeps a
    /// <c>decimal</c> through EVERY additive and multiplicative contributor and only
    /// `Creature.GainBlockInternal` casts to int, so a multiplier that comes after Frail
    /// (Unmovable, Vambrace, Paels Legion, Shadowmeld) works on 3.75, not on 3.
    /// </summary>
    public static decimal IncomingBlockExact(int baseBlock, List<BuffState> buffs, bool isDefend = false)
    {
        decimal blk = baseBlock;
        if (Get(buffs, BuffId.NoBlock) > 0)
        {
            return 0m;
        }

        // `Hook.ModifyBlock` runs every ADDITIVE contribution first and only then the
        // multiplicative ones. Fasten is `ModifyBlockAdditive` like Dexterity, so it has to
        // land before Frail rather than after it: added afterwards, a Frail Defend came out
        // as (base + Dex) * 0.75 + Fasten where the game pays (base + Dex + Fasten) * 0.75.
        blk += Get(buffs, BuffId.Dexterity);

        // `FastenPower.ModifyBlockAdditive` refuses a card that is not tagged Defend, which
        // is why the caller has to say which kind of block this is.
        if (isDefend)
        {
            blk += Get(buffs, BuffId.FastenPower);
        }

        // `FrailPower.ModifyBlockMultiplicative`, the only multiplicative one here.
        if (Get(buffs, BuffId.Frail) > 0)
        {
            blk *= 0.75m;
        }

        return Math.Max(0m, blk);
    }

    /// <summary>
    /// Whether a BuffId is a DEBUFF power: what Artifact stops, and what a dying Illusion
    /// sheds. `ArtifactPower.TryModifyPowerAmountReceived` asks the power's
    /// `GetTypeForAmount`, not a list, so this is read off the generated `IsBuff` flag for
    /// every BuffId that models a power (`FooPower` by name, like the converter's map).
    /// </summary>
    /// <remarks>
    /// It was a hand-kept list of twelve, and Hex, Dampen, No Draw, Sloth and twenty-five
    /// more went straight past an Artifact -- found by `scripts/audit_effects.py powers`.
    /// Two kinds stay out on purpose. The TEMPORARY powers hand their effect back by being
    /// removed, and the emulator keeps that as bookkeeping its call sites guard (a
    /// temporary Strength loss consumes Artifact where it is applied); and an id with no
    /// power behind it is an emulator counter, which is nothing's debuff.
    /// </remarks>
    internal static bool IsDebuff(BuffId id) => (int)id < DebuffIds.Length && DebuffIds[(int)id];

    private static readonly bool[] DebuffIds = BuildDebuffIds();

    private static bool[] BuildDebuffIds()
    {
        var ids = Enum.GetValues<BuffId>();
        var debuffs = new bool[ids.Max(id => (int)id) + 1];
        foreach (var id in ids)
        {
            if (
                id
                is BuffId.TemporaryStrength
                    or BuffId.TemporaryDexterity
                    or BuffId.TemporaryFocus
                    or BuffId.Illusion
            )
            {
                continue;
            }

            int? power =
                GeneratedData.Powers.FindId($"{id}Power") ?? GeneratedData.Powers.FindId($"{id}");
            debuffs[(int)id] = power is int found && !GeneratedData.Powers.Get(found).IsBuff;
        }

        return debuffs;
    }
}
