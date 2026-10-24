// game: CreatureCmd, the game-shaped damage command over the old state (docs/rebuild/MIGRATION_F08.md section 3 items
// 2-3). Dispatch is only through Hooks.*; HP and block are read and written only through OldCreatures (OldState.cs).
// Rounds R2 (Damage) and R3a (the kill, CreatureCmd.Kill.cs): written and tested, called from no old site yet (R4 reroutes the callers). Game code is cited by class and
// method only.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

public static partial class CreatureCmd
{
    // construct: game CreatureCmd.Damage
    //   foreach        Hook.ModifyDamage
    //   foreach        Hook.AfterModifyingDamageAmount
    //   foreach        Hook.BeforeDamageReceived
    //   foreach        Hook.ModifyHpLost
    //   foreach        Hook.AfterModifyingHpLostBeforeOsty
    //   foreach/cond   Hook.ModifyUnblockedDamageTarget
    //   foreach        Hook.ModifyHpLost
    //   foreach        Hook.AfterModifyingHpLostAfterOsty
    //   foreach/else   Hook.ModifyHpLost
    //   foreach/else   Hook.AfterModifyingHpLostAfterOsty
    //   foreach/if     Hook.AfterBlockBroken
    //   foreach/if     Hook.AfterCurrentHpChanged
    //   foreach/if     Hook.AfterDamageGiven
    //   foreach/if     Hook.AfterDamageReceived
    //   -              Kill
    // end construct
    /// <summary>
    /// game: CreatureCmd.Damage, the list overload every other reaches. A dead dealer: one empty result per target and
    /// nothing else. Per target in list order, skipping the dead: the ModifyDamage fold (all passes, decimal: additive,
    /// multiplicative, cap; the enchantment steps and the min-0 clamp), AfterModifyingDamageAmount, BeforeDamageReceived,
    /// block absorption on the target or its pet owner, ModifyHpLost before Osty, ModifyUnblockedDamageTarget,
    /// ModifyHpLost after Osty on that target, the HP loss (the one int cast, Creature.LoseHpInternal), and when another
    /// creature took the hit the overkill back onto the original target through a second after-Osty fold. Then over all
    /// results in order: AfterBlockBroken, AfterCurrentHpChanged (unblocked &gt; 0), AfterDamageGiven, and
    /// AfterDamageReceived for a receiver not killed, else it joins the killed list; then <see cref="Kill" /> once.
    /// </summary>
    /// <remarks>
    /// A creature this call kills stays in the combat (<see cref="OldWorld.MarkDying" />) from its HP loss until
    /// <see cref="Kill" /> returns, as the game keeps it listed until CreatureCmd.Kill removes it. The old state's
    /// damage history is owned here, through <see cref="OldCreatures" />: the counters its readers keep (the History.DamageReceived
    /// entry at the game's point in the first loop; MIGRATION_F08.md, "Fixes after R2's review"). The AfterDamageReceived
    /// listeners' tallies and the old player-side body are their handlers' (batch L2). Not in the old state: the run statistics (the player's damage-dealt counter, the map point's
    /// damage taken) and the visuals.
    /// </remarks>
    public static IReadOnlyList<DamageResult> Damage(KernelState k, IReadOnlyList<OldCreature> targets, decimal amount,
        ValueProp props, OldCreature? dealer, CardHandle? cardSource, ChoiceCtx? ctx = null)
    {
        var w = k.World ?? throw new InvalidOperationException("CreatureCmd.Damage needs a switched-on kernel state");
        var s = w.Combat ?? throw new InvalidOperationException("CreatureCmd.Damage outside a combat");
        if (dealer is { } dead && OldCreatures.IsDead(s, dead))
            return targets.Select(t => new DamageResult(t, props)).ToList();
        var results = new List<DamageResult>();
        if (targets.Count == 0)
            return results;
        ctx ??= new ChoiceCtx();
        var dying = new List<IDisposable>();
        IDisposable? keep = null;
        try
        {
            foreach (var original in targets)
            {
                // game: `if (originalTarget.IsDead) continue`, and a target a nested kill or escape took out of the combat
                // meanwhile has no position to name (see the results loop): skipped too.
                if (OldCreatures.IsDead(s, original) || !OldCreatures.InCombat(w, original))
                    continue;
                var to = R(w, original);
                var by = By(w, dealer);
                decimal modified = Hooks.ModifyDamage(k, new ModifyDamageArgs(to, by, amount, props, cardSource,
                    ModifyDamageHookType.All, PreviewModes.None), out var mods);
                var amountArgs = new AfterModifyingDamageAmountArgs(cardSource, mods);
                Settle(k, Hooks.AfterModifyingDamageAmount(k, amountArgs),
                    st => Hooks.AfterModifyingDamageAmountResume(k, amountArgs, st));
                var before = new BeforeDamageReceivedArgs(ctx, to, modified, props, by, cardSource);
                Settle(k, Hooks.BeforeDamageReceived(k, before), st => Hooks.BeforeDamageReceivedResume(k, before, st));

                decimal blocked = DamageBlock(s, OldCreatures.PetOwner(original) ?? original, modified, props);
                decimal unblocked = Hooks.ModifyHpLost(k, new ModifyHpLostArgs(R(w, original),
                    Math.Max(modified - blocked, 0m), props, by, cardSource, HpLossHookPhase.BeforeOsty), out mods);
                AfterHpLost(k, mods, beforeOsty: true);
                var hitRef = Hooks.ModifyUnblockedDamageTarget(k,
                    new ModifyUnblockedDamageTargetArgs(R(w, original), unblocked, props, by));
                var hit = OldCreatures.At(w, hitRef);
                unblocked = Hooks.ModifyHpLost(k, new ModifyHpLostArgs(hitRef, unblocked, props, by, cardSource,
                    HpLossHookPhase.AfterOsty), out mods);
                AfterHpLost(k, mods, beforeOsty: false);
                var first = LoseHpInternal(w, hit, unblocked, props, dying);
                var list = new List<DamageResult>(2) { first };
                int blockNow = OldCreatures.Block(s, original);
                bool broken = blockNow <= 0 && blocked > 0m;
                bool fully = !props.Has(ValueProps.Unblockable) && (blocked > 0m || blockNow > 0) && (int)unblocked == 0;
                var own = first;
                if (hit != original)
                {
                    decimal back = Hooks.ModifyHpLost(k, new ModifyHpLostArgs(R(w, original), first.OverkillDamage, props,
                        by, cardSource, HpLossHookPhase.AfterOsty), out mods);
                    AfterHpLost(k, mods, beforeOsty: false);
                    own = back > 0m ? LoseHpInternal(w, original, back, props, dying) : new DamageResult(original, props);
                    list.Add(own);
                }

                own.BlockedDamage = (int)blocked;
                own.WasBlockBroken = broken;
                own.WasFullyBlocked = fully;
                // game: CombatManager.History.DamageReceived for each result of this target, after both results' flags are set.
                foreach (var entry in list)
                    OldCreatures.NoteDamageReceived(w, entry, dealer);
                results.AddRange(list);
            }

            // A player this call killed keeps its hooks over the passes before the kill (the game deactivates them only
            // inside it, after AfterDeath).
            keep = results.Exists(r => r.WasTargetKilled && r.Receiver.Kind == CreatureKind.Player) ? w.KeepPlayerHooks() : null;
            var killed = new List<OldCreature>();
            foreach (var r in results)
            {
                var rr = k.Commands.Add(r);
                // A receiver a nested kill or escape removed since its hit has no position to name in a hook argument, so its
                // passes are skipped (the game visits a model only while the combat still holds it, IterateHookListeners'
                // per-visit Contains; the Creature object it would pass to the others cannot be carried here). Checked again
                // before each pass: a hook can remove the receiver in the middle of its own result.
                bool Gone() => !OldCreatures.InCombat(w, r.Receiver);
                if (Gone())
                    continue;
                if (r.WasBlockBroken)
                {
                    var a = new AfterBlockBrokenArgs(R(w, r.Receiver));
                    Settle(k, Hooks.AfterBlockBroken(k, a), st => Hooks.AfterBlockBrokenResume(k, a, st));
                }

                if (r.UnblockedDamage > 0 && !Gone())
                {
                    var a = new AfterCurrentHpChangedArgs(R(w, r.Receiver), -r.UnblockedDamage);
                    Settle(k, Hooks.AfterCurrentHpChanged(k, a), st => Hooks.AfterCurrentHpChangedResume(k, a, st));
                }

                if (Gone())
                    continue;
                var given = new AfterDamageGivenArgs(ctx, By(w, dealer), rr, props, R(w, r.Receiver), cardSource);
                Settle(k, Hooks.AfterDamageGiven(k, given), st => Hooks.AfterDamageGivenResume(k, given, st));
                if (Gone())
                    continue;
                if (!r.WasTargetKilled || !OldCreatures.IsDead(s, r.Receiver))
                {
                    var a = new AfterDamageReceivedArgs(ctx, R(w, r.Receiver), rr, props, By(w, dealer), cardSource);
                    Settle(k, Hooks.AfterDamageReceived(k, a), st => Hooks.AfterDamageReceivedResume(k, a, st));
                }
                else
                {
                    killed.Add(r.Receiver);
                }
            }

            keep?.Dispose();
            // game: `await Kill(killedCreatures)`, once per Damage call, after every target (CreatureCmd.Kill.cs).
            Kill(k, killed);
        }
        finally
        {
            keep?.Dispose();
            for (int i = dying.Count - 1; i >= 0; i--)
                dying[i].Dispose();
        }

        return results;
    }

    /// <summary>game: Creature.DamageBlockInternal: the blocked amount (none when unblockable) is min(block, amount),
    /// and the block loses it cast to int.</summary>
    private static decimal DamageBlock(CombatState s, OldCreature c, decimal amount, ValueProp props)
    {
        decimal blocked = props.Has(ValueProps.Unblockable) ? 0m : Math.Min(OldCreatures.Block(s, c), amount);
        if (blocked != 0m)
            OldCreatures.SetBlock(s, c, OldCreatures.Block(s, c) - (int)blocked);
        return blocked;
    }

    /// <summary>game: Creature.LoseHpInternal, the command's one cast: the decimal truncated to int (at most 999999999),
    /// HP floored at 0, the overkill kept. A creature it kills is marked dying until the Damage call ends.</summary>
    private static DamageResult LoseHpInternal(OldWorld w, OldCreature c, decimal amount, ValueProp props,
        List<IDisposable> dying)
    {
        var s = w.Combat!;
        int hp = OldCreatures.Hp(s, c);
        bool kills = hp > 0 && amount >= hp;
        int cast = (int)Math.Min(amount, 999999999m);
        OldCreatures.SetHp(s, c, Math.Max(hp - cast, 0));
        if (kills && c.Enemy is { } enemy)
            dying.Add(w.MarkDying(enemy));
        return new DamageResult(c, props)
        {
            UnblockedDamage = hp - OldCreatures.Hp(s, c),
            WasTargetKilled = kills,
            OverkillDamage = kills ? Math.Max(cast - hp, 0) : 0,
        };
    }

    private static void AfterHpLost(KernelState k, IReadOnlyList<ListenerRef> mods, bool beforeOsty)
    {
        if (beforeOsty)
        {
            var a = new AfterModifyingHpLostBeforeOstyArgs(mods);
            Settle(k, Hooks.AfterModifyingHpLostBeforeOsty(k, a), st => Hooks.AfterModifyingHpLostBeforeOstyResume(k, a, st));
        }
        else
        {
            var a = new AfterModifyingHpLostAfterOstyArgs(mods);
            Settle(k, Hooks.AfterModifyingHpLostAfterOsty(k, a), st => Hooks.AfterModifyingHpLostAfterOstyResume(k, a, st));
        }
    }

    private static CreatureRef R(OldWorld w, OldCreature c) => OldCreatures.Ref(w, c);

    /// <summary>The dealer's ref at each use: null once it has left the combat (a dealer a nested Damage killed and the
    /// kill removed; the game keeps the Creature object, but no position names it any more).</summary>
    private static CreatureRef? By(OldWorld w, OldCreature? dealer) => dealer is { } d ? OldCreatures.TryRef(w, d) : null;

    /// <summary>An event dispatch inside a command: a pause the command cannot resume itself is settled in place
    /// (<see cref="KernelState.Settle" />: keyed answer, or the step stops and replays).</summary>
    internal static void Settle(KernelState k, HookStatus st, Func<HookStatus, HookStatus> resume)
    {
        if (st.IsPaused)
            k.Settle(st, resume);
    }
}
