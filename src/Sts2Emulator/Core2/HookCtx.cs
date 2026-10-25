// The handler context (docs/rebuild/TIER1_KERNEL.md section 4): the first parameter of every handler delegate
// (HookEvent, HookFold, HookCap, HookTry, HookTryOrNull, HookTryInPlace, HookPredicate, HookMutate), and the only way a
// handler reaches the combat it acts in. A handler never takes the old state, an engine or a static; it asks the context.
// Tier 1 answers over the old state (OldState.cs, OldState.Models.cs); a Tier 2 runtime answers the same questions over
// its own state, so a handler written against the neutral members below ports without a change.
using Sts2Emulator.Core;

namespace Sts2Emulator.Core2;

/// <summary>A power listener as a handler reads it: the creature it sits on and its amount (the game's PowerModel.Amount).</summary>
public readonly record struct PowerView(CreatureRef Owner, decimal Amount);

/// <summary>
/// What a handler is handed (one per dispatcher call; no state of its own, so a handler cannot keep anything in it).
/// <list type="bullet">
/// <item>Engine-neutral members (the shape a Tier 2 context keeps): <see cref="Power" />, <see cref="Owner" />,
/// <see cref="Card(CardHandle)" />, <see cref="Card(ListenerRef)" />, <see cref="Relic" />, <see cref="IsAlive" />,
/// <see cref="IsMonster" />, <see cref="IsPlayer" />, <see cref="Damage" />, <see cref="CombatInProgress" />,
/// <see cref="HittableEnemies" />, <see cref="IsDebuff" /> and <see cref="Niche" />.</item>
/// <item>Tier 1 members (the old state, the one view of it): <see cref="K" /> (what the commands take:
/// <c>CreatureCmd.Damage(c.K, ...)</c>), <see cref="World" />, <see cref="OldCombat" /> (the <c>CombatState</c>, the
/// argument of most old bodies: <c>CardEffects.PhantomBlades_ModifyDamageAdditive(c.OldCombat)</c>), <see cref="OldBuffs" />
/// (a creature's buff list, for an extracted <c>BuffSystem.Strength_ModifyDamageAdditive(List&lt;BuffState&gt;)</c>) and
/// <see cref="Enemy" /> (the old <c>EnemyState</c>, for a body that takes one). A Tier 2 context has none of these; a
/// handler that uses them is rewritten when the model moves (docs/rebuild/TIER2_REWRITE_LIST.md lists each use).</item>
/// </list>
/// A context made by a runtime that is no <see cref="KernelState" /> (a test fake) answers none of them.
/// </summary>
public readonly struct HookCtx
{
    private readonly HookRuntime _rt;

    internal HookCtx(HookRuntime rt) => _rt = rt;

    /// <summary>The per-combat kernel state this dispatch runs on (Tier 1): the commands, the world, the marks.</summary>
    public KernelState K => _rt as KernelState
                            ?? throw new InvalidOperationException("this hook runtime is no kernel state (a test fake)");

    /// <summary>The old state, with its marks (Tier 1). Needs a switch on, as every dispatch that reaches a handler has.</summary>
    public OldWorld World => K.World ?? throw new InvalidOperationException("no switch is on: the kernel holds no world");

    /// <summary>
    /// The power listener <paramref name="self" /> as it is on its owner now; null when it is not there (a power removed
    /// from its owner mid-pass is still visited by the game, and the old state keeps no copy: such a handler returns the
    /// identity or is a NotModelled gap its batch names). The amount is the old buff's magnitude.
    /// </summary>
    public PowerView? Power(ListenerRef self) =>
        OldState.PowerBuff(World, self) is { } p
            ? new PowerView(new CreatureRef(self.Instance >> 8), p.Buff.Magnitude)
            : null;

    /// <summary>The creature the listener sits on (the game's Owner or Creature of the model); null for a listener with
    /// none (a modifier, a badge, a scaling model) or one that is not on the old state now.</summary>
    public CreatureRef? Owner(ListenerRef self) =>
        OldState.Locate(World, self) is { Creature: >= 0 } me ? new CreatureRef(me.Creature) : null;

    /// <summary>Whether a creature of the combat is alive now (game: Creature.IsAlive); false for a position no creature
    /// holds.</summary>
    public bool IsAlive(CreatureRef creature) =>
        OldCreatures.TryAt(World, creature) is { } it && !OldCreatures.IsDead(World.Combat!, it);

    /// <summary>Whether a creature is a monster of the game model <paramref name="gameClass" /> (its class name: game
    /// <c>creature.Monster is T</c>); false for the player's side and for a position no creature holds.</summary>
    public bool IsMonster(CreatureRef creature, string gameClass) =>
        OldCreatures.TryAt(World, creature)?.Enemy is { } e && e.DefId == World.Ids.Resolve(ListenerKind.Monster, gameClass);

    /// <summary>Whether a creature of the combat is a player (game: Creature.IsPlayer); false for a pet, a monster and a
    /// position no creature holds.</summary>
    public bool IsPlayer(CreatureRef creature) => OldCreatures.TryAt(World, creature)?.Kind == CreatureKind.Player;

    /// <summary>Whether the combat is in progress (game: CombatManager.IsInProgress): false once it has ended, or before
    /// it starts.</summary>
    public bool CombatInProgress => CombatEnd.IsInProgress(K);

    /// <summary>
    /// game: RunState.Rng.Niche, the run's Niche stream: CreatureCmd.Add's CombatState.CreateCreature rolls every new
    /// enemy's HP from it (Creature.SetUniqueMonsterHpValue), so it is what SurprisePower's, InfestedPower's and StockPower's
    /// AfterDeath draw, whatever kills the owner. Tier 1 answers with the combat's own copy of it,
    /// <c>CombatState.NicheHpRng</c>: taken at the run's Niche position when the fight starts, restreamed with Niche on a
    /// resample, handed back to the run when the fight ends, and cloned with the combat, so a fork, a snapshot or a probe
    /// draws its own copy and never the run's. A hand-made combat that no run built has none and answers with its Shuffle
    /// stream (the stand-in the old in-blow death answer passes); throws when it has neither.
    /// </summary>
    public Random Niche => OldCombat.NicheHpRng ?? OldCombat.ShuffleRng
        ?? throw new InvalidOperationException("the combat holds no Niche stream (NicheHpRng) and no Shuffle stream");

    /// <summary>The enemies a hit can reach now, in roster order (game: CombatState.HittableEnemies: alive, in the combat
    /// and hittable).</summary>
    public IReadOnlyList<CreatureRef> HittableEnemies()
    {
        var w = World;
        return OldCreatures.HittableEnemies(w).Select(e => OldCreatures.Ref(w, e)).ToList();
    }

    /// <summary>Whether a power is a debuff (game: PowerModel.Type == PowerType.Debuff), read from the game's power data
    /// by its model id.</summary>
    public bool IsDebuff(PowerRef power) => !Sts2Emulator.GeneratedData.Powers.Get(power.ModelId).IsBuff;

    /// <summary>The damage result an argument names (game: the DamageResult a damage hook is handed): its unblocked and
    /// blocked damage, overkill and kill.</summary>
    public DamageResult Damage(DamageResultRef result) => K.Commands.Result(result);

    /// <summary>The old <c>CombatState</c> (Tier 1: the argument of the old engine's bodies, whose first parameter is the
    /// state). Read it; write it only through an old body or a command.</summary>
    public CombatState OldCombat => World.Combat ?? throw new InvalidOperationException("the kernel's world holds no combat");

    /// <summary>
    /// A card of the combat by its handle (an argument's card source, a played card): its model id, pile and position, or
    /// null when no live card has that uid. Found by the K2b uid, so the answer is the card's place now.
    /// </summary>
    public CardView? Card(CardHandle card) => OldState.CardOf(World, card);

    /// <summary>
    /// The card a card-kind listener runs for (a card, or the enchantment or affliction on it): its model id, pile and
    /// position now, so a handler acts on its own copy only (the game's hook runs once per copy). Null when the listener
    /// is not a card kind, is not on the lists now, or names another model than the card has now.
    /// </summary>
    public CardView? Card(ListenerRef self) => OldState.CardOf(World, self);

    /// <summary>
    /// The relic listener <paramref name="self" /> as held now: its model id, its position in the obtain order and its
    /// counter (<c>RelicInstance.Counter</c>); null when it is not held. A relic's other saved properties are fields of the
    /// old <see cref="OldCombat" /> that its old body reads itself (Tier 1; on the rewrite list), not part of the view.
    /// </summary>
    public RelicView? Relic(ListenerRef self) => OldState.RelicOf(World, self);

    /// <summary>
    /// The creature that owns a card (the game's <c>CardModel.Owner.Creature</c>). The old state holds one card owner, the
    /// player (creature 0), so every handle names a card of the player; the handle is taken so that a handler written
    /// against this member ports unchanged to a runtime with several card owners.
    /// </summary>
    public CreatureRef CardOwner(CardHandle card) => new(0);

    /// <summary>
    /// Whether <paramref name="creature" /> is <paramref name="owner" /> or one of its pets (the game's
    /// <c>creature == Owner || Owner.Pets.Contains(creature)</c>, the test of a damage modifier for its owner's own hits):
    /// the old state has one pet, Osty, whose owner is the player. False for no creature.
    /// </summary>
    public bool IsOwnerOrPet(CreatureRef? creature, CreatureRef owner) =>
        creature is { } c && (c == owner || (OldCreatures.TryAt(World, c) is { } old
            && OldCreatures.PetOwner(old) is { } petOwner && OldCreatures.TryRef(World, petOwner) == owner));

    /// <summary>The pet owner of a creature (game: Creature.PetOwner, its creature): the player for Osty (the old state's one
    /// pet); null for any other creature and a position no creature holds.</summary>
    public CreatureRef? PetOwner(CreatureRef creature) =>
        OldCreatures.TryAt(World, creature) is { } old && OldCreatures.PetOwner(old) is { } owner
            ? OldCreatures.TryRef(World, owner)
            : null;

    /// <summary>The old enemy a creature reference names now (Tier 1: the argument of an old body that takes an
    /// <c>EnemyState</c>, as <c>RelicEffects.ApplyAfterBlockBroken</c> does); null for the player, Osty or a position no
    /// creature of the combat holds.</summary>
    public EnemyState? Enemy(CreatureRef creature) => OldCreatures.TryAt(World, creature)?.Enemy;

    /// <summary>The old buff list of a creature in the combat (Tier 1: the argument of an extracted
    /// <c>Model_Hook</c> function of BuffSystem). Throws for a creature that has none (a pet).</summary>
    public List<BuffState> OldBuffs(CreatureRef creature) =>
        OldCreatures.Buffs(World.Combat!, OldCreatures.At(World, creature))
        ?? throw new InvalidOperationException($"creature {creature.Index} has no old buff list");
}
