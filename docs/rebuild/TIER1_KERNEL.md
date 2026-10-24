# Tier 1 kernel: design note

2026-10-23, branch `rebuild/tier1-kernel` (from main 97b03d1e), game v0.107.1. Spireline decision D23 and
`PLAN.md` section 8.3: build the kernel in the game's shape, then migrate content onto it one hook family at a time,
the old code path serving every family not yet migrated. The full Tier 2 design is
`referee/design/rebuild-plan.md`; this note says what Tier 1 builds of it and how it sits on the old engine.
Game code is described by class and method identifiers only. Revised after review K0 (section 9).

## 1. Principle: one switch per site group, zero switches equals old

- The kernel is new code (`src/Sts2Emulator/Core2/`, namespace `Sts2Emulator.Core2`) that runs over the old engine's
  state (`CombatState`, `RunState`). It does not copy or replace that state.
- **Site inventory** (`tools/kernel/sites.json`, built in K3 from `tools/fidelity/hook_dispatch.json`): every old
  dispatch site (method, the hook moments it hand-dispatches) mapped to the game dispatchers it stands for. A
  **switch** is the smallest set of families whose dispatchers share an old site; families that share a site switch
  together. The turn-start flow (CombatManager's setup, AfterSideTurnStart and the orb queue's turn start: F02, F03,
  F13) is one switch for its pause timing, whatever the order of its listeners' migration.
- Switches are set per engine instance (`KernelOptions.Switches`); `--engine core2` reads them from
  `tools/fidelity/migrated_families.json`.
- With no switch on, `core2` is the old engine: no queue is used, no kernel state is written, and dumps and state hashes
  are byte-identical on the corpus. This is the kernel's first gate and the fuzzer's sanity check.
- A switched-on group moves its old sites onto `Hooks.<Dispatcher>(...)` behind `if (k.On(Switch.X)) ... else <old>`.
  Its listeners become handlers (section 4) that call the old engine's bodies. What is ported is order, membership,
  passes and pauses; the inside of a body stays the old one until a later family (damage, block, piles) replaces the
  commands it calls.
- The combat-over-or-ending guard of the game's guarded iterator reads the old engine's own flag
  (`CombatEngine.IsOverOrEnding`), so it changes nothing by itself; F10's switch changes when that flag is set.
- Built in K3 (`python -m tools.kernel.sites build|check|groups`, `tools/kernel/sites.json`, generated
  `Core2/Switches.g.cs`):
  - A site is a `hook_dispatch.json` method or region, plus the methods its `generic` entries name before the colon
    (`CombatEngine.EffectiveCost`, `IncrementPlayedCardTypeCounters`, `CaptureBeforePlayPowers`; a fold rule names no
    site). 279 sites (the two callers review K3 named, `CombatEngine.TurnPlayerTowardTarget` and `CardEffects.FirstEnemy`,
    and `RelicEffects.BiteForFurCoat`, which does the same, call `EnemyAI.IsTargetable`, the old `Hook.ShouldAllowHitting`,
    so they are sites of F08), each resolved to its file and line in the engine (a name that is gone is STALE); 126
    dispatchers have an old site, 21 have none; one hook with a site has no dispatcher (`AfterModifyingCardPlayResultPileOrPosition`,
    called outside Hook).
  - Groups by union over: families sharing a site not declared split, a dispatcher's passes (no dispatcher spans two
    families today), and the flows (`FLOWS`: F02+F03+F13). `check` fails when a family is in two groups, a dispatcher
    with a site or a flow is not in exactly one, or a site straddles two; each has a broken case
    (`tests/python/test_kernel_sites.py`).
  - **Result: one group, F01-F16.** 28 sites serve more than one family (the turn-start and first-turn methods of
    CombatEngine and RelicEffects, the EndTurn regions, `EffectiveCost`, `IncrementPlayedCardTypeCounters`, relic
    bodies such as `ApplyMusicBox`, `FinishBlockDoublingCard`, `RefreshBeltBuckle`, and the run's `AddCardToDeck`,
    `EnterShop`, `GenerateCombatRewards`); `sites groups` lists them. With every bridge split the groups would be F01,
    F02+F03+F13, and each other family alone. A coarse group is sound, never finer than the sites allow; before the first
    family migrates, its bridges are read and either recorded in `SPLITS` or left joined. A record is one of three
    kinds, each with a reason: `split` (the families' parts are separate statements of the site: the anchor text of each
    part, in source order), `by_caller` (a shared helper that serves one family per caller: each caller and the text of
    its call; the gate sits at the callers) and `false_listing` (a `hook_dispatch.json` entry that is not a dispatch
    there: the hook leaves the site, and the site leaves the inventory if it has no other). A split or by_caller record
    keeps the file and its hash (line ends normalised) as read, and `sites check` reads the source: a split's anchors
    must be present, unique and in order, a by_caller's present and its callers methods of the engine, and a changed
    file sends the record back to be read again. Each kind and each way to break it has a test
    (`tests/python/test_kernel_sites.py`). `SPLITS` is empty: no bridge has been read yet.
  - `Switch` has one member per family; `k.On(Switch.F06)` is true when F06's group is on. `KernelOptions` refuses a
    partial group (the families that share a site are named) and an unknown family. `SwitchGroups.FamilyOf(HookId)`
    gives a hook virtual's or dispatcher's family (hook_families.json).

## 2. The hook descriptor and its generator

- `tools/kernel/hooks.json`, checked in, one entry per public dispatcher of the game's `Hook` class (about 147, in
  about 14 shapes), identifiers and structure only:
  - `iterator`: `combat` (guarded: empty once the combat is over or ending, except during combat setup; read once at
    pass start), `combat-direct` (unguarded), `run+combat` (section 3), `run`;
  - `passes`, in order, each with:
    - `op`: `event` | `fold-add` | `fold-mul` | `fold-cap` (running minimum; changed when the listener lowers it) |
      `fold-set` | `try` (TryModify: the listener returns whether it changed the value) | `predicate-all` (first false
      is the preventer) | `predicate-any` (first true wins: ShouldStopCombatFromEnding, ShouldTakeExtraTurn,
      ShouldForcePotionReward) | `filter-set` (the AfterModifying* receivers: game order, only listeners in the passed
      changed set) | `preventer-only` (the AfterPreventing* receivers: the one preventer, if the iterator still yields
      it) | `mutate` (in-place: ModifyKeywordsInCombat, ModifyShuffleOrder);
    - `changed`: `nonzero` | `not-one` | `ne` | `int-ne` (compared after an int cast) | `trunc-ne` | `try-true` |
      `none` (not tracked);
    - `when`: the argument flag that selects the pass at run time (ModifyDamage's ModifyDamageHookType, ModifyHpLost's
      HpLossHookPhase), or absent;
    - `also`: for a `try-true` pass, the other parts of its condition beside the listener call, in our words
      (`out-not-null`: the value the listener gave back must be present, ModifyCardBeingAddedToDeck); any other part
      is an extractor error, so a new condition there is flagged;
  - `context`: `none` | `passed` (the caller's choice context, pushed and popped per listener) | `passed-unpushed`
    (passed without push: AfterAttack, AfterOrbEvoked, AfterPowerAmountChanged) | `caller-hook` (the caller's
    HookPlayerChoiceContext: AfterAutoPrePlayPhaseEntered, AfterAutoPostPlayPhaseEntered) | `per-listener` (each
    listener gets its own hook choice context; a listener that asks the player becomes a hook action and the dispatch
    moves on);
  - `wait_all`: the pass indices after which the dispatcher waits for every listener it started;
  - `pre`, `post`: named steps outside the loop (`enchant-add`, `enchant-mul`, `clamp-min0`, `netid-guard`,
    `preview-hittable`, `negative-cost-passthrough` (ModifyEnergyCostInCombat), `owner-in-combat`
    (ShouldPowerBeRemovedOnDeath));
  - `partner`: the receiving dispatcher of the changed set or the preventer, when the caller passes it on;
  - `args`: parameters in kernel types (hand-written: `CardHandle`, `CreatureRef`, `int`, `decimal`, `bool`, `Side`
    ...). The extractor does not write these. A `mutate` dispatcher needs a mutable argument (`MutableCardList` =
    `IList<CardHandle>`, `KeywordSet`, or an `Opaque` game object); the generator rejects one without.
- `tools/kernel/hooks_extract.py` extends `tools/fidelity/hook_table.dispatchers` to read every field above except
  `args` from the local decompile. `python -m tools.kernel.construct shape` diffs the checked-in descriptor against a
  fresh extraction both ways and checks that `Hooks.g.cs` equals a fresh generation. A dispatcher the extractor
  cannot classify is an error, never a default. Each mode has a broken case it catches.
- `tools/kernel/gen_hooks.py` writes `src/Sts2Emulator/Core2/Hooks.g.cs` and `HookId.g.cs` (committed; generated from
  our descriptor, not from game text): one static method per dispatcher, a fast path when no switched-on listener is
  registered for its passes, otherwise per pass the runtime's listener cursor (section 3) and the loop of its op.
  Event dispatchers whose context awaits each listener's whole task (every context but `per-listener`) return a
  `HookStatus` and have a `<Name>Resume` (section 5).
- Named steps are hand-written in `Core2/Steps.cs`, one function each, cited `game: Hook.<Method>`. The steps that read
  the dispatcher's arguments (`enchant-add`, `enchant-mul`, `owner-in-combat`, `preview-hittable`) are declared in
  `Hooks.g.cs` as partial methods typed by that dispatcher's argument record (so EnchantAdd reads `CardSource` and
  `Props`, PreviewHittable passes `a with { Target = enemy }` to the passes) and implemented per dispatcher in Steps.cs.
- Built in K1 (v0.107.1): 147 dispatchers, 178 passes over 178 hook virtuals, 15 shapes by (ops, context, wait) and
  22 counting changed criteria and steps. The extraction needed, beyond the list above: `changed: lowers` (fold-cap:
  the listener lowered the value); `try` with `changed: none` (the energy and star cost TryModify passes ignore the
  listener's answer); `negative-cost-passthrough` on ModifyStarCost too; a `when` on pre steps (ModifyDamage's enchant
  steps); the post step `dispatch:<Dispatcher>` (ModifyGeneratedMap ends by running ModifyGeneratedMapLate); `partner`
  as a list (ModifyHpLost feeds both AfterModifyingHpLost receivers). The runtime API Hooks.g.cs is written against is
  `Core2/HookRuntime.cs`; a dispatcher with steps has its passes as `<Name>Passes`.

## 3. Listener order over the old state

A pass reads its listeners through `HookCursor<THandler>` (`Core2/HookRuntime.cs`): `bool Next(out HookListener<T>)`
over the snapshot taken at pass start, an optional late part built once the first is exhausted (run+combat), a
visit-time membership check that skips an entry, and `Index`, the visited entry's position in the pass's order (both
parts, skipped entries counted), which a pause records. `Core2/Listeners.cs` builds the parts as the game's
`CombatState.IterateHookListeners` does:
1. creatures: allies (the player, then pets and Osty where the old engine models them) then enemies in roster order;
2. per creature: its powers in application order; then, for every creature without a player (enemies, pets, Osty),
   its monster; for a player: relics in obtain order (melted skipped), potions by slot, orbs in queue order, then the
   cards of Hand, Draw, Discard, Exhaust and Play in pile order, each followed by its affliction and its enchantment;
3. then the combat's modifiers, badges and the multiplayer scaling model.
- **run+combat iterator** (the run state's IterateHookListeners with a combat): the run's deck cards and their
  enchantments come first, visited as they are enumerated; the combat's list above is built only after them, so it
  is built when the deck part ends, not at pass start. Outside a combat the run's relics and potions too. ModifyDamage,
  ModifyHpLost, ShouldDie, AfterDeath and the damage-received dispatchers use it. `OldState` maps the run deck.
- **Membership at visit time**, by the game's `CombatState.Contains`: a power while its owner creature is in the
  combat and, for a player-owned power, while that player is active for hooks (a power removed from its owner
  mid-pass is still called, with the state it had); a relic, potion, card, orb, affliction or enchantment while it has
  not been removed from the combat's state and its owner is active for hooks; a monster while its creature is in the
  combat. Activity is read at the entry's turn (CreatureCmd deactivates hooks mid-pass): the old state cannot tell the
  player's activity (the game deactivates a creature's hooks only after its death-prevention hooks and AfterDeath, so
  the player's own death passes still visit its listeners at 0 HP), so it is a caller mark (`KeepPlayerHooks`,
  `DeactivatePlayerHooks`; default active). A listener added mid-pass is not visited.
- **Card identity first** (K2b, under the byte-identity gate): a per-combat card uid on `CardInstance`, left out of
  equality, the snapshot writer and the state walk; assigned when a card enters the combat (deck clone at combat
  start, generated or copied cards as created: a copy gets a new uid, a `with` change keeps it). A `CardHandle` is that
  uid, opaque to handlers, so Tier 2's card arena can replace it. An invariant (no two cards share a uid; every card in a
  pile has one) runs on the corpus and the fuzzer. Until K2b lands, no switch with card listeners turns on.
- Old-state mapping (`Core2/OldState.cs`, the only kernel file that knows the old representation): powers through
  `BuffId` to the game's power model (`tools/fidelity/power_aliases.json`), the named engine-only bookkeeping buffs left
  out, a buff no rule classifies NotModelled, one game power split over several `BuffId`s visited once at its first
  entry; relics, potions, orbs, piles as the old
  lists hold them; the old engine has no Play pile, so the card in flight is its one entry for the length of its play.
  What the old state cannot represent (folded power instances, melted relics) is listed in `Core2/NotModelled.cs`; a
  pass that would need it reports NotModelled, never a guess.
- Tests (`src/Sts2Emulator.Tests/Core2/ListenerOrderTests.cs`), each from the game's rule: creature order with pets and
  several enemies; powers before the monster, on enemies and pets; relic obtain order and its reverse; potions by slot
  with gaps; pile order with the affliction and enchantment after each card; modifiers, badges, scaling last;
  run+combat deck-first with the combat list built after; a power removed mid-pass still called; a creature removed
  mid-pass not called; a deactivated player's listeners skipped from that entry on; a listener added mid-pass not
  called; the guard read once; agreement with `hook_table.game_order_key` on the `hook_order` scenarios.
- Built in K2 (`Core2/OldState.cs`, `Listeners.cs`, `NotModelled.cs`), choices settled:
  - `ListenerRef` = (`ListenerKind`, model id, instance). Model ids are data/id_map.json's values (cards, relics,
    potions, monsters: the old def ids; powers: id_map's "powers" numbering); orbs and enchantments the old enums,
    afflictions `GameAffliction`. Instance: a power `creature << 8 | k` (its k-th instance on that creature), a
    monster its creature, a relic its obtain index, a potion its slot, an orb its queue position, a combat card and its
    affliction and enchantment the K2b uid, a run deck card `-(deck index + 1)`.
  - BuffId to game power (`OldModelIds`), in this order: engine-only only by an explicit list (Stunned, SummonCooldown,
    TerrorQueued, BackupCount, Reviving, FranticEscapePlayedCountUnused); a fold target or member of power_aliases.json
    (TemporaryStrength, TemporaryDexterity, TemporaryFocus, BlockNextTurn, and SetupStrikePower, a
    TemporaryStrengthPower subclass) is Folded and NotModelled; else the name (UPPER_SNAKE without _POWER) through
    "emu_rename" (plus ShivDamage = AccuracyPower, missing from that file and kept in OldModelIds so its other readers
    are unchanged), a game power if id_map has it; else a piece of the longest game power whose name prefixes it
    (visited once, at its first entry); else NotModelled, never engine-only (Burn, Metallicize, ClawDamage and
    WitherIntensity today: the game's power list has no class of that name, so a listener may be missing).
    `BuffClassTests` pins the class of every BuffId; a new BuffId or a changed alias or fold fails it until classified.
  - Creatures: the player 0, Osty 1 while `OstyMaxHp > 0` (its monster; it has no powers in the old state:
    NotModelled), then enemies in roster order. An enemy is in the combat while the roster holds it and it is alive,
    or the caller marks it dying (`OldWorld.MarkDying`, inside the kill), or a power vetoes its removal
    (`EnemyAI.ShouldRemoveOnDeath`). The player is active for hooks unless the caller declared them deactivated
    (`OldWorld.PlayerHooks`; not derived from `IsAboutToLose`, which turns true at 0 HP, before the game deactivates).
  - The Play pile: `OldWorld.Flying` (kernel play sites set it for a play's length); else `HeldPlayedCard` while a
    selection is pending and its uid is in no pile. The field can outlive its play (only ReleaseHeldPlayedCard clears
    it, and a snapshot never restores it): a held card whose uid is filed is stale; one whose result pile ends with a
    card of equal value under another uid is NotModelled (stale or twin).
  - NotModelled also: afflictions other than Bound, cards in `AutoPlayQueue` (members by uid, not listed), a run deck
    with no `RunState`. Orbs have no identity (matched by type and queue position) and deck cards are matched by
    index, so a pass that lists orbs or deck cards remembers the list at its start and reports NotModelled
    (`OrbsChangedMidPass`, `DeckChangedMidPass`) at the first such entry visited after any change to it: an evoke, a
    channel, a removal, a replaced list (a `List` enumerator taken at the snapshot detects every mutation, public API,
    AOT-safe). Melted relics are
    gone from the old list and skipped by the game, so the order is the same. Modifiers, badges and scaling: the old
    state has none; the caller supplies them on `OldWorld`.
  - **K3 obligations, the marks** (`OldWorld`; each a scoped setter, `using (w.X(..))`, restoring the previous value on
    exit; the site that opens the window sets it, with a test that a pass inside the window differs from one outside).
    A mark nobody sets leaves a default that silently drops or adds listeners unless the state shows it is missing:
    - `Flying(card)`: every play site, for the play's length. Missing: the card in flight is in no pile and no pass
      visits it. Not detectable (the old engine holds the card in a local); only the held-card case is.
    - `Setup()`: combat setup (IsStarting). Missing: a guarded pass in setup returns empty. Detected when the guard is
      set before turn 1 (NotModelled `SetupUndeclared`; a combat that ends before its first turn is reported too, the
      safe side); a setup pass with the guard clear is right either way.
    - `MarkDying(enemy)`: inside CreatureCmd.Kill, before the removal. Missing: the dying enemy's powers and monster
      are dropped from the death passes. Not detectable (the old engine keeps a dead enemy in the roster, so the state
      looks the same after the removal): the kill site's test is the only guard.
    - `KeepPlayerHooks()` over the player's own kill passes (death prevention, AfterDeath) and
      `DeactivatePlayerHooks()` from DeactivateHooks to the end of the combat. Missing with the player down
      (`IsAboutToLose`): NotModelled `PlayerHooksUndeclared`, at pass start and at any visit.
  - The handler table enters as `IHandlerSource` (`Find(hook, kind, model)`, `AnyOf(hook, kind)`); a gap is reported
    for a pass (`OldListeners.Report`, `LastPass`) only when a handler could stand for it. Each part's build runs
    `CardUidInvariant.Normalise`.
  - Agreement with `game_order_key` (`python -m tools.kernel.listener_order`, EmuDump `kernel-order` on the
    hook_order scenarios plus two of the tool's own: three enemies with powers, an orb queue and the exhaust pile, in both
    relic and potion orders, with the listener counts the state implies): 18 scenarios, 2,190 listeners, the list
    sorted by the key with 0 disagreements, 118,088 scenario pairs with 0 disagreements, 0 listener-count failures; the
    12 hook_order listeners not listed are all folded powers.

## 4. Handlers, the handler table and the dispatch

- A migrated model is one static class `Core2/Models/<Kind>/<GameClass>.cs` with one method per game override it
  ports, named and shaped like the override: events return `HookStep` (`Done`, or `Paused(request, resume)` at the
  game's await-of-choice site, where `resume` is a typed record per site holding what the rest of the body needs);
  folds take and return `decimal`; predicates return `bool`; Early and Late passes are their own methods. Each method
  cites `game: <Class>.<Method>` and calls the old engine for the body.
- `Core2/HandlerTable.cs`: per `HookId`, handlers keyed by the game's ModelId (through `data/id_map.json` and
  `power_aliases.json`), registered by each migrated model's `Register()` when its switch is on; a per-model mask
  derived from it; the old def id is resolved in `OldState` only.
- Folds return the value and the changed set by the pass's `changed` criterion; `predicate-all` returns the
  preventer, `predicate-any` the first true. Damage and block keep `decimal`; the old caller casts where the game casts.
- Only `Hooks` calls a handler; handlers change state only through the old engine's commands (a source-scan test until
  analyzers exist).
- Built in K3:
  - `HandlerTable` (an `IHandlerSource`): `t.Model(kind, "<GameClass>").On(HookId.<virtual>, handler)`, or
    `t.Register<TModel>()` for an `IMigratedModel` (static `Register(HandlerTable)`); migrated models are listed in
    `Core2/Models/Registry.cs` (empty). The name resolves through `OldModelIds.Resolve` (OldState.cs): id_map.json's
    number for powers, cards, relics, potions and monsters; the old enum for orbs (class name without "Orb"),
    enchantments and afflictions; modifiers, badges and scaling refused. A handler registers only when its hook's family
    switch is on (the family is the hook's, not the caller's word); a dispatcher id, a hook in no family, a second
    handler for the same (hook, model) and two names for one old id are errors. `Mask(kind, id|name)` is the model's
    hook set; `HasAny(hook)` is the dispatchers' fast path. The handler's delegate type is checked when a pass reads it
    (Listeners.cs), not at registration.
  - The kernel is two parts (review K3). `Kernel` (Core2/Kernel.cs) is shared and immutable: `new Kernel(KernelOptions,
    Func<OldModelIds>, models)`, `On(Switch)`, `AnyOn`, and `Handlers` and `Ids`, each built once on first use (a
    thread-safe `Lazy`) and only with a switch on; `Core2Engine` caches the options, the ids and the default kernel
    statically, so engines share them. `KernelState : HookRuntime` (Core2/KernelState.cs) is one per combat: the combat
    binding (`At`), the K2 caller marks and listener cursor (its `OldWorld`), and the recorded `HookActions`, `Stops`,
    `Waits` and the action counter. `Listeners` is `OldListeners.Cursor` over its `OldWorld`; `OnPaused` records a
    per-listener pause as a `RecordedHookAction` (id from one counter) and any other pause as a stop; `HookActionMark` is
    the counter; `WaitAll` records a `WaitPoint` with the pending count and leaves the actions to K4's executor. With no
    switch on a state builds no world, table or cursor, `OnPaused` and `WaitAll` return without recording, and `HasAny`
    is false, so every dispatcher returns at its fast path; the eight dispatchers with the net-id guard
    (`Steps.NetIdGuard`) test the fast path first (the generator orders it so), so the guard is not reached either.
  - Attachment: nothing is thread-bound. A state holds its own: `CombatState.Kernel` and `RunState.Kernel`
    (`KernelState?`, `[IdentityOnly]`, so the state walk, reanchor, dumps and snapshot writers never see it; null on the
    old engine, which pays one null field). `Kernel.Bind(combat, run)` makes the state and sets it; `Kernel.Attach` does
    that only with a switch on. `RunEngine.Kernel` (set by `Core2Engine`) binds the run's state at `Reset` and each
    combat the engine starts; a combat built elsewhere (a snapshot restore) holds none and its sites run old until it is
    bound. `Clone` and `CopyTo` copy the state (`KernelState.CopyTo`): the binding to the copy, the marks (a dying enemy
    matched by roster index), the recorded actions, stops, waits and the counter, with the pause contexts copied and
    nothing shared with the source. `CopyTo(into)` fills the `KernelState` that `into` already holds (a step restored in
    place by `HookChoiceReplay`, `Before.CopyTo(state)`), so a replay records the step's actions once; `RunState.Clone`
    copies the run's and rebinds the cloned combat's to the cloned run. A site's gate is
    `if (Kernel.If(Switch.X, combat, run) is { } k) Hooks.Y(k, ...) else <old>`: `k` is the combat's own state (null on
    the old engine and with the family's switch off); `Kernel.If(Switch.X, run)` is the run-level gate (the active
    combat's state while one is open, so the marks are one set). The marks are `using (Kernel.Marks.Flying(combat, card))`,
    `Setup(combat)`, `MarkDying(combat, enemy)`, `KeepPlayerHooks(combat)`, `DeactivatePlayerHooks(combat)`: a shared no-op
    for a combat with no state or no switch on, and a scope restores the previous value on exit. No old site calls them
    yet: each is placed with the first switch whose passes need it, with its test. Tests (`KernelTests`,
    `EngineFlagTests`): an old and a core2 combat on one thread share nothing, an old engine made after a core2 engine
    runs old, a clone carries its actions and marks and a replay from it records once, a marks scope restores on exit,
    no switch records nothing.
  - `--engine core2`: `Core2Engine : OldEngine` (tools/emudump/Engines/Core2, admitted by the engine-boundary test) with
    `KernelOptions` from `migrated_families.json` (copied next to EmuDump) and `Registry.All`; `EmuDump fight|reanchor|
    hookprobe|setupinfo --engine old|core2` (refused for other modes). byte_proof and `corpus.py identity` take
    `--a-args` / `--b-args`.

## 5. The action queue and pauses

- `Core2/ActionQueueSet.cs` and `Core2/ActionExecutor.cs`, as the game's `ActionQueueSet` and `ActionExecutor`: one
  queue per player of `GameActionRec` (kind: play card, use potion, end turn, ready-to-begin-enemy-turn, hook action;
  type: combat, combat-play-phase-only, non-combat, any; state: waiting, executing, gathering choice, ready to resume,
  finished, canceled; ids from one counter, a resumed action re-enqueued under a new id). The ready action is the
  lowest id at the front of a queue, skipping canceled fronts, queues gathering a choice, combat-play-phase-only actions
  of a paused queue and non-combat actions during a combat.
- Used only when a switch that needs it is on; then a player step enqueues its action and drains the queue, the
  action's body being the old engine's step.
- By switch: the win check after every action except end turn and ready-to-begin-enemy-turn (F10); at combat end the
  game's `CombatEnded` cancels the hook actions that have not started, and the player-driven cancel applies as the game
  sets it (F10); hook actions from `per-listener` passes (that dispatcher's switch).
- **Order is the queue's, always.** A handler that reaches a choice returns `Paused`. In a `per-listener` pass the
  dispatcher records a hook action (listener, resume record) and moves on to the later listeners; at a `wait_all`
  point the dispatcher drains the hook actions it started, in id order, before the next pass; elsewhere they run when
  the executor reaches them. The turn-start flow is a `flow` pause: CombatManager's own context waits for pause or
  completion without assigning a task, and its hook action is enqueued only when the auto-pre-play phase assigns it, so
  AfterSideTurnStart and the orb queue's turn start run before that choice (`hook_order.pause_floors`).
- **Continuation across the answer.** Where the hook action's screen opens inside a caller that can stop and resume
  (the executor, between actions), the answer runs the resume record directly. Where it opens inside old code that
  cannot (a drain at a `wait_all` point inside an old turn-end site), the step stops with the screen open through the old
  engine's `PendingSelection`, and the answer replays the step through `HookChoiceReplay`. The replayed step defers and
  drains exactly as the first run did, so the later listeners still run before the screen. Answers to kernel screens are
  keyed by (dispatcher, listener, resume site); answers to old screens keep their order. `HookChoiceReplay` gains the
  keyed lookup, and a step may mix both kinds.
- `none`, `passed`, `passed-unpushed` and `caller-hook` contexts: the game awaits each listener's whole task, so the
  later listeners run only after the choice is answered. The dispatcher stops at the pause and returns a paused
  `HookStatus` (its `PauseSite` with the pass and the listener's cursor `Index`, and the pass's cursor positioned
  after the listener); a pushed context keeps the listener pushed while the choice is open. Once the listener's resume
  record has run, `<Name>Resume(rt, a, status)` pops it and continues the same cursor, then the later passes (no
  snapshot is taken again; membership is still checked at visit). A choice inside a card's own play or a potion
  pauses the action that owns the context, as the game's player choice context does, and keeps the old replay; K4's
  flow pause continues from (pass, index) the same way. Such a dispatcher has no wait point or post step (the
  generator refuses one).
- Tests from the game's `ActionExecutor`, `ActionQueueSet` and `HookPlayerChoiceContext`: the boundary win check and
  its two exempt kinds; canceled fronts skipped; a paused action yields to the next ready action; a resumed action's new
  id; type rules; `CombatEnded` cancelling unstarted hook actions; later listeners before the screen in BeforeTurnEnd,
  AfterTurnEnd and BeforeFlush; the turn-start flow pause; clone parity of the queue.

- Built in K4 (settled):
  - **Queue** (`Core2/ActionQueueSet.cs`): `GameActionKind` (PlayCard, UsePotion, EndTurn, ReadyToBeginEnemyTurn, Hook),
    the game's `GameActionType` and `GameActionState`, and `GameActionRec` (plain data: kind, type, owner, id, state,
    the old step's action and target, the hook action's id, the answer it resumes with, the id it first ran under).
    `ActionQueueSet`: `Enqueue` (the game's three cancellations), `GetReadyAction`, `Start`/`Finish`,
    `PauseForChoice(a, cancelPlayCards)`, `Resume(id, answer)` (a new id from the one counter; a resumption that
    arrives first is kept), `PauseAll`/`UnpauseAll`, `StartCancellingPlayerDriven`, `CombatStarted`/`CombatEnded`
    (cancels every combat-typed action not executing: unstarted and gathering hook actions), `CancelWaitingFor`,
    `Copy`. It lives in `KernelState.Queue`, built on first use only with a switch on, copied by `CopyTo`.
  - **Executor** (`Core2/ActionExecutor.cs`): `ActionExecutor.Drain(k, host)` runs the ready action, then the win check
    unless it was EndTurn or ReadyToBeginEnemyTurn (F10's group; `WinChecks` counts), until nothing is ready or
    `ExecutorPaused`. An EndTurn action finishes before its body (the game's EndPlayerTurnAction only marks the player
    ready; the turn's end runs outside any action). `IActionHost` runs a player action's body. With a switch on,
    `RunEngine.Step` sends a combat step through `QueueStep`: a fresh action is enqueued and drained, its body the old
    step (`StepInner`); an answer to an old screen resumes the gathering action (new id) and its body replays; a
    replay's restore of the kernel state keeps the resumed id (`Carry`); the restore also replaces the queue object, so
    `Drain` reads `KernelState.Queue` again at every action and never holds one across a body. A canceled action does
    nothing, as in the game: the queue ran no body for it, and the old step is not run in its place (the step returns
    -1, nothing changed); the same for an executor that is paused (the action waits) and for an answer to a kernel
    screen (the hook action's resume record ran). The old step still checks the win inside itself, so the host's
    boundary check is empty until F10 moves it.
  - **Plain-data pauses.** `HookStatus` keeps `PassSnapshot` (the pass's listeners per part built, and the paused
    index), never the cursor; `<Name>Resume` calls `rt.Reopen` (generated), which rebuilds the cursor from one fresh
    listing (handlers and membership) in the snapshot's order: a listener gone is skipped, one added is not visited.
    Limits: the combat guard is read again at the fresh listing, and orb and deck change detection restarts there.
  - **Continuation across the answer.** (1) Per-listener: `OnPaused` records the hook action and enqueues a Hook action
    of type Combat; at a `wait_all` point `WaitAll` runs the dispatch's hook actions in id order, after every later
    listener; elsewhere the executor reaches it after the current action, opens its screen
    (`PendingSelection.KernelAction`) and the answer (`KernelState.Answer`, a new id) runs the resume record directly.
    (2) Inside old code that cannot resume (a drain at a wait point, a `none`/`passed`/`passed-unpushed`/`caller-hook`
    pause through `KernelState.Settle`): the answer is the step's keyed answer, or the step stops with the screen open
    (`PendingSelection.HookKey`, `HookChoicePending`) and its answer is stored keyed (`HookChoiceReplay.Keyed`, key =
    dispatcher, listener, `HookResume.ResumeSite`, occurrence; left out of the state walk, the core2 writer's) while old
    screens keep their order; the replay defers
    and drains as the first run did. (3) The turn start: `KernelState.TurnStart(ITurnStartFlow)` runs the setup to its
    pause, AfterSideTurnStart, the orb queue's turn start, then enqueues the flow's hook action (CombatPlayPhaseOnly) and
    checks the win condition (`ITurnStartFlow.CheckWin`, F10's behaviour, as CombatManager's StartTurn does before the
    executor is unpaused); only then the answer, the rest of the setup (`ResumeSetup`), AfterAutoPrePlayPhaseEntered and
    the boundary win check. A death before or during AfterSideTurnStart or the orb start that ends the combat calls
    `CombatEnded`, which cancels the pending choice: the choice, the rest of the setup and AfterAutoPrePlayPhaseEntered do
    not run. Test: `ADeathInAfterSideTurnStartEndsTheCombatAndCancelsTheTurnStartsChoice`.
  - **Entry gate.** `RunEngine.BindActiveCombat` and `BindCombat` bind a combat built outside the engine (the native
    run's anchor; OldReanchor binds the combat it anchors, right after the anchor); `RunEngine.StepCombat` is the gate of
    a combat stepped on its own: with no switch it is `CombatEngine.Step`, with one the combat must carry its KernelState
    (it throws otherwise) and the step goes through the queue as a run's combat step does (the action's body the combat
    engine's step). OldReanchor's every step (the action, the answers to its screens, the finish of a stopped step) goes
    through it, so with a switch on (`reanchor --engine core2`) it steps only bound combats and no step skips the queue
    (`ReanchorKernelGateTests`); OldHookProbe binds its scenario combats to the engine's kernel and `EmuDump kernel-order`
    takes `--engine`; the native run API refuses a switched-on kernel; `RunEngine.Step` throws when a switch is on and
    the combat about to be stepped has no KernelState. A test plays a whole run with a test switch on beside the old
    engine: the same masks, results and HP at every step, a KernelState on every combat step.
  - **G2891c25e0d** (a dead enemy at a mid-fight card selection): the game removes it inside
    `CreatureCmd.KillWithoutCheckingWinCondition` (AfterDeath, a per-listener dispatch that does not wait, then
    `CombatManager.RemoveCreature` and `CombatState.RemoveCreature`), so before any selection that follows in the same
    action and before an AfterDeath listener's own screen (its hook action runs after the action). The removal has two
    conditions, and the kill-site fix must keep both: it needs `ShouldCreatureBeRemovedFromCombatAfterDeath` to say so
    (the creature's model answers it), and `CombatState.RemoveCreature` is skipped while the monster `IsPerformingMove`
    (the move's own end removes it). The old engine
    answers and sweeps deaths after the card's effects (`HandleEnemyDeaths` after `CardEffects.Apply`,
    `AnswerDeathsSince` at the play's end), so a selection raised inside the effects stops the step with the corpse
    still in the roster. Test: `ADeadEnemyLeavesTheCombatBeforeAHookActionsScreenOpens`. The fix belongs to the kill
    site's switch (the `MarkDying` mark and a removal inside the kill).
  - **Plain data in a pause.** `HookStep.Request` (`object?`) and the `HookResume` records are shared by reference when
    a state is copied (`CopyTo`), so neither holds a live object: `Core2/HookPlain.cs` rejects the game's state types
    (`CombatState`, `EnemyState`, `CardInstance`, `RunState`, the runtime, `ChoiceCtx`, delegates), any reference type that
    is not a record, and collections of them; allowed are primitives, strings, enums, structs and records of the same
    (a member typed `object` or an interface is judged by its value). `HookStep.Paused` checks it in a debug build
    (`HookPlainTests` in every build).
  - A first family switch still needs: its bridges read into `SPLITS` (all 16 families are one group today); its old
    sites gated with `Kernel.If`, the marks placed, and `IKernelScreen` requests for its screens; F10 moving the win
    check to the host and calling `CombatEnded` at the game's point; the core2 writer (K5) writing the queue and the two
    screen fields.

## 6. Core2 as an engine, its writer, and the fuzzer

- `Core2Engine : IEngine` (tools/emudump/Engines/Core2/) is the old engine with the kernel and the switched-on set;
  `EmuDump --engine old|core2` picks it per process. Its native writer is the old engine's game-schema binding
  (`NetState`) plus the kernel's own state (queue, pending hook actions) only when a switch is on.
- Built in K5 (settled): `Core2Engine.GameView()` is old's view plus, only when a switch is on (the combat's or run's
  `KernelState` exists), the kernel's own state, written by `tools/emudump/Engines/Core2/Core2GameView.cs`. The game's
  NetFullCombatState has no field for a queue or a choice set the kernel keeps (its action fields are `lastExecutedActionId`,
  `lastExecutedHookId` and `nextChoiceIds`, the last being PlayerChoiceSynchronizer's counter), so: `net.last_action` and
  `net.last_hook` carry the first two under the game's names, from the queue's own counters (`ActionQueueSet.LastExecutedActionId/HookId`: the
  queue's own count and `RecordedHookAction.Id`, 1-based, where the game's HookId is `_nextHookId`'s, 0-based; so neither matches a
  record and the comparer lists both as surplus; set when an
  action finishes, a resumed action's last id), and everything else is `private.kernel`: `queue` (every action with id, kind,
  type, state, owner, first id, hook action, step action and target, answer; per queue the front-first ids, paused and the
  three cancel flags; next id), `hook_actions` (pause site: dispatcher, hook, pass, index, listener kind/model/instance,
  context; the screen request as plain data; the resume record's type and site), `stops`, `waits`, `screen` (the combat's open
  screen when the kernel opened it: its hook action id or its answer key), `keyed_answers` (the open screen's replay record),
  `answer_sequence` (`KernelState.AnswerSequence`), `turn_start_flow` (the flow's CombatPlayPhaseOnly hook action pending), the
  switches on, `executor_paused`, `win_checks`. The block is plain data read from the state alone, so a copy of the state writes
  the same block (test). With no switch on nothing is added: byte_proof old against `--engine core2` is 0 in every mode,
  `game-view` and `game-view-netstate` included. The comparer lists these as a surplus (docs/rebuild/COMPARER.md).
- The fuzzer (`tools/fidelity/fuzz_core2.py` driving `EmuDump fuzz`): setup fights built from models whose switches
  are on (random decks, relics, potions, encounters and legal actions, seeded), run on both engines, compared by state
  hash per step. Every difference is a listed `hook_open.txt` row the switch fixes, or goes to a decompile reading or a
  lab recording the same day. With no switch on it must find none.
- Built in K6 (`python -m tools.fidelity.fuzz_core2`, `EmuDump fuzz <specs.json> [--fields]`, `tools/fidelity/fuzz_explained.json`):
  - A spec is a seeded setup record (`relics_set`, `deck_set`, `potions_set`, `hp`, the `fight` item and an `act` item for acts
    2 and 3; no map room: the first option is taken and written out) built from the engine's own tables (`EmuDump tables` and
    `setupinfo`, no game files): 137 Ironclad, colourless and A10 cards, 119 relics (those whose pickup opens no screen), 48
    potions and the 80 encounters the old engine builds. A switch set keeps a model when its combat families (the port
    manifest's) are all on, an encounter when each of its monsters is (rosters read once from step 0 of each encounter). No
    switch draws everything. Spec i of seed S depends on (S, i) alone.
  - `EmuDump fuzz` plays each spec with a seeded player (SplitMix64) over `IEngine.LegalActions` (plays with their targets, end
    turn, potions; screens answered one pick or the decline at a time through the replay's own `Fight.ApplyChoice`) and writes
    the actions, the answers and `StateHash` per step. The same spec on `--engine old` and `--engine core2` plays the same
    fight while the states agree, so the first step whose action or hash differs is the first difference. A difference is
    re-run through `EmuDump fight --state-hash=fields` on its record cut at that step (saved under the rebuild folder's `fuzz`,
    a record `EmuDump fight` replays on either engine); the differing fields make its signature.
  - A test switch set is a scratch copy of the EmuDump build whose `migrated_families.json` holds it. `--switches F08,F10` is
    refused while the sites form one group (K3: F01-F16 switch together; the run says so and exits 2), so the no-handler
    sanity set is `--switches all`.
  - Triage: `fuzz_explained.json` maps a signature pattern to the `hook_open.txt` row (which must still be there) or the
    migration doc item that explains it; a difference matching none makes the run exit 1.
  - Sanity: no switch, 3000 fights (55,637 steps): 0 differences; `--switches all` (no handler ported), 2500 fights (56,182
    steps): 0 differences. About 9 fights per second per process (each fight runs on both engines), three processes.

## 7. Rounds and gates

K1 descriptor, extractor, generator and `construct shape` (Opus) | K2 listener order and its tests (Opus) | K2b card
uid under the byte-identity gate (Opus design, Sonnet audit of creation sites) | K3 site inventory, handler table,
`Hooks` runtime, switches, `--engine core2` with no switch byte-identical (Opus) | K4 queue, executor and pauses (Opus)
| K5 Core2Engine and writer (Sonnet) | K6 fuzzer (Sonnet). Every kernel round keeps the old engine's gates unchanged (C#
and EmuDump tests, corpus 0 regressed, Codex walk 0 worse) and core2's no-switch identity on the corpus. The
construction checks (`construct shape/commands/pauses/rng`) and `port_check` are built in P0 and run on the kernel;
`commands`, `pauses` and `rng` have only the kernel's own items until families migrate.

## 8. Risks

- Old state lacks identity for power instances and order for folded powers: the mapping reports what it cannot
  represent; a family that needs it adds it under the byte-identity gate, as K2b does for cards.
- Hybrid soundness: a migrated family's behaviour can re-enter through an unmigrated one (a handler calls an old
  command that dispatches by hand). Measured by per-family divergence and the transfer gap, not assumed; PLAN 8.3's
  escalation rule applies.
- Replay as the continuation inside old callers keeps its cost for those screens; order is the queue's either way.
- Escalation to Tier 2: handlers are keyed by game ModelId, card handles are opaque, resume state is a typed record per
  site, so Tier 2's state, arena and resume points replace Tier 1's without changing handler shapes.

## 9. Review K0 (Opus, 2026-10-23): CHANGES, all taken

1 descriptor shapes (cap, run-time passes, changed criteria, first-true predicates, partner receivers, in-place
mutation, unpushed contexts, two guards): section 2. 2 listener order (run+combat deck-first and late build,
IsActiveForHooks at visit, badges and scaling, every non-player creature lists its monster): section 3. 3 cross-family
seams: site inventory and switch groups, the guard reads old's flag, no queue or kernel state with no switch: section 1,
5, 6. 4 pauses (later listeners before the screen in wait-all passes, the turn-start flow pause, CombatEnded cancel,
new id on resume, type rules, mixed answers): section 5. 5 card identity is a blocker: K2b card uid before any switch
with card listeners: section 3. 6 escalation: ModelId keys, opaque handles, typed resume records: sections 3, 4, 8.
