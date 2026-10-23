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
    ...). The extractor does not write these.
- `tools/kernel/hooks_extract.py` extends `tools/fidelity/hook_table.dispatchers` to read every field above except
  `args` from the local decompile. `python -m tools.kernel.construct shape` diffs the checked-in descriptor against a
  fresh extraction both ways and checks that `Hooks.g.cs` equals a fresh generation. A dispatcher the extractor
  cannot classify is an error, never a default. Each mode has a broken case it catches.
- `tools/kernel/gen_hooks.py` writes `src/Sts2Emulator/Core2/Hooks.g.cs` and `HookId.g.cs` (committed; generated from
  our descriptor, not from game text): one static method per dispatcher, a fast path when no switched-on listener is
  registered for its passes, otherwise per pass a snapshot of listeners in game order and the loop of its op.
- Named steps are hand-written in `Core2/Steps.cs`, one function each, cited `game: Hook.<Method>`.

## 3. Listener order over the old state

`Core2/Listeners.cs` builds one pass's snapshot as the game's `CombatState.IterateHookListeners` does:
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
  combat. Activity is read at the entry's turn (CreatureCmd deactivates hooks mid-pass). A listener added mid-pass is
  not visited.
- **Card identity first** (K2b, under the byte-identity gate): a per-combat card uid on `CardInstance`, left out of
  equality, the snapshot writer and the state walk; assigned when a card enters the combat (deck clone at combat
  start, generated or copied cards as created: a copy gets a new uid, a `with` change keeps it). A `CardHandle` is that
  uid, opaque to handlers, so Tier 2's card arena can replace it. An invariant (no two cards share a uid; every card in a
  pile has one) runs on the corpus and the fuzzer. Until K2b lands, no switch with card listeners turns on.
- Old-state mapping (`Core2/OldState.cs`, the only kernel file that knows the old representation): powers through
  `BuffId` to the game's power model (`tools/fidelity/power_aliases.json`), engine-only bookkeeping buffs left out, one
  game power split over several `BuffId`s visited once at its first entry; relics, potions, orbs, piles as the old
  lists hold them; the old engine has no Play pile, so the card in flight is its one entry for the length of its play.
  What the old state cannot represent (folded power instances, melted relics) is listed in `Core2/NotModelled.cs`; a
  pass that would need it reports NotModelled, never a guess.
- Tests (`src/Sts2Emulator.Tests/Core2/ListenerOrderTests.cs`), each from the game's rule: creature order with pets and
  several enemies; powers before the monster, on enemies and pets; relic obtain order and its reverse; potions by slot
  with gaps; pile order with the affliction and enchantment after each card; modifiers, badges, scaling last;
  run+combat deck-first with the combat list built after; a power removed mid-pass still called; a creature removed
  mid-pass not called; a deactivated player's listeners skipped from that entry on; a listener added mid-pass not
  called; the guard read once; agreement with `hook_table.game_order_key` on the `hook_order` scenarios.

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
- `passed` and `passed-unpushed` contexts (a choice inside a card's own play or a potion) pause the action that owns
  the context, as the game's player choice context does, and keep the old replay.
- Tests from the game's `ActionExecutor`, `ActionQueueSet` and `HookPlayerChoiceContext`: the boundary win check and
  its two exempt kinds; canceled fronts skipped; a paused action yields to the next ready action; a resumed action's new
  id; type rules; `CombatEnded` cancelling unstarted hook actions; later listeners before the screen in BeforeTurnEnd,
  AfterTurnEnd and BeforeFlush; the turn-start flow pause; clone parity of the queue.

## 6. Core2 as an engine, its writer, and the fuzzer

- `Core2Engine : IEngine` (tools/emudump/Engines/Core2/) is the old engine with the kernel and the switched-on set;
  `EmuDump --engine old|core2` picks it per process. Its native writer is the old engine's game-schema binding
  (`NetState`) plus the kernel's own state (queue, pending hook actions) only when a switch is on.
- The fuzzer (`tools/fidelity/fuzz_core2.py` driving `EmuDump fuzz`): setup fights built from models whose switches
  are on (random decks, relics, potions, encounters and legal actions, seeded), run on both engines, compared by state
  hash per step. Every difference is a listed `hook_open.txt` row the switch fixes, or goes to a decompile reading or a
  lab recording the same day. With no switch on it must find none.

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
