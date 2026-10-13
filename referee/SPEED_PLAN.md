<!-- Synthesized 2026-10-13 by a five-lens brainstorm workflow checked against the decompiled game; numbers after step 2 are estimates. -->

# Plan: 1000 verified fights/min on one PC

## Verdict

Speed work alone could plausibly get to 1000 **compared** fights/min. That only works if the **CPU** cost per fight falls about 6x, not just the wall time. Today 6 copies use 33% of 16 threads, so each copy keeps about 0.88 of a thread busy, which is about 3.4 thread-seconds per fight. At 1000/min on roughly 10 core-equivalents, the budget is about 0.6 core-seconds per fight.

Reaching 1000 **identical** (matching) fights/min also depends on how often the emulator matches. In night03, 157 of 294 fights matched (53%), so 1000 compared fights would give about 530 matching ones. The headline therefore has to change (last section).

## Conflicts settled (spot-checked in the decompile)

- **Fade:** Instant mode does not skip `NTransition.RoomFadeIn`. It sets the alpha to 0 first, then still builds and awaits the 0.8 s tween (NTransition.cs:276-314). The driver pays it twice per fight.
- **Checksums:** `ChecksumTracker` is already off in single player. RunManager.cs:392-400 turns it on only for net types 2-4. Turning it off saves nothing, but we can switch it on ourselves to get a per-step state hash for A/B checks.
- **GC and asset purge:** `UnloadMissedCacheAssets` and `GC.Collect` run even with `Enabled = false` (PreloadManager.cs:138 and :146; the `Enabled` check is only at :168).
- **Card tweens:** the exhaust tween (0.3 s, CardPileCmd.cs:577) and the Play-pile tween (0.25 s, :767) are awaited in Instant mode too (:623-626).
- **Discarded:**
  - NonInteractiveMode: it changes how `ActionExecutor.Pause` works, which can reorder game events.
  - `FightConsoleCmd`: it seeds from the wall clock (EncounterModel.cs:351-355), so fights are not reproducible.
  - Several combats in one process via separate assembly load contexts: no parallel gain, and it breaks the one-referee-per-process rule.
  - Forking fights through `LoadRun`: it calls `IncrementNumReloads` and costs as much as a run start.
  - Synthetic tier-B states and single-step tests: these change the unit of verification, so they are deferred.

## Throughput model (12 copies)

Ceiling = copies × 60 / seconds per fight. "Expected" assumes the process overhead drops from today's ~33% efficiency to about 85-90%.

| Step | start run | Neow | enter room | fight | s/fight | ceiling/min | expected/min |
|---|---|---|---|---|---|---|---|
| 0 today | 1.0 | 0.2 | 0.85 | 1.9 | 3.95 | 182 | 60 (measured) |
| 1 long-lived servers | 1.0 | 0.2 | 0.85 | 1.9 | 3.95 | 182 | ~160 |
| 2 setup cuts | 0.2 | 0.2 | 0.1 | 1.9 | 2.4 | 300 | ~260 |
| 3 frame pacing + flag-based settle | 0.15 | 0.05 | 0.05 | ~0.45 | 0.7 | 1030 by wall time; ~860 if CPU-bound (~10 copies) | 500-850 |
| 4 frame-free tweens | 0.15 | 0.03 | 0.03 | ~0.12 | 0.33 | ~1800 (10 copies) | 1000-1400 |
| 5 chain fights in one run | – | – | 0.05 | 0.12 | ~0.18 | ~3300 | CPU-bound, >1500 |

How the fight times were estimated:
- **Step 3:** 20 actions × about 3 frames × about 4 ms ≈ 0.24 s, plus about 0.2 s of remaining draw tweens.
- **Step 4:** 20 actions × 2-5 ms of CPU ≈ 0.04-0.1 s.
- **Caveat:** every number from step 3 on depends on how much work each frame does, which nobody has measured yet. The emulator side (a few seconds per 300 fights) stays negligible.

## Roadmap

**Pass gate for every step.** Replay the 294 night03 seeds with the old and the new method. Both of these must hold:
- The game's step arrays are byte-identical, ignoring timing fields, and so is a per-step hash: `ChecksumTracker.GenerateChecksum(NetFullCombatState.FromRun(...))`.
- The emulator still matches the same 157/294 fights.

**0. Instrument (2 h).**
- Log per phase: frame count (`Engine.GetProcessFrames`), the `TimeProcess` monitor, CPU time (`Process.TotalProcessorTime`), gen-2 GC count, the number of "Asset not cached" lines, and the current `OS.LowProcessorUsageModeSleepUsec`.
- Pass: two runs of the same 20 fights give the same hash sequence.
- Outcome: we learn the CPU seconds per fight, and whether frames are spent sleeping (step 3 then pays off) or working (only step 4 helps).

**1. Long-lived servers (6 h).**
- The driver loops on an inbox folder and writes one JSONL line per fight. This replaces `Save`, which rewrites every result after each fight (Driver.cs:51-60), so the cost grows quadratically.
- A dispatcher hands fights to whichever server is free, keeps a hot spare and re-queues work after a stall.
- Write `skip_intro_logo` and `fps_limit: 0` into the settings file, so boots skip the 5.6 s+ logo.
- Replace the per-copy `netstat` polling with one shared iphlpapi poller.
- Run every copy at BelowNormal priority.
- Pass: one boot covers 2,000 fights, per-fight time stays flat, and the same spec run as fight #1 and fight #1000 is byte-identical.
- **Status (branch `speed/servers`): built** as `driver/Server.cs` + `serve.py`; see RUNBOOK.md. Measured (3 boots each): boot to menu 10.0 s -> 1.6 s with `skip_intro_logo` + `fps_limit: 0` (NGame.cs:679, :745), whole empty session 11.0 s -> 2.6 s. One server, 40 fights (the proof40 specs): 40/40 results equal to runs mode, 4.06 s/fight. Four servers, 160 fights (proof40 x4, every spec on whichever server was free): 160/160 equal, 0 stalls, 56 fights/min (4.19 s/fight/server, so no loss from sharing the machine at 4 copies). Not yet shown: 2,000 fights in one boot.
- **Finding:** without the logo the menu's background "Common" asset session (686 assets) overlaps the first run, and with 4 copies booting together it hung 3 of 8 servers (native, after `texture_2d_initialize: Parameter t is null`). Fix: with `no_preload` the driver now switches the preloader off in its mod initializer, so that session is never queued (also ~2 s less CPU per boot). 6 servers x 3 boots x 2 fights each: 0 stalls.

**2. Setup cuts (3 h).**
- Harmony prefix on `RoomFadeIn`: set the overlay alpha to 0, set `InTransition = false`, return at once.
- Make `Load{Run,Act,Room}Assets` no-ops, so there is no asset purge and no forced GC.
- Turn off `CombatReplayWriter.IsEnabled` after each run is set up.
- Pass: start run and enter room each drop by at least 0.7 s, gen-2 GCs per fight are about 0, and memory levels off over 1,000 fights. Track the stall rate.

**3. Frame pacing and settle (4 h).**
- Set `OS.LowProcessorUsageModeSleepUsec = 0` (fallback: `--fixed-fps 60`).
- Set a high `Engine.TimeScale` and re-apply it every frame, except while `RollingBoulderPower` is present.
- Replace the 6-stable-frame `Idle()` with the game's own flags: executor idle, queue empty, `CombatState == PlayPhase`, `Phase == Play`, no end-of-turn phase running (or `CombatEnded` seen). Then wait one frame for the deferred `RecalculateCardValues`.
- Keep the old 6-frame check plus the hash as an audit on 5% of fights.
- Pass: the gate holds, with zero audit violations over 3,000 fights.

**4. Frame-free tweens (6 h).**
- Harmony prefix on `TweenHelper.AwaitFinishedInternal` that calls `FastForwardToCompletion`, then run the pending Godot continuations.
- Pass: a card play takes 0 frames and an end turn at most 1, the gate holds over 300 fights, there are no new exceptions, and CPU per fight is at most 0.4 s.

**5. Chain fights in one run (6 h).**
- Between fights, rewind with `SyncWithSerializedPlayer`, `RunRngSet.LoadFromSerializable` and a truncated map history, then re-enter with `EnterMapCoordDebug`.
- Start a fresh run every 50 fights. Move deck additions after Neow on both sides, and cover Neow with its own cheap job.
- Pass: chained fight k is byte-identical to a fresh run of the same spec, over 50 specs including deaths.

**6. In parallel, emulator side only (10 h).**
- Re-anchor each step after the first divergence: load the game's state into the emulator, apply the action, compare. Stop re-anchoring a fight once enemy intents differ.
- Build decks that exercise more non-starter cards.
- Pass: 100% of re-anchored steps match on fights that are already identical.

**Fallback (20 h).** A host with no game scenes at all, audited against the normal host on 5% of fights. Only worth doing if CPU per fight is still above 0.6 s after step 4.
- **Status (branch `speed/nodeless`): built** as `driver/Nodeless.cs` + `driver/NodelessPatches.cs`, job flag `"nodeless": true` (RUNBOOK.md). 300 night04 coverage fights, both hosts at once: 333 ms CPU and 148 frames per nodeless-hosted fight vs 2,913 ms and 339 frames on the full host (nodeless 142-191 ms with `"frame_sleep_usec": 0`); 518 ms per fight on average once the 14 fallback replays (4.7%) are counted. Results: 269 identical, 14 fallbacks replayed on the full host, 17 where the full host stalled on a turn-start autoplay and nodeless went on; no nodeless errors. 3 processes: 56 fights/min full host, 111 nodeless, 271 nodeless with frame sleep 0. 2,000 fights in one nodeless server: no errors, orphan count flat (1,081), private memory 345 -> 352 MB, every result identical to the same spec's first run.

## Top risks

1. **CPU is the real limit.** At 0.88 thread per copy, much of each frame may already be real work. If so, step 3 gains little and everything rests on step 4 (or the fallback).
2. **Snapshots taken too early.** A chain of game work that runs outside the action queue could make the flag-based settle snapshot an unsettled state, which corrupts results silently. The 5% audit and the Rolling Boulder exclusion guard this.
3. **Frame-free side effects.** Tween callbacks now run synchronously. Exceptions in them (for example the "Hand size 11" error from the hand-layout UI) and the branch where the card's node has already left the tree need checking.
4. **Leaks between chained fights.** Progress counters, act room-visit counters, extra run fields and the death path could carry state from one fight into the next.
5. **Stalls cost more at speed.** A 40 s stall on one of 12 copies costs about 50 fights at 1000/min. The hot spare and re-queue are required.

## Headline metric

The current "verified fights/min" counts every compared fight, diverging ones included (batch.py:196-202). Proposed headline:

- **Verified transitions/min (VT/min):** one VT is one step where the emulator reproduces the game's next state.
- **Fight-equivalents/min:** VT/min ÷ 18.9 (night03's average steps per fight).
- **Alongside:** the identical-fight rate, and new (card, situation) cells per hour.

Today that is 60 × 18.9 × 0.67 ≈ 760 VT/min, which is about 40 fight-equivalents/min, not 60. The goal of 1000 fight-equivalents/min means about 18,900 VT/min.

Speed (steps 1-5) is the main lever toward that. The rest has to come from emulator fixes and re-anchoring. Single-step tests are only worth doing if the speed steps stall below about 400/min.
