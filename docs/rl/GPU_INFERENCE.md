# The batched GPU inference server for self-play (GPU lane, D27)

Design, 2026-10-25, revised the same day after the independent review (section 12). Status words: **measured** (run
and timed), **estimated** (reasoned), **decided** (the owner, the orchestrator or the review said so). The acceptance
numbers in section 9 are filled in when they are measured; until then this is a design, not a result.

## 0. Summary and the honest ceiling

Self-play is the generation loop's bottleneck. Every one of the lab's 16 self-play worker processes plays whole runs
on its own emulator and, at each combat decision, runs `live_search.search_live`, which asks the generation's combat
prior (`combat_prior.LivePrior`) for a probability vector at the root and at every tree node reaching
`prior_min_visits` (3). Today that prior is a 2-layer, 128-wide transformer forward on the worker's own CPU, one
torch thread, one position at a time.

**What the prior costs today (measured, week-3 A2c report, 2026-10-24, machine at 100% load):**

| quantity | value |
|---|---|
| prior calls per simulation | 0.22 |
| tokens per call | ~24 |
| forward on the CPU | 3.1-3.4 ms per call (the review's stand-in: 0.78 ms at 4 tokens, 1.35 at 24, 3.54 at 96 on an idle thread, so about half is compute and half dispatch) |
| encode + clone per call | 0.18 + 0.02 ms |
| plain search (no prior) | 541 sims per CPU-second (1.85 ms per simulation) |
| search with the prior | 375-400 sims per CPU-second |
| prior arm against no-prior arm, CPU time | 1.38x-1.44x |
| per-position cache hit rate (a cache was not built) | 1.8-4.0% |

So the prior adds about 0.22 x 3.5 ms = 0.77 ms of worker CPU per simulation on top of 1.85 ms. **Removing all of
it would make the prior arm 1.38x faster at most.** The server takes the forward off the worker and leaves about
0.3 ms of worker CPU per call (encode, clone, pack, two pipe syscalls, unpack, the float64 softmax), **but the
server's own CPU shares the same 16 logical CPUs and counts.** What it costs decides everything:

- **An eager torch forward on CUDA costs 3.5-3.6 ms of host CPU per batch at batch 1-4 and 2.7-2.9 ms at 16-64
  (measured by the review on this machine under load, a stand-in shaped like `CombatNet`).** That is as much CPU as
  today's forward: the server would serve ~280 batches per second, batches would have to grow to ~6 to meet demand,
  a call would wait 4-7 ms, and the job would lose about 15% at 16 processes (estimated). **So the server runs the
  forward as a captured CUDA graph per shape bucket, replayed: 0.7-1.0 ms of host CPU per batch including the copies
  (measured by the review).** The go/no-go of commit 1 is that number on the real checkpoint (section 9.0).
- Demand is bounded by the CPUs, not by the worker count: 16 logical CPUs x 520 sims/s x 0.22 = ~1,830 calls per
  second at best. With graph replay at mean batch 4-6 the server spends ~0.15-0.25 ms of CPU per call, 0.3-0.45 of a
  logical CPU in all (estimated).
- **Ceiling with graphs: about +25-30% decisions per second on the self-play job (estimated).** Per generation that is
  only +8-10% while the promotion arms (2 x 2,000 prior-arm runs) and the score's emulator arm (500) stay on the CPU
  path; **the promotion arms move to the server once section 9.3 passes (decided, review answer 2), both arms on the
  same transport within one comparison**, and then the lever covers ~6,000 of the ~6,500 prior-arm runs a generation
  plays. The score's emulator arm and the lockstep stay on the CPU path.

Two further effects, both measured in section 9:

- **Wall-clock needs more workers.** While a worker waits for the server its logical CPU idles. At 16 workers on 16
  logical CPUs (8 cores, SMT) the freed CPU is only partly recovered by the SMT sibling. The self-play job therefore
  runs `--server-procs` (20-24, section 9.1 picks) workers with the server and `--procs` (16) without; one emulator
  per process as before. Workers on the server path do not import torch (the saving is measured in 9.1, not assumed).
- **Per-call latency should not get worse.** The CPU forward takes 3.3 ms of CPU, 3.5-6 ms of wall on a saturated
  machine. The server round trip is a pipe hop, the queue (no timed wait, section 4), and one graph replay: about
  1.5-3 ms estimated, measured as p50/p99 in 9.1.

## 1. What runs today (facts the design rests on)

- `rl/gen0.py emu` (the lab's `selfplay` job) plays seeds on a `multiprocessing` spawn pool of `--procs`
  single-emulator processes (`run_block`, `maxtasksperchild=20`, workers at BelowNormal through
  `procutil.lower_priority`). Each task is `play_emu_run(seed, ...)`:
  `search_live(root.handle, **cfg.search_kwargs(...) | agent.search_extra())`.
- `gen_loop.Agent.search_extra()` returns the champion's `{"prior_fn": LivePrior(policy_path, probe), "c_puct",
  "prior_min_visits"}` (`combat_prior.find`, cached per process). `LivePrior.__call__(handle, legal)` clones the
  handle to read `obs`, encodes `combat_tokens.encode(handle, obs, encounter_id, probe)`, then
  `PolicyNet.weights([tokens], [legal])[0]`: the policy logits at the action slots, a float64 softmax over the legal
  actions only, uniform when no legal action has a finite logit; `live_search._prior_weights` renormalises in float64.
- The search calls the prior at the root (`_root_priors`, on a resampled probe fork) and at a node on its
  `prior_min_visits`-th visit (`_node_prior`). Calls are sequential inside a search; nothing in `live_search` batches.
- `model.json` carries `policy_sha256` and `vocab_sha256`; a worker refuses an export this build cannot play. The
  card attribute table (`combat_tokens.card_attribute_table`) is not covered by `vocab_sha256`.
- The run record schema (`run_train.shards.RUN_RECORD_SCHEMA`) has `additionalProperties: false` at the top level;
  five test modules validate records against it.
- The lab's `train` job runs under the GPU venv (Python 3.12.15, torch 2.14.1+cu132, confirmed from
  dist-info; `torch.cuda.is_available()` True on the GeForce RTX 3080, 10 GB). The emulator venv has torch
  2.12.0+cpu. The GPU idles between train jobs (point readings 8-11% utilisation, under 1 GB used by the desktop).
- Recorded self-play shards keep the root's token encoding of every sampled decision (`TOKEN_KEYS`), so the GPU
  path can be compared with the CPU path offline on real positions without an emulator.
- Hard rules: Windows-safe process code (no POSIX-only APIs; one emulator per process; spawn pools; priority classes),
  no network from game processes, results and logs free of host names, user names and absolute user paths (the
  interpreters live under the user profile, so logs carry exception type and message, never tracebacks).

## 2. Architecture

```
gen0.py emu (job main process, emulator venv)
  |-- starts: prior_server.py  (venv-gpu python, one process, owns the GPU model as CUDA graphs)
  |     named pipe \\.\pipe\spireline-prior-<pid>-<nonce>, local clients only, per-job authkey
  |     (AF_UNIX socket off Windows, for the tests)
  |-- spawn pool of P workers (emulator venv, no torch)
        each worker: RemotePrior(handle, legal) -> encode -> request -> 32 logits -> float64 softmax over legal
                     on any failure: LivePrior (in-process CPU prior) for the rest of the process
```

**One server per self-play job, owned by the job.** `gen0.py emu --prior-server auto` starts it, waits for its
`READY` line (every bucket's graph captured and warmed before it prints), hands the endpoint and authkey to the
`Agent`, and stops it (close stdin, then terminate after 5 s) when the block ends. The server exits on its own when
its parent is gone (a handle to the parent process opened once at start, so a reused pid cannot fool it, plus stdin
EOF), when the kill file appears, or when any of its threads raises (`os._exit(1)`: fail-stop, never a half-alive
server). There is no idle exit while the parent lives. In the lab the job step's Job Object kills the server with
the step; the server never breaks away from it. The job's main process restarts a dead server once, on the same
endpoint; a second death leaves the block on the CPU path. A second self-play job (the promotion arms, a bench)
starts its own server; the GPU holds several (the review measured ~290 MiB of context plus ~134 MiB reserved per
process on the stand-in; the net is 2.5 MB).

**The server does one thing:** policy logits for batches of token sequences. It does not see the emulator, the
search, seeds or handles. It loads `combat_policy.pt` with the same `CombatNet` constructor the CPU path uses,
`eval()`, fp32, `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`,
`torch.backends.mha.set_fastpath_enabled(False)` (the fused eval-mode encoder path is not capturable and not the
reference numerics), `torch.set_num_threads(1)` and `OMP_NUM_THREADS=1` (no NORMAL-priority thread may spin). It
uses `CombatNet.policy_logits` (A2c, `7b190cb0`) when the build has it and the forward's second output otherwise;
the two are `torch.equal`. The server imports `combat_tokens` for its constants and the card attribute table (which
needs `STS2_LIB_PATH`, as the `train` job already sets it). `--device cuda` is the only served device; without CUDA
the server refuses to start. `--device cpu` exists for the tests alone (section 6 says why).

**The client (`rl/prior_client.py`) imports numpy only.** `RemotePrior` is a frozen dataclass like `LivePrior`
(policy path, probe flag, endpoint, authkey, policy sha256) so it pickles into the workers through
`Agent.search_extra()`. A per-process connection is opened on the first call and reused; `maxtasksperchild=20`
reconnects every 20 runs. The client never calls `recv()` (which unpickles); frames travel as bytes.

**Threads in the server.** One acceptor; one thread per connection that owns that connection end to end
(`recv_bytes`, parse, enqueue, wait on its request's `Event`, `send_bytes`): a `PipeConnection` is used by one thread
only. One batcher thread drains the queue, pads into the bucket's static input tensors, replays the graph, copies the
logits back into pinned memory with a non-blocking copy and waits on a blocking `torch.cuda.Event` (the default sync
spins and charges GPU time as CPU), then sets the requests' events. The softmax is not the server's: it returns
logits.

**Priority.** The server runs at NORMAL priority class (flag `--priority normal|below`, default normal) because
20-24 BelowNormal workers wait on it: a BelowNormal server would round-robin with them. It is safe because it blocks
when idle, its torch threads are pinned to one, its sync blocks instead of spinning, and the lab's user-input guard
pauses the job (and so the server) whenever the owner uses the PC. Workers stay BelowNormal.

**Value head.** The live search evaluates leaves with rollouts and never reads the value head (A2c-target report);
the macro policy's V_run lookahead runs in numpy outside fights. The protocol reserves a flags bit for "value too",
unused this week.

## 3. Protocol

Binary, little-endian, one request and one reply per call, framed by `multiprocessing.connection` (`send_bytes`,
`recv_bytes`; AF_PIPE on Windows, AF_UNIX elsewhere, so the tests run on Linux too). Numbers are `struct` fields.

```
AUTH    multiprocessing.connection's hmac challenge with the job's authkey (bytes, no pickle), both ways,
        before anything else; a client that fails it is dropped.
HELLO   (client -> server, once per connection)
  magic "SPR1", kind=1 (u16), proto=1 (u16), policy_sha256 (64 ascii), vocab_sha256 (64 ascii),
  card_table_sha256 (64 ascii: sha256 of card_attribute_table()'s bytes and vocabulary size)
HELLO-OK / HELLO-NO (server -> client)
  magic, kind=1, status (u16: 0 ok, 1 policy mismatch, 2 vocab mismatch, 3 proto mismatch, 4 card table
  mismatch), reason (utf-8, rest)

PRIOR   (client -> server)
  magic, kind=2, req_id (u32), n_tokens T (u16), flags (u16: bit0 "value too", unused)
  types   u8[T]        (TYPE_* fit in a byte; N_TYPES = 17)
  ids     i32[T]
  owners  u8[T]        (0 the player, 1..MAX_ENEMIES)
  feats   f32[T*18]    (N_FEATURES = 18)
REPLY   (server -> client)
  magic, kind=2, status (u16: 0 ok, 1 bad request), req_id (u32)
  logits  f32[32]      (the policy head at the MAX_ACTIONS slots; -inf where no action token)

BYE     (client -> server) magic, kind=3  (optional; closing the pipe is enough)
```

A request is ~24 x (1 + 4 + 1 + 72) + header ~ 1.9 KB, a reply 140 bytes. The `req_id` lets a client detect a
reply that is not for its outstanding request (never expected; treated as a transport error). Anything the server
cannot parse is answered with status 1 and the connection closed; the client falls back.

**The reply carries logits, not probabilities, and the client turns them into probabilities with the same float64
helper `PolicyNet.weights` uses** (lifted out of `PolicyNet` into `prior_protocol.legal_weights(row, legal)`: the
legal slots, `-inf` elsewhere, softmax after subtracting the finite maximum, uniform when none is finite). So the
two paths differ only by where the logits were computed, a batch of one on the CPU device is bit-exact with
`LivePrior`, and the numpy work is off the server's GIL.

**Local clients only.** On Windows the listener is a `PipeListener` subclass whose pipe mode adds
`PIPE_REJECT_REMOTE_CLIENTS` (the standard-library listener does not set it, so a pipe is reachable over the network
by default); the pipe name carries a nonce; and the authkey is a fresh 32-byte secret per job that travels to the
workers inside the pickled `Agent`, never in a file or on the command line.

## 4. Batching policy

The batcher loop:

1. Block until the queue has one request (no GPU work on an empty queue).
2. Drain everything already queued, up to `--max-batch` (default 128). **No timed wait:** `--deadline-us` defaults
   to 0 and stays 0 on Windows, where Python 3.12 rounds lock timeouts up to whole milliseconds and wakes on the
   15.6 ms clock tick, so a 500 us wait would cost up to ~16 ms. Queueing during the replay is the batching.
3. Pick the bucket: batch size in {1, 2, 4, 8, 16, 32, 64, 128}, tokens rounded up to a multiple of 8 (bounded by
   the encoder's longest sequence; the buckets are enumerated at start and each gets one captured graph). Spare rows
   are filled with copies of a real row of the batch, never all-PAD rows (an all-PAD row has an all-masked attention
   and produces NaNs that the fused kernels handle differently from the reference). Copy into the bucket's static
   input tensors (pinned host staging, non-blocking copies), replay, copy the logits back, wait on a blocking event.
4. Answer every request of the batch (set each request's event; its connection thread sends); go to 1.

**Why no deadline is enough.** With arrivals at rate r and the replay taking f, the batch size settles at about
1 + r x (f + overhead) once the server is busy. At the demand bound of ~1,830 calls per second and f + overhead ~
0.8-1.0 ms, batches of 2-3 are typical and the GPU is far from saturated (a replay of 128 rows costs little more
than of 2). The server records histograms of batch sizes and of queue wait, replay and reply times (section 9).

**Capacity (estimated from the review's measurement; the go/no-go of section 9.0 measures it):** 0.7-1.0 ms of host
CPU per replayed batch gives ~1,000-1,400 batches per second, which at a mean batch of 2-3 is 2,000-4,000 calls per
second against a demand bound of ~1,830. If `--selftest` shows more than 1.2 ms of host CPU per replayed batch at
batch <= 8, or a capacity under twice the demand bound (~3,660 calls per second), the server does not proceed to plumbing
and the lane reports why (the criterion of section 9.0).

## 5. Numerics and determinism

**The CPU path is bit-identical and replayable:** torch CPU fp32, one thread, one unpadded sequence; the A2c
branch's tests (`7b190cb0`, on `w3/a2c`, not yet on main at the time of writing) pin the lean forward to the padded
one (`torch.equal`) and a fixed-seed search to its old result.

**The GPU path is not bit-identical to the CPU path,** and a row's logits can differ in the last bits between two
buckets (cuBLAS picks kernels and reduction orders by shape). TF32 is off, so the differences are fp32 rounding, not
10-bit mantissas. **Tolerance, gated on logits (review answer 3): max |logit_gpu - logit_cpu| <= 1e-4 over the action
slots (a TF32 tripwire), and zero argmax flips among positions whose top-2 logit gap exceeds 1e-5.** A 99.9%
argmax-agreement rule was dropped: it would pass a bug that flips one clear decision in a thousand. Section 9.2
measures the distribution on 10,000 recorded positions.

**What a last-bit difference does to the search.** PUCT compares scores; a prior that differs by 1e-6 flips a choice
only where two scores tie to that precision. Over the ~10^5 simulations of a run some flips may happen, so **a run
played on the server may not be replayable bit-for-bit from its seed on the CPU path.** Section 9.3 measures how
often: the CPU path is deterministic per seed, so the fraction of seeds whose server run is identical decision for
decision, and the first diverging decision of the rest, are direct measurements. Replays for diagnosis
(`gen0.play_emu_run(seed)`, `a3_devset`, `show_sample`) use the CPU path, check `prior_path`, and are exact for runs
the record tags `cpu`.

**Recording which path produced a sample (decided here, reviewed):**

- `RemotePrior` counts calls answered by the server and by the fallback in the worker process; `play_emu_run` reads
  the counters before and after the run and writes `row["prior_path"]`: `"cpu"`, `"server"`, `"mixed"`, or
  `"none"` (no prior), with the two counts. The run record (`RunRecorder.finish`) carries the same field, **and
  `RUN_RECORD_SCHEMA` gains it in the same commit** (the schema forbids unknown top-level properties).
- The combat parts (npz) carry `prior_path` per run (an int8 code per run, aligned with `seeds`), so
  `build_training_set` can exclude runs by path (`exclude_prior_paths`, default none). That is the way back if a
  server bug pollutes a training window.
- `ShardSink` adds `prior_paths` (counts of runs by path) to `shards.json`; `run_block`'s `summary.json` carries the
  same counts, the client's p50/p99 call latency, the server's stats file (batch histogram, replay and queue times,
  restarts) and the endpoint kind (`"pipe"`), never the endpoint name or the authkey.
- The promotion judge does not read the field. **This week the promotion arms run the CPU path** (`gen0 promote`
  takes no `--prior-server` until section 9.3 passes); after that both arms of a comparison run on the same transport.
- The score's emulator arm (`gen0 score`) and the lockstep's `search_kwargs(env_prior=True)` never receive a
  `RemotePrior` (a test pins it).

## 6. Failure modes and the kill switch

| event | what happens | recorded as |
|---|---|---|
| no server (`--prior-server off`, or `auto` with no GPU python / no CUDA / kill file present / server failed to start twice) | the job runs the CPU path at `--procs` workers (never `--server-procs` workers each loading torch) | `prior_path cpu` |
| `--prior-server require` and the server does not start | the job exits non-zero before playing | job failure |
| connect refused, auth failed or `HELLO` rejected (sha, vocab, card table, proto) | that worker process uses `LivePrior` for the rest of its life; one line on stderr | `mixed` or `cpu` |
| transport error mid-run (EOF, bad frame, wrong `req_id`) | the current call and all later calls of that process fall back; the search goes on | `mixed` |
| reply later than `--client-timeout-s` (5 s) | same as a transport error (the search has a 120 s budget per decision) | `mixed` |
| slow but live server | the client's p50/p99 latency goes into `summary.json`; p99 above `--latency-p99-ms` (20) marks the block `prior_server_degraded` | `server`, flagged |
| server thread exception, CUDA error, OOM | `os._exit(1)`; every worker falls back at its next call; the job's main process restarts the server once on the same endpoint (new workers connect; fallen-back workers stay on the CPU path until recycled) | `mixed` |
| parent job dies | server exits within ~1 s (parent handle signalled, or stdin EOF); the Job Object kills it anyway | - |
| kill file `<lab>/GPU_OFF` appears | server exits within 1 s; workers fall back; `auto` resolves to `off` for new jobs | `mixed`, then `cpu` |
| `SPIRELINE_PRIOR_SERVER=off` in the environment | overrides any flag to `off` | `cpu` |
| the owner uses the PC | the lab's guards pause self-play as today; the server idles (blocked, no spin) | - |

Fallback is per process and sticky: a worker that fell back once does not retry (`maxtasksperchild=20` already
renews every worker about every 5-6 minutes), so a flapping server cannot cost a reconnect per call. A block with
under 99% `server` runs under `auto`, or with p99 latency over the bound, is reported in the lab's `result.json` as
`prior_server_degraded`.

**Why `--device cpu` is test-only.** A CPU-device server is a net loss (its cost grows with rows, one batcher serves a
few hundred calls per second) while every run would be tagged `server`: a silent degradation. The served device is
CUDA or nothing.

The kill switch is a file because every other lab control is one (`PAUSE`, `HALT`, `main.lock`) and anyone,
including the owner from Explorer, can write it. Never kill the server by image name (it is one more
`python.exe`); the kill file or closing the job is the way.

## 7. Configuration

- `gen0.py emu|promote|bench-prior --prior-server off|auto|require` (default `off`: nothing changes until the lab
  config asks). `--gpu-python <path>` (or `SPIRELINE_GPU_PYTHON`) names the interpreter that runs the server; without
  it `auto` means `off` with one log line. `--server-procs N` is the worker count on the server path (`--procs` on
  the CPU path). `--prior-server-args "..."` passes batching flags through for benches.
- `gen_loop.Agent.prior_server: str | None` (the endpoint) and `prior_authkey: bytes | None`. `search_extra()` wraps
  the champion's `LivePrior` in a `RemotePrior` when set. `Agent.describe()` is unchanged (the transport is not part
  of the agent's identity).
- The lab's `jobs.json` `selfplay` command gains `--prior-server auto --gpu-python <venv-gpu python> --server-procs
  N` **only after the section 9.4 block**. `train` is unchanged. The promotion arms (`train --promote`, through
  `run_promote_subprocess`) gain the same flags only after 9.3 passes.
- `rl/prior_server.py --policy <pt> --endpoint <name> --authkey-fd|--authkey-env --device cuda --max-batch 128
  --deadline-us 0 --parent-pid N --kill-file <path> --priority normal|below --stats <json path>`; prints
  `READY <endpoint>` on stdout once every bucket is captured and warmed.
- `rl/prior_server.py --selftest N --clients C` runs synthetic clients against itself and prints calls/s, batches/s,
  host CPU per batch, batch histogram and latency percentiles: the go/no-go of section 9.0.

## 8. Alternatives considered

| option | why not |
|---|---|
| TorchScript / ONNX Runtime on the CPU inside each worker | changes numerics anyway, keeps 16 forwards on 16 CPUs; at best removes the dispatch half of a 1.4x cost (~+15%) |
| an eager GPU forward in the server | 3.5 ms of host CPU per batch on this machine: a net loss at 16 processes (review, finding 1) |
| a CUDA context in every worker (the worker venv on CUDA) | 16-24 contexts, batch size 1, context thrash; the workers would also have to run under venv-gpu |
| batching inside one worker (lockstep simulations, virtual loss) | changes the search, not bit-identical even on the CPU path; a search-quality question for Lane A |
| a per-search or cross-search cache keyed by tokens | measured hit rate 1.8-4.0% |
| a C# port of the net inside the emulator | large, and the net changes every generation |
| nothing | the prior arm stays at 1.4x of the no-prior arm on the bottleneck job |

## 9. Acceptance (measured; filled in before any lab switch)

All measurements on **one fixed seed list for every row** (the A2c bench's `seeds[:2P]` played different seeds per
process count), the current `agent/current.json` prior, search config `a3-v1`, the machine's load noted, the lab
paused and `QUIET.lock` held for the wall-clock rows (coordinated with the week-3 session: after 15:45Z on
2026-10-24 or any time on the 25th, a couple of hours between blocks). Results hold no host names, user names or
absolute user paths.

### Commands for the lane's window

Git Bash, lab paused and `QUIET.lock` held (section 9.4's rule). `<repo>` is the worktree of the commit under test, `<lab>`
the lab folder, `<gen dir>` the champion's generation folder, `<emulator python>` and `<gpu python>` the two venvs' interpreters,
`<native lib dir>` a built native library; `OUT` is any scratch folder outside the repo; `N` is the server count under test.

```
cd <repo>
export STS2_LIB_PATH=<native lib dir> STS2_ALLOW_STALE_NATIVE=1
PY=<emulator python>; GPU=<gpu python>; LAB=<lab>; GEN=<gen dir>; OUT=<scratch folder>

# 9.1 throughput rows (ABAB twice, one fixed seed list)
$PY rl/prior_bench.py throughput --seeds-file referee/seeds/E0-2026-10-23.json --seeds 48 \
    --rows "noprior:16,cpu:16,server:16,server:20,server:24" --decisions 100 --repeat 2 \
    --gpu-python $GPU --json $OUT/throughput.json

# 9.1 full-run A/B, N = the server row with the best wall decisions per second (15 minutes per arm plus the tails)
$PY rl/prior_bench.py fullrun --stream prior-bench --n 4000 --block-min 15 --procs 16 --server-procs N \
    --gpu-python $GPU --out $OUT/fullrun --json $OUT/fullrun.json

# 9.1 RAM case (the server is killed after 60 s, and its restart once more, so no worker finds a server again)
$PY rl/prior_bench.py ram --stream prior-bench --n 4000 --procs N --block-min 4 --after-s 60 \
    --gpu-python $GPU --out $OUT/ram --json $OUT/ram.json

# 9.3 sample quality on a fresh 1,000-seed block (a name no block has used)
$PY rl/gen0.py seeds --name Q93-<date> --n 1000 --out $OUT/Q93.json
$PY rl/prior_bench.py quality --seeds-file $OUT/Q93.json --seeds 1000 --procs 16 --server-procs N \
    --gpu-python $GPU --out $OUT/quality --json $OUT/quality.json

# 9.2 tolerance (any time, the GPU idles; about a minute): CPU logits from the workers' torch, the server from the GPU venv
$PY  rl/prior_tolerance.py --shards-glob "$LAB/runs/*-selfplay/shards/combat-g3-*.npz" \
    --policy $GEN/combat_policy.pt --n 10000 --dump-cpu $OUT/cpu.npy
$GPU rl/prior_tolerance.py --shards-glob "$LAB/runs/*-selfplay/shards/combat-g3-*.npz" \
    --policy $GEN/combat_policy.pt --n 10000 --cpu-logits $OUT/cpu.npy --json $OUT/tolerance.json
```

Each command prints its markdown table and writes numbers only to its JSON (no paths). The arithmetic is tested by
`$PY -m unittest tests/python/test_prior_bench.py`; a smoke run of every subcommand (2 workers, 2 seeds) is how the tools were
checked.

### 9.0 Go/no-go (commit 1, before plumbing): measured, GO

`--selftest` on the real checkpoint (generation 3, candidate-r1), measured by the commit-1 worker at ~15:50Z on 2026-10-24 at
machine load 22-70%. The first draft's line "1,000 batches per second" was a proxy for the quantity that matters (calls the
server answers per second); it held only at a mean batch of 2-3 and is replaced. **Go: <= 1.2 ms host CPU per replayed batch at
batch <= 8, and a calls-per-second capacity of at least twice the demand bound (~1,830 calls/s, section 0).**

| quantity | measured |
|---|---|
| calls/s, 16 synthetic clients at saturation | 4,898 (mean batch 8, 1.30 ms of whole-process host CPU per batch, client p50 / p99 3.0 / 4.5 ms) |
| calls/s, 8 clients | 4,134 (0.89 ms per batch) |
| calls/s, 4 clients | 2,820 (0.48 ms per batch) |
| graph-replay host CPU per batch | 0.09-0.20 ms at batch 1, 0.19-0.20 at 4, 0.25-0.27 at 8, 0.20-0.41 at 32 |
| eager forward, host CPU per batch (for comparison) | 2.5-3.4 ms |
| time to READY, warm, 256 buckets | 12.7-13.7 s (5.4-6.3 s of it capture) |
| server RSS | 1,245 MiB |
| GPU memory reserved | 2,216 MiB at 256 buckets, 76 MiB at 48 |

Capacity is 1.3 times what the doubled demand bound asks (4,898 against 3,660 calls/s) and replay CPU at batch <= 8 is at least 4 times under the
1.2 ms line, so the server proceeds. With the default `--max-tokens 64` (64 buckets) the server's own clock read 6.2-6.5 s to
READY in the commit-3 runs of 9.2 and the smoke tests (machine load 50-70%); the lane re-measures READY in 9.1. A cold start
(first venv-gpu torch and CUDA import after a reboot) was not measured here.

### 9.1 Throughput

*Measured in the lane's window.*

| arm | procs | dec/s (wall, total) | dec per CPU-s incl. server | sims per CPU-s incl. server | server CPU share | server calls/s | mean batch | p50 / p99 call latency | tasks tagged server |
|---|---|---|---|---|---|---|---|---|---|
| no prior | 16 | | | | - | - | - | - | - |
| prior, CPU path (today) | 16 | | | | - | - | - | - | - |
| prior, server | 16 | | | | | | | | |
| prior, server | 20 | | | | | | | | |
| prior, server | 24 | | | | | | | | |

Rules: the server's process CPU (from its stats file) goes into every CPU-time denominator; the arms run interleaved
ABAB twice (the A2c no-prior arm drifted 1-5% between runs); the deciding number is wall decisions per second with
the lab paused. **Plus a full-run A/B:** `gen0 emu --block-min 15` on the CPU path at 16 processes against the server
at the best server count, same seed stream: runs per hour, decisions per hour, peak commit charge (reference: block
`T0432` gave 1,115 runs and 118 dec/s at 16 processes). **Plus RAM:** peak commit at the server count with the
server killed mid-block (every worker falls back and loads torch). **Stop rule:** if the best server row is under
+10% wall decisions per second over the CPU-path row at 16 processes, the server does not go to `jobs.json` and the
lane reports why (the doc keeps the table).

How the tool measures (`rl/prior_bench.py throughput`): every row plays the same `--seeds` prefix of one seed file, each seed to
`--decisions` combat decisions, on a fresh spawn pool; every worker runs one warm task (imports, model load, the connection)
behind a barrier before the timed block. The server arms start one server per row (`require`) and stop it after the row. The
server's CPU is the difference of its stats file's `host_cpu_s` between the snapshot taken once it has counted exactly the
warm-up calls and the file it writes at exit; server calls/s and mean batch come from the same two files. The "no prior" arm is
the same agent without its combat prior. `load` is the OS processor load before and after a row and labels it, nothing more.
The timed block ends when the last task ends, so a row with few tasks per worker carries a tail: 48 seeds are 2 seeds per
worker on average at 24 workers (3 at 16). A server row whose tasks are not all tagged `server`, or whose server-CPU
baseline was not exact, prints a warning line under the table.

### 9.2 Tolerance: measured, PASS (commit 3, `rl/prior_tolerance.py`)

10,000 recorded positions (seed 20261025) drawn without replacement from the newest 11 combat-g3 parts of the lab's
self-play shards (a 41,956-position pool; 16-58 tokens, mean 26.3), generation 3's `combat_policy.pt`. CPU logits: `PolicyNet`'s
path (one unpadded sequence, `policy_logits`, one thread) under the emulator venv's torch 2.12.0+cpu, the torch the workers run
(under the GPU venv's torch 2.14.1 on the CPU the maximum is the same 1.91e-06). GPU logits: a real server over its pipe on
the RTX 3080 (torch 2.14.1+cu132, TF32 off, 64-token graphs, no position over 64 tokens), asked twice: by 16 client threads
(1,261 batches of 1-12 rows, mean 7.93: mixed rows, padded token buckets) and one position at a time (10,000 batches of one).
Differences are taken at the action slots; "gap" is the CPU path's top-2 logit gap.

| | batched, 16 client threads | alone, bucket of one |
|---|---|---|
| max abs dlogit | 1.91e-06 | 1.91e-06 |
| 99.9th percentile abs dlogit (61,937 action slots; the batched cell read 1.31e-06 in one of three runs, batch composition follows timing) | 1.43e-06 | 1.43e-06 |
| mean of the per-position maximum | 4.86e-07 | 4.90e-07 |
| slots finite in one path only | 0 | 0 |
| argmax flips, positions with gap > 1e-5, all action slots | 0 of 8,336 | 0 of 8,336 |
| argmax flips, same, legal slots only | 0 of 8,319 | 0 of 8,319 |
| flips among the 1,664 positions with no gap over 1e-5 (ties or near-ties, not counted) | 0 | 0 |
| float64 `legal_weights`: max abs dp / 99.9th percentile | 4.19e-07 / 2.81e-07 | 4.72e-07 / 2.87e-07 |
| float64 probability argmax flips (gap > 1e-5) | 0 of 8,319 | 0 of 8,319 |

The server against itself (batched against alone, the bucket-to-bucket spread) is at most 1.19e-06. The gate of section 5
(max abs dlogit <= 1e-4, no flip above a 1e-5 gap) holds with a factor 52 to spare, so TF32 or a wrong kernel would show; the
last-bit differences are the fp32 rounding section 5 expects, and no position is decided differently by the bucket it fell in.

### 9.3 Sample quality (the transport must not be a promotable change)

*Measured in the lane's window.*

Two arms on one fresh 1,000-seed block, same agent and config, one on the CPU path at 16 processes and one on the
server at its process count:

1. **Identity:** the fraction of seeds whose two runs are identical decision for decision (the CPU path is
   deterministic per seed), and for the rest the first diverging combat decision (its index and fight).
2. **Equivalence (pre-registered; two one-sided tests at 5%):** the transport passes when (a) the two-sided 90%
   cluster-bootstrap interval of the paired mean floor difference (server minus CPU path, resampling seeds, 10,000 draws)
   lies within +-0.30 floor (0.29 is the smallest gain the judge has ever promoted, so a transport effect inside +-0.3 is
   invisible to it), (b) the two-sided exact McNemar test on Act-1 boss kills is not significant at 0.05, (c) sims per
   decision and decisions per run are each within 2%, (d) the server arm is at least 99% `server` runs and the CPU arm 100%
   `cpu` runs (otherwise a dead server makes two CPU arms look equivalent), and (e) no pair is lost to a crash or a missing
   run (a crash on one side is a finding, not a dropped pair). The point estimate, its standard error, the 95% interval, the
   identical-seed fraction and the first-diverging-floor histogram are reported as information, with no threshold.
   (`compare_arms` gives only
   the judge's one-sided lower bound, which is not the statistic here.)

How the tool measures (`rl/prior_bench.py quality`): both arms play with `record=True`, and identity is reported on three levels:
the run's row (`decisions`, `combat_decisions`, `sims`, `floor`, `end`), the RunRecord (its `decisions` and `fights`, wall-clock
fields left out) and the search itself (a digest of each combat sample's tokens, root visit counts and simulation count, in
order). The first diverging floor is that of the first search that differs, else of the first differing record entry. The
statistics are the ones above: the 90% interval the rule reads, the 95% interval, the standard error and the point estimate for
information, the two-sided exact McNemar on Act-1 boss kills, the two ratios, and the verdict on conditions (a) to (e).

### 9.4 One generation through the judge

A 2,000-run self-play block on the server, recorded into the lab's shards with the lab paused for the block
(`LAB/PAUSE` carrying `gpu-lane <UTC>` and `QUIET.lock` with the owner and UTC line, written and removed by this lane;
**if another lane's PAUSE or QUIET.lock exists, wait**), the `train` job picking it up at the next ready marker, and
the promotion judge judging the generation normally on its fresh E-block. **Pass for the block itself** (the judge's
verdict says nothing about the transport): 0 crashes attributed to the prior, at least 99% `server` runs, runs per
hour at or above the 9.1 full-run A/B, p99 latency under the bound. If the block fails, its runs are excluded from
training windows by `prior_path` (section 5) and the lane reports. Only after a pass does `jobs.json` switch.

### 9.5 Gate

The Python suites (no C# changes: the .NET suites and the corpus baseline do not move; if any C# is touched, both
.NET suites run), landed on fork main commit by commit per HANDOFF section 9 (`main.lock`, `merge --ff-only`,
push once green). `lint-and-test.sh` is never run by Claude.

## 10. Rollout (commits, each reviewed by one Sonnet reviewer before landing)

1. `rl/prior_protocol.py` (frames, `legal_weights`, the card-table hash), `rl/prior_server.py` (CUDA graphs per
   bucket, connection threads, batcher, fail-stop, parent watch, kill file, stats, `--selftest`),
   `rl/prior_client.py` (`RemotePrior`, sticky fallback, counters, latency), tests on the CPU device over
   AF_PIPE/AF_UNIX: a batch of one is bit-exact with `LivePrior`, a padded bucket within 1e-6 on the CPU device;
   fallback on no server, on a killed server, on a bad authkey, on each `HELLO` mismatch, on the kill file;
   parent-death exit; a thread exception exits the process. **Then the 9.0 go/no-go on the GPU.** Nothing uses it
   yet: no behaviour change.
2. Plumbing: `Agent.prior_server`/`prior_authkey`, `gen0.py --prior-server/--gpu-python/--server-procs`, server
   start, restart and stop around `run_block`, `prior_path` in rows, records (**with the schema edit**), npz parts,
   `shards.json` and `summary.json`, `build_training_set(exclude_prior_paths=...)`, the 9.1 stats plumbing. Default
   `off`: no behaviour change. Tests: a 2-seed block through a CPU-device server started with the emulator venv's
   python as `--gpu-python` reproduces the CPU path exactly (decisions, sims, floors) and tags `server`; the score
   arm and the lockstep never see a `RemotePrior`.
3. `rl/prior_bench.py` (fixed seeds, ABAB, server CPU in the denominators, the full-run A/B, the RAM case) and the
   measured sections 9.0-9.3 written into this doc; a GPU test (skipped without CUDA) that pins 9.2's logit tolerance
   on the fixture prior.
4. The 9.4 block, then the `jobs.json` change (lab config, outside the repo), the promotion arms on the server, and
   a HANDOFF line.

Budget (the lane's cap is 50 reviewed rounds this week, raised from 25 on 2026-10-24): design and review ~2.7
Opus-equivalent rounds; commits 1-3 ~7 Sonnet rounds with reviews and amends; 4 ~1. Piece 1 ~11.

## 11. Questions the review answered (decided)

1. NORMAL priority for the server: yes, required (a BelowNormal server round-robins with the workers); safe with one
   torch thread, a blocking sync and the Job Object.
2. Promotion arms: on the server once 9.3 passes, both arms on the same transport; score's emulator arm and the
   lockstep stay on the CPU path.
3. Tolerance: on logits, max |dlogit| <= 1e-4 and zero argmax flips above a 1e-5 gap; no percentage rule.
4. Sticky fallback: yes; plus one server restart by the job.
5. Bucketing: kept; CUDA graphs now, not later.

## 12. Review log

- **2026-10-25, independent Opus review of the first draft: CHANGES.** One blocker (the eager forward's host CPU,
  measured on this machine), eight required findings and smalls, all applied above: CUDA graphs per bucket and the
  `--selftest` go/no-go before plumbing (sections 0, 2, 4, 9.0); the server's CPU in the ceiling and in every
  measurement, demand bounded by CPUs, the per-generation reading and the promotion arms' move (0, 9.1); logits in
  the reply and the float64 helper on the client (3); connection threads own their pipes, no timed wait on Windows,
  one torch thread, authkey and local-only pipes, no `recv()` (2, 3, 4); fail-stop, parent handle plus stdin, no idle
  exit, one restart, latency-based degradation, separate process counts, CPU device test-only (2, 6, 7); the schema
  edit, `prior_path` in the npz parts with an exclude switch, the exact-reproduction test (5, 10); the identity and
  equivalence statistics of 9.3; pass criteria and exclusion for 9.4; the smalls (replay tools check the path, the
  score arm and lockstep test, logs without tracebacks, real-row padding, the card-table hash, `policy_logits`,
  warm every bucket before `READY`). Unsupported "measured" claims of the draft were reworded or dropped (CUDA
  context size, Python per call, torch import cost, GPU point readings, "dispatch-bound").
- **2026-10-25, commit 3 (tools, measured 9.0 and 9.2):** `rl/prior_bench.py` (throughput, fullrun, ram, quality),
  `rl/prior_tolerance.py`, `tests/python/test_prior_bench.py`. 9.0 restated in calls per second and filled from commit 1's
  selftest; 9.2 measured on the GPU and passes with a factor 52 to spare (max abs dlogit 1.91e-06, 0 flips of 8,336). 9.1 and
  9.3 wait for the lane's window; their commands are in section 9.
