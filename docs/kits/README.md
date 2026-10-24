# Fix kits (Lane B, cloud pipeline)

One kit per divergence group of the agent-weighted ranking (`agent/gen0/rank-s2-m12`, block S2, generation 0) or per audit finding,
written on the PC from the lockstep records, the emulator source and the game's own code (described by class and method names, never
pasted). A cloud session implements a batch of kits against unit tests on a branch `cloud/<family>-<n>` with a PR to `main`; the PC
session gates (corpus, byte proof, EmuDump) and lands commit by commit. Rules: `CLAUDE.md`, `../../docs/CLOUD_CREDIT_PLAN.md` in Spireline.

## Batches for the cloud (2026-10-24)

Status 2026-10-24 evening: `cloud/run-flow-1` (PR 7) and `cloud/combat-core-1` (PR 9) are landed on main (`26c19ef2`), each after a local review and the full gate (corpus 0 regressed, 2 improved). Landed locally the same day, from the probes: the lone-End-Turn mechanism (potions the game cannot use were legal combat actions; `PotionUsage` table), the Neow Small Capsule claim, the multi-pick and chest comparer artefacts, and the ranking's combat-type and emulator-arm join fixes.

| batch | kits | code | confidence |
|---|---|---|---|
| `cloud/run-flow-1` | `G74caa1c59d` (owner) with `G8318fb1cf5`, `G88f7b7316f`, `Gb1aa8d4875`, `G5b5244da1e` (same mechanism: a Thieving Hopper steal leaves the run deck one card too large until the fight ends); `Gbc5227da53` (Fairy in a Bottle does not prevent death from non-combat HP loss) | C#, run layer (`Core/Run`), `tools/emudump/RunSnap.cs` | high; medium-high |
| `cloud/combat-core-1` | `Bolas` (the return-to-hand queue invents a card), `HeldPlayedCard` (a stale held card is filed twice; probe-first, test 1 is deterministic), `ShivDamage` (alias table: option A, landed with before/after counts) | C# combat engine outside `Core2/`; Python tooling | medium-high; low-medium; medium-high |

Probe-first, not yet batched (the eight synthetic trace fights of 2026-10-24 showed no difference for the F08 candidates C, D and E in isolation; the exact-fight trace tool is the week-4 item): `Gc44f78e430` (player hp at the reward screen after an enemy-phase fight end), `G1162c5de18` (re-run two seeds
first; likely the residue of a fixed lockstep artefact), `Gda7c3abc9d` (the emulator did not enter the chest room; probe).

## Local fixes (not cloud work): `local/`

Measurement artefacts of the lockstep comparer in `referee/lockstep.py`, fixed on the PC and re-measured there:
`local/G0a127979b6.md` (multi-pick card screens compared one pick ahead: 19 groups, G0a127979b6, G2955289b06, G04d74da38a, Gfb4c7f4acc,
G5120acf48c, G6f0314da18, Gd703a4790b, Gf65e1cf550, G3c94e2c2de, Gbdae48596f, Gc981582989, G7f549a3298, G87f40418b7, Gb46e127f40,
Gb8fd5e1c79, G9656df94b7, G15f476f926, G47d6f08135, Gf9721aeb9d) and `local/G96df219a44.md` with `G2fc811be54`, `Gac0a0b8a7e` (the chest
question compared after the emulator's auto-take of the relic).

Excluded from Lane B: `G2891c25e0d` and the F08 kill-timing mechanism (the rebuild lane's F08 migration, `docs/rebuild/MIGRATION_F08.md`).
