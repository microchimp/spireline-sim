# Building and testing this fork in a hosted Linux session

What it took to build the fork and run its unit suites in a Claude Code cloud session
(Ubuntu 24.04, x86_64, 4 cores, 15 GB RAM, root shell, outbound HTTPS through an allow-list
proxy), measured on `main` at `4a4b15f`. CI (`.github/workflows/ci.yml`) runs the same steps
on Windows with the .NET 9 SDK, Python 3.14 and uv; nothing below changes a test, the lock or
the project files.

## Setup

Already present in the image: `uv 0.11.32`, system CPython 3.13, `clang 18.1.3`, `gcc 13.3`,
`zlib1g-dev`, `libicu74`, `libssl3t64`. Missing: any .NET SDK, Python 3.14.

### .NET 9 SDK (about 1 minute)

The proxy denies Microsoft's download hosts (`builds.dotnet.microsoft.com`,
`download.visualstudio.microsoft.com`, `dotnetcli.azureedge.net`, `aka.ms`, `ci.dot.net`), so
`dotnet-install.sh` cannot run. Ubuntu 24.04's archive carries .NET 8 and .NET 10 but not 9,
Microsoft's apt feed for 24.04 (`packages.microsoft.com`) carries no .NET at all, and Canonical's
`dotnet/backports` PPA host is denied. `mcr.microsoft.com` is allowed, so the SDK comes out of
the official `dotnet/sdk:9.0-noble` image, pulled with the registry API and no container runtime:

```bash
mkdir -p layers rootfs
# 1. manifest list -> linux/amd64 image manifest
curl -sS -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
  -o index.json https://mcr.microsoft.com/v2/dotnet/sdk/manifests/9.0-noble
DIGEST=$(python3 -I -c "import json;d=json.load(open('index.json'));print([m['digest'] for m in d['manifests'] if m['platform']['os']=='linux' and m['platform']['architecture']=='amd64'][0])")
curl -sS -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  -o manifest.json "https://mcr.microsoft.com/v2/dotnet/sdk/manifests/$DIGEST"
# 2. every layer, checked against its digest
for dig in $(python3 -I -c "import json;d=json.load(open('manifest.json'));[print(l['digest']) for l in d['layers']]"); do
  curl -sSL -o "layers/${dig#sha256:}.blob" "https://mcr.microsoft.com/v2/dotnet/sdk/blobs/$dig"
  echo "${dig#sha256:}  layers/${dig#sha256:}.blob" | sha256sum -c -
done
# 3. only the dotnet tree, in layer order, then into place
for dig in $(python3 -I -c "import json;d=json.load(open('manifest.json'));[print(l['digest'][7:]) for l in d['layers']]"); do
  tar -xzf "layers/$dig.blob" -C rootfs --wildcards 'usr/share/dotnet*' 2>/dev/null || true
done
mv rootfs/usr/share/dotnet /usr/share/dotnet
ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
dotnet --version   # 9.0.319, runtime 9.0.21; 11 layers, 328 MB downloaded, 578 MB on disk
```

`api.nuget.org` is allowed, so `dotnet tool restore` (csharpier 1.2.6, 4 s) and package restore
work unchanged. The image's `libicu74` and `libssl3t64` are what the runtime needs.

### Python 3.14 and the locked environment (about 2.5 minutes)

`pypi.org`, `files.pythonhosted.org` and GitHub release downloads (uv's managed CPython) are
allowed.

```bash
uv python install 3.14            # cpython-3.14.6, 34 MB, 3 s
uv sync --locked --python 3.14    # 59 packages, 2 min 9 s; the dev group is on by default
```

Every pin in `uv.lock` has a Linux wheel, so the lock needed no change. One difference from the
PC to know about: on Linux the lock resolves PyPI's `torch 2.12.0+cu130`, which drags in the
CUDA 13 companions (`nvidia-*`, `cuda-*`, `triton`): 3.5 GB of the 4.8 GB `.venv`, and about
2.2 GB of the download. There is no GPU, `torch.cuda.is_available()` is `False`, and the suites
below do not care. A CPU-only resolution would need the `download.pytorch.org/whl/cpu` index,
which the proxy denies, and a lock change; neither was done.

### Native library for the gym tests (30 seconds)

`src/sts2_gym/native.py` loads `out/Sts2Emulator.so` on Linux, so the Windows publish step of
`lint-and-test.sh` becomes:

```bash
dotnet publish src/Sts2Emulator/Sts2Emulator.csproj -c Release -r linux-x64 --self-contained -o out
# 28 s: out/Sts2Emulator.so (10.2 MB, 92 exported entry points) + Sts2Emulator.so.dbg
```

NativeAOT needs `clang` and `zlib1g-dev`, both in the image. `scripts/build.sh` runs the same
command and picks the runtime from `uname`.

All of this is lost when the session's container is reclaimed; a SessionStart hook or a setup
script is the natural next step.

## Build and test results

Every project the task names builds; `dotnet build` is the default Debug configuration, as
`dotnet test` uses it. Wall time is `time` around the command, on an otherwise idle container.

| Step | Result | Wall time |
|---|---|---|
| `dotnet build src/Sts2Emulator.Tests/Sts2Emulator.Tests.csproj` | 0 errors, 24 warnings (CS8602 ×8, IL2070 ×4, CS8604 ×4, CS8669, CS8629, CS8509, CS0649 ×2 each) | 44 s incl. restore |
| `dotnet build tools/emudump.Tests/EmuDump.Tests.csproj` | 0 errors, 5 warnings (CS7022 ×4, CS8629) | 8 s (emulator already built) |
| `dotnet publish src/Sts2Emulator -c Release -r linux-x64` | `out/Sts2Emulator.so` | 28 s |
| `dotnet build referee/driver/SpirelineReferee.csproj` | 362 × CS0246: `HarmonyLib`, `MegaCrit.*`, Godot types not found | 2 s |

The referee driver is a mod the game loads: it references `sts2.dll`, `GodotSharp.dll` and
`0Harmony.dll` from the user's game copy (`-p:GameData=...`), none of which may be in this repo.
It is out of scope for the cloud, as `FORK.md` implies, and stays with the PC.

| Suite | Total | Passed | Failed | Skipped | Wall time |
|---|---|---|---|---|---|
| `dotnet test src/Sts2Emulator.Tests` (xunit) | 6871 | 6864 | 0 | 7 | 1 min 25 s (xunit: 1 min 20 s) |
| `dotnet test tools/emudump.Tests` (xunit) | 281 | 281 | 0 | 0 | 18 s (xunit: 16 s) |
| `uv run python -m unittest tests/python/test_sts2_gym.py` | 43 | 43 | 0 | 0 | 0.8 s |
| `uv run python -m unittest tests/python/test_sweep_buff_map.py` | 24 | 22 | 0 | 2 | 0.3 s |
| `uv run python -m unittest tests/python/test_trace_buff_slots.py` | 4 | 4 | 0 | 0 | 0.3 s |
| `uv run python -m unittest tests/python/test_observation_layout.py` | 17 | 17 | 0 | 0 | 0.4 s |

No test was changed. The `dotnet test` runs used `--no-build` after the builds above; with the
build folded in, the full build-and-test pass is about 3 minutes.

## Failures, classified

**Real defects: none.** Every suite is green.

**Missing local data: 2 skips** in `test_sweep_buff_map.py`, both `skipTest("decompiled sources
not present")`: they read the user's decompiled power classes from the gitignored `decompiled/`
tree, which `CLAUDE.md` keeps out of the repo. They skip on any machine without that tree.

**Linux-only: none** among the tests. The 7 xunit skips are static `Skip` attributes and skip
the same way on Windows: four `audit:` tests whose engine fixes are not merged (Fairy in a
Bottle ×2, Tender's Strength return, Pen Nib), one `audit:` Ashwater selection case, one open
Osty/Knockdown case, and the `StepCostProbe` performance probe that is run by name only.

## The lint steps, for completeness

`lint-and-test.sh` runs four read-only checks before building. They were run here the way the
script runs them (not the script itself). All four fail on this tree, and only a sliver of the
findings is Linux-specific:

| Check | Result | Linux-only part | Wall time |
|---|---|---|---|
| `uv run black . --check --target-version py314` | 236 of 342 files would be reformatted | none: the diffs re-wrap long lines; the repo is LF throughout (`.gitattributes`: `eol=lf`) | 45 s |
| `uv run ruff check .` | 3243 errors (D401 518, D205 393, D209 337, ARG005 169, PLC0415 113, ...) | 33 × EXE001 (shebang on a mode-644 file; ruff does not enforce it on Windows) | 0.6 s |
| `uv run ty check . --error-on-warning` | 1279 diagnostics | 19 in five `referee/` and `tools/lab/` files: `ctypes.WinDLL`, `subprocess.CREATE_NO_WINDOW`, `BELOW_NORMAL_PRIORITY_CLASS`, Windows-only names with no `sys.platform` guard | 3 s |
| `dotnet csharpier check .` | 264 of 762 `.cs` files "not formatted" | none: real layout differences (e.g. one-line `foreach`/`if` bodies, `with` initialisers) | 10 s |

The pinned tool versions, rule sets and line endings are the same on both OSes, so the rest of
these findings would be reported on Windows too. Whether the PC's gate passes them today is not
visible from here.

Not exercised in this session: `scripts/check_run_traces.py`, `scripts/audit_powers.py`,
`rl/train.py --check --run-env` and `rl/evaluate.py` (the script's remaining steps). The first
two read local data (retained run traces, decompiled sources); the RL two need only `out/`.

## Verdict

A cloud implementer can rely on these suites. The fork builds on Linux with the same pinned
toolchain as CI (.NET 9 SDK, Python 3.14, the locked environment), and all six suites pass
without a single test change: 6871 + 281 xunit tests and 88 Python tests, with zero failures and
only the by-design skips. Setup is about 4 minutes and 5.5 GB per fresh container (the SDK from
the registry image, `uv sync`, one NativeAOT publish) and must be repeated each session until a
setup hook exists. The game-dependent gates stay on the PC as planned: the referee driver does
not build without the game's assemblies, and the decompiled-source and run-trace checks skip or
cannot run. The script's lint steps are not a usable pass/fail signal for a cloud change on
`main` as it stands, since they fail on both OSes; a cloud PR should be judged on the test
suites and on `black`/`csharpier` diffs limited to the files it touches.
