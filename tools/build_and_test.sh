#!/usr/bin/env bash
# Build and test this fork on Windows (Git Bash), offline against the NuGet cache restored and
# recorded for b745217 (Spireline security addendum 1, conditions F1-F3).
#
#   bash tools/build_and_test.sh <log dir> <python with the upstream lock installed>
#
# Writes 1-restore.log .. 5-corpus-check.log (3b-emudump-test.log: tools/emudump.Tests; 3c-emudump-build.log:
# the Release EmuDump build) and results.json to the log dir. The Release EmuDump is built once, before the Python
# suite: tests that drive the real EmuDump (tools/fidelity/fuzz_core2.py, tools/emudump/bin/Release/net9.0/EmuDump.dll)
# must not run a stale build, and the corpus check reuses that build. The Python suite
# is compared against the reviewed b745217 baseline: on Windows ~60 upstream tests fail on
# POSIX-only process code (os.nice, os.killpg, ...), so "pass" means no NEW failures.
#
# Exit status (also "gate" in results.json): 0 only when every step passed: restore, publish, the C#
# suites (src/Sts2Emulator.Tests and tools/emudump.Tests: each exit 0 and a summary with Failed: 0 and
# Total > 0), the Release EmuDump build (exit 0), the Python suite (it ran tests and has no
# failure beyond BASELINE_FAILURES, or exit 0 without one) and the corpus check (exit 0). 1 when any of
# them failed, 3 when none failed but the corpus check was skipped, which is never a push gate.
# tests/python/test_build_and_test.py runs this script against a fake toolchain, broken step by step.
set -u -o pipefail
LOG="${1:?log dir}"; PY="${2:?python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${NUGET_PACKAGES:-$HOME/.nuget/packages}"
mkdir -p "$LOG"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 PYTHONDONTWRITEBYTECODE=1
# NativeAOT finds the MSVC linker through vswhere, which the Visual Studio installer keeps
# outside Git Bash's PATH.
export PATH="$PATH:/c/Program Files (x86)/Microsoft Visual Studio/Installer"
unset SPIRE_COACH_REPO STS2_LIB_PATH STS2_ALLOW_STALE_NATIVE VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
cd "$ROOT"
for d in "$ROOT" "$ROOT/.." "$ROOT/../.." /d; do
  for f in Directory.Build.props Directory.Build.targets Directory.Packages.props nuget.config global.json; do
    [ -e "$d/$f" ] && { echo "refusing: found $d/$f (review E5)"; exit 2; }
  done
done
t0=$(date +%s)
# Restore from the local package cache only: no network source at all.
# Tests first: restoring the test project rewrites the library's assets without a RID, so the
# library is restored for win-x64 last, right before the NativeAOT publish needs it.
dotnet restore tools/emudump.Tests/EmuDump.Tests.csproj --source "$CACHE" > "$LOG/1-restore.log" 2>&1; r1c=$?
dotnet restore src/Sts2Emulator.Tests/Sts2Emulator.Tests.csproj --source "$CACHE" >> "$LOG/1-restore.log" 2>&1; r1b=$?
dotnet restore src/Sts2Emulator/Sts2Emulator.csproj -r win-x64 --source "$CACHE" >> "$LOG/1-restore.log" 2>&1; r1=$?
dotnet publish src/Sts2Emulator/Sts2Emulator.csproj -c Release -r win-x64 --self-contained --no-restore -o out/ \
  > "$LOG/2-publish.log" 2>&1; r2=$?
sha=""; [ -f out/Sts2Emulator.dll ] && sha=$(sha256sum out/Sts2Emulator.dll | cut -d' ' -f1)
dotnet test src/Sts2Emulator.Tests/ --no-restore > "$LOG/3-dotnet-test.log" 2>&1; r3=$?
# The EmuDump tools' own suite: the codex walker, the dump, the setup mirror, the state lifecycle.
dotnet test tools/emudump.Tests/ --no-restore > "$LOG/3b-emudump-test.log" 2>&1; r3b=$?
# The Release EmuDump, built once here: the Python suite (tests/python/test_fuzz_core2.py's RealEmuDump) and the
# corpus check (step 5) both drive tools/emudump/bin/Release/net9.0/EmuDump.dll, and neither may run a stale one.
dotnet build tools/emudump/EmuDump.csproj -c Release --source "$CACHE" > "$LOG/3c-emudump-build.log" 2>&1; r3c=$?
r4=skipped
if [ -n "$sha" ]; then
  # STS2_REQUIRE_AOT=1: the solver's AOT-against-JIT test (tests/python/test_solver_aot_sites.py) fails here, never skips.
  STS2_REQUIRE_AOT=1 "$PY" -m unittest discover -s tests/python > "$LOG/4-python-test.log" 2>&1; r4=$?
fi
cs=$(grep -Eo 'Failed:\s*[0-9]+, Passed:\s*[0-9]+, Skipped:\s*[0-9]+, Total:\s*[0-9]+' "$LOG/3-dotnet-test.log" | tail -1)
emu=$(grep -Eo 'Failed:\s*[0-9]+, Passed:\s*[0-9]+, Skipped:\s*[0-9]+, Total:\s*[0-9]+' "$LOG/3b-emudump-test.log" | tail -1)
pyline=$(grep -E '^(OK|FAILED)' "$LOG/4-python-test.log" 2>/dev/null | tail -1)
newfail=""
if [ -f "$LOG/4-python-test.log" ] && [ -n "${BASELINE_FAILURES:-}" ]; then
  grep -E '^(FAIL|ERROR): ' "$LOG/4-python-test.log" | sed -E 's/^(FAIL|ERROR): //' | tr -d '\r' | sort -u > "$LOG/python-failures.txt"
  if [ -f "$BASELINE_FAILURES" ]; then
    tr -d '\r' < "$BASELINE_FAILURES" | sort -u > "$LOG/python-baseline.txt"
    newfail=$(comm -23 "$LOG/python-failures.txt" "$LOG/python-baseline.txt" | wc -l | tr -d ' ')
  fi
fi
pyran=$(grep -E '^Ran [0-9]+ tests' "$LOG/4-python-test.log" 2>/dev/null | tail -1)
# The recorded-fight corpus (tools/fidelity/corpus.py, referee/RUNBOOK.md): every fight ever recorded
# in the game, replayed by this build and compared with the stored baseline; exit 1 lists the
# regressions or baseline fights it could not compare. Skipped when this PC has no corpus, or with
# CORPUS_CHECK=0 ("skipped", which RUNBOOK step 7 never accepts for a push).
r5=skipped; corpus_line=""
CORPUS_DIR="${SPIRELINE_CORPUS:-/d/spireline/referee/corpus}"
if [ "${CORPUS_CHECK:-1}" != 0 ] && [ -f "$CORPUS_DIR/index.json" ]; then
  if [ "$r3c" = 0 ]; then
    "$PY" tools/fidelity/corpus.py --corpus "$CORPUS_DIR" check --procs "${CORPUS_PROCS:-16}" > "$LOG/5-corpus-check.log" 2>&1; r5=$?
  else
    echo "not run: the Release EmuDump build failed (3c-emudump-build.log)" > "$LOG/5-corpus-check.log"; r5=build-failed
  fi
  corpus_line=$(grep -E '^check: ' "$LOG/5-corpus-check.log" | tail -1 | tr -d '"\\')
fi
# The verdict. A step whose output cannot be read is a failure, never a pass.
fails=()
{ [ "$r1" = 0 ] && [ "$r1b" = 0 ] && [ "$r1c" = 0 ]; } || fails+=("restore")
[ "$r2" = 0 ] && [ -n "$sha" ] || fails+=("publish")
cs_failed=$(echo "$cs" | grep -Eo 'Failed:\s*[0-9]+' | grep -Eo '[0-9]+')
cs_total=$(echo "$cs" | grep -Eo 'Total:\s*[0-9]+' | grep -Eo '[0-9]+')
{ [ "$r3" = 0 ] && [ "${cs_failed:-x}" = 0 ] && [ "${cs_total:-0}" -gt 0 ]; } || fails+=("csharp")
emu_failed=$(echo "$emu" | grep -Eo 'Failed:\s*[0-9]+' | grep -Eo '[0-9]+')
emu_total=$(echo "$emu" | grep -Eo 'Total:\s*[0-9]+' | grep -Eo '[0-9]+')
{ [ "$r3b" = 0 ] && [ "${emu_failed:-x}" = 0 ] && [ "${emu_total:-0}" -gt 0 ]; } || fails+=("emudump-tests")
[ "$r3c" = 0 ] || fails+=("emudump-build")
py_ran=$(echo "$pyran" | grep -Eo '[0-9]+' | head -1)
if [ "${py_ran:-0}" -eq 0 ] || [ -z "$pyline" ]; then
  fails+=("python (no test run recorded)")
elif [ -n "${BASELINE_FAILURES:-}" ]; then
  { [ -f "$BASELINE_FAILURES" ] && [ "$newfail" = 0 ]; } || fails+=("python (new failures: ${newfail:-baseline file missing})")
elif [ "$r4" != 0 ]; then
  fails+=("python (exit $r4, no BASELINE_FAILURES)")
fi
gate=pass; rc=0
case "$r5" in
  0) ;;
  skipped) gate=incomplete; rc=3 ;;
  *) fails+=("corpus") ;;
esac
if [ "${#fails[@]}" -gt 0 ]; then gate=fail; rc=1; fi
failed_json=$(printf '"%s",' "${fails[@]}"); failed_json="[${failed_json%,}]"
[ "${#fails[@]}" -gt 0 ] || failed_json="[]"
cat > "$LOG/results.json" <<EOF
{"commit": "$(git rev-parse HEAD 2>/dev/null)", "tree": "$(git rev-parse 'HEAD^{tree}' 2>/dev/null)", "dotnet_sdk": "$(dotnet --version)",
 "restore_exit": [$r1, $r1b, $r1c], "publish_exit": $r2, "native_lib_sha256": "$sha",
 "dotnet_test_exit": $r3, "tests_csharp": "$cs", "emudump_test_exit": $r3b, "tests_emudump": "$emu", "emudump_build_exit": $r3c,
 "python_test_exit": "$r4", "tests_python": "$pyran / $pyline", "python_new_failures_vs_baseline": "$newfail",
 "corpus_check_exit": "$r5", "corpus_check": "$corpus_line", "gate": "$gate", "gate_failures": $failed_json,
 "minutes": $(( ($(date +%s) - t0) / 60 ))}
EOF
cat "$LOG/results.json"
exit "$rc"
