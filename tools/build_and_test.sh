#!/usr/bin/env bash
# Build and test this fork on Windows (Git Bash), offline against the NuGet cache restored and
# recorded for b745217 (Spireline security addendum 1, conditions F1-F3).
#
#   bash tools/build_and_test.sh <log dir> <python with the upstream lock installed>
#
# Writes 1-restore.log .. 5-corpus-check.log (3b-emudump-test.log: tools/emudump.Tests; 3c-emudump-build.log:
# the Release EmuDump build), one <step>.rc exit code file per test step, timing.log (each step's start, end and seconds
# since the script began) and results.json to the log dir. The Release EmuDump is built once, before the test steps: tests
# that drive the real EmuDump (tools/fidelity/fuzz_core2.py, tools/emudump/bin/Release/net9.0/EmuDump.dll)
# must not run a stale build, and the corpus check reuses that build. After restore, publish and that build, the three test
# steps run at once at BelowNormal (PARALLEL=0 runs them one after another): the C# suite then the EmuDump suite, the Python
# suite (PY_TIER=fast, the default, or slow or all: tools/py_tests.py, the list in tests/python/slow_tier.txt) and the corpus
# check (CORPUS_PROCS, default 4 in parallel, 16 serial). Each keeps its log and exit code; one failing step never stops
# the others. The Python suite
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
: > "$LOG/timing.log"
# One line per step in timing.log: name, start and end in seconds since the script began, and the step's own seconds.
timed() { local n="$1" s e; shift; s=$(date +%s); "$@"; local rc=$?; e=$(date +%s); echo "$n $((s - t0)) $((e - t0)) $((e - s))" >> "$LOG/timing.log"; return $rc; }
# Restore from the local package cache only: no network source at all.
# Tests first: restoring the test project rewrites the library's assets without a RID, so the library is restored for win-x64 last, right before the NativeAOT publish needs it.
dotnet restore tools/emudump.Tests/EmuDump.Tests.csproj --source "$CACHE" > "$LOG/1-restore.log" 2>&1; r1c=$?
dotnet restore src/Sts2Emulator.Tests/Sts2Emulator.Tests.csproj --source "$CACHE" >> "$LOG/1-restore.log" 2>&1; r1b=$?
dotnet restore src/Sts2Emulator/Sts2Emulator.csproj -r win-x64 --source "$CACHE" >> "$LOG/1-restore.log" 2>&1; r1=$?
now=$(( $(date +%s) - t0 )); echo "restore 0 $now $now" >> "$LOG/timing.log"
timed publish dotnet publish src/Sts2Emulator/Sts2Emulator.csproj -c Release -r win-x64 --self-contained --no-restore -o out/   > "$LOG/2-publish.log" 2>&1; r2=$?
sha=""; [ -f out/Sts2Emulator.dll ] && sha=$(sha256sum out/Sts2Emulator.dll | cut -d' ' -f1)
# The Release EmuDump, built once here and before anything that drives it: the Python suite (tests/python/test_fuzz_core2.py's
# RealEmuDump) and the corpus check both run tools/emudump/bin/Release/net9.0/EmuDump.dll, and neither may run a stale one. Its
# own restore rewrites the library's assets without a RID, so the library is restored for win-x64 again straight after: the
# C# tests below run in the state they always ran in, and no step restores while another builds.
timed emudump-build dotnet build tools/emudump/EmuDump.csproj -c Release --source "$CACHE" > "$LOG/3c-emudump-build.log" 2>&1; r3c=$?
dotnet restore src/Sts2Emulator/Sts2Emulator.csproj -r win-x64 --source "$CACHE" >> "$LOG/1-restore.log" 2>&1; r1d=$?
# The test steps. Run at once (PARALLEL=0: one after another), each its own process group at BelowNormal (nice -n 10 is the
# Windows BelowNormal class under Git Bash, inherited by the children), each keeping its own log and exit code file:
#   cs      the C# suite, then the EmuDump tools' own suite (both build the library in Debug, so they never overlap),
#   python  the Python suite's tier (PY_TIER, default fast; all = the whole suite) with STS2_REQUIRE_AOT=1: the solver's
#           AOT-against-JIT test (tests/python/test_solver_aot_sites.py) fails here, never skips,
#   corpus  the recorded-fight corpus (tools/fidelity/corpus.py, referee/RUNBOOK.md): every fight ever recorded in the game,
#           replayed by this build and compared with the stored baseline; exit 1 lists the regressions or baseline fights it
#           could not compare. Skipped when this PC has no corpus, or with CORPUS_CHECK=0 ("skipped", which RUNBOOK step 7
#           never accepts for a push).
PARALLEL="${PARALLEL:-1}"; TIER="${PY_TIER:-fast}"
if [ "$PARALLEL" = 1 ]; then NICE="nice -n 10"; CPROCS="${CORPUS_PROCS:-4}"; else NICE=""; CPROCS="${CORPUS_PROCS:-16}"; fi
command -v nice > /dev/null 2>&1 || NICE=""
CORPUS_DIR="${SPIRELINE_CORPUS:-/d/spireline/referee/corpus}"
cs_step() {
  timed csharp $NICE dotnet test src/Sts2Emulator.Tests/ --no-restore > "$LOG/3-dotnet-test.log" 2>&1; echo $? > "$LOG/3-dotnet-test.rc"
  timed emudump-tests $NICE dotnet test tools/emudump.Tests/ --no-restore > "$LOG/3b-emudump-test.log" 2>&1; echo $? > "$LOG/3b-emudump-test.rc"
}
py_step() {
  if [ -n "$sha" ]; then
    echo "tier: $TIER" > "$LOG/4-python-test.log"
    ( export STS2_REQUIRE_AOT=1; timed python $NICE "$PY" tools/py_tests.py "$TIER" >> "$LOG/4-python-test.log" 2>&1; echo $? > "$LOG/4-python-test.rc" )
  else echo skipped > "$LOG/4-python-test.rc"; fi
}
corpus_step() {
  if [ "${CORPUS_CHECK:-1}" != 0 ] && [ -f "$CORPUS_DIR/index.json" ]; then
    if [ "$r3c" = 0 ]; then
      timed corpus $NICE "$PY" tools/fidelity/corpus.py --corpus "$CORPUS_DIR" check --procs "$CPROCS" > "$LOG/5-corpus-check.log" 2>&1; echo $? > "$LOG/5-corpus-check.rc"
    else
      echo "not run: the Release EmuDump build failed (3c-emudump-build.log)" > "$LOG/5-corpus-check.log"; echo build-failed > "$LOG/5-corpus-check.rc"
    fi
  else echo skipped > "$LOG/5-corpus-check.rc"; fi
}
rm -f "$LOG"/*.rc
tp=$(date +%s)
if [ "$PARALLEL" = 1 ]; then
  cs_step & pid_cs=$!
  py_step & pid_py=$!
  corpus_step & pid_corpus=$!
  wait "$pid_cs" "$pid_py" "$pid_corpus"
else
  cs_step; py_step; corpus_step
fi
echo "parallel-phase $((tp - t0)) $(( $(date +%s) - t0 )) $(( $(date +%s) - tp ))" >> "$LOG/timing.log"
# A step that left no exit code file (its process died) is a failure, never a pass.
rc_of() { if [ -s "$1" ]; then tr -d '[:space:]' < "$1"; else echo 99; fi; }
r3=$(rc_of "$LOG/3-dotnet-test.rc"); r3b=$(rc_of "$LOG/3b-emudump-test.rc"); r4=$(rc_of "$LOG/4-python-test.rc"); r5=$(rc_of "$LOG/5-corpus-check.rc")
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
corpus_line=""
[ -f "$LOG/5-corpus-check.log" ] && corpus_line=$(grep -E '^check: ' "$LOG/5-corpus-check.log" | tail -1 | tr -d '"\\')
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
