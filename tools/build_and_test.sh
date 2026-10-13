#!/usr/bin/env bash
# Build and test this fork on Windows (Git Bash), offline against the NuGet cache restored and
# recorded for b745217 (Spireline security addendum 1, conditions F1-F3).
#
#   bash tools/build_and_test.sh <log dir> <python with the upstream lock installed>
#
# Writes 1-restore.log .. 5-corpus-check.log and results.json to the log dir. The Python suite
# is compared against the reviewed b745217 baseline: on Windows ~60 upstream tests fail on
# POSIX-only process code (os.nice, os.killpg, ...), so "pass" means no NEW failures.
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
dotnet restore src/Sts2Emulator.Tests/Sts2Emulator.Tests.csproj --source "$CACHE" > "$LOG/1-restore.log" 2>&1; r1b=$?
dotnet restore src/Sts2Emulator/Sts2Emulator.csproj -r win-x64 --source "$CACHE" >> "$LOG/1-restore.log" 2>&1; r1=$?
dotnet publish src/Sts2Emulator/Sts2Emulator.csproj -c Release -r win-x64 --self-contained --no-restore -o out/ \
  > "$LOG/2-publish.log" 2>&1; r2=$?
sha=""; [ -f out/Sts2Emulator.dll ] && sha=$(sha256sum out/Sts2Emulator.dll | cut -d' ' -f1)
dotnet test src/Sts2Emulator.Tests/ --no-restore > "$LOG/3-dotnet-test.log" 2>&1; r3=$?
r4=skipped
if [ -n "$sha" ]; then
  "$PY" -m unittest discover -s tests/python > "$LOG/4-python-test.log" 2>&1; r4=$?
fi
cs=$(grep -Eo 'Failed:\s*[0-9]+, Passed:\s*[0-9]+, Skipped:\s*[0-9]+, Total:\s*[0-9]+' "$LOG/3-dotnet-test.log" | tail -1)
pyline=$(grep -E '^(OK|FAILED)' "$LOG/4-python-test.log" 2>/dev/null | tail -1)
newfail=""
if [ -f "$LOG/4-python-test.log" ] && [ -n "${BASELINE_FAILURES:-}" ]; then
  grep -E '^(FAIL|ERROR): ' "$LOG/4-python-test.log" | sed -E 's/^(FAIL|ERROR): //' | tr -d '
' | sort -u > "$LOG/python-failures.txt"
  newfail=$(comm -23 "$LOG/python-failures.txt" "$BASELINE_FAILURES" | wc -l | tr -d ' ')
fi
pyran=$(grep -E '^Ran [0-9]+ tests' "$LOG/4-python-test.log" 2>/dev/null | tail -1)
# The recorded-fight corpus (tools/fidelity/corpus.py, referee/RUNBOOK.md): every fight ever recorded
# in the game, replayed by this build and compared with the stored baseline; exit 1 lists the
# regressions. Skipped when this PC has no corpus, or with CORPUS_CHECK=0.
r5=skipped; corpus_line=""
CORPUS_DIR="${SPIRELINE_CORPUS:-/d/spireline/referee/corpus}"
if [ "${CORPUS_CHECK:-1}" != 0 ] && [ -f "$CORPUS_DIR/index.json" ]; then
  if dotnet build tools/emudump/EmuDump.csproj -c Release --source "$CACHE" > "$LOG/5-corpus-check.log" 2>&1; then
    "$PY" tools/fidelity/corpus.py --corpus "$CORPUS_DIR" check --procs "${CORPUS_PROCS:-16}" >> "$LOG/5-corpus-check.log" 2>&1; r5=$?
  else
    r5=build-failed
  fi
  corpus_line=$(grep -E '^check: ' "$LOG/5-corpus-check.log" | tail -1 | tr -d '"\\')
fi
cat > "$LOG/results.json" <<EOF
{"commit": "$(git rev-parse HEAD)", "tree": "$(git rev-parse 'HEAD^{tree}')", "dotnet_sdk": "$(dotnet --version)",
 "restore_exit": [$r1, $r1b], "publish_exit": $r2, "native_lib_sha256": "$sha",
 "dotnet_test_exit": $r3, "tests_csharp": "$cs",
 "python_test_exit": "$r4", "tests_python": "$pyran / $pyline", "python_new_failures_vs_baseline": "$newfail",
 "corpus_check_exit": "$r5", "corpus_check": "$corpus_line", "minutes": $(( ($(date +%s) - t0) / 60 ))}
EOF
cat "$LOG/results.json"
