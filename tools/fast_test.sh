#!/usr/bin/env bash
# The C# suite, sharded across processes. The suite disables xUnit's in-process parallelism
# (AssemblyInfo.cs) because the emulator keeps static state (handle pools, factory fields), so
# tests must not share a process concurrently. Separate processes each get their own copy, so
# test classes are split into N shards that run at once.
#
#   bash tools/fast_test.sh [shards=8] [extra dotnet test --filter, e.g. "FullyQualifiedName~Splash"]
#
# Prints one line per shard and a total; exit 0 only if every shard passed.
set -u -o pipefail
N="${1:-8}"; EXTRA="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
LOG="$(mktemp -d)"
t0=$(date +%s)
dotnet build src/Sts2Emulator.Tests/ --no-restore -v q > "$LOG/build.log" 2>&1 || { tail -20 "$LOG/build.log"; exit 2; }
if [ -n "$EXTRA" ]; then
  dotnet test src/Sts2Emulator.Tests/ --no-build --filter "$EXTRA" 2>&1 | grep -E "Passed!|Failed!|  Failed "
  exit "${PIPESTATUS[0]}"
fi
# Test classes, from the test list (Namespace.Class.Method -> Namespace.Class).
dotnet test src/Sts2Emulator.Tests/ --no-build --list-tests 2>/dev/null \
  | sed -n 's/^    //p' | sed -E 's/\(.*//; s/\.[^.]+$//' | sort -u > "$LOG/classes.txt"
total=$(wc -l < "$LOG/classes.txt")
i=0
while read -r cls; do echo "$cls" >> "$LOG/shard$((i % N)).txt"; i=$((i + 1)); done < "$LOG/classes.txt"
pids=()
for s in $(seq 0 $((N - 1))); do
  [ -f "$LOG/shard$s.txt" ] || continue
  filter=$(sed 's/^/FullyQualifiedName~/; s/$/./' "$LOG/shard$s.txt" | paste -sd'|')
  dotnet test src/Sts2Emulator.Tests/ --no-build --filter "$filter" > "$LOG/test$s.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
passed=0; failed=0
for f in "$LOG"/test*.log; do
  line=$(grep -Eo 'Failed:\s*[0-9]+, Passed:\s*[0-9]+' "$f" | tail -1)
  echo "$(basename "$f"): ${line:-no summary}"
  f_=$(echo "$line" | grep -Eo 'Failed:\s*[0-9]+' | grep -Eo '[0-9]+'); p_=$(echo "$line" | grep -Eo 'Passed:\s*[0-9]+' | grep -Eo '[0-9]+')
  failed=$((failed + ${f_:-0})); passed=$((passed + ${p_:-0}))
  grep -E "^\s+Failed " "$f" | head -5
done
echo "total: $total classes, passed $passed, failed $failed, $(( $(date +%s) - t0 )) s (logs $LOG)"
[ "$rc" = 0 ] && [ "$failed" = 0 ]
