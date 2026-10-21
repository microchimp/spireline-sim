#!/usr/bin/env bash
# Holds the lab's PAUSE for the speed spike: removed on exit (trap), on a RELEASE file, or after 105 min at most.
LAB=/d/spireline/referee/LAB
SP=/d/spireline/referee/spike
trap 'rm -f "$LAB/PAUSE"; echo "$(date -u +%FT%TZ) PAUSE removed" >> "$SP/hold.log"' EXIT INT TERM HUP
printf 'speed-spike' > "$LAB/PAUSE"
echo "$(date -u +%FT%TZ) PAUSE written" >> "$SP/hold.log"
t0=$(date +%s)
while [ ! -f "$SP/RELEASE" ] && [ $(( $(date +%s) - t0 )) -lt 6300 ]; do
  if grep -q 'paused' "$LAB/status.txt" 2>/dev/null && [ ! -f "$SP/PAUSED" ]; then
    echo "$(date -u +%FT%TZ) lab paused: $(cut -c1-120 "$LAB/status.txt")" >> "$SP/hold.log"; touch "$SP/PAUSED"
  fi
  sleep 5
done
