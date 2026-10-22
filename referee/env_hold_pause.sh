#!/usr/bin/env bash
# Holds the lab's PAUSE ("env-proto") while envproto.py measures: removed on exit (trap), on envproto/RELEASE,
# or after 115 min at most.
LAB=/d/spireline/referee/LAB
EP=/d/spireline/referee/envproto
mkdir -p "$EP"; rm -f "$EP/RELEASE" "$EP/PAUSED"
trap 'rm -f "$LAB/PAUSE"; echo "$(date -u +%FT%TZ) PAUSE removed" >> "$EP/hold.log"' EXIT INT TERM HUP
printf 'env-proto' > "$LAB/PAUSE"; tw=$(date -u +%FT%TZ)
echo "$(date -u +%FT%TZ) PAUSE written" >> "$EP/hold.log"
t0=$(date +%s)
while [ ! -f "$EP/RELEASE" ] && [ $(( $(date +%s) - t0 )) -lt 6900 ]; do
  st=$(head -c 20 "$LAB/status.txt" 2>/dev/null)
  if grep -q paused "$LAB/status.txt" 2>/dev/null && [[ "$st" > "$tw" ]] && [ ! -f "$EP/PAUSED" ]; then
    echo "$(date -u +%FT%TZ) lab paused: $(cut -c1-100 "$LAB/status.txt")" >> "$EP/hold.log"; touch "$EP/PAUSED"
  fi
  sleep 5
done
