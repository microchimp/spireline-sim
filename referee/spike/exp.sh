#!/usr/bin/env bash
# Speed spike experiments (referee/design/real-game-speed.md), one after another, once hold_pause.sh has seen the lab
# paused. Usage: exp.sh <list file> (lines: name serve.py-args; see lists.txt). Skips a name whose run dir exists.
SP=/d/spireline/referee/spike
PY=/d/spireline/src/sts2/.venv/Scripts/python.exe
cd /d/spireline/fork/wt-spike || exit 1
export PYTHONDONTWRITEBYTECODE=1
while [ ! -f "$SP/PAUSED" ]; do sleep 5; done
while read -r name rest; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  [ -d "$SP/$name" ] && continue
  load=$(wmic cpu get loadpercentage 2>/dev/null | tr -dc '0-9')
  echo "$(date -u +%FT%TZ) start $name load=${load}% :: $rest" >> "$SP/exp.log"
  t0=$(date +%s.%N)
  eval "$PY referee/serve.py --game /d/spireline/referee/game-spike --spares 0 --run-dir $SP/$name $rest" > "$SP/$name.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) end $name rc=$rc wall=$(( $(date +%s) - ${t0%.*} ))s" >> "$SP/exp.log"
done < "$1"
echo "$(date -u +%FT%TZ) list done $1" >> "$SP/exp.log"
