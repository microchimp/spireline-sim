#!/usr/bin/env bash
# The run currency's fast tests (docs/rl/RUN_CURRENCY.md; big-step memo step 6): the audit, the board, the exam's Act 1 hallway and
# elite cells and a 200-run copy batch, a3-v1-r against a3-v1, then the stop rule's verdict in $OUT/report.json.
#
# Usage:  OUT=D:/spireline/referee/backlog/scratch-currency-fast bash tools/currency_fast.sh [audit|board|cells|batch|board2000|cells12|batch2|report|all]
# Run it from the worktree that holds the branch (it needs that worktree's out/ built: logs/gym-loop/bin/publish-wt.sh, WTN=<folder>).
#
# The QUIET steps are rl/currency_fast.py's own, so the script never needs to be wrapped:
#   1. it waits (a poll a minute, printing why) until the machine is free: no LAB/nightly/active, no LAB/score/active, no real-game job in
#      LAB/status.txt, no main.lock, and no QUIET.lock line except gym-loop, lab-score and its own (a `board-a` line, or any other tag,
#      holds it);
#   2. it writes `b-fast <ISO UTC> pid=<its Windows PID>` into QUIET.lock, rewritten every ten minutes while it plays;
#   3. it removes that line when each play step ends, and stops its workers at once (and goes back to 1) when the machine is taken;
#   4. a finished game or run is never played twice: rows are appended and finished ones skipped, so the script is safe to start again.
# The audit reads files only (seconds, no line). Do not start it while a lab night is due within the half hour unless it ends before the
# night (the board is about 15 min of six processes, the cells about 15, the batch about 40, the 2,000-game board about 55).
set -u
: "${OUT:?set OUT to the folder the readings go in}"
cd "$(dirname "$0")/.." || exit 9
PY=D:/spireline/src/sts2/.venv/Scripts/python.exe
step=${1:-all}
run() { "$PY" rl/currency_fast.py "$@" --out "$OUT"; }
case "$step" in
  audit)     run audit ;;
  board)     run board --games 400 ;;
  cells)     run cells ;;
  batch)     run batch --runs 200 ;;
  board2000) run board --games 2000 ;;   # extends the 400-game board to 2,000 paired games (the 400 already played are kept)
  cells12)   run cells --draws 12 ;;     # extends the cells to 12 draws a start (the four already played are kept)
  batch2)    run batch --runs 400 --seed-offset 200 --batch-name batch2 ;;   # 400 more runs on new seeds (the report reads every batch* folder)
  report)    run report ;;
  all)       run audit; a=$?
             if [ "$a" = 2 ]; then run report; exit 2; fi   # the audit's stop rule ends the run; its reasons are in report.json
             [ "$a" = 0 ] && run board --games 400 && run cells && run batch --runs 200 && run report ;;
  *) echo "usage: OUT=<dir> bash tools/currency_fast.sh [audit|board|cells|batch|board2000|cells12|batch2|report|all]" >&2; exit 2 ;;
esac
