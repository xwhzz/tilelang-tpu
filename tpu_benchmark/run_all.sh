#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"  # cd to repo root (tilelang-tpu)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"

# Collect all bench scripts
SCRIPTS=()
for f in "$SCRIPT_DIR"/swiglu/bench_swiglu_*.py; do
    [ -f "$f" ] && SCRIPTS+=("$f")
done
for f in "$SCRIPT_DIR"/rope/bench_rope_*.py; do
    [ -f "$f" ] && SCRIPTS+=("$f")
done
for f in "$SCRIPT_DIR"/elementwise_add/bench_add_*.py; do
    [ -f "$f" ] && SCRIPTS+=("$f")
done

NUM=${#SCRIPTS[@]}
PASS=0
FAIL=0
BAR_WIDTH=40

log() {
    echo "$1" | tee -a "$SUMMARY_LOG"
}

progress_bar() {
    local cur=$1 total=$2 pass=$3 fail=$4
    local pct=$((cur * 100 / total))
    local filled=$((cur * BAR_WIDTH / total))
    local empty=$((BAR_WIDTH - filled))
    local bar
    bar=$(printf '%0.s#' $(seq 1 "$filled") 2>/dev/null || true)
    bar+=$(printf '%0.s-' $(seq 1 "$empty") 2>/dev/null || true)
    printf "\r  [%s] %3d%% (%d/%d)  pass=%d fail=%d" "$bar" "$pct" "$cur" "$total" "$pass" "$fail"
}

log "============================================================"
log "  TPU Benchmark Runner  $(date)"
log "  Total: $NUM scripts"
log "============================================================"

for i in "${!SCRIPTS[@]}"; do
    idx=$((i + 1))
    script="${SCRIPTS[$i]}"
    name="$(basename "$script" .py)"
    logfile="$LOG_DIR/${name}_${TIMESTAMP}.log"

    progress_bar "$((idx - 1))" "$NUM" "$PASS" "$FAIL"

    if python "$script" > "$logfile" 2>&1; then
        PASS=$((PASS + 1))
        log "[$idx/$NUM] PASS  $name"
    else
        rc=$?
        FAIL=$((FAIL + 1))
        log "[$idx/$NUM] FAIL  $name  (rc=$rc)"
    fi

    progress_bar "$idx" "$NUM" "$PASS" "$FAIL"
done

printf "\n"
log ""
log "============================================================"
log "  Done: $NUM total, $PASS passed, $FAIL failed"
log "  Summary: $SUMMARY_LOG"
log "  Logs:    $LOG_DIR/*_${TIMESTAMP}.log"
log "============================================================"
