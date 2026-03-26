#!/bin/bash
# Pipeline controller for Hakko FM-203 firmware extraction.
#
# Usage:
#   ./pipeline.sh start                       # full pipeline (OCR + post-steps)
#   ./pipeline.sh start --post-only           # post-steps only (skip OCR)
#   ./pipeline.sh start --rebuild             # retrain classifier + full pipeline
#   ./pipeline.sh start --rebuild --reset     # full retrain (reset automated moves, keep manual)
#   ./pipeline.sh stop                        # stop running pipeline + all children
#   ./pipeline.sh status                      # show running state and progress

set -euo pipefail
cd "$(dirname "$0")"

PID_FILE=".pipeline.pid"
LOG_FILE="pipeline.log"
VENV="venv/bin/activate"

start_pipeline() {
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Pipeline already running (PID $pid). Use './pipeline.sh stop' first."
            exit 1
        else
            echo "Stale PID file found (process $pid not running). Cleaning up."
            rm -f "$PID_FILE"
        fi
    fi

    # Pass through any flags (--post-only, --rebuild, etc.)
    flags="$*"

    echo "Starting pipeline${flags:+ with flags: $flags}"
    echo "Log: $LOG_FILE"

    # Run in new session (setsid) so stop can kill entire process group
    source "$VENV"
    setsid python3 extract_pipeline.py $flags > "$LOG_FILE" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_FILE"

    echo "Pipeline started (PID $pid)"
    echo "Use './pipeline.sh status' to check progress"
    echo "Use './pipeline.sh stop' to cancel"
}

stop_pipeline() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No pipeline running (no PID file)"
        # Check for orphaned processes anyway
        orphans=$(pgrep -f "extract_pipeline.py|precompute.py|fix_address_trajectory|fix_outlier_votes|fix_byte_agreement|fix_duplicate_consensus|postprocess_firmware|ff_fill.py|r8c_validator|precompute_gaps" 2>/dev/null || true)
        if [ -n "$orphans" ]; then
            echo "Found orphaned pipeline processes: $orphans"
            echo "Killing them..."
            echo "$orphans" | xargs kill 2>/dev/null || true
            sleep 2
            echo "$orphans" | xargs kill -9 2>/dev/null || true
            echo "Done"
        fi
        return
    fi

    pid=$(cat "$PID_FILE")
    echo "Stopping pipeline (PID $pid)..."

    # Kill the process group (parent + all children)
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ] && [ "$pgid" != "0" ]; then
        kill -- "-$pgid" 2>/dev/null || true
        sleep 2
        # Force kill any survivors
        kill -9 -- "-$pgid" 2>/dev/null || true
    else
        kill "$pid" 2>/dev/null || true
    fi

    # Also kill any orphaned pipeline scripts
    pgrep -f "extract_pipeline.py|precompute.py|fix_address_trajectory|fix_outlier_votes|fix_byte_agreement|fix_duplicate_consensus|postprocess_firmware|ff_fill.py|r8c_validator|precompute_gaps" 2>/dev/null | xargs -r kill 2>/dev/null || true

    rm -f "$PID_FILE"
    echo "Pipeline stopped"
}

show_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Pipeline: not running"
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Last run (final lines of $LOG_FILE):"
            tail -5 "$LOG_FILE"
        fi
        return
    fi

    pid=$(cat "$PID_FILE")
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Pipeline: finished (PID $pid no longer running)"
        rm -f "$PID_FILE"
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Final output:"
            tail -10 "$LOG_FILE"
        fi
        return
    fi

    # Running — show progress
    echo "Pipeline: running (PID $pid)"

    # Elapsed time
    start_time=$(stat -c %Y "$PID_FILE" 2>/dev/null || stat -f %m "$PID_FILE" 2>/dev/null)
    now=$(date +%s)
    elapsed=$(( now - start_time ))
    mins=$(( elapsed / 60 ))
    secs=$(( elapsed % 60 ))
    echo "Elapsed: ${mins}m ${secs}s"

    # Current step
    if [ -f "$LOG_FILE" ]; then
        current_step=$(grep "^Running:" "$LOG_FILE" | tail -1)
        if [ -n "$current_step" ]; then
            echo "Current step: $current_step"
        fi
        echo ""
        echo "Recent output:"
        tail -5 "$LOG_FILE"
    fi
}

case "${1:-}" in
    start)
        shift
        start_pipeline "$@"
        ;;
    stop)
        stop_pipeline
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|status} [flags]"
        echo ""
        echo "Commands:"
        echo "  start              Start the pipeline (runs in background)"
        echo "  stop               Stop the running pipeline and all child processes"
        echo "  status             Show pipeline state and progress"
        echo ""
        echo "Flags (for start):"
        echo "  --post-only        Skip OCR extraction, run post-steps only"
        echo "  --rebuild          Retrain kNN classifier from scratch"
        echo "  --reset            Reset automated frame moves, keep manual moves"
        echo "                     (use with --rebuild for clean retrain)"
        exit 1
        ;;
esac
