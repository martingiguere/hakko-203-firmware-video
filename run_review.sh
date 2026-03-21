#!/bin/bash
# Review tool controller for Hakko FM-203 firmware extraction.
#
# Usage:
#   ./run_review.sh start     # start the review tool (background)
#   ./run_review.sh stop      # stop the review tool
#   ./run_review.sh status    # show running state
#   ./run_review.sh restart   # stop + start

set -euo pipefail
cd "$(dirname "$0")"

PID_FILE=".review.pid"
LOG_FILE="review.log"
VENV="venv/bin/activate"
PORT=8087

start_review() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Review tool already running (PID $pid). Use './run_review.sh stop' first."
            exit 1
        else
            rm -f "$PID_FILE"
        fi
    fi

    # Kill anything on the port
    lsof -ti:$PORT 2>/dev/null | xargs -r kill 2>/dev/null || true
    sleep 1

    source "$VENV"
    nohup python3 firmware_review_tool/app.py > "$LOG_FILE" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_FILE"

    # Wait for it to start
    sleep 3
    if curl -s -o /dev/null -w "" http://localhost:$PORT/ 2>/dev/null; then
        # Get the IP address for mobile access
        ip=$(grep -oP 'http://\d+\.\d+\.\d+\.\d+' "$LOG_FILE" | grep -v 127.0.0.1 | head -1)
        echo "Review tool started (PID $pid)"
        echo "  Desktop: http://localhost:$PORT/"
        echo "  Mobile:  ${ip:-http://<your-ip>}:$PORT/mobile"
    else
        echo "Review tool started (PID $pid) — waiting for startup..."
        echo "  Check: ./run_review.sh status"
    fi
}

stop_review() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        kill "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "Review tool stopped (PID $pid)"
    else
        echo "No PID file found"
    fi

    # Also kill anything on the port
    pids=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill 2>/dev/null || true
        echo "Killed processes on port $PORT"
    fi
}

show_status() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Review tool: running (PID $pid)"
            if [ -f "$LOG_FILE" ]; then
                # Show stats line
                stats=$(grep "Stats:" "$LOG_FILE" | tail -1)
                [ -n "$stats" ] && echo "  $stats"
                # Show URL
                ip=$(grep -oP 'http://\d+\.\d+\.\d+\.\d+' "$LOG_FILE" | grep -v 127.0.0.1 | head -1)
                echo "  Desktop: http://localhost:$PORT/"
                echo "  Mobile:  ${ip:-http://<your-ip>}:$PORT/mobile"
            fi
            return
        else
            echo "Review tool: not running (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        echo "Review tool: not running"
    fi
}

case "${1:-start}" in
    start)
        start_review
        ;;
    stop)
        stop_review
        ;;
    status)
        show_status
        ;;
    restart)
        stop_review
        sleep 1
        start_review
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
