#!/bin/bash
# Start the firmware review tool (kills any existing instance first)
cd "$(dirname "$0")"
lsof -ti:8087 | xargs -r kill 2>/dev/null
source venv/bin/activate
echo "Starting review tool on http://0.0.0.0:8087"
python3 firmware_review_tool/app.py
