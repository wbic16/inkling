#!/bin/bash
# Launch FP-Solve (mode=solve) detached + thermal watchdog.
set -u
cd ~/rse-fpsolve
W=$HOME/rse-fpsolve/work; mkdir -p "$W"
./venv/bin/python driver_fpsolve.py --base "$HOME/inkling-hf-base" --work "$W" --mode solve > "$W/solve-driver.log" 2>&1 &
PID=$!
echo $PID > "$W/solve.pid"
echo "solve pid $PID"
nohup ./watchdog.sh $PID "$W/thermal.log" > /dev/null 2>&1 &
echo "watchdog launched"
