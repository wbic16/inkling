#!/bin/bash
# Thermal watchdog for a running PID: SIGSTOP at >=88C, SIGCONT at <=75C. Logs every sample.
PID=$1
LOG=$2
while kill -0 "$PID" 2>/dev/null; do
  T=$(sensors 2>/dev/null | grep -i "tctl" | grep -oE "[0-9]+\.[0-9]" | head -1)
  echo "$(date -Is) tctl=$T pid=$PID" >> "$LOG"
  if [ -n "$T" ]; then
    TI=${T%.*}
    if [ "$TI" -ge 88 ]; then kill -STOP "$PID" 2>/dev/null; echo "$(date -Is) PAUSED at $T" >> "$LOG"; fi
    if [ "$TI" -le 75 ]; then kill -CONT "$PID" 2>/dev/null; fi
  fi
  sleep 60
done
echo "$(date -Is) watchdog exit (pid gone)" >> "$LOG"
