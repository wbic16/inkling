#!/bin/bash
setsid bash $HOME/probe2_run.sh > $HOME/rse-probe-launch.log 2>&1 < /dev/null &
echo "launched $!"
sleep 2
ps aux | grep -E 'llama-server|probe2' | grep -v grep | head -3
