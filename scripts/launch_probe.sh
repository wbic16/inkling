#!/bin/bash
setsid bash /home/wbic16/probe_run.sh > /home/wbic16/rse-probe-launch.log 2>&1 < /dev/null &
echo "launched pid $!"
sleep 2
ls -la /home/wbic16/rse-probe/ 2>/dev/null
ps aux | grep -E 'llama-cli|probe_run' | grep -v grep | head -3
