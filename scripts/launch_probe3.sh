#!/bin/bash
setsid bash $HOME/probe3_run.sh > $HOME/rse-probe3-launch.log 2>&1 < /dev/null &
echo "launched $!"
