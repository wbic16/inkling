#!/bin/bash
# Launcher: fully detach a script on this node (own session, redirected IO).
# usage: run-detached.sh <script> <logfile>
setsid nohup "$1" > "$2" 2>&1 < /dev/null &
echo "detached pid $!"
