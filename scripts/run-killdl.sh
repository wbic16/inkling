#!/bin/bash
pkill -f "run-download.sh" 2>/dev/null
sleep 1
pkill -f "curl -sL -C -" 2>/dev/null
sleep 1
ps aux | grep -E "run-download|curl" | grep -v grep | head -5
echo KILLED_OLD_DOWNLOAD
