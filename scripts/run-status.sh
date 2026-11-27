#!/bin/bash
echo "host=$(hostname) $(date -Is)"
echo "=== processes ==="
ps aux | grep -E "run-download|run-torch|curl|pip" | grep -v grep | head -10
echo "=== hub download log ==="
tail -5 /mnt/mirrorborn/shared/models/inkling-2-mini-hf/download.log 2>/dev/null
ls -la /mnt/mirrorborn/shared/models/inkling-2-mini-hf/ 2>/dev/null | head
echo "=== pv torch log ==="
tail -5 ~/rse-fpsolve/torch-setup.log 2>/dev/null
ls ~/rse-fpsolve/ 2>/dev/null
sensors 2>/dev/null | grep -iE "tctl"
free -g | head -2
