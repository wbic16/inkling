#!/bin/bash
pkill -f "run-parallel.sh" 2>/dev/null; sleep 1
pkill -f "safetensors.part" 2>/dev/null; sleep 2
ps aux | grep -E "curl" | grep -v grep | wc -l
rm -f /mnt/mirrorborn/shared/models/inkling-2-mini-hf/model-00001-of-00001.safetensors.part*
echo KILLED_PARALLEL
