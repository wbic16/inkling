#!/bin/bash
cd ~/rse-fpsolve/work || exit 1
echo "=== SOLVE_DONE ==="; cat SOLVE_DONE
echo "=== fpsolve_run.json ==="; cat fpsolve_run.json
echo "=== deltas sha256 ==="; sha256sum deltas.safetensors
echo "=== thermal log tail ==="; tail -4 ~/rse-fpsolve/work/thermal.log ~/thermal.log ~/rse-fpsolve/thermal.log 2>/dev/null | head -12
echo "=== temps now ==="; sensors 2>/dev/null | grep -iE "tctl|temp1" | head -3
mkdir -p /mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging
cp deltas.safetensors /mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging/deltas.safetensors
cp fpsolve_run.json SOLVE_DONE /mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging/ 2>/dev/null
echo "=== staged copy sha256 ==="; sha256sum /mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging/deltas.safetensors
