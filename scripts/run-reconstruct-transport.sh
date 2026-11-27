#!/bin/bash
set -u
cd ~/rse-fpsolve
echo "== orientation check =="
./venv/bin/python verify_orientation.py 2>&1 | grep -avE "Loading weights|it/s\]$" | tail -14
echo "== reconstruct =="
./venv/bin/python reconstruct_tf.py > work/reconstruct.log 2>&1
echo "reconstruct rc=$?"
grep -aE "loaded|deltas|edited_tf|saved|DONE|Error|Traceback" work/reconstruct.log | tail -8
echo "== transport =="
STAGE=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging
./venv/bin/python transport_gguf.py \
  --src-gguf /mnt/mirrorborn/shared/models/inkling-2-mini-gguf/Inkling-2-Mini-AD-Q4_K.gguf \
  --edited-tf "$STAGE/edited_tf.safetensors" \
  --out "$STAGE/Inkling-2-Mini-FPSOLVE-Q4_K.gguf" \
  --report "$STAGE/transport-report.json" > "$STAGE/transport.log" 2>&1
echo "transport rc=$?"
tail -4 "$STAGE/transport.log"
