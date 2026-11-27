#!/bin/bash
set -u
pkill -f transport_gguf.py 2>/dev/null; sleep 2
cd ~/rse-fpsolve
STAGE=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging
rm -f "$STAGE/Inkling-2-Mini-FPSOLVE-Q4_K.gguf"
./venv/bin/python transport_gguf.py \
  --src-gguf /mnt/mirrorborn/shared/models/inkling-2-mini-gguf/Inkling-2-Mini-AD-Q4_K.gguf \
  --edited-tf "$STAGE/edited_tf.safetensors" \
  --out "$STAGE/Inkling-2-Mini-FPSOLVE-Q4_K.gguf" \
  --report "$STAGE/transport-report.json" > "$STAGE/transport.log" 2>&1
echo "transport rc=$?"
tail -5 "$STAGE/transport.log"
sha256sum "$STAGE/Inkling-2-Mini-FPSOLVE-Q4_K.gguf" 2>/dev/null
ls -la "$STAGE/"
