#!/bin/bash
# Stage 5a: GGUF transport (self-verifying), then Stage 5b: eelbaz export (evidence).
set -u
cd ~/rse-fpsolve
STAGE=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging
SRC=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/Inkling-2-Mini-AD-Q4_K.gguf
echo "== transport start $(date -Is) =="
./venv/bin/python transport_gguf.py \
  --src-gguf "$SRC" \
  --deltas "$STAGE/deltas.safetensors" \
  --base-ckpt "$HOME/inkling-hf-base/model-00001-of-00001.safetensors" \
  --out "$STAGE/Inkling-2-Mini-FPSOLVE-Q4_K.gguf" \
  --report "$STAGE/transport-report.json" > "$STAGE/transport.log" 2>&1
echo "transport rc=$?"
tail -3 "$STAGE/transport.log"
echo "== export start $(date -Is) =="
./venv/bin/python driver_fpsolve.py --base "$HOME/inkling-hf-base" --work "$HOME/rse-fpsolve/work" \
  --mode export --out-model "$HOME/rse-fpsolve/edited-hf" > "$HOME/rse-fpsolve/work/export-driver.log" 2>&1
echo "export rc=$?"
tail -3 "$HOME/rse-fpsolve/work/driver-export.log" 2>/dev/null
echo "== all done $(date -Is) =="
