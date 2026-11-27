#!/bin/bash
# Stage 2: CPU torch home on plain-view + smoke test.
set -u
W=$HOME/rse-fpsolve
mkdir -p "$W"
LOG=$W/torch-setup.log
echo "=== torch setup start $(date -Is) host=$(hostname) ===" >> "$LOG"
sensors 2>/dev/null | grep -iE "tctl" >> "$LOG"
free -g >> "$LOG"; nproc >> "$LOG"

# Ranch proxy breaks uncached PyPI fetches: unset proxy vars in this shell only.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [ ! -x "$W/venv/bin/python" ]; then
  python3 -m venv "$W/venv" >> "$LOG" 2>&1 || { echo "VENV FAIL" >> "$LOG"; exit 1; }
fi
PIP="$W/venv/bin/pip"
$PIP install --upgrade pip >> "$LOG" 2>&1
echo "--- installing torch (CPU wheel index) $(date -Is)" >> "$LOG"
$PIP install torch --index-url https://download.pytorch.org/whl/cpu >> "$LOG" 2>&1 || { echo "TORCH INSTALL FAIL" >> "$LOG"; touch "$W/TORCH_FAIL"; exit 1; }
echo "--- installing transformers/safetensors/accelerate $(date -Is)" >> "$LOG"
$PIP install transformers safetensors accelerate numpy >> "$LOG" 2>&1 || { echo "TF INSTALL FAIL" >> "$LOG"; touch "$W/TORCH_FAIL"; exit 1; }

"$W/venv/bin/python" - >> "$LOG" 2>&1 <<'EOF'
import torch, transformers, safetensors
print("torch", torch.__version__, "threads", torch.get_num_threads())
print("transformers", transformers.__version__)
print("has Inkling2LoopForCausalLM:", hasattr(transformers, "Inkling2LoopForCausalLM"))
print("has Inkling2LoopForConditionalGeneration:", hasattr(transformers, "Inkling2LoopForConditionalGeneration"))
x = torch.randn(64, 64)
print("matmul ok", float((x @ x).sum()))
EOF
echo "=== torch setup done $(date -Is) ===" >> "$LOG"
sensors 2>/dev/null | grep -iE "tctl" >> "$LOG"
touch "$W/TORCH_DONE"
