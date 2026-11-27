#!/bin/bash
# Hub inspection for FP-Solve run: code, tools, disk, HF reachability.
set -u
echo "host=$(hostname) date=$(date -Is)"
echo "=== rse-looper-src ==="
cd ~/rse-looper-src 2>/dev/null && { git log --oneline -1; git remote -v | head -2; ls; } || echo "NO CLONE"
echo "=== pyproject ==="
cat ~/rse-looper-src/pyproject.toml 2>/dev/null
echo "=== fpsolve.py full ==="
cat ~/rse-looper-src/rse_looper/fpsolve.py 2>/dev/null | head -200
echo "=== disk NFS ==="
df -h /mnt/mirrorborn/shared | tail -1
ls -la /mnt/mirrorborn/shared/models/
echo "=== hf tools ==="
which huggingface-cli hf python3 pip3 2>/dev/null
python3 -c "import huggingface_hub; print('hf_hub', huggingface_hub.__version__)" 2>&1 | head -1
echo "=== HF API reachability ==="
curl -s -m 30 "https://huggingface.co/api/models/thinkingmachines/Inkling-2-Mini" | head -c 2000
echo
echo "INSPECT DONE"
