#!/bin/bash
echo "proxy env: http_proxy=$http_proxy https_proxy=$https_proxy"
T=/tmp/hf-test.bin; rm -f "$T"
S=$(date +%s)
curl -sL --max-time 60 -r 0-10485759 -o "$T" https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/model-00001-of-00001.safetensors
E=$(date +%s); SZ=$(stat -c%s "$T" 2>/dev/null || echo 0); rm -f "$T"
echo "fetched $SZ bytes in $((E-S))s"
echo HF_TEST_DONE
