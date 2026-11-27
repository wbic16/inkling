#!/bin/bash
echo "=== llama processes ==="
ps aux | grep -i llama | grep -v grep | head -5
echo "=== user services (llama/inkling/exo) ==="
systemctl --user list-units --type=service --all 2>/dev/null | grep -iE 'llama|inkling|exo' | head -10
echo "=== candidate llama.cpp dirs ==="
for d in /home/wbic16/llama.cpp /home/wbic16/src/llama.cpp /home/wbic16/llama /opt/llama.cpp /home/wbic16/exo /home/wbic16/nf-repo; do
  [ -d "$d" ] && echo "DIR: $d"
done
echo "=== find inkling2 sources (maxdepth 5, home + shared runtime) ==="
find /home/wbic16 -maxdepth 5 -iname '*inkling2*' 2>/dev/null | head -20
echo "=== runtime tar listing (inkling/llama bins) ==="
tar tzf /mnt/mirrorborn/shared/models/inkling-2-mini-gguf/runtime/llama-runtime.tar.gz 2>/dev/null | head -30
echo "=== which llama binaries on PATH / common spots ==="
which llama-server llama-cli llama-bench 2>/dev/null
ls /home/wbic16/llama.cpp/build/bin 2>/dev/null | head
find /home/wbic16 -maxdepth 4 -name 'llama-server' -type f 2>/dev/null | head -5
