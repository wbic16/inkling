#!/bin/bash
# Bisect transformers 5.x wheels for eelbaz-compatible inkling2_loop API markers.
set -u
cd ~/rse-fpsolve
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
mkdir -p tfbisect; cd tfbisect
for V in 5.23.0 5.24.0; do
  if [ ! -f "tf-$V.whl" ]; then
    ../venv/bin/pip download "transformers==$V" --no-deps -d . -q 2>/dev/null
    mv transformers-*.whl "tf-$V.whl" 2>/dev/null
  fi
  if [ -f "tf-$V.whl" ]; then
    M=$(unzip -p "tf-$V.whl" transformers/models/inkling2_loop/modeling_inkling2_loop.py 2>/dev/null | grep -c "class Inkling2LoopCache")
    C=$(unzip -p "tf-$V.whl" transformers/models/inkling2_loop/configuration_inkling2_loop.py 2>/dev/null | grep -c "loop_start_layer")
    F=$(unzip -p "tf-$V.whl" transformers/models/inkling2_loop/modeling_inkling2_loop.py 2>/dev/null | grep -A3 "class Inkling2LoopTextModel" | head -1)
    echo "v$V cache_class=$M config_loop_start=$C"
  else
    echo "v$V DOWNLOAD FAIL"
  fi
done
echo BISECT_DONE
