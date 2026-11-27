#!/bin/bash
cd ~/rse-fpsolve/tfbisect || exit 1
for V in 5.0.0 5.12.0 5.22.0; do
  echo "=== v$V ==="
  unzip -l "tf-$V.whl" | grep -c "inkling2_loop"
  unzip -p "tf-$V.whl" transformers/models/inkling2_loop/modeling_inkling2_loop.py 2>/dev/null | wc -l
  for pat in Inkling2LoopCache loop_start_layer loop_num_layers; do
    n=$(unzip -p "tf-$V.whl" "transformers/models/inkling2_loop/*.py" 2>/dev/null | grep -c "$pat")
    echo "  $pat: $n"
  done
done
echo WHEELCHECK_DONE
