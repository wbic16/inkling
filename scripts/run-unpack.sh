#!/bin/bash
set -u
cd ~/rse-fpsolve || exit 1
tar xzf rse-pkg.tgz
ls rse_looper/
echo "commit: $(cat COMMIT.txt)"
./venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import rse_looper.fpsolve as f, rse_looper.checkpoint as c, rse_looper.looping as l
print('rse_looper imports OK; submodules:', len(f.DEFAULT_FPSOLVE_SUBMODULES))
import torch, transformers
print('torch', torch.__version__, 'transformers', transformers.__version__)
"
echo UNPACK_DONE
