#!/bin/bash
set -u
cd ~/rse-fpsolve
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
./venv/bin/pip install "transformers==4.57.3" > tf-downgrade.log 2>&1
echo "pip rc=$?"
./venv/bin/python -c "
import transformers
print('transformers', transformers.__version__)
from transformers import Inkling2LoopForCausalLM
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopCache
print('both symbols OK')
"
