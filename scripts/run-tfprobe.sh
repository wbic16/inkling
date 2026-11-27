#!/bin/bash
./rse-fpsolve/venv/bin/python - <<'EOF'
import transformers, pkgutil
print("transformers", transformers.__version__)
import importlib
try:
    m = importlib.import_module("transformers.models.inkling2_loop.modeling_inkling2_loop")
    names = [n for n in dir(m) if "Inkling" in n or "Cache" in n]
    print("modeling_inkling2_loop symbols:", names)
except Exception as e:
    print("module import fail:", e)
try:
    c = importlib.import_module("transformers.models.inkling2_loop.configuration_inkling2_loop")
    print("config symbols:", [n for n in dir(c) if "Inkling" in n])
except Exception as e:
    print("config import fail:", e)
import os
d = os.path.join(os.path.dirname(transformers.__file__), "models", "inkling2_loop")
print("module dir:", os.listdir(d) if os.path.isdir(d) else "MISSING")
EOF
ls ~/inkling-hf-base/*.py 2>/dev/null; grep -o '"[^"]*\.py"' ~/inkling-hf-base/manifest.tsv 2>/dev/null | head; awk -F'\t' '{print $1}' ~/inkling-hf-base/manifest.tsv | grep -E "\.py$|auto_map" ; grep -o "auto_map[^}]*" ~/inkling-hf-base/config.json 2>/dev/null | head -c 400
