#!/bin/bash
cd ~/rse-fpsolve || exit 1
./venv/bin/python probe_looping2.py > probe-looping2.log 2>&1
echo "probe rc=$?" >> probe-looping2.log
