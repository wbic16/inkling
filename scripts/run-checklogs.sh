#!/bin/bash
ls -la ~/rse-fpsolve/*.log ~/rse-fpsolve/probe_looping*.py 2>/dev/null
echo "=== run-probe.sh ==="
cat ~/rse-fpsolve/run-probe.sh
echo "=== probe-looping2.log (filtered) ==="
grep -avE "Loading weights|LOAD REPORT|UNEXPECTED|Notes:|^Key |^-{3}|^\s*$" ~/rse-fpsolve/probe-looping2.log 2>/dev/null | tail -20
