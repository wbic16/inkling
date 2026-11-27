#!/bin/bash
grep -avE "Loading weights|LOAD REPORT|UNEXPECTED|Notes:|^Key |^-{3}|^\s*$" ~/rse-fpsolve/probe-looping2.log | tail -25
