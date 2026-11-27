#!/bin/bash
M=~/rse-fpsolve/venv/lib/python3.12/site-packages/transformers/models/inkling2_loop/modeling_inkling2_loop.py
sed -n '1308,1486p' "$M"
