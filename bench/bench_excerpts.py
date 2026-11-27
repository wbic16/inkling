#!/usr/bin/env python3
import json, os
B = os.path.expanduser("~/rse-fpsolve/bench")
for leg in ["fpsolve4", "base4"]:
    r = json.load(open(os.path.join(B, f"{leg}.json")))
    print(f"===== {leg} phenotype =====")
    for k, v in r["phenotype"].items():
        print(f"--- {k} (loop={v['loop']}):")
        print(v["text"][:500])
    rows = r["gsm8k"]["rows"]
    lp = next((x for x in rows if x["loop"]), None)
    ok = next((x for x in rows if x["correct"]), None)
    if lp:
        print(f"--- {leg} first LOOPING gsm8k row (i={lp['i']}, gold={lp['gold']}, got={lp['got']}):")
        print(lp["out"][:500])
    if ok:
        print(f"--- {leg} first CORRECT gsm8k row (i={ok['i']}, gold={ok['gold']}, got={ok['got']}):")
        print(ok["out"][:400])
print("===== thermal.log =====")
p = os.path.expanduser("~/rse-fpsolve/work/thermal.log")
print(open(p).read()[-800:] if os.path.exists(p) else "NO thermal.log")
