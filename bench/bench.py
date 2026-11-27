#!/usr/bin/env python3
"""Validation + GSM8K A/B harness (fleet llama-server, plain-view).

Legs: for each (model_label, gguf_path, loop_count): start server, run
(a) the two EE-1024 probe prompts (phenotype), (b) GSM8K test problems [0:n),
greedy, extract final number, detect repeat-loop degeneration.
Writes one JSON per leg + a summary. Same grader discipline for every leg.
"""
import json, os, re, subprocess, sys, time, urllib.request

SRV = os.path.expanduser("~/llama-bin/cpu/llama-server")
GSM = "/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/rse-bench/gsm8k-test.jsonl"
OUT = os.path.expanduser("~/rse-fpsolve/bench")
PORT = 18099


def wait_listen(logpath, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if "listening" in open(logpath).read():
                return True
        except FileNotFoundError:
            pass
        time.sleep(5)
    return False


def completion(prompt, n_predict):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["content"]


def gold_of(answer_field):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    return m.group(1).replace(",", "") if m else None


def extract_answer(text):
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def is_loop(text):
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 6]
    run = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        if run >= 5:
            return True
    return False


def num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def run_leg(label, gguf, loops, n_gsm):
    os.makedirs(OUT, exist_ok=True)
    logpath = os.path.join(OUT, f"{label}.server.log")
    args = [SRV, "-m", gguf, "--host", "127.0.0.1", "--port", str(PORT), "-c", "2048", "-t", "16"]
    if loops != 3:
        args += ["--override-kv", f"inkling2_loop.loop.count=int:{loops}"]
    with open(logpath, "w") as lf:
        proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT)
    result = {"label": label, "model": gguf, "loops": loops}
    try:
        if not wait_listen(logpath, proc):
            result["error"] = "server failed to start"
            return result
        # phenotype prompts (same as EE 1024 probe)
        ph = {}
        ph["counting"] = completion("What is 17 + 25? Give the number, then keep counting upward by 7 each step.", 48)
        ph["janet"] = completion("Janet has 3 baskets with 14 apples each. She gives away 9 apples. How many apples does she have left? Think step by step, then give the final number.", 96)
        result["phenotype"] = {k: {"text": v, "loop": is_loop(v)} for k, v in ph.items()}
        # GSM8K slice
        probs = [json.loads(l) for l in open(GSM)][:n_gsm]
        rows = []
        correct = loops_n = 0
        for i, p in enumerate(probs):
            prompt = p["question"] + "\n\nSolve step by step, and end with '#### <final number>'."
            try:
                out = completion(prompt, 384)
            except Exception as e:
                out = f"<error {e}>"
            gold = gold_of(p["answer"])
            got = extract_answer(out)
            ok = num_eq(got, gold)
            lp = is_loop(out)
            correct += ok; loops_n += lp
            rows.append({"i": i, "gold": gold, "got": got, "correct": ok, "loop": lp, "out": out[:600]})
            print(f"[{label}] {i+1}/{len(probs)} correct={correct} loops={loops_n}", flush=True)
        result["gsm8k"] = {"n": len(probs), "correct": correct, "loops": loops_n, "rows": rows}
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    # legs spec: label=gguf:loops,... via argv[1]; n via argv[2]
    spec = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    summary = []
    for leg in spec.split(","):
        label, gguf, loops = leg.rsplit(":", 2)
        r = run_leg(label, gguf, int(loops), n)
        json.dump(r, open(os.path.join(OUT, f"{label}.json"), "w"), indent=2)
        g = r.get("gsm8k", {})
        summary.append({"label": label, "loops": int(loops), "gsm_correct": g.get("correct"),
                        "gsm_n": g.get("n"), "gsm_loops": g.get("loops"),
                        "phenotype_loops": {k: v["loop"] for k, v in r.get("phenotype", {}).items()},
                        "error": r.get("error")})
        print(json.dumps(summary[-1]), flush=True)
        time.sleep(5)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
