#!/usr/bin/env python3
"""Transport: build derived GGUF from the fleet AD GGUF + FP-Solve deltas.

- Edited tensors (eelbaz FP-Solve targets only) are reconstructed in HF checkpoint
  space via rse_looper.checkpoint.transformers_name_to_ckpt_deltas (their mapping,
  unmodified), converted to GGUF orientation, stored F32.
- Every other tensor's raw bytes are copied verbatim; the script re-parses its own
  output and proves per-tensor sha256 equality for all unedited tensors, and numeric
  equality (round-trip through GGUF layout) for edited ones.
- Metadata KV section is copied byte-for-byte from the source GGUF.
Credit: FP-Solve + name mapping are Eric E. / Tensorbend.ai's work.
"""
import argparse, hashlib, json, struct, sys

import numpy as np

GGUF_MAGIC = 0x46554747
# ggml value types for metadata parsing
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(13)
FIXED = {T_UINT8: 1, T_INT8: 1, T_UINT16: 2, T_INT16: 2, T_UINT32: 4, T_INT32: 4, T_FLOAT32: 4, T_BOOL: 1, T_UINT64: 8, T_INT64: 8, T_FLOAT64: 8}
# ggml tensor type sizes: only needed for types present in this file
BLOCK_SIZES = {6: (32, 22), 8: (32, 34), 12: (256, 144), 13: (256, 176)}  # Q5_0, Q8_0, Q4_K, Q5_K -> (block elems, block bytes)


def tensor_nbytes(ttype, dims):
    n = 1
    for d in dims:
        n *= d
    if ttype in (0,):
        return n * 4
    if ttype == 1:
        return n * 2
    be, bb = BLOCK_SIZES[ttype]
    assert n % be == 0, (ttype, dims)
    return n // be * bb


class Reader:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        self.pos = 0

    def u32(self):
        v = struct.unpack_from("<I", self.buf, self.pos)[0]; self.pos += 4; return v

    def u64(self):
        v = struct.unpack_from("<Q", self.buf, self.pos)[0]; self.pos += 8; return v

    def string(self):
        n = self.u64(); s = self.buf[self.pos:self.pos + n]; self.pos += n; return s.decode("utf-8")

    def skip_value(self, vtype):
        if vtype == T_STRING:
            n = self.u64(); self.pos += n
        elif vtype == T_ARRAY:
            et = self.u32(); cnt = self.u64()
            if et == T_STRING:
                for _ in range(cnt):
                    n = self.u64(); self.pos += n
            else:
                self.pos += FIXED[et] * cnt
        else:
            self.pos += FIXED[vtype]


def parse_gguf(path):
    r = Reader(path)
    assert r.u32() == GGUF_MAGIC
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    kv_start = r.pos
    alignment = 32
    for _ in range(n_kv):
        k = r.string(); vt = r.u32()
        if k == "general.alignment":
            alignment = struct.unpack_from("<I", r.buf, r.pos)[0]
        r.skip_value(vt)
    kv_end = r.pos
    kv_bytes = r.buf[kv_start:kv_end]
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        dims = tuple(r.u64() for _ in range(n_dims))
        ttype = r.u32()
        off = r.u64()
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    data_start = (r.pos + alignment - 1) // alignment * alignment
    for t in tensors:
        t["data_off"] = data_start + t["offset"]
        t["nbytes"] = tensor_nbytes(t["type"], t["dims"])
    return {"version": version, "kv_bytes": kv_bytes, "n_kv": n_kv, "tensors": tensors,
            "alignment": alignment, "buf": r.buf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-gguf", required=True)
    ap.add_argument("--edited-tf", required=True, help="edited TF-space tensors (reconstruct_tf.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    sys.path.insert(0, "/home/wbic16/rse-fpsolve")
    import torch
    from safetensors.torch import load_file

    # P2: edited tensors arrive already reconstructed in TF space
    # (reconstruct_tf.py via eelbaz's apply_weight_deltas). eelbaz's ckpt-space
    # mapping targets a checkpoint layout the published checkpoint does not use.
    edited_tf = load_file(args.edited_tf)
    print(f"edited TF tensors: {len(edited_tf)}", flush=True)

    # TF -> GGUF (torch orientation)
    KIND = {"q": "q", "k": "k", "v": "v", "mlp": "ffn"}
    edited_gguf = {}
    for tf_name, t in edited_tf.items():
        parts = tf_name.split(".")
        layer = int(parts[2])
        if tf_name.endswith("_message_gate.down_proj.weight"):
            kind = KIND[parts[3].replace("_message_gate", "")]
            edited_gguf[f"blk.{layer}.loop_{kind}_ctrl_down.weight"] = t
        elif tf_name.endswith("_message_gate.forget_gate_proj.weight"):
            kind = KIND[parts[3].replace("_message_gate", "")]
            edited_gguf[f"blk.{layer}.loop_{kind}_ctrl_forget.weight"] = t
        elif tf_name.endswith("_message_gate.message_gate_proj.weight"):
            kind = KIND[parts[3].replace("_message_gate", "")]
            edited_gguf[f"blk.{layer}.loop_{kind}_ctrl_gain.weight"] = t
        elif tf_name.endswith(".self_attn.o_proj.linear.weight"):
            edited_gguf[f"blk.{layer}.attn_output.weight"] = t
        elif tf_name.endswith(".mlp.gate.weight"):
            edited_gguf[f"blk.{layer}.ffn_gate_inp.weight"] = t
        else:
            raise ValueError(f"unmapped edited TF tensor: {tf_name}")
    print(f"edited gguf tensors: {len(edited_gguf)}", flush=True)

    EXPECT = {"ctrl_down": (64, 4608), "ctrl_forget": (2304, 64), "ctrl_gain": (2304, 64),
              "attn_output": (2304, 3072), "ffn_gate_inp": (128, 2304)}
    for name, t in edited_gguf.items():
        for k, shp in EXPECT.items():
            if k in name:
                assert tuple(t.shape) == shp, (name, tuple(t.shape), shp)

    src = parse_gguf(args.src_gguf)
    by_name = {t["name"]: t for t in src["tensors"]}
    missing = [n for n in edited_gguf if n not in by_name]
    assert not missing, f"edited names missing from GGUF: {missing[:5]}"

    # write output
    alignment = src["alignment"]
    out_tensors = []
    offset = 0
    blobs = []
    for t in src["tensors"]:
        name = t["name"]
        if name in edited_gguf:
            W = edited_gguf[name].numpy().astype(np.float32)
            assert tuple(reversed(W.shape)) == tuple(t["dims"]), (name, W.shape, t["dims"])
            # GGUF flat data order == torch row-major order (proven by
            # verify_orientation.py against the source file: cos = 1.000000).
            data = np.ascontiguousarray(W).tobytes()
            ntype = 0
        else:
            data = src["buf"][t["data_off"]:t["data_off"] + t["nbytes"]]
            ntype = t["type"]
        pad = (alignment - offset % alignment) % alignment
        offset += pad
        out_tensors.append({"name": name, "dims": t["dims"], "type": ntype, "offset": offset})
        blobs.append((pad, data))
        offset += len(data)

    with open(args.out, "wb") as f:
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, src["version"], len(out_tensors), src["n_kv"]))
        f.write(src["kv_bytes"])
        for t in out_tensors:
            nb = t["name"].encode()
            f.write(struct.pack("<Q", len(nb))); f.write(nb)
            f.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", t["type"]))
            f.write(struct.pack("<Q", t["offset"]))
        pos = f.tell()
        f.write(b"\x00" * ((alignment - pos % alignment) % alignment))
        for pad, data in blobs:
            f.write(b"\x00" * pad)
            f.write(data)
    print(f"wrote {args.out}", flush=True)

    # verify: re-parse output, hash-compare unedited, numeric-compare edited
    outp = parse_gguf(args.out)
    ob = {t["name"]: t for t in outp["tensors"]}
    assert outp["kv_bytes"] == src["kv_bytes"], "metadata differs!"
    bad = []
    n_same = 0
    for t in src["tensors"]:
        o = ob[t["name"]]
        if t["name"] in edited_gguf:
            raw = outp["buf"][o["data_off"]:o["data_off"] + o["nbytes"]]
            W = edited_gguf[t["name"]].numpy()
            A = np.frombuffer(raw, dtype="<f4").reshape(W.shape)  # flat order == torch row-major
            diff = np.abs(A - W).max()
            if diff > 1e-5:
                bad.append((t["name"], f"edited numeric diff {diff}"))
        else:
            h1 = hashlib.sha256(src["buf"][t["data_off"]:t["data_off"] + t["nbytes"]]).hexdigest()
            h2 = hashlib.sha256(outp["buf"][o["data_off"]:o["data_off"] + o["nbytes"]]).hexdigest()
            if h1 != h2 or o["type"] != t["type"] or o["dims"] != t["dims"]:
                bad.append((t["name"], "unedited mismatch"))
            else:
                n_same += 1
    report = {"edited": len(edited_gguf), "unedited_byte_identical": n_same,
              "total": len(src["tensors"]), "bad": bad,
              "out_sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest()}
    json.dump(report, open(args.report, "w"), indent=2)
    print(json.dumps(report, indent=2), flush=True)
    assert not bad, bad[:3]
    print("TRANSPORT VERIFIED", flush=True)


if __name__ == "__main__":
    main()
