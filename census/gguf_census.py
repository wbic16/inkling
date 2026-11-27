#!/usr/bin/env python3
"""Read-only GGUF header census: metadata + tensor names/types. Header only."""
import struct, sys, collections

TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
         10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
         16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",
         22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"BF16",
         30:"Q4_0_4_4",31:"Q4_0_4_8",32:"Q4_0_8_8",34:"IQ1_M"}

def rstr(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def main(path):
    f = open(path, "rb")
    magic = f.read(4); assert magic == b"GGUF", magic
    version, = struct.unpack("<I", f.read(4))
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    print(f"version={version} tensors={n_tensors} metadata_kv={n_kv}")
    print("--- metadata ---")
    for _ in range(n_kv):
        k = rstr(f)
        vt, = struct.unpack("<I", f.read(4))
        if vt == 8: v = rstr(f)
        elif vt == 9:
            et, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
            sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
            if et == 8:
                v = [rstr(f) for _ in range(n)]
            else:
                f.seek(n * sizes.get(et, 8), 1); v = f"<array x{n}>"
        elif vt == 0: v = struct.unpack("<B", f.read(1))[0]
        elif vt == 1: v = struct.unpack("<b", f.read(1))[0]
        elif vt == 2: v = struct.unpack("<H", f.read(2))[0]
        elif vt == 3: v = struct.unpack("<h", f.read(2))[0]
        elif vt == 4: v = struct.unpack("<I", f.read(4))[0]
        elif vt == 5: v = struct.unpack("<i", f.read(4))[0]
        elif vt == 6: v = struct.unpack("<f", f.read(4))[0]
        elif vt == 7: v = struct.unpack("<?", f.read(1))[0]
        elif vt == 10: v = struct.unpack("<Q", f.read(8))[0]
        elif vt == 11: v = struct.unpack("<q", f.read(8))[0]
        elif vt == 12: v = struct.unpack("<d", f.read(8))[0]
        else: print("unknown vt", vt); return
        print(f"  {k} = {str(v)[:200]}")
    print("--- tensor type census (all) ---")
    tensors = []
    for _ in range(n_tensors):
        name = rstr(f)
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q"*nd, f.read(8*nd))
        tt, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tensors.append((name, dims, TYPES.get(tt, f"type{tt}")))
    c = collections.Counter(t[2] for t in tensors)
    for k, v in c.most_common(): print(f"  {k}: {v}")
    print("--- tensors in blk.6 .. blk.17 (looped block) ---")
    for name, dims, t in tensors:
        for L in range(6, 18):
            if name.startswith(f"blk.{L}."):
                print(f"  {name}  {t}  dims={dims}")
                break
    print("--- keyword tensors anywhere (mem/gate/router/conv/loop/recurr/expert) ---")
    for name, dims, t in tensors:
        low = name.lower()
        if any(k in low for k in ("mem", "gate", "router", "conv", "loop", "recurr")):
            print(f"  {name}  {t}  dims={dims}")
    print("--- first 12 tensor names ---")
    for name, dims, t in tensors[:12]: print(f"  {name}  {t}  dims={dims}")

if __name__ == "__main__":
    main(sys.argv[1])
