#!/bin/bash
# Stage 1: download thinkingmachines/Inkling-2-Mini (BASE ONLY) hub -> fleet NFS.
# Resumable (curl -C -), sha256-verified against HF API LFS metadata.
set -u
DEST=/mnt/mirrorborn/shared/models/inkling-2-mini-hf
mkdir -p "$DEST"
cd "$DEST"
LOG=$DEST/download.log
echo "=== download start $(date -Is) host=$(hostname) ===" >> "$LOG"

# Build manifest from HF API (with blob/LFS info)
python3 - <<'EOF' > manifest.tsv
import json, urllib.request
req = urllib.request.Request("https://huggingface.co/api/models/thinkingmachines/Inkling-2-Mini?blobs=true", headers={"User-Agent":"rse-run/1.0"})
d = json.load(urllib.request.urlopen(req, timeout=60))
print(f"# repo sha {d['sha']}")
for s in d["siblings"]:
    fn = s["rfilename"]
    lfs = s.get("lfs") or {}
    sha = lfs.get("sha256", "")
    size = s.get("size", lfs.get("size", 0))
    print(f"{fn}\t{size}\t{sha}")
EOF
head -1 manifest.tsv >> "$LOG"; wc -l manifest.tsv >> "$LOG"

fail=0
while IFS=$'\t' read -r fn size sha; do
  case "$fn" in \#*|"") continue;; esac
  mkdir -p "$(dirname "$fn")"
  url="https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/$fn"
  echo "--- $fn (size=$size sha=${sha:0:12}) $(date -Is)" >> "$LOG"
  curl -sL -C - --retry 5 --retry-delay 10 -o "$fn" "$url" >> "$LOG" 2>&1
  rc=$?
  actual=$(stat -c%s "$fn" 2>/dev/null || echo 0)
  if [ "$actual" != "$size" ]; then echo "SIZE MISMATCH $fn got=$actual want=$size rc=$rc" >> "$LOG"; fail=1; continue; fi
  if [ -n "$sha" ]; then
    got=$(sha256sum "$fn" | cut -d' ' -f1)
    if [ "$got" != "$sha" ]; then echo "SHA MISMATCH $fn got=$got want=$sha" >> "$LOG"; fail=1; continue; fi
    echo "OK $fn sha256=$got" >> "$LOG"
  else
    echo "OK $fn (no LFS sha; size verified)" >> "$LOG"
  fi
done < manifest.tsv

echo "=== download done fail=$fail $(date -Is) ===" >> "$LOG"
du -sh "$DEST" >> "$LOG"
touch "$DEST/DOWNLOAD_DONE_fail$fail"
