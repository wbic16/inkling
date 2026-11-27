#!/bin/bash
echo "host=$(hostname)"
df -h / /tmp $HOME /mnt/mirrorborn/shared 2>/dev/null | sort -u
echo "--- local write 256MB ---"
dd if=/dev/zero of=/tmp/speedtest.bin bs=4M count=64 conv=fdatasync 2>&1 | tail -1; rm -f /tmp/speedtest.bin
echo "--- nfs write 256MB ---"
dd if=/dev/zero of=/mnt/mirrorborn/shared/speedtest-$$.bin bs=4M count=64 conv=fdatasync 2>&1 | tail -1; rm -f /mnt/mirrorborn/shared/speedtest-$$.bin
echo SPEED_DONE
