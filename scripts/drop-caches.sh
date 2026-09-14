#!/usr/bin/env bash
# Drop the page cache on the head and the worker without host sudo: a one-shot privileged
# container can write /proc/sys/vm/drop_caches. On GB10 the driver allocates from MemFree and
# does not reclaim clean page cache itself, so a warm cache starves weight loading.
# Usage: scripts/drop-caches.sh [worker-ip]   (default worker 192.168.100.53)
set -euo pipefail
IMG="${DROP_CACHES_IMAGE:-lmsysorg/sglang@sha256:51a0563d41fd57bf2e3a531e7ae83ac52a0a5068de4660987be770aae5730c47}"
WORKER="${1:-192.168.100.53}"
drop() { docker run --rm --privileged --entrypoint sh "$IMG" -c 'sync; echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1; free -g | awk '/^Mem:/{printf "free %sG cache %sG\n", $7, $6}'; }
echo -n "head:   "; drop
echo -n "worker: "; ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "IMG='$IMG'; $(declare -f drop); drop"
