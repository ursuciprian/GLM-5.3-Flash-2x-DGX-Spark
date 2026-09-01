#!/usr/bin/env bash
# Hold the page cache down while a model loads.
#
# GB10 has no dedicated VRAM: every GPU allocation is host DRAM through the
# NVIDIA driver, and that path needs genuinely free pages. It fails rather
# than reclaiming clean page cache. Reading a 182 GiB checkpoint fills Cached
# and drives MemFree toward zero, so a KV allocation that "fits" according to
# MemAvailable can still fail.
#
# Run this alongside the load and stop it once the server is up.
#
# The page-cache mechanism, and the flusher-sidecar approach, are from
# tonyd2wild's GB10 KV-memory ladder study:
#   https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark
#   docs/GB10-KV-MEMORY-LADDER.md
# Confirmed here on 2026-08-28: MemFree 1.4G against MemAvailable 11.7G on a
# loaded node, an 8x gap.
set -uo pipefail
CEILING_GB="${CACHE_CEILING_GB:-15}"
INTERVAL="${CACHE_FLUSH_INTERVAL:-5}"
LOG="${CACHE_FLUSH_LOG:-$HOME/cache-flusher.log}"

meminfo() { awk -v k="^$1:" '$0 ~ k {print int($2/1048576)}' /proc/meminfo; }
log() { echo "$(date -Is) $*" >> "$LOG"; }

log "cache flusher started: ceiling ${CEILING_GB}G, interval ${INTERVAL}s"
while :; do
  cached=$(meminfo Cached)
  if [ "${cached:-0}" -gt "$CEILING_GB" ]; then
    free_before=$(meminfo MemFree)
    sync
    echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || \
      echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
    log "Cached ${cached}G > ${CEILING_GB}G, dropped; MemFree ${free_before}G -> $(meminfo MemFree)G"
  fi
  sleep "$INTERVAL"
done
