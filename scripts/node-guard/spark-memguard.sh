#!/usr/bin/env bash
# Stop inference before an over-allocation takes the node's userspace down.
#
# Must run ON the node: a watchdog reached over SSH dies with the thing it is
# watching. earlyoom as shipped (-m 2, ~2.5 GB) fires far below the point
# where sshd stops accepting connections.
#
# Two numbers matter here and they differ by nearly 10x on a loaded GB10:
#
#   MemAvailable  counts reclaimable page cache, and is what `free -g` prints
#                 in its last column. This governs whether userspace stays
#                 responsive, so it is the stop trigger.
#   MemFree       is what the NVIDIA driver actually needs, because on GB10
#                 every GPU allocation is host DRAM and the allocator fails
#                 rather than reclaiming cache. Measured at 1.4 GB while
#                 MemAvailable read 11.7 GB.
#
# MemFree sits low in normal operation, so it is logged rather than acted on.
# A sustained low reading during a load means the next big allocation will
# fail; that is what scripts/cache-flusher.sh is for.
#
# The MemFree-versus-MemAvailable distinction is from tonyd2wild's GB10
# KV-memory ladder study:
#   https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark
set -u
THRESHOLD_GB=${MEMGUARD_THRESHOLD_GB:-9}
GRACE_GB=${MEMGUARD_GRACE_GB:-14}
FREE_WARN_GB=${MEMGUARD_FREE_WARN_GB:-2}
INTERVAL=${MEMGUARD_INTERVAL:-5}
LOG="$HOME/memguard.log"

meminfo() { awk -v k="^$1:" '$0 ~ k {print int($2/1048576)}' /proc/meminfo; }
log() { echo "$(date -Is) $*" >> "$LOG"; }

log "memguard started: stop<${THRESHOLD_GB}G warn<${GRACE_GB}G free-warn<${FREE_WARN_GB}G"
warned=0
free_warned=0
while :; do
  a=$(meminfo MemAvailable)
  f=$(meminfo MemFree)
  c=$(meminfo Cached)

  if [ "${a:-99}" -lt "$THRESHOLD_GB" ]; then
    log "CRITICAL ${a}G available (MemFree ${f}G, Cached ${c}G) - stopping inference"
    "$HOME/.local/bin/sparkrun" stop --all >/dev/null 2>&1 || true
    docker ps -q --filter name=sparkrun | xargs -r docker rm -f >/dev/null 2>&1 || true
    sleep 20
    log "after stop: $(meminfo MemAvailable)G available, $(meminfo MemFree)G free"
    warned=0; free_warned=0
  elif [ "${a:-99}" -lt "$GRACE_GB" ]; then
    [ "$warned" = 0 ] && { log "WARN ${a}G available (MemFree ${f}G, Cached ${c}G)"; warned=1; }
  else
    warned=0
  fi

  # Low MemFree with a fat page cache is the signature of an allocation that
  # is about to fail even though MemAvailable looks comfortable.
  if [ "${f:-99}" -lt "$FREE_WARN_GB" ] && [ "${c:-0}" -gt 20 ]; then
    [ "$free_warned" = 0 ] && { log "WARN MemFree ${f}G with Cached ${c}G - GPU allocations may fail; consider cache-flusher.sh"; free_warned=1; }
  else
    free_warned=0
  fi

  sleep "$INTERVAL"
done
