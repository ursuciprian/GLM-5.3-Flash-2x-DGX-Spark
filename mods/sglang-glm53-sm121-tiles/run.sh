#!/usr/bin/env bash
# SM121 tile caps for SGLang's tilelang DSA kernels (GLM-5.3-Flash sparse MLA).
# GB10 exposes 101,376 B of dynamic shared memory per block; the stock tiles (block_I=64,
# 256/384 threads, 2 stages) request up to 169,984 B and the kernels fail to launch. The
# reference values below come from randomllama's GB10 ladder (2026-08-28): block_I=32, 128
# threads, single stage on the prefill factory; the dual head_per_block=4 launch keeps 256.
# Idempotent; every anchor must match exactly the expected number of times or nothing is written.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
# locate without importing: hooks run as root and an import would create root-owned kernel caches
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "kernels/ops/attention/dsa/tilelang_kernel.py"
if not p.exists():
    print(f"sm121-tiles: {p} missing; refusing"); sys.exit(1)
s = p.read_text()
if "[sm121-tiles]" in s:
    print("sm121-tiles: already applied"); sys.exit(0)
edits = [
    ("    block_I=64,\n    num_stages=2,\n    threads=256,", "    block_I=32,\n    num_stages=1,\n    threads=128,", 1),
    ("    block_I=64,", "    block_I=32,", 3),
    ("    block_I: int = 64,", "    block_I: int = 32,", 3),
    ("\n    threads=256,", "\n    threads=128,", 3),
    ("    threads = 384", "    threads = 128", 1),
]
out = s
for old, new, n in edits:
    c = out.count(old)
    if c != n:
        print(f"sm121-tiles: anchor {old!r} matched {c} times, expected {n}; refusing to patch"); sys.exit(1)
    out = out.replace(old, new)
out = out.replace('tilelang.set_log_level("WARNING")', 'tilelang.set_log_level("WARNING")  # [sm121-tiles] block_I=32, 128 threads, 1 stage', 1)
p.write_text(out); print(f"sm121-tiles: patched {p}")
PY
