#!/usr/bin/env bash
# Lenient (typical) acceptance for DFLASH sampled verification.
# --speculative-accept-threshold-single/-acc exist as server args but the DFLASH/DSpark sampled
# verify path hard-codes threshold_single=1.0, threshold_acc=1.0 in _accept_sampling_core, so at
# temperature 1.0 a greedy draft token survives only with probability p_target and the drafter
# lands about 1.3 tokens per step (14-17 tok/s against 50-65 greedy). This mod reads
# SGLANG_DFLASH_ACCEPT_THRESHOLD_SINGLE and SGLANG_DFLASH_ACCEPT_THRESHOLD_ACC (defaults 1.0 =
# stock, exact rejection sampling) and passes them to the kernel. Values below 1 accept more draft
# tokens at the cost of the exact sampled distribution. Anchored, idempotent, fail-closed.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "../kernels/ops/speculative/dspark/dspark_accept.py"
p = p.resolve(); s = p.read_text()
if "[dflash-lenient-accept]" in s:
    print("dflash-lenient-accept: already applied"); sys.exit(0)
old = "        threshold_single=1.0,\n        threshold_acc=1.0,\n        deterministic=True,\n"
new = ("        threshold_single=float(__import__(\"os\").environ.get(\"SGLANG_DFLASH_ACCEPT_THRESHOLD_SINGLE\", \"1.0\")),  # [dflash-lenient-accept]\n"
       "        threshold_acc=float(__import__(\"os\").environ.get(\"SGLANG_DFLASH_ACCEPT_THRESHOLD_ACC\", \"1.0\")),\n"
       "        deterministic=True,\n")
if s.count(old) != 1:
    print(f"dflash-lenient-accept: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
p.write_text(s.replace(old, new)); print(f"dflash-lenient-accept: patched {p}")
PY
