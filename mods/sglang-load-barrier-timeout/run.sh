#!/usr/bin/env bash
# Raise SGLang's hard-coded 480 s weight-load barrier to one hour.
# On this pair the worker's page cache is warm (the checkpoint was mirrored to it over CX-7) while
# the head reads cold from NVMe and first builds the multimodal processor, so rank 1 finishes
# loading minutes before rank 0 starts. After 480 s rank 1 raises "TP rank 1 could finish the
# model loading, but there are other ranks that didn't finish loading" and the head is killed at
# 45% of its shards. --dist-timeout does not cover this barrier. Idempotent, fail-closed.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/model_executor/model_runner_components/load_model_utils.py"
s = p.read_text()
old = "UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480"
if "[load-barrier-timeout]" in s:
    print("load-barrier-timeout: already applied"); sys.exit(0)
if s.count(old) != 1:
    print(f"load-barrier-timeout: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
p.write_text(s.replace(old, "UNBALANCED_MODEL_LOADING_TIMEOUT_S = 3600  # [load-barrier-timeout] cold head vs warm worker"))
print(f"load-barrier-timeout: patched {p}")
PY
