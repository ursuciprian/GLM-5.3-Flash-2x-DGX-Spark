#!/usr/bin/env bash
# Force greedy (argmax) verification for sampled requests on the DFLASH v2 worker.
# On the 2026-09-11 nightly the DFLASH sampled verify path (selector sampling + rejection kernel)
# produces token salad and repeated fragments at temperature 1.0 / top_p 0.95 on GLM-5.3-Flash,
# with and without the sparse top-p mod, while greedy verification is clean. SGLang already carries
# the fallback ("falling back to greedy argmax verification. The requested sampling distribution
# will not be preserved") for devices without the sampling kernel; this mod lets
# SGLANG_DFLASH_GREEDY_VERIFY=1 select it on purpose. Sampled requests then decode the target's
# greedy path: deterministic, no corrupted distribution, drafter acceptance as at temperature 0.
# Anchored, idempotent, fail-closed.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/speculative/dflash_worker_v2.py"
s = p.read_text()
if "[dflash-greedy-verify]" in s:
    print("dflash-greedy-verify: already applied"); sys.exit(0)
old = "        self._selector_sampling_enabled = not _is_npu\n"
new = ("        self._selector_sampling_enabled = not _is_npu and __import__(\"os\").environ.get(\n"
       "            \"SGLANG_DFLASH_GREEDY_VERIFY\", \"0\"\n"
       "        ) != \"1\"  # [dflash-greedy-verify] 1 = argmax verification for every request\n")
if s.count(old) != 1:
    print(f"dflash-greedy-verify: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
p.write_text(s.replace(old, new)); print(f"dflash-greedy-verify: patched {p}")
PY
