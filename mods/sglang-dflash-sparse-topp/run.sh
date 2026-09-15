#!/usr/bin/env bash
# Sampled DFLASH verification without the full-vocabulary top-p renorm.
# With the model's default sampling (temperature 1.0, top_p 0.95, no top_k) every verify step
# runs flashinfer's top-p renorm over all 154,880 logits for all block rows, eager, outside the
# CUDA graph: llama-benchy tg128 read 10 tok/s against 54 at temperature 0 on the same server.
# SGLang already has a sparse "top-k first" path (torch.topk, softmax and top-p on the kept
# columns, scatter back) but only takes it when the request sets top_k. This mod routes top-p-only
# requests through it with a synthetic top-k of SGLANG_DFLASH_SPARSE_TOPP_K (default 1024) tokens.
# Mass beyond the top 1024 tokens of a row is dropped before the nucleus cut; for a 0.95 nucleus
# on LLM decode rows that is far below the fp32 noise floor. Set the env var to 0 to disable.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/speculative/dflash_utils.py"
s = p.read_text()
if "[dflash-sparse-topp]" in s:
    print("dflash-sparse-topp: already applied"); sys.exit(0)
a_old = "    scaled_logits = next_token_logits / expanded_temperature\n    sparse_topk_applied = False\n\n    if use_sparse_topk and need_top_k:\n"
a_new = ("    scaled_logits = next_token_logits / expanded_temperature\n    sparse_topk_applied = False\n\n"
         "    # [dflash-sparse-topp] top-p-only requests take the sparse top-k-first path with a\n"
         "    # synthetic k, instead of a full-vocabulary renorm kernel per verify step.\n"
         "    import os as _os\n"
         "    _forced_top_ks = None\n"
         "    _sparse_k = int(_os.environ.get(\"SGLANG_DFLASH_SPARSE_TOPP_K\", \"1024\"))\n"
         "    if use_sparse_topk and need_top_p and not need_top_k and _sparse_k > 0:\n"
         "        _sparse_k = min(_sparse_k, int(scaled_logits.shape[-1]))\n"
         "        need_top_k = True\n"
         "        max_top_k = _sparse_k\n"
         "        uniform_top_k_value = _sparse_k\n"
         "        _forced_top_ks = torch.full(\n"
         "            (sampling_info.temperatures.shape[0],), _sparse_k, dtype=torch.int64, device=device\n"
         "        )\n\n"
         "    if use_sparse_topk and need_top_k:\n")
b_old = "        repeated_top_ks = torch.repeat_interleave(\n            sampling_info.top_ks, draft_token_num, dim=0\n        ).to(dtype=torch.int64)\n"
b_new = ("        repeated_top_ks = torch.repeat_interleave(\n"
         "            _forced_top_ks if _forced_top_ks is not None else sampling_info.top_ks,\n"
         "            draft_token_num,\n            dim=0,\n        ).to(dtype=torch.int64)\n")
for old, new in ((a_old, a_new), (b_old, b_new)):
    if s.count(old) != 1:
        print(f"dflash-sparse-topp: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
    s = s.replace(old, new)
p.write_text(s); print(f"dflash-sparse-topp: patched {p}")
PY
