"""Honour a KV cache spec's own dtype instead of the global --kv-cache-dtype.

Port of vllm-project/vllm#47618. Without it, a model whose MLA layers resolve
to fp8_ds_mla forces that dtype onto every other spec in the engine, including
a speculative drafter's sliding-window layers. Nothing can serve that
combination, and backend selection fails with:

  No valid attention backend found for cuda with AttentionSelectorConfig(
    head_size=128, kv_cache_dtype=fp8_ds_mla, use_mla=False,
    has_sliding_window=True, use_non_causal=True)
"""
import ast
import sys

P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/attn_utils.py"
src = open(P).read()

MARKER = 'getattr(kv_cache_spec, "cache_dtype_str", None) or cache_dtype'
if MARKER in src:
    print("per-layer kv dtype: already patched")
    sys.exit(0)

anchor = """                layer_cache_dtype = (
                    "auto"
                    if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE
                    else cache_dtype
                )"""
replacement = """                layer_cache_dtype = (
                    "auto"
                    if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE
                    else getattr(kv_cache_spec, "cache_dtype_str", None) or cache_dtype
                )"""

count = src.count(anchor)
if count != 1:
    raise SystemExit(f"expected 1 layer_cache_dtype anchor, found {count}")

src = src.replace(anchor, replacement)
ast.parse(src)
open(P, "w").write(src)
print("per-layer kv dtype patch applied + compiles")
