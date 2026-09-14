#!/usr/bin/env bash
# Chunked-prefill radix-insert race fix for hybrid (Mamba/KDA) radix caches, sgl-project/sglang#38319.
# MambaRadixCache.cache_unfinished_req inserts each prefill chunk's KV pages into the radix tree
# mid-prefill; if the request is then retracted or aborted, cache_finished_req frees those pages
# while the tree still references them. The next request sharing the prefix reads another
# request's pages and decodes an impossible token forever (the "!!!!" loop, token id 0 / 248319).
# Port of the closed PR sgl-project/sglang#38355 (andreasknopke) onto the 2026-09-11 nightly: the
# insert is skipped for chunked prefill and deferred to cache_finished_req. Controlled by
# SGLANG_DISABLE_CHUNKED_RADIX_INSERT (default 1 = fix on; 0 = stock behaviour). Anchored, idempotent.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
root = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/mem_cache"
edits = {
 "cache_init_params.py": [(
    "    chunked_prefill_size: Optional[int] = None\n",
    "    chunked_prefill_size: Optional[int] = None\n\n"
    "    # [radix-chunked-insert-fix] skip the radix insert during chunked prefill (sglang#38319)\n"
    "    disable_chunked_radix_insert: bool = False\n")],
 "kv_cache_builder.py": [(
    "    params = CacheInitParams(\n        disable=disable_radix_cache,\n",
    "    # [radix-chunked-insert-fix] sglang#38319: never let the radix tree reference pages of a\n"
    "    # request that can still be retracted. Default on; SGLANG_DISABLE_CHUNKED_RADIX_INSERT=0 restores stock.\n"
    "    import os as _os\n"
    "    _disable_chunked_radix_insert = _os.environ.get(\"SGLANG_DISABLE_CHUNKED_RADIX_INSERT\", \"1\") == \"1\"\n"
    "    if _disable_chunked_radix_insert:\n"
    "        logger.info(\"radix-chunked-insert-fix: chunked-prefill radix insert deferred to request completion (sglang#38319)\")\n"
    "    params = CacheInitParams(\n        disable=disable_radix_cache,\n"
    "        disable_chunked_radix_insert=_disable_chunked_radix_insert,\n")],
 "mamba_radix_cache.py": [(
    "        self.enable_mamba_extra_buffer_lazy = params.enable_mamba_extra_buffer_lazy\n",
    "        self.enable_mamba_extra_buffer_lazy = params.enable_mamba_extra_buffer_lazy\n"
    "        self.disable_chunked_radix_insert = getattr(params, \"disable_chunked_radix_insert\", False)  # [radix-chunked-insert-fix]\n"),
   ("            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)\n            return\n\n        token_ids = req.get_fill_ids()\n",
    "            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)\n            return\n\n"
    "        # [radix-chunked-insert-fix] chunked prefill: only track prefix_indices, insert at completion\n"
    "        if self.disable_chunked_radix_insert and chunked:\n"
    "            return _skip_cache_unfinished_req(req)\n\n"
    "        token_ids = req.get_fill_ids()\n")],
}
for name, reps in edits.items():
    p = root / name; s = p.read_text()
    if "[radix-chunked-insert-fix]" in s:
        print(f"radix-chunked-insert-fix: {name} already applied"); continue
    for old, new in reps:
        if s.count(old) != 1:
            print(f"radix-chunked-insert-fix: {name}: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
        s = s.replace(old, new)
    p.write_text(s); print(f"radix-chunked-insert-fix: patched {name}")
PY
