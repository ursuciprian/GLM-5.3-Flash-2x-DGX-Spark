#!/usr/bin/env bash
# FP8 at load for the BF16 linears of a compressed-tensors NVFP4 checkpoint.
# RedHatAI/GLM-5.3-Flash-NVFP4 quantizes only the routed experts. Attention, KDA projections,
# shared experts and the three dense MLPs stay BF16: about 12 GB per rank at TP=2 that every
# token reads, roughly 40 ms of the 69 ms plain-decode step on GB10 (273 GB/s). SGLang already
# has the hook (compressed-tensors "linear_fp8_config", meant for mixed checkpoints) but only
# reads it from the checkpoint's quantization_config. This mod lets SGLANG_CT_LINEAR_FP8=1 turn
# it on from the environment (dynamic per-token activations, weights quantized at load), and
# keeps the MoE router, the DSA indexer and the MTP eh_proj in BF16. Idempotent, fail-closed.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/layers/quantization/compressed_tensors/compressed_tensors.py"
s = p.read_text()
if "[dense-fp8]" in s:
    print("dense-fp8: already applied"); sys.exit(0)
a_old = '        linear_fp8_config = None\n        if "linear_fp8_config" in config:\n'
a_new = ('        linear_fp8_config = None\n'
         '        import os  # [dense-fp8]\n'
         '        if "linear_fp8_config" not in config and os.environ.get("SGLANG_CT_LINEAR_FP8") == "1":\n'
         '            from sglang.srt.layers.quantization.fp8 import Fp8Config\n'
         '            linear_fp8_config = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")\n'
         '        if "linear_fp8_config" in config:\n')
b_old = '            if self.linear_fp8_config is not None:\n                return Fp8LinearMethod(self.linear_fp8_config)\n'
b_new = ('            if self.linear_fp8_config is not None and not any(\n'
         '                t in prefix for t in (".mlp.gate", ".indexer.", "eh_proj")\n'
         '            ):  # [dense-fp8] router, indexer and MTP projection stay BF16\n'
         '                return Fp8LinearMethod(self.linear_fp8_config)\n')
for old, new in ((a_old, a_new), (b_old, b_new)):
    if s.count(old) != 1:
        print(f"dense-fp8: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
    s = s.replace(old, new)
p.write_text(s); print(f"dense-fp8: patched {p}")
PY
