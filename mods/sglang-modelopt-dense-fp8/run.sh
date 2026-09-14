#!/usr/bin/env bash
# FP8 at load for the BF16 linears of a ModelOpt NVFP4 checkpoint (LibertAIDAI / nvidia GLM-5.3-Flash).
# Those checkpoints quantize only the routed experts; attention, KDA projections, shared experts and
# the dense MLPs stay BF16, about 9 GB per rank at TP=2 that every decode step reads (the active
# NVFP4 experts are ~1.6 GB). With SGLANG_MODELOPT_DENSE_FP8=1 the excluded LinearBase layers get
# Fp8LinearMethod (weights quantized per output channel at load, dynamic per-token activations);
# lm_head, the MoE router, the DSA indexer and the MTP eh_proj stay BF16. Same idea as
# sglang-glm53-dense-fp8 for compressed-tensors and Mia's GLM53_DENSE_FP8 on vLLM. Idempotent, fail-closed.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/layers/quantization/modelopt_quant.py"
s = p.read_text()
tag = "[modelopt-dense-fp8]"
if tag in s:
    print("modelopt-dense-fp8: already applied"); sys.exit(0)
old = ("        if isinstance(layer, (LinearBase, ParallelLMHead)):\n"
       "            if is_layer_skipped(\n"
       "                prefix, self.exclude_modules, self.packed_modules_mapping\n"
       "            ) or self.is_layer_excluded(prefix):\n"
       "                return UnquantizedLinearMethod()\n"
       "            return Linear(self)\n")
new = ("        if isinstance(layer, (LinearBase, ParallelLMHead)):\n"
       "            if is_layer_skipped(\n"
       "                prefix, self.exclude_modules, self.packed_modules_mapping\n"
       "            ) or self.is_layer_excluded(prefix):\n"
       "                import os  # " + tag + "\n"
       "                if (\n"
       "                    os.environ.get(\"SGLANG_MODELOPT_DENSE_FP8\") == \"1\"\n"
       "                    and isinstance(layer, LinearBase)\n"
       "                    and not isinstance(layer, ParallelLMHead)\n"
       "                    and not any(t in prefix for t in (\".mlp.gate\", \".indexer.\", \"eh_proj\", \"lm_head\"))\n"
       "                ):\n"
       "                    if not hasattr(self, \"_dense_fp8_config\"):\n"
       "                        self._dense_fp8_config = Fp8Config(\n"
       "                            is_checkpoint_fp8_serialized=False, activation_scheme=\"dynamic\"\n"
       "                        )\n"
       "                    return Fp8LinearMethod(self._dense_fp8_config)\n"
       "                return UnquantizedLinearMethod()\n"
       "            return Linear(self)\n")
if s.count(old) != 1:
    print(f"modelopt-dense-fp8: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
s = s.replace(old, new)
import ast; ast.parse(s)
p.write_text(s); print(f"modelopt-dense-fp8: patched {p}")
PY
