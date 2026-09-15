#!/usr/bin/env bash
# compressed-tensors NVFP4 MoE on flashinfer_cutlass: collapse the per-expert activation global
# scales to one scalar, as the ModelOpt path does. The CUTLASS fused-MoE kernel quantizes the
# activations with a single global scale per GEMM; the compressed-tensors scheme passes the raw
# per-expert vector (w2_input_global_scale, and min-over-gate/up per expert for w13). In the
# RedHatAI GLM-5.3-Flash checkpoint down_proj input_global_scale differs 20-33x between experts of
# one layer, so the alphas and the activation quantization disagree for most experts and the
# second GEMM runs far off scale. min() keeps the widest range (compressed-tensors global scales
# are multipliers: smaller = larger amax). Idempotent, fail-closed.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py"
s = p.read_text()
tag = "[ct-nvfp4-scalar-act-scale]"
if tag in s:
    print("ct-nvfp4-scalar-act-scale: already applied"); sys.exit(0)
a_old = ("            w13_input_global_scale = layer.w13_input_global_scale.min(dim=1).values.to(\n"
         "                torch.float32\n"
         "            )\n")
a_new = ("            w13_input_global_scale = (  # " + tag + " one scalar for the whole GEMM, as ModelOpt does\n"
         "                layer.w13_input_global_scale.min().to(torch.float32).expand(layer.num_local_experts)\n"
         "            )\n")
b_old = "            w2_input_global_scale = layer.w2_input_global_scale\n"
b_new = ("            w2_input_global_scale = (  # " + tag + "\n"
         "                layer.w2_input_global_scale.min().to(torch.float32).expand(layer.num_local_experts)\n"
         "            )\n")
for name, old in (("w13", a_old), ("w2", b_old)):
    if s.count(old) != 1:
        print(f"ct-nvfp4-scalar-act-scale: anchor {name} matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
s = s.replace(a_old, a_new).replace(b_old, b_new)
import ast; ast.parse(s)
p.write_text(s)
print(f"ct-nvfp4-scalar-act-scale: patched {p}")
PY
