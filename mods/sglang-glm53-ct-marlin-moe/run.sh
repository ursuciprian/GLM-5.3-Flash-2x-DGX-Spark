#!/usr/bin/env bash
# Marlin NVFP4 MoE runner for compressed-tensors checkpoints.
# SGLang's compressed-tensors W4A4 NVFP4 MoE scheme knows two runners, flashinfer_cutlass and
# flashinfer_trtllm. Neither is good at batch 1 on GB10 (cutlass grouped GEMM with one row per
# expert; trtllm has no SM121 cubins). The ModelOpt FP4 path already has a Marlin W4A16 branch
# (prepare_moe_nvfp4_layer_for_marlin + MarlinMoeQuantInfo). This mod ports that branch to the
# compressed-tensors scheme so --moe-runner-backend marlin works on RedHatAI/GLM-5.3-Flash-NVFP4.
# The scale conventions line up: the scheme already stores w13/w2_weight_scale_2 = 1/global_scale,
# which is the ModelOpt weight_scale_2 the Marlin repack expects. Idempotent, fail-closed.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py"
s = p.read_text()
if "[ct-marlin]" in s:
    print("ct-marlin: already applied"); sys.exit(0)
edits = [
 # 1. flag
 ("        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm()\n",
  "        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm()\n"
  "        self.use_marlin = get_moe_runner_backend().is_marlin()  # [ct-marlin]\n"),
 # 2. weight prep: repack for Marlin instead of swizzling block scales
 ("        else:\n            # swizzle weight scales\n",
  "        elif self.use_marlin:  # [ct-marlin]\n"
  "            from sglang.srt.layers.quantization.marlin_utils_fp4 import (\n"
  "                prepare_moe_nvfp4_layer_for_marlin,\n"
  "            )\n"
  "            if not hasattr(layer.quant_config, \"group_size\"):\n"
  "                layer.quant_config.group_size = 16\n"
  "            prepare_moe_nvfp4_layer_for_marlin(layer)\n"
  "        else:\n            # swizzle weight scales\n"),
 # 3. runner
 ("        else:\n            import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401 – triggers @register_fused_func\n",
  "        elif self.use_marlin:  # [ct-marlin]\n"
  "            import sglang.srt.layers.moe.moe_runner.marlin  # noqa: F401\n\n"
  "            self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)\n"
  "        else:\n            import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401 – triggers @register_fused_func\n"),
 # 4. apply
 ("        x = dispatch_output.hidden_states\n\n        if self.use_flashinfer_trtllm:\n",
  "        x = dispatch_output.hidden_states\n\n"
  "        if self.use_marlin:  # [ct-marlin]\n"
  "            from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo\n\n"
  "            expert_map = None\n"
  "            global_num_experts = -1\n"
  "            if hasattr(layer, \"dispatcher\") and hasattr(layer.dispatcher, \"local_expert_mapping\"):\n"
  "                expert_map = layer.dispatcher.local_expert_mapping\n"
  "                if expert_map is not None:\n"
  "                    global_num_experts = self.moe_runner_config.num_experts\n"
  "            quant_info = MarlinMoeQuantInfo(\n"
  "                w13_qweight=layer.w13_weight,\n"
  "                w2_qweight=layer.w2_weight,\n"
  "                w13_scales=layer.w13_weight_scale,\n"
  "                w2_scales=layer.w2_weight_scale,\n"
  "                w13_g_idx_sort_indices=None,\n"
  "                w2_g_idx_sort_indices=None,\n"
  "                weight_bits=4,\n"
  "                w13_global_scale=layer.w13_weight_scale_2,\n"
  "                w2_global_scale=layer.w2_weight_scale_2,\n"
  "                expert_map=expert_map,\n"
  "                global_num_experts=global_num_experts,\n"
  "            )\n"
  "            return self.runner.run(dispatch_output, quant_info)\n\n"
  "        if self.use_flashinfer_trtllm:\n"),
]
for old, new in edits:
    if s.count(old) != 1:
        print(f"ct-marlin: anchor matched {s.count(old)} times, expected 1: {old[:60]!r}; refusing"); sys.exit(1)
    s = s.replace(old, new)
p.write_text(s); print(f"ct-marlin: patched {p}")
PY
