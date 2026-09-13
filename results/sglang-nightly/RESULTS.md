# GLM-5.3-Flash NVFP4 on SGLang, two DGX Sparks, 2026-09-13

First boots of GLM-5.3-Flash on a stock public SGLang image (the 2026-09-11 nightly, digest
`51a0563d`), TP=2 over ConnectX-7, DFlash2 drafter, thinking off. Recipe
`recipes/sglang/glm53-sglang-nvfp4-nightly.yaml`; arms under `recipes/arms/`.

## What it took to boot

Six boots, four mods, all under `mods/`:

| mod | why |
|---|---|
| `sglang-glm53-sm121-tiles` | tilelang DSA kernels request more shared memory than GB10 has; block_I 32, 128 threads, one stage |
| `sglang-glm53-ct-names` | RedHatAI's ignore list names the KDA forget gate `self_attn.forget_gate.f_a_proj`, SGLang flattens it to `self_attn.f_a_proj`; loading died with `Unable to find matching target`. One suffix rule in the weights mapper (SGLang's `WeightsMapper` applies one substring rule per name, so a second substring rule preempted the prefix strip) |
| `sglang-glm53-thinking-switch` | the checkpoint's chat template always opens `<think>`; the copy written by this mod honours `chat_template_kwargs {"enable_thinking": false}` |
| `sglang-load-barrier-timeout` | SGLang's 480 s weight-load barrier; the worker (warm page cache) finishes minutes before the head |

Two more found along the way: hooks run as root, so a mod that imports the engine leaves
root-owned kernel caches and the server dies with `Permission denied: /cache/runtime/tilelang/...`
(mods now locate files without importing); and the nvidia ModelOpt checkpoint (95 GB per rank)
exhausts the head's unified memory under parallel shard loading (single-threaded loading for
those arms).

## Probe results

Single stream, 400-800 tokens, thinking off, one fresh boot per arm, two probes per boot. The
acceptance figure is the last decode batch's gauge, so it is noisy. Prefill: an 8k prompt, first
request after boot pays JIT (about 1000 tok/s), steady state below.

| arm | code | prose | structured | prefill 8k | KV pool |
|---|---|---|---|---|---|
| control: d5, cutlass MoE, bf16 dense (4 boots) | 37-48 | 23-45 | 28-44 | 1880-1950 | 105-119k |
| d3 | 28-34 | 26-29 | 32 | 1920 | 115k |
| d7 | 32-33 | 40-44 | 49.5 | 1944 | 110k |
| adaptive k=7 | 45-54 | 35-41 | 37-40 | 1618 | 101k |
| chunked prefill 8192 | 46.5 | 29 | 44 | 1951 | 115k |
| 12 mamba slots | 33-35 | 37-38 | 35.5 | 1969 | 158k |
| mamba backend flashinfer | 37-46 | 29-36 | 35-42 | 1942 | 102k |
| dense FP8 (mod) | 43-49 | 29-45 | 39-53 | 1720-1820 | 247-262k |
| **dense FP8 + adaptive k=7 + cp8192 + 12 slots** | 42-46 | 32-50 | 45-60 | 1864 | 262k |
| no drafter | 14.4 | 14.4 | 14.4 | 1816 | 262k |

Failed arms: DSA `flashinfer_sparse_mla` (needs fp8 KV and accepts only the GLM-5 page layout),
MoE `flashinfer_trtllm` (no SM121 cubin), fp8 KV (same kernel wall), nvidia ModelOpt checkpoint
with cutlass / Marlin / CuTe DSL / Triton runners (head out of memory during load; rerun with
single-threaded loading pending).

## Reading

Plain decode is 69 ms per token, four times the bandwidth floor. Per rank per token: routed
experts 4.5 GB (16 ms), BF16 attention/KDA/shared/dense linears about 6 GB (22 ms, halved by the
dense-FP8 mod), lm_head 0.6 GB. The remaining 30-40 ms is kernel inefficiency at batch 1 (cutlass
grouped GEMM with one row per expert, 45 layers of KDA and DSA kernels, about 90 allreduces).
The drafter multiplies this by 3-4 on code and structured text, less on prose.

Against the published dual-Spark lanes: prose 2x tonyd2wild's vLLM NVFP4 lane (18.8) and ahead
of MiaAI-Lab's tuned EXL3 lane (32.1); code ties both (46.9 / 48.6) on the good boots; structured
behind both (54-65) by 10-25%; cold prefill behind Tony (2763) by a third. The only other SGLang
recipe for this model (randomllama, 28.6 code / 23.6 prose) is behind on every lane.

## Open

- Marlin NVFP4 MoE on the nvidia checkpoint (the small-batch kernel tonyd2wild uses on vLLM).
- llama-benchy pp2048/tg128 at depth 0 and 16k, c1 and c2, on the combo.
- tonyd2wild's 40-prompt harness and spark-bench on the winner, for the like-for-like table.
- fp8 KV: needs an SM121 sparse-MLA kernel for the NoPE layout with `index_kpool 4`.
- NVFP4 for the dense parts (offline requant), drafter at TP1, lm_head FP8.
