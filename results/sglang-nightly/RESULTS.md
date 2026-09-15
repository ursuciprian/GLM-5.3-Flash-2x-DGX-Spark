# GLM-5.3-Flash NVFP4 on SGLang, two DGX Sparks, 2026-09-14: the loop root cause and the promoted recipe

Every number in the 2026-09-13 section below was measured on `RedHatAI/GLM-5.3-Flash-NVFP4`. Those numbers are invalid as speed figures: that checkpoint produces repeating text on long prose, and repeating text is exactly what a block drafter accepts perfectly. The 2026-09-14 work found and replaced the cause.

## What loops, and what does not

Same image (`51a0563d`), same mods, same flags, same drafter, same probes. The perplexity probe feeds a fixed 1.7k-token passage (one paragraph repeated six times) and reads input-token log-probs; a healthy model shows a low first window and near-zero repeat windows.

| checkpoint | format | first-window NLL | tokens < -8 logprob | essay loops (24 tries, T 0 to 1.0) | gate | tool-eval short |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next NVFP4 (reference model) | ModelOpt | 1.27 | 3 | 0 | | |
| RedHatAI GLM-5.3-Flash-NVFP4 | compressed-tensors | 2.53 | 14 | 21 to 24, also at T=0 with no drafter | failed every arm | 44 to 57 |
| RedHatAI + Marlin MoE | compressed-tensors | 3.2 | 28 | loops | | |
| LibertAIDAI GLM-5.3-Flash-NVFP4 | ModelOpt | 1.26 | 4 | 0 | passed | 93 |
| LibertAIDAI + FP8 dense linears | ModelOpt | 1.21 | 2 | 0 | passed | |
| orcarouter Uncensored NVFP4 | compressed-tensors | null logprobs, token 0 | | | | |

Eliminated by measurement on the RedHatAI checkpoint before the swap: sampler and flashinfer sampling kernels (exact against CPU), DFLASH (the no-drafter arm loops identically at greedy), thinking on or off, the Marlin MoE kernel (loops too, worse perplexity), lenient, greedy-verify and typical acceptance mods (no effect), collapsing per-expert activation scales in the compressed-tensors MoE path (much worse perplexity, so the CUTLASS kernel honours per-expert scales). DSA triton is ROCm-only, KDA flashinfer needs page size 1, top-k torch needs `SGLANG_DSA_FUSE_TOPK=0`.

Corroboration: MiaAI-Lab GLM-5.3-Flash-NVFP4-Dual-DGX-Spark issue #1 and NVIDIA forum thread 381433 report the same repeated-garbage symptom for vLLM's native CUTLASS NVFP4 MoE on GB10; RandomLlama's SGLang recipe (forum 381703) serves the LibertAIDAI checkpoint cleanly on the same kernels.

## Promoted recipe

`recipes/sglang/glm53-sglang-nvfp4-nightly.yaml` recipe_version 2: LibertAIDAI checkpoint (`caca4e6a`), no explicit `--quantization` (the BF16 DFlash2 drafter otherwise inherits `modelopt_fp4`), `sglang-modelopt-dense-fp8` mod (FP8 at load for the 9 GB per rank of BF16 attention, KDA, shared-expert and dense-MLP weights; router, indexer, lm_head and eh_proj stay BF16), DFlash2 block 7, chunked prefill 8192, 12 mamba slots, thinking on at low effort. `--speculative-adaptive` is dropped: this nightly disables it for DFLASH ("only EAGLE/EAGLE3 are supported"), so every earlier "adaptive k=7" arm ran static k=7.

Speed at GLM's recommended sampling (T=1.0, top_p 0.95, thinking low), 512-token lanes, single stream:

| arm | code | prose | structured | prefill 8k tok/s | TTFT 8k | weights/rank | KV pool |
|---|---|---|---|---|---|---|---|
| LibertAIDAI, drafter fp4 | 35.9 (acc 4.15) | 19.2 (acc 2.17) | 51.2 (acc 4.85) | 1894 | 4.4 s | 87.9 GB | 131k |
| LibertAIDAI v2, drafter BF16 | 36.5 (acc 4.45) | 19.2 (acc 2.02) | 51.6 (acc 4.10) | 1660 | 5.0 s | 88.2 GB | 113k |
| **v2 + FP8 dense (promoted)** | **41.7 (acc 5.25)** | **23.4 (acc 2.12)** | **58.4 (acc 6.12)** | 1470 | 5.6 s | 84.5 GB | 262k |
| v2, block 4 | 34.0 (acc 3.48) | 23.3 (acc 2.12) | 40.4 (acc 3.95) | 840 | 9.9 s | 88.2 GB | 131k |

Greedy code alone: 45.2 tok/s (v2), 49.9 tok/s (promoted). Plain decode without a drafter is 14.4 tok/s (69 ms step); the drafter multiplies that by acceptance plus one. Prose acceptance of about 2.1 is the drafter's ceiling on this model; block 4 buys 21% on prose and loses 7% on code and 22% on structured, which bounds what an adaptive verify length could add.

Reference points on the same hardware: MiaAI-Lab EXL3 4bpw + vLLM: structured 62.9, code about 52, prose 27, at temp 0 with thinking off. Tony's vLLM NVFP4 route: 46.9 code.

## Why the step is 69 ms

Per rank, per decode step, the LibertAIDAI checkpoint reads about 6.2 GB of BF16 attention, KDA and indexer weights, 3.0 GB of BF16 shared experts, dense MLPs and embeddings, and 1.6 GB of NVFP4 routed experts (8 of 288 per layer, 42 layers), about 11 GB at roughly 250 GB/s effective, plus about 90 NCCL all-reduces over ConnectX-7 and the DSA indexer and top-k kernels. The FP8-dense mod halves the dominant BF16 read and took the step to about 55 ms. NVFP4 for the dense linears (an offline requant) would halve it again and is the next lever; fp8 KV matters only past 32k context.

## Load-time memory

The nvidia ModelOpt checkpoint (191 GB, 33 shards of 5.8 GB, 96.7 GB per rank) cannot load on either node: about 2.5 GB of anonymous memory per shard accumulates during the load until RAM and the 16 GB swap are exhausted and `earlyoom` (active on both nodes, `-m 2 -s 80`) SIGTERMs the scheduler, visible as `Rank 0 scheduler died during initialization (exit code: -15)`. SGLang's default 8-thread loader makes it worse (`enable_multithread_load: false` helps but is not enough). More swap and a softer earlyoom threshold fix it; re-sharding to 2 GB shards is the no-sudo alternative. Not pursued: that checkpoint keeps more BF16 than LibertAIDAI and would decode slower.

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
