# GLM-5.3-Flash on two DGX Sparks (GB10, TP=2)

Serving [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B
total, 18B active, NoPE sparse-MLA) on a pair of DGX Sparks. Nothing about
this model runs on GB10 out of the box: the MoE kernel the engines pick on
SM121 computes wrong math silently, the sparse-MLA backend asserts on the
head layout, and the checkpoint plus KV does not leave the host enough
memory unless every allocation is placed by hand. This repo carries the
fixes, the recipes for **vLLM** (MTP or DFlash2) and **SGLang** (DFlash2),
the EXL3 path, and every measurement.

**Headline, llama-benchy pp2048/tg128, prefix caching on, one boot each:**

| depth | vLLM MTP-5 c1 / c2 | vLLM DFlash2 c1 / c2 | vLLM DFlash2, fp8 KV c1 / c2 |
|---|---|---|---|
| 4096 | 25.5 / 26.3 | 19.9 / 23.6 | 23.0 / 25.8 |
| 8192 | 24.5 / 23.7 | 18.6 / 19.2 | - |
| 16384 | - | - | 23.6 / 16.8 |

Fresh 2k prefill at depth is 445-530 tok/s; context reuse prefill 1300-1680.
Full tables in [results/](results/). The same generation-length caveat as on
every model here: a 700-token measurement reads 1.4x lower than a 400-token
one on the same server, and that alone explained our first "gap" against a
published table.

The published reference for this model on this hardware is
[tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark).
At matched generation length we beat it at four of six concurrency levels
and the GB10 memory mechanism below is theirs; see Credits.

## Hardware

- 2x DGX Spark, GB10 / SM121, 128 GB LPDDR5X **unified** per node, ~273 GB/s.
- Two CX-7 links, both active at 200 Gb/s: `192.168.100.x` and
  `192.168.101.x`. WiFi `192.168.68.x` is for SSH only.
- Checkpoints: `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `11d73216` (182 GiB,
  ~74.5 GB of weights per node at TP=2), `RedHatAI` NVFP4 compressed-tensors
  as the alternative, EXL3 4bpw via MiaAI-Lab's image. The BF16 base does not
  fit on two Sparks.
- Drafter: `incoai/GLM-5.3-Flash-DFlash2`, k=7.

Unified memory is the fact behind everything here. Host and GPU share one
pool; an over-allocated server does not OOM, it starves sshd and needs a power
cycle. With 74.5 GB of weights per node this model runs 5-10 GB from that
edge for its entire life.

---

## Quick start

```sh
# once, both nodes: the memory guard. Not decorative, see below.
for h in dgx-01 dgx-02; do ssh $h 'nohup setsid ~/memguard.sh > /tmp/memguard.log 2>&1 < /dev/null &'; done

# every launch, both nodes
for h in dgx-01 dgx-02; do ssh $h 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'; done

ssh dgx-01 'cd ~/GEN-AI/glm53 && export PATH="$HOME/.local/bin:$PATH" && \
  nohup setsid sparkrun run recipes/vllm/glm53-vllm-dflash2-v9.yaml \
    --cluster dgx-cluster-cx7 --tp 2 > results/serve.log 2>&1 < /dev/null &'
```

Load takes ~15 minutes with the InstantTensor loader in the v9 image, ~10
without it because that image reads the whole 164 GB checkpoint through the
page cache and starves the driver. Watch `free -g` on **both** nodes; the
worker runs out first.

Benchmark the running server; never a plain `sparkrun benchmark`:

```sh
uvx llama-benchy@0.4.0 --base-url http://192.168.100.62:8000/v1 \
  --model glm-5.3-flash --tokenizer LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  --extra-body return_token_ids=false \
  --depth 0 4096 8192 16384 --pp 2048 --tg 128 --enable-prefix-caching \
  --concurrency 1 2 --save-result results/my-run.csv
```

`scripts/bench.sh` wraps this with a sanity gate that reads the *text* of a
completion first: the SM121 MoE trap passes health checks while producing
`lockLockLock` from the first token.

## Recipes

| recipe | engine | image | use |
|---|---|---|---|
| `vllm/glm53-vllm-dflash2-v9.yaml` | vLLM | `glm53-sm121:v9-dflash2` (built here) | **default.** SM121 NoPE-MLA stack, DFlash2 k=7, InstantTensor direct-IO load, NCCL re-pinned to 2.30.7 |
| `vllm/glm53-vllm-dflash2-sm121.yaml` | vLLM | `glm53-sm121:v8-dflash2` | same without the direct-IO loader |
| `vllm/glm53-vllm-mtp.yaml` | vLLM | `vllm-glm53:patched` | built-in MTP-5, the first configuration that served correctly |
| `vllm/glm53-vllm-dflash2-redhat*.yaml` | vLLM | `glm53-sm121:v9-dflash2` | RedHatAI compressed-tensors NVFP4 target, util 0.78 variant |
| `vllm/glm53-flash-nvfp4-dflash2-vllm.yaml` | vLLM | tonyd2wild `sm121-v11-dflash2` | the reference stack, verbatim, for A/B |
| `vllm/glm53-exl3-*.yaml` | vLLM fork | MiaAI-Lab / spark-arena EXL3 images | EXL3 4bpw, DFlash2 k=7; `-64k` and `-super` add context and fast-core pinning |
| `sglang/glm53-sglang-dflash2.yaml` | SGLang | `lmsysorg/sglang:glm-5.3-flash` | DFlash2 with the hidden-state capture overlay bind-mounted |
| `sglang/glm53-*-sglang-dflash2.yaml` | SGLang | same | RedHatAI and abliterated targets |

---

## Why it does not work out of the box, and the fixes

Everything below is measured or hit here; the recipes exist because of it.

**MoE kernel.** `--moe-backend marlin` is mandatory on vLLM. The
auto-selected FLASHINFER_CUTLASS NVFP4 MoE path computes wrong math on sm_121
with no error. Nearly every parameter is an expert, so all output is garbage.

**NoPE sparse MLA.** GB10's only sparse-MLA backend requires an `fp8_ds_mla`
cache and asserts `pe_dim == 64`; GLM-5.3 ships `qk_rope_head_dim: 0`.
`patches/patch_mla.py` (from kingjones30) builds MLA with rope 64 and
zero-pads q and k_pe. Zeros carry no rotation and add exactly 0 to the
logits, so it changes no number. It also widens the sm120 topk, extends the
dispatch allowlist so this head layout is routed to the sparse backend at
all, and fixes tail compaction. Four patches, anchor-asserted, loud on
mismatch.

**It has to be baked into the image.** sparkrun runs the container
unprivileged and the patcher writes into `dist-packages`, so a start-time
patch dies with `PermissionError`. `patches/Dockerfile.vllm-patched` and the
`patches/sm121/Dockerfile.*` ladder build the images; build once and
`docker save | ssh dgx-02 docker load`, so both nodes hold the same image ID,
or sparkrun re-syncs ~30 GB on every launch. Converting these four files to
bind-mounted overlays, the way the Qwen repos do it, is the next cleanup: run
the patcher once against copies pulled from the stock image, commit the
results, mount them.

**DFlash2 on vLLM** is a backport of upstream PR #52816 onto the image's
older vLLM tree: `patches/dflash2/` carries the drafter model file, the
speculator, and two anchored patchers for the registry and for the GLM
target's aux-hidden-state capture. `patches/dflash2/NOTES.md` and
`GLUE-NOTES.md` document every edit.

**DFlash2 on SGLang** needs one file: `patches/glm5_next.py`, the hidden-state
capture hook from sgl-project/sglang#36708, merged 13 hours *after* the public
`glm-5.3-flash` image was built. Bind-mounted; diff it against the image's
copy before trusting it, and expect ~36 changed lines, all capture logic.

**`--language-model-only`.** The multimodal processor inflates the API
front-end during chat-template init and the kernel OOM-kills it regardless of
GPU settings.

**Arch flags.** `TORCH_CUDA_ARCH_LIST=12.1f`, `CUTE_DSL_ARCH=sm_121a`,
`FLASHINFER_CUDA_ARCH_LIST=12.1a`. Configs copied from SM120 writeups say
`12.0f` and are wrong here.

## Memory on GB10: the part that costs a day

The driver allocates GPU memory from **genuinely free** pages and fails
rather than reclaiming clean page cache. The number that decides whether the
KV allocation lands is `MemFree`, not the `MemAvailable` that `free -g`
prints last and that every habit reaches for. Measured mid-load, 2026-08-28:

| | MemFree | MemAvailable | Cached |
|---|---|---|---|
| dgx-01 | **1.4 G** | 11.7 G | 13.4 G |
| dgx-02 | **1.5 G** | 13.6 G | 15.2 G |

Consequences, every one of them hit here:

- `gpu_memory_utilization` has a **floor and a ceiling** on this model. 0.78
  gave a KV pool of -0.6 GiB and vLLM refused to start; 0.84 starved the host
  to 8 GB and died on the first request. 0.81 sits between two known-bad
  endpoints. `--kv-cache-memory` with vLLM's *conservative* suggestion is the
  better knob; the larger "to fully utilize" figure is computed from
  MemAvailable and the driver cannot deliver it.
- **Load format matters as much as the fraction.** `auto` materialises the
  whole 164 GB checkpoint through the page cache on each node; InstantTensor
  direct-IO does not, and that difference is three boots we lost. The v9
  image installs it, and re-pins NCCL, because the install silently downgrades
  it to 2.29.7.
- `scripts/cache-flusher.sh` holds the page cache down through the load.
- **The memory guard must know the model.** A fixed 15 GB floor killed
  legitimate GLM boots that run at 5-10 GB free; earlyoom at 2 percent fires
  far too late. Run the guard with a per-model threshold, on the node, never
  over SSH: a watchdog reached over SSH dies with the thing it watches.
  Disabling it entirely cost one physical restart of both nodes.

## Reading results honestly

- Boot-to-boot variance is 15-25 percent, including acceptance length.
  Alternate A/B/A in one scripted run; never compare two boots.
- Generation length dominates measured tok/s. Match the other party's
  `max_tokens` before concluding anything about their config.
- Read acceptance from `/metrics` (`vllm:spec_decode_num_accepted_tokens_total`
  over `num_draft_tokens_total`), not from a throughput delta. Our DFlash2
  acceptance on this model on benchy prompts is ~0.42 per draft position,
  which is the same value the reference stack reports.

## Layout

```
recipes/vllm/  recipes/sglang/     one file per configuration, default first in the table above
patches/                           patch_mla.py + Dockerfile (vLLM base), sm121/ image ladder v3-v9,
                                   dflash2/ backport, upstream/ the three PR diffs, glm5_next.py (SGLang)
scripts/bench.sh                   sanity gate + grid; scripts/cache-flusher.sh; scripts/node-guard/
results/                           CSVs named by stack and grid, serve logs, spec-metrics
```

## Credits

- GB10 memory mechanism, KV ladder, flusher sidecar, and the reference
  benchmark table:
  [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
- vLLM SM121 patches and the NoPE-MLA approach:
  [kingjones30/GLM-5.3-Flash-2x-DGX-Spark](https://github.com/kingjones30/GLM-5.3-Flash-2x-DGX-Spark)
- EXL3 build and image:
  [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
- DFlash2 drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2);
  method z-lab / inco.ai, arXiv 2602.06036
- NVFP4 quant: [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
- SGLang hidden-state capture: sgl-project/sglang#36708
- Companion repos on the same hardware:
  [qwen3.8-flash-next-dgx-spark-tp-2](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2),
  [qwen3.8-27b-dgx-spark](https://github.com/ursuciprian/qwen3.8-27b-dgx-spark)
