# GLM-5.3-Flash on two DGX Sparks

Serving recipes for [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(320B MoE, 18B active, NoPE sparse MLA, KDA linear attention, vision) at tensor
parallel 2 across two NVIDIA DGX Spark (GB10, SM121) over ConnectX-7, packaged for
[sparkrun](https://sparkrun.dev). One SGLang recipe on a stock public nightly with
five small mods, and the vLLM lanes (DFlash2, MTP, EXL3) this repo grew from.
Nothing about this model runs on GB10 out of the box; these boot, and every
number here comes from a fresh boot on this pair.

## Quick start

```sh
sparkrun registry add https://github.com/ursuciprian/GLM-5.3-Flash-2x-DGX-Spark
sparkrun run @glm53/glm53-sglang-nvfp4-nightly --cluster <your-cluster> --tp 2 --trust
```

Or from a clone, the day's best configuration:

```sh
git clone https://github.com/ursuciprian/GLM-5.3-Flash-2x-DGX-Spark
cd GLM-5.3-Flash-2x-DGX-Spark
sparkrun run recipes/arms/glm-dense-fp8-combo.yaml --cluster <your-cluster> --tp 2 --trust --no-follow
```

`--trust` accepts the mod hooks that patch the engine inside the container (see
[Why the mods](#why-the-mods)). Boot takes 10-13 minutes warm. The server answers on
port 8000 with the OpenAI API, model name `glm-5.3-flash`. Thinking is off unless a
request sends `chat_template_kwargs: {"enable_thinking": true}`. Stop with
`sparkrun stop --all`.

Checkpoint: `RedHatAI/GLM-5.3-Flash-NVFP4` (compressed-tensors, 185 GB, about 87 GB of
weights per Spark) and the drafter `incoai/GLM-5.3-Flash-DFlash2`. Download once on the
head and mirror to the worker over the fast link; the checkpoint is too large to pull
twice comfortably.

### Which recipe

| You are serving | Recipe | What you get |
|---|---|---|
| Chat, code, agents | `recipes/arms/glm-dense-fp8-combo.yaml` | Best decode measured here: code 63-65 tok/s, prose 49-52, structured 60, single stream. 262k-token KV pool, 131k context. Dense linears in FP8, adaptive DFlash2 k=7, chunked prefill 8192. |
| The conservative baseline | `recipes/sglang/glm53-sglang-nvfp4-nightly.yaml` | Same stack with BF16 dense weights and fixed k=5: code 37-48, prose 23-45, structured 28-44, 105-119k pool. |
| vLLM lanes, historical | `recipes/vllm/*.yaml` | DFlash2 and MTP on the SM121-patched vLLM images built here, plus the EXL3 4bpw path on MiaAI-Lab's image. 19-26 tok/s on llama-benchy; kept for A/B against the reference stacks. |

Full arm tables in [results/sglang-nightly/RESULTS.md](results/sglang-nightly/RESULTS.md).

## How this works

A sparkrun recipe is one YAML: container digest, checkpoint revision, environment,
serve command. sparkrun starts the same container on both Sparks, wires NCCL over
ConnectX-7 and runs the serve command with `--tp 2`. Anything the engine needs beyond
its stock image is a **mod**: a directory under `mods/` that sparkrun copies into each
container and runs as a pre-serve hook. Mods here are small, anchored, idempotent and
fail closed; a patch that does not apply exactly stops the boot.

**Engine.** SGLang, the 2026-09-11 nightly (`lmsysorg/sglang@sha256:51a0563d...`),
the first public image with GLM-5.3-Flash support merged (sgl-project/sglang#36507),
the NVFP4 loader fixes (#38621) and the DFLASH v2 speculative worker. Attention:
`dsa` with tilelang prefill and decode kernels, triton KDA, `flashinfer_cutlass`
NVFP4 MoE, bf16 KV. DFlash2 block-diffusion drafter with flashinfer draft attention;
the combo recipe uses `--speculative-adaptive` at k=7, which shortens the verify block
when acceptance drops.

**Why the dense FP8 mod.** The checkpoint quantizes only the routed experts. Attention,
KDA projections, shared experts and the three dense MLPs stay BF16, about 6 GB per rank
that every token reads. The mod turns on an existing SGLang hook
(`linear_fp8_config`) from the environment: those linears are quantized to FP8 at load
with dynamic per-token activations, router and indexer stay BF16. Frees 3 GB per rank
and lifts the KV pool from 110k to 262k tokens.

**Measurement.** Fresh boot per arm, two probes per boot: 800 tokens of code, 500 of
prose, 400 of structured output, an 8k cold prefill, a cached turn. Four boots of the
baseline spread about 13% at one stream, so single-arm differences inside that band are
not results. Plain decode without the drafter is 14.4 tok/s (69 ms per token); the
drafter multiplies that by 3-4.5 depending on the text.

## Why the mods

| mod | what it fixes |
|---|---|
| `sglang-glm53-sm121-tiles` | The tilelang DSA kernels request up to 170 KB of shared memory; GB10 has 101 KB. Tiles shrink to block_I 32, 128 threads, one stage on the prefill factory. |
| `sglang-glm53-ct-names` | The checkpoint's ignore list names the KDA forget gate `self_attn.forget_gate.f_a_proj`; SGLang flattens it to `self_attn.f_a_proj`, so loading died with `Unable to find matching target`. One suffix rule in the weights mapper. SGLang's mapper applies one substring rule per name, longest first, so the rule has to live in the suffix stage. |
| `sglang-glm53-thinking-switch` | The chat template opens every assistant turn with `<think>` and has no switch. The mod writes a copy where thinking is opt-in via `enable_thinking`, so clients without kwargs get an answer in `content` and the glm45 reasoning parser splits correctly when thinking is on. |
| `sglang-load-barrier-timeout` | SGLang aborts if one TP rank finishes loading 480 s before the others. The worker loads from a warm page cache after the mirror, the head reads cold. Raised to an hour. |
| `sglang-glm53-dense-fp8` | FP8 at load for the BF16 linears, above. |
| `sglang-glm53-ct-marlin-moe` | Ports the ModelOpt Marlin NVFP4 MoE branch to the compressed-tensors scheme so `--moe-runner-backend marlin` works. Measured slower on GB10 (Marlin is W4A16 and skips the FP4 tensor cores). Kept for reference, not in a shipped recipe. |

All mods locate files by path instead of importing the engine: hooks run as root, and an
import creates root-owned kernel caches that kill the unprivileged server later.

## Gotchas

- **Fabric names are pinned** to this pair (`enp1s0f1np1`, `rocep1s0f1,roceP2p1s0f1`,
  GID 3). sparkrun's `-o` does not reach `env:`; edit the YAML for `f0` wiring.
- **Unified memory has no OOM.** An over-allocated server starves sshd and needs a power
  cycle. This model runs 9-11 GB from that edge. The nvidia ModelOpt checkpoint (95 GB
  per rank) took the head past it during load every time; the RedHatAI checkpoint fits.
- **Drop the page cache before loading** on both nodes; `free -g`'s available column
  lies to the driver, `MemFree` is what counts.
- **Thinking off makes GLM ramble.** With `<think></think>` prefilled, the model writes
  untagged planning prose into `content` ("The user wants me to..."), on every engine.
  Faster (+8% acceptance) but agent harnesses that parse `content` suffer. Turn thinking
  on for tool use and let the parser split.
- **Server log is inside the container** at `/tmp/sparkrun_serve.log`; `docker logs`
  shows only the CUDA banner. The worker's log holds the reason when the head dies
  with `Rank 0 scheduler died during initialization`.
- **Mods resolve beside the recipe directory**; keep `recipes/<dir>/mods -> ../../mods`.
- **First 8k prefill after boot runs at half speed** (about 1000 tok/s against 1900
  steady): JIT warm-up, not a config effect.
- **llama-benchy** needs `--extra-body return_token_ids=false` on SGLang, and its
  coherence probe sends no chat kwargs, which is why the template defaults to thinking
  off.

## Not working properly

- **Quality gate is open.** Tool-eval short scored 33 (thinking on default) and 47
  (thinking off) against Qwen3.8-Flash-Next's 97-100 on the same 15 scenarios; the
  25k-token four-marker retrieval passed on one arm and failed on another; a 1024-token
  essay flagged repetition on the FP8-dense arm and not on the BF16 one. The rambling
  thinking-off style is part of it. Being rerun with thinking on and per-arm; do not
  put this in front of agents until it is settled.
- **One silent collapse observed.** A run of `!` (token id 0) once on a coherence probe
  right after boot, not reproduced in five later boots. The known SM121 failure mode;
  under watch.
- **llama-benchy tg128 reads 8-12 tok/s** while every other lane reads 40-65 on the
  same server. Under investigation (sampling defaults or how the client counts DFlash
  chunks); the benchy grid is not published until resolved.
- **fp8 KV cache** is blocked: the only SM121 sparse-MLA kernel accepting fp8 knows the
  GLM-5 page layout (nope 448, rope 64); GLM-5.3-Flash has no rope dims and
  `index_kpool 4`. Tilelang refuses fp8 KV, trtllm has no SM121 cubins.
- **Prefill** 1900 tok/s steady against 2763 on the vLLM reference and 4100 on Qwen.
- **nvidia/GLM-5.3-Flash-NVFP4** runs the head out of memory during load at TP2.

## Models tested

One model, two checkpoints, two engines, all on this pair.

| | SGLang nightly, RedHatAI NVFP4, combo | SGLang nightly, baseline | vLLM v9, LibertAIDAI NVFP4, DFlash2 |
|---|---|---|---|
| decode, code, 1 stream | **63-65 tok/s** | 37-48 | 19.9 (benchy tg128) |
| decode, prose | **49-52** | 23-45 | - |
| decode, structured | **60** | 28-44 | - |
| no-drafter decode | 14.4 | 14.4 | - |
| cold prefill 8k | 1860 tok/s | 1900 | 445-530 (2k at depth) |
| cached turn | 0.7 s | 0.7 s | 1300-1680 tok/s |
| KV pool | **262k** | 105-119k | - |
| context | 131k | 131k | 32k |
| boot | 10-13 min | 10-13 min | 15 min |

Against the published dual-Spark lanes, their numbers from their READMEs, ours from the
combo recipe: tonyd2wild vLLM NVFP4 + DFlash2, code 46.9 and prose 18.8; MiaAI-Lab
EXL3 4bpw + DFlash2, code 48.6 and prose 32.1 (adaptive-k), structured 62.9-65.1;
randomllama SGLang NVFP4, code 28.6 and prose 23.6. We lead on code and prose, tie on
structured, trail on prefill and on KV capacity (their fp8 KV pools are 581k to 1.75M).

## Hardware

- 2x DGX Spark: GB10, SM121, 128 GB LPDDR5X unified, about 273 GB/s per node.
- ConnectX-7 RoCE between the nodes, GID index 3.
- 185 GB checkpoint plus 4 GB drafter per node.

## Layout

| Path | What |
|---|---|
| `recipes/sglang/` | The shipped SGLang recipe |
| `recipes/arms/` | Ladder arms, one file per configuration; `glm-dense-fp8-combo.yaml` is the day's best |
| `recipes/vllm/` | The vLLM and EXL3 lanes |
| `mods/` | The six SGLang mods, each a `run.sh` with the reason in its header |
| `patches/` | vLLM SM121 patches and image ladder, DFlash2 backport, upstream diffs |
| `scripts/` | `bench.sh` (sanity gate + llama-benchy grid), `cache-flusher.sh`, `node-guard/` |
| `results/sglang-nightly/` | Boot notes, ladder log, probe and quality logs, `RESULTS.md` |
| `results/` | vLLM-era CSVs and serve logs |

## Credits

- [tonyd2wild](https://github.com/tonyd2wild): the GB10 memory mechanism, KV ladder and
  flusher sidecar, the reference vLLM NVFP4 + DFlash2 stack and its benchmark table, the
  thinking-off acceptance finding.
- [MiaAI-Lab](https://github.com/MiaAI-Lab): the EXL3 build and image, the adaptive-k
  and dense-FP8 ideas this recipe's combo follows, the llama-benchy measurement spec.
- [kingjones30](https://github.com/kingjones30): the vLLM SM121 patches and the NoPE-MLA
  padding approach.
- [randomllama](https://huggingface.co/randomllama): the first SGLang GLM-5.3-Flash
  recipe on GB10 and the tile map behind the `sm121-tiles` mod.
- [incoai](https://huggingface.co/incoai): the DFlash2 drafter (z-lab / inco.ai, arXiv
  2602.06036).
- [RedHatAI](https://huggingface.co/RedHatAI) and
  [LibertAIDAI](https://huggingface.co/LibertAIDAI): the NVFP4 checkpoints.
- [eugr](https://github.com/eugr): sparkrun.
- Upstream: sgl-project/sglang#36507, #36708 and #38621, whose authors made the stock
  image boot this model.

The measurements, the mods and the quality findings here are mine, and so are any
mistakes. Companion repos on the same hardware:
[qwen3.8-flash-next-dgx-spark-tp-2](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2),
[qwen3.8-27b-dgx-spark](https://github.com/ursuciprian/qwen3.8-27b-dgx-spark).

## License

Apache-2.0, see `LICENSE`. The vLLM patches under `patches/` are modified copies of
Apache-2.0 vLLM files and keep their upstream headers. The mods under `mods/` edit
SGLang (Apache-2.0) in place at container start and ship no engine code.
