# Patches

## `glm5_next.py` (SGLang, required for DFlash2)

DFlash2 needs the target to expose fused hidden states from three layers so the
drafter can condition on them. That capture path landed in
[sgl-project/sglang#36708](https://github.com/sgl-project/sglang/pull/36708)
("[DFLASH] Support GLM-5.3-Flash hidden-state capture"), merged into the
`xinyuan/glm-5.3-flash-support` branch on 2026-08-27 18:23 UTC.

The public image `lmsysorg/sglang:glm-5.3-flash` was built 2026-08-27 05:22
UTC, **13 hours before that merge**, so it does not have the support. Without
it, `--speculative-algorithm DFLASH` has no hidden states to draft from.

The PR touches exactly one production file, so no image rebuild is needed:
bind-mount this file over the container's copy, which is what
`recipes/glm53-sglang-dflash2.yaml` does.

What it adds: a `dflash_capture` flag, `_prepare_aux_hidden_state()` fusing
hidden state with residual, and
`layers_to_capture = [2, num_layers // 2, num_layers - 3]`.

Source: merge commit `c4d5d45e506dcd978a65661a503eda1a272c39a4`.

### Verify before trusting it

This copy is the whole file from the merge commit of a moving feature branch,
not a diff. If the image's own `glm5_next.py` has diverged, mounting this
could regress something else. Diff them on the node first:

```sh
C=$(docker create lmsysorg/sglang:glm-5.3-flash)
docker cp $C:/sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py /tmp/img_glm5.py
docker rm $C
diff /tmp/img_glm5.py patches/glm5_next.py
```

If the only differences are the ~84 added lines of capture logic, mount this
file. If the image has unrelated newer changes, apply just those hunks to the
image's version instead.

## patch_mla.py — SM120/121 vLLM fixes (four patches, not one)

GLM-5.3-Flash uses MLA with `qk_rope_head_dim = 0`. vLLM's fast
`fp8_ds_mla` KV-cache path and its SM120/121 kernels are written for
DeepSeek's 512+64 head layout, so a NoPE model does not reach them and falls
back to a much slower path.

The patcher rewrites `vllm/model_executor/layers/mla.py` in place at
container start, adding a `pe_pad = 64` when the layer is genuinely NoPE and
`VLLM_MLA_NOPE_PAD_ROPE=1`. Query and `k_pe` are zero-padded immediately
before the attention call. Zeros carry no rotation and contribute exactly 0
to the logits, so this is arithmetically a no-op — it only makes the tensor
shapes match what the kernel expects.

Despite the name, the script applies four independent patches. Verified
against `vllm/vllm-openai:glm53-flash-arm64-cu130`, it reports:

```
mla.py NoPE-pad patch applied + compiles
sm120 topk width patch applied + compiles
sm120 dispatch allowlist extended + compiles
sm120 tail-compaction applied + compiles
```

The three `sm120` patches widen the topk, extend the kernel dispatch
allowlist so this head layout is routed to the sparse-MLA backend at all, and
fix tail compaction. The rope padding alone is not sufficient.

Source: `mods/fix-glm53-nope-rope-pad/patch_mla.py` in
kingjones30/GLM-5.3-Flash-2x-DGX-Spark.

It is anchor-based: it asserts each anchor string appears exactly once and
exits non-zero otherwise, so a vLLM version bump fails loudly rather than
silently serving wrong numbers. It also no-ops cleanly if already applied.

**Apply it at build time, not at container start.** It writes into
`dist-packages`, and sparkrun runs the container unprivileged:

```
PermissionError: [Errno 13] Permission denied:
'/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mla.py'
```

`Dockerfile.vllm-patched` bakes it into `vllm-glm53:patched`, which is what
the recipe references. Build once and `docker save | docker load` to the
second node so both hold the same image ID — building separately on each node
yields different IDs and sparkrun then re-syncs ~30 GB on every launch.
