# Per-layer KV cache dtype, so a DFlash2 drafter is not forced onto the
# target's fp8_ds_mla. Port of vllm-project/vllm#47618.
FROM vllm-glm53:dflash2
COPY patch_perlayer_kvdtype.py /tmp/
RUN python3 /tmp/patch_perlayer_kvdtype.py && rm /tmp/patch_perlayer_kvdtype.py
