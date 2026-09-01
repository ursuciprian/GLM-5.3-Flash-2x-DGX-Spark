# DFlash2 speculative decoding for GLM-5.3-Flash, layered on our patched image.
# Upstream builds this FROM radixark/vllm-glm53-flash:sm121-v8, which is not
# public; vllm/vllm-openai:glm53-flash-arm64-cu130 carries the same
# vllm/models/glm5next/nvidia layout, so the same patches apply.
FROM vllm-glm53:patched
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm
COPY qwen3_dflash2.py $VLLM/model_executor/models/qwen3_dflash2.py
COPY dflash2/ $VLLM/v1/worker/gpu/spec_decode/dflash2/
COPY patch_registry_and_select.py /tmp/
COPY patch_glm_aux_capture.py /tmp/
COPY patch_kv_page_lcm2.py /tmp/
COPY patch_glm5_drafter_group.py /tmp/
RUN python3 /tmp/patch_registry_and_select.py \
 && python3 /tmp/patch_glm_aux_capture.py \
 && python3 /tmp/patch_kv_page_lcm2.py \
 && python3 /tmp/patch_glm5_drafter_group.py \
 && python3 -c "from vllm.model_executor.models.registry import ModelRegistry; \
assert 'DFlash2DraftModel' in ModelRegistry.get_supported_archs(); \
print('DFlash2 drafter registered')"
