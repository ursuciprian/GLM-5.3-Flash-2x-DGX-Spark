#!/usr/bin/env bash
# compressed-tensors name mapping for GLM-5.3-Flash on SGLang.
# RedHatAI/GLM-5.3-Flash-NVFP4 lists its unquantized modules under Hugging Face names. SGLang's
# Glm5Next mapper strips "model.language_model." but the KDA layers' forget gate is flattened in
# SGLang (self_attn.f_a_proj) while the checkpoint nests it (self_attn.forget_gate.f_a_proj), so
# the ignore list never matches and loading dies with "Unable to find matching target for
# model.layers.0.self_attn.f_a_proj". WeightsMapper applies one substring rule per name, so the
# fix goes in the suffix stage, which runs after the prefix strip. Idempotent, fail-closed.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
# locate without importing: hooks run as root and an import would create root-owned kernel caches
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/models/glm5_next.py"
if not p.exists():
    print(f"ct-names: {p} missing; refusing"); sys.exit(1)
s = p.read_text()
old = ('    hf_to_sglang_mapper = WeightsMapper(\n'
       '        orig_to_new_substr={\n'
       '            "model.language_model.": "model.",\n'
       '            "model.visual": "visual",\n'
       '        }\n'
       '    )\n')
new = ('    hf_to_sglang_mapper = WeightsMapper(\n'
       '        orig_to_new_substr={\n'
       '            "model.language_model.": "model.",\n'
       '            "model.visual": "visual",\n'
       '        },\n'
       '        orig_to_new_suffix={  # [ct-names] checkpoint nests the KDA forget gate, SGLang flattens it\n'
       '            "self_attn.forget_gate.f_a_proj": "self_attn.f_a_proj",\n'
       '            "self_attn.forget_gate.f_b_proj": "self_attn.f_b_proj",\n'
       '        },\n'
       '    )\n')
if "[ct-names]" in s:
    print("ct-names: already applied"); sys.exit(0)
if s.count(old) != 1:
    print(f"ct-names: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
p.write_text(s.replace(old, new)); print(f"ct-names: patched {p}")
PY
