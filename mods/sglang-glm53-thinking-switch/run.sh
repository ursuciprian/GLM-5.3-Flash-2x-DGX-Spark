#!/usr/bin/env bash
# Chat template with a working enable_thinking switch for GLM-5.3-Flash.
# The checkpoint's template opens every assistant turn with "<think>", unconditionally, and only
# knows clear_thinking (history pruning). So chat_template_kwargs {"enable_thinking": false}
# changes nothing in the prompt while SGLang's reasoning parser, which honours that kwarg, stops
# splitting; the client then receives the model's private reasoning followed by "</think>" inside
# content. This writes a copy of the template whose generation prompt emits "<think></think>" when
# enable_thinking is false, to the path the recipe passes as --chat-template. Fail-closed anchor.
set -euo pipefail
python3 - <<'PY'
import glob, pathlib, sys
srcs = sorted(glob.glob("/cache/huggingface/hub/models--*--GLM-5.3-Flash*/snapshots/*/chat_template.jinja"))
if not srcs:
    print("thinking-switch: checkpoint chat_template.jinja not found under /cache/huggingface; refusing"); sys.exit(1)
s = pathlib.Path(srcs[-1]).read_text()
old = "    <|assistant|>{{- '<think>' -}}"
new = ("    <|assistant|>{%- if enable_thinking is defined and not enable_thinking -%}{{- '<think></think>' -}}"
       "{%- else -%}{{- '<think>' -}}{%- endif -%}  {#- [thinking-switch] -#}")
if s.count(old) != 1:
    print(f"thinking-switch: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
out = pathlib.Path("/tmp/glm53_chat_template.jinja"); out.write_text(s.replace(old, new)); out.chmod(0o644)
print(f"thinking-switch: wrote {out} from {srcs[-1]}")
PY
