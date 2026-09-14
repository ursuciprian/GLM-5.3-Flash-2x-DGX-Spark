#!/usr/bin/env bash
# Typical acceptance for DFLASH sampled verification.
# The exact rejection kernel is correct (verified on GB10 against CPU references) but it needs the
# draft probability q that produced each draft token; the DFLASH selector's declared q does not
# match how its tokens are chosen on this nightly, and the residual sampling then emits token salad
# and repeated fragments at temperature 1.0. This mod adds a q-free verify: accept draft token j
# while p_target(token_j | accepted prefix) >= SGLANG_DFLASH_TYPICAL_ACCEPT (e.g. 0.2); at the first
# rejection sample the replacement from the target distribution at that position; if every draft
# survives, sample the bonus from the last position. Target probabilities come from the existing
# build_dflash_verify_target_probs (temperature, top-k, top-p already applied). Unset = stock path.
set -euo pipefail
python3 - <<'PY'
import importlib.util, pathlib, sys
p = pathlib.Path(importlib.util.find_spec("sglang").origin).parent / "srt/speculative/dflash_worker_v2.py"
s = p.read_text()
if "[dflash-typical-accept]" in s:
    print("dflash-typical-accept: already applied"); sys.exit(0)
old = '''        """Scatter the selector's sparse q into a dense one for DSpark's kernel."""
        bs, block = candidates.shape
        gamma = block - 1
        vocab = int(next_token_logits.shape[-1])
'''
new = '''        """Scatter the selector's sparse q into a dense one for DSpark's kernel."""
        bs, block = candidates.shape
        gamma = block - 1
        vocab = int(next_token_logits.shape[-1])
        # [dflash-typical-accept] q-free verification: accept while the target likes the draft
        # token, otherwise sample the replacement from the target distribution itself.
        _tau = __import__("os").environ.get("SGLANG_DFLASH_TYPICAL_ACCEPT")
        if _tau:
            from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs

            tau = float(_tau)
            p = build_dflash_verify_target_probs(
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=block,
                bs=bs,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )  # (bs, block, vocab), temperature/top-k/top-p applied
            drafts = candidates[:, 1:].long()  # (bs, gamma); position j is verified by p[:, j]
            p_draft = torch.gather(p[:, :gamma], 2, drafts.unsqueeze(-1)).squeeze(-1)  # (bs, gamma)
            ok = p_draft >= tau
            # accept the prefix of consecutive ok positions
            first_bad = torch.where(ok, gamma, torch.arange(gamma, device=ok.device).expand_as(ok)).amin(dim=1)
            accept_len = first_bad.clamp(max=gamma)  # number of accepted drafts, 0..gamma
            # replacement / bonus sampled from the target at position accept_len
            rows = p[torch.arange(bs, device=p.device), accept_len]  # (bs, vocab)
            rows = rows.clamp_min(0)
            rows = torch.where(rows.sum(-1, keepdim=True) > 0, rows, torch.softmax(next_token_logits.view(bs, block, vocab)[torch.arange(bs, device=p.device), accept_len].float(), -1))
            bonus = torch.multinomial(rows.float(), 1).squeeze(-1)
            return accept_len.to(torch.int32), bonus.to(torch.int64)
'''
if s.count(old) != 1:
    print(f"dflash-typical-accept: anchor matched {s.count(old)} times, expected 1; refusing"); sys.exit(1)
p.write_text(s.replace(old, new)); print(f"dflash-typical-accept: patched {p}")
PY
