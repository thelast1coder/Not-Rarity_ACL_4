"""Exact candidate scoring: log p(candidate string | prompt) under the model.

For every (prompt, candidate) pair we encode prompt+candidate jointly and sum the
log-probabilities of the candidate tokens. If the tokenizer merges across the
prompt/candidate boundary, we fall back to encoding the two parts separately and
record boundary_ok=False (the analysis reports how often this happens).
"""
from __future__ import annotations

import numpy as np
import torch


def encode_pair(tok, prompt: str, cand: str, add_special_tokens: bool):
    p_ids = tok(prompt, add_special_tokens=add_special_tokens).input_ids
    full = tok(prompt + cand, add_special_tokens=add_special_tokens).input_ids
    if full[:len(p_ids)] == p_ids and len(full) > len(p_ids):
        return p_ids, full[len(p_ids):], True
    c_ids = tok(cand, add_special_tokens=False).input_ids
    return p_ids, c_ids, False


@torch.no_grad()
def score_pairs(model, tok, prompts: list[str], cands: list[list[str]], add_special_tokens: bool,
                batch_size: int = 32, progress: bool = True):
    """prompts[i] is scored against every string in cands[i].

    Returns (logp, ntok, boundary_ok): lists of float arrays, one per prompt.
    logp[i][k] = sum_t log p(cand_k token t | prompt_i, previous cand tokens).
    """
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    jobs = []  # (i, k, input_ids, n_prompt, n_cand)
    for i, (p, cs) in enumerate(zip(prompts, cands)):
        for k, c in enumerate(cs):
            p_ids, c_ids, ok = encode_pair(tok, p, c, add_special_tokens)
            jobs.append((i, k, p_ids + c_ids, len(p_ids), len(c_ids), ok))
    logp = [np.zeros(len(cs), dtype=np.float64) for cs in cands]
    ntok = [np.zeros(len(cs), dtype=np.int64) for cs in cands]
    bok = [np.ones(len(cs), dtype=bool) for cs in cands]
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][2]))
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        L = max(len(jobs[j][2]) for j in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):  # right padding: positions are unaffected
            seq = jobs[j][2]
            ids[r, :len(seq)] = torch.tensor(seq)
            att[r, :len(seq)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float()
        lsm = torch.log_softmax(logits, dim=-1)
        for r, j in enumerate(idx):
            i, k, seq, n_p, n_c, ok = jobs[j]
            # token at position t is predicted by logits at t-1
            pos = torch.arange(n_p - 1, n_p - 1 + n_c, device=lsm.device)
            tgt = torch.tensor(seq[n_p:n_p + n_c], device=lsm.device)
            logp[i][k] = float(lsm[r, pos, tgt].sum())
            ntok[i][k] = n_c
            bok[i][k] = ok
        if progress and (b // batch_size) % 50 == 0:
            print(f"[score] {min(b + batch_size, len(order))}/{len(order)} pairs")
    return logp, ntok, bok


def softmax_rows(logp: np.ndarray) -> np.ndarray:
    z = logp - logp.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
