"""Batched generation with Hugging Face transformers (default) or vLLM (optional, faster).

The HF path is tested on CPU with a tiny model. The vLLM path is NOT tested in the
container: on the first GPU run, compare 20 greedy outputs between the two backends
and log the agreement in notes/decisions.md before using vLLM at scale.
"""
from __future__ import annotations

import torch


def cut_at_stops(text: str, stops: list[str] | None) -> str:
    if not stops:
        return text
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


@torch.no_grad()
def generate_hf(model, tok, prompts: list[str], max_new_tokens: int = 32, do_sample: bool = False,
                temperature: float = 1.0, top_p: float = 1.0, num_return_sequences: int = 1,
                batch_size: int = 16, stops: list[str] | None = None, add_special_tokens: bool | None = None,
                seed: int = 0, progress: bool = True) -> list[list[str]]:
    """Returns one list of num_return_sequences strings per prompt.

    Pure sampling means temperature T, top_p=1 and top_k=0, so no hidden truncation.
    Prompts that already contain chat special tokens should be encoded with
    add_special_tokens=False; base prompts with the tokenizer default.
    """
    from transformers import GenerationConfig

    torch.manual_seed(seed)
    old_side = tok.padding_side
    tok.padding_side = "left"
    device = next(model.parameters()).device
    gc = GenerationConfig(
        do_sample=do_sample, max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else None, top_p=top_p if do_sample else None,
        top_k=0 if do_sample else None, num_return_sequences=num_return_sequences,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list[list[str] | None] = [None] * len(prompts)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        batch = [prompts[i] for i in idx]
        kw = {} if add_special_tokens is None else {"add_special_tokens": add_special_tokens}
        enc = tok(batch, return_tensors="pt", padding=True, **kw).to(device)
        gen = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, generation_config=gc)
        new = gen[:, enc.input_ids.shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for j, i in enumerate(idx):
            chunk = texts[j * num_return_sequences:(j + 1) * num_return_sequences]
            out[i] = [cut_at_stops(t, stops) for t in chunk]
        if progress and (b // batch_size) % 20 == 0:
            print(f"[gen] {min(b + batch_size, len(order))}/{len(order)}")
    tok.padding_side = old_side
    return out  # type: ignore[return-value]


def generate_vllm(hf_id: str, prompts: list[str], max_new_tokens: int = 32, do_sample: bool = False,
                  temperature: float = 1.0, top_p: float = 1.0, num_return_sequences: int = 1,
                  stops: list[str] | None = None, gpu_memory_utilization: float = 0.5, seed: int = 0,
                  revision: str | None = None) -> list[list[str]]:
    """Optional fast path. Set gpu_memory_utilization from MEASURED free memory on a
    shared card, never the vLLM default of 0.9."""
    from vllm import LLM, SamplingParams  # noqa: import only on GPU machines

    llm = LLM(model=hf_id, revision=revision, gpu_memory_utilization=gpu_memory_utilization, seed=seed,
              dtype="bfloat16")
    sp = SamplingParams(n=num_return_sequences, max_tokens=max_new_tokens,
                        temperature=temperature if do_sample else 0.0, top_p=top_p if do_sample else 1.0,
                        top_k=-1, stop=stops or None, seed=seed)
    res = llm.generate(prompts, sp)
    return [[cut_at_stops(o.text, stops) for o in r.outputs] for r in res]
