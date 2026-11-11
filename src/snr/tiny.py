"""Tiny local tokenizer and model for CPU tests (no downloads).

A byte-level BPE tokenizer is trained on the fly so that prompt/answer boundary
handling is tested on real BPE merges, and a 2-layer random Llama matches its vocab.
"""
from __future__ import annotations

import torch

CHAT_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def tiny_tokenizer(corpus: list[str], vocab_size: int = 600):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1,
                             special_tokens=["<pad>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|system|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus * 3, tr)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    tok.chat_template = CHAT_TEMPLATE
    return tok


def tiny_model(tok, seed: int = 0, sharpen: float = 4.0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(sharpen)
    return m
