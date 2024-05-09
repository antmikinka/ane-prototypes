## REVISED VERSION OF CasualSelfAttention from more-ane-transformers/gpt2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by the number of heads."
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # Linear transformation for Q, K, V
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)  # Linear transformation for the output
        self.attn_dropout = nn.Dropout(config.dropout)  # Dropout for attention
        self.resid_dropout = nn.Dropout(config.dropout)  # Dropout for the output
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = False  # Optionally use flash attention

    def forward(self, x, attention_mask, kv_config):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Split the linear output into Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        if kv_config is not None and hasattr(kv_config, 'kv_cache') and kv_config.kv_cache is not None:
            kv_cache = kv_config.kv_cache
            new_q, new_k, new_v = q, k, v  # New queries, keys, values
            old_k, old_v = kv_cache.chunk(2, dim=2)  # Split cached keys and values

            k = torch.cat([old_k, new_k], dim=1)
            v = torch.cat([old_v, new_v], dim=1)
            q = new_q

            B, T, C = k.size()

        current_cache = torch.cat([k, v], dim=2)

        # Reshape for multi-head attention and compute dot-product attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.multi_head_attention_forward(
                q, k, v, 
                num_heads=self.n_head,
                dropout_p=self.dropout,
                add_bias_kv=False,
                add_zero_attn=False,
                need_weights=False
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att + attention_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # Combine attention weights with values

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Concatenate heads

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, current_cache
