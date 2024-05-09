
import math
import inspect
from dataclasses import dataclass
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

# Note: Original implementation of distilbert uses an epsilon value of 1e-12
# which is not friendly with the float16 precision that ANE uses by default
EPS = 1e-7

WARN_MSG_FOR_TRAINING_ATTEMPT = \
    "This model is optimized for on-device execution only. " \
    "Please use the original implementation from Hugging Face for training"

WARN_MSG_FOR_DICT_RETURN = \
    "coremltools does not support dict outputs. Please set return_dict=False"


# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix +
               'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix +
                                                                  'weight']
    return state_dict



'''
LlamaModel(
  (embed_tokens): Embedding(32000, 4096)
  (layers): ModuleList(
    (0-31): 32 x LlamaDecoderLayer( #transported from modeling_llama.py
      (self_attn): LlamaSdpaAttention(
        (q_proj): Linear(in_features=4096, out_features=4096, bias=False) ## needs to be nn.Conv2d
        (k_proj): Linear(in_features=4096, out_features=4096, bias=False) ## needs to be nn.Conv2d
        (v_proj): Linear(in_features=4096, out_features=4096, bias=False) ## needs to be nn.Conv2d
        (o_proj): Linear(in_features=4096, out_features=4096, bias=False) ## needs to be nn.Conv2d
        (rotary_emb): LlamaRotaryEmbedding() 
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
        (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
        (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): LlamaRMSNorm()
      (post_attention_layernorm): LlamaRMSNorm()
    )
  )
  (norm): LlamaRMSNorm()
)
'''

@dataclass
class CacheConfig:
    kv_cache: torch.Tensor # [num_layers, 1, 2*seqlen, n_embd]
    qk_mask: torch.Tensor # [1, 1, seqlen, maxseqlen]
    output_mask: torch.Tensor # [1]
    head_index: int

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))



## LlamaDecoderLayer needs
# hidden_size = config.hidden_size
# self_attention[attn.implementation](config,layer_idx=layer_idx)
# mlp(config)
# inport_layernorm - LlamaRMSNorm(hidden_size, config.rms_norm_eps)
# post_attention_layer_norm(hidden_size, eps=config.rms_norm_eps)

class LayerNorm(nn.Module): ## TODO - Cross Compare this with LlamaRMSNorm

    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    ## layers.0.input_layernorm.weight torch.Size([2048])
    ## layers.0.input_layernorm.bias torch.Size([2048])

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epilson = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        ariance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act_fn = SiLU() # TODO - check this update it

    def forward(self, x):
        # TODO - need to finish this once I figure it out
        slice = self.intermediate_size // self.config.pretraining_tp
        gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        gate_proj = torch.cat(
            [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
        up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
        #else:
            #down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    





class LlamaRotaryEmbedding(nn.Module): # TODO - check this over and make sure it good. 
    # not sure if this would be LlamaRotaryEmbedding or LlamaLinearScalingRotaryEmbedding

    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, n_head, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(x.device)
    


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    


'''
self.gate_proj = nn.Conv2d(
in_features=,
out_features=,
bias=False
)
self.up_proj = nn.Conv2d(
in_features=,
out_features=,
bias=False
)
self.down_proj = nn.Conv2d(
in_features=,
out_features=,
bias=False
)
self.act_fn = nn.SiLU()
'''

class LlamaSdpaAttention(
        # we have to convert the Linear to nn.Conv2d for higher optimization with ANE
        # og and modified multihead_attention.py have scaled dot attention
    
        # (q_proj): Linear(in_features=4096, out_features=4096, bias=False)    
        self.q_proj = nn.Conv2d(
            in_features=,
            out_features,
            bias=False
        )

        # (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
        self.k_proj = nn.Conv2d(
            in_features=,
            out_features,
            bias=False
        )

        # (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
        self.v_proj = nn.Conv2d(
            in_features=,
            out_features,
            bias=False
        )

        # (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        self.o_proj = nn.Conv2d(
            in_features=,
            out_features,
            bias=False
        )
)



class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = nn.LlamaSdpaAttention(config) # TODO - not sure if it would be config, but ensure thats right
        self.mlp = nn.LlamaMLP(config) # TODO - not sure
        self.input_layernorm = nn.LlamaRMSNorm() # TODO - not sure
        self.post_attention_layernorm = nn.LlamaRMSNorm() # TODO - not sure

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    


    

    

'''
  "_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.32.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}
'''

'''
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 3200,
  "initializer_range": 0.02,
  "intermediate_size": 8640,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 26,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.31.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}
'''



# NOT SURE WHAT TO DO WITH THIS
@dataclass
class LlamaConfig:
    block_size: int = #is this model dim or 
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    dropout: float = 0.0
    bias: bool = True # bias in linears and layersnorms, false, is a bit faster
    layer_norm_eps: float = 1e-06
    rotary_pct: float = 0.25
    rotary_emb_base: int = 10000


class Llama(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = 
        self.transformer = nn.ModuleDict(dict(
            wte = nn. 
        ))


def transform_hf_weights(state_dict): ## TODO - Refine for state_dict of llama
    for k in list(state_dict.keys()):
        # print(k, state_dict[k].shape)
        # In HF defined as:
        # self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        # Here:
        # self.out_proj = nn.Conv2d(self.d_v, self.d_out, 1)
        # To go from Conv1D to Conv2d, we need to transpose and unsqueeze twice.
        if "attn.c_proj.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            newk = k.replace("attn.c_proj.weight", "attn.out_proj.weight")
            state_dict[newk] = state_dict.pop(k).t()[:, :, None, None]
            # after = state_dict[newk].shape
            # print(before, "->", after)

        # Similar to c_proj.weight
        elif "attn.c_proj.bias" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            newk = k.replace("attn.c_proj.bias", "attn.out_proj.bias")
            state_dict[newk] = state_dict.pop(k)
            # after = state_dict[newk].shape
            # print(before, "->", after)

        # In HF defined as:
        # self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        # Here:
        # self.q_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        # self.v_proj = nn.Conv2d(embed_dim, self.d_v, 1)
        # self.k_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        # To go from Conv1D to Conv2d, we need to transpose and unsqueeze twice.
        elif "attn.c_attn.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            # From HF: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            qm,km,vm = state_dict.pop(k).t()[:, :, None, None].chunk(3, dim=0)
            state_dict[k.replace("c_attn", "q_proj")] = qm
            state_dict[k.replace("c_attn", "k_proj")] = km
            state_dict[k.replace("c_attn", "v_proj")] = vm
            # newks = [k.replace("c_attn", f"{l}_proj") for l in ["q", "k", "v"]]
            # afters = [state_dict[newk].shape for newk in newks]
            # print(before, "->", afters)

        # Similar to c_attn.weight
        elif "attn.c_attn.bias" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            qm,km,vm = state_dict.pop(k).chunk(3, dim=0)
            state_dict[k.replace("c_attn", "q_proj")] = qm
            state_dict[k.replace("c_attn", "k_proj")] = km
            state_dict[k.replace("c_attn", "v_proj")] = vm
            # newks = [k.replace("c_attn", f"{l}_proj") for l in ["q", "k", "v"]]
            # afters = [state_dict[newk].shape for newk in newks]
            # print(before, "->", afters)

        # In HF defined as:
        # self.c_fc = Conv1D(intermediate_size, embed_dim)
        # Here:
        # self.c_fc = nn.Conv2d(embed_dim, ffn_dim, 1)
        # To go from Conv1D to Conv2d, we need to transpose and unsqueeze twice.
        elif "mlp.c_fc.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            state_dict[k] = state_dict[k].t()[:, :, None, None]
            # after = state_dict[k].shape
            # print(before, "->", after)

        # In HF defined as:
        # self.c_proj = Conv1D(embed_dim, intermediate_size)
        # Here:
        # self.c_proj = nn.Conv2d(ffn_dim, embed_dim, 1)
        # To go from Conv1D to Conv2d, we need to transpose and unsqueeze twice.
        elif "mlp.c_proj.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            state_dict[k] = state_dict[k].t()[:, :, None, None]
            # after = state_dict[k].shape
            # print(before, "->", after)

        # In HF defined as:
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Here:
        # self.lm_head = nn.Conv2d(config.n_embd, config.vocab_size, 1, bias=False)
        # To go from Linear to Conv2d, we need to unsqueeze twice (NO TRANSPOSE).
        elif "lm_head.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            state_dict[k] = state_dict[k][:, :, None, None]
            # after = state_dict[k].shape
            # print(before, "->", after)

        # Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
        # apply scale and bias terms in opposite orders. In order to accurately restore a
        # state_dict trained using the former into the the latter, we adjust the bias term
        elif ".ln_" in k and ".bias" in k and not OVERRIDE_LAYER_NORM:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            weight_key = k.replace(".bias", ".weight")
            state_dict[k] = state_dict[k] / state_dict[weight_key]
            # after = state_dict[k].shape
            # print(before, "->", after)

        elif ".ln_" in k and OVERRIDE_LAYER_NORM:
            newk = k.replace(".weight", ".bn.weight")
            newk = newk.replace(".bias", ".bn.bias")
            state_dict[newk] = state_dict.pop(k)


        # else:
        #     if "bias" in k:
        #         print(k)



















 @staticmethod
    def tokenizer_by_name():
        return {n:f"openlm-research{n}" for n in Llama.model_names()}

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        model_type = model_type.replace('openlm-research/', '')
        assert model_type in Llama.model_names()
        model_type = 'openlm-research/' + model_type
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import LlamaForCausalLM
        print("loading weights from pretrained model: %s" % model_type)

        # n_layer, n_head and hidden_size are determined from model_type
        config_args = Llama.config_args()[model_type.replace('openlm-research/', '')]

        # create a from-scratch initialized model
        config = LlamaConfig(**config_args)
        model = Llama(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys]

        # init a huggingface/transformers model
        model_hf = LlamaNeoXForCausalLM.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        for k in list(sd_hf.keys()):
            if k.startswith("gpt_neox."):
                newk = k.replace("gpt_neox.", "")
                assert newk not in sd_hf
                sd_hf[newk] = sd_hf.pop(k)

        for k in list(sd_hf.keys()):
            if k.startswith("embed_out."):
                newk = k.replace("embed_out.", "lm_head.")
                sd_hf[newk] = sd_hf.pop(k)
            elif k.endswith(".masked_bias"):
                del sd_hf[k]
            elif k.endswith(".attention.bias"): #FIXME what is this
                del sd_hf[k]

        in_hf = set(sd_hf.keys()) - set(sd_keys)
        missing_hf = set(sd_keys) - set(sd_hf.keys())
        if len(in_hf) != 0 or len(missing_hf) != 0:
            print([x for x in in_hf if "" in x])
            print([x for x in missing_hf if "" in x])

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            assert sd_hf[k].shape == sd[k].shape, f"{k}: {sd_hf[k].shape} != {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model

    ## convert.py

    def sample_inputs(self):
        vocab_size = self.config.vocab_size
        max_length = 512

        return OrderedDict({
            'input_ids': torch.randint(0, vocab_size, (1, max_length,), dtype=torch.int32),
            'output_mask': torch.tensor([0], dtype=torch.int32),
        })

    def output_types(self):
        return OrderedDict({
            'logits': torch.float16,
        })

    ## generate.py

    @staticmethod
    def build_inputs(seq, input_length=None, outputs={}, pad_to_length=None, pad_token_id=-1, prompt_chunk_idx=None):
        seqlen = seq.shape[1]

        if not pad_to_length:
            pad_to_length = seqlen
        length = pad_to_length

        assert length == seqlen or pad_token_id != -1, "pad token must be provided when padding"

        # Pad the sequence itself too.
        input_ids = torch.cat([
            seq.squeeze(),
            torch.full((pad_to_length - seqlen,), pad_token_id)
        ]).unsqueeze(0)

        # Used to mask outputs before they exit the model.
        # input_ids: [0,1,2,3] length = 4, result is in index 3
        output_mask = torch.tensor([seqlen-1], dtype=torch.int32)

        return {
            "input_ids": input_ids.int(),
            "output_mask": output_mask,
        }

if __name__ == "__main__":
    import argparse
    from transformers import LlamaForCausalLM, AutoTokenizer
    from src.utils.psnr import compute_psnr
    parser = argparse.ArgumentParser(description='Convert a model to CoreML.')
    parser.add_argument('--model_name', choices=GPT.model_names(), default="pythia-160m", type=str)
    args = parser.parse_args()

    model_name = 'openlm-research/' + args.model_name

    nano = GPT.from_pretrained(model_name).eval()
    hf = GPTNeoXForCausalLM.from_pretrained(model_name, use_cache=False).eval()

    tok = AutoTokenizer.from_pretrained(model_name)

    inputs = tok("this washed coffee comes from huila, colombia", return_tensors="pt")
    with torch.no_grad():
        # print(inputs)
        # inputs = torch.rand((1,1,512), requires_grad=False)
        # position_ids = torch.arange(0, inputs.shape[1], dtype=torch.long).unsqueeze(0) # shape (1, t)
        # attention_mask = (1 - torch.tril(torch.ones((1,1,inputs.shape[1],inputs.shape[1]), dtype=torch.float16))) * -1e4
        # attention_mask = (1 - torch.tril(torch.ones((1,1,inputs.shape[1],inputs.shape[1]), dtype=torch.float32))) * -1e9
        # hf_out_a = hf.gpt_neox.layers[0].attention(hf.gpt_neox.layers[0].input_layernorm(inputs), None)[0]
        # nano_out_a = nano.layers[0].attention(nano.layers[0].input_layernorm(inputs), position_ids, attention_mask)
        # assert torch.equal(hf_out_a, nano_out_a), "zzz"
        # hf_out_mlp = hf.gpt_neox.layers[0].mlp(hf.gpt_neox.layers[0].post_attention_layernorm(inputs))
        # nano_out_mlp = nano.layers[0].mlp(nano.layers[0].post_attention_layernorm(inputs))
        # assert torch.equal(hf_out_mlp, nano_out_mlp), "lkjlkjlkj"
        # hf_out = hf_out_mlp + hf_out_a + inputs
        # nano_out = nano_out_mlp + nano_out_a + inputs
        # assert torch.equal(hf_out, nano_out), "wqwwqwqwq"

        # hf_out_l = hf.gpt_neox.layers[0](inputs, None)[0]
        # nano_out_l = nano.layers[0](inputs, position_ids, attention_mask)
        # wtf = nano_out_a + nano_out_mlp + inputs
        # print(wtf - nano_out_l)
        # assert torch.equal(hf_out_l, hf_out)
        # assert torch.equal(wtf, nano_out_l)
        # assert torch.equal(hf_out_l, nano_out_l)


        inputs = {k:v for k,v in inputs.items() if k in ["input_ids"]}
        hf_out = hf(**inputs)['logits']
        nano_out = nano(**inputs)
        # nano = nano.to(device="mps",dtype=torch.float16)
        # inputs = {k: v.to(device="mps") for k, v in inputs.items()}
        # nano_out = nano(**inputs).cpu().float()

    assert hf_out.shape == nano_out.shape, f"{hf_out.shape} != {nano_out.shape}"
    # psnr should be ~240 if perfect.
    print("psnr:", compute_psnr(hf_out, nano_out))
    print("eq", torch.equal(hf_out, nano_out))



