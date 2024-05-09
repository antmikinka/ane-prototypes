#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

#from .home.antmi.more_ane_transformers.src.ml_ane_transformers.ane.layer_norm import LayerNormANE
#/home/antmi/more-ane-transformers/src/ml_ane_transformers/ane/layer_norm.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

#from modeling_llama import LlamaMLP, LlamaAttention

# Note: Original implementation of distilbert uses an epsilon value of 1e-12
# which is not friendly with the float16 precision that ANE uses by default
EPS = 1e-6

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
    print(state_dict)
    return state_dict
"""
class LayerNormANE(LayerNormANE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(
            correct_for_bias_scale_order_inversion)
"""



def _get_unpad_data_optimized(attention_mask):
    # Compute sequence lengths for each batch
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    
    # Efficiently find indices of non-zero elements (unpadded positions)
    # Using torch.where can be more efficient for certain operations on the ANE
    non_zero_indices = torch.where(attention_mask.view(-1) == 1)[0]
    
    # Find the maximum sequence length in the batch for potential buffer size optimizations
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    
    # Compute cumulative sequence lengths with padding for easy indexing
    # Padding at the beginning for cumulative indexing might be optimized differently based on model architecture
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    
    return (
        non_zero_indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

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

#ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)




class LlamaRotaryEmbedding(nn.Module):
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
    """LlamaMLP Module optimized for the Apple Neural Engine"""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Conv2d(self.hidden_size, self.intermediate_size,kernel_size=1, bias=False)
        self.up_proj = nn.Conv2d(self.hidden_size, self.intermediate_size,kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(self.intermediate_size, self.hidden_size,kernel_size=1, bias=False)
        self.act_fn = nn.SiLU()

        ''' # TODO - properly set this up like distilbert.py #LlamaMLP(modeling_llama.LlamaMLP)
        setattr(
            self, 'gate_proj',
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=self.intermediate_size,
                kernel_size=1,

            ))

        setattr(
            self, 'up_proj',
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=self.intermediate_size,
                kernel_size=1,

            ))


        setattr(
            self, 'down_proj',
            nn.Conv2d(
                in_channels=self.intermediate_size,
                out_channels=self.hidden_size,
                kernel_size=1,

            ))

        setattr(
            self, 'act_fn',
            nn.SiLU()
        )
        '''        

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            # Optionally, if bias is used, split it as well.
            #if self.gate_proj.bias is not None:
            #gate_proj_bias_slices = self.gate_proj.bias.split(slice_size, dim=0)
            #else:
            #gate_proj_bias_slices = [None] * self.config.pretraining_tp 

             # Applying Conv2d slices
            gate_proj_outputs = [F.conv2d(x, gate_proj_slices[i], bias=None, stride=1, padding=0) for i in range(self.config.pretraining_tp)]
            gate_proj = torch.cat(gate_proj_outputs, dim=1)
        
            up_proj_outputs = [F.conv2d(x, up_proj_slices[i], bias=None, stride=1, padding=0) for i in range(self.config.pretraining_tp)]
            up_proj = torch.cat(up_proj_outputs, dim=1)
        
            # Activation and element-wise multiplication
            intermediate_states = self.act_fn(gate_proj) * up_proj
        
            # Handling down_proj; assuming concatenation followed by a reduction through Conv2d is intended
            # Since Conv2d cannot directly sum slices like linear, we'll concatenate and then apply a single Conv2d to reduce dimension if needed
            down_proj_outputs = torch.cat([intermediate_states for _ in range(self.config.pretraining_tp)], dim=1)
            down_proj = self.down_proj(down_proj_outputs)
        else:
            # For non-parallel processing, directly apply Conv2d layers
            gate_proj = self.gate_proj(x)
            up_proj = self.up_proj(x)
            intermediate_states = self.act_fn(gate_proj) * up_proj
            down_proj = self.down_proj(intermediate_states)

        return down_proj

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
    """ LlamaAttention module optimized for the Apple Neural Engine """   
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer_idx = layer.idx

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

        #self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim,kernel_size=(1, 1), bias=config.attention_bias)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim,kernel_size=(1, 1), bias=config.attention_bias)
        self.v_proj = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim,kernel_size=(1, 1), bias=config.attention_bias)
        self.o_proj = nn.Conv2d(self.hidden_size, self.hidden_size,kernel_size=(1, 1), bias=config.attention_bias)

        ''' # TODO - properly set this up like distilbert.py #LlamaAttention(modeling_llama.LlamaAttention):
        # layers.0.self_attn.q_proj.weight torch.Size([4096, 4096])
        setrattr(
            self, 'q_proj',
            nn.Conv2d(
            in_channels=self.hidden_size, #this matches the input feature dimension
            out_channels=self.num_heads * self.head_dim, #for QKV projections
            kernel_size=(1, 1),
            bias=config.attention_bias
        ))

        # layers.0.self_attn.k_proj.weight torch.Size([4096, 4096])
        setrattr(
            self, 'k_proj',
            nn.Conv2d(
            in_channels=self.hidden_size, #this matches the input feature dimension
            out_channels=self.num_key_value_heads * self.head_dim, # Adjust based on key/value heads
            kernel_size=(1, 1),
            bias=config.attention_bias
        ))

        # layers.0.self_attn.v_proj.weight torch.Size([4096, 4096])
        setrattr(
            self, 'v_proj',
            nn.Conv2d(
            in_channels=self.hidden_size, 
            out_channels=self.num_key_value_heads * self.head_dim, 
            kernel_size=(1, 1),
            bias=config.attention_bias
        ))

        # layers.0.self_attn.o_proj.weight torch.Size([4096, 4096])
        setrattr(
            self, 'o_proj',
            nn.Conv2d(
            in_channels=self.hidden_size, # input form concat head outputs
            out_channels=self.hidden_size, # projecting back to the hidden size
            kernel_size=(1, 1),
            bias=config.attention_bias
        ))
        '''
        self._init_rope()

    def _init_rope(self):
        rotary_embedding_classes = {
            "linear": LlamaLinearScalingRotaryEmbedding,
            "dynamic": LlamaDynamicNTKScalingRotaryEmbedding,
        }
    
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling.get("type")
            scaling_factor = self.config.rope_scaling.get("factor", 1)  # Default factor
            embedding_class = rotary_embedding_classes.get(scaling_type)
        
            if embedding_class is not None:
                self.rotary_emb = embedding_class(
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
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Ensure hidden_states is in 4D format (B, C, 1, S) for Conv2d operations
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.unsqueeze(2)  # Add the singleton dimension

        # Apply Conv2d projections directly without manual weight splitting
        query_states = self.q_proj(hidden_states).squeeze(2)
        key_states = self.k_proj(hidden_states).squeeze(2)
        value_states = self.v_proj(hidden_states).squeeze(2)

        # Adjust shapes for multi-head attention calculation
        # Assuming hidden_states originally (B, C, S), where C is channels (hidden_size)
        query_states = query_states.view(bsz, self.config.num_heads, q_len, -1).permute(0, 2, 1, 3)
        key_states = key_states.view(bsz, self.config.num_heads, q_len, -1).permute(0, 2, 1, 3)
        value_states = value_states.view(bsz, self.config.num_heads, q_len, -1).permute(0, 2, 1, 3)

        # Compute attention scores and apply mask
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * (self.config.head_dim ** -0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape attn_output to (B, C, 1, S) for final projection
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, -1, 1, q_len)

        # Apply final projection
        attn_output = self.o_proj(attn_output).squeeze(2)  # Assuming o_proj is also a Conv2d layer

        if not output_attentions:
            attn_weights = None


        print("attn_output")
        print(attn_output)
        print("attn_weights")
        print(attn_weights)
        print("past_key_value") 
        print(past_key_value)    
        return attn_output, attn_weights, past_key_value
       

class LlamaConfig:
    def __init__(self):
        self.hidden_size = 3200
        self.intermediate_size = 8640
        self.max_position_embeddings = 2048
        self.num_attention_heads = 32
        self.pad_token_id = 0
        self.rms_norm_eps = 1e-06
        self.use_cache = True
        self.torch_dtype = "float16"
        # Add more attributes as necessary

# Instantiate the config with specified values
config = LlamaConfig()

# Assuming LlamaMLP, LlamaRMSNorm (or nn.LayerNorm), and LLAMA_ATTENTION_CLASSES are appropriately defined
# Here's the optimized LlamaDecoderLayer using the provided config





class LlamaForCausalLMConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.hidden_size = kwargs.get('hidden_size', 3200)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 26)
        self.num_attention_heads = kwargs.get('num_attention_heads', 32)
        self.intermediate_size = kwargs.get('intermediate_size', 8640)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 2048)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-06)
        self.hidden_act = kwargs.get('hidden_act', 'silu')

# Assuming Block and Attention classes are properly defined elsewhere to match LLaMA's architecture
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        #self.layer_idx = layer.idx
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)#), layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Ensure hidden_states is in the expected dtype (float16 for ANE optimization)
        hidden_states = hidden_states.to(self.config.torch_dtype)

        # Apply input layer normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # Assuming self.self_attn properly handles dtype conversion internally
        attention_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        # Add & Norm
        hidden_states = hidden_states + attention_output
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        print(outputs)
        return outputs



class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=layer_idx) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, output_mask=None):
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.max_position_embeddings, f"Cannot forward sequence of length {t}, max position embeddings is only {self.config.max_position_embeddings}"
        
        position_ids = torch.arange(t, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
        position_embeddings = self.position_embeddings(position_ids)

        inputs_embeds = self.embed_in(input_ids)
        x = inputs_embeds + position_embeddings

        attention_mask = (1 - torch.tril(torch.ones((t, t), dtype=torch.float16, device=device))) * -1e4

        for layer in self.layers:
            x = layer(x, position_ids, attention_mask)

        if output_mask is not None:
            x = torch.index_select(x, 1, output_mask)

        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        return logits

# Initialize the model with the provided configuration
config = LlamaForCausalLMConfig(
    vocab_size=32000,
    hidden_size=3200,
    num_hidden_layers=26,
    num_attention_heads=32,
    intermediate_size=8640,
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-06,
    hidden_act='silu',
    torch_dtype='float16',  # This setting would be used to set global dtype if necessary
    use_cache=True
)
model = LlamaForCausalLM(config)

@staticmethod
def config_args():
    return OrderedDict({
        'llama2': dict(n_layer=32, n_head=32, hidden_size=3200, intermediate_size=8640, vocab_size=32000)
    })
@staticmethod
def model_names():
    return list(GPT.config_args().keys())



@staticmethod
def tokenizer_by_name():
    return {n:f"openlm-research/{n}" for n in GPT.model_names()}

def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """ Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
    """
    for k in state_dict:
        is_internal_proj = all(substr in k for substr in ['proj', '.weight'])
        is_output_proj = all(substr in k
                             for substr in ['classifier', '.weight'])
        if is_internal_proj or is_output_proj:
            if len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]
            


@classmethod
def from_pretrained(cls, model_type, override_args=None):
    model_type = model_type.replace('openlm-research/', '')
    assert model_type in GPT.model_names()
    model_type = "openlm-research/" + model_type
    override_args = override_args or {} # default to empty dict
    # only dropout can be overridden see more notes below
    assert all(k == 'dropout' for k in override_args)
    from transformers import GPTNeoXForCausalLM
    print("loading weights from pretrained model: %s" % model_type)

    # n_layer, n_head and hidden_size are determined from model_type
    config_args = GPT.config_args()[model_type.replace("openlm-research/", '')]

    #generate.py

    # create a from-scratch initialized model
    config = LlamaConfig(**config_args)
    model = LlamaForCausalLM(config)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys()]

    print("sd_keys")
    print(sd_keys)

    # init a huggingface/transformers model
    model_hf = LlamaForCausalLM.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    print(sd_hf)

    # Adjusting the keys to match your custom model's expectations
    adjusted_sd_hf = {}
    for k, v in sd_hf.items():
        new_key = k
        if k.startswith("model."):
            new_key = k.replace("model.", "")  # Fix this line according to your previous message
        if k.startswith("embed_out."):
            new_key = k.replace("embed_out.", "lm_head.")
        # Perform additional key adjustments if necessary
        
        adjusted_sd_hf[new_key] = v

    # Now use adjusted_sd_hf for comparison and copying
    in_hf = set(adjusted_sd_hf.keys()) - set(sd_keys)
    missing_hf = set(sd_keys) - set(adjusted_sd_hf.keys())
    if len(in_hf) != 0 or len(missing_hf) != 0:
        print("In HF model but not in custom model:", [x for x in in_hf])
        print("In custom model but not in HF model:", [x for x in missing_hf])

    # Ensure all of the parameters are aligned and match in names and shapes
    assert len(adjusted_sd_hf.keys()) == len(sd_keys), f"mismatched keys: {len(adjusted_sd_hf.keys())} != {len(sd_keys)}"
    for k in adjusted_sd_hf:
        if k in sd:  # Ensure the key exists in your model before copying
            assert adjusted_sd_hf[k].shape == sd[k].shape, f"{k}: {adjusted_sd_hf[k].shape} != {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(adjusted_sd_hf[k])

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
    from transformers import LlamaModel, AutoTokenizer
    #from src.utils.psnr import compute_psnr
    #/home/antmi/more-ane-transformers/src/utils/psnr.py
    import sys
    sys.path.append('/home/antmi/more-ane-transformers/')
    from src.utils.psnr import compute_psnr
    parser = argparse.ArgumentParser(description='Convert a model to CoreML.')
    parser.add_argument('--model_name', choices=GPT.model_names(), default="open_llama_3b_v2", type=str)
    args = parser.parse_args()
    
    model_name = "openlm-research/" + args.model_name

    nano = GPT.from_pretrained(model_name).eval()
    hf = LlamaForCausalLM.from_pretrained(model_name, use_cache=False).eval()

    tok = AutoTokenizer.from_pretrained(model_name)

    inputs = tok("this washed coffee comes from huila, colombia", return_tensors="pt")
    with torch.no_grad():

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


'''


## would be the automatic conversion type beat like distilbert    

class LlamaDecoderLayer(modeling_llama.LlamaDecoderLayer):
    
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'self_attn', LlamaAttention(config))
        setattr(self, 'mlp', LlamaMLP(config))
        setattr(self, 'input_layernorm', LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        setattr(self, 'post_attention_layernorm', LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
'''



