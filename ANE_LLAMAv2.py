#import LayerNormANE
import torch
import torch.nn as nn
from transformers.models.llama import modeling_llama
from dataclasses import dataclass

#from transformers import cache_utils
#from cache_utils import StaticCache

#impsrc/transformers/cache_utils.py
EPS = 1e-6

WARN_MSG_FOR_TRAINING_ATTEMPT = \
	"This model is optimized for on-device execution only. " \
	"Please use the original implementation from Hugging Face for training"

WARN_MSG_FOR_DICT_RETURN = \
	"coremltools does not support dict outputs. Please set return_dict=False"
	
## Currently this file completely computes properly. I just need to continue ANE on the classes.
## This file should have no problem with state_dicts like how I was with the previous things. 
## ane-llama.py is the file that I will have to cross ref the weights with and its what applies the weights to the model
## weights look great, made the txt file for after loading. One weird thing, I dont know how layers were loaded when classes werent specified in this file
	

	
# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
	# apply scale and bias terms in opposite orders. In order to accurately restore a
	# state_dict trained using the former into the the latter, we adjust the bias term
	
	
def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata, strict, missing_keys,unexpected_keys, error_msgs):
	state_dict[prefix +'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix +'weight']
	return state_dict

##distilbert.py has
## config, config.dim, config.hidden_dim, config.n_layers, config.activation, config.vocab_size, config.use_return_dict, config.num_labels, config.hidden_size
def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
	""" Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
	"""
	for k in state_dict:
		is_internal_proj = all(substr in k for substr in ['proj', '.weight'])
		is_output_proj = all(substr in k for substr in ['head', '.weight'])
		print(is_internal_proj)
		if is_internal_proj or is_output_proj:
			if len(state_dict[k].shape) == 2:
				state_dict[k] = state_dict[k][:, :, None, None]

"""
class LayerNormANE(LayerNormANE):
		
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self._register_load_state_dict_pre_hook(
				correct_for_bias_scale_order_inversion)
"""
"""
class LlamaRMSNorm(modeling_llama.LlamaRMSNorm):
	
	def __init__(self, config):
		super().__init__(config)
		
		setattr(self,)
"""
"""
class LlamaRMSNorm(modeling_llama.LlamaRMSNorm): #TODO need to look overthis
		def __init__(self, hidden_size, eps=1e-6):
			super().__init__(hidden_size, eps)
			self.weight = nn.Parameter(torch.ones(hidden_size))
			self.variance_epilson = eps
			
		def forward(self, hidden_states):
			# Principle 16: Data Format (B, C, 1, S ) 
			if hidden_states.ndim != 4:  # Assuming (B, S, C) or similar coming in
				hidden_states = hidden_states.to_format("channels_last")  # Adjust as needed
		
			input_dtype = hidden_states.dtype
		
			# Principle 18: Minimize Copies to/from ANE, keep computation on device where possible
			variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)  
		
			# Principle 18 (continued) + Potential for Principle 19 (`einsum` if suitable)
			try:
				# Explore einsum as ANE may optimize, but depends on full LLM + hardware behavior
				hidden_states = torch.einsum(
					'bcis,c->bcis', hidden_states, torch.rsqrt(variance + self.variance_epsilon)
				)
			except RuntimeError:
				# Fallback - ANE toolchain or expression complexity may not support it
				hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
		
			# Use setattr due to LlamaRMSNorm lacking direct access to a scaling paramter 
			setattr(self, 'weight', self.weight.to(hidden_states.device))  # Principle 18  again
			hidden_states = hidden_states * self.weight  
		
			return hidden_states.to(input_dtype)

"""
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

"""
class LlamaRotaryEmbedding(nn.Module):
	def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
		super().__init__()
		self.dim = dim
		self.max_position_embeddings = max_position_embeddings
		self.base = base
		inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
		self.register_buffer("inv_freq", inv_freq, persistent=False)

	@property
	def sin_cached(self):
		logger.warning_once(
            "The sin_cached attribute will be removed in 4.40. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead."
        )
		return self._sin_cached

	@property
	def cos_cached(self):
		logger.warning_once(
			"The cos_cached attribute will be removed in 4.40. Bear in mind that its contents changed in v4.38. Use "
			"the forward method of RoPE from now on instead."
		)
		return self._cos_cached

	def forward(self, x, position_ids, seq_len=None):
		if seq_len is not None:
			logger.warning_once("The `seq_len` argument is deprecated and unused. It will be removed in v4.40.")

		# x: [bs, num_attention_heads, seq_len, head_size]
		inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
		position_ids_expanded = position_ids[:, None, :].float()
		freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)

		emb = torch.cat((freqs, freqs), dim=-1)
		cos = emb.cos().to(dtype=x.dtype)
		sin = emb.sin().to(dtype=x.dtype)
		# backwards compatibility
		self._cos_cached = cos
		self._sin_cached = sin
		return cos, sin
"""
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

#ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

class LlamaLinearScalingRotaryEmbedding(modeling_llama.LlamaLinearScalingRotaryEmbedding): #TODO need to lookover this
				"""Optimized LlamaRotaryEmbedding with linear scaling for ANE workloads."""
				
				def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
					self.scaling_factor = scaling_factor
					super().__init__(dim, max_position_embeddings, base, device)
				
				def forward(self, x, seq_len=None):
					if seq_len is None:
						seq_len = x.shape[-2]  # Infer sequence length from input
				
					# Device Flexibility: Match calculations to the device of the input tensor
					t = torch.arange(seq_len, device=x.device) / self.scaling_factor
					freqs = torch.outer(t, self.inv_freq.to(x.device))
				
					emb = torch.cat((freqs, freqs), dim=-1)
				
					# Type Consistency: Maintain accuracy and prevent surprises 
					cos_emb = emb.cos().to(dtype=x.dtype)
					sin_emb = emb.sin().to(dtype=x.dtype) 
				
					return cos_emb, sin_emb

"""
class LlamaLinearScalingRotaryEmbedding(modeling_llama.LlamaLinearScalingRotaryEmbedding):
	def __init__(self, config):
		super().__init__(config)
		
		setattr(self,)
"""

class LlamaDynamicNTKScalingRotaryEmbedding(modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding): #TODO need to lookover this
		"""Optimized LlamaDynamicNTKScalingRotaryEmbedding for ANE usage."""
		
		def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
			self.scaling_factor = scaling_factor
			super().__init__(dim, max_position_embeddings, base, device)
		
		def forward(self, x, seq_len=None):
			if seq_len is None:
				seq_len = x.shape[-2]  # Infer sequence length dynamically
		
			# Device Flexibility: Move computations to match data
			device = x.device 
		
			 # Adaptive Computation & Type Handling
			if seq_len > self.max_position_embeddings:
				adjusted_base = self.base * (
					(self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
				) ** (self.dim / (self.dim - 2))
				inv_freq = 1.0 / (adjusted_base ** (
						torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))  # Ensure float types
		
				# Dynamically set inv_freq attribute (used by parent class)
				setattr(self, 'inv_freq', inv_freq) 
		
			# Utilize superclass logic
			return super().forward(x, seq_len=seq_len) 

'''
class LlamaDynamicNTKScalingRotaryEmbedding(modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding):
	def __init__(self, config):
		super().__init__(config)
			setattr(self,)
'''

def rotate_half(x):
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
	print(q.shape, cos.shape, sin.shape, rotate_half(q).shape)
	"""Applies Rotary Position Embedding to the query and key tensors.

	Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
	print("before cos and sin unsqueeze")
	print(q.shape)
	print(cos.shape)
	print(sin.shape)
	print(rotate_half(q).shape)
	cos = cos.unsqueeze(unsqueeze_dim)
	sin = sin.unsqueeze(unsqueeze_dim)

	print("after cos and sin unsqueeze")
	print(q.shape)
	print(cos.shape)
	print(sin.shape)
	print(rotate_half(q).shape)

	q_embed = (q * cos) + (rotate_half(q) * sin)
	print(f"q_embed shape",q_embed.shape)

	k_embed = (k * cos) + (rotate_half(k) * sin)

	print(f"q_embed shape",k_embed.shape)
	return q_embed, k_embed

class LlamaMLP(modeling_llama.LlamaMLP):
	"""LlamaMLP Module optimized for the Apple Neural Engine"""

	def __init__(self, config):
		super().__init__(config)
		self.config = config
		self.hidden_size = config.hidden_size
		self.intermediate_size = config.intermediate_size
		#self.gate_proj = nn.Conv2d(self.hidden_size, self.intermediate_size,kernel_size=1, bias=False)
		#self.up_proj = nn.Conv2d(self.hidden_size, self.intermediate_size,kernel_size=1, bias=False)
		#self.down_proj = nn.Conv2d(self.intermediate_size, self.hidden_size,kernel_size=1, bias=False)
		#self.act_fn = nn.SiLU()
	
		setattr(
			self, 'gate_proj',
			nn.Conv2d(
				in_channels=self.hidden_size,
				out_channels=self.intermediate_size,
				kernel_size=1,
				bias=config.attention_bias,
			))
	
		setattr(
			self, 'up_proj',
			nn.Conv2d(
				in_channels=self.hidden_size,
				out_channels=self.intermediate_size,
				kernel_size=1,
				bias=config.attention_bias,
			))
	
	
		setattr(
			self, 'down_proj',
			nn.Conv2d(
				in_channels=self.intermediate_size,
				out_channels=self.hidden_size,
				kernel_size=1,
				bias=config.attention_bias,
			))
	
		setattr(
			self, 'act_fn',
			nn.SiLU()
		)
	
	
	def forward(self, x): #TODO need to lookover this
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
			gate_proj_outputs = [F.Conv2d(x, gate_proj_slices[i], bias=None, stride=1, padding=0) for i in range(self.config.pretraining_tp)]
			gate_proj = torch.cat(gate_proj_outputs, dim=1)
		
			up_proj_outputs = [F.Conv2d(x, up_proj_slices[i], bias=None, stride=1, padding=0) for i in range(self.config.pretraining_tp)]
			up_proj = torch.cat(up_proj_outputs, dim=1)
		
			# Activation and element-wise multiplication
			intermediate_states = self.act_fn(gate_proj) * up_proj
		
			# Handling down_proj; assuming concatenation followed by a reduction through Conv2d is intended
			# Since Conv2d cannot directly sum slices like linear, we'll concatenate and then apply a single Conv2d to reduce dimension if needed
			down_proj_outputs = torch.cat([intermediate_states for _ in range(self.config.pretraining_tp)], dim=1)
			down_proj = self.down_proj(down_proj_outputs)
		else:
			# For non-parallel processing, directly apply Conv2d layers
			print("ANE_LLAMA LLAMAMLP, ELSE NON-PARALLEL PROCESSING")
			gate_proj = self.gate_proj(x)
			up_proj = self.up_proj(x)
			intermediate_states = self.act_fn(gate_proj) * up_proj
			down_proj = self.down_proj(intermediate_states)
	
		return down_proj
		
	
#def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: #TODO need to lookover this
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



class LlamaAttention(modeling_llama.LlamaAttention):
					
	def __init__(self, config, layer_idx: 0):
		super().__init__(config)

		self.config = config
		self.layer_idx = layer_idx=0
		self.attention_dropout = config.attention_dropout
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads
		self.head_dim = self.hidden_size // self.num_heads
		self.num_key_value_heads = config.num_key_value_heads
		self.num_key_value_groups = self.num_heads // self.num_key_value_heads
		self.max_position_embeddings = config.max_position_embeddings
		self.rope_theta = config.rope_theta
		self.is_causal = True
		

		
		setattr(
			self, 'q_proj',
			nn.Conv2d(
				in_channels=self.hidden_size,
				out_channels=self.num_heads*self.head_dim,
				kernel_size=1,
				bias=config.attention_bias,
			))

		setattr(
			self, 'k_proj',
			nn.Conv2d(
				in_channels=self.hidden_size,
				out_channels=self.num_key_value_heads * self.head_dim,
				kernel_size=1,
				bias=config.attention_bias,
			))

		setattr(
			self, 'v_proj',
			nn.Conv2d(
				in_channels=self.hidden_size,
				out_channels=self.num_key_value_heads*self.head_dim,
				kernel_size=1,
				bias=config.attention_bias,
			))

		setattr(
			self, 'o_proj',
			nn.Conv2d(
				in_channels=self.hidden_size,
				out_channels=self.hidden_size,
				kernel_size=1,
				bias=config.attention_bias,
			))

		self._init_rope()

		def _init_rope(self):
			if self.config.rope_scaling is None:
				self.rotary_emb = LlamaRotaryEmbedding(
					self.head_dim, 
					max_position_embeddings=self.max_position_embeddings,
					base=self.rope_theta,
				)
			else:
				raise NotImplementedError
				#TODO linear & dynamic scaling rotary embeddings

	def forward( #TODO finish forward method
		self,
		hidden_states,
		attention_mask,
		position_ids,
		past_key_value,
		output_attentions,
		use_cache,
		cache_position,
		**kwargs):

		bsz, q_len, _ = hidden_states.size()


		#query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		#key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
		value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
		query_states = query_states.view(bsz, q_len, self.head_dim)  
		key_states = key_states.view(bsz, q_len, self.head_dim)
		#value_states = value_states.view(bsz, q_len, self.head_dim)

		past_key_value = getattr(self, "past_key_value", past_key_value)
		cos, sin = self.rotary_emb(value_states, position_ids)
		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

		'''
			print(f"hidden_states size shape llama att", hidden_states.shape())
			# Ensure hidden_states is in 4D format (B, C, 1, S) for Conv2d operations
			if hidden_states.dim() == 3:
				hidden_states = hidden_states.unsqueeze(2)  # Add the singleton dimension

			# Apply Conv2d projections directly without manual weight splitting
			query_states = self.q_proj(hidden_states).squeeze(2)
			print("query_states shape:", query_states.shape) 
			key_states = self.k_proj(hidden_states).squeeze(2)
			print("key_states shape:", key_states.shape)
			value_states = self.v_proj(hidden_states).squeeze(2)
			print("value_states shape:", value_states.shape)

			# Adjust shapes for multi-head attention calculation
			# Assuming hidden_states originally (B, C, S), where C is channels (hidden_size)
			query_states = query_states.view(bsz, self.config.num_heads, q_len, -1).permute(0, 2, 1, 3)
			print("query_states shape view permute:", query_states.shape)

			key_states = key_states.view(bsz, self.config.num_heads, q_len, -1).permute(0, 2, 1, 3)
			print("key_statess shape view permute:", key_states.shape)

			value_states = value_states.view(bsz, self.config.num_heads, q_len, -1).permute(0, 2, 1, 3)
			print("value_states shape view permute:", value_states.shape)


			# Compute attention scores and apply mask
			attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * (self.config.head_dim ** -0.5)
			print("attn_weights shape torch matmul:", attn_weights.shape)

			if attention_mask is not None:
				attn_weights = attn_weights + attention_mask

			attn_weights = F.softmax(attn_weights, dim=-1)
			if self.attention_dropout > 0:
				attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

			# Compute attention output
			attn_output = torch.matmul(attn_weights, value_states)
			print("Compute attention output attn_output shape torch matmul:", attn_output.shape)


			# Reshape attn_output to (B, C, 1, S) for final projection
			attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, -1, 1, q_len)
			print(" Reshape attn_output to (B, C, 1, S) for final projection attn_output shape torch matmul:", attn_output.shape)


			# Apply final projection
			attn_output = self.o_proj(attn_output).squeeze(2)  # Assuming o_proj is also a Conv2d layer
			print("Apply final projection attn_output shape selfo_proj:", attn_output.shape)


			if not output_attentions:
				attn_weights = None

			return attn_output, None, past_key_value
		'''
			#setattr(self,)


class LlamaSdpaAttention(modeling_llama.LlamaSdpaAttention(modeling_llama.LlamaAttention)):
						
	def __init__(self, config, layer_idx):
		super().__init__(config, layer_idx)
		#print(f"hidden_states, shape:", hidden_states.shape)
		self.layer_idx = layer_idx

		setattr(self, 'q_proj',nn.Conv2d(in_channels=self.hidden_size,out_channels=self.num_heads*self.head_dim,kernel_size=1,bias=config.attention_bias,))

		setattr(self, 'k_proj',nn.Conv2d(in_channels=self.hidden_size,out_channels=self.num_key_value_heads * self.head_dim,kernel_size=1,bias=config.attention_bias,))

		setattr(self, 'v_proj',nn.Conv2d(in_channels=self.hidden_size,out_channels=self.num_key_value_heads*self.head_dim,kernel_size=1,bias=config.attention_bias,))

		setattr(self, 'o_proj',nn.Conv2d(in_channels=self.hidden_size,out_channels=self.hidden_size,kernel_size=1,bias=config.attention_bias,))


	def forward(
		self,
		hidden_states,
		attention_mask=None,
		position_ids=None,
		past_key_value=None,
		output_attentions=False,
		use_cache=False,
		cache_position=None,
	):

		#I want to cringe so into a pringle and then be tossed into the ocean
		# while referencing the self 
		q = self.q_proj(query)

		print(f"hidden_states, shape:", hidden_states.shape)
		bsz, q_len, _ = hidden_states.size() #torch.Size([1, 10, 2048])
		# Reshape hidden_states to [batch_size, hidden_size, seq_length, 1]
		hidden_states_reshaped = hidden_states.transpose(1 , 2).unsqueeze(-1)
		print(f"after hidden_states, shape:", hidden_states_reshaped.shape)

		query_states = self.q_proj(hidden_states_reshaped)
		print(f"query_states shape llamasdpaattention: ", query_states.shape)

		key_states = self.k_proj(hidden_states)#.unsqueeze(2) #torch.Size([1, 10, 1, 256])
		print(f"key_states shape llamasdpaattention: ", key_states.shape)

		value_states = self.v_proj(hidden_states)#.unsqueeze(2) #torch.Size([1, 10, 1, 256])
		print(f"value_states shape llamasdpaattention: ", value_states.shape)

		# Apply rotary positional embeddings
		cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)  
		# Print the shapes of cos and sin here for debugging
		#cos shape: torch.Size([1, 10, 64]) sin shape: torch.Size([1, 10, 64])
		print('cos shape:', cos.shape, 'sin shape:', sin.shape)

		#torch.Size([1, 10, 1, 2048]) torch.Size([1, 10, 64]) torch.Size([1, 10, 64]) torch.Size([1, 10, 1, 2048])
		query_states, key_states = apply_rotary_pos_emb(
			query_states, key_states, cos, sin, None
		)

		# Reshape for attention calculation 
		query_states = query_states.view(bsz, self.num_heads, q_len, self.head_dim).permute(0, 2, 1, 3)
		print(f"query_states shape view: ", query_states.shape)

		key_states = key_states.view(bsz, self.num_key_value_heads, q_len, self.head_dim).permute(0, 2, 1, 3)
		print(f"key_states shape view: ", key_states.shape)

		value_states = value_states.view(bsz, self.num_key_value_heads, q_len, self.head_dim).permute(0, 2, 1, 3)
		print(f"value_states shape view: ", value_states.shape)

		# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged withnon-contiguous inputs with custom attn_mask,
		# Reference: https://github.com/pytorch/pytorch/issues/112577.
		if query_states.device.type == "cuda" and causal_mask is not None:
			query_states = query_states.contiguous()
			key_states = key_states.contiguous()
			value_states = value_states.contiguous()
			
		# SDPA with memory-efficient backend  ... (Handling as discussed before) ...

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

		# Optionally return attention weights and past key/value states
		# depending on the parameters `output_attentions` and `use_cache`

		return attn_output, None, past_key_value  

class LlamaPreTrainedModel(modeling_llama.LlamaPreTrainedModel):
	#config_class = LlamaConfig
	base_model_prefix = "model"
	supports_gradient_checkpointing = True
	_no_split_modules = ["LlamaDecoderLayer"]
	_skip_keys_device_placement = ["past_key_values", "causal_mask"]
	_supports_flash_attn_2 = True
	_supports_sdpa = True
	_supports_cache_class = True
	
	def _init_weights(self, module):
		std = self.config.initializer_range
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()
	
	def _setup_cache(self, cache_cls, max_batch_size, max_cache_len = None):
		if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
			raise ValueError(
				"`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
				"make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
			)
	
		if max_cache_len > self.model.causal_mask.shape[-1] or self.device != self.model.causal_mask.device:
			causal_mask = torch.full((max_cache_len, max_cache_len), fill_value=1, device=self.device)
			self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
	
		for layer in self.model.layers:
			weights = layer.self_attn.o_proj.weight
			layer.self_attn.past_key_value = cache_cls(
				self.config, max_batch_size, max_cache_len, device=weights.device, dtype=weights.dtype
			)
	
	def _reset_cache(self):
		for layer in self.model.layers:
			layer.self_attn.past_key_value = None


class LlamaDecoderLayer(modeling_llama.LlamaDecoderLayer):
						
	def __init__(self, config, layer_idx):
		super().__init__(config, layer_idx)
		self.hidden_size = config.hidden_size
		
		setattr(self, 'self_attn', LlamaSdpaAttention(config, layer_idx))  #TODO implement the llama_attention types, needed because of various used from diff amt of heads
		setattr(self, 'mlp', LlamaMLP(config))
		setattr(self, 'input_layernorm', LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)) #maybe config.hidden_size, eps=config.rms_norm_eps
		setattr(self, 'post_attention_layernorm', LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)) #maybe config.hidden_size, eps=config.rms_norm_eps
		
	def forward(
		self,
		hidden_states,
		attention_mask = None,
		position_ids = None,
		past_key_value = None,
		output_attentions = False,
		use_cache = False,
		cache_position = None,
		**kwargs,
	):
		
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


"""
class LlamaPreTrainedModel(modeling_llama.LlamaPreTrainedModel):
							
	def __init__(self, config):
		super().__init__(config)
		# setattr(self,)
			
		# Register hook for unsqueezing nn.Linear parameters to match nn.Conv2d parameter spec
		self._register_load_state_dict_pre_hook(linear_to_conv2d_map)
"""
@dataclass
class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states, #: torch.Tensor,
        value_states,# torch.Tensor,
        layer_idx: int,#: int,
        cache_kwargs = None,#: Optional[Dict[str, Any]] = None,
	):
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx = 0):
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self):
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length, layer_idx = 0):
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache = []
        self.value_cache = []
        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx = 0):
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states,
        value_states,
        layer_idx: int,
        cache_kwargs = None,
    ):
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx = 0):
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self):
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self):
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values):
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

PretrainedConfig = {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": False,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5632,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 22,
  "num_key_value_heads": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": 0,
  "rope_theta": 10000.0,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.35.0",
  "use_cache": True,
  "vocab_size": 32000
}

class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the `max_position_embeddings`, `hidden_size` and `num_attention_heads`
            required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        self.key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        self.value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        self.seen_tokens = 0

    def update(
        self,
        key_states,
        value_states,
        layer_idx = 0,
        cache_kwargs = None,
    ):
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for. Kept for backward compatibility
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` just needs the `q_len`
                to know how much of the cache it should overwrite.

        Return:
            A tuple containing the updated key and value states.
        """
        new_cache_positions = cache_kwargs.get("cache_position")
        k_out = self.key_cache
        v_out = self.value_cache

        k_out[:, :, new_cache_positions] = key_states
        v_out[:, :, new_cache_positions] = value_states

        self.seen_tokens += key_states.shape[2]
        return k_out, v_out

    def get_seq_length(self, layer_idx = 0):
        """Returns the sequence length of the cached states that were seen by the model. `layer_idx` kept for BC"""
        return self.seen_tokens

    def get_usable_length(self, new_sequence_length=None, layer_idx = 0):
        return self.seen_tokens

    def get_max_length(self):
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return self.max_cache_len

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        device = self.key_cache.device
        self.key_cache = self.key_cache.index_select(0, beam_idx.to(device))
        device = self.value_cache.device
        self.value_cache = self.value_cache.index_select(0, beam_idx.to(device))

    def to_legacy_cache(self):
        """Dummy function for BC. We have to keep it because otherwise the call in the forward of models will break it"""
        return None




class LlamaModel(modeling_llama.LlamaModel):
								
	def __init__(self, config):
		super().__init__(config)
		setattr(self, 'embed_tokens', nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx))
		self._register_load_state_dict_pre_hook(linear_to_conv2d_map)
		setattr(self, 'layers', nn.ModuleList([LlamaDecoderLayer(config, layer_idx=0) for _ in range(config.num_hidden_layers)])) #TODO change config.n_layers, is not real for llama
		setattr(self, 'norm', LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps))

		causal_mask = torch.full((config.max_position_embeddings, config.max_position_embeddings), fill_value=1)
		self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
		# Initialize weights and apply final processing
		self.post_init()
		
		causal_mask = torch.full((config.max_position_embeddings, config.max_position_embeddings), fill_value=1)
		self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
		# Initialize weights and apply final processing
		self.post_init()
			
	


class LlamaForCausalLM(modeling_llama.LlamaForCausalLM):
									
	def __init__(self, config):
		super().__init__(config)
		self.vocab_size = config.vocab_size
		
		setattr(self, 'model', LlamaModel(config))
		
		self._register_load_state_dict_pre_hook(linear_to_conv2d_map)
		
		setattr(self, 'lm_head', 
			nn.Conv2d(
				in_channels=self.config.hidden_size,
				out_channels=self.config.vocab_size,
				kernel_size=1,
				bias=config.attention_bias,
		))

		self.post_init()

		def forward(
			self,
			input_ids = None,
			attention_mask = None,
			position_ids = None,
			past_key_values = None,
			inputs_embeds = None,
			labels = None,
			use_cache = None,
			output_attentions = None,
			output_hidden_states = None,
			return_dict = None,
			cache_position = None,):
		
			output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		
			output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
		
			return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		
			# decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
			print(f"outputs shape", outputs.shape())
			outputs = self.model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_values=past_key_values,
				inputs_embeds=inputs_embeds,
				use_cache=use_cache,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
				cache_position=cache_position,
			)
			
			hidden_states = outputs[0]
			if self.config.pretraining_tp > 1:
				raise NotImplementedError()
					#lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
					#logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
					#logits = torch.cat(logits, dim=-1)
			else:
				logits = self.lm_head(hidden_states)
				
			logits = logits.float()
		
			loss = None
			if labels is not None:
				raise NotImplementedError("labels not implemented, must be none")
			
			if not return_dict:
				output = (logits,) + outputs[1:]
				return (loss,) + output if loss is not None else output
			
			return {
				"loss" : loss,
				"logits" : logits,
				"past_key_values" : outputs.past_key_values,
				"hidden_states" : outputs.hidden_states,
				"attentions" : outputs.attentions,
			}
		
		
		def prepare_inputs_for_generation(
			self,
			input_ids,
			past_key_values=None,
			attention_mask=None,
			inputs_embeds=None,
			**kwargs):
			
			past_length = 0
			if past_key_values is not None:
				if isinstance(past_key_values, Cache):
					cache_length = past_key_values.get_seq_length()
					past_length = past_key_values.seen_tokens
					max_cache_length = past_key_values.get_max_length()
				else:
					cache_length = past_length = past_key_values[0][0].shape[2]
					max_cache_length = None
		
				# Keep only the unprocessed tokens:
				# 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
				# some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
				# input)
				if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
					input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
				# 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
				# input_ids based on the past_length.
				elif past_length < input_ids.shape[1]:
					input_ids = input_ids[:, past_length:]
				# 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
		
				# If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
				if (
					max_cache_length is not None
					and attention_mask is not None
					and cache_length + input_ids.shape[1] > max_cache_length
				):
					attention_mask = attention_mask[:, -max_cache_length:]
		
			position_ids = kwargs.get("position_ids", None)
			if attention_mask is not None and position_ids is None:
				# create position_ids on the fly for batch generation
				position_ids = attention_mask.long().cumsum(-1) - 1
				position_ids.masked_fill_(attention_mask == 0, 1)
				if past_key_values:
					position_ids = position_ids[:, -input_ids.shape[1] :]
		
			if past_key_value := getattr(self.model.layers[0].self_attn, "past_key_value", None):
				# generation with static cache
				past_length = past_key_value.get_seq_length()
				input_ids = input_ids[:, past_length:]
				position_ids = position_ids[:, past_length:]
		
			# TODO @gante we should only keep a `cache_position` in generate, and do +=1.
			# same goes for position ids. Could also help with continued generation.
			cache_position = kwargs.get("cache_position", None)
			if cache_position is None:
				cache_position = torch.arange(
					past_length, past_length + position_ids.shape[-1], device=position_ids.device
				)
		
			# if `inputs_embeds` are passed, we only want to use them in the 1st generation step
			if inputs_embeds is not None and past_key_values is None:
				model_inputs = {"inputs_embeds": inputs_embeds}
			else:
				model_inputs = {"input_ids": input_ids}
		
			model_inputs.update(
				{
					"position_ids": position_ids,
					"cache_position": cache_position,
					"past_key_values": past_key_values,
					"use_cache": kwargs.get("use_cache"),
					"attention_mask": attention_mask,
				}
			)
			return model_inputs
	
	@staticmethod
	def _reorder_cache(past_key_values, beam_idx):
		reordered_past = ()
		for layer_past in past_key_values:
			reordered_past += (
				tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
			)
		return reordered_past
		