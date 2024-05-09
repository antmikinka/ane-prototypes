#

#from llama import LlamaForCausalLM as ane_llama

#optimized_model = ane_llama.LlamaForCausalLM(
#    baseline_model.config).eval()
#optimized_model.load_state_dict(baseline_model.state_dict())
import torch
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

   

import transformers
from transformers import AutoModelForCausalLM
import coremltools as ct
import numpy as np
#model_name = "openlm-research/open_llama_3b_v2"
model_name = "google/gemma-2b"
baseline_model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=False, torch_dtype=torch.float16, torchscript=True).eval()

print("baseline_model state_dict----------------------------------------------")
#for param_tensor in baseline_model.state_dict():
    #print(param_tensor, "\t", baseline_model.state_dict()[param_tensor].size())
print(baseline_model)
print("baseline_model state_dict----------------------------------------------")

#print(baseline_model.weight)


#import ANE_LLAMA
#from ANE_LLAMA import LlamaForCausalLM
#optimized_model = ANE_LLAMA.LlamaForCausalLM(baseline_model.config).eval()
#print(optimized_model)
#optimized_model.load_state_dict(baseline_model.state_dict())

#print(optimized_model.sample_inputs())

#print(optimized_model.output_types())

def transform_hf_weights(state_dict):
    for k in list(state_dict.keys()):
        print(k, state_dict[k].shape)
    
    
print("optimized_model state_dict---------AFTER LOADING-------------------------------------")
#print(optimized_model)
#transform_hf_weights(optimized_model.state_dict())
#for param_tensor in optimized_model.state_dict():
    #print(param_tensor, "\t", optimized_model.state_dict()[param_tensor].size())
print("optimized_model state_dict----------AFTER LOADING------------------------------------")
"""
batch_size = 512
sequence_length=2048
num_attention_heads=32
hidden_size = 2048
num_hidden_layers = 22
vocab_size = 32000

input_ids = ct.TensorType(shape=(batch_size, sequence_length), dtype=np.int32)  # Assuming batch size of 1 for simplicity
attention_mask = ct.TensorType(shape=(batch_size, sequence_length), dtype=np.int32)  # Matching input_ids shape

# Example for a single layer's past_key_values; you'll need to replicate this based on num_hidden_layers
pkv_shape = (batch_size, num_attention_heads, sequence_length, hidden_size // num_attention_heads)
past_key_values = [ct.TensorType(shape=pkv_shape, dtype=np.float32) for _ in range(2 * num_hidden_layers)]  # 2 for each key and value
"""
"""
logits = ct.TensorType(shape=(batch_size, sequence_length, vocab_size), dtype=np.float32)


model_input = [input_ids, attention_mask] + past_key_values
model_output = [logits] + past_key_values  # Define past_key_values_output similarly

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
"""
#input_ids = tokenizer.encode(tokenizer.bos_token, return_tensors="pt")
#attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
#position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0)

# Example function to be traced
#def simplified_forward(input_ids, attention_mask, position_ids):
    #return optimized_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
#inputs=[ct.TensorType(shape=input_ids.shape),
#                                           ct.TensorType(shape=attention_mask.shape),
#                                           ct.TensorType(shape=position_ids.shape)],
# Tracing
#traced_model = torch.jit.trace(simplified_forward, (input_ids, attention_mask, position_ids))
"""
import torch
traced_optimized_model = torch.jit.trace(
    optimized_model,
    (tokenized["input_ids"], tokenized["attention_mask"]
))
"""
"""
import coremltools as ct
import numpy as np
ane_mlpackage_obj = ct.convert(
    traced_optimized_model,
    inputs=model_input,
    outputs=model_output,
    compute_precision=compute_precision,
    minimum_deployment_target=ct.target.iOS17, # To allow float16 inputs + outputs.
    convert_to="mlprogram",
)
"""
out_path = "gemma-ane-7b.mlpackage"
#ane_mlpackage_obj.save(out_path)



