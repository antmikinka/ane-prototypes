import transformers
model_name = "openlm-research/open_llama_3b_v2"
baseline_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=False,
    torchscript=True,
).eval()


import llama as ane_llama
optimized_model = ane_llama.LlamaForCausalLM(
    baseline_model.config).eval()
optimized_model.load_state_dict(baseline_model.state_dict())





tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenized = tokenizer(
    ["Sample input text to trace the model"],
    return_tensors="pt",
    max_length=128,  # token sequence length
    padding="max_length",
)