# ane-prototypes
ANE optimized prototypes

Due to the hierarchical structure of my environment and duplicate file names
*I pre-named files*
- example models-llama.py, this file should be modified to remove “models-“ making it just “llama.py”
- (yes that file would be in the “models” folder”)

**info restated above:**
- all files in “models” folder should be renamed, removing the “models-“ prefix in the file names. 
- These files should be moved into “more-ane-transformes/models”
---------------------------------------------------------------------

**text - results/python3_.models-ane-llama_results.txt**
- missing some weights in state_dict
- also some unexpected keys in state_dict
---------------------------------------------------------------------
**text - results/python3_.models-llama.py_results.txt**

- rebuilt more-ane-transformers with Llama in mind, kept the GPT class (didn’t understand  enough) 
- had issues with GPT class
---------------------------------------------------------------------

**text - results/python3_.models-llama2ane.py_results.txt**
- tried a method like distilbert from apple (where you switch classes for more optimized ones, should have been easy convert that is dynamic for any llama model, if corrected would be huge)
 
---------------------------------------------------------------------

text - results/python3_ANE_LLAMA.py_results.txt - file:
- computes with no problems, I think it just needs to be implemented
 
---------------------------------------------------------------------

text - results/python3_ANE_LLAMAv2.py_results.txt
- had an issue with LlamaAttention Class, missing / extra attribute “attention_dropout”
- not sure why there are two ANE_LLAMA files…
 
---------------------------------------------------------------------

models/models-modeling_llama.py 
- yes I realize that whatever file was using this import, was actually using the env import and not that modified file import lol
- otherwise just used as a quick ref for llama architecture

 
---------------------------------------------------------------------

models/models-revisedCasualSelfAttention.py
- freebie, haven’t comprehended this yet, not sure if it’ll help
 
 ---------------------------------------------------------------------




ane-gemma-7b.py
- pretty sure would trace etc, but ran out of ram
- I didn’t see a class in here, may not do much
- started to comment things out to see the parameters matching correctly

ane-llama.py
- pretty sure would trace etc, but ran out of ram
- this file uses the ANE_LLAMA class, so this would CONVERT
- started to comment things out to see the parameters matching correctly

correct_env_settings.txt
- my windows pc machine env
