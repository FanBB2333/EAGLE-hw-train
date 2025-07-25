import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载原始 LLaMA3 模型和分词器
model_name = "model/LLM/Llama-3.2-1B-Instruct"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# 从 Safetensor 文件中加载参数
safetensor_path = "model/LLM/Llama-3.2-1B-Instruct"
safetensor_params = torch.load(safetensor_path)

# 替换模型参数
for name, param in model.named_parameters():
    if name in safetensor_params:
        param.data.copy_(safetensor_params[name])

# 保存新的 LLaMA3 模型
new_model_name = "model/LLM/Llama-3.2-1B-Instruct"
model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)