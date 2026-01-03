import torch
from transformers import AutoTokenizer, Qwen3Config, AutoModelForCausalLM
from custom_qwen3 import  NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding
import onnx
import numpy as np
from transformers.cache_utils import DynamicCache

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

####Offical Model####

config = Qwen3Config.from_pretrained(model_name,attn_implementation="eager")  
official_model = AutoModelForCausalLM.from_pretrained(model_name,config=config).eval()

qwenConfig = official_model.config
custom_model =  NNTrainerQwen3ForCausalLM(qwenConfig).eval()
custom_model.load_state_dict(official_model.state_dict(),strict=False)

head_dim = config.head_dim
num_layers = config.num_hidden_layers

#Making dummy inputs
input_ids = torch.rand(256,1)
cos = torch.rand(1,256,head_dim)
sin = torch.rand(1,256,head_dim)
variance_epsilon = torch.tensor([[1e-6,]])
causal_mask = torch.rand(1,1,1,256)
# Making dummy inputs for onnx model with dynamic dimensions
past_cache = {}
num_layers = 28

# Create sample inputs with concrete dimensions for export
sample_seq_len = 1
sample_past_seq_len = 1

for layer_id in range(num_layers):
    past_cache[layer_id] = {
        "key": torch.rand(1, 16, sample_past_seq_len, 128),  # (batch=1, heads=16, past_seq, hidden_dim=128)
        "value": torch.rand(1, 16, sample_past_seq_len, 128),
    }
    
# Prepare input names
input_names = ['input', 'cos', 'sin', 'variance_epsilon']
for i in range(num_layers):
    input_names.append(f'past_cache_{i}_key')
    input_names.append(f'past_cache_{i}_value')
input_names.append('causal_mask')    

# Prepare output names
output_names = ['output']
for i in range(num_layers):
    output_names.append(f'present_cache_{i}_key')
    output_names.append(f'present_cache_{i}_value')

# Define dynamic shapes for ONNX export (batch size is static = 1)
seq_dim = torch.export.Dim("seq_len")
past_seq_dim = torch.export.Dim("past_seq_len")

dynamic_shapes = [
    {0: seq_dim},  # input - [1, seq_len]
    {1: seq_dim},  # cos - [1, seq_len, hidden_dim]  
    {1: seq_dim},  # sin - [1, seq_len, hidden_dim]
    None,  # variance_epsilon - [1, 1]
    None,
]

# Add dynamic shapes for past_cache inputs
# for i in range(num_layers):
#     dynamic_shapes.append({2: past_seq_dim})  # past_cache_{i}_key - [1, 16, past_seq_len, 128]
#     dynamic_shapes.append({2: past_seq_dim})  # past_cache_{i}_value - [1, 16, past_seq_len, 128]

# dynamic_shapes.append({2: seq_dim, 3: seq_dim})  # causal_mask - [1, 1, seq_len, seq_len]

# Export with dynamic shapes
torch.onnx.export(
    custom_model, 
    (input_ids, cos, sin, variance_epsilon, past_cache, causal_mask),
    './qwen3_model.onnx',
    export_params=True,
    opset_version=23,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    keep_initializers_as_inputs=False,
    dynamo=True,
)

print("<Model exported successfully>")
