import torch
from transformers import AutoTokenizer, Qwen3Config, AutoModelForCausalLM
from custom_qwen3 import  NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding
from transformers.cache_utils import DynamicCache

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

####Offical Model####

config = Qwen3Config.from_pretrained(model_name,attn_implementation="eager")  
print(config)
exit()
official_model = AutoModelForCausalLM.from_pretrained(model_name,config=config).eval()

# Prompt
prompt = "What is the capital of India ?" 

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)

print("\nInput prompt:",prompt)

enc = tokenizer(text, return_tensors="pt")
input_ids=enc.input_ids
generated = input_ids.clone()
    
output = official_model.generate(
    input_ids=input_ids,
    do_sample=False,
    max_new_tokens=2,
    use_cache=True,
)

decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print("Official model output: ",decoded,"\n")

####Custom Model####

qwenConfig = official_model.config
custom_model =  NNTrainerQwen3ForCausalLM(qwenConfig).eval()
custom_model.load_state_dict(official_model.state_dict(),strict=False)

rotary_emb = Qwen3RotaryEmbedding(qwenConfig)

generated = input_ids.clone() # input to official Qwen model
cur_len =  enc.input_ids.size(1)

position_ids = torch.arange(cur_len).unsqueeze(0)
cos, sin = rotary_emb(generated.to(torch.float32), position_ids)
variance_epsilon = torch.tensor([[1e-6,]])

# Tokens to generate
num_tokens_to_generate = 2
causal_mask = torch.full((1, 1, generated.shape[1], generated.shape[1]), float(-3.4028e+38))
causal_mask = torch.triu(causal_mask, diagonal=1) 
past_cache = DynamicCache()
response = []

outputs,_, = custom_model(
        generated.transpose(0,1), 
        cos,
        sin,
        variance_epsilon,
        past_cache,
        causal_mask,
    )

next_token_logits = outputs[0][:, cur_len - 1, :]
next_token_id = torch.argmax(next_token_logits,dim=-1)
response.append(next_token_id.item())

for step in range(num_tokens_to_generate - 1):  
    
    past_seen_tokens = past_cache.get_seq_length()
    position_ids = torch.arange(past_seen_tokens,past_seen_tokens+1).unsqueeze(0)
    cos, sin = rotary_emb(next_token_id.to(torch.float32), position_ids)
    
    attention_mask = torch.zeros((1,past_seen_tokens+1)).to(torch.int64)
  
    outputs,_ = custom_model(
        next_token_id.unsqueeze(1), 
        cos,
        sin,
        variance_epsilon,
        past_cache,
        attention_mask,
    )

    next_token_logits = outputs[0][:,-1, :]
    next_token_id = torch.argmax(next_token_logits,dim=-1)
    if next_token_id == tokenizer.eos_token_id:
        break
    response.append(next_token_id.item())

decoded = tokenizer.decode(response, skip_special_tokens=True)

print("\nCustom Model: ",decoded)