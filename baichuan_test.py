from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import torch
tokenizer = AutoTokenizer.from_pretrained(
    "baichuan-inc/baichuan-7B",
    model_max_length=512,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.unk_token


conversations = ['你好','世界']
input_ids = tokenizer(
    conversations,
    return_tensors="pt",
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
).input_ids
for input in input_ids:
    print(tokenizer.decode(input))

sys.exit(0)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B", device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))