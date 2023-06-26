from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
import sys
import torch



# model_path = 'THUDM/chatglm2-6b'

# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
# model = model.eval()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)

model_path = 'baichuan-inc/baichuan-7B'
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=True,trust_remote_code=True)
inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))