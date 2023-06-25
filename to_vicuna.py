#!/usr/bin/env python
#coding=utf8
import sys
import json


# num = 0
# with open('./data/tigerbot-alpaca-zh-0.5m.json', 'r') as f:
#     with open('./data/tigerbot-alpaca-zh-0.5m-vicuna.json', 'w', encoding='utf8') as wf:
#       for line in f:
#           item = json.loads(line.strip())
#           num+=1
#           vicuna_item = {}
#           human = {'from':'human', 'value':item['instruction'] + item['input']}
#           gpt = {'from':'gpt', 'value':item['output']}
#           vicuna_item['id'] = f'tigerbot_{num}'
#           vicuna_item['conversations'] = [human, gpt]
#           wf.write(json.dumps(vicuna_item, ensure_ascii=False)+'\n')


# num = 0
# with open('./data/CoT_Chinese_data.json', 'r') as f:
#     with open('./data/CoT_Chinese_data_vicuna.json', 'w', encoding='utf8') as wf:
#       items = json.loads(f.read())
#       for item in items:
#           num+=1
#           vicuna_item = {}
#           human = {'from':'human', 'value':item['instruction'] + item['input']}
#           gpt = {'from':'gpt', 'value':item['output']}
#           vicuna_item['id'] = f'tigerbot_{num}'
#           vicuna_item['conversations'] = [human, gpt]
#           wf.write(json.dumps(vicuna_item, ensure_ascii=False)+'\n')

with open('/data/FastChat/data/dummy_conversation.json', 'r') as f:
    with open('/data/FastChat/data/dummy_conversation_vicuna.json', 'w') as wf:
      datas = json.loads(f.read())
      for data in datas:
         wf.write(json.dumps(data, ensure_ascii=False)+'\n')
