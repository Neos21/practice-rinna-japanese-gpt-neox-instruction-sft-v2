import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# - Python v3.8 以降というのは Poetry にお任せ
# - import してるので必須で入れた (poetry add でバージョン番号はお任せ
#   - torch = "^2.4.0"
#   - transformers = "^4.44.0"
# - Cuda 利用のためにいるらしいので入れた
#   - accelerate = "^0.33.0"
# - ないと実行時に怒られたので入れた
#   - sentencepiece = "^0.2.0"
#   - protobuf = "^5.27.3"
# - 参考
#   - https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2
#   - https://zenn.dev/fusic/articles/try-various-llms

model_name = 'rinna/japanese-gpt-neox-3.6b-instruction-sft-v2'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Start : Model Name [', model_name, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Tokenizer')  # ワーニングが出るから legacy と clean_up_tokenization_spaces を入れた・float16 は高速化のため
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False, clean_up_tokenization_spaces=True, torch_dtype=torch.float16)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model')
model = AutoModelForCausalLM.from_pretrained(model_name)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Is Cuda Available?')
if torch.cuda.is_available():
    print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Use Cuda')
    model = model.to('cuda')

input_text = 'こんにちは世界！'
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt : [', input_text, ']')

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Prompt Tokenizer')
inputs = tokenizer(input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Model Generate')
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), 'Decode')
output_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)

print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '----------')
print(output_text)
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))), '==========')
