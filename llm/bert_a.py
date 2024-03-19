import torch
from transformers import BertModel, BertTokenizer  # pip install transformers==2.2.0

# 这里我们调用bert-base模型，同时模型的词典经过小写处理
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

print(tokenizer.tokenize("I have a good time, thank you."))

input_text = "Here is some text to encode"
input_ids = tokenizer.encode(input_text, add_special_tokens=True)

print(f"\ninput_text: \n {input_text}")
print(f"\ntoken_ids: \n {input_ids}")

input_ids = torch.tensor([input_ids])

# 获得BERT模型最后一个隐层结果
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
    print(f"\nlast_hidden_states:\n {last_hidden_states}")
