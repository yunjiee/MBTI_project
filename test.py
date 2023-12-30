import torch
from transformers import BertTokenizerFast , AutoModel, AutoTokenizer
#from transformers import BertTokenizer
#from IPython.display import clear_output
#pip install -U transformers

PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型
'''
# 取得此預訓練模型所使用的 tokenizer
#tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
from transformers import (
   BertTokenizerFast,
   AutoModelForMaskedLM,
   AutoModelForCausalLM,
   AutoModelForTokenClassification,
)

# masked language model (ALBERT, BERT)
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
#maskedLM_model = AutoModelForMaskedLM.from_pretrained('ckiplab/albert-tiny-chinese') # or other models above
maskedLM_model = BertTokenizerFast.from_pretrained('ckiplab/albert-tiny-chinese') # or other models above

'''
from transformers import (
  BertTokenizerFast,
  AutoModel,
)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
maskedLM_model = AutoModel.from_pretrained('ckiplab/bert-base-chinese')

#clear_output()
#print("PyTorch 版本：", torch.__version__)

vocab = tokenizer.vocab #作為中文分詞仍然太少了，有很多字並未收入進去的
#print("字典大小：", len(vocab))
#print(vocab)
#幫你把一個個詞切開
text = "等到潮水 [MASK] 了，就知道誰沒穿褲子。"
#text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"

tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(text)
print(tokens[:10], '...')
print(ids[:10], '...')

##################### language model #####################
"""
這段程式碼載入已經訓練好的 masked 語言模型並對有 [MASK] 的句子做預測
"""
from transformers import BertForMaskedLM

# 除了 tokens 以外我們還需要辨別句子的 segment ids
#拿id 製作成 token tensor  #轉成tensor格式
tokens_tensor = torch.tensor([ids])  # (1, seq_len) 
#複製一個長度一樣的且都為零的segment tensor
segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
#用這model來進行預估
#maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
#clear_output()

# 使用 masked LM 估計 [MASK] 位置所代表的實際 token 
maskedLM_model.eval()
with torch.no_grad():
    outputs = maskedLM_model(tokens_tensor, segments_tensors) #輸出對整句話的預測
    predictions = outputs[0] #取的預測值
    # (1, seq_len, num_hidden_units)
del maskedLM_model

# 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
masked_index = 4
k = 10 #有點類似正規法 #把預測前幾高的"字"，表示出來
#取得[MASK]的topk預測
probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k) #k=取前幾的排名的方式
#把預測的id轉回token以利後續處理 #有點類似查表的功能
predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

# 顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
print("輸入 tokens ：", tokens[:10], '...')
print('-' * 50)
for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
    tokens[masked_index] = t
    print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokens))


