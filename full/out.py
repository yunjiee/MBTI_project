### 加載訓練好的模型
from transformers import BertForSequenceClassification, BertTokenizer
model_path = r"D:\project\MBTI_project\full\output"
model = BertForSequenceClassification.from_pretrained(model_path)
print(model)
### 加載詞彙表
tokenizer = BertTokenizer.from_pretrained(model_path)
print(tokenizer)

### 進行預測
import torch
import pandas as pd
df = pd.read_csv("D:\project\MBTI_project\data_personality\enfj_data.csv")  # 替换为您 CSV 文件的实际路径
texts = df['Content'].tolist()  # 替换 'text_column_name' 为您的文本所在列的列名

## 假设 new_data 是一个包含新文本的列表
'''
# 这里需要先从CSV文件中读取数据，然后进行处理
# 假设您已经将CSV中的文本载入到一个名为texts的列表中
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
#truncation=True 将过长的序列截断至max_length

import sys
sys.path.append('/content/drive/My Drive/full/')  # 更新为您的路径
'''
from clean import preprocess_text

# 预处理文本
texts = [preprocess_text(text) for text in texts]
##對模型進行預處理
inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
#tokenizer 用于将文本转换为模型可以理解的格式
print(inputs)
#錯誤:　要求截断至 max_length，但未提供最大长度，且模型没有预定义的最大长度。默认为不截断
##使用模型進行預測
#torch.no_grad() 表示在此过程中不计算梯度
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
print(logits)
label_list = ["infp", "infj", "intj", "intp", "isfp", "isfj", "istj", "istp", "enfp", "enfj", "entj", "entp", "esfp", "esfj", "estj", "estp"]
id_to_label = {i: label for i, label in enumerate(label_list)}

predictions = torch.argmax(logits, dim=-1) #torch.argmax(logits, dim=-1)选出每个输入文本最可能的类别
predictions = [id_to_label[id] for id in predictions.numpy()]

print(predictions)
