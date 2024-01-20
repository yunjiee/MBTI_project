from transformers import AutoTokenizer, BertForPreTraining
#pip install xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import numpy as np
import torch

# 数据预处理函数
def preprocess_text(text):
    text = str(text).lower()  # 确保文本是字符串类型并转换为小写
    text = re.sub(r'http\S+', ' ', text)  # 移除URLs
    text = re.sub(r"\\t", "  ", text)
    text = re.sub(r"\s{2,}", " ", text)  # 去除多余空格
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格为一个空格
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?\(\)\']', '', text)
    labels = ["infp", "infj", "intj", "intp", "isfp", "isfj", "istj", "istp", "enfp", "enfj", "entj", "entp", "esfp", "esfj", "estj", "estp"]
    pattern = r'(' + '|'.join(map(re.escape, labels)) + ')'
    text = re.sub(pattern, 'type', text)
    text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " 've", text)
    text = re.sub(r"\'t", " 't", text)
    text = re.sub(r"\'re", " 're", text)
    text = re.sub(r"\'d", " 'd", text)
    text = re.sub(r"\'ll", " 'll", text)
    text = re.sub(r", ", " , ", text)
    text = re.sub(r"\. ", " . ", text)
    text = re.sub(r"'", " ' ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\? ", " ? ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\?{2,}", " ? ", text)
    text = re.sub(r"\s{2,}", "", text)  # 去除多余空格
    text = re.sub(r"\!{2,}", " ! ", text)  # 去除多余空格
    return text

# 定义函数以训练和评估模型
def read_and_preprocess_data(csv_path): 
    data = pd.read_csv(csv_path)
    data['processed_content'] = data['processed_content'].apply(preprocess_text)
    data_texts = data['processed_content']
    labels = data['type']
    return labels ,data_texts

def evaluate_model(model, X_test, y_test):
    # 模型评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return y_pred

def predict(model, new_texts):
    new_texts = new_data(new_texts)
    predictions = model.predict(new_texts)
    # 将数字标签转换为相应的标签
    inverse_label_mapping = {k: i for i, k in label_mapping.items()}
    predicted_labels = [inverse_label_mapping[prediction] for prediction in predictions]
    return predicted_labels

# 使用模型进行新数据预测
def new_data(new_texts):
    new_texts = [preprocess_text(text) for text in new_texts]
    new_texts = input_list.transform(new_texts)
    return new_texts

# 调用 train_model 函数训练模型
csv_path = 'C:/Users/student/yunjiee-python/MBTI_project/full2/data/new_processed_all_posts_data.csv'
labels ,data_texts =read_and_preprocess_data(csv_path)
label_mapping = {'infp': 0, 'infj': 1, "intj": 2, "intp": 3, "isfp": 4, "isfj": 5, "istj": 6, "istp": 7, "enfp": 8, "enfj": 9, "entj": 10, "entp": 11, "esfp": 12, "esfj": 13, "estj": 14, "estp": 15}
label_mapping_inverse = {k: i for i, k in label_mapping.items()}
# 将标签转换为数字
y = [label_mapping[label] for label in labels]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForPreTraining.from_pretrained("bert-base-uncased")

# 使用tokenizer处理每个文本
inputs = [tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) for text in data_texts]

# 打印结果
features = []
for inputs in data_texts:
    with torch.no_grad():
        outputs = model(**inputs)
        # 假设我们使用最后一层的隐藏状态的平均值作为特征
        features.append(outputs.last_hidden_state.mean(dim=1).numpy())
    
X = np.vstack(features)


# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 模型训练
model = XGBClassifier(objective='multi:softprob', num_class=16)
model.fit(X_train, y_train)
# 评估模型
evaluate_model(model, X_test, y_test)

# 预测新数据
new_texts = ["so i had this idea for a thread a couple of days ago , and then today i saw jonkay asking for one , how' s that for synchronicity astralflame ?"]
predicted_labels = predict(model, new_texts)
print("Predicted MBTI type:", predicted_labels)










def bert_encode(texts, tokenizer, max_len=50):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)