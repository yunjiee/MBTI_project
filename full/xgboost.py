import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import re

def preprocess_text(text):
    text = str(text).lower()  # 确保文本是字符串类型并转换为小写
    text = re.sub(r'http\S+', ' ', text)  # 移除URLs
    text = re.sub(r"\\t", "  ", text)
    pattern = r'(' + '|'.join(map(re.escape, labels)) + ')'
    text = re.sub(r"\s{2,}", " ", text) #去除多餘空
    text = re.sub(r'\s+', ' ', text) # 合并多个空格为一个空格
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?\(\)\']', '', text)
    text = re.sub(pattern, 'type', text)
    text = re.sub(r"\'s", " \'s", text) 
    text = re.sub(r"\'ve", " \'ve", text) 
    text = re.sub(r"\'t", " \'t", text) 
    text = re.sub(r"\'re", " \'re", text) 
    text = re.sub(r"\'d", " \'d", text) 
    text = re.sub(r"\'ll", " \'ll", text) 
    text = re.sub(r", ", " , ", text)
    text = re.sub(r"\. ", " . ", text) 
    text = re.sub(r"'", " ' ", text)  
    text = re.sub(r"!", " ! ", text) 
    text = re.sub(r"\(", " ( ", text) 
    text = re.sub(r"\)", " ) ", text) 
    text = re.sub(r"\? ", " ? ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\?{2,}", " ? ", text)
    text = re.sub(r"\s{2,}", "", text) #去除多餘空
    text = re.sub(r"\!{2,}", " ! ", text) #去除多餘空
    return text

# 示例数据集
csv_path = 'D:/project/MBTI_project/full/data/data.csv'  # 您的文本数据
labels = ["infp", "infj", "intj", "intp", "isfp", "isfj", "istj", "istp", "enfp", "enfj", "entj", "entp", "esfp", "esfj", "estj", "estp"] # 对应的MBTI类型


# 特征提取(特徵集)
vectorizer = TfidfVectorizer()
with open('D:/project/MBTI_project/full/output/vocab.txt', 'r', encoding='utf-8') as file:
    texts = [preprocess_text(line.strip()) for line in file]
X = vectorizer.fit_transform(texts)

# 数据划分(设置目标为多类分类 (multi:softprob) 和类别数量)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 模型训练
model = xgb.XGBClassifier(objective='multi:softprob', num_class=16)  # 假设MBTI类型有16个类别
model.fit(X_train, y_train)

# 模型评估(使用测试集（X_test, y_test）来预测 MBTI 类型)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# 使用模型进行新数据预测
new_texts = ["so i had this idea for a thread a couple of days ago , and then today i saw jonkay asking for one , how' s that for synchronicity astralflame ?"]
new_texts = preprocess_text(new_texts)
new_texts = vectorizer.transform(new_texts)
predictions = model.predict(new_texts)
print("Predicted MBTI type:", predictions)