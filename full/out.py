from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
import re
label_list = ["infp", "infj", "intj", "intp", "isfp", "isfj", "istj", "istp", "enfp", "enfj", "entj", "entp", "esfp", "esfj", "estj", "estp"]

def preprocess_text(text):
    text = str(text).lower()  # 确保文本是字符串类型并转换为小写
    text = re.sub(r'http\S+', ' ', text)  # 移除URLs
    text = re.sub(r"\\t", "  ", text)
    # 构建正则表达式，使用 | 分隔不要的词语，并添加括号来捕获匹配项
    pattern = r'(' + '|'.join(map(re.escape, label_list)) + ')'
    text = re.sub(r"\s{2,}", " ", text) #去除多餘空
    text = re.sub(r'\s+', ' ', text) # 合并多个空格为一个空格
    #print("111111clean11111",text)
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?\(\)\']', '', text)
    # 将匹配到的词语替换为 <type>
    text = re.sub(pattern, 'type', text)
    text = re.sub(r"\'s", " \'s", text) 
    text = re.sub(r"\'ve", " \'ve", text) 
    text = re.sub(r"\'t", " \'t", text) 
    text = re.sub(r"\'re", " \'re", text) 
    text = re.sub(r"\'d", " \'d", text) 
    text = re.sub(r"\'ll", " \'ll", text) 
    text = re.sub(r", ", " , ", text)
    #print("11111111111111111111111",text)
    text = re.sub(r"\. ", " . ", text) 
    #print("22222222222222222222222",text)
    text = re.sub(r"'", " ' ", text)  
    text = re.sub(r"!", " ! ", text) 
    text = re.sub(r"\(", " ( ", text) 
    text = re.sub(r"\)", " ) ", text) 
    text = re.sub(r"\? ", " ? ", text)
    #print("22222222222222222222222",text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\?{2,}", " ? ", text)
    text = re.sub(r"\s{2,}", "", text) #去除多餘空
    text = re.sub(r"\!{2,}", " ! ", text) #去除多餘空
    #print("33333333333333333333333",text)
    return text

def load_model_and_tokenizer(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

def preprocess_csv(csv_path, text_column_name):
    df = pd.read_csv(csv_path, encoding='iso-8859-1')
    texts = df[text_column_name].tolist()
    # 预处理文本
    texts = [preprocess_text(text) for text in texts]
    return texts

def predict_text(model, tokenizer, texts):
    ##對模型進行預處理
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    id_to_label = {i: label for i, label in enumerate(label_list)}

    predictions = torch.argmax(logits, dim=-1)
    predictions = [id_to_label[id] for id in predictions.numpy()]

    return predictions
'''
if __name__ == "__main__":
    model_path = f"D:/project/MBTI_project/full/output/"  # 替换为您的模型路径
    model, tokenizer = load_model_and_tokenizer(model_path)

    csv_file_path = "D:/project/MBTI_project/full/data/dev.csv"  # 替换为您的CSV文件路径
    column_name = "Content"  # 替换为文本所在列的列名
    texts = preprocess_csv(csv_file_path, column_name)
    predictions = predict_text(model, tokenizer, texts)
    print(predictions)
'''
if __name__ == "__main__":
    model_path = f"D:/project/MBTI_project/full/output/"  # 替换为您的模型路径
    model, tokenizer = load_model_and_tokenizer(model_path)
    # 定义一个名为 text 的列表，如果它还不存在
    texts = []
    # 要添加的句子
    sentence = "anyone ever have something happen that you just have to share with someone or have a random thought or idea that you feel like putting out there    or do you simply feel like rambling or ruminating    well   here  s a place to make it known if there is n t a thread for it   go for it here  anything and everything is welcome and encouraged "
    # 将句子添加到列表中
    texts = preprocess_text(texts.append(sentence))
    predictions = predict_text(model, tokenizer, texts)
    print(predictions)
