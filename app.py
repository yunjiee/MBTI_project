from flask import Flask, render_template, request, jsonify
# from transformers import BertForSequenceClassification, BertTokenizer
#import torch
import re

app = Flask(__name__)

# label_list = ["infp", "infj", "intj", "intp", "isfp", "isfj", "istj", "istp", "enfp", "enfj", "entj", "entp", "esfp", "esfj", "estj", "estp"]
# model_path = "output/"
# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained(model_path)

# def preprocess_text(text):
#     text = str(text).lower()  # 确保文本是字符串类型并转换为小写
#     text = re.sub(r'http\S+', ' ', text)  # 移除URLs
#     text = re.sub(r"\\t", "  ", text)
#     # 构建正则表达式，使用 | 分隔不要的词语，并添加括号来捕获匹配项
#     pattern = r'(' + '|'.join(map(re.escape, label_list)) + ')'
#     text = re.sub(r"\s{2,}", " ", text) #去除多餘空
#     text = re.sub(r'\s+', ' ', text) # 合并多个空格为一个空格
#     #print("111111clean11111",text)
#     text = re.sub(r'[^a-zA-Z\s\.\,\!\?\(\)\']', '', text)
#     # 将匹配到的词语替换为 <type>
#     text = re.sub(pattern, 'type', text)
#     text = re.sub(r"\'s", " \'s", text) 
#     text = re.sub(r"\'ve", " \'ve", text) 
#     text = re.sub(r"\'t", " \'t", text) 
#     text = re.sub(r"\'re", " \'re", text) 
#     text = re.sub(r"\'d", " \'d", text) 
#     text = re.sub(r"\'ll", " \'ll", text) 
#     text = re.sub(r", ", " , ", text)
#     #print("11111111111111111111111",text)
#     text = re.sub(r"\. ", " . ", text) 
#     #print("22222222222222222222222",text)
#     text = re.sub(r"'", " ' ", text)  
#     text = re.sub(r"!", " ! ", text) 
#     text = re.sub(r"\(", " ( ", text) 
#     text = re.sub(r"\)", " ) ", text) 
#     text = re.sub(r"\? ", " ? ", text)
#     #print("22222222222222222222222",text)
#     text = re.sub(r"\.{2,}", ".", text)
#     text = re.sub(r"\?{2,}", " ? ", text)
#     text = re.sub(r"\s{2,}", "", text) #去除多餘空
#     text = re.sub(r"\!{2,}", " ! ", text) #去除多餘空
#     #print("33333333333333333333333",text)
#     return text

# def predict_text(model, tokenizer, texts):
#     preprocessed_text = preprocess_text(texts)
#     ##對模型進行預處理
#     inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits

#     # label_list = ["infp", "infj", "intj", "intp", "isfp", "isfj", "istj", "istp", "enfp", "enfj", "entj", "entp", "esfp", "esfj", "estj", "estp"]
#     id_to_label = {i: label for i, label in enumerate(label_list)}
#     predictions = torch.argmax(logits, dim=-1)
#     predictions = [id_to_label[id] for id in predictions.numpy()]

#     return predictions ###結果類型變數

@app.route('/')
def index():
   # 將變數傳遞到模板
    return render_template('mbti.html')

@app.route('/detail')
def detail():
    return render_template('detail.html')

@app.route('/data_support')
def data_support():
    return render_template('data_support.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        article_content = request.form['article_content']
        # prediction = predict_text(article_content)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
