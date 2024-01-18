from flask import Flask, render_template, request, jsonify
from out import load_model_and_tokenizer, predict_text, preprocess_text
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import re

app = Flask(__name__)

model_path = "output/"
model, tokenizer = load_model_and_tokenizer(model_path)

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
        article_content = request.form.get('article_content')
        print("Article Content:", article_content)
        # 預處理輸入文本
        preprocessed_text = preprocess_text(article_content)
        # 進行預測
        predictions = predict_text(model, tokenizer, [preprocessed_text])
        prediction = predictions[0]
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
