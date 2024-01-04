import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# MBTI类型列表
mbti_types = ["infp", "infj", "intj", "intp", "isfp", "isfj", "istj", "istp", "enfp", "enfj", "entj", "entp", "esfp", "esfj", "estj", "estp"]
all_data = pd.DataFrame()
# 预处理函数
def preprocess_text(text):
    text = str(text).lower()  # 确保文本是字符串类型并转换为小写
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\s{2,}", " ", text) #去除多餘空
    text = re.sub(r'http\S+', ' ', text)  # 移除URLs
    text = re.sub(r'[^a-z .]', ' ', text)  # 替换非英文字符为一个空格
    text = re.sub("[^a-zA-Z.,!?']", " ", text)  # 保留字母和标点符号，替换其他字符为一个空格
    # 构建正则表达式，使用 | 分隔不要的词语，并添加括号来捕获匹配项
    pattern = r'(' + '|'.join(map(re.escape, mbti_types)) + ')'
    # 将匹配到的词语替换为 <type>
    text = re.sub(pattern, '<type>', text)
    # 分词并应用词干提取
    words = word_tokenize(text)
    return words

for mbti_type in mbti_types:
    file_name = f'./MBTI_project/data_personality/{mbti_type}_posts_data.csv'
    data = pd.read_csv(file_name)
    data = data.dropna(subset=['Content'])

    data['processed_content'] = data['Content'].astype(str).apply(preprocess_text)
    data['type'] = mbti_type

    data = data[['type', 'processed_content']]
    all_data = pd.concat([all_data, data], ignore_index=True)
    
all_data.to_csv('./MBTI_project/data_personality/processed_all_posts_data.csv', index=False)

print("All processed data saved to: ./MBTI_project/data_personality/processed_all_posts_data.csv")