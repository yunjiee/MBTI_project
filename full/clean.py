import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

'''
Q1.資料有些都是同篇作者的文章
Q2.資料要設置字數小於50字的刪除
Q3.##沒有反映資料原始的樣子=>因為有刪掉一些奇怪的
###刪掉那些咚咚，可以說一下
'''

# MBTI类型列表
mbti_types = ["infp", "infj", "intj", "intp", "isfp", "isfj", "istj", "istp", "enfp", "enfj", "entj", "entp", "esfp", "esfj", "estj", "estp"]
all_data = pd.DataFrame()
# 预处理函数
def preprocess_text(text):
    text = str(text).lower()  # 确保文本是字符串类型并转换为小写
    text = re.sub(r"\\t", "  ", text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text) 
    text = re.sub(r"\'ve", " \'ve", text) 
    text = re.sub(r"n\'t", " n\'t", text) 
    text = re.sub(r"\'re", " \'re", text) 
    text = re.sub(r"\'d", " \'d", text) 
    text = re.sub(r"\'ll", " \'ll", text) 
    text = re.sub(r",", " , ", text) 
    text = re.sub(r"!", " ! ", text) 
    text = re.sub(r"\(", " \( ", text) 
    text = re.sub(r"\)", " \) ", text) 
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\s{2,}", " ", text) #去除多餘空
    text = re.sub(r'http\S+', ' ', text)  # 移除URLs
    text = re.sub(r'[^a-z .]', ' ', text)  # 替换非英文字符为一个空格
    text = re.sub(r'[^a-z0-9 .,!?\'\"]', " ", text)  # 保留字母和标点符号，替换其他字符为一个空格
    # 构建正则表达式，使用 | 分隔不要的词语，并添加括号来捕获匹配项
    pattern = r'(' + '|'.join(map(re.escape, mbti_types)) + ')'
    # 将匹配到的词语替换为 <type>
    text = re.sub(pattern, '<type>', text)
    return text

for mbti_type in mbti_types:
    file_name = f'D:/project/MBTI_project/data_personality/{mbti_type}_posts_data.csv'
    data = pd.read_csv(file_name)
    data = data.dropna(subset=['Content'])
    data = data[data['Content'].str.strip().astype(bool)]  # 然后删除内容仅包含空格的行
    data['word_count'] = data['Content'].apply(lambda x: len(x.split()))
    data = data[data['word_count'] >= 3]


    data['processed_content'] = data['Content'].astype(str).apply(preprocess_text)
    data['type'] = mbti_type

    data = data[['type', 'processed_content']]
    all_data = pd.concat([all_data, data], ignore_index=True)
    
all_data.to_csv('./data_personality/processed_all_posts_data.csv', index=False)

print("All processed data saved to: ./data_personality/processed_all_posts_data.csv")