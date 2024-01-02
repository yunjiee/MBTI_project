'''import re
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import download
from nltk import punkt
import random

# 初始化停用词和词干提取器
#nltk.download('stopwords')
#nltk.download('punkt')
cachedStopWords = stopwords.words("english")
stemmer = PorterStemmer()

# 设置单个帖子的单词限制
MAX_WORDS = 50

# 读取CSV文件
file_name = '.\MBTI_project\data_personality\enfj_posts_data.csv'
data = pd.read_csv(file_name)  # 替换为你的数据文件路径
#print(data.head())
# 从文件名提取MBTI类型

type_from_file = file_name.split('\\')[-1].split('_')[0]

# 预处理函数
def preprocess_text(text):
     # 确保文本是字符串类型
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # 移除URLs
    text = re.sub(r'http\S+', ' ', text)
    text = text.replace("'", " ")
    # 保留字母和标点符号
    text = re.sub("[^a-zA-Z.,!?']", " ", text)
    not_words = ["infp","infj","intj","intp","isfp","isfj","istj","istp","enfp","enfj","entj","entp","esfp","esfj","estj","estp","<type>"]
    # 构建正则表达式，使用 | 分隔不要的词语
    not_words_regex = '|'.join(map(re.escape, not_words))
    # 使用正则表达式进行替换
    text = re.sub(not_words_regex, '', text)
    #print(text)
    # 缩短或填充文本以满足单词限制
    words = word_tokenize(text)
    if len(words) > MAX_WORDS:
        words = words[:MAX_WORDS]
    else:
        words += [' '] * (MAX_WORDS - len(words))
    # 移除停用词和词干提取
    words = [stemmer.stem(word) for word in words if word.lower() not in cachedStopWords]
    # 重新组合为单个字符串
    return ' '.join(words)

# 分割文本并应用预处理
def split_and_preprocess(text):
    words = word_tokenize(text)
    if len(words) <= MAX_WORDS:
        return [preprocess_text(text)]
    return [preprocess_text(' '.join(words[i:i+MAX_WORDS])) for i in range(0, len(words), MAX_WORDS)]

# 应用分割和预处理函数
split_texts = data['Content'].astype(str).apply(split_and_preprocess)

# 将分割和处理后的文本转换为单独的行
processed_data = []
for texts in split_texts:
    for text in texts:
        processed_data.append({'type': type_from_file, 'processed_content': text})

# 将处理后的文本存储到新的DataFrame
processed_df = pd.DataFrame(processed_data)

# 打印处理后的前几行，以检查结果
print(processed_df.head())

# 保存到新的CSV文件
processed_df.to_csv('processed_data.csv', index=False)

# 应用预处理函数到DataFrame的每一行
data['processed_content'] = data['Content'].apply(preprocess_text)

# 打印处理后的前几行，以检查结果
print(data['processed_content'].head())

# 将提取到的type添加到DataFrame中
data['type'] = type_from_file

# 将Type列移动到第一列
data = data[['type', 'processed_content']]

# 保存到新的CSV文件
data.to_csv('processed_data.csv', index=False)
'''

import pandas as pd
import re

def preprocess_text(cell):
    # 转换为字符串类型
    cell = str(cell)
    # 去除跨行符号并转换为小写
    cell = cell.replace('\n', ' ').replace('\r', ' ').lower()
    # 替换非英文字符为一个空格
    cell = re.sub(r'[^a-z .]', ' ', cell)
    print(cell)

    print("--------------------------------")

    # 根据句号进行分割
    sentences = cell.split('.')
    processed_sentences = []

    for sentence in sentences:
        words = sentence.split()
        # 按照单词数量对句子进行处理
        if len(words) > 90:
            # 大于90个单词，分为多个部分
            for i in range(0, len(words), 50):
                processed_sentences.append(' '.join(words[i:i+50]))
        elif len(words) >= 50:
            # 50到90个单词之间，只取前50个单词
            processed_sentences.append(' '.join(words[:50]))
        else:
            # 少于50个单词，填充空格
            processed_sentences.append(' '.join(words) + ' ' * (50 - len(words)))
    print(processed_sentences)    
    return ' '.join(processed_sentences)


# 读取CSV文件
file_name = '.\MBTI_project\data_personality\enfj_posts_data.csv'
data = pd.read_csv(file_name)  # 替换为你的数据文件路径

# 应用预处理函数
data['processed_content'] = data['Content'].apply(preprocess_text)

# 保存处理后的数据
processed_file_path = './processed_enfj_posts_data.csv'
data.to_csv(processed_file_path, index=False)

processed_file_path
