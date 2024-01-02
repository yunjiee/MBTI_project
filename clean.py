'''
import re
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import download
from nltk import punkt
'''
import random
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# 初始化停用词和词干提取器
cachedStopWords = stopwords.words("english")
stemmer = PorterStemmer()

# 设置单个帖子的单词限制
MAX_WORDS = 50

# 读取CSV文件
file_name = '.\MBTI_project\data_personality\enfj_posts_data.csv'
data = pd.read_csv(file_name)  # 替换为你的数据文件路径
type_from_file = file_name.split('\\')[-1].split('_')[0]  # 从文件名提取MBTI类型

# 预处理函数
def preprocess_text(text):
    # 确保文本是字符串类型
    text = str(text).lower()
    # 移除URLs
    text = re.sub(r'http\S+', ' ', text)
    text = text.replace("'", " ")
    # 保留字母和标点符号
    text = re.sub("[^a-zA-Z.,!?']", " ", text)
    not_words = ["infp", "infj", "intj", "intp", "isfp", "isfj", "istj", "istp", "enfp", "enfj", "entj", "entp", "esfp", "esfj", "estj", "estp", "<type>"]
    # 使用正则表达式进行替换
    text = re.sub('|'.join(map(re.escape, not_words)), '', text)
    words = word_tokenize(text)
    if len(words) > MAX_WORDS:
        words = words[:MAX_WORDS]
    else:
        words += [' '] * (MAX_WORDS - len(words))
    words = [stemmer.stem(word) for word in words if word.lower() not in cachedStopWords]
    return ' '.join(words)

# 应用分割和预处理函数
split_texts = data['Content'].astype(str).apply(preprocess_text)

# 打印处理后的前几行，以检查结果
print(split_texts.head())

# 保存到新的CSV文件
data.to_csv('processed_data.csv', index=False)

'''
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
'''
import pandas as pd
import re
def preprocess_text(cell):
    # 去除跨行符号并转换为小写
    cell = re.sub(r'\s+', ' ', cell).lower()
    # 替换非英文字符为一个空格
    cell = re.sub(r'[^a-z .]', ' ', cell)
    cell   = re.sub("[^a-zA-Z.,!?']", " ", cell )

    # 根据句号进行分割
    sentences = cell.split('.')
    processed_cells = []

    for sentence in sentences:
        words = sentence.split()
        # 分割超过50个单词的句子
        while words:
            processed_cells.append(' '.join(words[:50]))
            words = words[50:]

    return processed_cells

# 读取CSV文件
file_name = '.\MBTI_project\data_personality\enfj_posts_data.csv'
data = pd.read_csv(file_name)

# 确保 'Content' 列是字符串类型
data['Content'] = data['Content'].astype(str)

# 应用预处理函数并扁平化结果
processed_cells = data['Content'].apply(preprocess_text)
flattened_cells = [cell for sublist in processed_cells for cell in sublist]

# 创建新的DataFrame
processed_data = pd.DataFrame({'processed_content': flattened_cells})

# 保存处理后的数据
processed_file_path = '.\MBTI_project\data_personality\processed_enfj_posts_data.csv'
processed_data.to_csv(processed_file_path, index=False)

processed_file_path




#file_name = '.\MBTI_project\data_personality\enfj_posts_data.csv'
'''
