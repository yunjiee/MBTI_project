import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
#import seaborn as sns

############################## 讀取Kaggle資料   ##############################
csv_file_path = "d:/project/MBTI_project/data/archive/mbti_1.csv"
data = pd.read_csv(csv_file_path)
###print(data.head(5))

data_list = []
for row in data["posts"]:
    parts = row.split("|||")
    data_list.append(parts)
#print(data_list[7]) #包含了分割後的结果列表


############################## 文本清理 ##############################
'''注意:
1.資料有:非英文字、格行 的問題
2.把和類型有關字 都刪除掉
3.把無法識別字，以空格取代
4.留住標點符號 =>為了他更好判斷
5.文句限定為50個英文單字,含標點符號 => 其餘的要截長補短
6.把csv加上type(共兩格)，然後合併成一個檔案，合併檔(類型、文章、隨機放入)
7.其餘:清除網址,去除多餘的空格,去除停詞,构建正则表达式，使用 | 分隔不要的词语
8.
'''
print()
import time
import nltk

##### Compute list of subject with Type | list of comments 

from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import download
#nltk.download('stopwords')


# Lemmatizer | Stemmatizer
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

# Cache the stop words for speed 
cachedStopWords = stopwords.words("english")

# 创建一个空列表，用于存储预处理后的帖子
preprocessed_posts = []

# 初始化停用词和词干提取器
cachedStopWords = stopwords.words("english")
stemmer = PorterStemmer()

# 遍历数据中的每一行
for index, row in data.iterrows():
    # 获取当前行的 'posts' 列的值
    Post = row['posts']
    # List all urls
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', Post)
    # Remove urls #清除網址
    temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', Post)
    # Keep only words #把非字母的換成空格
    temp = re.sub("[^a-zA-Z]", " ", temp)
    
    not_words = ["infp","infj","intj","intp","isfp","isfj","istj","istp","enfp","enfj","entj","entp","esfp","esfj","estj","estp","<type>"]
    # 构建正则表达式，使用 | 分隔不要的词语
    not_words_regex = '|'.join(map(re.escape, not_words))
    # 使用正则表达式进行替换
    temp = re.sub(not_words_regex, '', temp)
    # Remove spaces > 1 #去除多餘的空格
    temp = re.sub(' +', ' ', temp).lower()
    # Remove stopwords and lematize #去除停詞
    stemmed_words = [stemmer.stem(word) for word in temp.split(' ') if word not in cachedStopWords]
    result = " ".join(stemmed_words)
    #print("\nBefore preprocessing:\n\n", OnePost[0:500])
    #print("\nAfter preprocessing:\n\n", temp[0:500])
    #print("\nList of urls:")


############################## 1.先看資料的分布情形 ##############################

#設圖形大小
plt.figure(figsize=(40, 20))
# 设置 x 軸
plt.xticks(fontsize=16, rotation=0)
# 设置 y 軸
plt.yticks(fontsize=16, rotation=0)
# 使用 Matplotlib 的 bar 
type_counts = data['type'].value_counts()#每個類型不同的數量
#colors = [plt.cm.viridis(np.random.random()) for _ in range(len(type_counts))] #每個類型隨機對一個顏色
unique_colors = set()
colors = []
for _ in range(len(type_counts)):
    while True:
        random_color = plt.cm.viridis(np.random.random())
        if random_color != unique_colors:
            unique_colors.add(random_color)
            colors.append(random_color)
            break
plt.bar(type_counts.index, type_counts, color=colors)
# 设置 x 轴標籤
plt.xlabel('type',fontsize=16)
plt.ylabel('number', fontsize=16)

# 顯示圖形
###plt.show()

############################## 把文本標籤數字化 ##############################
from sklearn.preprocessing import LabelEncoder
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
lab_encoder = LabelEncoder().fit(unique_type_list)
#print(lab_encoder.transform(['INFJ']))

#创建了四个新的特征列 ie、ns、ft 和 pj，这些特征代表了性格类型的不同维度（I/E、N/S、F/T、P/J）
data['ie'] = data.type
data['ns'] = data.type
data['ft'] = data.type
data['pj'] = data.type

for i, t in enumerate(data.type):
    if 'I' in t:
        data.ie[i] = 'I'
    elif 'E' in t:
        data.ie[i] = 'E'
        
    if 'N' in t:
        data.ns[i] = 'N'
    elif 'S' in t:
        data.ns[i] = 'S'
        
    if 'F' in t:
        data.ft[i] = 'F'
    elif 'T' in t:
        data.ft[i] = 'T'
        
    if 'P' in t:
        data.pj[i] = 'P'
    elif 'J' in t:
        data.pj[i] = 'J'

posts = data.posts.values
yIE = data.ie.values
yNS = data.ns.values
yFT = data.ft.values
yPJ = data.pj.values
y = data.type

print(data.ie.value_counts(), end='\n\n')
print(data.ns.value_counts(), end='\n\n')
print(data.ft.value_counts(), end='\n\n')
print(data.pj.value_counts(), end='\n\n')

plt.figure()
data.ie.hist(); plt.show()
data.ns.hist(); plt.show()
data.ft.hist(); plt.show()
data.pj.hist(); plt.show()

##############################正則表達式 ##############################





'''
#https://www.kaggle.com/code/rantan/multiclass-and-multi-output-classification#Text-Analysis-with-(MBTI)-Myers-Briggs-Personality-Type-Dataset
# 计算具有类型的主题列表 | 评论列表
from nltk.stem import PorterStemmer, WordNetLemmatizer

# 词形还原
stemmer = PorterStemmer()  # 词干提取器
lemmatiser = WordNetLemmatizer()  # 词形还原器
def pre_process_data(data, remove_stop_words=True):

    list_personality = []
    list_posts = []
    len_data = len(data)
    i=0
    
    for row in data.iterrows():
        i+=1
        if i % 500 == 0:
            print("%s | %s rows" % (i, len_data))

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        type_labelized = lab_encoder.transform([row[1].type])[0]
        list_personality.append(type_labelized)
        list_posts.append(temp)

    #del data
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality

list_posts, list_personality = pre_process_data(data, remove_stop_words=True)           


'''
