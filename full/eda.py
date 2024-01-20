import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############## Q:為甚麼資料會蠻一致的 =>需要想回答  ####################
'''
資料呈現方式:
1.全部類型一起的分布圖(content文章的數量)
2.四個類型的各自資料量(content文章的數量)
3.跑出全部資料的content和number
4.##全體字數量
5.##看有沒有kaggle有沒有甚麼圖的

Q1.顏色要另外下載套組，讓他全部分色
Q2.是否要以number => 如果以單詞數量呈現的話，就要統整出各類別最常用的單詞會更好

'''
##################################
#收集到的全部資料
file_path = 'D:/project/MBTI_project/full/data/processed_all_posts_data.csv'
data = pd.read_csv(file_path)

####

def plot_mbti_distribution(data):
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
    plt.ylabel('content', fontsize=16)
    plt.title('MBTI data number', fontsize=24)
    # 顯示圖形
    plt.show()

############## 四個類別的分佈圖 ##############
from sklearn.preprocessing import LabelEncoder
unique_type_list = ["infp","infj","intj","intp","isfp","isfj","istj","istp","enfp","enfj","entj","entp","esfp","esfj","estj","estp"]
lab_encoder = LabelEncoder().fit(unique_type_list)
#print(lab_encoder.transform(['INFJ']))

def plot_mbti_dimensions(data):
    #创建了四个新的特征列 ie、ns、ft 和 pj，这些特征代表了性格类型的不同维度（I/E、N/S、F/T、P/J）
    data['ie'] = data['type'].apply(lambda t: 'i' if 'i' in t else 'e')
    data['ns'] = data['type'].apply(lambda t: 'n' if 'n' in t else 's')
    data['ft'] = data['type'].apply(lambda t: 'f' if 'f' in t else 't')
    data['pj'] = data['type'].apply(lambda t: 'p' if 'p' in t else 'j')

    # 打印每个维度的值计数
    print(data['ie'].value_counts(), end='\n\n')
    print(data['ns'].value_counts(), end='\n\n')
    print(data['ft'].value_counts(), end='\n\n')
    print(data['pj'].value_counts(), end='\n\n')

    # 绘制直方图
    plt.figure()
    data['ie'].hist(); plt.show()
    data['ns'].hist(); plt.show()
    data['ft'].hist(); plt.show()
    data['pj'].hist(); plt.show()

    ###########說不定可以把照片全部一起出來
    ##看看要不要以字數的方式計算
    #是以content文本的方式計算
def mix(data):
    data['ie'] = data['type'].apply(lambda t: 'i' if 'i' in t else 'e')
    data['ns'] = data['type'].apply(lambda t: 'n' if 'n' in t else 's')
    data['ft'] = data['type'].apply(lambda t: 'f' if 'f' in t else 't')
    data['pj'] = data['type'].apply(lambda t: 'p' if 'p' in t else 'j')

    ie_count = data['ie'].value_counts()
    ns_count = data['ns'].value_counts()
    ft_count = data['ft'].value_counts()
    pj_count = data['pj'].value_counts()

    # 创建四个新的特征列 ie、ns、ft 和 pj，这些特征代表了性格类型的不同维度（I/E、N/S、F/T、P/J）
    categories = ['i', 'e', 'n', 's', 'f', 't', 'p', 'j']
    counts = [ie_count['i'], ie_count['e'], ns_count['n'], ns_count['s'], ft_count['f'], ft_count['t'], pj_count['p'], pj_count['j']]
    colors =['skyblue', 'skyblue', 'lightgreen', 'lightgreen', 'lavender', 'lavender', 'lightyellow', 'lightyellow']
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    # 创建条形图
    plt.bar(categories, counts,color=colors,width=0.6)
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Counts by Categories')
    plt.show()


import csv

#跑出全部資料的content和number的數量
def getstat(file_path):
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) == 2:  # 假设您的 CSV 文件有两列
                lines.append(line[1])

    print("Number of lines: {}".format(len(lines)))

    wordcnt = 0
    for line in lines:
        words = line.split()  # 正确地分词每行文本
        wordcnt += len(words)

    print("Number of words: {}".format(wordcnt))

################### 執行def ##################
getstat(file_path)
    
# 绘制所有类型的分布图
plot_mbti_distribution(data)

# 绘制四个维度的分布图
#plot_mbti_dimensions(data)

mix(data)