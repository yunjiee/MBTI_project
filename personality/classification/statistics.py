import csv
import sys
from cleaning import standard_clean
'''
#目的是读取两个 CSV 文件 =>統計其基本數據資料
def getstat():
    #提取数据
    lines = []
    lines2 = []
    with open('./data_personality/processed_all_posts_data.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if len(line) == 2:
                lines.append(line[1])
    with open("dev.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if len(line) == 2:
                lines2.append(line[1])

    print("Number of training: {}".format(len(lines)))
    print("Number of dev: {}".format(len(lines2)))
    print("Number of total: {}".format(len(lines) + len(lines2)))


    wordcnt = 0
    for i in lines:
        a = standard_clean(i)
        a = a.split()
        wordcnt += len(a)
    for i in lines2:
        a = standard_clean(i)
        a = a.split()
        wordcnt += len(a)

    print("Number of words: {}".format(wordcnt))

    
getstat()'''

import csv
import sys
from cleaning import standard_clean

def getstat():
    lines = []
    with open('./MBTI_project/data_personality/processed_all_posts_data.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) == 2:  # 假设您的 CSV 文件有两列
                lines.append(line[1])

    print("Number of lines: {}".format(len(lines)))

    wordcnt = 0
    for i in lines:
        a = standard_clean(i)
        a = a.split()
        wordcnt += len(a)

    print("Number of words: {}".format(wordcnt))

getstat()
