import csv
import sys

### sametest 函数用于检查并报告两个 CSV 文件之间共有多少相同的数据行，
# 这有助于评估数据集的质量和适用性。###

#目的是比较两个 CSV 文件（train.csv 和 dev.csv）中的数据，并计算两者之间有多少相同的行
def sametest():
    lines = []
    lines2 = []
    #读取 train.csv 文件
    with open("train.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if len(line) == 2:
                lines.append(line)
    #读取 dev.csv 文件
    with open("dev.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if len(line) == 2:
                lines2.append(line)
    ##比較兩個文件中的行中有多少重複的部分
    cntr = 0
    for i in lines2: 
        if i in lines:
            cntr += 1
            print(i)
    print(cntr)

sametest()