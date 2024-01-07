import os
import csv
from sklearn.model_selection import train_test_split

#条件包括：行必须有两个元素，第二个元素（假设为文本）的长度大于 50 字符，并且该行尚未出现在 lines 列表中。
def combine(directory):
    lines = []
    with open(directory, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")# 使用 DictReader 读取 CSV 文件
        for line in reader:
            # 检查每行是否有 "type" 和 "processed_content" 字段
            if "type" in line and "processed_content" in line:
                # 检查 "processed_content" 的长度是否大于 50
                if len(line["processed_content"]) > 50:
                    # 将满足条件的行添加到列表中
                    lines.append([line["type"], line["processed_content"]])
                #if len(lines) >= 5000: break
    return lines

#将lines列表分割成训练集(train)和开发集(dev)，開發集大小为 15%
def create_combined_dataset(lines):
	train, dev = train_test_split(lines, test_size = 0.15)
	return train, dev

#把得到的 train和dev 列表的内容，分別寫入 train和dev .csv文件。
def dataset_exporter(train, dev, directory):
    train_file = os.path.join(directory, 'train.csv')
    dev_file = os.path.join(directory, 'dev.csv')

    with open(train_file, 'w', newline='', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(train)

    with open(dev_file, 'w', newline='', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(dev)

#lines = combine("./dataset")
path = "C:/Users/student/yunjiee-python/MBTI_project/full/data/processed_all_posts_data.csv"
directory = "C:/Users/student/yunjiee-python/MBTI_project/full/data"  # 保存文件的目录

lines = combine(path)
train, dev = create_combined_dataset(lines)
dataset_exporter(train, dev,directory)