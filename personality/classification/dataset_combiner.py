import os
import csv
import sys
from sklearn.model_selection import train_test_split

#条件包括：行必须有两个元素，第二个元素（假设为文本）的长度大于 50 字符，并且该行尚未出现在 lines 列表中。
def combine(directory):
	lines = []
	for filename in os.listdir(directory):
		input_file = os.path.join(directory, filename)
		with open(input_file, "r", encoding="utf-8") as f:
			reader = csv.reader(f, delimiter="\t")
			cntr = 0
			for line in reader:
				if sys.version_info[0] == 2:
					line = list(unicode(cell, 'utf-8') for cell in line)
				if len(line) == 2 and len(line[1]) > 50 and line not in lines:
					lines.append(line)
					cntr += 1
				#if cntr >= 5000: break
	return lines

#将lines列表分割成训练集(train)和开发集(dev)，開發集大小为 15%
def create_combined_dataset(lines):
	train, dev = train_test_split(lines, test_size = 0.15)
	return train, dev

#把得到的 train和dev 列表的内容，分別寫入 train和dev .csv文件。
def dataset_exporter(train, dev):
	with open('train.csv', 'w', newline='') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(train)
	csvFile.close()
	with open('dev.csv', 'w', newline='') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(dev)
	csvFile.close()

#lines = combine("./dataset")
lines = combine("D:\project\MBTI_project\data_personality\processed_all_posts_data.csv")
train, dev = create_combined_dataset(lines)
dataset_exporter(train, dev)