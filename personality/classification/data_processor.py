import os
import csv
import sys
import re
#from cleaning import clean
from clean import preprocess_text
#表示用於單個訓練或測試的例子
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid #self:看guid的屬性和方法#guid有點類似於地標
        self.text = text
        self.label = label

# 定義DataProcessor類，作為序列分類數據集的數據轉換器的基類
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    # 用於獲取訓練集的InputExample的集合，需要在子類中實現
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        #獲取訓練集的`InputExample`集合
        raise NotImplementedError()
#當代碼執行到 raise 語句時，指定的例外會被觸發，並且通常會中斷當前執行的函數或方法
#NotImplementedError通常用於表示某個方法或功能還沒有被實現

    # 用於獲取開發（驗證）集的InputExample的集合，需要在子類中實現
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    # 用於獲取該數據集的標籤列表，需要在子類中實現
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                if len(line) == 2:
                    lines.append(line)
            return lines

class PersonalityProcessor(DataProcessor):
    #類定義或構造函數
    def __init__(self, mode):
        self.mode = mode #该参数可能用于指定处理数据的模式（如对 MBTI 类型的不同处理方式）
        self.mode = self.mode.upper()

    #獲取訓練和驗證數據的方式
    #從从指定的目录data_dir中读取训练和开发（验证）数据集。
    def get_train_examples(self, data_dir):
        #create_examples 方法用于将读取的数据转换为一定格式的例子
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "train.csv")), "train")
    def get_dev_examples(self, data_dir):
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    #獲取標籤的方法
    def get_labels(self, data_dir):
        #获取训练数据集中所有不同的标签，并将它们存储在一个列表中
        labels_list = []
        train_examples = self.get_train_examples(data_dir)
        for i in train_examples: 
            if i.label not in labels_list:
                labels_list.append(i.label)
        return labels_list

    #創建例子的方法
    def create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if (i == 0): continue
            id_num = "%s-%s" % (set_type, i)
            text = line[1]
            text = clean(text)
            print(text)
            label = line[0]
            label = re.sub("[^a-zA-Z]", '', label)
            label = label.lower()
            if (len(label) > 4): continue
            
            if (self.mode == "E/I" or self.mode == "I/E"): label = label[0]
            elif (self.mode == "N/S" or self.mode == "S/N"): label = label[1]
            elif (self.mode == "T/F" or self.mode == "F/T"): label = label[2]
            elif (self.mode == "J/P" or self.mode == "P/J"): label = label[3]

            examples.append(InputExample(guid=id_num, text=text, label=label))
        return examples

