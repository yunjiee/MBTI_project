import os
import csv
import sys
import re
from cleaning import clean

#表示用於單個訓練或測試的例子
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid #self:看guid的屬性和方法#guid有點類似於地標
        self.text = text
        self.label = label
# 定義DataProcessor類，作為序列分類數據集的數據轉換器的基類
#class DataProcessor(object): 類似於大綱的設計，如果沒有執行到裡面任何一個def，就會出現NotImplementedError()的錯誤
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    #这个方法会根据特定数据集的格式和结构来解析数据。
    # 用於獲取訓練集的InputExample的集合，需要在子類中實現
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        #獲取訓練集的`InputExample`集合
        raise NotImplementedError()
    #當代碼執行到 raise 語句時，指定的例外會被觸發，並且通常會中斷當前執行的函數或方法
    #NotImplementedError通常用於表示某個方法或功能還沒有被實現
    # 用於獲取開發（驗證）集的InputExample的集合，需要在子類中實現
    #目的是将开发集数据转换为 InputExample 对象，以便用于模型的验证和调优。
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    #    # 用於獲取該數據集的標籤列表，需要在子類中實現

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    #方法对于了解数据集的结构和训练模型时定义输出层非常重要。
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file.制表符分隔"""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                if len(line) == 2:
                    lines.append(line)
            return lines

#子目錄=> 用於取代大綱裡面的內容，來執行個別化的程式
class PersonalityProcessor(DataProcessor):
    def __init__(self, mode):
        self.mode = mode
        self.mode = self.mode.upper()

    def get_train_examples(self, data_dir):
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self, data_dir):
        labels_list = []
        train_examples = self.get_train_examples(data_dir)
        for i in train_examples: 
            if i.label not in labels_list:
                labels_list.append(i.label)
        return labels_list

    def create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if (i == 0): continue
            id_num = "%s-%s" % (set_type, i)
            text = line[1]
            text = clean(text)

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

