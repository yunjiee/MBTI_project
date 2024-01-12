import os
import csv
import re
#from cleaning import clean
from clean import preprocess_text

##### 處理和準備數據以進行序列分類任務 =>轉成適合模型使用 ####

#表示用於單個訓練或測試的例子
class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label=None):
        self.guid = guid #self:看guid的屬性和方法#guid有點類似於地標
        #print("guid     ",guid)
        #guid=>目的用於區分不同樣本
        self.text = text #=>樣本的文本內容
        self.label = label#=>樣本的標籤(樣本是可選的，因為測試時不需要標籤)

#大綱
# 定義DataProcessor類，作為序列分類數據集的數據轉換器的基類
### class DataProcessor(object):類似於大綱的設計，如果沒有執行到裡面任何一個def，就會出現NotImplementedError()的錯誤
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    #这个方法会根据特定数据集的格式和结构来解析数据。
    # 用於獲取訓練集的InputExample的集合，需要在子類中實現
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        ##獲取訓練集的樣本
        raise NotImplementedError()
    #當代碼執行到 raise 語句時，指定的例外會被觸發，並且通常會中斷當前執行的函數或方法
    #NotImplementedError通常用於表示某個方法或功能還沒有被實現
    # 用於獲取開發（驗證）集的InputExample的集合，需要在子類中實現
    #目的是将开发集数据转换为 InputExample 对象，以便用于模型的验证和调优。
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        ##獲取驗證集的樣本
        raise NotImplementedError()
        #用於獲取該數據集的標籤列表，需要在子類中實現

    def get_labels(self):
        """Gets the list of labels for this data set."""
        ##獲取數據的所有標籤
        raise NotImplementedError()
    #方法对于了解数据集的结构和训练模型时定义输出层非常重要。
    @classmethod
    #表示後面的程式是和"類相關"
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file.使用逗號分隔"""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f) #使用逗號分隔
            lines = []
            for line in reader:
                if len(line) == 2:
                    lines.append(line)
            return lines

#子目錄=> 用於取代大綱裡面的內容，來執行個別化的程式
'''
#數據讀取的轉換流程部分，负责从 CSV 文件中读取原始文本数据，并将其转换为一种更结构化的格式（InputExample 对象），
这种格式适合用于后续的机器学习模型训练和验证。
通过这种方式，您可以确保数据以一种标准和一致的方式被处理和呈现给模型。
'''
### 處立準備數據 => 如何读取、解析和转换特定于个性类型数据集的数据。 ###
class PersonalityProcessor(DataProcessor):
    #類定義或構造函數
    def __init__(self, mode):
        self.mode = mode #该参数可能用于指定处理数据的模式（如对 MBTI 类型的不同处理方式）
        self.mode = self.mode.upper()

    #獲取訓練和驗證數據的方式
    ############數據讀取的轉換流程部分##############
    #從从指定的目录data_dir中獲取训练和驗證的數據集。
    def get_train_examples(self, data_dir):
        #create_examples 方法用于将读取的数据转换为InputExample的例子
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "train.csv")), "train")
        #os.path.join(data_dir, "train.csv")生成完整的文件路徑
    def get_dev_examples(self, data_dir):
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")
    #_read_tsv 方法用于读取以制表符分隔的值（TSV）文件，并返回每行的数据

##### 我資料要有一列是專門放標籤數據的在data_dir中

    #獲取訓練數據集中標籤(列表)的方法
    def get_labels(self, data_dir):
        #获取训练数据集中所有不同的标签，并将它们存储在一个列表中
        labels_list = []
        train_examples = self.get_train_examples(data_dir)
        #data_dir这个参数告诉 get_train_examples 方法去哪个文件夹中寻找训练数据。
        ##get_train_examples通過這裡取來訓練數據的標籤
        for i in train_examples: 
            #檢查標籤是否已存在於labels_list列表中
            if i.label not in labels_list:
                labels_list.append(i.label)
                print("標籤在其中")
        return labels_list

    #創建例子的方法
    def create_examples(self, lines, set_type):
        print(f"一共執行幾行 : {len(lines)}")
        examples = [] #設一個空list
        for (i, line) in enumerate(lines):
            print(f"準備行數 {i}: 內容{line}")
            if (i == 0): continue
            id_num = "%s-%s" % (set_type, i)
            #id_num 每个数据样本生成的唯一标识符
            text = line[1]
            text = preprocess_text(text)
            #針對標籤進行確認
            label = line[0]
            label = re.sub("[^a-zA-Z]", '', label)
            label = label.lower()
            print("全部",label)
            if (len(label) > 4): continue
            #從四個維度
            if (self.mode == "E/I" or self.mode == "I/E"): label = label[0],print(label)
            elif (self.mode == "N/S" or self.mode == "S/N"): label = label[1]
            elif (self.mode == "T/F" or self.mode == "F/T"): label = label[2]
            elif (self.mode == "J/P" or self.mode == "P/J"): label = label[3]
            #舉例:用於簡單序列分類任務的數據結構
            examples.append(InputExample(guid=id_num, text=text, label=label))
        return examples
'''
#data_dir = "./MBTI_project/full/data"
data_dir = "/content/drive/My Drive/full/data"
processor = PersonalityProcessor("YOUR_MODE")  # 替换为您的模式
train_examples = processor.get_train_examples(data_dir)
dev_examples = processor.get_dev_examples(data_dir)
print("11111111111111111111111",train_examples)
print("22222222222222222222222",dev_examples)
'''