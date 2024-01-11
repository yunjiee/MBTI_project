
#用于存储处理后的文本数据 
#InputFeatures 是對象的列表，每个对象都包含了一个训练/评估样本的轉換後的數據。
#这些数据可以直接用于 BERT 模型的训练或评估

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids #把前後句子，分為0或是1來判斷
        print('input_ids1          ',input_ids)
        self.input_mask = input_mask #标记序列中哪些位置是真实Token，哪些是填充的
        print('input_mask1        ',input_mask)
        self.segment_ids = segment_ids #区分两个序列的段ID（在处理两个序列时使用）
        print('segment_ids1            ',segment_ids)
        self.label_id = label_id #样本的标签ID
        print('label_id1            ',label_id)
        

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):

        tokens = tokenizer.tokenize(example.text)
        print('example.text          ',example.text)
        print('最原始的tokens             ',tokens)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs: 序列對
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences: 單個序列
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # [CLS]（用于分类任务）和 [SEP]（序列分隔符）分隔
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        #当处理两个序列（例如，两个句子）时，它们被连接在一起
        #[SEP] 用于明确地分隔两个序列或表示单个序列的结束。
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        print('加上後的tokens             ',tokens)

        segment_ids = [0] * len(tokens)
        #input_ids : 代表 tokens 的数字 ID 序列
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        #input_mask：标记哪些是真实 token，哪些是填充
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        #segment_ids：标识序列中不同部分的段落 ID
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #label_id：序列的标签 ID，用于训练或评估
        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
        
        print('input_ids          ',input_ids)
        print('input_mask        ',input_mask)
        print('segment_ids          ',segment_ids)
        print('label_id            ',label_id)


    return features


'''
from processor import PersonalityProcessor
data_dir = "./MBTI_project/full/data"
processor = PersonalityProcessor("YOUR_MODE")  # 替换为您的模式
train_examples = processor.get_train_examples(data_dir)

train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
'''