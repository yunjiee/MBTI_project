import torch
from torch.utils.data import (DistributedSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
### dataloader 數據加載器 =>基于 PyTorch 的模型准备训练和评估数据。它将数据转换为模型所需的格式 ###
#PyTorch 中用于加载数据的一个工具，可以为模型训练提供批量数据
#预处理的数据转换成 PyTorch DataLoader 的形式，这是为了让数据能够被 PyTorch 模型有效地使用

#訓練
def get_train_dataloader(args, train_features):
    #args：包含训练配置的参数 
    #train_features : 訓練術具的特徵值列表
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    library(writexl)
    # 将数据框写入 Excel 文件
    # write_xlsx(train_data_df, "train_data.xlsx")

    #創建數據庫    
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    print(train_data)
    #輸入張量和標籤和數據一起，方便模型訓練
    #採樣數據
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
        #RandomSampler隨機打亂數據樣本然後危機抽取來使用
    else:
        train_sampler = DistributedSampler(train_data)
        #DistributedSampler當在多個GPU上進行分布式訓練時，這採樣器確保使用在每個設備上的數據是非重複的
    #加載數據
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    #DataLoader把數據轉換為可更模型的批量數據
    return train_dataloader

#測試
def get_eval_dataloader(args, eval_features):
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    
    #評估 
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    return eval_dataloader

#把樣本標籤轉為張量
def get_label_ids(args, features):
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return all_label_ids
