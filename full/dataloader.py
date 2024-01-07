import torch
from torch.utils.data import (DistributedSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
### dataloader 數據加載器 =>基于 PyTorch 的模型准备训练和评估数据。它将数据转换为模型所需的格式 ###
#PyTorch 中用于加载数据的一个工具，可以为模型训练提供批量数据
#预处理的数据转换成 PyTorch DataLoader 的形式，这是为了让数据能够被 PyTorch 模型有效地使用

def get_train_dataloader(args, train_features):
    #args：包含训练配置的参数 
    #train_features : 训练数据集的特征列表
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader


def get_eval_dataloader(args, eval_features):
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    return eval_dataloader


def get_label_ids(args, features):
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return all_label_ids
