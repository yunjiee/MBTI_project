import os
#pip install pytorch-pretrained-bert
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

#獲取bert模型 
def get_bert_model(args, num_labels):
    #args 參數 =>用來配置模型和指定類別的數量(num_labels)#設置緩存目錄
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))#預訓練的BERT模板
    model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)
    return model
    
#優化器=>主要目的還是在針對預處理bert model進行參數上的修正，希望讓他更吻合
#選取優化器
def get_optimizer(args, model, num_train_optimization_steps):
    #選取模型的參數
    #args 參數 =>用來配置模型和指定類別的數量(num_train_optimization_steps)
    # 如果 t_total 未提供，则设置默认值
    param_optimizer = list(model.named_parameters())
    #不需要權重衰減的參數
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #分組模型參數，應用不同的權重衰減
    optimizer_grouped_parameters = [
        #對於weight參數，weight_decay'設為0.01
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #對於bias參數，weight_decay'設為0.0
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    #創建BertAdam優化器
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    return optimizer
# 调用 get_optimizer 时传递 t_total 参数，或者设置一个合适的值
optimizer = get_optimizer(args, model, num_train_optimization_steps, t_total)
