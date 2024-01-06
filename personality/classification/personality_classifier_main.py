# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############"""BERT finetuning runner.微調"""########################
#理想狀態來說: 應該只要給他下面兩個資料data_dir和output_dir，就可以先跑跑看
#parser.add_argument("--data_dir", default="./", type=str)
#parser.add_argument("--output_dir", default="./output/", type=str)
#=>都測試好後,再來看base-large的部分
from __future__ import absolute_import, division, print_function
#用於确保代码在不同版本的Python中具有一致的行为(维护同时需要在Python 2和Python 3环境下运行的代码非常有用)

import argparse #解析命令行参数 =>运行程序时从命令行指定这些参数
import csv
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
#pip install scipy
from scipy.stats import pearsonr, spearmanr
#pip install scikit-learn
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from compute_metrics import compute_metrics
from data_processor import InputExample, DataProcessor, PersonalityProcessor
from features import convert_examples_to_features
from dataloader import get_train_dataloader, get_eval_dataloader, get_label_ids
from bert_model import get_bert_model, get_optimizer


def main():
    #創建一個解析器，用於處理命令行參數
    parser = argparse.ArgumentParser()
    #添加各種命令行參數
    #调用定义了一个命令行参数的规则，包括如何解析该参数以及该参数的一些元数据
    #参数名称以两个连字符（--）开头，它被视为一个可选参数(是那些在命令行中可以省略的参数。意味着在命令行中使用这些参数时，需要使用其完整的名称)
    parser.add_argument("--data_dir", default="./", type=str)#數據目錄
    ##data_dire資料夾內要有: train.csv 用于训练，dev.csv 或 eval.csv 用于模型评估
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str)#使用的bert模型
    parser.add_argument("--output_dir", default="./output/", type=str)#輸出目錄
    #output_dir資料夾: 训练过程中生成的模型和输出数据将保存在这个目录中

    parser.add_argument("--cache_dir", default="", type=str)#緩存目錄
    parser.add_argument("--max_seq_length", default=128, type=int)#最大序列長度

    parser.add_argument("--do_train", action='store_true')#是否經過訓練
    parser.add_argument("--do_eval", action='store_true')#是否進行評估
    parser.add_argument("--do_lower_case", action='store_true')#是否將文本轉為小寫

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)#評估時的批次大小
    parser.add_argument("--learning_rate", default=5e-5, type=float)#學習率

    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--no_cuda", action='store_true')

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    parser.add_argument('--mode', default="ALL", type=str)
    #用于解析命令行参数(将每个参数与之前通过 add_argument 定义的参数进行匹配，并基于提供的信息对这些参数进行适当的类型转换)
    #如果命令行参数不符合预定义的规则，parse_args()会自动显示错误信息并退出程序
    #这在创建需要用户输入参数的脚本时非常有用。
    args = parser.parse_args()


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = PersonalityProcessor(args.mode)
    label_list = processor.get_labels(args.data_dir)
    print(label_list)
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_examples = processor.get_train_examples(args.data_dir)

    num_train_optimization_steps = None
    if args.do_train:
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = get_bert_model(args, num_labels)
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    optimizer = get_optimizer(args, model, num_train_optimization_steps)

    global_step = 0

    if args.do_train:
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        train_dataloader = get_train_dataloader(args, train_features)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        eval_dataloader = get_eval_dataloader(args, eval_features)
        all_label_ids = get_label_ids(args, eval_features)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        
        result = compute_metrics(preds, all_label_ids.numpy(), label_list)
        loss = tr_loss/global_step if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        print(result)
        
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in result.keys():
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()