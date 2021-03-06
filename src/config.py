#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 模型训练参数
@Author       : Qinghe Li
@Create time  : 2021-02-22 17:07:11
@Last update  : 2021-03-06 15:07:45
"""

import torch
import random

# CPU/显卡
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机数种子
SEED = 2021
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# train 路径
# model_file_path = "../log/train_elec/model/model_elec_0"
model_file_path = None
train_data_path = "../data_oaag/home/chunked/train_*"
vocab_path = "../data_oaag/home/vocab"
glove_emb_path = "../../glove/glove.42B.300d.txt"
log_root = "../log/home"

# decode path
decode_model_path = "../log/train_sport/model/model_sport_1"
decode_data_path = "../data_oaag/sport/chunked/test_*"

# eval Path
eval_model_path = None
eval_data_path = None

# 模型超参数
hidden_dim = 256
emb_dim = 300
batch_size = 100
beam_size = 4
max_que_steps = 20
max_dec_steps = 100
min_dec_steps = 20
vocab_size = 50000

pointer_gen = True
is_coverage = False

lr = 0.001
cov_loss_wt = 1.0
eps = 1e-12
max_grad_norm = 2.0
max_epochs = 20
