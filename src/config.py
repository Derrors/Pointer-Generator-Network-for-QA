#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 模型训练参数
@Author       : Qinghe Li
@Create time  : 2021-02-22 17:07:11
@Last update  : 2021-03-03 15:51:31
"""

import torch
import random

# CPU/显卡
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机数种子
SEED = 2021
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 路径
model_file_path = None
train_data_path = "../data/cell/chunked/train_*"
decode_data_path = "../data/cell/chunked/test_*"
eval_data_path = "../data/cell/chunked/test_*"
vocab_path = "../../glove/glove.42B.300d.txt"
log_root = "../log"

# 模型超参数
hidden_dim = 256
emb_dim = 128
batch_size = 100
beam_size = 4
max_enc_steps = 30
max_dec_steps = 40
min_dec_steps = 5
vocab_size = 50000

pointer_gen = True
is_coverage = False

lr = 0.001
cov_loss_wt = 1.0
eps = 1e-12
max_grad_norm = 2.0
max_epochs = 200
