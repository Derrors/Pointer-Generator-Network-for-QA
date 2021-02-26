#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 模型结构
@Author       : Qinghe Li
@Create time  : 2021-02-23 15:08:26
@Last update  : 2021-02-25 20:19:51
"""

import torch
import config
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# 权重初始化，默认xavier
def init_network(model, method="xavier"):
    for name, w in model.named_parameters():
        if "weight" in name:
            if method == "xavier":
                nn.init.xavier_normal_(w)
            elif method == "kaiming":
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)
        elif "bias" in name:
            nn.init.constant_(w, 0)
        else:
            pass


class Encoder(nn.Module):
    """编码器"""
    def __init__(self):
        super(Encoder, self).__init__()
        # TODO: 使用训练好的 GLOVE 词向量？
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)

        # 编码得到每个单词的隐层表示、最后一个step的h和c
        output, hidden = self.lstm(packed)           # hidden : ((2, b, h), (2, b, h))

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # (b, l, 2h)
        encoder_outputs = encoder_outputs.contiguous()                      # (b, l, 2h)

        return encoder_outputs, hidden


class ReduceState(nn.Module):
    """将编码器最后一个step的隐层状态进行降维以适应解码器"""
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

    def forward(self, hidden):
        h, c = hidden       # ((2, b, h), (2, b, h))

        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)   # (b, 2h)
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)   # (b, 2h)

        hidden_reduced_h = F.relu(self.reduce_h(h_in))                          # (b, h)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))   # (1, b, h)


class Attention(nn.Module):
    """计算当前解码词与编码序列的Attention"""
    def __init__(self):
        super(Attention, self).__init__()

        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.W_s = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, enc_padding_mask, coverage):
        """
        Args:
            s_t_hat:                    当前解码器的状态
            encoder_outputs:            编码器输出的编码序列的各隐层表示
            enc_padding_mask:           编码序列的mask
            coverage:                   当前记录的coverage
        Return:
            c_t:                        加权求和得到的当前解码词的上下文表示
            atten_dis:                  当前解码词与编码序列的注意力权重
            coverage:                   更新后的coverage
        """
        batch, seq_len, hidden_size = list(encoder_outputs.size())          # (b, l, 2h)

        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)   # (b * l, 2h)
        encoder_feature = self.W_h(encoder_feature)                         # (b * l, 2h)

        dec_fea = self.W_s(s_t_hat)                                 # (b, 2h)
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(
            batch, seq_len, hidden_size).contiguous()               # (b, l, 2h)
        dec_fea_expanded = dec_fea_expanded.view(-1, hidden_size)   # (b * l, 2h)
        # W_h * h_i + W_s * s_t + b
        att_features = encoder_feature + dec_fea_expanded           # (b * l, 2h)

        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)                   # (b * l, 1)
            coverage_feature = self.W_c(coverage_input)             # (b * l, 2h)
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)                                # (b * l, 2h)
        scores = self.v(e).view(-1, seq_len)                        # (b, l)

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask    # (b, l)
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor               # (b, l)

        attn_dist = attn_dist.unsqueeze(1)                          # (b, 1, l)
        c_t = torch.bmm(attn_dist, encoder_outputs).squeeze(1)      # (b, 2h)

        attn_dist = attn_dist.view(-1, seq_len)                     # (b, l)

        if config.is_coverage:
            coverage = coverage.view(-1, seq_len)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    """解码器"""
    def __init__(self):
        super(Decoder, self).__init__()

        # TODO: GLOVE
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.attention_network = Attention()

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        self.pred_vocab = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.Linear(config.hidden_dim, config.vocab_size))

    def forward(self, y_t_1, s_t_1, c_t_1, encoder_outputs, enc_padding_mask,
                extra_zeros, enc_batch_extend_vocab, coverage, step):
        """
        Args:
            y_t_1:                      当前decoder的输入单词
            s_t_1:                      解码器的前一个step的状态
            c_t_1:                      解码器的前一个step的上下文
            encoder_outputs:            编码器的输出
            enc_padding_mask:           编码序列的mask
            extra_zeros:                OOV词预留，用于拓展词表
            enc_batch_extend_vocab:     包含OOV词的编码序列
            coverage:                   当前记录的coverage
            step:                       当前步
        Return:
            final_dist:                 当前解码得到的词的分布
            s_t:                        当前解码器的状态
            c_t:                        当前解码器的上下文
            attn_dist:                  注意力权重分布
            p_gen:                      拷贝概率
            coverage:                   累计coverage
        """
        # 解码器初始状态
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)     # (b, 2h)
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)        # 解码器在当前step的解码输出

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)         # (b, 2h)
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)                   # (b, 4h + e_d)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # (b, 3h)
        output = self.pred_vocab(output)                                    # (b, v)
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class Model(nn.Module):
    def __init__(self, model_file_path=None, is_eval=False):
        super(Model, self).__init__()
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # 参数初始化
        for model in [encoder, decoder, reduce_state]:
            init_network(model)

        # decoder与encoder参数共享
        decoder.embedding.weight = encoder.embedding.weight

        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        self.encoder = encoder.to(config.DEVICE)
        self.decoder = decoder.to(config.DEVICE)
        self.reduce_state = reduce_state.to(config.DEVICE)

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state["encoder_state_dict"])
            self.decoder.load_state_dict(state["decoder_state_dict"], strict=False)
            self.reduce_state.load_state_dict(state["reduce_state_dict"])
