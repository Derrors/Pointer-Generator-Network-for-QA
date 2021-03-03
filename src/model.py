#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 模型结构
@Author       : Qinghe Li
@Create time  : 2021-02-23 15:08:26
@Last update  : 2021-03-03 15:34:35
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

    def __init__(self, embeddings=None):
        super(Encoder, self).__init__()
        # TODO: 使用训练好的 GLOVE 词向量？
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)

        # 编码得到每个单词的隐层表示、最后一个step的h和c
        output, state = self.lstm(packed)           # hidden : ((2, b, h), (2, b, h))

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # (b, l, 2h)
        encoder_outputs = encoder_outputs.contiguous()                      # (b, l, 2h)

        return encoder_outputs, state


class ReduceState(nn.Module):
    """将编码器最后一个step的隐层状态与观点表示进行拼接、降维以适应解码器"""

    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 6 + 5, config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim * 6 + 5, config.hidden_dim)

    def forward(self, hidden, opinion):
        h, c = hidden       # ((2, b, h), (2, b, h))

        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)   # (b, 2h)
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)   # (b, 2h)

        hidden_reduced_h = F.relu(self.reduce_h(torch.cat([h_in, opinion], dim=-1)))    # (b, h)
        hidden_reduced_c = F.relu(self.reduce_c(torch.cat([c_in, opinion], dim=-1)))    # (b, h)

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))           # (1, b, h)


class CoAttention(nn.Module):
    def __init__(self):
        super(CoAttention, self).__init__()

        self.U = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    def forward(self, H_q, H_rs, q_mask, r_masks):
        b, l_q, _ = H_q.size()
        _, l_r, _ = H_rs.size()

        H_qs = H_q.unsqueeze(1).expand(
            b, config.review_num, l_q, config.hidden_dim * 2).reshape(b * config.review_num, l_q, -1)
        q_masks = q_mask.unsqueeze(1).expand(b, config.review_num, l_q)

        M = torch.tanh(torch.bmm(self.U(H_qs), H_rs.transpose(1, 2)))

        M_q, _ = torch.max(M, dim=2)                        # (b * k, l_q)
        M_r, _ = torch.max(M, dim=1)                        # (b * k, l_r)
        alpha_q = F.softmax(M_q, dim=-1).view(b, config.review_num, -1)     # (b, k, l_q)
        alpha_r = F.softmax(M_r, dim=-1).view(b, config.review_num, -1)     # (b, k, l_r)

        pai_q = torch.mean((alpha_q * q_masks).unsqueeze(3) * H_qs.view(b,
                                                                        config.review_num, l_q, -1), dim=1)      # (b, l_q, 2h)
        _pai_r = (alpha_r * r_masks).unsqueeze(3) * H_rs.view(b, config.review_num,
                                                              l_r, -1)                        # (b, k, l_r, 2h)
        pai_r = _pai_r.reshape(b, -1, config.hidden_dim * 2)        # (b, k * l_r, 2h)

        m_q = alpha_q.bmm(H_q)                              # (b, k, 2h)
        m_r = alpha_r.unsqueeze(2).matmul(H_rs.view(b, config.review_num, l_r, -1)).squeeze(2)     # (b, k, 2h)
        m = torch.cat([m_q, m_r], dim=-1)                   # (b, k, 4h)

        return pai_q, pai_r, m


class OpinionClassifier(nn.Module):
    def __init__(self):
        super(OpinionClassifier, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dim * 4 + 5, config.hidden_dim * 4 + 5, bias=False),
            nn.Tanh(),
            nn.Linear(config.hidden_dim * 4 + 5, 1, bias=False))

        self.classifier = nn.Linear(config.hidden_dim * 4 + 5, 3)

    def forward(self, m, ratings):
        ratings = F.one_hot(ratings, num_classes=5).float()     # (b, k, 5)
        _m = torch.cat([m, ratings], dim=-1)                    # (b, k, 4h + 5)

        beta = F.softmax(self.attention(_m), dim=-1)            # (b, k, 1)
        opinion = _m.transpose(1, 2).bmm(beta).squeeze(2)       # (b, 4h + 5)
        p_o = F.softmax(self.classifier(opinion), dim=1)

        return _m, beta, p_o, opinion


class OpinionFusion(nn.Module):
    def __init__(self):
        super(OpinionFusion, self).__init__()

        if config.opinion_fusion_mode == "dynamic":
            self.W_o = nn.Linear(config.hidden_dim * 4 + 5, config.hidden_dim * 2, bias=False)
            self.W_os = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
            self.vo = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, alpha_r_t, rev_padding_mask, beta, m, s_t_hat):
        batch, k, _ = m.size()
        if config.opinion_fusion_mode == "dynamic":
            o = beta * m                                                        # (b, k, 4h +5)
            s_t_hat = s_t_hat.unsqueeze(1).expand(
                batch, k, config.hidden_dim * 2).contiguous()                   # (b, k, 2h)
            opinion_feature = torch.tanh(self.W_o(o) + self.W_os(s_t_hat))
            beta = F.softmax(self.vo(opinion_feature), dim=1)                   # (b, k, 1)

        _alpha_r_t = beta * alpha_r_t.view(batch, k, -1) * rev_padding_mask     # (b, k, l_r)
        _alpha_r_t = _alpha_r_t.reshape(batch, -1)                              # (b, k * l_r)
        normalization_factor = _alpha_r_t.sum(1, keepdim=True)
        alpha_r_t_hat = (_alpha_r_t / normalization_factor)                     # (b, k * l_r)

        return alpha_r_t_hat


class QuestionAttention(nn.Module):
    """计算当前解码词与问题序列的Attention"""

    def __init__(self):
        super(QuestionAttention, self).__init__()

        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)

        self.W_q = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.W_qs = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v_q = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, pai_q, que_padding_mask, que_coverage):
        """
        Args:
            s_t_hat: (b, 2h)                当前解码器的状态
            pai_q: (b, l_q, 2h)             co-attention后的问题表示
            que_padding_mask: (b, l_q)      问题序列的mask
            que_coverage: (b, l_q)          当前记录的que_coverage
        Return:
            c_q_t: (b, 2h)                  当前step的问题的上下文
            alpha_q_t: (b, l_q)             当前解码词与问题序列的注意力权重
            que_coverage: (b, l_q)          更新后的que_coverage
        """
        batch, seq_len, hidden_size = pai_q.size()                      # (b, l_q, 2h)

        encoder_feature = self.W_q(pai_q)                               # (b, l_q, 2h)

        dec_fea = self.W_qs(s_t_hat)                                    # (b, 2h)
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(
            batch, seq_len, hidden_size).contiguous()                   # (b, l_q, 2h)
        att_features = encoder_feature + dec_fea_expanded               # (b, l_q, 2h)

        if config.is_coverage:
            coverage_input = que_coverage.view(batch, -1, 1)            # (b, l_q, 1)
            coverage_feature = self.W_c(coverage_input)                 # (b, l_q, 2h)
            att_features = att_features + coverage_feature

        e_q_t = torch.tanh(att_features)                                # (b, l_q, 2h)
        alpha_q_t = self.v_q(e_q_t).view(-1, seq_len)                   # (b, l_q)

        alpha_q_t_ = F.softmax(alpha_q_t, dim=1) * que_padding_mask     # (b, l_q)
        normalization_factor = alpha_q_t_.sum(1, keepdim=True)
        alpha_q_t = (alpha_q_t_ / normalization_factor).unsqueeze(1)    # (b, 1, l_q)

        c_q_t = torch.bmm(alpha_q_t, pai_q).squeeze(1)                  # (b, 2h)

        alpha_q_t = alpha_q_t.squeeze(1)                                # (b, l_q)

        if config.is_coverage:
            que_coverage = que_coverage.view(-1, seq_len)
            que_coverage = que_coverage + alpha_q_t

        return c_q_t, alpha_q_t, que_coverage


class ReviewAttention(nn.Module):
    """计算当前解码词与评论序列的Attention"""

    def __init__(self):
        super(ReviewAttention, self).__init__()

        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)

        self.W_r = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.W_rs = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v_r = nn.Linear(config.hidden_dim * 2, 1, bias=False)

        self.opinion_fusion = OpinionFusion()

    def forward(self, s_t_hat, pai_r, rev_padding_mask, rev_coverage, beta, m):
        """
        Args:
            s_t_hat: (b, 2h)                    当前解码器的状态
            pai_r: (b, k * l_r, 2h)             co-attention后的评论表示
            rev_padding_mask: (b, k, l_r)       评论序列的mask
            rev_coverage: (b, k * l_r)          当前记录的rev_coverage
            beta: (b, k, 1)                     评论级别的attention权重
            m: (b, k, 4h + 5)                   观点表示
        Return:
            c_r_t: (b, 2h)                      当前step的评论的上下文
            alpha_r_t_hat: (b, k * l_r)         融合了观点意见的评论注意力权重
            rev_coverage: (b, k * l_r)          更新后的rev_coverage
        """
        batch, seq_len, hidden_size = pai_r.size()                      # (b, k * l_r, 2h)

        encoder_feature = self.W_r(pai_r)                               # (b, k * l_r, 2h)

        dec_fea = self.W_rs(s_t_hat)                                    # (b, 2h)
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(
            batch, seq_len, hidden_size).contiguous()                   # (b, k * l_r, 2h)
        att_features = encoder_feature + dec_fea_expanded               # (b, k * l_r, 2h)

        if config.is_coverage:
            coverage_input = rev_coverage.view(batch, seq_len, 1)       # (b, k * l_r, 1)
            coverage_feature = self.W_c(coverage_input)                 # (b, k * l, 2h)
            att_features = att_features + coverage_feature

        e_r_t = torch.tanh(att_features)                                # (b, k * l, 2h)
        alpha_r_t = self.v_r(e_r_t).view(batch, seq_len)                # (b, k * l_r)

        alpha_r_t_ = F.softmax(alpha_r_t, dim=1) * rev_padding_mask.view(batch, -1)         # (b, k * l_r)
        normalization_factor = alpha_r_t_.sum(1, keepdim=True)
        alpha_r_t = (alpha_r_t_ / normalization_factor).unsqueeze(1)    # (b, 1, k * l_r)

        c_r_t = torch.bmm(alpha_r_t, pai_r).squeeze(1)                  # (b, 2h)

        alpha_r_t = alpha_r_t.squeeze(1)                                # (b, k * l_r)
        alpha_r_t_hat = self.opinion_fusion(alpha_r_t, rev_padding_mask, beta, m, s_t_hat)  # (b, k * l_r)

        if config.is_coverage:
            rev_coverage = rev_coverage.view(batch, seq_len)
            rev_coverage = rev_coverage + alpha_r_t_hat

        return c_r_t, alpha_r_t_hat, rev_coverage


class Decoder(nn.Module):
    """解码器"""

    def __init__(self, embeddings=None):
        super(Decoder, self).__init__()

        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)

        self.x_context = nn.Linear(config.emb_dim + config.hidden_dim * 4, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.que_attention = QuestionAttention()
        self.rev_attention = ReviewAttention()

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 6, 3)

        self.pred_vocab = nn.Sequential(
            nn.Linear(config.hidden_dim * 6, config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.vocab_size))

    def forward(self, y_t, s_t_0, c_q_t_0, c_r_t_0, pai_q, que_padding_mask, que_batch_extend_vocab,
                pai_r, rev_padding_mask, rev_batch_extend_vocab, extra_zeros, que_coverage, rev_coverage, beta, _m, step):
        """
        Args:
            y_t: (b, )                              当前decoder的输入单词
            s_t_0: ((1, b, h), (1, b, h))           解码器的前一个step的状态
            c_q_t_0: (b, 2h)                        解码器的前一个step问题的上下文
            c_r_t_0: (b, 2h)                        解码器的前一个step评论的上下文
            H_q: (b, l_q, 2h)                       编码器的输出
            que_padding_mask: (b, l_q)              编码序列的mask
            que_batch_extend_vocab: (b, l_q)        包含OOV词的问题序列
            H_rs: (b * k, l_r, 2h)
            rev_padding_mask: (b, k, l_r)           编码序列的mask
            rev_batch_extend_vocab: (b, k, l_r)     包含OOV词的问题序列
            extra_zeros:                            OOV词预留，用于拓展词表
            que_coverage
            rev_coverage:                           当前记录的coverage
            beta: (b, k, 1)
            _m: (b, k, 4h + 5)
            step:                                   当前步
        Return:
            final_dist:                             当前解码得到的词的分布
            s_t:                                    当前解码器的状态
            c_t:                                    当前解码器的上下文
            attn_dist:                              注意力权重分布
            p_gen:                                  拷贝概率
            que_coverage:                           累计coverage
            rev_coverage
        """
        b, k, l_r = rev_padding_mask.size()
        # 解码器初始状态
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_0
            s_t_hat = torch.cat([h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)], dim=1)     # (b, 2h)
            c_q_t, _, que_coverage_next = self.que_attention(s_t_hat, pai_q,
                                                             que_padding_mask, que_coverage)
            c_r_t, _, rev_coverage_next = self.rev_attention(s_t_hat, pai_r, rev_padding_mask,
                                                             rev_coverage, beta, _m)
            que_coverage = que_coverage_next
            rev_coverage = rev_coverage_next

        y_t_embd = self.embedding(y_t)
        x = self.x_context(torch.cat([y_t_embd, c_q_t_0, c_r_t_0], dim=1))
        _, s_t = self.lstm(x.unsqueeze(1), s_t_0)        # 解码器在当前step的解码输出

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat([h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)], 1)         # (b, 2h)

        c_q_t, alpha_q_t, q_coverage_next = self.que_attention(s_t_hat, pai_q,
                                                               que_padding_mask, que_coverage)
        c_r_t, alpha_r_t, r_coverage_next = self.rev_attention(s_t_hat, pai_r, rev_padding_mask,
                                                               rev_coverage, beta, _m)

        if self.training or step > 0:
            que_coverage = q_coverage_next                              # (b, l_q)
            rev_coverage = r_coverage_next                              # (b, k * l_r)

        h_s_t = torch.cat([s_t_hat, c_q_t, c_r_t], dim=-1)              # (b, 6h)

        p_gen = None
        if config.pointer_gen:
            p_gen = F.softmax(self.p_gen_linear(h_s_t), dim=1)          # (b, 3)

        p_v = F.softmax(self.pred_vocab(h_s_t), dim=1)                  # (b, v)

        if config.pointer_gen:
            vocab_dist_ = p_gen[:, 0].view(-1, 1) * p_v
            alpha_q_t_ = p_gen[:, 1].view(-1, 1) * alpha_q_t            # (b, l_q)
            alpha_r_t_ = p_gen[:, 2].view(-1, 1) * alpha_r_t            # (b, k * l_r)

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            rev_batch_extend_vocab = rev_batch_extend_vocab.view(b, k * l_r)
            final_dist = vocab_dist_.scatter_add(1, que_batch_extend_vocab, alpha_q_t_)
            final_dist = vocab_dist_.scatter_add(1, rev_batch_extend_vocab, alpha_r_t_)
        else:
            final_dist = p_v

        return final_dist, s_t, c_q_t, c_r_t, alpha_q_t, alpha_r_t, que_coverage, rev_coverage


class Model(nn.Module):
    def __init__(self, model_file_path=None, embeddings=None, is_eval=False):
        super(Model, self).__init__()
        encoder = Encoder(embeddings)
        decoder = Decoder(embeddings)
        reduce_state = ReduceState()
        co_attention = CoAttention()
        opinion_classifier = OpinionClassifier()

        # 参数初始化
        for model in [encoder, decoder, reduce_state, co_attention, opinion_classifier]:
            init_network(model)

        # decoder与encoder参数共享
        decoder.embedding.weight = encoder.embedding.weight

        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()
            co_attention = co_attention.eval()
            opinion_classifier = opinion_classifier.eval()

        self.encoder = encoder.to(config.DEVICE)
        self.decoder = decoder.to(config.DEVICE)
        self.reduce_state = reduce_state.to(config.DEVICE)
        self.co_attention = co_attention.to(config.DEVICE)
        self.opinion_classifier = opinion_classifier.to(config.DEVICE)

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state["encoder_state_dict"])
            self.decoder.load_state_dict(state["decoder_state_dict"], strict=False)
            self.reduce_state.load_state_dict(state["reduce_state_dict"])
            self.co_attention.load_state_dict(state["co_attention_state_dict"])
            self.opinion_classifier.load_state_dict(state["opinion_classifier_state_dict"])
