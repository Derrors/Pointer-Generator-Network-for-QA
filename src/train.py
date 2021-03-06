#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 模型训练
@Author       : Qinghe Li
@Create time  : 2021-02-22 17:18:38
@Last update  : 2021-03-06 15:12:47
"""

import os
import time

import tensorflow as tf
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad, Adam

import config
from data import Vocab, get_batch_data_list, get_input_from_batch, get_output_from_batch, get_init_embeddings
from model import Model
from utils import calc_running_avg_loss


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = get_batch_data_list(config.train_data_path,
                                           self.vocab,
                                           batch_size=config.batch_size,
                                           mode="train")
        time.sleep(10)

        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        train_dir = os.path.join(config.log_root, "train_{}".format(stamp))

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, "model")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.compat.v1.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter_step):
        """保存模型"""
        state = {
            "iter": iter_step,
            "encoder_state_dict": self.model.encoder.state_dict(),
            "decoder_state_dict": self.model.decoder.state_dict(),
            "reduce_state_dict": self.model.reduce_state.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_loss
        }
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        model_save_path = os.path.join(self.model_dir, "model_{}_{}".format(iter_step, stamp))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        """模型初始化或加载、初始化迭代次数、损失、优化器"""

        # 初始化模型
        self.model = Model(model_file_path, get_init_embeddings(self.vocab._id_to_word))

        # 定义优化器
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)

        # 初始化迭代次数和损失
        start_iter, start_loss = 0, 0

        # 如果传入的已存在的模型路径，加载模型继续训练
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state["iter"]
            start_loss = state["current_loss"]

            if not config.is_coverage:
                self.optimizer.load_state_dict(state["optimizer"])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(config.DEVICE)

        return start_iter, start_loss

    def train_one_batch(self, batch):
        """训练一个batch，返回该batch的loss"""

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens, target_batch = \
            get_output_from_batch(batch)

        self.optimizer.zero_grad()
        # (b, l, 2h), ((2, b, h), (2, b, h))
        encoder_outputs, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)     # (h,c) = ((1, b, h), (1, b, h))

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]        # 当前step解码器的输入单词
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = \
                self.model.decoder(y_t_1,
                                   s_t_1,
                                   c_t_1,
                                   encoder_outputs,
                                   enc_padding_mask,
                                   extra_zeros,
                                   enc_batch_extend_vocab,
                                   coverage,
                                   di)
            target = target_batch[:, di]    # 当前step解码器的目标词
            # final_dist 是词汇表每个单词的概率，词汇表是扩展之后的词汇表
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()     # 取出目标单词的概率
            step_loss = -torch.log(gold_probs + config.eps)     # 最大化gold_probs，也就是最小化step_loss

            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        avg_losses = torch.sum(torch.stack(step_losses, 1), 1) / dec_lens
        loss = torch.mean(avg_losses)

        loss.backward()

        clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, epochs, model_file_path=None):
        # 训练设置
        iter_step, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        for _ in range(epochs):
            # 获取下一个batch数据
            for batch in self.batcher:
                loss = self.train_one_batch(batch)

                running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter_step)
                iter_step += 1

                if iter_step % 50 == 0:
                    self.summary_writer.flush()

                if iter_step % 200 == 0:
                    print("steps %d, seconds for %d batch: %.2f , loss: %f" % (iter_step, 200, time.time() - start, loss))
                    start = time.time()

                if iter_step % 1000 == 0:
                    self.save_model(running_avg_loss, iter_step)


if __name__ == "__main__":
    train_processor = Train()
    train_processor.trainIters(config.max_epochs, config.model_file_path)
