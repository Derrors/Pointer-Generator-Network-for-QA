#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 在测试集上评估模型的损失水平
@Author       : Qinghe Li
@Create time  : 2021-02-19 11:37:30
@Last update  : 2021-03-03 15:57:24
"""

import os
import time

import tensorflow as tf
import torch

import config
from data import (Vocab, get_batch_data_list, get_input_from_batch,
                  get_output_from_batch)
from model import Model
from utils import calc_running_avg_loss


class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = get_batch_data_list(config.eval_data_path, self.vocab,
                                           batch_size=config.batch_size, mode="eval")
        time.sleep(10)

        model_name = os.path.basename(model_file_path)
        eval_dir = os.path.join(config.log_root, "eval_%s" % (model_name))

        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)

        self.summary_writer = tf.compat.v1.summary.FileWriter(eval_dir)
        self.model = Model(model_file_path, self.vocab.embeddings())

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens, target_batch = \
            get_output_from_batch(batch)

        encoder_outputs, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1,
                                                                                           s_t_1,
                                                                                           c_t_1,
                                                                                           encoder_outputs,
                                                                                           enc_padding_mask,
                                                                                           extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage,
                                                                                           di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens
        loss = torch.mean(batch_avg_loss)

        return loss.item()

    def run_eval(self):
        running_avg_loss, iter_step = 0, 0
        start = time.time()

        for batch in self.batcher:
            loss = self.eval_one_batch(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter_step)

            iter_step += 1

            if iter_step % 100 == 0:
                self.summary_writer.flush()

            if iter_step % 1000 == 0:
                print("steps %d, seconds for %d batch: %.2f , loss: %f" %
                      (iter_step, 1000, time.time() - start, running_avg_loss))
                start = time.time()


if __name__ == "__main__":
    model_path = "../log/train_20210224_212437/model/model_50000_20210225_035719"
    eval_processor = Evaluate(model_path)
    eval_processor.run_eval()
