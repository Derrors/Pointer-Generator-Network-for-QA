#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : decode 阶段，使用 beam search 算法
@Author       : Qinghe Li
@Create time  : 2021-02-23 16:37:33
@Last update  : 2021-03-06 15:03:42
"""

import os
import time
import torch

import config
import data
from data import Vocab, get_batch_data_list, get_input_from_batch, get_init_embeddings
from model import Model
from utils import write_for_eval, eval_decode_result


class Beam(object):
    def __init__(self, tokens, log_probs, state, que_context, rev_context, que_coverage, rev_coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.que_context = que_context
        self.rev_context = rev_context
        self.que_coverage = que_coverage
        self.rev_coverage = rev_coverage

    def extend(self, token, log_prob, state, que_context, rev_context, que_coverage, rev_coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    que_context=que_context,
                    rev_context=rev_context,
                    que_coverage=que_coverage,
                    rev_coverage=rev_coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, "decode_%s" % (model_name))
        self.ref_dir = os.path.join(self._decode_dir, "ref_dir")
        self.dec_dir = os.path.join(self._decode_dir, "dec_dir")
        # 创建3个目录
        for p in [self._decode_dir, self.ref_dir, self.dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        # 读取并分批测试数据
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = get_batch_data_list(config.decode_data_path, self.vocab,
                                           batch_size=config.beam_size, mode="decode")
        time.sleep(15)
        # 加载模型
        self.model = Model(model_file_path, get_init_embeddings(self.vocab._id_to_word), is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch):
        # 每个batch中只有1个样例被复制beam_size次
        que_batch, que_padding_mask, que_lens, que_batch_extend_vocab, rev_batch, rev_padding_mask, rev_lens, rev_batch_extend_vocab, extra_zeros, rating_batch, c_t_0, que_coverage, rev_coverage = \
            get_input_from_batch(batch)

        H_q, q_state = self.model.encoder(que_batch, que_lens)          # (b, l_q, 2h), ((2, b, h), (2, b, h))
        H_rs, r_states = self.model.encoder(
            rev_batch.view(config.beam_size * config.review_num, -1),
            rev_lens.view(config.beam_size * config.review_num, ))     # (b * k, l_r, 2h), ((2, b * k, h), (2, b * k, h))

        pai_q, pai_r, m = self.model.co_attention(H_q, H_rs, que_padding_mask, rev_padding_mask)

        _m, beta, p_o, opinion = self.model.opinion_classifier(m, rating_batch)

        s_t = self.model.reduce_state(q_state, opinion)              # (h, c) = ((1, b, h), (1, b, h))

        dec_h, dec_c = s_t
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      que_context=c_t_0[0],
                      rev_context=c_t_0[0],
                      que_coverage=(que_coverage[0] if config.is_coverage else None),
                      rev_coverage=(rev_coverage[0] if config.is_coverage else None),)
                 for _ in range(config.beam_size)]

        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN)
                             for t in latest_tokens]
            y_t_1 = torch.tensor(latest_tokens, dtype=torch.long, device=config.DEVICE)

            all_state_h = []
            all_state_c = []
            all_que_context = []
            all_rev_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_que_context.append(h.que_context)
                all_rev_context.append(h.rev_context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_q_t_1 = torch.stack(all_que_context, 0)
            c_r_t_1 = torch.stack(all_rev_context, 0)

            que_coverage_t_1 = None
            rev_coverage_t_1 = None
            if config.is_coverage:
                all_que_coverage = []
                all_rev_coverage = []
                for h in beams:
                    all_que_coverage.append(h.que_coverage)
                    all_rev_coverage.append(h.rev_coverage)
                que_coverage_t_1 = torch.stack(all_que_coverage, 0)
                rev_coverage_t_1 = torch.stack(all_rev_coverage, 0)

            final_dist, s_t, c_q_t, c_r_t, alpha_q_t, alpha_r_t, que_coverage_t, rev_coverage_t = \
                self.model.decoder(y_t_1, s_t_1, c_q_t_1, c_r_t_1,
                                   pai_q, que_padding_mask, que_batch_extend_vocab,
                                   pai_r, rev_padding_mask, rev_batch_extend_vocab,
                                   extra_zeros, que_coverage_t_1, rev_coverage_t_1, beta, _m, steps)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                que_context_i = c_q_t[i]
                rev_context_i = c_r_t[i]
                que_coverage_i = (que_coverage_t[i] if config.is_coverage else None)
                rev_coverage_i = (rev_coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        que_context=que_context_i,
                                        rev_context=rev_context_i,
                                        que_coverage=que_coverage_i,
                                        rev_coverage=rev_coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break
            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

    def decode(self):
        start = time.time()
        counter = 0
        for batch in self.batcher:
            # 运行beam search得到解码结果
            best_summary = self.beam_search(batch)

            # 提取解码得到的单词ID，忽略解码的第1个[START]单词的ID，然后将单词ID转换为对应的单词
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids,
                                                 self.vocab,
                                                 (batch.oovs[0] if config.pointer_gen else None))

            # 如果解码结果中有[STOP]单词，那么去除它
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_answer = batch.original_answers[0]

            # 将解码结果以及参考答案处理并写入文件，以便后续计算ROUGE分数
            write_for_eval(original_answer,
                           decoded_words,
                           counter,
                           self.ref_dir,
                           self.dec_dir)

            counter += 1
            if counter % 1000 == 0:
                print("%d example in %d sec" % (counter, time.time() - start))
                start = time.time()

        print("Decoder has finished reading dataset.")
        print("Now starting eval...")
        eval_decode_result(self.ref_dir, self.dec_dir)


if __name__ == "__main__":
    beam_Search_processor = BeamSearch(config.decode_model_path)
    beam_Search_processor.decode()
