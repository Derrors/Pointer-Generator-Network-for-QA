#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 生成单个问题的答案
@Author       : Qinghe Li
@Create time  : 2021-02-23 16:49:00
@Last update  : 2021-03-03 15:57:49
"""

import time
import torch
import config
import data
from data import Batch, Example, Vocab, get_input_from_batch
from decode import Beam
from model import Model


def build_batch_by_text(text, vocab):
    """
    预处理：构建一个Batch对象
        1. 创建一个 Example(text, "", vocab)
        2. 构建一个 Batch
    """
    example = Example(text, "", vocab)
    ex_list = [example for _ in range(config.beam_size)]
    batch = Batch(ex_list, vocab, config.beam_size)
    return batch


class BeamSearch(object):
    def __init__(self, model_file_path, vocab):
        self.vocab = vocab
        self.model = Model(model_file_path, self.vocab.embeddings())

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self, batch):
        best_summary = self.beam_search(batch)

        output_ids = [int(t) for t in best_summary.tokens[1:]]
        decoded_words = data.outputids2words(output_ids,
                                             self.vocab,
                                             (batch.oovs[0] if config.pointer_gen else None))

        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words

        print("decode_words:", decoded_words)

        return "".join(decoded_words)

    def beam_search(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch)

        encoder_outputs, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
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
            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1,
                                                                                    s_t_1,
                                                                                    c_t_1,
                                                                                    encoder_outputs,
                                                                                    enc_padding_mask,
                                                                                    extra_zeros,
                                                                                    enc_batch_extend_vocab,
                                                                                    coverage_t_1,
                                                                                    steps)
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
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
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


if __name__ == "__main__":
    test_text = "......."
    model_path = "../log/train_20210223_101755/model/**"

    vocab = Vocab(config.vocab_path, config.vocab_size)
    batch = build_batch_by_text(test_text, vocab)
    beam_processor = BeamSearch(model_path, vocab)
    beam_processor.decode(batch)
