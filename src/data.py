#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 词表处理
@Author       : Qinghe Li
@Create time  : 2021-02-23 15:58:19
@Last update  : 2021-03-03 16:39:31
"""

import glob
import random
import struct
import time

import numpy as np
import tensorflow as tf
import torch
from tensorflow.core.example import example_pb2

import config

PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"
START_DECODING = "[START]"
STOP_DECODING = "[STOP]"


class Vocab(object):
    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0     # 记录当前词表的词数
        self._embeddings = None

        # [PAD], [UNK], [START] and [STOP] 对应的id分别为 0,1,2,3.
        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # 如果词表文件为预训练的Glove向量，则读取对应的词向量
        if vocab_file.endswith(".txt"):
            self._embeddings = []
            for _ in range(4):
                self._embeddings.append(np.random.rand(config.emb_dim, ))
            self._embeddings[0] = np.zeros((config.emb_dim, ))

            # 读取词表文件并建立词典，最大词数为 max_size
            with open(vocab_file, "r", encoding="utf-8") as vocab_f:
                for line in vocab_f.readlines():
                    pieces = line.split()

                    if len(pieces) != config.emb_dim + 1:
                        continue

                    w = pieces[0]
                    if w in self._word_to_id:
                        continue

                    self._word_to_id[w] = self._count
                    self._id_to_word[self._count] = w
                    self._embeddings.append(np.asarray(pieces[1:], dtype=float))
                    self._count += 1

                    if max_size != 0 and self._count >= max_size:
                        print(
                            "max_size of vocab was specified as %i; we now have %i words. Stopping reading." %
                            (max_size, self._count))
                        break
            assert len(self._word_to_id) == len(self._id_to_word) == len(self._embeddings)
        else:
            # 读取词表文件并建立词典，最大词数为 max_size
            with open(vocab_file, "r") as vocab_f:
                for line in vocab_f:
                    pieces = line.split()

                    if len(pieces) != 2:
                        continue

                    w = pieces[0]
                    if w in self._word_to_id:
                        continue

                    self._word_to_id[w] = self._count
                    self._id_to_word[self._count] = w
                    self._count += 1

                    if max_size != 0 and self._count >= max_size:
                        print(
                            "max_size of vocab was specified as %i; we now have %i words. Stopping reading." %
                            (max_size, self._count))
                        break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" %
              (self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        """获取单个词语的id"""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """根据id解析出对应的词语"""
        if word_id not in self._id_to_word:
            raise ValueError("Id not found in vocab: %d" % word_id)
        return self._id_to_word[word_id]

    def embeddings(self):
        if self._embeddings is not None:
            return torch.tensor(self._embeddings, dtype=torch.float)
        else:
            return None

    def size(self):
        """获取加上特殊符号后的词汇表数量"""
        return self._count


class Example(object):
    def __init__(self, question, answer, reviews, vocab):
        start_decoding = vocab.word2id(START_DECODING)
        stop_decoding = vocab.word2id(STOP_DECODING)

        # 处理问题文本
        question_words = question.split()
        if len(question_words) > config.max_que_steps:
            question_words = question_words[:config.max_que_steps]
        self.que_input = [vocab.word2id(w) for w in question_words]     # 编码问题，这里OOV被编码为UNK对应的ID
        self.que_len = len(question_words)

        # 处理答案文本
        answer_words = answer.split()
        ans_ids = [vocab.word2id(w) for w in answer_words]
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(
            ans_ids, config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # 处理评论文本
        self.rev_lens = []
        self.rev_inputs = []
        reviews_words = []
        for review in reviews:
            review_words = review.split()
            if len(review_words) > config.max_rev_steps:
                review_words = review_words[:config.max_rev_steps]
            reviews_words.append(review_words)
            self.rev_lens.append(len(review_words))
            self.rev_inputs.append([vocab.word2id(w) for w in review_words])

        # 如果使用pointer-generator模式, 需要保存一些额外信息
        if config.pointer_gen:
            self.oovs = []

            # 包含OOV词的文本编码，同时得到问题文本中的OOV词表
            self.que_input_extend_vocab, self.oovs = question2ids(question_words, vocab, self.oovs)

            self.rev_inputs_extend_vocab = []
            for review_words in reviews_words:
                rev_input_extend_vocab, self.oovs = question2ids(review_words, vocab, self.oovs)
                self.rev_inputs_extend_vocab.append(rev_input_extend_vocab)

            # 考虑问题文本内的OOV单词的目标序列编码
            ans_ids_extend_vocab, self.oovs = answer2ids(answer_words, vocab, self.oovs)
            _, self.target = self.get_dec_inp_targ_seqs(
                ans_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

        # 存储原始的问题和答案
        self.original_question = question
        self.original_answer = answer
        self.original_reviews = reviews

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:
            inp = inp[:max_len]
            target = target[:max_len]   # 没有结束标志
        else:
            target.append(stop_id)      # 加入结束标志
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_question_input(self, max_len, pad_id):
        while len(self.que_input) < max_len:
            self.que_input.append(pad_id)
        if config.pointer_gen:
            while len(self.que_input_extend_vocab) < max_len:
                self.que_input_extend_vocab.append(pad_id)

    def pad_review_input(self, max_len, pad_id):
        for i in range(len(self.rev_inputs)):
            while len(self.rev_inputs[i]) < max_len:
                self.rev_inputs[i].append(pad_id)
        if config.pointer_gen:
            for i in range(len(self.rev_inputs_extend_vocab)):
                while len(self.rev_inputs_extend_vocab[i]) < max_len:
                    self.rev_inputs_extend_vocab[i].append(pad_id)


class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(PAD_TOKEN)          # PAD-id
        self.init_encoder_seq(example_list)             # 初始化编码器输入序列
        self.init_decoder_seq(example_list)             # 初始化解码器的输入序列和目标序列
        self.store_orig_strings(example_list)           # 保存原始文本

    def init_encoder_seq(self, example_list):
        # 获取当前batch中编码器最大的输入长度（问题文本的最大长度）
        max_que_seq_len = max([ex.que_len for ex in example_list])
        max_rev_seq_len = max([max(ex.rev_lens) for ex in example_list])

        # 将小于最大长度的问题文本进行补全填充
        for ex in example_list:
            ex.pad_question_input(max_que_seq_len, self.pad_id)
            ex.pad_review_input(max_rev_seq_len, self.pad_id)

        # 初始化
        self.que_batch = np.zeros((self.batch_size, max_que_seq_len), dtype=np.int32)
        self.que_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.que_padding_mask = np.zeros((self.batch_size, max_que_seq_len), dtype=np.float32)

        self.rev_batch = np.zeros((self.batch_size, config.review_num, max_rev_seq_len), dtype=np.int32)
        self.rev_lens = np.zeros((self.batch_size, config.review_num), dtype=np.int32)
        self.rev_padding_mask = np.zeros((self.batch_size, config.review_num, max_rev_seq_len), dtype=np.float32)

        # 填充编码器的输入序列
        for i, ex in enumerate(example_list):
            self.que_batch[i, :] = ex.que_input[:]
            self.que_lens[i] = ex.que_len
            self.rev_batch[i, :] = ex.rev_inputs[:]
            self.rev_lens[i, :] = ex.rev_lens[:]

            for j in range(ex.que_len):
                self.que_padding_mask[i][j] = 1

            for j in range(len(ex.rev_lens)):
                for k in range(ex.rev_lens[j]):
                    self.rev_padding_mask[i][j][k] = 1

        # pointer-generator 模式下的序列填充
        if config.pointer_gen:
            # 当前batch中OOV词的最大数量
            self.max_oovs = max([len(ex.oovs) for ex in example_list])
            # 保存当前batch中的OOV词
            self.oovs = [ex.oovs for ex in example_list]
            # 考虑OOV词的编码器输入序列
            self.que_batch_extend_vocab = np.zeros((self.batch_size, max_que_seq_len), dtype=np.int32)
            self.rev_batch_extend_vocab = np.zeros(
                (self.batch_size, config.review_num, max_rev_seq_len), dtype=np.int32)

            for i, ex in enumerate(example_list):
                self.que_batch_extend_vocab[i, :] = ex.que_input_extend_vocab[:]

            for i, ex in enumerate(example_list):
                self.rev_batch_extend_vocab[i, :] = ex.rev_inputs_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # 填充解码器的输入、目标序列
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len

            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_questions = [ex.original_question for ex in example_list]
        self.original_answers = [ex.original_answer for ex in example_list]
        self.original_reviews = [ex.original_reviews for ex in example_list]


def question2ids(question_words, vocab, oovs):
    """返回两个列表：包含oov词汇的问题文本id; 问题文本的oov词汇列表"""
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in question_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
            # 计算OOV词的id
            oov_num = oovs.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def answer2ids(answer_words, vocab, oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in answer_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w in oovs:
                vocab_idx = vocab.size() + oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids, oovs


def outputids2words(id_list, vocab, question_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)
        except ValueError:  # OOV
            assert question_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            question_oov_idx = i - vocab.size()
            try:
                w = question_oovs[question_oov_idx]
            except ValueError:
                raise ValueError(
                    "Error: model produced word ID %i which corresponds to question OOV %i but this example only has %i question OOVs" %
                    (i, question_oov_idx, len(question_oovs)))
        words.append(w)
    return words


def show_que_oovs(question, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = question.split(" ")
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = " ".join(words)
    return out_str


def show_ans_oovs(answer, vocab, question_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = answer.split(" ")
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # OOV
            if question_oovs is None:
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in question_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:
            new_words.append(w)
    out_str = " ".join(new_words)
    return out_str


def get_input_from_batch(batch):
    """处理batch为模型的输入数据"""
    que_batch = torch.tensor(batch.que_batch, dtype=torch.long, device=config.DEVICE)
    que_padding_mask = torch.tensor(batch.que_padding_mask, dtype=torch.float, device=config.DEVICE)
    que_lens = torch.tensor(batch.que_lens, dtype=torch.long)
    rev_batch = torch.tensor(batch.rev_batch, dtype=torch.long, device=config.DEVICE)
    rev_padding_mask = torch.tensor(batch.rev_padding_mask, dtype=torch.float, device=config.DEVICE)
    rev_lens = torch.tensor(batch.rev_lens, dtype=torch.long)

    extra_zeros = None
    que_batch_extend_vocab = None
    rev_batch_extend_vocab = None

    if config.pointer_gen:
        que_batch_extend_vocab = torch.tensor(batch.que_batch_extend_vocab, dtype=torch.long, device=config.DEVICE)
        rev_batch_extend_vocab = torch.tensor(batch.rev_batch_extend_vocab, dtype=torch.long, device=config.DEVICE)

        if batch.max_oovs > 0:
            extra_zeros = torch.zeros((batch.batch_size, batch.max_oovs), device=config.DEVICE)

    c_t_0 = torch.zeros((batch.batch_size, 2 * config.hidden_dim), device=config.DEVICE)

    que_coverage = None
    rev_coverage = None
    if config.is_coverage:
        que_coverage = torch.zeros(que_batch.size(), device=config.DEVICE)
        rev_coverage = torch.zeros(rev_batch.size(), device=config.DEVICE).view(batch.batch_size, -1)

    return que_batch, que_padding_mask, que_lens, que_batch_extend_vocab, rev_batch, rev_padding_mask, rev_lens, rev_batch_extend_vocab, extra_zeros, c_t_0, que_coverage, rev_coverage


def get_output_from_batch(batch):
    dec_batch = torch.tensor(batch.dec_batch, dtype=torch.long, device=config.DEVICE)
    dec_padding_mask = torch.tensor(batch.dec_padding_mask, dtype=torch.float, device=config.DEVICE)
    dec_lens = torch.tensor(batch.dec_lens, dtype=torch.long, device=config.DEVICE)

    max_dec_len = np.max(batch.dec_lens)
    target_batch = torch.tensor(batch.target_batch, dtype=torch.long, device=config.DEVICE)

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens, target_batch


def get_batch_data_list(data_path, vocab, batch_size, mode):
    """读取目录下的数据文件并处理为batch数据"""
    examples_list = []

    # 获取路径下的所有文件
    filelist = glob.glob(data_path)
    assert filelist, ("Error: Empty filelist at %s" % data_path)

    if mode == "train":
        random.shuffle(filelist)
    else:
        filelist = sorted(filelist)
    # 依次读取数据样例
    for f in filelist:
        reader = open(f, "rb")
        while True:
            len_bytes = reader.read(8)
            if not len_bytes:
                break
            str_len = struct.unpack("q", len_bytes)[0]
            example_str = struct.unpack("%ds" % str_len, reader.read(str_len))[0]
            e = example_pb2.Example.FromString(example_str)

            question_text = e.features.feature["question"].bytes_list.value[0].decode().strip()
            answer_text = e.features.feature["answer"].bytes_list.value[0].decode().strip()
            reviews = eval(e.features.feature["reviews"].bytes_list.value[0].decode().strip())

            if len(question_text) == 0 or len(answer_text) == 0:
                continue

            # 处理为一个Example对象,并加入到数据列表中
            example = Example(question_text, answer_text, reviews, vocab)
            examples_list.append(example)
        reader.close()

    # mini-batch列表
    batches_list = []
    if mode == "decode":
        for e in examples_list:
            # decode模式时，beam search 的 batch 中只有一个样例的多个副本
            b = [e for _ in range(batch_size)]
            batches_list.append(Batch(b, vocab, batch_size))
    else:
        # 根据输入长度进行降序排列
        examples = sorted(examples_list, key=lambda inp: inp.que_len, reverse=True)
        for i in range(0, len(examples), batch_size):
            if i + batch_size < len(examples):
                batches_list.append(Batch(examples[i:i + batch_size], vocab, batch_size))

        random.shuffle(batches_list)
    return batches_list
