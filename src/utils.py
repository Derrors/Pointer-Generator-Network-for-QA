#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : ROUGE 计算、结果处理
@Author       : Qinghe Li
@Create time  : 2021-02-16 10:07:28
@Last update  : 2021-03-06 15:04:19
"""

import glob
import logging
import os

import pyrouge
import tensorflow as tf
from distinct_n import distinct_n_corpus_level
from nltk.translate.bleu_score import corpus_bleu
from prettytable import PrettyTable


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def print_results(question, answer, decoded_output):
    print("")
    print("Question:  %s", question)
    print("Reference answer: %s", answer)
    print("Generated answer: %s", decoded_output)
    print("")


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()

    r.model_filename_pattern = "#ID#_reference.txt"
    r.system_filename_pattern = r"(\d+)_decoded.txt"

    r.model_dir = ref_dir
    r.system_dir = dec_dir

    logging.getLogger("global").setLevel(logging.WARNING)
    rouge_results = r.convert_and_evaluate()
    rouge_dict = r.output_to_dict(rouge_results)

    result_dict = {}
    for x in ["1", "l"]:
        key = "rouge_%s_f_score" % (x)
        val = rouge_dict[key]
        result_dict[key] = val

    rouge_1 = result_dict["rouge_1_f_score"]
    rouge_l = result_dict["rouge_l_f_score"]

    return rouge_1, rouge_l


def eval_decode_result(ref_dir, dec_dir):
    rouge_1, rouge_l = rouge_eval(ref_dir, dec_dir)

    ref_filelist = glob.glob(os.path.join(ref_dir, "*_reference.txt"))
    dec_filelist = glob.glob(os.path.join(dec_dir, "*_decoded.txt"))

    ref_sentences = []
    dec_sentences = []

    for f in ref_filelist:
        with open(f, "r") as rf:
            for s in rf.readlines():
                if len(s) > 0:
                    ref_sentences.append([s])
    for f in dec_filelist:
        with open(f, "r") as rf:
            for s in rf.readlines():
                if len(s) > 0:
                    dec_sentences.append(s)

    assert len(ref_sentences) == len(dec_sentences)

    bleu_1 = corpus_bleu(ref_sentences, dec_sentences, weights=[1.0, 0.0])
    bleu_2 = corpus_bleu(ref_sentences, dec_sentences, weights=[0.0, 1.0])

    distinct_1 = distinct_n_corpus_level(dec_sentences, 1)
    distinct_2 = distinct_n_corpus_level(dec_sentences, 2)

    table = PrettyTable()
    table.field_names = ["ROUGE-1", "ROUGE-L", "BLEU-1", "BLEU-2", "Distinct-1", "Distinct-2"]
    table.add_row((round(rouge_1, 4), round(rouge_l, 4), round(bleu_1, 4), round(bleu_2, 4), round(distinct_1, 4), round(distinct_2, 4)))
    print(table)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)

    loss_sum = tf.compat.v1.Summary()
    tag_name = "running_avg_loss/decay=%f" % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)

    return running_avg_loss


def write_for_eval(reference_sent, decoded_words, ex_index, ref_dir, dec_dir):

    decoded_sent = " ".join(decoded_words).strip()
    decoded_sent = make_html_safe(decoded_sent)
    reference_sent = make_html_safe(reference_sent)

    ref_file = os.path.join(ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as fr:
        fr.write(reference_sent)
    with open(decoded_file, "w") as fd:
        fd.write(decoded_sent)


if __name__ == "__main__":
    eval_decode_result("../log/decode_model_elec_0/ref_dir", "../log/decode_model_elec_0/dec_dir")
