import torch
import numpy as np
import re
from collections import Counter
import string
import pickle
import random
from torch.autograd import Variable
import copy
import ujson as json
import traceback

IGNORE_INDEX = -100

RE_D = re.compile('\d')
def has_digit(string):
    return RE_D.search(string)

def prepro(token):
    return token if not has_digit(token) else 'N'

class DataIterator(object):
    def __init__(self, buckets, bsz, para_limit, ques_limit, char_limit, shuffle, sent_limit):
        self.buckets = buckets
        self.bsz = bsz
        if para_limit is not None and ques_limit is not None:
            self.para_limit = para_limit
            self.ques_limit = ques_limit
        else:
            para_limit, ques_limit = 0, 0
            for bucket in buckets:
                for dp in bucket:
                    para_limit = max(para_limit, dp['context_idxs'].size(0))
                    ques_limit = max(ques_limit, dp['ques_idxs'].size(0))
            self.para_limit, self.ques_limit = para_limit, ques_limit
        self.char_limit = char_limit
        self.sent_limit = sent_limit

        self.num_buckets = len(self.buckets)
        self.bkt_pool = [i for i in range(self.num_buckets) if len(self.buckets[i]) > 0]
        if shuffle:
            for i in range(self.num_buckets):
                random.shuffle(self.buckets[i])
        self.bkt_ptrs = [0 for i in range(self.num_buckets)]
        self.shuffle = shuffle

    def __iter__(self):
        context_idxs = torch.LongTensor(self.bsz, self.para_limit).cuda()
        ques_idxs = torch.LongTensor(self.bsz, self.ques_limit).cuda()
        context_char_idxs = torch.LongTensor(self.bsz, self.para_limit, self.char_limit).cuda()
        ques_char_idxs = torch.LongTensor(self.bsz, self.ques_limit, self.char_limit).cuda()
        y1 = torch.LongTensor(self.bsz).cuda()
        y2 = torch.LongTensor(self.bsz).cuda()
        q_type = torch.LongTensor(self.bsz).cuda()
        start_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit).cuda()
        end_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit).cuda()
        all_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit).cuda()
        is_support = torch.LongTensor(self.bsz, self.sent_limit).cuda()

        while True:
            if len(self.bkt_pool) == 0: break
            bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]
            start_id = self.bkt_ptrs[bkt_id]
            cur_bucket = self.buckets[bkt_id]
            cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

            ids = []

            cur_batch = cur_bucket[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: (x['context_idxs'] > 0).long().sum(), reverse=True)

            max_sent_cnt = 0
            for mapping in [start_mapping, end_mapping, all_mapping]:
                mapping.zero_()
            is_support.fill_(IGNORE_INDEX)

            for i in range(len(cur_batch)):
                context_idxs[i].copy_(cur_batch[i]['context_idxs'])
                ques_idxs[i].copy_(cur_batch[i]['ques_idxs'])
                context_char_idxs[i].copy_(cur_batch[i]['context_char_idxs'])
                ques_char_idxs[i].copy_(cur_batch[i]['ques_char_idxs'])
                if cur_batch[i]['y1'] >= 0:
                    y1[i] = cur_batch[i]['y1']
                    y2[i] = cur_batch[i]['y2']
                    q_type[i] = 0
                elif cur_batch[i]['y1'] == -1:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1
                elif cur_batch[i]['y1'] == -2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2
                elif cur_batch[i]['y1'] == -3:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 3
                else:
                    assert False
                ids.append(cur_batch[i]['id'])

                for j, cur_sp_dp in enumerate(cur_batch[i]['start_end_facts']):
                    if j >= self.sent_limit: break
                    if len(cur_sp_dp) == 3:
                        start, end, is_sp_flag = tuple(cur_sp_dp)
                    else:
                        start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
                    if start < end:
                        start_mapping[i, start, j] = 1
                        end_mapping[i, end-1, j] = 1
                        all_mapping[i, start:end, j] = 1
                        is_support[i, j] = int(is_sp_flag)

                max_sent_cnt = max(max_sent_cnt, len(cur_batch[i]['start_end_facts']))

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            max_q_len = int((ques_idxs[:cur_bsz] > 0).long().sum(dim=1).max())

            self.bkt_ptrs[bkt_id] += cur_bsz
            if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
                self.bkt_pool.remove(bkt_id)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'ques_idxs': ques_idxs[:cur_bsz, :max_q_len].contiguous(),
                'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                'ques_char_idxs': ques_char_idxs[:cur_bsz, :max_q_len].contiguous(),
                'context_lens': input_lengths,
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
                'start_mapping': start_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                'end_mapping': end_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt]}

def get_buckets(record_file):
    # datapoints = pickle.load(open(record_file, 'rb'))
    datapoints = torch.load(record_file)
    return [datapoints]

def convert_tokens(eval_file, qa_id, pp1, pp2, p_type):
    answer_dict = {}
    for qid, p1, p2, type in zip(qa_id, pp1, pp2, p_type):
        if type == 0:
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx: end_idx]
        elif type == 1:
            answer_dict[str(qid)] = 'yes'
        elif type == 2:
            answer_dict[str(qid)] = 'no'
        elif type == 3:
            answer_dict[str(qid)] = 'noanswer'
        else:
            assert False
    return answer_dict

def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answer"]
        prediction = value
        assert len(ground_truths) == 1
        cur_EM = exact_match_score(prediction, ground_truths[0])
        cur_f1, _, _ = f1_score(prediction, ground_truths[0])
        exact_match += cur_EM
        f1 += cur_f1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

# def evaluate(eval_file, answer_dict, full_stats=False):
#     if full_stats:
#         with open('qaid2type.json', 'r') as f:
#             qaid2type = json.load(f)
#         f1_b = exact_match_b = total_b = 0
#         f1_4 = exact_match_4 = total_4 = 0

#         qaid2perf = {}

#     f1 = exact_match = total = 0
#     for key, value in answer_dict.items():
#         total += 1
#         ground_truths = eval_file[key]["answer"]
#         prediction = value
#         cur_EM = metric_max_over_ground_truths(
#             exact_match_score, prediction, ground_truths)
#         # cur_f1 = metric_max_over_ground_truths(f1_score,
#                                             # prediction, ground_truths)
#         assert len(ground_truths) == 1
#         cur_f1, cur_prec, cur_recall = f1_score(prediction, ground_truths[0])
#         exact_match += cur_EM
#         f1 += cur_f1
#         if full_stats and key in qaid2type:
#             if qaid2type[key] == '4':
#                 f1_4 += cur_f1
#                 exact_match_4 += cur_EM
#                 total_4 += 1
#             elif qaid2type[key] == 'b':
#                 f1_b += cur_f1
#                 exact_match_b += cur_EM
#                 total_b += 1
#             else:
#                 assert False

#         if full_stats:
#             qaid2perf[key] = {'em': cur_EM, 'f1': cur_f1, 'pred': prediction,
#                     'prec': cur_prec, 'recall': cur_recall}

#     exact_match = 100.0 * exact_match / total
#     f1 = 100.0 * f1 / total

#     ret = {'exact_match': exact_match, 'f1': f1}
#     if full_stats:
#         if total_b > 0:
#             exact_match_b = 100.0 * exact_match_b / total_b
#             exact_match_4 = 100.0 * exact_match_4 / total_4
#             f1_b = 100.0 * f1_b / total_b
#             f1_4 = 100.0 * f1_4 / total_4
#             ret.update({'exact_match_b': exact_match_b, 'f1_b': f1_b,
#                 'exact_match_4': exact_match_4, 'f1_4': f1_4,
#                 'total_b': total_b, 'total_4': total_4, 'total': total})

#         ret['qaid2perf'] = qaid2perf

#     return ret

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

