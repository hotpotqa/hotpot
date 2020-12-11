import random
from tqdm import tqdm
import spacy
import ujson as json
import jsonlines
import pickle
import collections
from collections import Counter
import numpy as np
import os.path
import argparse
import torch
import os
from joblib import Parallel, delayed
from util import normalize_answer
import torch
import bisect
import re
import string
import copy
from transformers import BertTokenizer
import shutil

BERT_MAX_SEQ_LEN = 512
tokenizer = BertTokenizer.from_pretrained('bert-large-cased', return_token_type_ids=True)


def find_nearest(a, target, test_func=lambda x: True):
    idx = bisect.bisect_left(a, target)
    if (0 <= idx < len(a)) and a[idx] == target:
        return idx, 0
    elif idx == 0:
        return 0, abs(a[0] - target)
    elif idx == len(a):
        return idx - 1, abs(a[-1] - target)
    else:
        d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
        d2 = abs(a[idx-1] - target) if test_func(a[idx-1]) else 1e200
        if d1 > d2:
            return idx-1, d2
        else:
            return idx, d1


def fix_span(parastr, offsets, span):
    span = span.strip()
    assert span in parastr, '{}\t{}'.format(span, parastr)
    begins, ends = map(list, zip(*[x for x in offsets]))

    best_dist = 1e200
    best_indices = None

    if span == parastr:
        return parastr, (0, len(parastr)), 0

    for m in re.finditer(re.escape(span), parastr):
        begin_offset, end_offset = m.span()
        fixed_begin, d1 = find_nearest(begins, begin_offset, lambda x: x < end_offset)
        fixed_end, d2 = find_nearest(ends, end_offset, lambda x: x > begin_offset)
        if d1 + d2 < best_dist:
            best_dist = d1 + d2
            best_indices = (fixed_begin, fixed_end)
            if best_dist == 0:
                break

    assert best_indices is not None
    return best_indices, best_dist


def convert_idx(text, tokens, filter=lambda x: x):
    current = 0
    spans = []
    for token in tokens:
        if token == '[UNK]':
            token = ''
        clear_token = filter(token)
        current = text.find(clear_token, current)
        if current < 0:
            print(token, clear_token, text)
            raise Exception()
        spans.append((current, current + len(clear_token)))
        current += len(clear_token)
    return spans


def tokenize(sent):
    """
    :param sent: (str) sentence in natural language
    :return: (list of str) list of tokens
    """
    return tokenizer.tokenize(sent)


def filter(token):
    """
    :param token: (str) word or part of word after tokenization
    :return: (str) token as a word or part of word (whithout extra chars)
    """
    symbols = '#' #chars that need to be filtered out (e. g. '#' for BERT-like tokenization)
    return token.lstrip(symbols)


def remove_punctuation(sent):
    """
    :param sent: (str) possibly with punctuation
    :return: (str) definitely without punctuation
    """
    return re.sub('[%s]' % re.escape(string.punctuation), '', sent)

def preprocess(sent):
    return sent.encode('latin-1', 'ignore').decode('latin-1')

def encode(question, context, max_length, config):
    """
    :param question: (str) question in nl
    :param context: (str) the whole context concatenated
    :return: BERT-like tokenization (with special tokens: [CLS] question [SEP] context [SEP])
    """
    encoding = tokenizer.encode_plus(question, context, padding='max_length',
                                     max_length=max_length,  truncation='only_second')
    return encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"]


def _process_article(article, config):
    """
    :param article: dict corresponding to a single qa pair
    article.keys(): 'supporting_facts' (list of lists), 'level' (str 'easy', 'medium', 'hard'), 'question' (str),
    'context' (list of lists [title (str), list of sentences (str)], 'answer' (str),  '_id' (str), 'type' (str e.g. 'comparison', 'bridge')
    :param config: class with training params as fields
    :return: dicts with features for train and eval
    """
    paragraphs = article['context']
    # some articles in the fullwiki dev/test sets have zero paragraphs
    if len(paragraphs) == 0:
        paragraphs = [['some random title', ['some random stuff']]]

    text_context, context_tokens = '', []
    flat_offsets = []
    start_end_facts = []  # (start_token_id, end_token_id, is_sup_fact=True/False)
    N_tokens = 0

    def _process(sent, is_sup_fact=False):
        sent = preprocess(sent)
        sent = " " + sent + " "
        nonlocal text_context, context_tokens, flat_offsets, start_end_facts, N_tokens
        N_chars = len(text_context)
        sent_tokens = tokenize(sent)
        sent_spans = convert_idx(sent, sent_tokens, filter)
        sent_spans = [[N_chars+e[0], N_chars+e[1]] for e in sent_spans]
        text_context += sent
        context_tokens.extend(sent_tokens)
        flat_offsets.extend(sent_spans)
        sent_N_tokens = len(sent_tokens)
        start_end_facts.append((N_tokens, N_tokens + sent_N_tokens, is_sup_fact))
        N_tokens += sent_N_tokens

    if 'supporting_facts' in article:
        sp_set = set(list(map(tuple, article['supporting_facts'])))
    else:
        sp_set = set()

    sp_fact_cnt = 0

    for para in paragraphs:
        cur_title, cur_para = para[0], para[1]
        _process(cur_title)
        for sent_id, sent in enumerate(cur_para):
            is_sup_fact = (cur_title, sent_id) in sp_set
            sp_fact_cnt += is_sup_fact
            _process(sent, is_sup_fact)

    is_answerable = True
    is_yes_no = False
    yes_no = None  # 0 if 'no', 1 otherwise
    
    if 'answer' in article.keys():
        best_indices = [0, 0]
        answer = preprocess(article['answer'].strip())
        if answer.lower() in ['yes', 'no']:
            is_yes_no = True
            answer = answer.lower()
            yes_no = int(answer == 'yes')
        else:
            if answer not in text_context:
                # in the fullwiki setting, the answer might not have been retrieved
                # use (0, 1) so that we can proceed
                answer = ''
                is_answerable = False
            else:
                best_indices, _ = fix_span(text_context, flat_offsets, answer)
    else:
        # some random stuff
        answer = ''
        best_indices = [0, 0]
        is_answerable = False

    answer_tokens = tokenizer.encode_plus(answer, padding='max_length', max_length=BERT_MAX_SEQ_LEN // 2,  truncation=True)['input_ids']
    question_tokens = tokenize(preprocess(article['question']))

    example = {'question_tokens': question_tokens, 'context_tokens': context_tokens,
               'y1s': best_indices[0], 'y2s': best_indices[1], 'id': article['_id'],
               'start_end_facts': start_end_facts, 'is_answerable': is_answerable, 'is_yes_no': is_yes_no, 'yes_no': yes_no}
    eval_example = {'context': text_context, 'spans': flat_offsets, 'answer': answer, 'id': article['_id']}
    return example, eval_example


def process_file(filename, config):
    data = json.load(open(filename, 'r'))

    eval_examples = {}

    outputs = Parallel(n_jobs=12, verbose=10)(delayed(_process_article)(article, config) for article in data)
    examples = [e[0] for e in outputs]
    eval_examples = [e[1] for e in outputs]

    random.shuffle(examples)
    print("{} questions in total".format(len(examples)))

    return examples, eval_examples


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.

    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def build_features(config, examples, data_type, out_file):
    if data_type == 'test':
        max_seq_len = 100000
    else:
        max_seq_len = BERT_MAX_SEQ_LEN
    print("Processing {} examples...".format(data_type))
    datapoints = []
    doc_stride = config.doc_stride
    
    unique_id = 0

    unanswerable_count = 0
    yes_no_count = 0
    span_count = 0
    total_count = 0

    unanswerable_ex_count = 0
    yes_no_ex_count = 0
    span_ex_count = 0
    total_ex_count = len(examples)
    max_chunks = 0
    for example in tqdm(examples):
        question_tokens = example['question_tokens']
        context_tokens = example['context_tokens']
        if len(question_tokens) > config.max_query_length:
            question_tokens = question_tokens[0 : config.max_query_length]
        question_shift = len(question_tokens) + 2  # [CLS] + question + [SEP]
        # answer span is based on context_text tokens only but after encoding
        #  question and special tokens are added in front

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_seq = config.max_seq_length - len(question_tokens) - 3

        if config.is_training:
            start, end = example["y1s"], example["y2s"]

        _DocSpan = collections.namedtuple(
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0

        while start_offset < len(context_tokens):
            length = len(context_tokens) - start_offset
            if length > max_tokens_for_seq:
                length = max_tokens_for_seq
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(context_tokens):
                break
            start_offset += min(length, doc_stride)
       
        is_answerable_example = example['is_answerable']
        is_yes_no_example = example['is_yes_no']
        yes_no_example = example['yes_no']

        unanswerable_ex_count += (1 - is_answerable_example)
        yes_no_ex_count += is_yes_no_example
        span_ex_count += (1 - is_yes_no_example) * is_answerable_example

        supportive_facts = example['start_end_facts']
        max_chunks = max(max_chunks, len(doc_spans))
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            segment_ids = []
            max_context = []

            tokens.append("[CLS]")
            segment_ids.append(0)
            max_context.append(0)

            tokens.extend(question_tokens)
            segment_ids.extend([0] * len(question_tokens))
            max_context.extend([0] * len(question_tokens))

            tokens.append("[SEP]")
            segment_ids.append(0)
            max_context.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i

                max_context.append(int(_check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)))
                tokens.append(context_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            max_context.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            pad_len = max_seq_len - len(input_ids)
            pad = [0] * pad_len

            input_ids.extend(pad)
            input_mask.extend(pad)
            segment_ids.extend(pad)
            max_context.extend(pad)

            assert len(input_ids) == config.max_seq_length
            assert len(input_mask) == config.max_seq_length
            assert len(segment_ids) == config.max_seq_length
            assert len(max_context) == config.max_seq_length

            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1

            y1, y2 = 0, 0
            is_answerable = 0
            is_yes_no = 0
            # yes -- 1, no -- 0
            yes_no = 0
            answer_options = [1, 0]
            if config.is_training:
                if not is_answerable_example or \
                        (is_answerable_example and not is_yes_no_example and not (start >= doc_start and end <= doc_end)):
                    y1, y2 = 0, 0
                elif is_answerable_example and (start >= doc_start and end <= doc_end):
                    y1 = start - doc_start + question_shift
                    y2 = end - doc_start + question_shift
                    is_answerable = 1
                if is_yes_no_example:
                    is_yes_no = 1
                    for s, e, sup in supportive_facts:
                        if sup and (s >= doc_start and e <= doc_end):
                            is_answerable = 1
                            yes_no = answer_options[start]
                            y1, y2 = start, end
                            break
                    else:
                        y1, y2 = 0, 0
                yes_no_count += is_yes_no
                span_count += is_answerable * (1 - is_yes_no)
                unanswerable_count += (1 - is_answerable)
                total_count += 1
            labels = [is_answerable, is_yes_no, yes_no, y1, y2]
            datapoints.append({'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': segment_ids,
                               'max_context': max_context, 'feature_id': unique_id, 'example_id': example['id'],
                               'labels' : labels})
            unique_id += 1

    print(f'max chunks: {max_chunks}')
    print(f"unanswerable: {unanswerable_count / total_count}, yes_no: {yes_no_count / total_count}, span: {span_count / total_count}")
    print(f"unansweravle examples: {unanswerable_ex_count}, yes_no_examples: {yes_no_ex_count}, span: {span_ex_count}")

    yes_no_examples = []
    for datapoint in datapoints:
        is_y_n = datapoint['labels'][1]
        ex_id = datapoint['example_id']
        if is_y_n:
            yes_no_examples.append(ex_id)
    print(f'YES NO EXAMPLES IN THE DATA: {len(set(yes_no_examples))}')

    dir_name = data_type
    try:
        os.mkdir(dir_name)
        print(f"Directory {dir_name} created")
    except OSError:
        print(f"Removing directory {dir_name} and creating a new one")
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)
    fileparts = out_file.split('.')
    name, ext = '.'.join(fileparts[:-1]), fileparts[-1]
    num_objects = len(datapoints)
    num_files = config.num_files if config.num_files > 0 else num_objects
    batch_size = num_objects // num_files
    st = 0
    # dividing datapoints between mltiple files for using lazy dataloading with multiple workers
    # files are supposed to be of approximately equal size, but we do not want to split examples between files

    examples_total = 0
    for i in range(num_files - 1):
        examples = [] # holds starting and ending lines of the examples in the respective file
        start = 0
        ex_id = None
        is_y_n = False
        with jsonlines.open(f'{dir_name}/{name}_{str(i)}.{ext}', mode='w') as writer:
            for dp_idx, datapoint in enumerate(datapoints[st: (i + 1) * batch_size]):
                if ex_id and datapoint['example_id'] != ex_id:
                    end = dp_idx
                    if is_y_n:
                        examples.append((start, end))
                    start = end
                ex_id = datapoint['example_id']
                is_y_n = datapoint['labels'][1]
                writer.write(datapoint)
            last_ex_id = datapoints[(i + 1) * batch_size - 1]['example_id']
            j = (i + 1) * batch_size
            while datapoints[j]['example_id'] == last_ex_id:
                writer.write(datapoints[j])
                j += 1
            if is_y_n:
                examples.append((start, j - st))
            st = j
        examples_total += len(examples)
        with open(f'{dir_name}/{name}_{str(i)}_meta.pickle', mode='wb') as fp:
            pickle.dump(examples, fp)
    
    examples = [] # holds starting and ending lines of the examples in the respective file
    start = 0
    ex_id = None
    is_y_n = False

    with jsonlines.open(f'{dir_name}/{name}_{str(num_files - 1)}.{ext}', mode='w') as writer:
        for dp_idx, datapoint in enumerate(datapoints[st:]):
            if ex_id and datapoint['example_id'] != ex_id:
                end = dp_idx
                if is_y_n:
                    examples.append((start, end))
                start = end
            ex_id = datapoint['example_id']
            is_y_n = datapoint['labels'][1]
            writer.write(datapoint)
        end = dp_idx
        if is_y_n:
            examples.append((start, end))
    examples_total += len(examples)
    print(f'EXAMPLES IN THE DATASET: {examples_total}')
    with open(f'{dir_name}/{name}_{str(num_files - 1)}_meta.pickle', mode='wb') as fp:
        pickle.dump(examples, fp)

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "w") as fh:
        json.dump(obj, fh)


def prepro(config):
    random.seed(13)
    examples, eval_examples = process_file(config.data_file, config)

    print(len(examples), len(eval_examples))
    if config.data_split == 'train':
        record_file = config.train_record_file
        eval_file = config.train_eval_file
    elif config.data_split == 'dev':
        record_file = config.dev_record_file
        eval_file = config.dev_eval_file
    elif config.data_split == 'test':
        record_file = config.test_record_file
        eval_file = config.test_eval_file

    build_features(config, examples, config.data_split, record_file)
    save(eval_file, eval_examples, message='{} eval'.format(config.data_split))

