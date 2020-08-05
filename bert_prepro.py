import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path
import argparse
import torch
import torch
import os
from joblib import Parallel, delayed
from util import normalize_answer
import torch
import bisect
import re
from transformers import BertTokenizer

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


def encode(question, context, config):
    """
    :param question: (str) question in nl
    :param context: (str) the whole context concatenated
    :return: BERT-like tokenization (with special tokens: [CLS] question [SEP] context [SEP])
    """
    encoding = tokenizer.encode_plus(question, context, padding='max_length',
                                     max_length=BERT_MAX_SEQ_LEN,  truncation='only_second')  # cls + 2 * sep
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

    def _process(sent):
        sent = " " + sent + " "
        nonlocal text_context, context_tokens, flat_offsets
        N_chars = len(text_context)
        sent_tokens = tokenize(sent)
        sent_spans = convert_idx(sent, sent_tokens, filter)

        sent_spans = [[N_chars+e[0], N_chars+e[1]] for e in sent_spans]
        text_context += sent
        context_tokens.extend(sent_tokens)
        flat_offsets.extend(sent_spans)

    for para in paragraphs:
        cur_title, cur_para = para[0], para[1]
        _process(cur_title)
        for sent in cur_para:
            _process(sent)
    if 'answer' in article:
        answer = article['answer'].strip()
        if answer.lower() == 'yes':
            best_indices = [-3, -3]
        elif answer.lower() == 'no':
            best_indices = [-2, -2]
        else:
            if answer not in text_context:
                # in the fullwiki setting, the answer might not have been retrieved
                # use (0, 1) so that we can proceed
                best_indices = [-1, -1]
            else:
                best_indices, _ = fix_span(text_context, flat_offsets, article['answer'])
    else:
        # some random stuff
        answer = 'random'
        best_indices = [-1, -1]
    example = {'context' : text_context, 'question' : article['question'],
               'context_tokens': context_tokens, 'question_tokens': tokenize(article['question']),
               'y1s': best_indices[0], 'y2s': best_indices[1] + 1, 'id': article['_id']}
    eval_example = {'context': text_context, 'spans': flat_offsets, 'answer': [answer], 'id': article['_id']}
    return example, eval_example


def process_file(filename, config):
    data = json.load(open(filename, 'r'))

    eval_examples = {}

    outputs = Parallel(n_jobs=12, verbose=10)(delayed(_process_article)(article, config) for article in data)
    examples = [e[0] for e in outputs]
    for _, e in outputs:
        if e is not None:
            eval_examples[e['id']] = e

    random.shuffle(examples)
    print("{} questions in total".format(len(examples)))

    return examples, eval_examples


def build_features(config, examples, data_type, out_file):
    if data_type == 'test':
        BERT_MAX_SEQ_LEN = 100000

    print("Processing {} examples...".format(data_type))
    datapoints = []
    total = 0
    total_ = 0
    context_filtered, question_filtered = 0, 0
    for example in tqdm(examples):
        total_ += 1
        
        question_shift = len(example['question_tokens']) + 2  # cls + question + sep
        # answer span is based on context_text tokens only but after encoding
        #  question and special tokens are added in front

        start, end = example["y1s"], example["y2s"]
        y1, y2 = start + question_shift, end + question_shift

        if y2 > BERT_MAX_SEQ_LEN - 1: # sep
            total += 1
            continue

        input_ids, attention_mask, token_type_ids = encode(example['question'], example['context'], config)

        datapoints.append(
            {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids' : token_type_ids,
                'y1': y1, 'y2': y2, 'id': example['id'], 'question_tokens' : question_tokens}
        )
    print("Filtered {} / {} instances of features in total".format(total, total_))
    dir_name = data_type
    try:
        os.mkdir(dir_name)
        print(f"Directory {dir_name} created")
    except OSError:
        print(f"Directory {dir_name} already exists, writing files there")
    fileparts = out_file.split('.')
    name, ext = '.'.join(fileparts[:-1]), fileparts[-1]  # separating file into name and extention
    num_objects = len(datapoints)
    num_files = config.num_files if config.num_files > 0 else num_objects
    batch_size = num_objects // num_files
    for i in range(num_files - 1):
        torch.save(datapoints[i * batch_size: (i + 1) * batch_size], f'{dir_name}/{name}_{str(i)}.{ext}')
    torch.save(datapoints[(num_files - 1) * batch_size:], f'{dir_name}/{name}_{str(num_files - 1)}.{ext}')


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "w") as fh:
        json.dump(obj, fh)


def prepro(config):
    random.seed(13)
    examples, eval_examples = process_file(config.data_file, config)

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

