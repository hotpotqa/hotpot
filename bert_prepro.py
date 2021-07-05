import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import csv
import random
from tqdm import tqdm
import ujson as json
import jsonlines
import pickle
import collections
import os.path
import os
from joblib import Parallel, delayed
import bisect
import re
import string
from transformers import BertTokenizer, T5Tokenizer
import shutil
import numpy as np
import matplotlib.pyplot as plt

N_ARTICLES = 10
BERT_MAX_SEQ_LEN = 512
MAX_SENT_LEN = 256 - 65 - 3
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', return_token_type_ids=True)
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer = 'placeholder'


def print_chunks(chunks):
    """
    prints chunks' context text
    :param chunks: a list of chunks
    """
    for i, chunk in enumerate(chunks):
        text = tokenizer.decode(chunk['input_ids'])
        # 'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': segment_ids,
        print(f'chunk num: {i}, chunk text: \n{text}\n')


def find_nearest(a, target, test_func=lambda x: True):
    """
    find a token start/end which is closest to the answer span start/end
    :param a: (List[int]) sorted list of starts/ends of the tokens in the context
    :param target: (int) start/end of the answer span
    :param test_func: filter function to ensure result is in the correct range
    :return: idx of the token s/e closest to target, dissimilarity score
    """
    # bisect.bisect_left(list, x) returns a position to insert x in the sorted list
    idx = bisect.bisect_left(a, target)
    # s/e of the answer span is the same as s/e of one of the tokens
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


def fix_span(text_context, offsets, span):
    """
    find start-end indices of the span in the text_context nearest to the existing token start-end indices
    :param text_context: (str) text to search for span in
    :param offsets: (List(Tuple[int, int]) list of begins and ends for each token in the text
    :param span: (str) the answer span to find in the text_context
    :return: span indices, distance to the nearest token indices
    """
    span = span.strip()
    assert span in text_context, f'answer span:{span} is not in the context: {text_context}'
    begins, ends = map(list, zip(*[x for x in offsets]))

    best_dist = 1e200
    best_indices = None

    if span == text_context:
        return text_context, (0, len(text_context)), 0

    # re.escape(pattern) escapes (adds '\' to special characters in pattern)
    # re.finditer(pattern, string) returns match objects for all matches of pattern in the string
    for m in re.finditer(re.escape(span), text_context):
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


def convert_idx(text, tokens):
    """
    finds spans for each token in the text
    :param text: (str) context to look for token in
    :param tokens: (List[str]) tokenized text
    :param filter: a function to filter tokenization artefacts (e.g. '[UNK]' or '#' in the middle of a word)
    :return: (List[Tuple(int, int)]) spans for each token
    """
    current = 0
    spans = []
    is_bert = tokenizer == bert_tokenizer
    for token in tokens:
        clear_token = filter_func(tokenizer.convert_tokens_to_string([token]))
        current = text.find(clear_token, current)
        if is_bert:
            assert current >= 0, f'token: {token}, cleared token: {clear_token}, text: {text}'
        spans.append((current, current + len(clear_token)))
        current += len(clear_token)
    return spans


def filter_func(token):
    """
    :param token: (str) word or part of word after tokenization
    :return: (str) token as a word or part of word (whithout extra chars)
    """
    if token == '[UNK]':
        token = ''
    symbols = ['\u2581', '#'] #chars that need to be filtered out (e. g. '#' for BERT-like tokenization)
    for symbol in symbols:
        token = token.lstrip(symbol)
    return token


def remove_punctuation(sent):
    """
    :param sent: (str) possibly with punctuation
    :return: (str) definitely without punctuation
    """
    return re.sub('[%s]' % re.escape(string.punctuation), '', sent)


def preprocess(sent):
    """
    substitute multiple spaces with one and remove non latin-1 symbols
    :param sent: string to process
    :return: the string without multiple spaces and latin-1 as encoding
    """
    whitespaces = re.compile(r"\s+")
    sent = whitespaces.sub(" ", sent)
    sent = sent.replace("\xad", "")
    return sent.encode('latin-1', 'ignore').decode('latin-1')


def _process_article(article):
    """
    processes the article and turns it into a train dict with features and eval dict with context and answer
    :param article: dict corresponding to a single qa pair
    article.keys(): 'supporting_facts' (List[List[int]]), 'level' (str 'easy', 'medium', 'hard'),
    'question' (str), 'context' (List[List[Tuple(title (str), sentences (List[str])],
    'answer' (str),  '_id' (str), 'type' (str e.g. 'comparison', 'bridge')
    :return: dicts with features for train and eval e.g
     'start_end_facts' (List[Tuple[int]]) (s_token_id, e_token_id, is_sup_fact),
     'question tokens' [List[int]] token indices for the question
    ...
    """
    paragraphs = article['context']

    context_text, context_tokens = '', []
    flat_offsets = []
    start_end_facts = []  # (start_token_id, end_token_id, is_sup_fact=True/False)
    sup_paragraphs = [0] * len(paragraphs) # 1 if paragraph contains supportive fact, else 0
    para_context_text = []
    para_context_tokens = []
    para_start_end_facts = []
    N_tokens, N_chars = 0, 0

    def _process(sent, is_sup_fact=False):
        sent = preprocess(sent)
        nonlocal context_text, context_tokens, flat_offsets, start_end_facts, N_tokens, N_chars
        sent_tokens = tokenizer.tokenize(sent)
        sent_spans = convert_idx(sent, sent_tokens)
        sent_spans = [[N_chars+e[0], N_chars+e[1]] for e in sent_spans]
        flat_offsets.extend(sent_spans)
        context_text += sent
        context_tokens.extend(sent_tokens)
        N_chars += len(sent)
        sent_N_tokens = len(sent_tokens)
        start_end_facts.append((N_tokens, N_tokens + sent_N_tokens, is_sup_fact))
        N_tokens += sent_N_tokens

    if 'supporting_facts' in article:
        sp_set = set(list(map(tuple, article['supporting_facts'])))
    else:
        sp_set = set()

    sp_fact_cnt = 0

    for i, para in enumerate(paragraphs):
        cur_title, cur_para = para[0], para[1]
        _process(cur_title.strip() + " ")
        for sent_id, sent in enumerate(cur_para):
            is_sup_fact = (cur_title, sent_id) in sp_set
            sup_paragraphs[i] |= is_sup_fact
            sp_fact_cnt += is_sup_fact
            _process(sent, is_sup_fact)
        para_context_text.append(context_text)
        para_context_tokens.append(context_tokens)
        para_start_end_facts.append(start_end_facts)
        context_text = "\n"
        context_tokens = []
        start_end_facts = []
        N_chars += 1
        N_tokens = 0

    question = preprocess(article['question'])

    full_text = ''.join(para_context_text)

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
            if answer not in full_text:
                answer = ''
            else:
                best_indices, _ = fix_span(full_text, flat_offsets, answer)
    else:
        # some random stuff
        answer = ''
        best_indices = [0, 0]
        print('UNANSWERABLE: ', article['_id'])

    assert 2 <= sum(sup_paragraphs) <= 2, f'wrong number of sup paragraphs: {sum(sup_paragraphs)}'

    question_tokens = tokenizer.tokenize(question)

    example = {'question_tokens': question_tokens, 'context_tokens': para_context_tokens,
               's': best_indices[0], 'e': best_indices[1], 'id': article['_id'],
               'start_end_facts': para_start_end_facts, 'sup_paragraphs': sup_paragraphs,
               'is_yes_no': is_yes_no, 'yes_no': yes_no}
    eval_example = {'context': full_text, 'spans': flat_offsets, 'answer': answer, 'id': article['_id']}
    return example, eval_example


def process_file(filename, config):
    """
    processes all articles
    :param filename: (str) path to the .json file
    :param config: config class
    :return: List[Dict] train features, List[Dict] eval features (such as full context text and correct answer)
    """
    data = json.load(open(filename, 'r'))


    with open('filter_ids.csv', newline='') as f:
        reader = csv.reader(f)
        _, ids_to_filter = zip(*list(reader))
    outputs = Parallel(n_jobs=8, verbose=10, require='sharedmem')(delayed(_process_article)(article) for article in data if article['_id'] not in ids_to_filter)
    examples, eval_examples = zip(*outputs)

    random.shuffle(list(examples))
    print(f"{len(examples)} questions in total")

    return examples, eval_examples


def _check_is_max_context(doc_spans, cur_span_index, position):
    """
    Check if this is the 'max context' doc span for the token.
    :param doc_spans: List[NamedTuple(start: int, length: int)] sorted by start
    :param cur_span_index: index of current doc_span (in doc_spans)
    :param position: position of the token to check for max_span
    :return: bool whether token in position has max context
    """

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
            break
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def dump_dp(config, datapoints, dir_name, out_file):
    """
    dividing datapoints into multiple files for using lazy data loading with multiple workers
    files are supposed to be of approximately equal size, but we do not want to split examples between files
    :param config: config class
    :param datapoints: List[Dict] a batch of data points
    :param out_file: dumping file main part, data will be dumped to {out_file_name}_{file_num}.{out_file_ext}
    """

    fileparts = out_file.split('.')
    name, ext = '.'.join(fileparts[:-1]), fileparts[-1]

    num_objects = len(datapoints)
    num_files = config.num_files if config.num_files > 0 else num_objects
    batch_size = num_objects // num_files

    st = 0
    examples_total = 0
    for i in range(num_files - 1):
        with jsonlines.open(f'{dir_name}/{name}_{i}.{ext}', mode='w') as writer:
            end = (i + 1) * batch_size
            examples, start = write_datapoints(datapoints[st: end], writer)
            last_ex_id = datapoints[end - 1]['example_id']

            while datapoints[end]['example_id'] == last_ex_id:
                writer.write(datapoints[end])
                end += 1
            examples.append((start, end - st))
            st = end

        examples_total += len(examples)
        with open(f'{dir_name}/{name}_{str(i)}_meta.pickle', mode='wb') as fp:
            pickle.dump(examples, fp)

    with jsonlines.open(f'{dir_name}/{name}_{num_files - 1}.{ext}', mode='w') as writer:
        examples, start = write_datapoints(datapoints[st:], writer)
        end = len(datapoints)
        examples.append((start, end - st))
    examples_total += len(examples)
    with open(f'{dir_name}/{name}_{num_files - 1}_meta.pickle', mode='wb') as fp:
        pickle.dump(examples, fp)
    return examples_total


def write_datapoints(datapoints, writer):
    """
    write datapoints to a file, remember the beginning and ending lines for the yes/no questions
    :param datapoints: List[Dict] a batch of data points
    :param writer: file handler of a file to write to
    :return: examples List[Tuple[int, int]] the beginning and ending lines for the yes/no questions
    """
    examples = []  # holds beginning and ending lines of each example in the respective file
    start = 0
    ex_id = None
    for dp_idx, datapoint in enumerate(datapoints):
        if ex_id is not None and datapoint['example_id'] != ex_id:
            end = dp_idx
            examples.append((start, end))
            start = end
        ex_id = datapoint['example_id']
        writer.write(datapoint)
    return examples, start


def build_bert_datapoint(config,
                         question_tokens, context_tokens,
                         doc_spans, doc_span_index):

    doc_span = doc_spans[doc_span_index]
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
        tokens.append(context_tokens[doc_span.para][split_token_index])
        #except:
        #    print('EXCEPTION:')
        #    print('span index:', i, 'start index:', doc_span.start, 'prev article len:', len(context_tokens[doc_span.para - 1]))
        #    print('split token index:', split_token_index, f'len context_tokens[{doc_span.para}]', len(context_tokens[doc_span.para]))
        #    return
        segment_ids.append(1)

    tokens.append("[SEP]")
    segment_ids.append(1)
    max_context.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    input_len = len(input_ids)
    pad_len = config.max_seq_length - input_len
    pad = [0] * pad_len

    input_ids.extend(pad)
    input_mask.extend(pad)
    segment_ids.extend(pad)
    max_context.extend(pad)

    assert len(input_ids) == config.max_seq_length, len(input_ids)
    assert len(input_mask) == config.max_seq_length
    assert len(segment_ids) == config.max_seq_length
    assert len(max_context) == config.max_seq_length

    datapoint = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': segment_ids,
                 'max_context': max_context, 'seq_len': input_len}
    return datapoint


def build_t5_datapoint(config,
                       question_tokens, context_tokens,
                       doc_spans, doc_span_index):

    doc_span = doc_spans[doc_span_index]
    tokens = []

    tokens.append(tokenizer.tokenize('Query:')[0])
    tokens.extend(question_tokens)
    tokens.append(tokenizer.tokenize('Document:')[0])
    split_token_index = doc_span.start
    tokens.extend(context_tokens[doc_span.para][split_token_index: split_token_index + doc_span.length])
    tokens.append(tokenizer.tokenize('Relevant:')[0])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    input_len = len(input_ids)
    pad_len = config.max_seq_length - input_len
    pad = [0] * pad_len

    input_ids.extend(pad)
    input_mask.extend(pad)

    assert len(input_ids) == config.max_seq_length
    assert len(input_mask) == config.max_seq_length

    datapoint = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': [0] * config.max_seq_length,
            'max_context': [0] * config.max_seq_length, 'seq_len': input_len}
    return datapoint


def build_features(config, examples, split, out_file):
    """
    make chunks of size config.max_seq_length to train a BERT-like model
    each chunk corresponds to a single context text or part of the text
    :param config: config class
    :param examples: (List[Dict]) example dict for each question context pair
    :param split: 'train', 'dev' or 'test'
    :param out_file: (.pickle) dumps all features to config.num_files pickle
    files {split}/{out_file (without extention)}_{i}.pickle
    """
    print(f"Processing {split} examples...")
    datapoints = []
    chunk_lens = []
    datapoint_lens = []
    all_sup_labels = []
    yes_no_example_id_to_ans = {}

    unique_id = 0

    yes_no_count, span_count , total_count = 0, 0, 0

    yes_no_ex_count = 0
    span_ex_count = 0
    max_chunks = 0
    sup_chunks = 0
    ten_chunks = 0

    cropped_questions = 0
    cropped_sentences = 0

    start_token, sep_token, end_token = ('[CLS]', '[SEP]', '[SEP]') if config.tokenizer == 'bert'\
        else ('Question:', 'Document:', 'Relevant:')
    is_relevant_labels = ['false', 'true']
    for i, example in enumerate(tqdm(examples)):
        question_tokens = example['question_tokens']
        context_tokens = example['context_tokens']
        start_end_facts = example['start_end_facts']
        sup_labels = []
        if len(question_tokens) > config.max_query_length:
            question_tokens = question_tokens[0: config.max_query_length]
            cropped_questions += 1
        question_shift = len(question_tokens) + len(tokenizer.tokenize(start_token)) + \
                         len(tokenizer.tokenize(sep_token))
        # answer span is based on context_text tokens only but after encoding
        # question and special tokens are added in front

        max_tokens_for_seq = config.max_seq_length - question_shift - len(tokenizer.tokenize(end_token))

        if config.is_training:
            start, end = example['s'], example['e']

        _DocSpan = collections.namedtuple(
            'DocSpan', ['para', 'start', 'length'])
        doc_spans = []
        sup_paras = np.array(example["sup_paragraphs"])
        
        assert sum(sup_paras) == 2, 'wrong number of supportive facts'

        if config.level == 'paragraph':
            for para_i, para_tokens in enumerate(context_tokens):
                start_offset = 0
                length = len(para_tokens)

                while length > max_tokens_for_seq:
                    doc_spans.append(_DocSpan(para_i, start=start_offset, length=max_tokens_for_seq))
                    start_offset += config.doc_stride
                    length -= config.doc_stride
                    datapoint_lens.append(max_tokens_for_seq)

                doc_spans.append(_DocSpan(para_i, start=start_offset, length=length))
                datapoint_lens.append(length)
        elif config.level == 'sentence':
            for para_i, sent_spans in enumerate(start_end_facts):
                for el in sent_spans:
                    s, e, is_sup_fact = el
                    length = min(e - s, max_tokens_for_seq)
                    if split == 'train' and e - s > MAX_SENT_LEN:
                        continue
                    if e - s > max_tokens_for_seq:
                        cropped_sentences += 1
                    sup_labels.append(is_sup_fact)
                    doc_spans.append(_DocSpan(para_i, start=s, length=length))
                    datapoint_lens.append(e - s)
        else:
            print(f'sorry, there is no such level {config.level}')
        num_chunks = len(doc_spans)

        ten_chunks += int(num_chunks == 10)
        chunk_lens.append(num_chunks)

        is_yes_no_example = example['is_yes_no']
        is_yes_example = example['yes_no']

        if is_yes_no_example:
            yes_no_example_id_to_ans[example['id']] = example['yes_no']
        yes_no_ex_count += is_yes_no_example
        span_ex_count += 1 - is_yes_no_example
        max_chunks = max(max_chunks, num_chunks)

        for doc_span_index, doc_span in enumerate(doc_spans):

            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1

            is_sup_binary = int(sup_paras[doc_span.para]) if config.level == 'paragraph' else sup_labels[doc_span_index]
            all_sup_labels.append(is_sup_binary)
            is_sup = is_sup_binary if config.tokenizer == 'bert' else tokenizer.encode(is_relevant_labels[is_sup_binary])[0]
            sup_chunks += is_sup_binary

            span_s, span_e = 0, 0
            is_yes_no = is_yes_no_example
            # yes -- 1, no -- 0
            is_yes = -1
            if is_yes_no:
                is_yes = is_yes_example

            if config.is_training and start >= doc_start and end <= doc_end:
                span_s = start - doc_start + question_shift
                span_e = end - doc_start + question_shift
                assert 0 <= span_s < config.max_seq_length, f'starts are wrong, {span_s}'
                assert 0 <= span_e < config.max_seq_length, f'ends are wrong, {span_e}'

            labels = [is_yes_no, is_yes, span_s, span_e, is_sup]

            yes_no_count += is_yes_no
            span_count += (1 - is_yes_no)
            total_count += 1
            unique_id += 1

            dp = build_bert_datapoint(config, question_tokens, context_tokens, doc_spans, doc_span_index)\
                if config.tokenizer == 'bert' else \
                build_t5_datapoint(config, question_tokens, context_tokens, doc_spans, doc_span_index)
            dp.update({'feature_id': unique_id, 'example_id': example['id'], 'para': doc_span.para, 'labels': labels})
            datapoints.append(dp)

    chunk_lens = np.array(chunk_lens)
    all_sup_labels = np.array(all_sup_labels)
    datapoint_lens = np.array(datapoint_lens)
    plot_stats(config, chunk_lens, datapoint_lens, all_sup_labels, split)
    
    assert len(datapoint_lens) == len(all_sup_labels), f'len of datapoints lens: {len(datapoints_lens)} does not match len of sup labels: {len(all_sup_labels)}'
    print(f"max chunks: {max_chunks}, sup chunks: {sup_chunks}, avg num of sup chunks: {sup_chunks / span_ex_count}")
    print(f"yes_no: {yes_no_count}, span: {span_count}, total: {total_count}")
    print(f"yes_no: {yes_no_count / total_count:.5f}, span: {span_count / total_count:.5f}")
    print(f"yes_no_examples: {yes_no_ex_count}, span: {span_ex_count}")
    print(f"Examples in 10 chunks: {ten_chunks}, total examples: {len(examples)}")
    print(f"Examples with questions longer than {config.max_query_length} tokens: {cropped_questions}, cropped sentences: {cropped_sentences}")
    print(f"Chunk lens: {np.quantile(chunk_lens, q=0.5)}, {np.quantile(chunk_lens, q=0.75)}, {np.quantile(chunk_lens, q=0.9)}, \
            {np.quantile(chunk_lens, q=0.95)}, {np.quantile(chunk_lens, q=0.99)}")

    dir_name = f'{config.level}_{split}'

    try:
        os.mkdir(dir_name)
        print(f"Directory {dir_name} created")
    except OSError:
        print(f"Removing directory {dir_name} and creating a new one")
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)

    with open(f'{dir_name}/{config.yes_no_example_file}', "wb") as fp:
        pickle.dump([yes_no_example_id_to_ans], fp)

    dumped = dump_dp(config, datapoints, dir_name, out_file)
    assert dumped == len(examples), f'number of examples dumped ({dumped}) is not equal to' \
                                    f' total number of examples: {len(examples)}'


def plot_stats(config, chunk_lens, datapoint_lens, all_sup_labels, split):
    fig, axs = plt.subplots(ncols=2, nrows=2, sharey='all')
    axs = axs.flatten()
    
    for i in range(4):
        axs[i].set_yscale('log')
    shared_list = axs[1:]
    shared_list[0].get_shared_x_axes().join(*shared_list)
    name = 'Fact' if config.level == 'sentence' else 'Chunk'

    axs[0].hist(chunk_lens, bins=35)
    axs[0].set_title(f'{name} number\n distribution\n')

    axs[1].hist(datapoint_lens, bins=35)
    axs[1].set_title(f'{name} token number \n distribution\n')

    axs[2].hist(datapoint_lens[all_sup_labels.astype(bool)], bins=35)
    axs[2].set_title(f'Supporting {name} token number \n distribution\n')

    axs[3].hist(datapoint_lens[(1 - all_sup_labels).astype(bool)], bins=35)
    axs[3].set_title(f'Non-supporting {name} token number \n distribution\n')
    fig.tight_layout()
    plt.savefig(f'plots/{split}_{config.level}_eda.png')



def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "w") as fh:
        json.dump(obj, fh)


def prepro(config):
    """
    process all files and build features for BERT-like models, save them to files
    :param config: config class
    """
    random.seed(13)
    # remove sample weights file
    weights_file = '/mnt/localdata/rak/kg_qa/models/SpanBERT/outputs/train_weights.pickle'
    if os.path.exists(weights_file):
        os.remove(weights_file)
    global tokenizer
    tokenizer = bert_tokenizer if config.tokenizer == 'bert' else t5_tokenizer
    examples, eval_examples = process_file(config.data_file, config)
    test_example = eval_examples[0]
    test_spans = test_example['spans']

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

