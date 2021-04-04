import os
from prepro import prepro
from bert_prepro import prepro as BERT_prepro
from run import train, test
import argparse

parser = argparse.ArgumentParser()

glove_word_file = "glove.840B.300d.txt"

word_emb_file = "word_emb.json"
char_emb_file = "char_emb.json"
train_eval = "train_eval.json"
dev_eval = "dev_eval.json"
test_eval = "test_eval.json"
word2idx_file = "word2idx.json"
char2idx_file = "char2idx.json"
idx2word_file = 'idx2word.json'
idx2char_file = 'idx2char.json'
train_record_file = 'train_record.jsonl'
dev_record_file = 'dev_record.jsonl'
test_record_file = 'test_record.jsonl'
yes_no_example_file = 'yn_id_to_ans.pickle'



parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data_file', type=str)
parser.add_argument('--glove_word_file', type=str, default=glove_word_file)
parser.add_argument('--save', type=str, default='HOTPOT')

parser.add_argument('--word_emb_file', type=str, default=word_emb_file)
parser.add_argument('--char_emb_file', type=str, default=char_emb_file)
parser.add_argument('--train_eval_file', type=str, default=train_eval)
parser.add_argument('--dev_eval_file', type=str, default=dev_eval)
parser.add_argument('--test_eval_file', type=str, default=test_eval)
parser.add_argument('--word2idx_file', type=str, default=word2idx_file)
parser.add_argument('--char2idx_file', type=str, default=char2idx_file)
parser.add_argument('--idx2word_file', type=str, default=idx2word_file)
parser.add_argument('--idx2char_file', type=str, default=idx2char_file)

parser.add_argument('--train_record_file', type=str, default=train_record_file)
parser.add_argument('--dev_record_file', type=str, default=dev_record_file)
parser.add_argument('--test_record_file', type=str, default=test_record_file)

parser.add_argument('--yes_no_example_file', type=str, default=yes_no_example_file)

parser.add_argument('--glove_char_size', type=int, default=94)
parser.add_argument('--glove_word_size', type=int, default=int(2.2e6))
parser.add_argument('--glove_dim', type=int, default=300)
parser.add_argument('--char_dim', type=int, default=8)

parser.add_argument('--para_limit', type=int, default=1000)
parser.add_argument('--ques_limit', type=int, default=80)
parser.add_argument('--sent_limit', type=int, default=100)
parser.add_argument('--char_limit', type=int, default=16)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--checkpoint', type=int, default=1000)
parser.add_argument('--period', type=int, default=100)
parser.add_argument('--init_lr', type=float, default=0.5)
parser.add_argument('--keep_prob', type=float, default=0.8)
parser.add_argument('--hidden', type=int, default=80)
parser.add_argument('--char_hidden', type=int, default=100)
parser.add_argument('--patience', type=int, default=1)
parser.add_argument('--seed', type=int, default=13)

parser.add_argument('--sp_lambda', type=float, default=0.0)

parser.add_argument('--data_split', type=str, default='train')
parser.add_argument('--fullwiki', action='store_true')
parser.add_argument('--prediction_file', type=str)
parser.add_argument('--sp_threshold', type=float, default=0.3)

parser.add_argument('--num_files', type=int, default=1)
parser.add_argument('--tokenizer', type=str, default='spacy')
parser.add_argument('--doc_stride', type=int, default=128)
parser.add_argument('--is_training', type=bool, default=True)
parser.add_argument('--max_query_length', type=int, default=75)
parser.add_argument('--max_seq_length', type=int, default=512)
parser.add_argument('--level', type=str, default='paragraph')

config = parser.parse_args()

def _concat(config, filename):
    new_name = filename
    #new_name = '{}_{}'.format(config.level, filename)
    if config.fullwiki:
        new_name = 'fullwiki.{}'.format(new_name)
    return new_name

config.train_record_file = _concat(config, config.train_record_file)
config.dev_record_file = _concat(config, config.dev_record_file)
config.test_record_file = _concat(config, config.test_record_file)
config.dev_eval_file = _concat(config, config.dev_eval_file)
config.test_eval_file = _concat(config, config.test_eval_file)

print(config.dev_record_file)

if config.mode == 'train':
    train(config)
elif config.mode == 'prepro':
    if config.tokenizer == 'spacy':
        prepro(config)
    else:
        BERT_prepro(config)
elif config.mode == 'test':
    test(config)
elif config.mode == 'count':
    cnt_len(config)
