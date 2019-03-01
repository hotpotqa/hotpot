from convert import convert_hotpot_to_squad_format
import json

def prepro_bert(config):
    json_dict = json.load(open(config.data_file, 'r'))

    if config.data_split == 'train':
        eval_file = config.train_eval_file
    elif config.data_split == 'dev':
        eval_file = config.dev_eval_file
    elif config.data_split == 'test':
        eval_file = config.test_eval_file

    squad = convert_hotpot_to_squad_format(json_dict, gold_paras_only=False, combine_context=True)

    with open(eval_file, 'w') as fp:
        json.dump(squad, fp)