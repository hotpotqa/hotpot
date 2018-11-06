# HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering

This repository contains the baseline model code, as well as the entire pipeline of running experiments on the HotpotQA dataset,
including data download, data preprocessing, training, and evaluation.

## Requirements

Python 3, pytorch 0.3.0, spacy

To install pytorch 0.3.0, follow the instructions at https://pytorch.org/get-started/previous-versions/ . For example, with
CUDA8 and conda you can do
```
conda install pytorch=0.3.0 cuda80 -c pytorch
```

To install spacy, run
```
conda install spacy
```

## Data Download and Preprocessing

Run the script to download the data, including HotpotQA data and GloVe embeddings, as well as spacy packages.
```
./download.sh
```

There are three HotpotQA files:
- Training set http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
- Dev set in the distractor setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
- Dev set in the fullwiki setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json This is just `hotpot_dev_distractor_v1.json` without the gold paragraphs, but instead with the top 10 paragraphs obtained using our
retrieval system. If you want to use your own IR system (which is encouraged!), you can replace the paragraphs in this json
with your own retrieval results. Please note that the gold paragraphs might or might not be in this json because our IR system
is pretty basic.
- Test set in the fullwiki setting http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json Because in the fullwiki setting, you only need to submit your prediction to our evaluation server without the code, we publish the test set without the answers and supporting facts. The context in the file is paragraphs obtained using our retrieval system, which might or might not contain the gold paragraphs. Again you are encouraged to use your own IR system in this setting --- simply replace the paragraphs in this json with your own retrieval results.


## JSON Format

The top level structure of each JSON file is a list, where each entry represents a question-answer data point. Each data point is
a dict with the following keys:
- `_id`: a unique id for this question-answer data point. This is useful for evaluation.
- `question`: a string.
- `answer`: a string. The test set does not have this key.
- `supporting_facts`: a list. Each entry in the list is a list with two elements `[title, sent_id]`, where `title` denotes the title of the 
paragraph, and `sent_id` denotes the supporting fact's id (0-based) in this paragraph. The test set does not have this key.
- `context`: a list. Each entry is a paragraph, which is represented as a list with two elements `[title, sentences]` and `sentences` is a list
of strings.

There are other keys that are not used in our code, but might be used for other purposes (note that these keys are not present in the test sets, and your model should not rely on these two keys for making preditions on the test sets):
- `type`: either `comparison` or `bridge`, indicating the question type. (See our paper for more details).
- `level`: one of `easy`, `medium`, and `hard`. (See our paper for more details).

## Preprocessing

Preprocess the training and dev sets in the distractor setting:
```
python main.py --mode prepro --data_file hotpot_train_v1.1.json --para_limit 2250 --data_split train
python main.py --mode prepro --data_file hotpot_dev_distractor_v1.json --para_limit 2250 --data_split dev
```

Preprocess the dev set in the full wiki setting:
```
python main.py --mode prepro --data_file hotpot_dev_fullwiki_v1.json --data_split dev --fullwiki --para_limit 2250
```

Note that the training set has to be preprocessed before the dev sets because some vocabulary and embedding files are produced
when the training set is processed.

## Training

Train a model
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0 \ 
--sp_lambda 1.0
```

Our implementation supports running on multiple GPUs. Remove the `CUDA_VISIBLE_DEVICES` variable to run on all GPUs you have
```
python main.py --mode train --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0 --sp_lambda 1.0
```

You will be able to see the perf reach over 58 F1 on the dev set. Record the file name (something like `HOTPOT-20180924-160521`)
which will be used during evaluation.

## Local Evaluation

First, make predictions and save the predictions into a file (replace `--save` with your own file name).
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --data_split dev --para_limit 2250 --batch_size 24 --init_lr 0.1 \ 
--keep_prob 1.0 --sp_lambda 1.0 --save HOTPOT-20180924-160521 --prediction_file dev_distractor_pred.json
```

Then, call the evaluation script:
```
python hotpot_evaluate_v1.py dev_distractor_pred.json hotpot_dev_distractor_v1.json
```

The same procedure can be repeated to evaluate the dev set in the fullwiki setting.
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --data_split dev --para_limit 2250 --batch_size 24 --init_lr 0.1 \ 
--keep_prob 1.0 --sp_lambda 1.0 --save HOTPOT-20180924-160521 --prediction_file dev_fullwiki_pred.json --fullwiki
python hotpot_evaluate_v1.py dev_fullwiki_pred.json hotpot_dev_fullwiki_v1.json
```

## Prediction File Format

The prediction files `dev_distractor_pred.json` and `dev_fullwiki_pred.json` should be JSON files with the following keys:
- `answer`: a dict. Each key of the dict is a QA pair id, corresponding to the field `_id` in data JSON files. Each value of the dict is a string representing the predicted answer.
- `sp`: a dict. Each key of the dict is a QA pair id, corresponding to the field `_id` in data JSON files. Each value of the dict is a list representing the predicted supporting facts. Each entry of the list is a list with two elements `[title, sent_id]`, where `title` denotes the title of the paragraph, and `sent_id` denotes the supporting fact's id (0-based) in this paragraph.

## Model Submission and Test Set Evaluation

We use Codalab for test set evaluation. In the distractor setting, you must submit your code and provide a Docker environment. Your code will run on the test set. In the fullwiki setting, you only need to submit your prediction file. See https://worksheets.codalab.org/worksheets/0xa8718c1a5e9e470e84a7d5fb3ab1dde2/ for detailed instructions.

## License
The HotpotQA dataset is distribued under the [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/legalcode) license.
The code is distribued under the Apache 2.0 license.

## References

The preprocessing part and the data loader are adapted from https://github.com/HKUST-KnowComp/R-Net . The evaluation script is
adapted from https://rajpurkar.github.io/SQuAD-explorer/ .



