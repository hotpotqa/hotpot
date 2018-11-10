
# Download Hotpot Data
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

# Download GloVe
GLOVE_DIR=./
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

# Download Spacy language models
python3 -m spacy download en
