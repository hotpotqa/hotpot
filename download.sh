or file_name in hotpot_dev_distractor_v1 hotpot_dev_fullwiki_v1 hotpot_train_v1.1
do
    json_name="$file_name.json"
    tmp_name="$file_name.tmp"
    if [ ! -f $json_name ]; then
        if [ -f $tmp_name ]; then
            rm -f $tmp_name
        fi
        wget http://curtis.ml.cmu.edu/datasets/hotpot/$json_name -O $tmp_name
        mv $tmp_name $json_name
    fi
done

# Download GloVe
GLOVE_DIR=./
mkdir -p $GLOVE_DIR
if [ ! -f $GLOVE_DIR/glove.840B.300d.zip ]; then
    if [ -f $GLOVE_DIR/glove.840B.300d.tmp ]; then
        rm -f $GLOVE_DIR/glove.840B.300d.tmp
    fi
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.tmp
    mv $GLOVE_DIR/glove.840B.300d.tmp $GLOVE_DIR/glove.840B.300d.zip
    unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR
fi

# Download Spacy language models
python3 -m spacy download en
