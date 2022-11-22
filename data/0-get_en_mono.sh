#!/usr/bin/env bash
# Adapted from https://github.com/pytorch/fairseq/blob/simulastsharedtask/examples/translation/prepare-iwslt14.sh
source ./data_path.sh
SCRIPTS=/root/Mono4SiM/utility/mosesdecoder/scripts
# source ~/envs/apex/bin/activate

vocab=32000
vtype=unigram
workers=4

# TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
# NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LC=$SCRIPTS/tokenizer/lowercase.perl

spm_train=$FAIRSEQ/scripts/spm_train.py
spm_encode=$FAIRSEQ/scripts/spm_encode.py

CORPORA=(
    "news.2017.en.shuffled.deduped"
    "news.2016.en.shuffled"
)

orig=${MONO}/orig
prep=${MONO}/prep
ready=${MONO}/ready
mkdir -p $orig $prep $ready

echo "downloading data"
cd $orig

wget http://data.statmt.org/wmt17/translation-task/news.2016.en.shuffled.gz
wget http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz
gzip -dk news.2016.en.shuffled.gz
gzip -dk news.2017.en.shuffled.deduped.gz
cd ..

echo "pre-processing train data..."
for l in ${SRC}; do
    rm -f $prep/train.dirty.$l
    for f in "${CORPORA[@]}"; do
        echo "precprocess train $f"
        cat $orig/$f | \
            perl $REM_NON_PRINT_CHAR | \
            perl $LC >> $prep/train.dirty.$l
    done
done

# filter empty pairs
perl $CLEAN -ratio 1000 $prep/train.dirty ${SRC} ${SRC} $prep/train 1 10000

# SPM
SPM_PREFIX=/root/Mono4SiM/data/cwmt-enzh/prep/spm_${vtype}${vocab}
for l in ${SRC}; do
    SPM_MODEL=${SPM_PREFIX}_${l}.model
    echo "Using SPM model $SPM_MODEL"
    for split in train; do
        if [ -f $ready/$split.$l ]; then
            echo "found $ready/$split.$l, skipping spm_encode"
        else
            echo "spm_encode to $split.$l..."
            python $spm_encode --model=$SPM_MODEL \
                --output_format=piece \
                < $prep/$split.$l > $ready/$split.$l
        fi
    done
done

# filter ratio and maxlen < 1024
perl $CLEAN -ratio 9 $ready/train ${SRC} ${SRC} $ready/train.clean 1 1024
