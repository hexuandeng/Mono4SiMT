#!/usr/bin/env bash
# Adapted from https://github.com/pytorch/fairseq/blob/simulastsharedtask/examples/translation/prepare-iwslt14.sh
source ./data_path.sh
SCRIPTS=/root/Mono4SiM/utility/mosesdecoder/scripts
LC_ALL=en_US.UTF-8
LANG=en_US.UTF-8
# source ~/envs/apex/bin/activate

DATA=$1
SRC=$3
TGT=$4
vocab=32000
vtype=unigram
workers=4
SPM_PREFIX=$2

# TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
# NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LC=$SCRIPTS/tokenizer/lowercase.perl

spm_train=$FAIRSEQ/scripts/spm_train.py
spm_encode=$FAIRSEQ/scripts/spm_encode.py

prep=${DATA}/prep
ready=${DATA}/ready
bin=${DATA}/data-bin
mkdir -p $prep $ready $bin

echo "pre-processing train data..."
for l in ${SRC} ${TGT}; do
    rm -f $prep/train.dirty.$l
    cat ${DATA}/interactive/detok.$l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC >> $prep/train.dirty.$l
done

# filter empty pairs
perl $CLEAN -ratio 1000 $prep/train.dirty ${SRC} ${TGT} $prep/train 1 10000

# SPM
for l in ${SRC} ${TGT}; do
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

# filter ratio and maxlen < 256
perl $CLEAN -ratio 9 $ready/train ${SRC} ${TGT} $ready/train.clean 1 256

if [[ $5 == "true" ]]; then
    python -m fairseq_cli.preprocess \
        --source-lang ${SRC} \
        --target-lang ${TGT} \
        --trainpref ${ready}/train.clean \
        --validpref ${ready}/valid \
        --testpref ${ready}/test \
        --destdir ${bin} \
        --workers ${workers} \
        --srcdict ${SPM_PREFIX}_${SRC}.txt \
        --tgtdict ${SPM_PREFIX}_${TGT}.txt
fi
