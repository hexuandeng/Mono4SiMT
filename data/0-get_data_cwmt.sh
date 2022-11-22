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
    "cwmt/parallel/casia2015/casia2015/casia2015"
    "cwmt/parallel/casict2011/casict2011/casict-A"
    "cwmt/parallel/casict2011/casict2011/casict-B"
    "cwmt/parallel/casict2015/casict2015/casict2015"
    "cwmt/parallel/neu2017/neu2017/NEU"
)

orig=${BASE}/orig
prep=${BASE}/prep
ready=${BASE}/ready
bin=${BASE}/data-bin
mkdir -p $orig $prep $ready $bin

echo "downloading data"
cd $orig

file=cwmt-data.zip
if [ -f $file ]; then
    echo "$file already exists, skipping download"
else
    kaggle datasets download -d warmth/cwmt-data
    if [ -f $file ]; then
        echo "$file successfully downloaded."
    else
        echo "$file not successfully downloaded."
        exit -1
    fi
    unzip $file
fi
cd ..

echo "pre-processing train data..."
for l in ${SRC} ${TGT}; do
    rm -f $prep/train.dirty.$l
    for f in "${CORPORA[@]}"; do
        if [ "$l" == "zh" ]; then
            if [[ "$f" == *"NEU"* ]]; then
                t="_cn.txt"
            else
                t="_ch.txt"
            fi
        else
            t="_en.txt"
        fi

        echo "precprocess train $f$t"
        cat $orig/$f$t | \
            perl $REM_NON_PRINT_CHAR | \
            perl $LC >> $prep/train.dirty.$l
    done
done

echo "pre-processing valid data..."
for l in ${SRC} ${TGT}; do
    if [ "$l" == "zh" ]; then
        DEV=(
            "$orig/cwmt/dev/NJU-newsdev2018-zhen/NJU-newsdev2018-zhen/CWMT2017-ce-news-test-src.xml"
            "$orig/cwmt/dev/NJU-newsdev2018-enzh/NJU-newsdev2018-enzh/CWMT2017-ec-news-test-ref.xml"
        )
    else
        DEV=(
            "$orig/cwmt/dev/NJU-newsdev2018-zhen/NJU-newsdev2018-zhen/CWMT2017-ce-news-test-ref.xml"
            "$orig/cwmt/dev/NJU-newsdev2018-enzh/NJU-newsdev2018-enzh/CWMT2017-ec-news-test-src.xml"
        )
    fi

    rm -f $prep/valid.dirty.$l
    for f in "${DEV[@]}"; do
        echo "precprocess valid $f"
        grep '<seg id' $f | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
            perl $REM_NON_PRINT_CHAR | \
            perl $LC >> $prep/valid.dirty.$l
    done
done

# testset en -> zh
rm -f $prep/test.*
for y in 2008 2009 2011; do
    tail +2 $orig/cwmt${y}_ec_news.tsv | cut -f6 | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC >> $prep/test.en-zh.en
    for c in 1 2 3; do
        tail +2 $orig/cwmt${y}_ec_news.tsv | cut -f$(($c+6)) | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC >> $prep/test.en-zh.zh.$c
    done
done

# zh -> en
for y in 2008 2009; do
    tail +2 $orig/cwmt${y}_ce_news.tsv | cut -f6 | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC > $prep/test.zh-en.zh
    for c in 1 2 3; do
        tail +2 $orig/cwmt${y}_ce_news.tsv | cut -f$(($c+6)) | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC > $prep/test.zh-en.en.$c
    done
done

cp $prep/test.${SRC}-${TGT}.${SRC} $prep/test.${SRC}
cp $prep/test.${SRC}-${TGT}.${TGT}.1 $prep/test.${TGT}        

# filter empty pairs
perl $CLEAN -ratio 1000 $prep/train.dirty ${SRC} ${TGT} $prep/train 1 10000
perl $CLEAN -ratio 1000 $prep/valid.dirty ${SRC} ${TGT} $prep/valid 1 10000

# SPM
SPM_PREFIX=$prep/spm_${vtype}${vocab}
for l in ${SRC} ${TGT}; do
    SPM_MODEL=${SPM_PREFIX}_${l}.model
    DICT=${SPM_PREFIX}_${l}.txt
    BPE_TRAIN=$prep/bpe-train.$l

    if [[ ! -f $SPM_MODEL ]]; then
        if [ -f $BPE_TRAIN ]; then
            echo "$BPE_TRAIN found, skipping concat."
        else
            train=$prep/train.$l
            default=1000000
            total=$(cat $train | wc -l)
            echo "lang $l total: $total."
            if [ "$total" -gt "$default" ]; then
                cat $train | \
                shuf -r -n $default >> $BPE_TRAIN
            else
                cat $train >> $BPE_TRAIN
            fi     
        fi               

        echo "spm_train on $BPE_TRAIN..."
        ccvg=1.0
        if [[ ${l} == "zh" ]]; then
            ccvg=0.9995
        fi
        python $spm_train --input=$BPE_TRAIN \
            --model_prefix=${SPM_PREFIX}_${l} \
            --vocab_size=$vocab \
            --character_coverage=$ccvg \
            --model_type=$vtype \
            --normalization_rule_name=nmt_nfkc_cf
        
        cut -f1 ${SPM_PREFIX}_${l}.vocab | tail -n +4 | sed "s/$/ 100/g" > $DICT
        cp $SPM_MODEL $bin/$(basename $SPM_MODEL)
        cp $DICT $bin/$(basename $DICT)
    fi

    echo "Using SPM model $SPM_MODEL"
    for split in train valid test; do
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
perl $CLEAN -ratio 9 $ready/train ${SRC} ${TGT} $ready/train.clean 1 1024

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
