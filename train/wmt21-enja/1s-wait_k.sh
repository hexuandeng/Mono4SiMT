#!/usr/bin/env bash

source ./data_path.sh

WAITK=$1
DATA=$2
TASK=$3
SUBSET=$5

if [ "$DATA" == "raw" ]; then
    DATA=/root/Mono4SiM/data/${DATASET}/data-bin
fi

pip install sacrebleu[ja]
CUDA_VISIBLE_DEVICES=$4 python -m fairseq_cli.train \
    ${DATA}/${SUBSET} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --fp16 --ddp-backend=no_c10d \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch waitk_transformer \
    --waitk ${WAITK} \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --attention-dropout 0.0 --activation-dropout 0.0 --dropout 0.3 \
    --max-tokens 25600 --update-freq 2 \
    --lr 1e-3 --lr-scheduler cosine --warmup-init-lr 1e-07 --weight-decay 0.0 \
    --lr-shrink 1 --lr-period-updates 20000 --min-lr 1e-09 \
    --warmup-updates 10000 --clip-norm 0.1 \
    --max-update 50000 \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy \
    --max-source-positions 10000 --max-target-positions 10000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --save-dir checkpoints/${TASK}_${SUBSET} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --log-format simple --log-interval 50 \
    --seed 1
