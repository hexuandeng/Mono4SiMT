#!/usr/bin/env bash

source ./data_path.sh

WAITK=$1
DATA=$2
TASK=$3
SUBSET=$5

if [ "$DATA" == "raw" ]; then
    DATA=/root/Mono4SiM/data/${DATASET}/data-bin
fi

CUDA_VISIBLE_DEVICES=$4 python -m fairseq_cli.train \
    ${DATA}/${SUBSET} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --max-tokens 25600 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch waitk_transformer \
    --waitk ${WAITK} \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --save-dir checkpoints/${TASK}_${SUBSET} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --fp16 --local_rank $SLURM_LOCALID \
    --seed 2
