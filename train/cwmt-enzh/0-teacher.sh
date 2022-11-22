#!/usr/bin/env bash
source ./data_path.sh
TASK=teacher_cwmt_${SRC}${TGT}

CUDA_VISIBLE_DEVICES=$1 python -m fairseq_cli.train \
    ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --max-tokens 25600 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch transformer \
    --encoder-normalize-before --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-best-checkpoints 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --seed 1 \
    --fp16
