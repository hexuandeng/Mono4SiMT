#!/usr/bin/env bash
source ./data_path.sh
TASK=teacher_wmt21_${SRC}${TGT}

pip install sacrebleu[ja]
CUDA_VISIBLE_DEVICES=$1 python -m fairseq_cli.train \
    ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --fp16 --ddp-backend=no_c10d \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --attention-dropout 0.0 --activation-dropout 0.0 --dropout 0.3 \
    --max-tokens 25600 --update-freq 2 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch transformer \
    --lr 1e-3 --lr-scheduler cosine --warmup-init-lr 1e-07 --weight-decay 0.0 \
    --lr-shrink 1 --lr-period-updates 20000 --min-lr 1e-09 \
    --warmup-updates 10000 --clip-norm 0.1 \
    --max-update 50000 \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy \
    --max-source-positions 10000 --max-target-positions 10000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-best-checkpoints 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --log-format simple --log-interval 50 \
    --seed 1 
