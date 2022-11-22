#!/usr/bin/env bash
SPLIT=test
EXP=/home/XXXX-2/Projects/sinkhorn-simultrans/expcwmt
SRC=en
TGT=zh
DATA=/media/XXXX-2/Data/cwmt/zh-en/data-bin
FAIRSEQ=/home/XXXX-2/Projects/sinkhorn-simultrans/fairseq
USERDIR=`realpath ../../simultaneous_translation`
export PYTHONPATH="$USERDIR:$FAIRSEQ:$PYTHONPATH"

function run_oracle () {
    CHECKDIR=${EXP}/checkpoints/${1}
    DATANAME=$(basename $(dirname $(dirname ${DATA})))
    OUTPUT=${DATANAME}_${TGT}-results/${1}
    AVG=false
    BLEU_TOK=13a
    WORKERS=2

    if [[ ${TGT} == "zh" ]]; then
        BLEU_TOK=zh
    fi
    GENARGS="--beam 1 --remove-bpe sentencepiece"

    if [[ $AVG == "true" ]]; then
        CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
        python ../scripts/average_checkpoints.py \
            --inputs ${CHECKDIR} --num-best-checkpoints 5 \
            --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
    else
        CHECKPOINT_FILENAME=checkpoint_best.pt
    fi

    # python -m fairseq_cli.generate ${DATA} \
    python generate.py ${DATA} \
        --user-dir ${USERDIR} \
        -s ${SRC} -t ${TGT} \
        --gen-subset ${SPLIT} \
        --task translation_infer \
        --max-tokens 8000 --fp16 \
        --inference-config-yaml ../exp/infer_mt.yaml \
        --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
        --model-overrides '{"load_pretrained_encoder_from": None}' \
        --results-path ${OUTPUT} \
        ${GENARGS}

    grep -E "D-[0-9]+" ${OUTPUT}/generate-${SPLIT}.txt | \
        sed "s/^D-//" | \
        sort -k1 -n | \
        cut -f3 > ${OUTPUT}/oracle_prediction

    REF=(
        "/media/XXXX-2/Data/cwmt/zh-en/prep/test.${SRC}-${TGT}.${TGT}.1"
        "/media/XXXX-2/Data/cwmt/zh-en/prep/test.${SRC}-${TGT}.${TGT}.2"
        "/media/XXXX-2/Data/cwmt/zh-en/prep/test.${SRC}-${TGT}.${TGT}.3"
    )
    SYSTEMS=(
        # "../${OUTPUT}/prediction"
        "${OUTPUT}/oracle_prediction"
    )

    python -m sacrebleu ${REF[@]} -i ${SYSTEMS[@]} \
        -m bleu \
        --width 2 \
        --tok zh -lc | tee ${OUTPUT}/score
}

for k in 1 3 5 7 9; do
    run_oracle sinkhorn_delay${k}_ft
done