source ~/utility/sacrebleu/sacrebleu2/bin/activate
SRC=de
TGT=en
DIR=wmt15_${TGT}-results
WORKERS=2
REF=(
    "/media/XXXX-2/Data/wmt15/de-en/prep/test.${TGT}"
)

# Normal
for DELAY in 1 3 5 7 9; do
    BASELINE="${DIR}/wait_${DELAY}_${SRC}${TGT}_distill.wmt15/prediction"
    SYSTEMS=(
        "${DIR}/wait_${DELAY}_${SRC}${TGT}_mon.wmt15/prediction"
        "${DIR}/wait_${DELAY}_${SRC}${TGT}_reorder.wmt15/prediction"
        "${DIR}/ctc_delay${DELAY}.wmt15/prediction"
        "${DIR}/ctc_delay${DELAY}_mon.wmt15/prediction"
        "${DIR}/ctc_delay${DELAY}_reorder.wmt15/prediction"
        "${DIR}/sinkhorn_delay${DELAY}.wmt15/prediction"
        "${DIR}/sinkhorn_delay${DELAY}_ft.wmt15/prediction"
    )

    OUTPUT=${DIR}/quality-results.wmt15/delay${DELAY}-systems
    mkdir -p $(dirname ${OUTPUT})
    python -m sacrebleu ${REF[@]} -i ${BASELINE} ${SYSTEMS[@]} \
        --paired-jobs ${WORKERS} \
        -m bleu chrf \
        --width 2 \
        --tok 13a -lc \
        --chrf-lowercase \
        --paired-bs | tee ${OUTPUT}
done

# Full-sentence
TEACHER="${DIR}/teacher_wmt15_${SRC}${TGT}.wmt15/prediction"
OUTPUT=${DIR}/quality-results.wmt15/full_sentence-system
mkdir -p $(dirname ${OUTPUT})
python -m sacrebleu ${REF[@]} -i ${TEACHER} \
    --paired-jobs ${WORKERS} \
    -m bleu chrf \
    --width 2 \
    --tok 13a -lc \
    --chrf-lowercase \
    --confidence | tee ${OUTPUT}