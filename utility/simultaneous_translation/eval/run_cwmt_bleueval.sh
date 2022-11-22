source ~/utility/sacrebleu/sacrebleu2/bin/activate
SRC=en
TGT=zh
DIR=cwmt_${TGT}-results
WORKERS=2
REF=(
    "/root/Datasets/cwmt-enzh/prep/test.${SRC}-${TGT}.${TGT}.1"
    "/root/Datasets/cwmt-enzh/prep/test.${SRC}-${TGT}.${TGT}.2"
    "/root/Datasets/cwmt-enzh/prep/test.${SRC}-${TGT}.${TGT}.3"
)

# Normal
for DELAY in 1 3 5 7 9; do
    BASELINE="${DIR}/wait_${DELAY}_${SRC}${TGT}_distill.cwmt/prediction"
    SYSTEMS=(
        "${DIR}/wait_${DELAY}_${SRC}${TGT}_mon.cwmt/prediction"
        "${DIR}/wait_${DELAY}_${SRC}${TGT}_reorder.cwmt/prediction"
        "${DIR}/ctc_delay${DELAY}.cwmt/prediction"
        "${DIR}/ctc_delay${DELAY}_mon.cwmt/prediction"
        "${DIR}/ctc_delay${DELAY}_reorder.cwmt/prediction"
        "${DIR}/sinkhorn_delay${DELAY}.cwmt/prediction"
        "${DIR}/sinkhorn_delay${DELAY}_ft.cwmt/prediction"
    )
    OUTPUT=${DIR}/quality-results.cwmt/delay${DELAY}-systems
    mkdir -p $(dirname ${OUTPUT})
    python -m sacrebleu ${REF[@]} -i ${BASELINE} ${SYSTEMS[@]} \
        --paired-jobs ${WORKERS} \
        -m bleu chrf \
        --width 2 \
        --tok zh -lc \
        --chrf-lowercase \
        --paired-bs | tee ${OUTPUT}
done

# Full-sentence
TEACHER="${DIR}/teacher_cwmt_${SRC}${TGT}.cwmt/prediction"
OUTPUT=${DIR}/quality-results.cwmt/full_sentence-systems
mkdir -p $(dirname ${OUTPUT})
python -m sacrebleu ${REF[@]} -i ${TEACHER} \
    --paired-jobs ${WORKERS} \
    -m bleu chrf \
    --width 2 \
    --tok zh -lc \
    --chrf-lowercase \
    --confidence | tee ${OUTPUT}

# # Ablation
# BASELINE="${DIR}/sinkhorn_delay3.cwmt/prediction"
# SYSTEMS=(
#     "${DIR}/sinkhorn_delay3_unittemp.cwmt/prediction"
#     "${DIR}/sinkhorn_delay3_nonoise.cwmt/prediction"
#     "${DIR}/sinkhorn_delay3_softmax.cwmt/prediction"
# )
# OUTPUT=${DIR}/quality-results.cwmt/ablation-systems
# mkdir -p $(dirname ${OUTPUT})
# python -m sacrebleu ${REF[@]} -i ${BASELINE} ${SYSTEMS[@]} \
#     --paired-jobs ${WORKERS} \
#     -m bleu chrf \
#     --width 2 \
#     --tok zh -lc \
#     --chrf-lowercase \
#     --paired-bs | tee ${OUTPUT}

# # Verbose scores
# OUTDIR=${DIR}/quality-results.cwmt/verbose
# mkdir -p ${OUTDIR}
# for DELAY in 1 3 5 7 9; do
#     SYSTEMS=(
#         "sinkhorn_delay${DELAY}"
#         "sinkhorn_delay${DELAY}_ft"
#     )
#     for s in "${SYSTEMS[@]}"; do
#         python -m sacrebleu ${REF[@]} \
#             -i ${DIR}/${s}.cwmt/prediction \
#             -m bleu \
#             --width 2 \
#             --tok zh -lc | tee ${OUTDIR}/${s}
#     done
# done