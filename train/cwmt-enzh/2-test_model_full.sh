source ./data_path.sh

SPLIT=test
ROOT=/root/Mono4SiM/data/${DATASET}
DATA=${ROOT}/data-bin
SRC_FILE=${ROOT}/prep/test.${SRC}-${TGT}.${SRC}
TGT_FILE=${ROOT}/prep/test.${SRC}-${TGT}.${TGT}.1

GENARGS="--beam 5 --lenpen 1.5 --max-len-a 1.2 --max-len-b 10 --remove-bpe sentencepiece"
EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer zh --sacrebleu-lowercase"
REF=(
    "${BASE}/prep/test.${SRC}-${TGT}.${TGT}.1"
    "${BASE}/prep/test.${SRC}-${TGT}.${TGT}.2"
    "${BASE}/prep/test.${SRC}-${TGT}.${TGT}.3"
)

PORT=11000
TASK=teacher_cwmt_${SRC}${TGT}
CHECKDIR=checkpoints/${TASK}
CHECKPOINT=${CHECKDIR}/checkpoint_best.pt
OUTPUT=${CHECKDIR}/log
mkdir -p ${OUTPUT}

echo "Evaluating ${TASK}!"

AGENT=/root/Mono4SiM/utility/simultaneous_translation/eval/agents/simul_t2t_waitk.py
SPM_PREFIX=${DATA}/spm_unigram32000
WORKERS=2

BLEU_TOK=13a
UNIT=word
if [[ ${TGT} == "zh" ]]; then
    BLEU_TOK=zh
    UNIT=char
    NO_SPACE="--no-space"
fi

simuleval --gpu 0 \
    --agent ${AGENT} \
    --user-dir ${USERDIR} \
    --source ${SRC_FILE} \
    --target ${TGT_FILE} \
    --data-bin ${DATA} \
    --model-path ${CHECKPOINT} \
    --src-splitter-path ${SPM_PREFIX}_${SRC}.model \
    --tgt-splitter-path ${SPM_PREFIX}_${TGT}.model \
    --output ${OUTPUT} \
    --sacrebleu-tokenizer ${BLEU_TOK} \
    --eval-latency-unit ${UNIT} \
    --segment-type ${UNIT} \
    ${NO_SPACE} \
    --scores \
    --full-sentence \
    --port ${PORT} \
    --workers ${WORKERS}

echo "Simuleval ${TASK} finished!"

CUDA_VISIBLE_DEVICES=0 python -m \
    fairseq_cli.generate ${ROOT}/data-bin \
    -s ${SRC} -t ${TGT} \
    --user-dir ${USERDIR} \
    --gen-subset ${SPLIT} \
    --skip-invalid-size-inputs-valid-test \
    --task translation_infer \
    --inference-config-yaml pre_monotonic.yaml \
    --path ${CHECKDIR}/checkpoint_best.pt \
    --max-tokens 4096 --fp16 \
    --results-path ${OUTPUT} \
    ${GENARGS} ${EXTRAARGS}

python 2s-proc_generate.py -r ${OUTPUT} -f generate-${SPLIT}.txt
echo "Generation ${TASK} finished!"

python -m sacrebleu ${REF[@]} -i ${OUTPUT}/detok.txt \
    -m bleu chrf \
    --chrf-lowercase \
    --width 2 \
    --tok zh -lc | tee ${OUTPUT}/sacrebleu.txt

echo "SacreBLEU ${TASK} finished!"
