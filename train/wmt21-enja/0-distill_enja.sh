#!/usr/bin/env bash
source ./data_path.sh
TASK=teacher_wmt21
CHECKDIR=/root/Mono4SiM/train/checkpoints/${TASK}_${SRC}${TGT}
AVG=true

GENARGS="--beam 6 --lenpen 1.0 --max-len-a 1.2 --max-len-b 10 --remove-bpe sentencepiece"
EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer ja-mecab --sacrebleu-lowercase"

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  python /root/Mono4SiM/utility/scripts/average_checkpoints.py \
    --inputs ${CHECKDIR} --num-best-checkpoints 5 \
    --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

ROOT=/root/Mono4SiM/generate/${TASK}
FILES=/root/Mono4SiM/data/${DATASET}
mkdir -p ${ROOT}
mkdir -p ${ROOT}/interactive
mkdir -p ${FILES}/split

split -l $(($((`wc -l < ${FILES}/ready/train.clean.en`/8))+1)) -d -a 1 ${FILES}/ready/train.clean.en ${FILES}/split/train.en.

for i in {0..7}
do
  cat ${FILES}/split/train.en.${i} | CUDA_VISIBLE_DEVICES=${i} \
  python -m fairseq_cli.interactive /root/Mono4SiM/data/${DATASET}/data-bin \
    -s ${SRC} -t ${TGT} \
    --user-dir ${USERDIR} \
    --skip-invalid-size-inputs-valid-test \
    --task translation \
    --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
    --batch-size 64 --buffer-size 128 --fp16 \
    ${GENARGS} ${EXTRAARGS} > ${ROOT}/interactive/generate-train.${i}.txt 2>&1 &
done
