#!/usr/bin/env bash
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    -e|--expdir)
      EXP="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--source)
      SRC_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--target)
      TGT_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

AGENT=./agents/simul_t2t_waitk.py
source ${EXP}/data_path.sh

CHECKPOINT=${EXP}/checkpoints/${MODEL}/checkpoint_best.pt
SPM_PREFIX=${DATA}/spm_unigram32000

PORT=23451
WORKERS=2
BLEU_TOK=13a
UNIT=word
DATANAME=$(basename $(dirname $(dirname ${DATA})))
OUTPUT=${DATANAME}_${TGT}-results/${MODEL}.${DATANAME}
mkdir -p ${OUTPUT}

if [[ ${TGT} == "zh" ]]; then
  BLEU_TOK=zh
  UNIT=char
  NO_SPACE="--no-space"
fi

simuleval \
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
