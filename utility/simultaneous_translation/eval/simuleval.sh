#!/usr/bin/env bash
# credits: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -a|--agent)
      AGENT="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    -k|--waitk)
      WAITK="$2"
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
    -l|--lm-path)
      KENLM="$2"
      shift # past argument
      shift # past value
      ;;
    -w|--lm-weight)
      LM_WEIGHT="$2"
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


source ${EXP}/data_path.sh

CHECKPOINT=${EXP}/checkpoints/${MODEL}/checkpoint_best.pt
SPM_PREFIX=${DATA}/spm_unigram32000

PORT=12345
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
  --incremental-encoder \
  --sacrebleu-tokenizer ${BLEU_TOK} \
  --eval-latency-unit ${UNIT} \
  --segment-type ${UNIT} \
  ${NO_SPACE} \
  --scores \
  --test-waitk ${WAITK} \
  --port ${PORT} \
  --lm-path ${KENLM} \
  --lm-weight ${LM_WEIGHT} \
  --workers ${WORKERS}
