#!/usr/bin/env bash
MODEL=bert-base-multilingual-cased
PREFIX=$1
SRC=$2
TGT=$3
N=1000000

OUTDIR=./alignments

SRCTOK=$(mktemp)
TGTTOK=$(mktemp)
CORPUS=$(mktemp)

head -n ${N} ${PREFIX}.${SRC} | sed 's/▁//g' > ${SRCTOK}
head -n ${N} ${PREFIX}.${TGT} | sed 's/▁//g' > ${TGTTOK}

echo "aligning ..."
mkdir -p ${OUTDIR}
ALIGNOUT=${OUTDIR}/$(basename ${PREFIX}).${SRC}-${TGT}_${N}
ALIGNOUT_R=${OUTDIR}/$(basename ${PREFIX}).${TGT}-${SRC}_${N}
if [ -f "${ALIGNOUT}" ]; then
	echo "${ALIGNOUT} exists, skipping alignment"
elif [ -f "${ALIGNOUT_R}" ]; then
    echo "${ALIGNOUT_R} exists, skipping alignment"
else
	paste ${SRCTOK} ${TGTTOK} | sed "s/\t/ ||| /" > ${CORPUS}
	python -m awesome_align.run_align \
    --output_file=${ALIGNOUT} \
    --model_name_or_path=${MODEL} \
    --data_file=${CORPUS} \
    --extraction 'softmax' \
    --batch_size 128
fi

echo "calculating anticipation"
for k in {1..9}; do 
    if [ -f "${ALIGNOUT}" ]; then
        python count_anticipation.py -k $k < ${ALIGNOUT}
    elif [ -f "${ALIGNOUT_R}" ]; then
        python count_anticipation.py -k $k -r < ${ALIGNOUT_R}
    fi
done


rm -f $SRCTOK
rm -f $TGTTOK
rm -f $CORPUS