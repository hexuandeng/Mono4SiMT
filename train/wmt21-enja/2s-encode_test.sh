source ./data_path.sh

OUTDIR=$1
PREFIX=$1
NAME=$2
TOK=$3
mkdir -p ${OUTDIR}

SRCTOK=$(mktemp)
TGTTOK=$(mktemp)
CORPUS=${OUTDIR}/align_process

vocab=32000
vtype=unigram
spm_encode=$FAIRSEQ/scripts/spm_encode.py

if [ "$TOK" == "true" ]; then
    cat ${PREFIX}/$NAME.$SRC | sed 's/▁//g' > ${SRCTOK}
    cat ${PREFIX}/$NAME.$TGT | sed 's/▁//g' > ${TGTTOK}
else
    SPM_PREFIX=/root/Mono4SiM/data/wmt21-enja/prep/spm_${vtype}${vocab}
    for l in ${SRC} ${TGT}; do
        SPM_MODEL=${SPM_PREFIX}_${l}.model
        echo "Using SPM model $SPM_MODEL"
        if [ -f $ready/$split.$l ]; then
            echo "found $ready/$split.$l, skipping spm_encode"
        else
            echo "spm_encode to $split.$l..."
            python $spm_encode --model=$SPM_MODEL \
                --output_format=piece \
                < ${PREFIX}/$NAME.$l > ${PREFIX}/$l.tok
        fi
    done
    cat ${PREFIX}/${SRC}.tok | sed 's/▁//g' > ${SRCTOK}
    cat ${PREFIX}/${TGT}.tok | sed 's/▁//g' > ${TGTTOK}
fi

paste ${SRCTOK} ${TGTTOK} | sed "s/\t/ ||| /" > ${CORPUS}

rm -f $SRCTOK
rm -f $TGTTOK
rm -f ${PREFIX}/${SRC}.tok
rm -f ${PREFIX}/${TGT}.tok
