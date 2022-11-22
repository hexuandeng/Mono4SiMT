source ./data_path.sh

OUTDIR=${BASE}/score
PREFIX=${BASE}/ready/train.clean
mkdir -p ${OUTDIR}

SRCTOK=$(mktemp)
TGTTOK=$(mktemp)
CORPUS=${OUTDIR}/align_process.${SRC}-${TGT}

cat ${PREFIX}.${SRC} | sed 's/▁//g' > ${SRCTOK}
cat ${PREFIX}.${TGT} | sed 's/▁//g' > ${TGTTOK}

paste ${SRCTOK} ${TGTTOK} | sed "s/\t/ ||| /" > ${CORPUS}

rm -f $SRCTOK
rm -f $TGTTOK

cd /root/Mono4SiM/utility/fast_align/build
./fast_align -i ${CORPUS} -d -v -o -p fwd_params >${OUTDIR}/fwd_align 2>${OUTDIR}/fwd_err
./fast_align -i ${CORPUS} -r -d -v -o -p rev_params >${OUTDIR}/rev_align 2>${OUTDIR}/rev_err
