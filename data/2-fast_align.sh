source ./data_path.sh

OUTDIR=${DATA}/score
PREFIX=${DATA}/ready/train.clean
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
./fast_align -i ${CORPUS} -d -o -v > ${OUTDIR}/forward.align
./fast_align -i ${CORPUS} -d -o -v -r > ${OUTDIR}/reverse.align
./atools -i ${OUTDIR}/forward.align -j ${OUTDIR}/reverse.align -c grow-diag-final-and > ${OUTDIR}/diag.align
