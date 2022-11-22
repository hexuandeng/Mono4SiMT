source ./data_path.sh

KENLMBIN=/root/Mono4SiM/utility/kenlm/build/bin
CORPUS=${BASE}/ready/train.clean.${TGT}
LMDATA=${BASE}/score
NGRAM=3

mkdir -p ${LMDATA}

export PATH="/root/.local/bin:$PATH"
# estimate ngram
${KENLMBIN}/lmplz -o ${NGRAM} -S 50% < ${CORPUS} > ${LMDATA}/$(basename $BASE)_$TGT.arpa

# binarize
${KENLMBIN}/build_binary -s ${LMDATA}/$(basename $BASE)_$TGT.arpa ${LMDATA}/$(basename $BASE)_$TGT.bin
