KENLMBIN=/home/XXXX-2/utility/kenlm/build/bin
FAIRSEQ=/home/XXXX-2/utility/fairseq
DATA=/media/XXXX-2/Data/cwmt/en-zh
TGT=zh
CORPUS=${DATA}/prep/train.dirty.${TGT}
SPM_MODEL=${DATA}/prep/spm_unigram32000_zh.model
LMDATA=./
NGRAM=3

mkdir -p ${LMDATA}

# split bpe
if [ -f "${LMDATA}/corpus" ]; then
    echo "${LMDATA}/corpus exists. skipping spm_encode."
else 
    python ${FAIRSEQ}/scripts/spm_encode.py \
        --model=$SPM_MODEL \
        --output_format=piece \
        < ${CORPUS} > ${LMDATA}/corpus
fi

# estimate ngram
${KENLMBIN}/lmplz -o ${NGRAM} -S 50% < ${LMDATA}/corpus > ${LMDATA}/lm.arpa

# binarize
${KENLMBIN}/build_binary -s ${LMDATA}/lm.arpa ${LMDATA}/lm.bin