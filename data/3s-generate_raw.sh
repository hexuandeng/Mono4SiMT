SCORE=$1
folder=$2
SRC=$3
TGT=$4
ROOT=$5
BASE=$6
ADD=$7
ADDD=$8
RAW=$9
workers=4
bin=${ROOT}/data-bin/${SCORE}_${folder}${RAW}
SPM_PREFIX=${BASE}/prep/spm_unigram32000
ready=${BASE}/ready


cd ${ROOT}
cat ${ROOT}/${folder}/${SCORE}.${SRC} ${ADD}.${SRC} ${ADDD}.${SRC} > ${ROOT}/${folder}/${SCORE}_tmp.${SRC}
cat ${ROOT}/${folder}/${SCORE}.${TGT} ${ADD}.${TGT} ${ADDD}.${TGT} > ${ROOT}/${folder}/${SCORE}_tmp.${TGT}

python -m fairseq_cli.preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${ROOT}/${folder}/${SCORE}_tmp \
    --validpref ${ready}/valid \
    --testpref ${ready}/test \
    --destdir ${bin} \
    --workers ${workers} \
    --srcdict ${SPM_PREFIX}_${SRC}.txt \
    --tgtdict ${SPM_PREFIX}_${TGT}.txt

rm ${ROOT}/${folder}/${SCORE}_tmp.${SRC}
rm ${ROOT}/${folder}/${SCORE}_tmp.${TGT}
