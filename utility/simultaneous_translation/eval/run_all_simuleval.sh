DATA=cwmt
TGT=zh
EXP=../expcwmt
SRC_FILE=/media/XXXX-2/Data/cwmt/zh-en/prep/test.en-zh.en
TGT_FILE=/media/XXXX-2/Data/cwmt/zh-en/prep/test.en-zh.zh.1

for t in 2 3; do
    for k in 1 3 5 7 9; do
        MODEL=sinkhorn_delay${k}_ft
        bash simuleval.sh \
            -a agents/simul_t2t_ctc.py \
            -m ${MODEL} \
            -k ${k} \
            -e ${EXP} \
            -s ${SRC_FILE} \
            -t ${TGT_FILE}

        OUTPUT=${DATA}_${TGT}-results/${MODEL}.${DATA}
        mv ${OUTPUT}/scores ${OUTPUT}/scores.${t}
    done
done
bash run_cwmt_bleueval.sh


DATA=wmt15
TGT=en
EXP=../expwmt15
SRC_FILE=/media/XXXX-2/Data/wmt15/de-en/prep/test.de
TGT_FILE=/media/XXXX-2/Data/wmt15/de-en/prep/test.en

for t in 1 2 3; do
    for k in 1 3 5 7 9; do
        MODEL=sinkhorn_delay${k}_ft
        bash simuleval.sh \
            -a agents/simul_t2t_ctc.py \
            -m ${MODEL} \
            -k ${k} \
            -e ${EXP} \
            -s ${SRC_FILE} \
            -t ${TGT_FILE}

        OUTPUT=${DATA}_${TGT}-results/${MODEL}.${DATA}
        mv ${OUTPUT}/scores ${OUTPUT}/scores.${t}
    done
done
bash run_wmt15_bleueval.sh