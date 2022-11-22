source ./data_path.sh

OUTPUT=${DATA}/score/k-anticipation.log
rm ${OUTPUT}
for k in {1..9}; do 
    echo "calculating forward ${k} anticipation:" >> ${OUTPUT}
    python /root/Mono4SiM/utility/simultaneous_translation/eval/anticipation/count_anticipation.py \
    -k $k < ${DATA}/score/forward.align >> ${OUTPUT}
    echo "calculating reverse ${k} anticipation:" >> ${OUTPUT}
    python /root/Mono4SiM/utility/simultaneous_translation/eval/anticipation/count_anticipation.py \
    -k $k < ${DATA}/score/reverse.align >> ${OUTPUT}
    echo "calculating diag ${k} anticipation:" >> ${OUTPUT}
    python /root/Mono4SiM/utility/simultaneous_translation/eval/anticipation/count_anticipation.py \
    -k $k < ${DATA}/score/diag.align >> ${OUTPUT}
done
