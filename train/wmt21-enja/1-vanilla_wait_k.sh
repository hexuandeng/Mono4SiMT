source ./data_path.sh

DATA=$1
SUBSET=$2
GPU=$3
if [ ! $3 ]; then
    GPU='0,1,2,3,4,5,6,7'
fi

for i in 1 3 5 7 9; do
    TASK=wait_${i}_${SRC}${TGT}_distill
    if [ "$DATA" == "raw" ]; then
        TASK=wait_${i}_${SRC}${TGT}
    fi
    echo ">> Begin training ${TASK}_${SUBSET}"
    bash 1s-wait_k.sh \
    ${i} ${DATA} ${TASK} ${GPU} \
    ${SUBSET} >> log/${TASK}_${SUBSET}.log
done
