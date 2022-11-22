export SRC=en
export TGT=ja
export DATASET=wmt21-${SRC}${TGT}
export BASE=/root/Mono4SiM/data/${DATASET}
export DATA=/root/Mono4SiM/data/teacher_wmt21_mono/data-bin
export FAIRSEQ=/root/Mono4SiM/utility/fairseq
export USERDIR=/root/Mono4SiM/utility/simultaneous_translation
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
# . ~/envs/apex/bin/activate
