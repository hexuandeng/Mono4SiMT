export SRC=en
export TGT=zh
export DATASET=cwmt-${SRC}${TGT}
export BASE=/root/Mono4SiM/data/${DATASET}
export DATA=/root/Mono4SiM/generate/teacher_cwmt_mono/data-bin
export FAIRSEQ=/root/Mono4SiM/utility/fairseq
export USERDIR=/root/Mono4SiM/utility/simultaneous_translation
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
# . ~/envs/apex/bin/activate
