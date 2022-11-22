#!/usr/bin/env bash
ALIGNOUT=$1
ALIGNOUT_R=$2
cd /root/Mono4SiM/utility/scripts
for k in {1..9}; do 
    if [ -f "${ALIGNOUT}" ]; then
        echo "calculating $k anticipation"
        python count_anticipation.py -k $k < ${ALIGNOUT}
    elif [ -f "${ALIGNOUT_R}" ]; then
        echo "calculating $k anticipation rev"
        python count_anticipation.py -k $k -r < ${ALIGNOUT_R}
    fi
done
