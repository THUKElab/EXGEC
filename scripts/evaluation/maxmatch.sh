#!/bin/bash

while getopts "s:h:d:r:l:" optname; do
    case $optname in
    s)
        FILE_SRC=${OPTARG};;
    h)
        FILE_HYP=${OPTARG};;
    d)
        DIR_HYP=${OPTARG};;
    r)
        FILE_REF=${OPTARG};;
    l)
        FILE_LOG=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

DIR_DATA="/data/yejh/nlp/datasets/GEC/EGEC/conll14/"

DIR_HYP=${DIR_HYP:-"${DIR_DATA}/CoNLL2014-Systems"}

FILE_REF=${FILE_REF:-"${DIR_DATA}/conll14st-test.m2"}

FILE_LOG=${FILE_LOG:-"metrics/logs/temp.txt"}

echo FILE_REF: ${FILE_REF[*]}

#if [ -f ${FILE_LOG} ]; then
#    rm -rf ${FILE_LOG}
#fi

for FILE_HYP in ${DIR_HYP}/*; do
    if test -f ${FILE_HYP}; then
        echo FILE_HYP: ${FILE_HYP}
        nohup python metrics/M2/m2scorer.py ${FILE_HYP} ${FILE_REF} 2>&1 | tee -a ${FILE_LOG} &
    fi
done

wait
sort -n -r -t ' ' -k 2 ${FILE_LOG} > ${FILE_LOG}.sort

#python metrics/M2/m2scorer.py scripts/eval/example/system scripts/eval/example/source_gold
#python metrics/M2/m2scorer.py ${DIR_DATA}/CoNLL2014-Systems/AMU ${DIR_DATA}/conll14st-test.m2