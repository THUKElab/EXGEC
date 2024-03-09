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

FILE_REF=${FILE_REF:-"${DIR_DATA}/conll14st-test.m2.ieval.xml"}

DIR_HYP=${DIR_HYP:-"${DIR_DATA}/CoNLL2014-Systems"}

FILE_LOG=${FILE_LOG:-"metrics/logs/conll2014-imeasure.txt"}

echo FILE_REF: ${FILE_REF[*]}

#python metrics/imeasure/m2_to_ixml.py -in:${FILE_REF} -out:temp.xml

python metrics/imeasure/ieval.py -hyp:${FILE_HYP} -ref:${FILE_REF} -v
#python metrics/imeasure/ieval.py -hyp:${FILE_HYP} -ref:${FILE_REF} | tee -a ${FILE_LOG}

#for FILE_HYP in ${DIR_HYP}/*; do
#    if test -f ${FILE_HYP}; then
#        echo FILE_HYP: ${FILE_HYP}
#        nohup python metrics/imeasure/ieval.py -hyp:${FILE_HYP} -ref:${FILE_REF} | tee -a ${FILE_LOG} &
#    fi
##    exit 1
#done

#python metrics/imeasure/ieval.py \
#    -hyp:scripts/eval/example_imeasure/sys.txt \
#    -ref:scripts/eval/example_imeasure/gold.xml


#----------------------------------------------------------------------------------------------------------
#Hypothesis file    : scripts/eval/example_imeasure/sys.txt
#Gold standard file : scripts/eval/example_imeasure/gold.xml
#Maximising metric  : WACC - CORRECTION
#Optimise for       : SENTENCE
#WAcc weight        : 2.0
#F beta             : 1.0
#----------------------------------------------------------------------------------------------------------
#
#OVERALL RESULTS
#----------------------------------------------------------------------------------------------------------
#Aspect            TP      TN      FP      FN     FPN       P       R  F_1.00     Acc   Acc_b    WAcc  WAcc
#----------------------------------------------------------------------------------------------------------
#Detection          2       5       0       1       0  100.00   66.67   80.00   87.50   62.50   90.00   62.
#Correction         2       5       0       1       0  100.00   66.67   80.00   87.50   62.50   90.00   62.