#!/bin/bash

while getopts "s:h:d:r:l:t:" optname; do
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
    t)
        TOOL=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

DIR_DATA=/data/yejh/nlp/datasets/GEC/EGEC/conll14

FILE_SRC=${FILE_SRC:-"${DIR_DATA}/conll14st-test.tok.src"}

#FILE_REF=("${DIR_DATA}/conll14st-test.trg0" "${DIR_DATA}/conll14st-test.trg1")

#FILE_REF=${FILE_SRC:-"${DIR_DATA}/conll14st-test.m2"}

#FILE_REF=${FILE_REF:-"${DIR_DATA}/conll14st-test.errant"}


FILE_LOG=${FILE_LOG:-"metrics/logs/temp.txt"}

echo Evaluation Tool: ${TOOL}
echo FILE_SRC: ${FILE_SRC}
echo FILE_REF: ${FILE_REF[*]}

if [[ ! ${DIR_HYP} && ! ${FILE_HYP} ]]; then
    echo "Specify either of DIR_HYP or FILE_HYP"
    exit 1
elif [[ ${DIR_HYP} && ${FILE_HYP} ]]; then
    echo "Specify either of DIR_HYP or FILE_HYP"
    echi 1
elif [[ ${DIR_HYP} && -d ${DIR_HYP} ]]; then
    echo DIR_HYP: ${DIR_HYP}
    LIST_HYP=()
    for FILE_HYP in ${DIR_HYP}/*; do
        LIST_HYP[${#LIST_HYP[@]}]=${FILE_HYP}
    done
elif [[ ${FILE_HYP} && -f ${FILE_HYP} ]]; then
    echo FILE_HYP: ${FILE_HYP}
    LIST_HYP=(${FILE_HYP})
fi

echo LIST_HYP: ${LIST_HYP[@]}

if [ -f ${FILE_LOG} ]; then
    rm -rf ${FILE_LOG}
fi


for FILE_HYP in ${LIST_HYP[@]}; do
    if [ ! -f ${FILE_HYP} ]; then
        continue
    fi

    if [[ ${TOOL} == "errant" ]]; then
        DIR_M2=${FILE_HYP%/*}/errant
        if [ ! -d ${DIR_M2} ]; then
            mkdir ${DIR_M2}
        fi

        FILE_M2=${DIR_M2}/$(basename ${FILE_HYP}).errant
        bash scripts/metrics/errant.sh -s ${FILE_SRC} -h ${FILE_HYP} -r ${FILE_REF} -m ${FILE_M2} -l ${FILE_LOG}

    elif [[ ${TOOL} == "errant-sent" ]]; then
        exit 1

    elif [[ ${TOOL} == "m2" ]]; then
        bash scripts/metrics/m2.sh -d ${DIR_HYP} -r ${FILE_REF} -l ${FILE_LOG}
        exit 1

    elif [[ ${TOOL} == "m2-sent" ]]; then
        bash scripts/metrics/pt_m2.sh -d ${DIR_HYP} -r ${FILE_REF} -l ${FILE_LOG}
        exit 1

    elif [[ ${TOOL} == "gleu" ]]; then
        bash scripts/metrics/gleu.sh -s ${FILE_SRC} -d ${DIR_HYP} -l ${FILE_LOG} -r ${FILE_REF}
        exit 1
        
    elif [[ ${TOOL} == "imeasure" ]]; then
#        bash scripts/metrics/imeasure.sh -s
        exit 1

    else
        echo "Error Tool ${TOOL}"
        exit 1
    fi
done

#wait
#sort -n -r -t ' ' -k 2 ${FILE_LOG} > ${FILE_LOG}.sort
