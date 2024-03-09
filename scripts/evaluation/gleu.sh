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

DIR_DATA="metrics/data"

FILE_SRC=${FILE_SRC:-"${DIR_DATA}/conll14st-test.tok.src"}
#FILE_REF=${FILE_REF:-"${DIR_DATA}/conll14st-test.trg0" "${DIR_DATA}/conll14st-test.trg1"}
#DIR_HYP=${DIR_HYP:-"${DIR_DATA}/CoNLL2014-Systems"}

FILE_LOG=${FILE_LOG:-"metrics/logs/temp.txt"}

#echo FILE_SRC: ${FILE_SRC}
#echo FILE_REF: ${FILE_REF[*]}

#if [ -f ${FILE_LOG} ]; then
#    rm -rf ${FILE_LOG}
#fi

# Get FILE_REF
if [ -d ${FILE_REF} ]; then
    FILE_REF_LIST=()
    for REF in ${FILE_REF}/*.tgt; do
        FILE_REF_LIST[${#FILE_REF_LIST[@]}]=${REF}
    done
    echo FILE_REF_LIST: ${FILE_REF_LIST[*]}
    FILE_REF=${FILE_REF_LIST[*]}
fi

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


for FILE_HYP in ${LIST_HYP[@]}; do
    if [ ! -f ${FILE_HYP} ]; then
        continue
    fi
    nohup python metrics/gleu/compute_gleu -s ${FILE_SRC} -r ${FILE_REF[*]} -o ${FILE_HYP} -n 4 >&1 \
        | tee -a ${FILE_LOG} &

done



#for FILE_HYP in ${DIR_HYP}/*; do
#    if test -f ${FILE_HYP}; then
#        echo FILE_HYP: ${FILE_HYP}
##        nohup python metrics/GLEU/compute_gleu -s ${FILE_SRC} -r ${FILE_REF[*]} -o ${FILE_HYP} -n 4 >&1 \
##            | tee -a ${FILE_LOG} &
#        nohup python metrics/GLEU/deprecated/compute_gleu -s ${FILE_SRC} -r ${FILE_REF[*]} -o ${FILE_HYP} -n 4 >&1 \
#            | tee -a ${FILE_LOG} &
#    fi
#done

#wait
#sort -n -r -t ' ' -k 2 ${FILE_LOG} > ${FILE_LOG}.sort

#python metrics/GLEU/compute_gleu -s ${FILE_SRC} -r ${FILE_REF[*]} -o ${FILE_HYP} -n 4