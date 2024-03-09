#!/bin/bash

while getopts "g:m:v:f:s:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    v)
        VALID_NAME=${OPTARG};;
    f)
        EXPLANATION_FORMAT=${OPTARG};;
    s)
        EXPLANATION_SETTING=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

EXPLANATION_SETTING=${EXPLANATION_SETTING:-""}

# Generate Hypothesis
for PATH_MODEL in ${DIR_MODEL}/checkpoint*.pt; do
    bash scripts/fairseq/eng/explainable_bart/predict.sh \
        -g ${GPU_list} \
        -m ${DIR_MODEL} \
        -n $(basename $PATH_MODEL) \
        -v ${VALID_NAME} \
        -f "${EXPLANATION_FORMAT}" \
        -s "${EXPLANATION_SETTING}"
done
