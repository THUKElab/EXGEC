#!/bin/bash

while getopts "b:i:o:s:" optname; do
    case $optname in
    b)
        DIR_BPE=${OPTARG};;
    i)
        DIR_INPUT=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
    s)
        SUFFIX=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

DIR_WORKING="models/fairseq/explainable_bart"

DIR_BPE=${DIR_BPE:-"${DIR_WORKING}/preprocess/eng"}

DIR_INPUT=${DIR_INPUT:-"../datasets/GEC/EGEC/expect/denoise"}

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_WORKING}/preprocess/eng/expect_denoise"}

SUFFIX=${SUFFIX:-"denoise"}

mkdir -p ${DIR_OUTPUT}

# BPE for original datasets
#for SPLIT in "train" "valid" "test"; do
#    if [ ! -f "${DIR_OUTPUT}/${SPLIT}/${SPLIT}.bpe.src" ]; then
#        echo "Apply BPE: ${DIR_INPUT}/${SPLIT}.json -> ${DIR_OUTPUT}/${SPLIT}"
#        mkdir -p ${DIR_OUTPUT}/${SPLIT}
#        python ${DIR_WORKING}/preprocess/eng/explanation_preprocess.py \
#            --encoder-json "${DIR_BPE}/encoder.json" \
#            --vocab-bpe "${DIR_BPE}/vocab.bpe" \
#            --input "${DIR_INPUT}/${SPLIT}.json" \
#            --source-output "${DIR_OUTPUT}/${SPLIT}/${SPLIT}.bpe.src" \
#            --target-output "${DIR_OUTPUT}/${SPLIT}/${SPLIT}.bpe.tgt" \
#            --explanation-output "${DIR_OUTPUT}/${SPLIT}/${SPLIT}.bpe.exp" \
#            --workers 32 \
#            --keep-empty
#    fi
#done

# BPE for denoise datasets
for SPLIT in "train" "valid" "test"; do
    if [ ! -f "${DIR_OUTPUT}/${SPLIT}/${SPLIT}.bpe.src" ]; then
        echo "Apply BPE: ${DIR_INPUT}/${SPLIT}/expect_${SPLIT}_${SUFFIX}.json -> ${DIR_OUTPUT}/${SPLIT}"
        mkdir -p ${DIR_OUTPUT}/${SPLIT}
        python ${DIR_WORKING}/preprocess/eng/explanation_preprocess_denoise.py \
            --encoder-json "${DIR_BPE}/encoder.json" \
            --vocab-bpe "${DIR_BPE}/vocab.bpe" \
            --input "${DIR_INPUT}/${SPLIT}/expect_${SPLIT}_${SUFFIX}.json" \
            --source-output "${DIR_OUTPUT}/${SPLIT}/${SPLIT}.bpe.src" \
            --target-output "${DIR_OUTPUT}/${SPLIT}/${SPLIT}.bpe.tgt" \
            --explanation-output "${DIR_OUTPUT}/${SPLIT}/${SPLIT}.bpe.exp" \
            --workers 32 \
            --keep-empty
    fi
done

# Fairseq Preprocess
#python ${DIR_WORKING}/preprocess/eng/preprocess.py \
#    --source-lang "src" \
#    --target-lang "tgt" \
#    --trainpref "${DIR_OUTPUT}/valid/valid.bpe" \
#    --validpref "${DIR_OUTPUT}/valid/valid.bpe" \
#    --destdir "${DIR_OUTPUT}/valid/bin/" \
#    --workers 32 \
#    --srcdict "${DIR_BPE}/dict.txt" \
#    --tgtdict "${DIR_BPE}/dict.txt"

if [ ! -d "${DIR_OUTPUT}/bin" ]; then
    python ${DIR_WORKING}/preprocess/eng/preprocess.py \
        --source-lang "src" \
        --target-lang "tgt" \
        --trainpref "${DIR_OUTPUT}/train/train.bpe" \
        --validpref "${DIR_OUTPUT}/valid/valid.bpe" \
        --destdir "${DIR_OUTPUT}/train/bin/" \
        --workers 32 \
        --srcdict "${DIR_BPE}/dict.txt" \
        --tgtdict "${DIR_BPE}/dict.txt"
fi
