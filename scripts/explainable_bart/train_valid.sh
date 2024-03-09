#!/bin/bash

while getopts "g:i:b:m:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    i)
        DIR_INPUT=${OPTARG};;
    b)
        DIR_BPE=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

SEED=42

MAX_TOKENS=1024

UPDATE_FREQ=1

BART_PATH="../resources/bart.large/model.pt"

DIR_BPE=${DIR_BPE:-"models/fairseq/explainable_bart/preprocess/eng"}

EXPLANATION_FORMAT="evidence-type"
EXPLANATION_SETTING="rationalization"

#DIR_MODEL=${DIR_MODEL:-"expect-baseline-MT16384*1-lr3e-5-warmup500_temp"}
DIR_MODEL=${DIR_MODEL:-"expect_valid-${EXPLANATION_SETTING}_${EXPLANATION_FORMAT}-before"}
DIR_MODEL=models/fairseq/explainable_bart/exps/eng/${DIR_MODEL}

DIR_INPUT=models/fairseq/explainable_bart/preprocess/eng/expect/valid/bin_temp

if [ ! -d ${DIR_INPUT}/bin ]; then
    bash scripts/fairseq/eng/explainable_bart/preprocess.sh -i ${DIR_INPUT}  -o ${DIR_INPUT}
fi

mkdir -p ${DIR_MODEL}
mkdir -p ${DIR_MODEL}/results

# 2023.01.25修改：使用 lr-scheduler inverse_sqrt
CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT} \
    --save-dir ${DIR_MODEL} \
    --user-dir models/fairseq/explainable_bart \
    --task explainable_gec \
    --arch egec_bart_large \
    --restore-file ${BART_PATH} \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --max-tokens ${MAX_TOKENS} \
    --update-freq ${UPDATE_FREQ} \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --lr 3e-05 \
    --warmup-updates 100 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 30 \
    --patience 10 \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 1 \
    --eval-gec \
    --eval-gec-min-update 500 \
    --eval-gec-metric "errant_eng" \
    --eval-gec-output-prefix "${DIR_MODEL}/results/output" \
    --eval-gec-m2-filepath "../datasets/GEC/EGEC/expect/valid/expect_valid.errant" \
    --eval-gec-raw-filepath "../datasets/GEC/EGEC/expect/json/valid.json" \
    --eval-gec-exp-filepath "models/fairseq/explainable_bart/preprocess/eng/expect/valid/valid.bpe.exp" \
    --beam 5 \
    --bpe gpt2 \
    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
    --min-len 0 \
    --left-pad-source \
    --explanation-format ${EXPLANATION_FORMAT} \
    --explanation-setting ${EXPLANATION_SETTING} \
    --explanation-before \
    --seed $SEED >${DIR_MODEL}/nohup.log 2>&1 &
wait

#    --use-encoder-mlp \
#    --use-decoder-mlp \

#bash scripts/fairseq/eng/predict.sh -g ${GPU_list}\
#    -m ${DIR_MODEL} \
#    -n checkpoint_best_score.pt \
#    -v expect_test
#
#bash scripts/fairseq/eng/predict.sh -g ${GPU_list}\
#    -m ${DIR_MODEL} \
#    -n checkpoint_best_score.pt \
#    -v expect_valid

