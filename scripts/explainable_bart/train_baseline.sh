#!/bin/bash

while getopts "g:i:b:m:d:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    i)
        DIR_INPUT=${OPTARG};;
    b)
        DIR_BPE=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    d)
        DATASET=${OPTARG};;
    ?)
        echo "Unknown option $OPTARG"
        exit 1;;
    esac
    # echo "option index is $OPTIND"
done

SEED=42

MAX_TOKENS=4096

UPDATE_FREQ=4

BART_PATH="../Resources/bart.large/model.pt"

DIR_WORKING="models/fairseq/explainable_bart"

DIR_DATASET="../Datasets/GEC/EGEC/expect"

DIR_BPE=${DIR_BPE:-"${DIR_WORKING}/preprocess/eng"}

EXPLANATION_SETTING="infusion"

EXPLANATION_FORMAT="evidence"

SRC_DROPOUT="0.1"

EXPLANATION_WEIGHT="1.0"

TAGGING_WEIGHT="1.0"

DATASET=${DATASET:-""}

if [ ${DATASET} = "denoise" ]; then
    DIR_INPUT=${DIR_INPUT:-"${DIR_WORKING}/preprocess/eng/expect_denoise"}
    FILE_EXPECT=${FILE_EXPECT:-"${DIR_DATASET}/denoise/valid/expect_valid_denoise.json"}
    FILE_EXPLANATION=${FILE_EXPLANATION:-"${DIR_WORKING}/preprocess/eng/expect_denoise/valid/valid.bpe.exp"}
elif [ ${DATASET} = "random" ]; then
    DIR_INPUT=${DIR_INPUT:-"${DIR_WORKING}/preprocess/eng/expect_random"}
    FILE_EXPECT=${FILE_EXPECT:-"${DIR_DATASET}/artifact/random/valid/expect_valid_random.json"}
    FILE_EXPLANATION=${FILE_EXPLANATION:-"${DIR_WORKING}/preprocess/eng/expect_random/valid/valid.bpe.exp"}
elif [ ${DATASET} = "adjacent_random" ]; then
    DIR_INPUT=${DIR_INPUT:-"${DIR_WORKING}/preprocess/eng/expect_random"}
    FILE_EXPECT=${FILE_EXPECT:-"${DIR_DATASET}/artifact/adjacent_random/valid/expect_valid_adjacent_random.json"}
    FILE_EXPLANATION=${FILE_EXPLANATION:-"${DIR_WORKING}/preprocess/eng/expect_adjacent_random/valid/valid.bpe.exp"}
else
    echo "Error Dataset: ${DATASET}"
    exit 1
fi

if [ ! -d ${DIR_INPUT}/train/bin ]; then
    bash scripts/fairseq/eng/explainable_bart/preprocess.sh \
        -i "${DIR_DATASET}/denoise"  \
        -o ${DIR_INPUT}
fi

# Baseline
#DIR_MODEL=${DIR_MODEL:-"expect_denoise/expect_denoise-baseline-MT4096*4-lr3e-5-warmup500"}
# Explanation
#DIR_MODEL=${DIR_MODEL:-"expect_denoise/expect_denoise-explanation_evidence_type"}
# Infusion
#DIR_MODEL=${DIR_MODEL:-"expect_denoise/expect_denoise-infusion_evidence_type-src_drop0.1"}
# Rationalization
#DIR_MODEL=${DIR_MODEL:-"expect_denoise/expect_denoise-rationalization_evidence_type_before-enc_mlp-ew1.0_eps0"}
# Sequence Labeling
#DIR_MODEL=${DIR_MODEL:-"expect_denoise/expect_denoise-tag_bo_l2_tw0.8_eps0.1"}

# ============================== Artifact-Random ==============================
# Infusion
DIR_MODEL=${DIR_MODEL:-"expect_random/expect_random-infusion_evidence-src_drop0.1"}
# Rationalization
#DIR_MODEL=${DIR_MODEL:-"expect_random/expect_random-rationalization_evidence_type_before-enc_mlp-ew1.0_eps0-src_drop0.1"}
#DIR_MODEL=${DIR_MODEL:-"expect_adjacent_random/expect_adjacent_random-rationalization_evidence_type_after-enc_mlp-ew1.0_eps0-src_drop0.1"}

DIR_MODEL=${DIR_WORKING}/exps/eng/${DIR_MODEL}

mkdir -p ${DIR_MODEL} && mkdir -p ${DIR_MODEL}/results

# ========================= Baseline =========================
#CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT}/train/bin \
#    --save-dir ${DIR_MODEL} \
#    --user-dir ${DIR_WORKING} \
#    --task explainable_gec \
#    --arch egec_bart_large \
#    --restore-file ${BART_PATH} \
#    --reset-lr-scheduler \
#    --reset-optimizer \
#    --reset-meters \
#    --reset-dataloader \
#    --max-tokens ${MAX_TOKENS} \
#    --update-freq ${UPDATE_FREQ} \
#    --optimizer adam \
#    --layernorm-embedding \
#    --share-all-embeddings \
#    --share-decoder-input-output-embed \
#    --lr 3e-05 \
#    --warmup-updates 500 \
#    --weight-decay 0.01 \
#    -s src \
#    -t tgt \
#    --dropout 0.3 \
#    --source-word-dropout ${SRC_DROPOUT} \
#    --lr-scheduler inverse_sqrt \
#    --clip-norm 0.1 \
#    --criterion label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --max-epoch 100 \
#    --patience 20 \
#    --adam-betas '(0.9,0.999)' \
#    --log-format tqdm \
#    --fp16 \
#    --skip-invalid-size-inputs-valid-test \
#    --find-unused-parameters \
#    --keep-last-epochs 1 \
#    --eval-gec \
#    --eval-gec-min-update 300 \
#    --eval-gec-metric "errant_eng" \
#    --eval-gec-output-prefix ${DIR_MODEL}/results/output \
#    --eval-gec-m2-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.errant" \
#    --eval-gec-exp-filepath "${DIR_WORKING}/preprocess/eng/expect_denoise/valid/valid.bpe.exp" \
#    --beam 10 \
#    --bpe gpt2 \
#    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
#    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
#    --left-pad-source \
#    --seed $SEED >${DIR_MODEL}/nohup.log 2>&1 &
#wait

# ========================= Rationalization Training on denoised datasets =========================
#CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT}/train/bin \
#    --save-dir ${DIR_MODEL} \
#    --user-dir ${DIR_WORKING} \
#    --task explainable_gec \
#    --arch egec_bart_large \
#    --restore-file ${BART_PATH} \
#    --reset-lr-scheduler \
#    --reset-optimizer \
#    --reset-meters \
#    --reset-dataloader \
#    --max-tokens ${MAX_TOKENS} \
#    --update-freq ${UPDATE_FREQ} \
#    --optimizer adam \
#    --layernorm-embedding \
#    --share-all-embeddings \
#    --share-decoder-input-output-embed \
#    --lr 3e-05 \
#    --warmup-updates 500 \
#    --weight-decay 0.01 \
#    -s src \
#    -t tgt \
#    --dropout 0.3 \
#    --source-word-dropout ${SRC_DROPOUT} \
#    --lr-scheduler inverse_sqrt \
#    --clip-norm 0.1 \
#    --criterion explainable_label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --max-epoch 100 \
#    --patience 20 \
#    --adam-betas '(0.9,0.999)' \
#    --log-format tqdm \
#    --fp16 \
#    --skip-invalid-size-inputs-valid-test \
#    --find-unused-parameters \
#    --keep-last-epochs 1 \
#    --eval-gec \
#    --eval-gec-min-update 500 \
#    --eval-gec-metric "errant_eng" \
#    --eval-gec-output-prefix "${DIR_MODEL}/results/output" \
#    --eval-gec-m2-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.errant" \
#    --eval-gec-raw-filepath "${DIR_DATASET}/artifact/adjacent_random/valid/expect_valid_adjacent_random.json" \
#    --eval-gec-exp-filepath "${DIR_WORKING}/preprocess/eng/expect_adjacent_random/valid/valid.bpe.exp" \
#    --beam 10 \
#    --bpe gpt2 \
#    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
#    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
#    --min-len 0 \
#    --left-pad-source \
#    --explanation-format "${EXPLANATION_FORMAT}" \
#    --explanation-setting "${EXPLANATION_SETTING}" \
#    --explanation-weight ${EXPLANATION_WEIGHT} \
#    --use-encoder-mlp \
#    --seed ${SEED} >${DIR_MODEL}/nohup.log 2>&1 &
#wait

#    --explanation-before \
#    --eval-gec-raw-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.json" \
#    --eval-gec-exp-filepath "${DIR_WORKING}/preprocess/eng/expect_denoise/valid/valid.bpe.exp" \
#    --use-decoder-mlp \

# ========================= Infusion Training on denoised datasets =========================
#CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT}/train/bin \
#    --save-dir ${DIR_MODEL} \
#    --user-dir ${DIR_WORKING} \
#    --task explainable_gec \
#    --arch egec_bart_large \
#    --restore-file ${BART_PATH} \
#    --reset-lr-scheduler \
#    --reset-optimizer \
#    --reset-meters \
#    --reset-dataloader \
#    --max-tokens ${MAX_TOKENS} \
#    --update-freq ${UPDATE_FREQ} \
#    --optimizer adam \
#    --layernorm-embedding \
#    --share-all-embeddings \
#    --share-decoder-input-output-embed \
#    --lr 3e-05 \
#    --warmup-updates 500 \
#    --weight-decay 0.01 \
#    -s src \
#    -t tgt \
#    --dropout 0.3 \
#    --source-word-dropout ${SRC_DROPOUT} \
#    --lr-scheduler inverse_sqrt \
#    --clip-norm 0.1 \
#    --criterion explainable_label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --max-epoch 100 \
#    --patience 20 \
#    --adam-betas '(0.9,0.999)' \
#    --log-format tqdm \
#    --fp16 \
#    --skip-invalid-size-inputs-valid-test \
#    --find-unused-parameters \
#    --keep-last-epochs 1 \
#    --eval-gec \
#    --eval-gec-min-update 300 \
#    --eval-gec-metric "errant_eng" \
#    --eval-gec-output-prefix "${DIR_MODEL}/results/output" \
#    --eval-gec-m2-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.errant" \
#    --eval-gec-raw-filepath "${DIR_DATASET}/artifact/random/valid/expect_valid_random.json" \
#    --eval-gec-exp-filepath "${DIR_WORKING}/preprocess/eng/expect_random/valid/valid.bpe.exp" \
#    --beam 10 \
#    --bpe gpt2 \
#    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
#    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
#    --min-len 0 \
#    --left-pad-source \
#    --explanation-format "${EXPLANATION_FORMAT}" \
#    --explanation-setting "${EXPLANATION_SETTING}" \
#    --seed ${SEED} >${DIR_MODEL}/nohup.log 2>&1 &
#wait


# ========================= Explanation Training on denoised datasets =========================
#CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT}/train/bin \
#    --save-dir ${DIR_MODEL} \
#    --user-dir ${DIR_WORKING} \
#    --task explainable_gec \
#    --arch egec_bart_large \
#    --restore-file ${BART_PATH} \
#    --reset-lr-scheduler \
#    --reset-optimizer \
#    --reset-meters \
#    --reset-dataloader \
#    --max-tokens ${MAX_TOKENS} \
#    --update-freq ${UPDATE_FREQ} \
#    --optimizer adam \
#    --layernorm-embedding \
#    --share-all-embeddings \
#    --share-decoder-input-output-embed \
#    --lr 3e-05 \
#    --warmup-updates 500 \
#    --weight-decay 0.01 \
#    -s src \
#    -t tgt \
#    --dropout 0.3 \
#    --source-word-dropout ${SRC_DROPOUT} \
#    --lr-scheduler inverse_sqrt \
#    --clip-norm 0.1 \
#    --criterion label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --max-epoch 100 \
#    --patience 20 \
#    --adam-betas '(0.9,0.999)' \
#    --log-format tqdm \
#    --fp16 \
#    --skip-invalid-size-inputs-valid-test \
#    --find-unused-parameters \
#    --keep-last-epochs 1 \
#    --eval-gec \
#    --eval-gec-min-update 300 \
#    --eval-gec-metric "errant_eng" \
#    --eval-gec-output-prefix "${DIR_MODEL}/results/output" \
#    --eval-gec-m2-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.errant" \
#    --eval-gec-raw-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.json" \
#    --eval-gec-exp-filepath "${DIR_WORKING}/preprocess/eng/expect_denoise/valid/valid.bpe.exp" \
#    --beam 10 \
#    --bpe gpt2 \
#    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
#    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
#    --min-len 0 \
#    --left-pad-source \
#    --explanation-format ${EXPLANATION_FORMAT} \
#    --explanation-setting ${EXPLANATION_SETTING} \
#    --seed $SEED >${DIR_MODEL}/nohup.log 2>&1 &
#wait


# ========================= Sequence Labeling Training on denoised datasets =========================
#CUDA_VISIBLE_DEVICES=${GPU_list} nohup python models/fairseq/train.py ${DIR_INPUT}/train/bin \
#    --save-dir ${DIR_MODEL} \
#    --user-dir ${DIR_WORKING} \
#    --task explainable_gec \
#    --arch egec_bart_large \
#    --restore-file ${BART_PATH} \
#    --reset-lr-scheduler \
#    --reset-optimizer \
#    --reset-meters \
#    --reset-dataloader \
#    --max-tokens ${MAX_TOKENS} \
#    --update-freq ${UPDATE_FREQ} \
#    --optimizer adam \
#    --layernorm-embedding \
#    --share-all-embeddings \
#    --share-decoder-input-output-embed \
#    --lr 3e-05 \
#    --warmup-updates 500 \
#    --weight-decay 0.01 \
#    -s src \
#    -t tgt \
#    --dropout 0.3 \
#    --source-word-dropout ${SRC_DROPOUT} \
#    --lr-scheduler inverse_sqrt \
#    --clip-norm 0.1 \
#    --criterion explainable_label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --max-epoch 100 \
#    --patience 20 \
#    --adam-betas '(0.9,0.999)' \
#    --log-format tqdm \
#    --fp16 \
#    --skip-invalid-size-inputs-valid-test \
#    --find-unused-parameters \
#    --keep-last-epochs 1 \
#    --eval-gec \
#    --eval-gec-min-update 0 \
#    --eval-gec-metric "errant_eng" \
#    --eval-gec-output-prefix "${DIR_MODEL}/results/output" \
#    --eval-gec-m2-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.errant" \
#    --eval-gec-raw-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.json" \
#    --eval-gec-exp-filepath "${DIR_WORKING}/preprocess/eng/expect_denoise/valid/valid.bpe.exp" \
#    --beam 10 \
#    --bpe gpt2 \
#    --gpt2-encoder-json ${DIR_BPE}/encoder.json \
#    --gpt2-vocab-bpe ${DIR_BPE}/vocab.bpe \
#    --left-pad-source \
#    --sequence-tagging \
#    --tagging-weight ${TAGGING_WEIGHT} \
#    --seed ${SEED} >${DIR_MODEL}/nohup.log 2>&1 &
#wait


bash scripts/fairseq/eng/explainable_bart/predict_batch.sh \
    -g ${GPU_list} \
    -m ${DIR_MODEL} \
    -v "expect_test" \
    -f "${EXPLANATION_FORMAT}" \
    -s "${EXPLANATION_SETTING}"

#bash scripts/fairseq/eng/explainable_bart/predict.sh \
#    -g ${GPU_list} \
#    -m ${DIR_MODEL} \
#    -n ${CHECKPOINT} \
#    -v expect_valid \
#    -f "${EXPLANATION_FORMAT}" \
#    -s "${EXPLANATION_SETTING}"

#--eval-gec-m2-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.errant" \
#    --eval-gec-raw-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.json" \
#    --eval-gec-exp-filepath "${DIR_WORKING}/preprocess/eng/expect_denoise/valid/valid.bpe.exp" \


#--eval-gec-m2-filepath "${DIR_DATASET}/denoise/valid/expect_valid_denoise.errant" \
#    --eval-gec-raw-filepath "${DIR_DATASET}/artifact/adjacent_random/valid/expect_valid_adjacent_random.json" \
#    --eval-gec-exp-filepath "${DIR_WORKING}/preprocess/eng/expect_adjacent_random/valid/valid.bpe.exp" \