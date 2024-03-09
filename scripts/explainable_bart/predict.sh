#!/bin/bash

while getopts "g:p:m:n:o:v:f:s:" optname; do
    case $optname in
    g)
        GPU_list=${OPTARG};;
    p)
        DIR_PROCESSED=${OPTARG};;
    m)
        DIR_MODEL=${OPTARG};;
    n)
        FILE_MODEL=${OPTARG};;
    o)
        DIR_OUTPUT=${OPTARG};;
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

if [ -z ${DIR_MODEL} ]; then
    echo "DIR_MODEL not specified"
    exit -1
fi

if [ ! -n ${VALID_NAME} ]; then
    echo "VALID_NAME not specified"
    exit -1
fi

DIR_WORKING="models/fairseq/explainable_bart"

DIR_DATASET="../Datasets/GEC/EGEC/expect"

DIR_PROCESSED=${DIR_PROCESSED:-"${DIR_WORKING}/preprocess/eng/expect_denoise/train/bin"}

FILE_MODEL=${FILE_MODEL:-"checkpoint_best_score.pt"}

DIR_OUTPUT=${DIR_OUTPUT:-"${DIR_MODEL}/results"}

EXPLANATION_FORMAT=${EXPLANATION_FORMAT:-"evidence-type"}

EXPLANATION_SETTING=${EXPLANATION_SETTING:-""}

FILE_LOG="${DIR_OUTPUT}/${VALID_NAME}.log"

mkdir -p ${DIR_OUTPUT} && touch ${FILE_LOG}

# Default files for validation
if [ ${VALID_NAME} = "expect_valid" ]; then
    # Original Dataset
    # FILE_INPUT=${FILE_INPUT:-"${DIR_DATASET}/valid/expect_valid.src"}
    # FILE_REF=${FILE_REF:-"${DIR_DATASET}/valid/expect_valid.errant"}
    # Denoised Dataset
    FILE_ERRANT=${FILE_ERRANT:-"${DIR_DATASET}/denoise/valid/expect_valid_denoise.errant"}
    FILE_EXPECT=${FILE_EXPECT:-"${DIR_DATASET}/denoise/valid/expect_valid_denoise.json"}
    FILE_EXPLANATION=${FILE_EXPLANATION:-"${DIR_WORKING}/preprocess/eng/expect_denoise/valid/valid.bpe.exp"}
elif [ ${VALID_NAME} = "expect_test" ]; then
    # Original Dataset
    # FILE_INPUT=${FILE_INPUT:-"${DIR_DATASET}/test/expect_test.src"}
    # FILE_REF=${FILE_REF:-"${DIR_DATASET}/test/expect_test.errant"}
    # Denoised Dataset
    FILE_ERRANT=${FILE_ERRANT:-"${DIR_DATASET}/denoise/test/expect_test_denoise.errant"}
    FILE_EXPECT=${FILE_EXPECT:-"${DIR_DATASET}/denoise/test/expect_test_denoise.json"}
    FILE_EXPLANATION=${FILE_EXPLANATION:-"${DIR_WORKING}/preprocess/eng/expect_denoise/test/test.bpe.exp"}
else
    echo "Unknown VALID_NAME=${VALID_NAME}"
    exit -1
fi

echo "#################### predicting ####################" | tee -a ${FILE_LOG}
echo "Model: ${DIR_MODEL}/${FILE_MODEL}" | tee -a ${FILE_LOG}
echo "Valid: ${VALID_NAME}" | tee -a ${FILE_LOG}
echo "Output: ${DIR_OUTPUT}" | tee -a ${FILE_LOG}


# Generate Hypothesis
CUDA_VISIBLE_DEVICES=${GPU_list} python ${DIR_WORKING}/interactive.py ${DIR_PROCESSED} \
    --task explainable_gec \
    --arch egec_bart_large \
    --user-dir ${DIR_WORKING} \
    --path ${DIR_MODEL}/${FILE_MODEL} \
    --beam 10 \
    --nbest 1 \
    -s src \
    -t tgt \
    --bpe gpt2 \
    --buffer-size 10000 \
    --batch-size 128 \
    --num-workers 4 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    --min-len 0 \
    --left-pad-source \
    --explanation-format "${EXPLANATION_FORMAT}" \
    --explanation-setting "${EXPLANATION_SETTING}" \
    --eval-gec-m2-filepath ${FILE_ERRANT} \
    --eval-gec-raw-filepath ${FILE_EXPECT} \
    --eval-gec-exp-filepath ${FILE_EXPLANATION} \
    --explanation-before \
    >> ${FILE_LOG}


# ============================== EXPECT_valid Results ============================== #
#                                                           ckpt  P       R       F0.5    eval_loss   TP    FP    FN    P       R       F1      F0.5    EM      ACC     PS      RS      F1S     F0.5S
# bart-baseline-MT4096*1-lr3e-5                               13  30.38   35.22   31.24   2.386       914   2095  1681
# bart-baseline-MT8192*1-lr3e-5                               22  31.12   34.14   31.68   2.468       886   1961  1709
# bart-baseline-MT12288*1-lr3e-5                              21  30.22   33.91   30.89   2.523       880   2032  1715
# bart-baseline-MT16384*1-lr1e-4                              23  30.99   35.22   31.76   2.484       914   2035  1681
# bart-baseline-MT16384*1-lr8e-5                              16  31.73   33.53   32.07   2.488       870   1872  1725
# bart-baseline-MT16384*1-lr5e-5                              32  31.61   32.29   31.74   2.480       838   1813  1757
# bart-baseline-MT16384*1-lr3e-5                              24  31.03   34.68   31.70   2.446       900   2000  1695
# bart-baseline-MT16384*1-lr1e-5                              84  30.54   34.99   31.34   2.443       908   2065  1687
# bart-baseline-MT16384*1-lr5e-6                              47  28.63   29.29   28.76   2.563       760   1895  1835
# bart-baseline-MT16384*1-lr3e-6                              54  26.83   22.04   25.71   2.578       572   1560  2023

# bart-baseline-MT16384*1-lr3e-5-warmup100                    24  30.34   35.26   31.21   2.510       915   2101  1680
# bart-baseline-MT16384*1-lr3e-5-warmup200                    27  30.03   35.03   30.91   2.496       909   2118  1686
# bart-baseline-MT16384*1-lr3e-5-warmup500                    28  30.59   33.72   31.17   2.511       875   1985  1720
# bart-baseline-MT16384*1-lr3e-5-warmup1000                   33  30.40   34.61   31.16   2.547       898   2056  1697
# bart-baseline-MT16384*1-lr3e-5-warmup2000                   74  30.36   33.99   31.02   2.529       882   2023  1713


# 以下实验统一使用 MT16384*1-lr3e-5-warmup500
# bart-infusion_type-MT16384*1-lr3e-5                         28  31.15   35.14   31.87   2.504       912   2016  1683
# bart-infusion_evid-MT16384*1-lr3e-5                         22  40.72   43.31   41.22   2.479       1124  1636  1471
# bart-infusion_evid-type-MT16384*1-lr3e-5                    19  40.79   42.50   41.11   2.421       1103  1601  1492
# bart-infusion_type-evid-MT16384*1-lr3e-5                    26  40.25   45.39   41.18   2.474       1178  1749  1417
# bart-infusion_type-evid-MT16384*1-lr3e-5                    30  39.82   45.05   40.76   2.417       1169  1767  1426

# Explainable_GEC Error_only Paper                                                                                                                      39.77   56.13   50.39   33.41   40.18   45.74
# Explainable_GEC Error_only Reproduce                                                                                  57.01   27.95   37.51   47.20   40.12   52.09   47.38   23.23   31.17   39.22

# bart-explanation_evid                                       19                                                        41.05   34.59   37.54   39.57   36.01   -
# bart-rational-evid_type_before                              24  29.73   32.25   30.21   2.713       837   1978  1758  39.22   35.31   37.16   38.37   32.95   23.50
# bart-rational-evid_type_before                              25  29.35   33.14   30.04   2.667       860   2070  1735  40.74   33.68   36.87   39.10   32.78   24.04
# bart-rational-evid_type_before-enc_mlp                      36  30.23   33.37   30.81   2.690       866   1999  1729  36.56   41.88   39.04   37.51   28.51   30.13
# bart-rational-evid_type_before-dec_mlp                      37  30.34   33.60   30.94   2.635       872   2002  1723  42.41   33.56   37.47   40.29   33.78   31.33
# bart-rational-evid_type_before-enc_dec_mlp                  35  32.62   31.29   32.35   2.687       812   1677  1783  33.75   44.12   38.25   35.41   21.09   28.22

# bart-rational-evid_type_before-enc_dec_mlp-ew1              25  30.38   34.64   31.15   4.554       899   2060  1696  49.83   16.94   25.29   35.89   39.12   22.38
# bart-rational-evid_type_before-enc_dec_mlp-ew2              50  30.38   35.68   31.26   6.876       918   2104  1677  45.76   31.95   37.63   42.12   35.97   32.78
# bart-rational-evid_type_before-enc_dec_mlp-ew4              33  30.28   35.07   31.13   11.598      910   2095  1685  48.26   32.74   39.01   44.08   37.75   32.16
# bart-rational-evid_type_before-enc_dec_mlp-ew6              35  29.90   32.22   30.34   16.322      836   1960  1759  49.04   29.87   37.12   43.46   38.33   30.96
# bart-rational-evid_type_before-enc_dec_mlp-ew8              33  29.10   32.33   29.69   20.992      893   2044  1756  46.47   33.72   39.08   43.20   36.47   32.37

# bart-rational-evid_type_before-enc_dec_mlp-ew1_eps0         25  30.84   32.99   31.24   2.424       856   1920  1739  44.33   25.75   32.58   38.74   36.43   25.78
# bart-rational-evid_type_before-enc_dec_mlp-ew2_eps0         33  30.87   34.10   31.47   2.719       885   1982  1710  42.95   37.88   40.26   41.83   33.57   32.86
# bart-rational-evid_type_before-enc_dec_mlp-ew3_eps0         25  30.36   32.60   30.78   2.959       846   1946  1749  43.37   32.41   37.10   40.62   34.69   29.09
# bart-rational-evid_type_before-enc_dec_mlp-ew4_eps0         30  29.62   33.91   30.39   2.224       880   2091  1715  42.55   35.71   38.83   40.98   33.73   32.32

# bart-rational-evid_type_after                               25  30.31   34.45   31.05   2.659       894   2056  1701  42.17   41.53   41.85   42.04   32.12   31.41
# bart-rational-evid_type_after                               41  29.41   35.68   30.48   2.654       926   2223  1669  44.55   43.82   44.18   44.40   33.20   37.55
# bart-rational-evid_type_after-enc_mlp                       32  30.93   35.03   31.67   2.587       909   2030  1686  49.93   40.22   44.55   47.63   37.17   37.55
# bart-rational-evid_type_after-dec_mlp                       27  30.39   34.22   31.09   2.724       888   2034  1707  38.64   40.66   39.63   39.03   29.01   29.30
# bart-rational-evid_type_after-enc_dec_mlp                   33  30.47   35.38   31.34   2.688       918   2095  1677  35.09   46.55   40.02   36.91   23.17   33.11

# bart-rational-evid_type_after-enc_dec_mlp-ew1_eps0          34  30.94   35.49   31.75   2.415       921   2056  1674  45.92   38.42   41.84   44.19   35.64   37.63
# bart-rational-evid_type_after-enc_dec_mlp-ew2_eps0          30  30.56   35.34   31.41   2.624       917   2084  1678  46.30   38.47   42.02   44.49   35.89   39.08
# bart-rational-evid_type_after-enc_dec_mlp-ew3_eps0          34  29.32   34.53   30.23   2.847       896   2160  1699  48.01   37.20   41.92   45.37   36.59   39.37
# bart-rational-evid_type_after-enc_dec_mlp-ew4_eps0          32  30.01   31.10   30.22   2.961       807   1882  1788  51.52   36.34   42.62   47.55   40.12   38.29



# ============================== EXPECT_denoise_valid Results ============================== #
#                                                           ckpt  P       R       F0.5    eval_loss   TP    FP    FN    P       R       F1      F0.5    EM      ACC     PS      RS      F1S     F0.5S   PT      RT      F1T     F0.5T   EMT     ACCT
# Explainable_GEC Error_only Reproduce                                                                                  53.60   35.46   42.68   48.63   36.84   52.09   42.76   28.29   34.05   38.79
# bart-explanation_evid_type                                  20                                                        42.19   31.47   36.05   39.50   30.29   28.14   24.32   18.14   20.78   22.77
# bart-explanation_evid_type                                  40                                                        44.43   32.93   37.82   41.53   31.29   33.36   32.19   23.86   27.41   30.09

# bart-baseline-MT4096*4-lr3e-5-warmup500                     20  36.11   38.30   36.53   2.478       994   1759  1601
# bart-baseline-MT4096*4-lr3e-5-warmup500                     20  36.20   38.27   36.60   2.476       993   1750  1602
# bart-baseline-MT4096*4-lr3e-5-warmup500-src_drop0.1         51  36.14   34.87   35.88   2.491       905   1599  1690
# bart-baseline-MT4096*4-lr3e-5-warmup500-src_drop0.2         75  35.99   34.95   35.78   2.402       907   1613  1688

# bart-infusion_type                                          25  36.28   38.54   36.71   2.458       1000  1756  1595
# bart-infusion_type-src_drop0.1                              60  35.31   34.87   35.22   2.489       905   1658  1690
# bart-infusion_evid                                          24  45.51   46.24   45.65   2.447       1200  1437  1395
# bart-infusion_evid-src_drop0.1                              36  45.78   44.55   45.53   2.377       1156  1369  1439
# bart-infusion_evid_type                                     23  44.55   48.02   45.20   2.439       1246  1551  1349
# bart-infusion_evid_type-src_drop0.1                         62  44.28   47.55   44.90   2.448       1234  1553  1361
# bart-infusion_type_evid                                     28  45.38   46.90   45.67   2.433       1217  1465  1378
# bart-infusion_type_evid-src_drop0.1                         49  44.25   46.86   44.75               1216  1532  1379


# bart-rational-evid_type_after-enc_dec_mlp-ew0.5             37  35.67   37.50   36.02   2.296       973   1755  1622  49.40   37.75   42.80   46.53   37.46   37.22   37.57   28.71   32.55   35.39
# bart-rational-evid_type_after-enc_dec_mlp-ew1.0             23  35.39   39.15   36.08   2.338       1016  1855  1579  47.35   38.50   42.47   45.27   36.39   35.64   34.18   27.79   30.66   32.68
# bart-rational-evid_type_after-enc_dec_mlp-ew2.0             19  34.18   37.80   34.85   2.610       981   1889  1614  45.53   36.04   40.24   43.25   35.72   31.66   31.62   25.03   27.94   30.04

# bart-rational-evid_type_after-ew1.0                         33  35.50   38.11   35.99   2.355       989   1797  1606  46.88   45.65   46.26   46.63   35.72   39.91   36.05   35.11   35.57   35.86
# bart-rational-evid_type_after-enc_mlp-ew0.5                 35  35.34   37.74   35.81   2.269       982   1797  1613  49.14   43.54   46.17   47.91   37.26   38.58   37.19   32.95   34.94   36.26
# bart-rational-evid_type_after-enc_mlp-ew0.5-src_drop_0.1    46  35.40   38.03   35.90   2.287       987   1801  1608  39.77   38.88   39.32   39.59   27.60   32.82   28.39   27.75   28.06   28.25
# bart-rational-evid_type_after-enc_mlp-ew0.5-src_drop_0.2    48  35.95   38.46   36.43   2.305       998   1778  1597  43.51   36.44   39.66   41.88   29.76   32.82   31.37   26.27   28.59   30.20
# bart-rational-evid_type_after-enc_mlp-ew1.0                 29  36.12   38.61   36.59   2.302       1002  1772  1593  49.32   45.32   47.24   48.46   36.72   39.78   38.79   35.65   37.15   38.12
# bart-rational-evid_type_after-enc_mlp-ew1.0                 29  35.57   38.61   36.14   2.366       1002  1815  1593  46.54   47.64   47.09   46.76   35.31   40.03   36.20   37.05   36.62   36.37
# bart-rational-evid_type_after-enc_mlp-ew1.0-src_drop0.1     74  36.34   40.15   37.05   2.366       1042  1825  1553  48.95   42.72   45.63   47.56   36.05   40.32   38.10   33.26   35.51   37.02
# bart-rational-evid_type_after-enc_mlp-ew1.0-src_drop0.2     53  35.47   39.19   36.16   2.357       1017  1850  1578  48.56   40.05   43.90   46.58   34.15   37.71   37.45   30.89   33.86   35.92
# bart-rational-evid_type_after-enc_mlp-ew1.5                 36  34.63   39.11   35.44   2.446       1015  1916  1580  48.58   45.21   46.83   47.87   35.89   41.90   38.40   35.74   37.02   37.84
# bart-rational-evid_type_after-enc_mlp-ew1.5-src_drop0.1     37  36.03   38.42   36.49   2.448       997   1770  1598  43.90   42.82   43.35   43.68   30.96   36.88   32.94   32.13   32.53   32.77
# bart-rational-evid_type_after-enc_mlp-ew2.0                 24  35.16   36.61   35.44   2.517       950   1752  1645  50.85   43.50   46.89   49.19   38.75   39.78   39.56   33.84   36.48   38.27
# bart-rational-evid_type_after-enc_mlp-ew2.0-src_drop0.1     46  35.41   38.61   36.00   2.515       1002  1828  1593  47.98   42.86   45.28   46.86   35.35   40.07   37.28   33.30   35.18   36.41
# bart-rational-evid_type_after-dec_mlp-ew1.0                 31  34.79   37.76   35.35   2.379       980   1837  1615  44.07   42.98   43.52   43.85   33.61   37.46   33.21   32.39   32.79   33.04

# bart-tag_bo_l2_tw0.2_eps0.1                                 39  36.77   35.03   36.41   2.733       909   1563  1686  -
# bart-tag_bo_l2_tw0.5_eps0.1                                 40  37.34   34.99   36.84   3.077       908   1524  1687  61.18   2.18    4.21    9.54    40.53   17.03   27.63   0.98    1.90    4.29
# bart-tag_bo_l2_tw0.5_eps0.1                                 53  36.16   35.68   36.06   3.058       926   1635  1669  57.00   6.87    12.26   23.18   39.70   19.15   32.10   3.87    6.90    13.05
# bart-tag_bo_l2_tw0.8_eps0.1                                 60  35.47   36.92   35.74   3.427       958   1743  1637  51.77   21.63   30.51   40.49   36.43   23.46   29.33   12.26   17.29   22.94
# bart-tag_bo_l2_tw1.0_eps0.1                                 60  35.10   36.96   35.46   3.666       959   1773  1636  48.82   26.55   34.40   41.81   35.76   25.94   29.21   15.89   20.58   25.02
# bart-tag_bo_l2_tw1.5_eps0.1                                 42  36.12   36.34   36.16   4.241       943   1668  1652  50.95   22.01   30.74   40.34   36.97   24.66   30.66   13.24   18.49   24.27
# bart-tag_bo_l2_tw1.5_eps0.1                                 47  36.04   35.72   35.98   4.218       927   1645  1668  52.69   19.31   28.26   39.15   38.38   25.24   34.46   12.63   18.49   25.61
# bart-tag_bo_l2_tw2.0_eps0.1                                 47  35.93   35.38   35.82   4.797       918   1637  1677  52.48   22.29   31.29   41.29   37.96   28.06   37.03   15.73   22.08   29.14

# bart-rational-evid_type_after-tag_bio_l1_tw1.0_eps0.1-ew1.0
#                                                             31  34.43   36.61   34.85   3.521       950   1809  1645  46.08   43.19   44.59   45.47   35.27   37.22   34.02   31.90   32.93   33.57   56.58   15.21   23.97   36.65   38.96   21.55
# bart-rational-evid_type_after-tag_bio_l2_tw1.0_eps0.1-ew1.0
#                                                             44  33.85   37.15   34.46   3.596       964   1884  1631  45.82   44.57   45.19   45.56   35.39   40.53   35.00   34.05   34.52   34.81   48.79   27.79   35.41   42.38   35.89   25.11
# bart-rational-evid_type_after-enc_mlp-tag_bo_l2_tw0.2_eps0.1-ew1.0
#                                                             36  35.57   38.88   36.18   2.570       1009  1828  1586  46.11   46.85   46.48   46.26   35.23   41.44   35.99   36.56   36.27   36.10   58.38   12.82   21.02   34.13   38.13   20.31
# bart-rational-evid_type_after-enc_mlp-tag_bo_l2_tw0.5_eps0.1-ew1.0
#                                                             27  34.80   38.65   35.51   2.882       1003  1879  1592  49.33   43.43   46.19   48.03   36.14   39.74   38.45   33.84   36.00   37.43   58.16   12.87   21.07   34.14   38.42   21.09
# bart-rational-evid_type_after-enc_mlp-tag_bo_l2_tw1.0_eps0.1-ew1.0
#                                                             24  34.81   38.19   35.44   3.356       991   1856  1604  49.05   42.28   45.41   47.53   37.75   36.59   36.19   31.19   33.51   35.07   56.54   20.46   30.05   41.80   37.59   22.38

# bart-rational-evid_type_before-enc_mlp-ew1.0                25  36.23   36.74   36.35   2.366       956   1683  1639  35.47   42.68   38.74   36.71   25.69   28.80   22.83   27.47   24.93   23.63
# bart-rational-evid_type_before-enc_mlp-ew1.0-src_drop0.1    42  38.25   34.18   37.36   2.375       887   1432  1708  36.01   35.58   35.79   35.92   26.61   26.56   23.79   23.51   23.65   23.73

# bart-rational-evid_type_before-enc_dec_mlp-ew0.5            51  35.53   37.34   35.88   2.280       969   1758  1626  47.46   32.39   38.50   43.42   36.84   32.28   47.46   32.39   38.50   43.42
# bart-rational-evid_type_before-enc_dec_mlp-ew1.0            26  35.83   36.72   36.00   2.364       953   1707  1642  51.20   28.05   36.25   43.95   38.83   28.18   34.26   18.77   24.25   29.41
# bart-rational-evid_type_before-enc_dec_mlp-ew2.0            30  35.62   38.27   36.12   2.599       993   1795  1602  47.43   34.22   39.75   39.75   36.72   32.99   34.83   25.12   29.19   32.33




# ============================== EXPECT_denoise_test Results ============================== #
#                                                           ckpt  P       R       F0.5    eval_loss   TP    FP    FN    P       R       F1      F0.5    EM      ACC     PS      RS      F1S     F0.5S   PT      RT      F1T     F0.5T   EMT     ACCT
# Explainable_GEC Error_only Reproduce                                                                                  51.73   36.34   42.69   47.69   36.38   50.83   41.31   29.02   34.09   38.08
# bart-explanation_evid_type                                  20                                                        42.34   33.13   37.18   40.11   29.06   26.95   23.64   18.49   20.75   22.39

# bart-infusion_type                                          25  35.54   37.50   35.91               968   1756  1613
# bart-infusion_type-src_drop0.1                              60  36.00   35.37   35.87               913   1623  1668
# bart-infusion_evid                                          24  46.32   47.07   46.47               1215  1408  1366
# bart-infusion_evid-src_drop0.1                              36  46.02   44.13   45.63               1139  1336  1442
# bart-infusion_evid_type                                     23  46.03   49.36   46.66               1274  1494  1307
# bart-infusion_evid_type-src_drop0.1                         62  44.96   47.50   45.44               1226  1501  1355
# bart-infusion_type_evid                                     28  46.21   47.46   46.45               1225  1426  1356
# bart-infusion_type_evid-src_drop0.1                         49  47.38   47.38   45.78               1223  1471  1358

# bart-baseline-MT4096*4-lr3e-5-warmup500                     20  36.35   38.94   36.84               1005  1760  1576
# bart-baseline-MT4096*4-lr3e-5-warmup500                     20  35.79   38.16   36.24               985   1767  1596
# bart-baseline-MT4096*4-lr3e-5-warmup500-src_drop0.1         51  36.33   35.49   36.16               916   1605  1665
# bart-baseline-MT4096*4-lr3e-5-warmup500-src_drop0.2         75  36.19   34.68   35.88               895   1578  1686

# bart-rational-evid_type_after-enc_mlp-ew0.5-src_drop_0.1    46  35.65   38.12   36.12               984   1776  1597  39.77   38.47   39.11   39.50   28.68   31.75   26.86   25.98   26.41   26.68
# bart-rational-evid_type_after-enc_mlp-ew0.5-src_drop_0.2    48  36.55   39.21   37.05               1012  1757  1569  43.69   37.06   40.10   42.18   30.22   32.16   29.97   25.42   27.51   28.93
# bart-rational-evid_type_after-enc_mlp-ew1.0                 29  35.84   38.98   36.43               1006  1801  1575  47.15   48.83   47.98   47.48   35.80   40.73   35.13   36.39   35.75   35.37
# bart-rational-evid_type_after-enc_mlp-ew1.0-src_drop0.1     74  36.52   40.41   37.24               1043  1813  1538  49.43   44.10   46.61   48.26   35.43   39.86   37.19   33.18   35.07   36.31
# bart-rational-evid_type_after-enc_mlp-ew1.0-src_drop0.2     53  36.26   39.56   36.87               1021  1795  1560  48.42   40.71   44.23   46.65   34.64   38.04   36.91   31.03   33.72   35.56
# bart-rational-evid_type_after-enc_mlp-ew1.5-src_drop0.1     37  35.92   38.16   36.35               985   1757  1596  45.12   44.54   44.83   45.00   31.58   36.30   32.89   32.46   32.67   32.80
# bart-rational-evid_type_after-enc_mlp-ew2.0-src_drop0.1     46  36.00   39.02   36.57               1007  1790  1574  47.24   41.93   44.43   46.07   34.48   38.99   35.56   31.56   33.44   34.68

# bart-rational-evid_type_before-enc_mlp-ew1.0                25  36.58   37.93   36.85               979   1697  1602  35.81   42.81   38.99   37.02   25.66   27.32   21.32   25.49   23.22   22.04
# bart-rational-evid_type_before-enc_mlp-ew1.0-src_drop0.1    42  38.68   35.41   37.98               914   1449  1667  36.77   36.85   36.81   36.79   27.11   26.24   23.13   23.18   23.15   23.14



# ============================== EXPECT_random_valid Results ============================== #
# bart-infusion_evid-src_drop0.1                              32  35.88   33.26   35.33   2.515       863   1542  1732

# bart-rational-evid_type_after-enc_mlp-ew1.0-src_drop0.1     40  36.36   34.37   35.95   2.588       892   1561  1703  14.39   0.45    0.86    2.00    40.49   16.04   3.03    0.09    0.18    0.40
# bart-rational-evid_type_before-enc_mlp-ew1.0-src_drop0.1    38  36.17   33.72   35.65   2.554       875   1544  1720  13.60   0.40    0.77    1.79    40.24   15.83   2.40    0.07    0.14    0.31
# bart-rational-evid_type_before-enc_mlp-ew1.0-src_drop0.1    39  37.04   35.53   36.73   2.572       922   1567  1673  9.72    0.16    0.32    0.75    40.36   15.87   2.78    0.05    0.09    0.23


# ============================== EXPECT_random_test Results ============================== #
# bart-infusion_evid-src_drop0.1                              32  36.44   33.20   35.74               857   1495  1724

# bart-rational-evid_type_after-enc_mlp-ew1.0-src_drop0.1     40  36.86   34.87   36.44               900   1542  1681  7.45    0.16    0.32    0.74    39.90   15.02   3.19    0.07    0.14    0.32
# bart-rational-evid_type_before-enc_mlp-ew1.0-src_drop0.1    38  37.63   34.83   37.04               899   1490  1682  14.38   0.53    1.02    2.31    39.65   15.02   3.75    0.14    0.27    0.61
# bart-rational-evid_type_before-enc_mlp-ew1.0-src_drop0.1    39  37.53   35.84   37.18               925   1540  1656  15.00   0.28    0.54    1.30    39.90   15.02   7.75    0.07    0.14    0.33


# ============================== EXPECT_adjacent_random_valid Results ============================== #
# bart-infusion_evid-src_drop0.1                              73  38.46   42.81   39.26   2.463       1111  1778  1484
# bart-infusion_evid-src_drop0.2                              51  38.87   39.15   38.92   2.486       1016  1598  1579

# bart-rational-evid_type_after-enc_mlp-ew1.0-src_drop0.1     31  36.36   34.14   35.89   2.542       886   1551  1709  23.68   2.53    4.57    8.86    39.41   15.79   6.63    0.68    1.23    2.38
# bart-rational-evid_type_before-enc_mlp-ew1.0-src_drop0.1    47  36.53   38.73   36.95   2.509       1005  1746  1590  26.97   3.37    6.00    11.23   39.54   17.03   7.87    0.98    1.75    3.27


# ============================== EXPECT_adjacent_random_test Results ============================== #
# bart-infusion_evid-src_drop0.1                              73  39.66   43.01   40.28               1110  1689  1471
# bart-infusion_evid-src_drop0.2                              51  39.70   39.21   39.60               1012  1537  1569

# bart-rational-evid_type_after-enc_mlp-ew1.0-src_drop0.1     31  37.34   35.18   36.88               908   1524  1673  26.74   3.28    5.84    11.00   38.78   15.48   10.73   1.32    2.34    4.42
# bart-rational-evid_type_before-enc_mlp-ew1.0-src_drop0.1    47  37.09   39.52   37.55               1020  1730  1561  29.00   4.02    7.06    12.93   38.66   16.02   10.17   1.41    2.47    4.54

