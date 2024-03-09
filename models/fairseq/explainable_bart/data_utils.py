import json
from typing import List
from itertools import chain
from utils import get_logger
from dataclasses import dataclass, field
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    performance_measure,
)

try:
    from .egec_language_pair_dataset import ERROR_TYPE_TOKENS, TAGGING_LABELS
except ImportError:
    from egec_language_pair_dataset import ERROR_TYPE_TOKENS, TAGGING_LABELS

LOGGER = get_logger(__name__)


@dataclass
class ExplainableSample(object):
    index: int = field(
        default=None, metadata={"help": "Sample Index"}
    )
    src_tokens: List[str] = field(
        default=None, metadata={"help": "Source tokens, which are usually ungrammatical"}
    )
    tgt_tokens: List[str] = field(
        default=None, metadata={"help": "Target tokens, which are grammatical"}
    )
    error_type: str = field(
        default=None, metadata={"help": "Error type defined in the EXPECT paper"}
    )
    src_correction: List[str] = field(
        default=None, metadata={"help": "Source correction words"}
    )
    tgt_correction: List[str] = field(
        default=None, metadata={"help": "Target correction words"}
    )
    src_correction_index: List[int] = field(
        default=None, metadata={"help": "Index of corrections in src_tokens"}
    )
    tgt_correction_index: List[int] = field(
        default=None, metadata={"help": "Index of corrections in tgt_tokens"}
    )
    src_evidence: List[str] = field(
        default=None, metadata={"help": "Source evidence words (extractive rationals)"}
    )
    tgt_evidence: List[str] = field(
        default=None, metadata={"help": "Target evidence words (extractive rationals)"}
    )
    src_evidence_index: List[int] = field(
        default=None, metadata={"help": "Index of evidence words (extractive rationals) in src_tokens"}
    )
    tgt_evidence_index: List[int] = field(
        default=None, metadata={"help": "Index of evidence words (extractive rationals) in tgt_tokens"}
    )

    def __repr__(self):
        return f"ExplainableSample(index={self.index}, " \
               f"source={' '.join(self.src_tokens)}, " \
               f"target={' '.join(self.tgt_tokens)}, " \
               f"error_type={self.error_type})"

    def to_json(self):
        result = {
            "source": self.src_tokens,
            "target": self.tgt_tokens,
            "error_type": self.error_type,
            "source_evidence_index": self.src_evidence_index,
            "target_evidence_index": self.tgt_evidence_index,
        }
        if self.src_correction_index is not None and self.tgt_correction_index is not None:
            result.update({
                "source_correction_index": self.src_correction_index,
                "target_correction_index": self.tgt_correction_index,
            })
        return result


def load_expect(file_input: str, max_sample: int = -1) -> List[ExplainableSample]:
    """
    Input: {
        "target": ["That", "'s", "the", "reason", "why", "basketball", "is", "my", "favorite", "sport", "."],
        "source": ["That", "'s", "[NONE]", "reason", "why", "basketball", "is", "my", "favorite", "sport", "."],
        "evidence_index": [0, 1, 3, 12, 13, 15],
        "correction_index": [2, 14],
        "error_type": "Article",
        "predicted_parsing_order": {"0": 2, "1": 2, "2": 1, "3": 2, "8": 3, "10": 2, "12": 2, "13": 2, "14": 1, "15": 2, "20": 3, "22": 2},
        "origin": "A"
    }

    Output: {
        "source": ["That", "'s", "reason", "why", "basketball", "is", "my", "favorite", "sport", "."],
        "target": ["That", "'s", "the", "reason", "why", "basketball", "is", "my", "favorite", "sport", "."],
        "source_evidence_index": [0, 1, 2],
        "target_evidence_index": [0, 1, 2],
        "error_type": "Article",
    }

    src: That 's [NONE] reason why basketball is my favorite sport .
    tgt: That 's the reason why basketball is my favorite sport .
    src_evidence: That 's reason
    tgt_evidence: That 's reason
    """
    samples = []
    with open(file_input, "r", encoding="utf-8") as f:
        for line in f:
            if 0 <= max_sample <= len(samples):
                break
            raw_sample = json.loads(line)
            src_tokens, tgt_tokens = raw_sample["source"], raw_sample["target"]

            # Acquire source_correction_index and source_correction_index
            src_correction_idx, src_correction = [], []
            tgt_correction_idx, tgt_correction = [], []
            for idx in raw_sample["correction_index"]:
                if 0 <= idx < len(tgt_tokens):
                    tgt_correction_idx.append(idx)
                    tgt_correction.append(tgt_tokens[idx])
                elif idx > len(tgt_tokens):
                    idx -= len(tgt_tokens) + 1
                    src_correction_idx.append(idx)
                    src_correction.append(src_tokens[idx])
                else:
                    raise ValueError(f"Invalid sample: {raw_sample}")

            # Acquire source_evidence_index and target_evidence_index
            src_evidence_idx, src_evidence = [], []
            tgt_evidence_idx, tgt_evidence = [], []
            for idx in raw_sample["evidence_index"]:
                if 0 <= idx < len(tgt_tokens):
                    if tgt_tokens[idx] not in ["[None]", "-NONE-"]:
                        tgt_evidence_idx.append(idx)
                        tgt_evidence.append(tgt_tokens[idx])
                    else:
                        LOGGER.warning(f"Invalid evidence: `{tgt_tokens[idx]}` ({idx}) in {' '.join(tgt_tokens)}")
                elif idx > len(tgt_tokens):
                    idx -= len(tgt_tokens) + 1
                    if src_tokens[idx] not in ["[None]", "-NONE-"]:
                        src_evidence_idx.append(idx)
                        src_evidence.append(src_tokens[idx])
                    else:
                        LOGGER.warning(f"Invalid evidence: `{src_tokens[idx]}` ({idx}) in {' '.join(src_tokens)}")
                else:
                    raise ValueError(f"Invalid sample: {raw_sample}")

            samples.append(ExplainableSample(
                index=len(samples),
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
                error_type=raw_sample["error_type"],
                src_correction=src_correction,
                tgt_correction=tgt_correction,
                src_correction_index=src_correction_idx,
                tgt_correction_index=tgt_correction_idx,
                src_evidence=src_evidence,
                tgt_evidence=tgt_evidence,
                src_evidence_index=src_evidence_idx,
                tgt_evidence_index=tgt_evidence_idx,
            ))
    return samples


def process_expect(samples: List[ExplainableSample]) -> List[ExplainableSample]:
    """ Created by yejh on 2023.08.22
        Process expect by following rules:
        1) Remove `[NONE]` and `-NONE-` tokens in official dataset
        2) Rearrange correction_index
        3) Rearrange evidence_index
    """
    new_samples = []
    # Number of samples with NONE tokens
    cnt_src_none_token, cnt_tgt_none_token = 0, 0

    for sample in samples:
        src_tokens, tgt_tokens = sample.src_tokens, sample.tgt_tokens
        src_correction_idx, tgt_correction_idx = sample.src_correction_index, sample.tgt_correction_index
        src_evidence_idx, tgt_evidence_idx = sample.src_evidence_index, sample.tgt_evidence_index

        # Handle invalid samples which contain "-NONE-"
        if any([True if x == "-NONE-" else False for x in src_tokens]) \
                or any([True if x == "-NONE-" else False for x in tgt_tokens]):
            LOGGER.warning(f"NONE token in {sample}")
            src_tokens = [x.replace("-NONE-", "[NONE]") for x in src_tokens]
            tgt_tokens = [x.replace("-NONE-", "[NONE]") for x in tgt_tokens]

        src_none_idx = [i for i, x in enumerate(src_tokens) if x == "[NONE]"]
        tgt_none_idx = [i for i, x in enumerate(tgt_tokens) if x == "[NONE]"]
        if len(src_none_idx):
            cnt_src_none_token += 1
        if len(tgt_none_idx):
            cnt_tgt_none_token += 1

        if len(src_none_idx):
            # Rearrange correction_index and evidence_index
            _src_correction_idx, _src_evidence_idx = [], []
            for idx in src_correction_idx:
                if src_tokens[idx] == "[None]":
                    LOGGER.warning(f"Remove [NONE] from correction: {sample}")
                    continue
                if idx > src_none_idx[0]:
                    idx -= len(src_none_idx)
                _src_correction_idx.append(idx)
            src_correction_idx = src_correction_idx

            for idx in src_evidence_idx:
                assert src_tokens[idx] != "[None]", f"[NONE] cannot be evidence, {sample}"
                if idx > src_none_idx[0]:
                    idx -= len(src_none_idx)
                _src_evidence_idx.append(idx)
            src_evidence_idx = _src_evidence_idx
        src_correction = list(filter(lambda x: x not in ["[NONE]", "-NONE-"], sample.src_correction))
        src_evidence = list(filter(lambda x: x not in ["[NONE]", "-NONE-"], sample.src_evidence))

        if len(tgt_none_idx):
            # Rearrange correction_index and evidence_index
            _tgt_correction_idx, _tgt_evidence_idx = [], []
            for idx in tgt_correction_idx:
                if tgt_tokens[idx] == "[None]":
                    LOGGER.warning(f"Remove [NONE] from correction: {sample}")
                    continue
                if idx > tgt_none_idx[0]:
                    idx -= len(tgt_none_idx)
                _tgt_correction_idx.append(idx)
            tgt_correction_idx = _tgt_correction_idx

            for idx in tgt_evidence_idx:
                assert tgt_tokens[idx] != "[None]", f"[NONE] cannot be evidence, {sample}"
                if idx > tgt_none_idx[0]:
                    idx -= len(tgt_none_idx)
                _tgt_evidence_idx.append(idx)
            tgt_evidence_idx = _tgt_evidence_idx
        tgt_correction = list(filter(lambda x: x not in ["[NONE]", "-NONE-"], sample.tgt_correction))
        tgt_evidence = list(filter(lambda x: x not in ["[NONE]", "-NONE-"], sample.tgt_evidence))

        # if len(src_none_idx):
        #     # Assure [NONE] cannot be evidence
        #     assert all([1 if i != src_none_idx[0] else 0 for i in src_evidence_idx]), \
        #         f"[NONE] cannot be evidence, {sample}"
        #     src_evidence_idx = [i - 1 if i > src_none_idx[0] else i for i in src_evidence_idx]
        # if len(tgt_none_idx):
        #     assert all([1 if i != tgt_none_idx[0] else 0 for i in tgt_evidence_idx]), \
        #         f"[NONE] cannot be evidence, {sample}"
        #     tgt_evidence_idx = [i - 1 if i > tgt_none_idx[0] else i for i in tgt_evidence_idx]

        new_samples.append(ExplainableSample(
            index=sample.index,
            src_tokens=list(filter(lambda x: x != "[NONE]", src_tokens)),
            tgt_tokens=list(filter(lambda x: x != "[NONE]", tgt_tokens)),
            error_type=sample.error_type,
            src_correction=src_correction,
            tgt_correction=tgt_correction,
            src_correction_index=src_correction_idx,
            tgt_correction_index=tgt_correction_idx,
            src_evidence=src_evidence,
            tgt_evidence=tgt_evidence,
            src_evidence_index=src_evidence_idx,
            tgt_evidence_index=tgt_evidence_idx,
        ))
    return new_samples


def load_expect_denoise(file_input: str, max_sample: int = -1) -> List[ExplainableSample]:
    samples = []
    with open(file_input, "r", encoding="utf-8") as f:
        for line in f:
            if 0 <= max_sample <= len(samples):
                break
            raw_sample = json.loads(line)
            src_tokens, tgt_tokens = raw_sample["source"], raw_sample["target"]

            # src_evidence = raw_sample["source_evidence"]
            # tgt_evidence = raw_sample["target_evidence"]
            src_evidence_idx = raw_sample["source_evidence_index"]
            # tgt_evidence_idx = raw_sample["target_evidence_index"]

            samples.append(ExplainableSample(
                index=len(samples),
                src_tokens=src_tokens.copy(),
                tgt_tokens=tgt_tokens.copy(),
                src_evidence_index=src_evidence_idx.copy(),
                # tgt_evidence_index=tgt_evidence_idx.copy(),
                error_type=raw_sample["error_type"],
            ))
    return samples


def evaluate_explanation(
        samples: List[ExplainableSample],
        pred_error_type_list: List[str],
        pred_evidence_idx_list: List[List[int]],
        src_word_bpes_list: List[List[List[str]]],
        strict_bpe: bool = False,
):
    """ Created by yejh on 2023.08.19
        Evaluate explanation performance using seqeval
        1) Token-level evaluation: P, R, F
           1.1) A single word is counted only once despite it may consist of bpes
           1.2) Samples labeled with "Others" error type are not considered, following ACL2023
        2) Sentence-level evaluation: EM, Label Accuracy

        Args:
            samples: index=1839
                tgt: However , things did n't happen like what I hoped .
                src: However , things did n't happened like what I hoped .
                word_evidence: 3, 4     did n't
            pred_error_type_list
            pred_evidence_idx_list: bpe_evidence: 3, 4, 5   did n 't
            src_word_bpes_list: [[However], ..., [did], [n, 't], ...]
            strict_bpe

        Build inputs for seqeval
            gold_cls: list[str]
            gold_seq_list: list[list[str]]
            pred_cls: list[str]
            pred_seq_list: list[list[str]]
    """
    pred_seq_list, pred_seq_strict_list = [], []
    gold_error_type_list, gold_seq_list, gold_seq_strict_list = [], [], []

    # Build gold inputs
    for idx, sample in enumerate(samples):
        assert len(sample.src_tokens) == len(src_word_bpes_list[idx])
        gold_error_type = ERROR_TYPE_TOKENS[sample.error_type]
        gold_error_type_list.append(gold_error_type)

        if gold_error_type == "<Others>":
            gold_seq_list.append(["O"] * len(sample.src_tokens))
            gold_seq_strict_list.append(["O"] * len(sample.src_tokens))
        else:
            gold_seq_list.append([
                "B-" if x in sample.src_evidence_index else "O"
                for x in range(len(sample.src_tokens))
            ])
            gold_seq_strict_list.append([
                "B-" + gold_error_type if x in sample.src_evidence_index else "O"
                for x in range(len(sample.src_tokens))
            ])

    # Build pred inputs
    for pred_error_type, pred_evidence_idx, src_word_bpes in zip(
            pred_error_type_list,
            pred_evidence_idx_list,
            src_word_bpes_list,
    ):
        if pred_error_type == "<Others>":
            pred_seq_list.append(["O"] * len(src_word_bpes))
            pred_seq_strict_list.append(["O"] * len(src_word_bpes))
            continue

        bpe2word = {}  # Map bpe_idx to word_idx
        for word_idx, word_bpes in enumerate(src_word_bpes):
            for _ in word_bpes:
                bpe2word[len(bpe2word)] = word_idx

        pred_seq = [0] * len(src_word_bpes)
        for idx in pred_evidence_idx:
            if idx < len(bpe2word):
                pred_seq[bpe2word[idx]] += 1
            else:
                print(f"Error Explanation: {pred_evidence_idx}")
                break

        if strict_bpe:  # A word is considered as evidence only if all its bpes make sense
            pred_seq = [
                True if x == len(src_word_bpes[i]) else False
                for i, x in enumerate(pred_seq)
            ]
        else:
            pred_seq = [True if x > 0 else False for x in pred_seq]

        pred_seq_list.append(["B-" if x else "O" for x in pred_seq])
        pred_seq_strict_list.append(["B-" + pred_error_type if x else "O" for x in pred_seq])

    return calc_explanation_score(
        gold_seq_list,
        pred_seq_list,
        gold_seq_strict_list,
        pred_seq_strict_list,
        gold_error_type_list,
        pred_error_type_list,
        strict_bpe=strict_bpe,
    )

    # # Evaluate by seqeval
    # p = round(precision_score(gold_seq_list, pred_seq_list), 4)
    # r = round(recall_score(gold_seq_list, pred_seq_list), 4)
    # f1 = round(f1_score(gold_seq_list, pred_seq_list), 4)
    # f0_5 = round(1.25 * p * r / (0.25 * p + r), 4)
    # em = round(sum([
    #     1 for g, p in zip(gold_seq_list, pred_seq_list) if g == p]
    # ) / len(pred_seq_list), 4)
    #
    # p_strict = round(precision_score(gold_seq_strict_list, pred_seq_strict_list), 4)
    # r_strict = round(recall_score(gold_seq_strict_list, pred_seq_strict_list), 4)
    # f1_strict = round(f1_score(gold_seq_strict_list, pred_seq_strict_list), 4)
    # f0_5_strict = round(1.25 * p_strict * r_strict / (0.25 * p_strict + r_strict), 4)
    # em_strict = round(sum([
    #     1 for g, p in zip(gold_seq_strict_list, pred_seq_strict_list) if g == p]
    # ) / len(pred_seq_strict_list), 4)
    #
    # acc = round(accuracy_score(gold_error_type_list, pred_error_type_list), 4)
    # return {
    #     "P": p, "R": r, "F1": f1, "F0.5": f0_5, "EM": em,
    #     "P_strict": p_strict, "R_strict": r_strict, "F1_strict": f1_strict,
    #     "F0.5_strict": f0_5_strict, "EM_strict": em_strict,
    #     "ACC": acc, "strict_bpe": strict_bpe,
    # }


def evaluate_tagging(
        samples: List[ExplainableSample],
        tag_preds: List[List[int]],
        src_word_bpes_list: List[List[List[str]]],
        # strict_bpe: bool = False,
):
    """ Created by yejh on 2023.09.11
        Evaluate tagging performance using seqeval
        1) Token-level evaluation: P, R, F
           1.1) A single word is counted only once despite it may consist of bpes
           1.2) Samples labeled with "Others" error type are not considered, following ACL2023
        2) Sentence-level evaluation: EM, Label Accuracy

        Args:
            samples: index=1839
                tgt: However , things did n't happen like what I hoped .
                src: However , things did n't happened like what I hoped .
                word_evidence: 3, 4     did n't
            tag_preds (bpe_evidence): 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0
            src_word_bpes_list: [[However], ..., [did], [n, 't], ...]

        Build inputs for seqeval
            gold_cls: list[str]
            gold_seq_list: list[list[str]]
            pred_cls: list[str]
            pred_seq_list: list[list[str]]
    """
    pred_seq_list, pred_seq_strict_list, pred_error_type_list = [], [], []
    gold_error_type_list, gold_seq_list, gold_seq_strict_list = [], [], []

    # Build gold inputs
    for idx, sample in enumerate(samples):
        assert len(sample.src_tokens) == len(src_word_bpes_list[idx])
        assert len(tag_preds[idx]) == len(list(chain(*src_word_bpes_list[idx]))), \
            f"idx={idx}: {len(tag_preds[idx])}, {len(list(chain(*src_word_bpes_list[idx])))}"

        gold_error_type = ERROR_TYPE_TOKENS[sample.error_type][1:-1]
        gold_error_type_list.append(gold_error_type)

        if gold_error_type == "Others":
            gold_seq_list.append(["O"] * len(sample.src_tokens))
            gold_seq_strict_list.append(["O"] * len(sample.src_tokens))
        else:
            gold_seq_list.append([
                "B-" if x in sample.src_evidence_index else "O"
                for x in range(len(sample.src_tokens))
            ])
            gold_seq_strict_list.append([
                "B-" + gold_error_type if x in sample.src_evidence_index else "O"
                for x in range(len(sample.src_tokens))
            ])

    # Build pred inputs
    for tag_pred, src_word_bpes in zip(
            tag_preds,
            src_word_bpes_list,
    ):
        bpe2word = {}  # Map bpe_idx to word_idx
        for word_idx, word_bpes in enumerate(src_word_bpes):
            for _ in word_bpes:
                bpe2word[len(bpe2word)] = word_idx

        pred_seq = []
        for idx, tag in enumerate(tag_pred):
            if len(pred_seq) <= bpe2word[idx]:
                pred_seq.append([TAGGING_LABELS[tag]])
            else:
                pred_seq[bpe2word[idx]].append(TAGGING_LABELS[tag])
        pred_seq = [x[0] for x in pred_seq]
        pred_seq = ["O" if x == "None" else x for x in pred_seq]

        pred_error_type = [x.split("-")[1] for x in pred_seq if x != "O"]
        if pred_error_type:
            pred_error_type_list.append(pred_error_type[0])
        else:
            pred_error_type_list.append("Others")

        if pred_error_type_list[-1] == "Others":
            pred_seq_list.append(["O"] * len(src_word_bpes))
            pred_seq_strict_list.append(["O"] * len(src_word_bpes))
        else:
            pred_seq_list.append(['B-' if x != "O" else "O" for x in pred_seq])
            pred_seq_strict_list.append(['B-' + x.split('-')[1] if 'I-' in x else x for x in pred_seq])

    return calc_explanation_score(
        gold_seq_list,
        pred_seq_list,
        gold_seq_strict_list,
        pred_seq_strict_list,
        gold_error_type_list,
        pred_error_type_list,
        strict_bpe=False,
    )


def calc_explanation_score(
        gold_seq_list,
        pred_seq_list,
        gold_seq_strict_list,
        pred_seq_strict_list,
        gold_error_type_list,
        pred_error_type_list,
        strict_bpe=False,
):
    # Evaluate by seqeval
    p = round(precision_score(gold_seq_list, pred_seq_list), 4)
    r = round(recall_score(gold_seq_list, pred_seq_list), 4)
    f1 = round(f1_score(gold_seq_list, pred_seq_list), 4)
    f0_5 = round(1.25 * p * r / (0.25 * p + r), 4)
    em = round(sum([
        1 for g, p in zip(gold_seq_list, pred_seq_list) if g == p]
    ) / len(pred_seq_list), 4)
    detail_dict = performance_measure(gold_seq_list, pred_seq_list)

    p_strict = round(precision_score(gold_seq_strict_list, pred_seq_strict_list), 4)
    r_strict = round(recall_score(gold_seq_strict_list, pred_seq_strict_list), 4)
    f1_strict = round(f1_score(gold_seq_strict_list, pred_seq_strict_list), 4)
    f0_5_strict = round(1.25 * p_strict * r_strict / (0.25 * p_strict + r_strict), 4)
    em_strict = round(sum([
        1 for g, p in zip(gold_seq_strict_list, pred_seq_strict_list) if g == p]
    ) / len(pred_seq_strict_list), 4)
    detail_dict_strict = performance_measure(gold_seq_strict_list, pred_seq_strict_list)

    acc = round(accuracy_score(gold_error_type_list, pred_error_type_list), 4)
    return {
        "P": p, "R": r, "F1": f1, "F0.5": f0_5, "EM": em, "ACC": acc,
        "P_strict": p_strict, "R_strict": r_strict, "F1_strict": f1_strict,
        "F0.5_strict": f0_5_strict, "EM_strict": em_strict, "strict_bpe": strict_bpe,
        "TP": detail_dict["TP"], "FP": detail_dict["FP"],
        "FN": detail_dict["FN"], "TN": detail_dict["TN"],
        "TP_strict": detail_dict_strict["TP"], "FP_strict": detail_dict_strict["FP"],
        "FN_strict": detail_dict_strict["FN"], "TN_strict": detail_dict_strict["TN"],
    }
