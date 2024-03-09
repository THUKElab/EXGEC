# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.data import LanguagePairDataset, data_utils

from utils import get_logger

LOGGER = get_logger(__name__)
SEPERATOR_TOKEN = "<sep>"
ERROR_TYPE_TOKENS = {
    "None": "<None>",
    # Syntax
    "Infinitives": "<Infinitives>",
    "Gerund": "<Gerund>",
    "Participle": "<Participle>",
    "Subject-Verb Agreement": "<Subject_Verb_Agreement>",
    "Auxiliary Verb": "<Auxiliary_Verb>",
    "Pronoun-Antecedent Agreement": "<Pronoun_Antecedent_Agreement>",
    "Possessive": "<Possessive>",
    # Morphology
    "Collocation": "<Collocation>",
    "Preposition": "<Preposition>",
    "POS Confusion": "<POS_Confusion>",
    "Number": "<Number>",
    "Transitive Verb": "<Transitive_Verb>",
    # Discourse Level
    "Verb Tense": "<Verb_Tense>",
    "Article": "<Article>",
    "Others": "<Others>",
}
# Labels for sequence tagging
# TAGGING_LABELS = [
#     "O", "None",  # Add `None` as padding index for criterion
#     "B-Infinitives", "I-Infinitives",
#     "B-Gerund", "I-Gerund",
#     "B-Participle", "I-Participle",
#     "B-Subject_Verb_Agreement", "I-Subject_Verb_Agreement",
#     "B-Auxiliary_Verb", "I-Auxiliary_Verb",
#     "B-Pronoun_Antecedent_Agreement", "I-Pronoun_Antecedent_Agreement",
#     "B-Possessive", "I-Possessive",
#     "B-Collocation", "I-Collocation",
#     "B-Preposition", "I-Preposition",
#     "B-POS_Confusion", "I-POS_Confusion",
#     "B-Number", "I-Number",
#     "B-Transitive_Verb", "I-Transitive_Verb",
#     "B-Verb_Tense", "I-Verb_Tense",
#     "B-Article", "I-Article",
#     "B-Others", "I-Others",
# ]
TAGGING_LABELS = [
    "O", "None",  # Add `None` as padding index for criterion
    "B-Infinitives",
    "B-Gerund",
    "B-Participle",
    "B-Subject_Verb_Agreement",
    "B-Auxiliary_Verb",
    "B-Pronoun_Antecedent_Agreement",
    "B-Possessive",
    "B-Collocation",
    "B-Preposition",
    "B-POS_Confusion",
    "B-Number",
    "B-Transitive_Verb",
    "B-Verb_Tense",
    "B-Article",
    "B-Others",
]


def collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=True,
        left_pad_target=False,
        input_feeding=True,
        pad_to_length=None,
        pad_to_multiple=1,
):
    """ Revised by yejh on 2023.08.09
        Build Explainable Samples
        batch = {
            'id': torch.tensor,
            'nsentences': int,
            'ntokens': int,
            'net_input': {
                'src_tokens': torch.tensor,
                'src_lengths': torch.tensor,
                'prev_output_tokens': torch.tensor,
            },
            'target': torch.tensor,
            'tagging_target': torch.tensor,  # if sequence_tagging=True
        }
    """
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, pad_idx=1, move_eos_to_beginning=False, pad_to_length=None):
        # Convert a list of 1d tensors into a padded 2d tensor
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    ids = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )

    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    src_tokens = src_tokens.index_select(0, sort_order)
    ids = ids.index_select(0, sort_order)

    prev_output_tokens, target = None, None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"] if pad_to_length is not None else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"] if pad_to_length is not None else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    # Handle tagging_target
    tagging_target = None
    if samples[0].get("tagging_target", None) is not None:
        tagging_target = merge(
            "tagging_target",
            left_pad=left_pad_source,
            pad_idx=TAGGING_LABELS.index("None"),
            pad_to_length=pad_to_length["tagging_target"] if pad_to_length is not None else None,
        )
        tagging_target = tagging_target.index_select(0, sort_order)

    batch = {
        "id": ids,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(0, sort_order)
    if tagging_target is not None:
        batch["tagging_target"] = tagging_target
    return batch


class ExplainableLanguagePairDataset(LanguagePairDataset):
    def __init__(
            self,
            src,
            src_sizes,
            src_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=None,
            left_pad_source=True,
            left_pad_target=False,
            shuffle=True,
            input_feeding=True,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            align_dataset=None,
            constraints=None,
            append_bos=False,
            eos=None,
            num_buckets=0,
            src_lang_id=None,
            tgt_lang_id=None,
            pad_to_multiple=1,
            explanation=None,
            explanation_sizes=None,
            explanation_setting=None,
            explanation_format=None,
            explanation_before=False,
            sequence_tagging=False,
    ):
        assert left_pad_source == left_pad_target == False
        super().__init__(
            src,
            src_sizes,
            src_dict,
            tgt=tgt,
            tgt_sizes=tgt_sizes,
            tgt_dict=tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            shuffle=shuffle,
            input_feeding=input_feeding,
            remove_eos_from_source=remove_eos_from_source,
            append_eos_to_target=append_eos_to_target,
            align_dataset=align_dataset,
            constraints=constraints,
            append_bos=append_bos,
            eos=eos,
            num_buckets=num_buckets,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
            pad_to_multiple=pad_to_multiple,
        )

        self.error_types, self.error_types_index = [], {}
        self.explanation = explanation
        self.explanation_sizes = explanation_sizes
        self.explanation_setting = explanation_setting
        self.explanation_format = explanation_format
        self.explanation_before = explanation_before
        self.sequence_tagging = sequence_tagging
        self.num_labels = len(TAGGING_LABELS)
        self.id2label = {i: label for i, label in enumerate(TAGGING_LABELS)}
        self.label2id = {label: i for i, label in enumerate(TAGGING_LABELS)}

    def error_type_dict_index(self, index):
        return self.error_types_index[self.error_types[index.item()]]

    def init_error_types(self):
        if not self.error_types:
            for k, v in ERROR_TYPE_TOKENS.items():
                if self.src_dict.index(v) == self.src_dict.unk():
                    LOGGER.info(f"Add Error Type {v}: {len(self.src_dict)}")

                self.error_types.append(v)
                self.error_types_index[v] = self.src_dict.add_symbol(v)
                error_types_index = self.tgt_dict.add_symbol(v)
                assert self.error_types_index[v] == error_types_index, \
                    f"Invalid dictionary: {v} | {self.error_types_index[v]} | {error_types_index}"

    def reorder_explanation(self, explanation_format, explanation_item):
        format_split = explanation_format.split("-")
        error_type_item, evidence_item = None, None
        if "type" in format_split:
            error_type_item = self.error_type_dict_index(explanation_item[0])
        if "evidence" in format_split:
            evidence_item = explanation_item[1:] + len(self.src_dict)

        if format_split[0] == "type":
            explanation_item[0] = error_type_item
            if evidence_item is not None:
                explanation_item[1:] = evidence_item
        elif format_split[0] == "evidence":
            explanation_item[:-1] = evidence_item
            if error_type_item is not None:
                explanation_item[-1] = error_type_item
            else:
                explanation_item = evidence_item
        return explanation_item

    def build_tagging_target(self, src_item, explanation_item, simple=False):
        """ Created by yejh on 2023.09.08
            1) Build Tagging Target

            @param src_item: source tokens
            @param explanation_item: explanation
            @param simple:
        """
        src_len = len(src_item)
        error_type = self.error_types[explanation_item[0].item()][1:-1]  # e.g. Infinitives
        evidence_item = explanation_item[1:]
        tagging_target = [self.label2id["O"]] * src_len

        for idx, evi_idx in enumerate(evidence_item):
            if simple:
                tagging_target[evi_idx] = self.label2id["B-" + error_type]
            else:
                if evi_idx == 0 or tagging_target[evi_idx - 1] == self.label2id["O"]:
                    tagging_target[evi_idx] = self.label2id["B-" + error_type]
                else:
                    tagging_target[evi_idx] = self.label2id["I-" + error_type]
        return tagging_target

    def __getitem__(self, index):
        """ Return a sample for collater
            example = {
                "id": index,
                "source": src_item,
                "target": tgt_item,
                "evidence": evidence,
                "error_type": error_type,
            }
        """
        if not self.error_types:
            self.init_error_types()

        example = super().__getitem__(index)
        src_item, tgt_item = example["source"], example["target"]
        assert src_item[-1] == self.eos

        if self.sequence_tagging and tgt_item is not None:
            tagging_target = self.build_tagging_target(
                src_item=src_item,
                explanation_item=self.explanation[index],
                simple=True,
            )
            example["tagging_target"] = torch.LongTensor(tagging_target)

        if self.explanation_setting and self.explanation_format:
            if self.explanation_setting != "infusion" and tgt_item is None:
                return example

            sep_index = self.src_dict.index(SEPERATOR_TOKEN)
            explanation_item = self.reorder_explanation(
                self.explanation_format,
                self.explanation[index],
            )

            if self.explanation_setting == "infusion":  # [Source] <sep> [Explanation]
                src_item[-1] = sep_index
                example["source"] = torch.cat(
                    (src_item, explanation_item, src_item.new(1).fill_(self.eos)),
                    dim=0,
                )
            elif self.explanation_setting == "explanation":
                example["target"] = torch.cat(
                    (explanation_item, tgt_item.new(1).fill_(self.eos)),
                    dim=0,
                )
            elif self.explanation_setting == "rationalization":
                if self.explanation_before:  # [Explanation] <sep> [Correction]
                    example["target"] = torch.cat(
                        (explanation_item, tgt_item.new(1).fill_(sep_index), tgt_item),
                        dim=0,
                    )
                else:  # [Correction] <sep> [Explanation]
                    tgt_item[-1] = sep_index
                    example["target"] = torch.cat(
                        (tgt_item, explanation_item, tgt_item.new(1).fill_(self.eos)),
                        dim=0,
                    )
        return example

    def collater(self, samples, pad_to_length=None):
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    # def num_tokens(self, index):
    #     """ Return the number of tokens in a sample. This value is used to
    #         enforce ``--max-tokens`` during batching.
    #     """
    #     if not self.explanation_setting:
    #         return max(self.src_sizes[index], self.tgt_sizes[index])
    #     elif self.explanation_setting == "infusion":
    #         return max(
    #             self.src_sizes[index] + self.explanation_sizes[index] + 1,
    #             self.tgt_sizes[index],
    #         )
    #     elif self.explanation_setting == "rationalization":
    #         return max(
    #             self.src_sizes[index],
    #             self.tgt_sizes[index] + self.explanation_sizes[index] + 1,
    #         )
    #     else:
    #         raise NotImplementedError
    #
    # def num_tokens_vec(self, indices):
    #     """ Return the number of tokens for a set of positions defined by indices.
    #         This value is used to enforce ``--max-tokens`` during batching.
    #     """
    #     if not self.explanation_setting:
    #         sizes = np.maximum(self.src_sizes[indices], self.tgt_sizes[indices])
    #     elif self.explanation_setting == "infusion":
    #         sizes = np.maximum(
    #             self.src_sizes[indices] + self.explanation_sizes[indices] + 1,
    #             self.tgt_sizes[indices],
    #         )
    #     elif self.explanation_setting == "rationalization":
    #         sizes = np.maximum(
    #             self.src_sizes[indices],
    #             self.tgt_sizes[indices] + self.explanation_sizes[indices] + 1,
    #         )
    #     else:
    #         raise NotImplementedError
    #     return sizes
