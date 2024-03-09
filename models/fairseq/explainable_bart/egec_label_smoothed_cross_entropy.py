# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    label_smoothed_nll_loss,
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)

from utils import get_logger
from .egec_language_pair_dataset import SEPERATOR_TOKEN, TAGGING_LABELS

LOGGER = get_logger(__name__)


@dataclass
class ExplainableLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    explanation_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for explanation tokens"}
    )
    tagging_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for sequence tagging"}
    )


# def explainable_label_smoothed_nll_loss(
#         lprobs,
#         target,
#         epsilon,
#         ignore_index=None,
#         reduce=True,
#         exp_mask=None,
#         exp_weight=1.0,
#         vocab_size=None,
# ):
#     if target.dim() == lprobs.dim() - 1:
#         target = target.unsqueeze(-1)  # [B, 1]
#     if exp_mask.dim() == lprobs.dim() - 1:
#         exp_mask = exp_mask.unsqueeze(-1)  # [B, 1]
#
#     nll_loss = -lprobs.gather(dim=-1, index=target)  # [B, 1]
#
#     if exp_mask is None:
#         smooth_loss = -lprobs.sum(dim=-1, keepdim=True)  # [B, 1]
#     else:
#
#     if ignore_index is not None:
#         pad_mask = target.eq(ignore_index)
#         nll_loss.masked_fill_(pad_mask, 0.0)
#         smooth_loss.masked_fill_(pad_mask, 0.0)
#     else:
#         nll_loss = nll_loss.squeeze(-1)
#         smooth_loss = smooth_loss.squeeze(-1)
#
#     loss_cor, loss_exp, nll_loss_cor, nll_loss_exp = None, None, None, None
#     # Apply weight for explanation part
#     if exp_mask is not None:
#         nll_loss_cor = nll_loss.masked_fill(exp_mask, 0.0)
#         nll_loss_exp = nll_loss.masked_fill(exp_mask.eq(0), 0.0)
#         smooth_loss_cor = smooth_loss.masked_fill_(exp_mask, 0.0)
#         smooth_loss_exp = smooth_loss.masked_fill_(exp_mask.eq(0), 0.0)
#         if reduce:
#             nll_loss_cor = nll_loss_cor.sum()
#             nll_loss_exp = nll_loss_exp.sum()
#             smooth_loss_cor = smooth_loss_cor.sum()
#             smooth_loss_exp = smooth_loss_exp.sum()
#
#         eps_cor = epsilon / (vocab_size - 1)
#         eps_exp = epsilon / (vocab_size - 1)
#
#         loss_cor = (1.0 - epsilon - eps_i) * nll_loss_cor + eps_i * smooth_loss_cor
#         loss_exp = (1.0 - epsilon - eps_i) * nll_loss_exp + eps_i * smooth_loss_exp
#         loss = loss_cor + exp_weight * loss_exp
#         nll_loss = nll_loss_cor + exp_weight * nll_loss_exp
#     else:
#         if reduce:
#             nll_loss = nll_loss.sum()
#             smooth_loss = smooth_loss.sum()
#         eps_i = epsilon / (lprobs.size(-1) - 1)
#         loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
#
#     return {
#         "loss": loss,
#         "nll_loss": nll_loss,
#         "loss_cor": loss_cor,
#         "loss_exp": loss_exp,
#         "nll_loss_cor": nll_loss_cor,
#         "nll_loss_exp": nll_loss_exp,
#     }


@register_criterion(
    "explainable_label_smoothed_cross_entropy",
    dataclass=ExplainableLabelSmoothedCrossEntropyCriterionConfig,
)
class ExplainableLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            explanation_weight=1.0,
            tagging_weight=1.0,
            label_smoothing_exp=0.0,
            label_smoothing_tag=0.1,
    ):
        super().__init__(
            task=task,
            sentence_avg=sentence_avg,
            label_smoothing=label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy,
        )
        self.explanation_weight = explanation_weight
        self.tagging_weight = tagging_weight
        self.eps_exp = label_smoothing_exp
        self.eps_tag = label_smoothing_tag
        LOGGER.info(f"Build ExplainableLabelSmoothedCrossEntropyCriterion - "
                    f"explanation_weight: {self.explanation_weight}, "
                    f"tagging_weight: {self.tagging_weight}, "
                    f"exp_cor: {self.eps}, "
                    f"eps_exp: {self.eps_exp}, "
                    f"eps_tag: {self.eps_tag}")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss_dict = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        logging_output = {
            "loss": loss_dict["loss"].data,
            "loss_cor": loss_dict["loss_cor"].data,
            "loss_exp": loss_dict["loss_exp"].data,
            "loss_tag": loss_dict["loss_tag"].data,
            "nll_loss": loss_dict["nll_loss"].data,
            "nll_loss_cor": loss_dict["nll_loss_cor"].data,
            "nll_loss_exp": loss_dict["nll_loss_exp"].data,
            "nll_loss_tag": loss_dict["nll_loss_tag"].data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss_dict["loss"], sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        # [B, V+LS], [B,]
        lprobs, target, tag_lprobs, tag_target = self.get_lprobs_and_target(model, net_output, sample)

        # Split net_output into correction and explanation parts
        sep_idx = model.encoder.dictionary.index(SEPERATOR_TOKEN)
        exp_mask = target.ge(sep_idx)

        lprobs_cor, lprobs_exp = lprobs[exp_mask.eq(0), :], lprobs[exp_mask, :]
        target_cor, target_exp = target[exp_mask.eq(0)], target[exp_mask]

        # Shrink logit space
        lprobs_cor, lprobs_exp = lprobs_cor[:, :sep_idx], lprobs_exp[:, sep_idx:]
        target_exp = target_exp - sep_idx

        loss_cor, nll_loss_cor = label_smoothed_nll_loss(
            lprobs_cor,
            target_cor,
            epsilon=self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        loss_exp, nll_loss_exp = label_smoothed_nll_loss(
            lprobs_exp,
            target_exp,
            epsilon=self.eps_exp,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        loss_tag, nll_loss_tag = loss_exp.new(1).fill_(0.0), loss_exp.new(1).fill_(0.0)
        if tag_lprobs is not None and tag_target is not None:
            loss_tag, nll_loss_tag = label_smoothed_nll_loss(
                tag_lprobs,
                tag_target,
                epsilon=self.eps_tag,
                ignore_index=TAGGING_LABELS.index("None"),
                reduce=reduce,
            )
        loss = loss_cor + self.explanation_weight * loss_exp + self.tagging_weight * loss_tag
        nll_loss = nll_loss_cor + self.explanation_weight * nll_loss_exp + self.tagging_weight * nll_loss_tag

        return {
            "loss": loss,
            "nll_loss": nll_loss,
            "loss_cor": loss_cor,
            "loss_exp": loss_exp,
            "loss_tag": loss_tag,
            "nll_loss_cor": nll_loss_cor,
            "nll_loss_exp": nll_loss_exp,
            "nll_loss_tag": nll_loss_tag,
        }

    def get_lprobs_and_target(self, model, net_output, sample):
        # Adapt to Explainable Transformer with Sequence Tagging
        lprobs, tag_lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target, tag_target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        return (
            lprobs.view(-1, lprobs.size(-1)),
            target.view(-1),
            tag_lprobs.view(-1, tag_lprobs.size(-1)) if tag_lprobs is not None else None,
            tag_target.view(-1) if tag_target is not None else None,
        )

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        loss_cor_sum = sum(log.get("loss_cor", 0) for log in logging_outputs)
        loss_exp_sum = sum(log.get("loss_exp", 0) for log in logging_outputs)
        loss_tag_sum = sum(log.get("loss_tag", 0) for log in logging_outputs)
        nll_loss_cor_sim = sum(log.get("nll_loss_cor", 0) for log in logging_outputs)
        nll_loss_exp_sim = sum(log.get("nll_loss_exp", 0) for log in logging_outputs)
        nll_loss_tag_sim = sum(log.get("nll_loss_tag", 0) for log in logging_outputs)

        metrics.log_scalar("loss_cor", loss_cor_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("loss_exp", loss_exp_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("loss_tag", loss_tag_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_cor", nll_loss_cor_sim / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_exp", nll_loss_exp_sim / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss_tag", nll_loss_tag_sim / sample_size / math.log(2), sample_size, round=3)
