# This file customizes some specific implementation for GEC task
import itertools
import os
import time
from dataclasses import dataclass, field

import math
import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.token_generation_constraints import unpack_constraints
from fairseq_cli.interactive import make_batches, get_symbols_to_strip_from_output

from data import DATA_GEC_EN, DATA_GEC_ZH, M2DataReader, Dataset, Sample
from data.augmenters import MixEditAugmenter
from utils import get_logger
from .language_pair_dataset_augmented import AugmentedLanguagePairDataset

LOGGER = get_logger(__name__)


def load_augmented_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        prepend_bos_src=None,
        augmenter=None,
        corrupt_target=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        LOGGER.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        LOGGER.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return AugmentedLanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        augmenter=augmenter,
        corrupt_target=corrupt_target,
    )


@dataclass
class GECConfig(TranslationConfig):
    # options for reporting GEC scores during validation
    eval_gec: bool = field(
        default=False,
        metadata={"help": "evaluation with BLEU scores"}
    )
    eval_gec_min_update: int = field(
        default=0,
        metadata={"help": "only evaluate gec if num_updates > eval_gec_min_update"}
    )
    eval_gec_dataset: str = field(
        default="wi_dev",
        metadata={"help": "evaluation dataset"}
    )
    eval_gec_dataset_path: str = field(
        default="",
        metadata={"help": "path of evaluation dataset"}
    )
    # eval_gec_input_path: str = field(
    #     default="",
    #     # default=os.path.join(os.path.dirname(__file__), "preprocess/zho/mucgec_dev/MuCGEC_dev.seg.char.src"),
    #     metadata={"help": "evaluation input"}
    # )
    eval_gec_output_prefix: str = field(
        default="temp",
        metadata={"help": ""}
    )
    eval_gec_metric: str = field(
        default="errant_eng", metadata={
            "help": "GEC metric",
            "choices": ["errant_eng", "errant_zho", "m2", "gleu"],
        },
    )
    eval_gec_sent_level: bool = field(
        default=False,
        metadata={"help": "evaluation with sentence-level metric"}
    )

    # options for task-specific data augmentation
    augmentation_schema: str = field(
        default="none",
        metadata={
            "help": "augmentation schema: e.g. `cut_off`, `src_cut_off`, `trg_cut_off`",
            "choices": ["none", "cut_off", "src_cut_off", "trg_cut_off", "copy"],
        }
    )
    augmentation_enable_mixedit: bool = field(
        default=False,
        metadata={
            "help": "use MixEdit",
        }
    )
    file_dataset_m2: str = field(
        default="",
        metadata={
            "help": "used for mixedit augmentation schema"
        }
    )
    file_pattern: str = field(
        default="",
        metadata={
            "help": "json file of edit_pattern",
        }
    )
    mixedit_temperature: float = field(
        default=1.0,
        metadata={"help": "used for reshaping pattern distribution"}
    )
    mixedit_filter_pattern_min_freq: int = field(
        default=3,
        metadata={"help" :"filter patterns whose frequency < value"}
    )
    mixedit_filter_pattern_max_diff: int = field(
        default=2,
        metadata={"help": "filter patterns whose length difference > value"}
    )
    mixedit_enable_extend_pattern: bool = field(
        default=False,
        metadata={"help": "where extend error patterns"}
    )
    mixedit_min_pattern: int = field(
        default=2,
        metadata={"help": "min pattern for extending error patterns"}
    )
    mixedit_copy: bool = field(
        default=False,
        metadata={"help": "copy target to source"}
    )
    mixedit_remove_bpe: str = field(
        default="",
        metadata={"help": "remove bpe for Chinese dataset"}
    )
    augmentation_pattern_noise_rate: float = field(
        default=0.0,
        metadata={"help": "rate for pattern noising"}
    )
    augmentation_pattern_noise_step: int = field(
        default=0,
        metadata={"help": "steps for pattern noising"}
    )
    augmentation_merge_sample: bool = field(
        default=False,
        metadata={"help": "merge original and augmented samples together"}
    )
    augmentation_masking_schema: str = field(
        default="word",
        metadata={
            "help": "augmentation masking schema: e.g. `word`, `span`",
            "choices": ["word", "span"],
        }
    )
    augmentation_masking_probability: float = field(
        default=0.15,
        metadata={"help": "augmentation masking probability"}
    )
    augmentation_replacing_schema: str = field(
        default="mask",
        metadata={
            "help": "augmentation replacing schema: e.g. `mask`, `random`, `mixed`",
            "choices": ["mask", "random", "mixed"],
        }
    )
    augmentation_span_type: str = field(
        default="sample",
        metadata={
            "help": "augmentation span type e.g. sample, w_sample, ws_sample, etc.",
            "choices": ["sample", "w_sample", "ws_sample"],
        }
    )
    augmentation_span_len_dist: str = field(
        default="geometric",
        metadata={"help": "augmentation span length distribution e.g. geometric, poisson, etc."}
    )
    augmentation_max_span_len: int = field(
        default=10,
        metadata={"help": "augmentation maximum span length"}
    )
    augmentation_min_num_spans: int = field(
        default=5,
        metadata={"help": "augmentation minimum number of spans"}
    )
    augmentation_geometric_prob: float = field(
        default=0.2,
        metadata={"help": "augmentation minimum number of spans"}
    )
    augmentation_poisson_lambda: float = field(
        default=5.0,
        metadata={"help": "augmentation lambda of poisson distribution"}
    )


@register_task("gec", dataclass=GECConfig)
class GECTask(TranslationTask):
    def __init__(self, cfg: GECConfig, src_dict, tgt_dict, score_key="F"):
        super().__init__(cfg, src_dict, tgt_dict)
        self.cfg: GECConfig = cfg
        self.cfg_all = None
        self.trainer = None
        self.bpe = None
        self.tokenizer = None
        self.metric = None
        self.eval_data = None
        self.eval_input = None
        self.sequence_generator = None
        self.model_score = []
        self.augmenter = None
        self.score_key = score_key

    def build_model(self, cfg: GECConfig, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        if cfg.eval_gec:
            assert cfg.eval_gec_dataset_path
            self.eval_data = M2DataReader().read(cfg.eval_gec_dataset_path)
            self.eval_input = [x.source[0] for x in self.eval_data]

            # self.eval_data = M2DataReader().read(
            #     DATA_GEC_EN[cfg.eval_gec_dataset]["errant"]
            #     if cfg.eval_gec_dataset in DATA_GEC_EN else DATA_GEC_ZH[cfg.eval_gec_dataset]["errant"]
            # )
            #
            # if cfg.eval_gec_dataset_path:
            #     with open(cfg.eval_gec_dataset_path, "r", encoding="utf-8") as f:
            #         self.eval_input = [x.strip() for x in f]
            # else:
            #     self.eval_input = [x.source[0] for x in self.eval_data]

            # self.tokenizer = encoders.build_tokenizer(self.cfg_all.tokenizer)
            if self.cfg_all.bpe is not None and self.cfg_all.bpe._name == "gpt2":
                self.bpe = GPT2BPE(self.cfg_all.bpe)

            # build_generator 会根据 generation_args 修改 MultiheadAttention 的 beam_size
            self.sequence_generator = self.build_generator([model], self.cfg_all.generation)
            self.set_beam_size(1)

            if cfg.eval_gec_metric == "errant_eng":
                from metrics import ErrantEN, SystemScorer
                self.metric = ErrantEN(SystemScorer())
            elif cfg.eval_gec_metric == "errant_zho":
                from metrics import ErrantZH, SystemScorer
                from utils.pre_processors.pre_process_chinese import pre_process
                self.metric = ErrantZH(SystemScorer())
                self.eval_input_processed, self.eval_input_ids = pre_process(
                    self.eval_input,
                    file_vocab=os.path.join(os.path.dirname(__file__), "preprocess/zho/vocab_v2.txt"),
                )
                LOGGER.info(f"Split {len(self.eval_input)} into {self.eval_input_ids} sub-sentences.")
            else:
                raise ValueError

        if cfg.augmentation_enable_mixedit:
            assert cfg.file_dataset_m2
            dataset = M2DataReader().read(cfg.file_dataset_m2)
            self.augmenter = MixEditAugmenter(
                temperature=cfg.mixedit_temperature,
                enable_filter_pattern=True,
                filter_pattern_min_freq=cfg.mixedit_filter_pattern_min_freq,
                filter_pattern_max_diff=cfg.mixedit_filter_pattern_max_diff,
                enable_extend_pattern=cfg.mixedit_enable_extend_pattern,
                min_pattern=cfg.mixedit_min_pattern,
                pattern_noise_rate=cfg.augmentation_pattern_noise_rate,
                pattern_noise_step=cfg.augmentation_pattern_noise_step,
                file_pattern=cfg.file_pattern,
                verbose=True,
                encode_fn=self.encode_fn,
                decode_fn=self.decode_fn,
                remove_bpe=cfg.mixedit_remove_bpe if cfg.mixedit_remove_bpe else None,
            )
            self.augmenter.setup(
                data=dataset,
                build_tgt2sample=True,
            )
        return model

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if split != "train" or not self.cfg.augmentation_enable_mixedit:
            super().load_dataset(
                split,
                epoch=epoch,
                combine=combine,
                kwargs=kwargs,
            )
            return

        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = load_augmented_langpair_dataset(
            data_path,
            split,
            self.cfg.source_lang,
            self.src_dict,
            self.cfg.target_lang,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            augmenter=self.augmenter,
        )

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False,
    ):
        sample = self.augment_sample(sample)
        return super().train_step(
            sample=sample,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            update_num=update_num,
            ignore_grad=ignore_grad,
        )

    def augment_sample(self, sample):
        if self.cfg.augmentation_schema in ["none"]:
            return sample

        if "secondary" in sample:
            augmented_sample = sample["secondary"]
        else:
            augmented_sample = {
                'id': sample['id'].clone(),
                'nsentences': sample['nsentences'],
                'ntokens': sample['ntokens'],
                'net_input': {
                    'src_tokens': None,
                    'src_lengths': sample['net_input']['src_lengths'].clone(),
                    'prev_output_tokens': None,
                },
                'target': sample['target'].clone()
            }

        if self.cfg.augmentation_schema == 'cut_off':
            augmented_sample['net_input']['src_tokens'] = self._mask_tokens(
                augmented_sample['net_input']['src_tokens']
                if "secondary" in sample else sample['net_input']['src_tokens'],
                self.src_dict,
            )
            # 使用 src_dict 而不是 tgt_dict，因为当使用 transformer 模型时会出现未知原因导致 tgt_dict 扩容
            augmented_sample['net_input']['prev_output_tokens'] = self._mask_tokens(
                augmented_sample['net_input']['prev_output_tokens']
                if "secondary" in sample else sample['net_input']['prev_output_tokens'],
                self.src_dict,
            )
        elif self.cfg.augmentation_schema == 'src_cut_off':
            augmented_sample['net_input']['src_tokens'] = self._mask_tokens(
                augmented_sample['net_input']['src_tokens']
                if "secondary" in sample else sample['net_input']['src_tokens'],
                self.src_dict,
            )
            if not "secondary" in sample:
                augmented_sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].clone()
        elif self.cfg.augmentation_schema == 'trg_cut_off':
            if not "secondary" in sample:
                augmented_sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].clone()
            augmented_sample['net_input']['prev_output_tokens'] = self._mask_tokens(
                augmented_sample['net_input']['prev_output_tokens']
                if "secondary" in sample else sample['net_input']['prev_output_tokens'],
                self.src_dict,
            )
        elif self.cfg.augmentation_schema == "copy":
            # Just for debug
            augmented_sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].clone()
            augmented_sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].clone()
        else:
            raise ValueError("Augmentation schema {0} is not supported".format(self.cfg.augmentation_schema))

        if self.cfg.augmentation_merge_sample:
            sample = {
                'id': torch.cat((sample['id'], augmented_sample['id']), dim=0),
                'nsentences': sample['nsentences'] + augmented_sample['nsentences'],
                'ntokens': sample['ntokens'] + augmented_sample['ntokens'],
                'net_input': {
                    'src_tokens': torch.cat(
                        [
                            sample['net_input']['src_tokens'],
                            augmented_sample['net_input']['src_tokens'],
                        ], dim=0,
                    ),
                    'src_lengths': torch.cat(
                        [
                            sample['net_input']['src_lengths'],
                            augmented_sample['net_input']['src_lengths'],
                        ], dim=0,
                    ),
                    'prev_output_tokens': torch.cat(
                        [
                            sample['net_input']['prev_output_tokens'],
                            augmented_sample['net_input']['prev_output_tokens'],
                        ], dim=0
                    ),
                },
                'target': torch.cat(
                    [
                        sample['target'],
                        augmented_sample['target'],
                    ], dim=0,
                )
            }
        elif "secondary" not in sample:
            sample = {
                'primary': sample,
                'secondary': augmented_sample,
            }
        return sample

    def _mask_tokens(self, inputs, vocab_dict):
        if self.cfg.augmentation_masking_schema == 'word':
            masked_inputs = self._mask_tokens_by_word(inputs, vocab_dict)
        elif self.cfg.augmentation_masking_schema == 'span':
            masked_inputs = self._mask_tokens_by_span(inputs, vocab_dict)
        else:
            raise ValueError("The masking schema {0} is not supported".format(self.cfg.augmentation_masking_schema))
        return masked_inputs

    def _mask_tokens_by_word(self, inputs, vocab_dict):
        vocab_size = len(vocab_dict)
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()

        available_token_indices = (inputs != bos_index) & (inputs != eos_index) \
                                  & (inputs != pad_index) & (inputs != unk_index)
        random_masking_indices = torch.bernoulli(torch.full(
            inputs.shape,
            self.cfg.augmentation_masking_probability,
            device=inputs.device,
        )).bool()

        masked_inputs = inputs.clone()
        masking_indices = random_masking_indices & available_token_indices
        self._replace_token(masked_inputs, masking_indices, unk_index, vocab_size)

        return masked_inputs

    def _mask_tokens_by_span(self, inputs, vocab_dict):
        vocab_size = len(vocab_dict)
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()

        span_info_list = self._generate_spans(inputs)

        num_spans = len(span_info_list)
        masked_span_list = np.random.binomial(
            1,
            self.cfg.augmentation_masking_probability,
            size=num_spans,
        ).astype(bool)
        masked_span_list = [span_info_list[i] for i, masked in enumerate(masked_span_list) if masked]

        available_token_indices = (inputs != bos_index) & (inputs != eos_index) \
                                  & (inputs != pad_index) & (inputs != unk_index)
        random_masking_indices = torch.zeros_like(inputs)
        for batch_index, seq_index, span_length in masked_span_list:
            random_masking_indices[batch_index, seq_index:seq_index + span_length] = 1

        masked_inputs = inputs.clone()
        masking_indices = random_masking_indices.bool() & available_token_indices
        self._replace_token(
            masked_inputs,
            masking_indices,
            unk_index,
            vocab_size,
        )
        return masked_inputs

    def _sample_span_length(self, span_len_dist, max_span_len, geometric_prob=0.2, poisson_lambda=5.0):
        if span_len_dist == 'geometric':
            span_length = min(np.random.geometric(geometric_prob) + 1, max_span_len)
        elif span_len_dist == 'poisson':
            span_length = min(np.random.poisson(poisson_lambda) + 1, max_span_len)
        else:
            span_length = np.random.randint(max_span_len) + 1
        return span_length

    def _get_default_spans(self, batch_index, seq_length, num_spans):
        span_length = int((seq_length - 2) / num_spans)
        last_span_length = seq_length - 2 - (num_spans - 1) * span_length
        span_infos = []
        for i in range(num_spans):
            span_info = (batch_index, 1 + i * span_length, span_length if i < num_spans - 1 else last_span_length)
            span_infos.append(span_info)

        return span_infos

    def _generate_spans(self, inputs):
        if self.cfg.augmentation_span_type == 'sample':
            span_info_list = self._generate_spans_by_sample(inputs)
        elif self.cfg.augmentation_span_type == 'w_sample':
            span_info_list = self._generate_spans_by_w_sample(inputs)
        elif self.cfg.augmentation_span_type == 'ws_sample':
            span_info_list = self._generate_spans_by_ws_sample(inputs)
        else:
            raise ValueError("Span type {0} is not supported".format(self.cfg.augmentation_span_type))

        return span_info_list

    def _generate_spans_by_sample(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]
        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index <= max_index:
                span_length = self._sample_span_length(
                    self.cfg.augmentation_span_len_dist,
                    self.cfg.augmentation_max_span_len,
                    self.cfg.augmentation_geometric_prob,
                    self.cfg.augmentation_poisson_lambda,
                )
                span_length = min(span_length, max_index - seq_index + 1)
                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.cfg.augmentation_min_num_spans:
                span_infos = self._get_default_spans(
                    batch_index,
                    seq_length,
                    self.cfg.augmentation_min_num_spans,
                )
            span_info_list.extend(span_infos)
        return span_info_list

    def _generate_spans_by_w_sample(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]
        input_words = ((inputs & ((1 << 25) - 1)) >> 16) - 1
        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index < max_index:
                span_length = self._sample_span_length(
                    self.cfg.augmentation_span_len_dist,
                    self.cfg.augmentation_max_span_len,
                    self.cfg.augmentation_geometric_prob,
                    self.cfg.augmentation_poisson_lambda,
                )
                span_length = min(span_length, max_index - seq_index + 1)

                word_id = input_words[batch_index, seq_index + span_length - 1]
                if word_id >= 0:
                    word_index = (input_words[batch_index, :] == word_id + 1).nonzero().squeeze(-1)
                    span_length = (word_index[0] - seq_index).item() if word_index.nelement() > 0 \
                        else max_index - seq_index + 1

                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.cfg.augmentation_min_num_spans:
                span_infos = self._get_default_spans(
                    batch_index,
                    seq_length,
                    self.cfg.augmentation_min_num_spans,
                )
            span_info_list.extend(span_infos)
        return span_info_list

    def _generate_spans_by_ws_sample(self, inputs):
        batch_size, seq_length = inputs.size()[0], inputs.size()[1]
        input_segments = (inputs >> 25) - 1
        input_words = ((inputs & ((1 << 25) - 1)) >> 16) - 1
        span_info_list = []
        for batch_index in range(batch_size):
            span_infos = []
            seq_index = 1
            max_index = seq_length - 2
            while seq_index < max_index:
                span_length = self._sample_span_length(
                    self.cfg.augmentation_span_len_dist,
                    self.cfg.augmentation_max_span_len,
                    self.cfg.augmentation_geometric_prob,
                    self.cfg.augmentation_poisson_lambda,
                )
                span_length = min(span_length, max_index - seq_index + 1)

                segment_start_id = input_segments[batch_index, seq_index]
                segment_end_id = input_segments[batch_index, seq_index + span_length - 1]
                if segment_start_id != segment_end_id:
                    segment_index = (input_segments[batch_index, :] == segment_start_id).nonzero().squeeze(-1)
                    span_length = (segment_index[-1] - seq_index + 1).item()

                word_id = input_words[batch_index, seq_index + span_length - 1]
                if word_id >= 0:
                    word_index = (input_words[batch_index, :] == word_id + 1).nonzero().squeeze(-1)
                    span_length = (word_index[0] - seq_index).item() if word_index.nelement() > 0 \
                        else max_index - seq_index + 1

                span_infos.append((batch_index, seq_index, span_length))
                seq_index += span_length

            if len(span_infos) < self.cfg.augmentation_min_num_spans:
                span_infos = self._get_default_spans(
                    batch_index,
                    seq_length,
                    self.cfg.augmentation_min_num_spans,
                )
            span_info_list.extend(span_infos)
        return span_info_list

    def _replace_token(self, inputs, masking_indices, mask_index, vocab_size):
        if self.cfg.augmentation_replacing_schema == 'mask':
            inputs[masking_indices] = mask_index
        elif self.cfg.augmentation_replacing_schema == 'random':
            random_words = torch.randint(
                vocab_size,
                inputs.shape,
                device=inputs.device,
                dtype=torch.long,
            )
            inputs[masking_indices] = random_words[masking_indices]
        elif self.cfg.augmentation_replacing_schema == 'mixed':
            # 80% of the time, we replace masked input tokens with <unk> token
            mask_token_indices = torch.bernoulli(torch.full(
                inputs.shape,
                0.8,
                device=inputs.device,
            )).bool() & masking_indices
            inputs[mask_token_indices] = mask_index

            # 10% of the time, we replace masked input tokens with random word
            random_token_indices = torch.bernoulli(torch.full(
                inputs.shape,
                0.5,
                device=inputs.device,
            )).bool() & masking_indices & ~mask_token_indices
            random_words = torch.randint(
                vocab_size,
                inputs.shape,
                device=inputs.device,
                dtype=torch.long,
            )
            inputs[random_token_indices] = random_words[random_token_indices]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        else:
            raise ValueError(
                "The replacing schema: {0} is not supported. "
                "Only support ['mask', 'random', 'mixed']".format(self.cfg.augmentation_replacing_schema)
            )

    def set_beam_size(self, beam_size):
        for model in self.sequence_generator.model.models:
            # if hasattr(model, "set_beam_size"):
            model.set_beam_size(beam_size)

    def post_validate(self, model, stats, *args, **kwargs):
        if self.augmenter is not None:
            self.augmenter.num_updates = stats['num_updates']
            LOGGER.info(f"Augment.num_updates: {self.augmenter.num_updates}")

        if not self.cfg.eval_gec or stats['num_updates'] < self.cfg.eval_gec_min_update:
            return

        def encode_fn(x):
            if self.tokenizer is not None:
                x = self.tokenizer.encode(x)
            if self.bpe is not None:
                x = self.bpe.encode(x)
            return x

        use_cuda = not model.args.cpu
        max_positions = utils.resolve_max_positions(
            self.max_positions(), model.max_positions()
        )
        # Set beam search for sequence generation
        self.set_beam_size(self.cfg_all.generation.beam)

        start_id = 0
        start_time = time.time()

        results = []
        f = open(f"{self.cfg.eval_gec_output_prefix}-{stats['num_updates']}.out.nbest", "w", encoding="utf-8")
        for batch in make_batches(
                self.eval_input_processed if hasattr(self, "eval_input_processed") else self.eval_input,
                self.cfg_all,
                self,
                max_positions,
                encode_fn,
        ):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }

            translate_start_time = time.time()
            translations = self.inference_step(
                self.sequence_generator, [model], sample, constraints=constraints
            )
            translate_time = time.time() - translate_start_time

            list_constraints = [[] for _ in range(bsz)]
            if self.cfg_all.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]

            for i, (hypo_id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                    (
                        start_id + hypo_id,
                        src_tokens_i,
                        hypos,
                        {
                            "constraints": constraints,
                            "time": translate_time / len(translations),
                        },
                    )
                )

        detok_hypo_str_list = []

        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            src_str = ""
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.cfg_all.common_eval.post_process)
                f.write("S-{}\t{}\n".format(id_, src_str))
                f.write("W-{}\t{:.3f}\tseconds\n".format(id_, info["time"]))
                for constraint in info["constraints"]:
                    f.write(
                        "C-{}\t{}\n".format(
                            id_,
                            self.tgt_dict.string(constraint, self.cfg_all.common_eval.post_process),
                        )
                    )

            # Process top predictions
            for hypo_idx, hypo in enumerate(hypos[: min(len(hypos), self.cfg_all.generation.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=None,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.cfg_all.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.sequence_generator),
                )
                detok_hypo_str = self.decode_fn(hypo_str)
                if hypo_idx == 0:
                    detok_hypo_str_list.append(detok_hypo_str)
                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                f.write("H-{}\t{}\t{}\n".format(id_, score, hypo_str))
                # detokenized hypothesis
                f.write("D-{}\t{}\t{}\n".format(id_, score, detok_hypo_str))
                f.write(
                    "P-{}\t{}\n".format(
                        id_,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                # convert from base e to base 2
                                hypo["positional_scores"].div_(math.log(2)).tolist(),
                            )
                        ),
                    )
                )
                if self.cfg_all.generation.print_alignment:
                    alignment_str = " ".join(
                        ["{}-{}".format(src, tgt) for src, tgt in alignment]
                    )
                    f.write("A-{}\t{}\n".format(id_, alignment_str))

        # update running id_ counter
        # start_id += len(inputs)
        LOGGER.info("Generation time: {:.3f}".format(time.time() - start_time))

        # For MuCGEC
        if hasattr(self, "eval_input_processed"):
            from utils.post_processors.zho.post_process_bart import post_process
            detok_hypo_str_list = post_process(self.eval_input_processed, detok_hypo_str_list, self.eval_input_ids)

        # Write Hypotheses
        with open(f"{self.cfg.eval_gec_output_prefix}-{stats['num_updates']}.out", "w", encoding="utf-8") as f:
            for line in detok_hypo_str_list:
                f.write(line + "\n")

        # Construct dataset
        assert len(self.eval_data) == len(detok_hypo_str_list)
        dataset_hyp = Dataset(samples=[])
        for sample, hyp in zip(self.eval_data, detok_hypo_str_list):
            dataset_hyp.samples.append(Sample(
                index=len(dataset_hyp),
                source=sample.source.copy(),
                target=[hyp],
            ))

        # Evaluate using GEC metric
        score, results = self.metric.evaluate(dataset_hyp, self.eval_data)
        self.model_score.append(score[self.score_key])
        LOGGER.info(f"Evaluate with {self.metric.__class__}: {score}")

        # Set beam_size to 1 for training
        self.set_beam_size(1)
        self.backup_best_model()

    def backup_best_model(self):
        if self.model_score and self.model_score[-1] == max(self.model_score):
            save_dir = self.cfg_all.checkpoint.save_dir
            LOGGER.info(f"Save model with best score: {round(self.model_score[-1], 4)}")
            os.system(f"rm -f {os.path.join(save_dir, 'checkpoint_best_score.pt')}")
            os.system(f"cp {os.path.join(save_dir, 'checkpoint_last.pt')} "
                      f"{os.path.join(save_dir, 'checkpoint_best_score.pt')}")

    def encode_fn(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x
