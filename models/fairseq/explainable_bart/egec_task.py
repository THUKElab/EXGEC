import itertools
import os
from dataclasses import dataclass, field

from data import Dataset, M2DataReader, Sample
from fairseq import search, utils
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
from metrics import SystemScorer

from utils import get_logger

from .data_utils import evaluate_explanation, evaluate_tagging, load_expect_denoise
from .egec_language_pair_dataset import (
    ERROR_TYPE_TOKENS,
    ExplainableLanguagePairDataset,
)
from .interactive import inference
from .sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment

LOGGER = get_logger(__name__)


def load_explainable_langpair_dataset(
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
    # max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    explanation_setting=None,
    explanation_format=None,
    explanation_before=False,
    sequence_tagging=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    explanation_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer lang-code
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

        # Load explanation
        if sequence_tagging or (explanation_setting and explanation_format):
            explanation_dataset = data_utils.load_indexed_dataset(
                path=os.path.join(data_path, "{}.{}".format(split_k, "exp")),
                dataset_impl=dataset_impl,
            )
            if explanation_dataset is not None:
                explanation_datasets.append(explanation_dataset)

        LOGGER.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        explanation_dataset = (
            explanation_datasets[0] if len(explanation_datasets) > 0 else None
        )
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = (
            ConcatDataset(tgt_datasets, sample_ratios)
            if len(tgt_datasets) > 0
            else None
        )
        explanation_dataset = (
            explanation_datasets[0] if len(explanation_datasets) > 0 else None
        )

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
    explanation_dataset_sizes = (
        explanation_dataset.sizes if explanation_dataset is not None else None
    )
    return ExplainableLanguagePairDataset(
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
        explanation=explanation_dataset,
        explanation_sizes=explanation_dataset_sizes,
        explanation_setting=explanation_setting,
        explanation_format=explanation_format,
        explanation_before=explanation_before,
        sequence_tagging=sequence_tagging,
    )


@dataclass
class ExplainableGECConfig(TranslationConfig):
    explanation_setting: str = field(
        default="",
        metadata={
            "choices": ["", "infusion", "rationalization", "explanation"],
            "help": "Explanation format. No explanation if empty",
        },
    )
    explanation_format: str = field(
        default="evidence",
        metadata={
            "choices": ["type", "evidence", "type-evidence", "evidence-type"],
            "help": "Explanation format",
        },
    )
    explanation_before: bool = field(
        default=False, metadata={"help": "Explanation position relative to the target"}
    )
    # options for reporting GEC scores during validation
    eval_gec: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_gec_min_update: int = field(
        default=0,
        metadata={"help": "only evaluate gec if num_updates > eval_gec_min_update"},
    )
    eval_gec_m2_filepath: str = field(
        default="",
        metadata={"help": "Evaluation M2 dataset. No evaluation if not specified"},
    )
    eval_gec_raw_filepath: str = field(
        default="",
        metadata={
            "help": "Evaluation official dataset. No evaluation if not specified"
        },
    )
    eval_gec_exp_filepath: str = field(
        default="", metadata={"help": "Evaluation explanation filepath"}
    )
    eval_gec_output_prefix: str = field(default="temp", metadata={"help": ""})
    eval_gec_metric: str = field(
        default="errant_eng",
        metadata={
            "help": "GEC metric",
            "choices": ["errant_eng", "errant_zho", "m2", "gleu"],
        },
    )
    eval_gec_sent_level: bool = field(
        default=False, metadata={"help": "evaluation with sentence-level metric"}
    )


@register_task("explainable_gec", dataclass=ExplainableGECConfig)
class ExplainableGECTask(TranslationTask):
    def __init__(self, cfg: ExplainableGECConfig, src_dict, tgt_dict, score_key="F"):
        super().__init__(cfg, src_dict, tgt_dict)
        self.cfg: ExplainableGECConfig = cfg
        self.cfg_all = None
        self.bpe = None
        self.tokenizer = None
        self.eval_data = None
        self.eval_src = None
        self.eval_exp = None
        self.eval_raw = None
        self.eval_bpe = []
        self.eval_src_ids = None
        self.eval_src_processed = None
        self.sequence_generator = None
        self.metric = None
        self.model_score = []
        self.score_key = score_key

    def build_model(self, cfg: ExplainableGECConfig, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        if cfg.eval_gec_m2_filepath and self.cfg_all is not None:
            # self.tokenizer = encoders.build_tokenizer(self.cfg_all.tokenizer)
            if hasattr(self.cfg_all, "bpe") and self.cfg_all.bpe._name == "gpt2":
                self.bpe = GPT2BPE(self.cfg_all.bpe)
            self.eval_data = M2DataReader().read(cfg.eval_gec_m2_filepath)
            self.eval_src = [x.source[0] for x in self.eval_data]
            self.eval_exp = [
                x.strip()
                for x in open(cfg.eval_gec_exp_filepath, "r", encoding="utf-8")
            ]

            if self.cfg_all.model.sequence_tagging or cfg.explanation_setting in [
                "rationalization",
                "explanation",
            ]:
                # self.eval_raw = load_expect(cfg.eval_gec_raw_filepath)
                # self.eval_raw = process_expect(self.eval_raw)
                self.eval_raw = load_expect_denoise(cfg.eval_gec_raw_filepath)

                # from .preprocess.eng.explanation_preprocess import MultiprocessingEncoder
                from .preprocess.eng.explanation_preprocess_denoise import (
                    MultiprocessingEncoder,
                )

                encoder = MultiprocessingEncoder(
                    encoder_json=self.cfg_all.bpe.gpt2_encoder_json,
                    vocab_bpe=self.cfg_all.bpe.gpt2_vocab_bpe,
                )
                encoder.initializer()
                with open(cfg.eval_gec_raw_filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        src_word_bpes, tgt_word_bpes, _, _, _ = encoder.process_line(
                            line
                        )
                        self.eval_bpe.append(src_word_bpes)

            self.build_metric(cfg)
            # build_generator 会根据 generation_args 修改 MultiheadAttention 的 beam_size
            self.sequence_generator = self.build_generator(
                [model], self.cfg_all.generation
            )
            self.sequence_generator.vocab_size += 1 + len(ERROR_TYPE_TOKENS)
            self.set_beam_size(1)
        return model

    def build_metric(self, cfg):
        if cfg.eval_gec_metric == "errant_eng":
            from metrics import ErrantEN

            self.metric = ErrantEN(SystemScorer())
        elif cfg.eval_gec_metric == "errant_zho":
            from metrics import ErrantZH

            from utils.pre_processors.pre_process_chinese import pre_process

            self.metric = ErrantZH(SystemScorer())
            self.eval_src_processed, self.eval_src_ids = pre_process(
                self.eval_src,
                file_vocab=os.path.join(
                    os.path.dirname(__file__), "preprocess/zho/vocab_v2.txt"
                ),
            )
        else:
            raise NotImplementedError

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load Explainable Language Pair Dataset
        1) Language pair
        2) Explanation
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = load_explainable_langpair_dataset(
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
            # max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            explanation_setting=self.cfg.explanation_setting,
            explanation_format=self.cfg.explanation_format,
            explanation_before=self.cfg.explanation_before,
            sequence_tagging=self.cfg_all.model.sequence_tagging,
        )

    def build_dataset_for_inference(
        self,
        sent_tokens,
        sent_lengths,
        constraints=None,
        explanation_tokens=None,
        explanation_lengths=None,
        explanation_before=False,
    ):
        return ExplainableLanguagePairDataset(
            src=sent_tokens,
            src_sizes=sent_lengths,
            src_dict=self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            explanation=explanation_tokens,
            explanation_sizes=explanation_lengths,
            explanation_setting=self.cfg.explanation_setting,
            explanation_format=self.cfg.explanation_format,
            explanation_before=explanation_before,
            sequence_tagging=False,
        )

    def set_beam_size(self, beam_size):
        for model in self.sequence_generator.model.models:
            model.set_beam_size(beam_size)

    def should_validate_gec(self, stats):
        LOGGER.debug(f"stats: {stats}")
        return stats["num_updates"] >= self.cfg.eval_gec_min_update

    def post_validate(self, model, stats):
        if not self.should_validate_gec(stats):
            return
        # Set beam search for sequence generation
        self.set_beam_size(self.cfg_all.generation.beam)

        output_filepath = (
            f"{self.cfg.eval_gec_output_prefix}-{stats['num_updates']}.out"
        )
        results = inference(
            cfg=self.cfg_all,
            task=self,
            generator=self.sequence_generator,
            src_dict=self.source_dictionary,
            tgt_dict=self.target_dictionary,
            encode_fn=self.encode_fn,
            decode_fn=self.decode_fn,
            max_positions=utils.resolve_max_positions(
                self.max_positions(), model.max_positions()
            ),
            input_source_lines=(
                self.eval_src
                if self.eval_src_processed is None
                else self.eval_src_processed
            ),
            input_explanation_lines=self.eval_exp,
            buffer_size=10000,
            use_cuda=not model.args.cpu,
            align_dict=None,
            sout=open(output_filepath + ".nbest", "w", encoding="utf-8"),
        )
        # Set beam_size to 1 for training
        self.set_beam_size(1)

        if self.cfg.explanation_setting in ["rationalization", "explanation"]:
            num_invalid = sum([x["invalid"] for x in results])
            hypo_error_type = [x["hypo_error_type"] for x in results]
            hypo_evidence_idx = [x["hypo_evidence_idx"] for x in results]
            results_exp = evaluate_explanation(
                samples=self.eval_raw,
                pred_error_type_list=hypo_error_type,
                pred_evidence_idx_list=hypo_evidence_idx,
                src_word_bpes_list=self.eval_bpe,
                strict_bpe=True,
            )
            LOGGER.info(f"Invalid samples: {num_invalid}")
            LOGGER.info(f"Evaluate Explanation: {results_exp}")

        if self.cfg_all.model.sequence_tagging:
            hypo_tagging = [x["tag_pred"] for x in results]
            with open(output_filepath + ".tag", "w", encoding="utf-8") as f:
                for tag in hypo_tagging:
                    f.write(f"{tag}\n")
            results_tag = evaluate_tagging(
                samples=self.eval_raw,
                tag_preds=hypo_tagging,
                src_word_bpes_list=self.eval_bpe,
            )
            LOGGER.info(f"Evaluate Tagging: {results_tag}")

        if self.cfg.explanation_setting == "explanation":
            return

        detok_hypo_sent = [x["detok_hypo_sent"] for x in results]
        if self.eval_src_processed is not None:  # For MuCGEC
            from utils.post_processors.zho.post_process_bart import post_process

            detok_hypo_sent = post_process(
                srcs=self.eval_src_processed,
                tgts=detok_hypo_sent,
                ids=self.eval_src_ids,
            )

        # Write Hypotheses
        with open(output_filepath, "w", encoding="utf-8") as f:
            for line in detok_hypo_sent:
                f.write(line + "\n")

        # Construct dataset
        assert len(self.eval_data) == len(detok_hypo_sent)
        dataset_hyp = Dataset(samples=[])
        for sample, hyp in zip(self.eval_data, detok_hypo_sent):
            dataset_hyp.samples.append(
                Sample(
                    index=len(dataset_hyp),
                    source=sample.source.copy(),
                    target=[hyp],
                )
            )

        # Evaluate using GEC metric
        score, results = self.metric.evaluate(dataset_hyp, self.eval_data)
        self.model_score.append(score[self.score_key])
        LOGGER.info(f"Evaluate Correction with {self.metric.__class__}: {score}")
        self.backup_best_model()

    def backup_best_model(self):
        if self.model_score and self.model_score[-1] == max(self.model_score):
            save_dir = self.cfg_all.checkpoint.save_dir
            LOGGER.info(f"Save model with best score: {round(self.model_score[-1], 4)}")
            os.system(f"rm -f {os.path.join(save_dir, 'checkpoint_best_score.pt')}")
            os.system(
                f"cp {os.path.join(save_dir, 'checkpoint_last.pt')} "
                f"{os.path.join(save_dir, 'checkpoint_best_score.pt')}"
            )

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

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                # Use custom NoisingSequenceGenerator for noising back translation
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )
