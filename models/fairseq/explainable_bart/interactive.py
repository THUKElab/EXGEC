#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

SEPERATOR_TOKEN = "<sep>"
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


# @dataclass
# class EvaluationConfig(FairseqDataclass):
#     # input_source: str = field(
#     #     default="../../../../datasets/GEC/EGEC/expect/valid/expect_valid.src",
#     #     metadata={"help": "file to read from; use - for stdin"},
#     # )
#     input_errant: str = field(
#         default="../../../datasets/GEC/EGEC/expect/denoise/valid/expect_valid_denoise.errant",
#         metadata={"help": "file to read from; use - for stdin"},
#     )
#     input_explanation: str = field(
#         default="preprocess/eng/expect/valid/valid.bpe.exp",
#         metadata={"help": "file to read from; use - for stdin"},
#     )
#     input_expect: str = field(
#         default="../../../datasets/GEC/EGEC/expect/denoise/valid/expect_valid_denoise.json",
#         metadata={"help": "file to read from; use - for stdin"},
#     )
#     output: str = field(
#         default="/dev/null",
#         metadata={"help": "file to write to; use - for stdout"},
#     )


def buffered_read(
    input_source_files=None,
    input_source_lines=None,
    input_explanation_files=None,
    input_explanation_lines=None,
    buffer_size=10000,
):
    assert (input_source_files and input_explanation_files) or (
        input_source_lines and input_explanation_lines
    )

    if isinstance(input_source_files, str):
        input_source_files = [input_source_files]
    if isinstance(input_explanation_files, str):
        input_explanation_files = [input_explanation_files]

    if input_source_files:
        data_iter = zip(
            fileinput.input(
                files=input_source_files, openhook=fileinput.hook_encoded("utf-8")
            ),
            fileinput.input(
                files=input_explanation_files, openhook=fileinput.hook_encoded("utf-8")
            ),
        )
    else:
        data_iter = zip(input_source_lines, input_explanation_lines)

    buffer = []
    for src_str, exp_str in data_iter:
        buffer.append((src_str.strip(), exp_str.strip()))
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    fileinput.close()
    if len(buffer) > 0:
        yield buffer


def make_batches(sents, explanations, cfg, task, max_positions, encode_fn):
    """Revised by yejh on 2023.08.14
    1) Input both sents and explanations
    2) Build src_tokens with explanations if explanation_setting == infusion
    """

    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) constraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in sents]
        for i, line in enumerate(sents):
            if "\t" in line:
                sents[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    sent_tokens, sent_lengths = task.get_interactive_tokens_and_lengths(
        sents, encode_fn
    )

    explanation_tokens, explanation_lengths = [], []
    # if task.cfg.explanation_setting == "infusion":
    for explanation in explanations:
        # [ERROR_TYPE] [EVIDENCE_INDEX]
        explanation = list(map(int, explanation.split()))
        explanation_tokens.append(torch.LongTensor(explanation))
    explanation_lengths = [t.numel() for t in explanation_tokens]

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            sent_tokens,
            sent_lengths,
            constraints=constraints_tensor,
            explanation_tokens=explanation_tokens,
            explanation_lengths=explanation_lengths,
            explanation_before=cfg.task.explanation_before,
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        num_workers=cfg.dataset.num_workers,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )


def inference(
    cfg,
    task,
    generator,
    src_dict,
    tgt_dict,
    encode_fn,
    decode_fn,
    max_positions,
    input_source_files=None,
    input_source_lines=None,
    input_explanation_files=None,
    input_explanation_lines=None,
    buffer_size=5000,
    use_cuda=True,
    align_dict=None,
    sout=sys.stdout,
):
    """Created by yejh on 2023.08.14
    return a list of samples
    [
        {
            "src_str": str,
            "detok_src_str": str,
            "hypo_str": str,
            "detok_hypo_sent: str,
            "hypo_error_type": str,
            "hypo_evidence_idx": List[int],
        },
    ]
    """
    returns = []
    start_time, start_id, total_translate_time = time.time(), 0, 0

    for inputs in buffered_read(
        input_source_files=input_source_files,
        input_source_lines=input_source_lines,
        input_explanation_files=input_explanation_files,
        input_explanation_lines=input_explanation_lines,
        buffer_size=buffer_size,
    ):
        results = []
        source_inputs = [x[0] for x in inputs]
        explanation_inputs = [x[1] for x in inputs]
        for batch in make_batches(
            source_inputs, explanation_inputs, cfg, task, max_positions, encode_fn
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

            # Sequence Tagging
            tag_preds = None
            decoder = generator.model.models[0].decoder
            if decoder.sequence_tagging:
                encoder = generator.model.models[0].encoder
                encoder_out = encoder(
                    src_tokens=sample["net_input"]["src_tokens"],
                    src_lengths=sample["net_input"]["src_lengths"],
                )
                # print(encoder_out["encoder_out"][0].size())  # [LS, B, D]
                tag_logits = decoder.forward_tagging(encoder_out)  # [B, LS, C]
                tag_preds = torch.argmax(tag_logits, dim=-1)  # [B, LS]
                tag_preds = [x.detach().cpu().tolist() for x in tag_preds]

            translate_start_time = time.time()
            translations = task.inference_step(
                generator,
                generator.model.models,
                sample,
                constraints=constraints,
            )
            translate_time = time.time() - translate_start_time
            total_translate_time += translate_time
            list_constraints = [[] for _ in range(bsz)]
            if cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (idx, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                tag_preds_i = (
                    tag_preds[i][: len(src_tokens_i) - 1] if tag_preds else None
                )
                constraints = list_constraints[i]
                results.append(
                    (
                        start_id + idx,
                        src_tokens_i,
                        hypos,
                        tag_preds_i,
                        {
                            "constraints": constraints,
                            "time": translate_time / len(translations),
                        },
                    )
                )

        # sort output to match input order
        for id_, src_tokens, hypos, tag_pred, info in sorted(
            results, key=lambda x: x[0]
        ):
            src_str, detok_src_sent = "", ""
            if src_dict is not None:
                # src_str: [Source] <sep> [ERROR_TYPE] <unk> <unk>
                # 1135 4255 20328 286 2057 290 16759 764 <sep> <Others> 50283 50284
                src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)

                if cfg.task.explanation_setting == "infusion":
                    src_str_list = src_str.split()
                    sep_idx = src_str_list.index(SEPERATOR_TOKEN)

                    src_sent = src_str_list[:sep_idx]
                    detok_src_sent = decode_fn(" ".join(src_sent))

                    src_evidence_idx = [
                        i for i in src_tokens.int().cpu().tolist() if i >= len(src_dict)
                    ]
                    detok_src_evidence_idx = [
                        i - len(src_dict) for i in src_evidence_idx
                    ]

                    detok_src_str = f"{detok_src_sent} {SEPERATOR_TOKEN}"
                    format_split = cfg.task.explanation_format.split("-")
                    if format_split[0] == "type":
                        detok_src_str += " " + src_str_list[sep_idx + 1]
                        if detok_src_evidence_idx:
                            detok_src_str += " " + " ".join(
                                list(map(str, detok_src_evidence_idx))
                            )
                    elif format_split[0] == "evidence":
                        detok_src_str += " " + " ".join(
                            list(map(str, detok_src_evidence_idx))
                        )
                        if len(format_split) > 1 and format_split[1] == "type":
                            detok_src_str += " " + src_str_list[-1]
                else:
                    detok_src_str = detok_src_sent = decode_fn(src_str)

                returns.append(
                    {
                        "src_str": src_str,
                        "detok_src_str": detok_src_str,
                        "tag_pred": tag_pred,
                    }
                )
                # print("S-{}\t{}".format(id_, src_str))
                print("S-{}\t{}".format(id_, detok_src_str), file=sout)
                print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]), file=sout)
                for constraint in info["constraints"]:
                    print(
                        "C-{}\t{}".format(
                            id_,
                            tgt_dict.string(constraint, cfg.common_eval.post_process),
                        ),
                        file=sout,
                    )

            # Process top predictions
            for hypo_idx, hypo in enumerate(
                hypos[: min(len(hypos), cfg.generation.nbest)]
            ):
                # 1) Special error type tokens
                # 2) Pointing tokens
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                invalid_sample = False
                detok_hypo_str, detok_hypo_sent = None, detok_src_sent
                hypo_error_type, detok_hypo_evidence_idx = "<Others>", []
                try:
                    if cfg.task.explanation_setting == "rationalization":
                        hypo_str_list = hypo_str.split()

                        assert (
                            len(
                                list(
                                    filter(
                                        lambda x: x == SEPERATOR_TOKEN, hypo_str_list
                                    )
                                )
                            )
                            == 1
                        ), f"Invalid hypo: {hypo_str_list}"
                        sep_idx = hypo_str_list.index(SEPERATOR_TOKEN)

                        hypo_evidence_idx = [
                            i
                            for i in hypo["tokens"].int().cpu().tolist()
                            if i >= len(src_dict)
                        ]
                        detok_hypo_evidence_idx = [
                            i - len(src_dict) for i in hypo_evidence_idx
                        ]

                        hypo_str, detok_hypo_str = "", ""
                        format_split = cfg.task.explanation_format.split("-")
                        if cfg.task.explanation_before:
                            # hypo_str: [ERROR_TYPE] [EVIDENCE_INDEX] <sep> [CORRECTION]
                            hypo_sent = hypo_str_list[sep_idx + 1 :]
                            detok_hypo_sent = decode_fn(" ".join(hypo_sent))
                            if format_split[0] == "type":
                                hypo_error_type = hypo_str_list[0]
                                hypo_str += hypo_error_type
                                detok_hypo_str += hypo_error_type
                                if detok_hypo_evidence_idx:
                                    hypo_str += " " + " ".join(
                                        list(map(str, hypo_evidence_idx))
                                    )
                                    detok_hypo_str += " ".join(
                                        list(map(str, detok_hypo_evidence_idx))
                                    )
                            elif format_split[0] == "evidence":
                                hypo_str += " ".join(list(map(str, hypo_evidence_idx)))
                                detok_hypo_str += " ".join(
                                    list(map(str, detok_hypo_evidence_idx))
                                )
                                if len(format_split) > 1 and format_split[1] == "type":
                                    hypo_error_type = hypo_str_list[sep_idx - 1]
                                    hypo_str += " " + hypo_error_type
                                    detok_hypo_str += " " + hypo_error_type
                            hypo_str += (
                                " " + SEPERATOR_TOKEN + " " + " ".join(hypo_sent)
                            )
                            detok_hypo_str += (
                                " " + SEPERATOR_TOKEN + " " + detok_hypo_sent
                            )
                        else:
                            # hypo_Str: [CORRECTION] <sep> [ERROR_TYPE] [EVIDENCE_INDEX]
                            hypo_sent = hypo_str_list[:sep_idx]
                            detok_hypo_sent = decode_fn(" ".join(hypo_sent))
                            hypo_str += " ".join(hypo_sent) + " " + SEPERATOR_TOKEN
                            detok_hypo_str += detok_hypo_sent + " " + SEPERATOR_TOKEN
                            if format_split[0] == "type":
                                hypo_error_type = hypo_str_list[0]
                                hypo_str += " " + hypo_error_type
                                detok_hypo_str += " " + hypo_error_type
                                if detok_hypo_evidence_idx:
                                    hypo_str += " " + " ".join(
                                        list(map(str, hypo_evidence_idx))
                                    )
                                    detok_hypo_str += " ".join(
                                        list(map(str, detok_hypo_evidence_idx))
                                    )
                            elif format_split[0] == "evidence":
                                hypo_str += " " + " ".join(
                                    list(map(str, hypo_evidence_idx))
                                )
                                detok_hypo_str += " " + " ".join(
                                    list(map(str, detok_hypo_evidence_idx))
                                )
                                if len(format_split) > 1 and format_split[1] == "type":
                                    hypo_error_type = hypo_str_list[-1]
                                    hypo_str += " " + hypo_error_type
                                    detok_hypo_str += " " + hypo_error_type
                    elif cfg.task.explanation_setting == "explanation":
                        # hypo_Str: [ERROR_TYPE] [EVIDENCE_INDEX]
                        detok_hypo_sent = None
                        hypo_str_list = hypo_str.split()

                        hypo_evidence_idx = [
                            i
                            for i in hypo["tokens"].int().cpu().tolist()
                            if i >= len(src_dict)
                        ]
                        detok_hypo_evidence_idx = [
                            i - len(src_dict) for i in hypo_evidence_idx
                        ]

                        hypo_str, detok_hypo_str = "", ""
                        format_split = cfg.task.explanation_format.split("-")
                        if format_split[0] == "type":
                            hypo_error_type = hypo_str_list[0]
                            hypo_str += hypo_error_type
                            detok_hypo_str += hypo_error_type
                            if detok_hypo_evidence_idx:
                                hypo_str += " " + " ".join(
                                    list(map(str, hypo_evidence_idx))
                                )
                                detok_hypo_str += " " + " ".join(
                                    list(map(str, detok_hypo_evidence_idx))
                                )
                        elif format_split[0] == "evidence":
                            hypo_str += " ".join(list(map(str, hypo_evidence_idx)))
                            detok_hypo_str += " ".join(
                                list(map(str, detok_hypo_evidence_idx))
                            )
                            if len(format_split) > 1 and format_split[1] == "type":
                                hypo_error_type = hypo_str_list[-1]
                                hypo_str += " " + hypo_error_type
                                detok_hypo_str += " " + hypo_error_type
                    else:
                        detok_hypo_str = detok_hypo_sent = decode_fn(hypo_str)
                except Exception as e:
                    invalid_sample = True
                    print(e)
                    print(
                        f"Error hypo_tokens: {' '.join(map(str, hypo['tokens'].int().cpu().tolist()))}"
                    )

                if hypo_idx == 0:
                    returns[-1].update(
                        {
                            "invalid": invalid_sample,
                            "hypo_tokens": " ".join(
                                map(str, hypo["tokens"].int().cpu().tolist())
                            ),
                            "detok_hypo_sent": detok_hypo_sent,
                            "hypo_error_type": hypo_error_type,
                            "hypo_evidence_idx": detok_hypo_evidence_idx,
                        }
                    )

                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print("H-{}\t{}\t{}".format(id_, score, hypo_str), file=sout)
                # detokenized hypothesis
                print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str), file=sout)
                print(
                    "P-{}\t{}".format(
                        id_,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                # convert from base e to base 2
                                hypo["positional_scores"].div_(math.log(2)).tolist(),
                            )
                        ),
                    ),
                    file=sout,
                )
                if cfg.generation.print_alignment:
                    alignment_str = " ".join(
                        ["{}-{}".format(src, tgt) for src, tgt in alignment]
                    )
                    print("A-{}\t{}".format(id_, alignment_str), file=sout)

        # update running id_ counter
        start_id += len(inputs)

    logger.info(
        "Total time: {:.3f} seconds; translation time: {:.3f}".format(
            time.time() - start_time, total_translate_time
        )
    )
    # with open("exps/eng/expect_denoise/temp/results/output.json", "w", encoding="utf-8") as f:
    #     json.dump(returns, f)
    return returns


def main(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if cfg.generation.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if cfg.interactive.buffer_size > 1:
        logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")

    from data import Dataset, M2DataReader, Sample
    from data_utils import evaluate_explanation, evaluate_tagging, load_expect_denoise

    eval_data = M2DataReader().read(cfg.task.eval_gec_m2_filepath)
    eval_src = [x.source[0] for x in eval_data]
    eval_exp = [
        x.strip() for x in open(cfg.task.eval_gec_exp_filepath, "r", encoding="utf-8")
    ]
    eval_raw, eval_bpe = None, None
    if cfg.task.explanation_setting in ["rationalization", "explanation"]:
        eval_raw = load_expect_denoise(cfg.task.eval_gec_raw_filepath)
        eval_bpe = []

        from preprocess.eng.explanation_preprocess_denoise import MultiprocessingEncoder

        encoder = MultiprocessingEncoder(
            encoder_json=f"{os.path.dirname(__file__)}/preprocess/eng/encoder.json",
            vocab_bpe=f"{os.path.dirname(__file__)}/preprocess/eng/vocab.bpe",
        )
        encoder.initializer()
        with open(cfg.task.eval_gec_raw_filepath, "r", encoding="utf-8") as f:
            for line in f:
                src_word_bpes, tgt_word_bpes, _, _, _ = encoder.process_line(line)
                eval_bpe.append(src_word_bpes)

    # Inference
    with open(
        f"{os.path.dirname(cfg.common_eval.path)}/results/output.out.nbest",
        "w",
        encoding="utf-8",
    ) as f:
        results = inference(
            cfg,
            task,
            generator=generator,
            input_source_lines=eval_src,
            input_explanation_lines=eval_exp,
            src_dict=src_dict,
            tgt_dict=tgt_dict,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            max_positions=max_positions,
            buffer_size=cfg.interactive.buffer_size,
            use_cuda=use_cuda,
            align_dict=align_dict,
            sout=f,
        )

    # Evaluate Explanation
    if cfg.task.explanation_setting in ["rationalization", "explanation"]:
        num_invalid = sum([x["invalid"] for x in results])
        hypo_error_type = [x["hypo_error_type"] for x in results]
        hypo_evidence_idx = [x["hypo_evidence_idx"] for x in results]
        results_exp = evaluate_explanation(
            samples=eval_raw,
            pred_error_type_list=hypo_error_type,
            pred_evidence_idx_list=hypo_evidence_idx,
            src_word_bpes_list=eval_bpe,
            strict_bpe=True,
        )
        print(f"Invalid samples: {num_invalid}")
        print(f"Evaluate Explanation: {results_exp}")
        logger.info(f"Invalid samples: {num_invalid}")
        logger.info(f"Evaluate Explanation: {results_exp}")

    if cfg.model.sequence_tagging:
        hypo_tagging = [x["tag_pred"] for x in results]
        results_tag = evaluate_tagging(
            samples=eval_raw,
            tag_preds=hypo_tagging,
            src_word_bpes_list=eval_bpe,
        )
        print(f"Evaluate Tagging: {results_tag}")
        logger.info(f"Evaluate Tagging: {results_tag}")

    if cfg.task.explanation_setting == "explanation":
        return

    # Construct dataset
    detok_hypo_sent = [x["detok_hypo_sent"] for x in results]
    assert len(eval_data) == len(detok_hypo_sent)
    dataset_hyp = Dataset(samples=[])
    for sample, hyp in zip(eval_data, detok_hypo_sent):
        dataset_hyp.samples.append(
            Sample(
                index=len(dataset_hyp),
                source=sample.source.copy(),
                target=[hyp],
            )
        )

    # Evaluate Correction using GEC metric
    from metrics import ErrantEN, SystemScorer

    metric = ErrantEN(SystemScorer())
    score, results = metric.evaluate(dataset_hyp, eval_data)
    print(f"Evaluate Correction with {metric.__class__}: {score}")
    logger.info(f"Evaluate Correction with {metric.__class__}: {score}")


def cli_main():
    from fairseq.options import add_model_args

    parser = options.get_interactive_generation_parser()

    # Add ModelConfig to enable sequence tagging
    add_model_args(parser)

    # NOTE: Add custom EvaluationConfig is in vain, since `convert_namespace_to_omegaconf`
    # identifies `Config` defined in `FairseqConfig`
    # group = parser.add_argument_group("Evaluation")
    # gen_parser_from_dataclass(group, EvaluationConfig())

    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()

"""
======================================= EXPECT =======================================
preprocess/eng/expect_denoise/train/bin
--task explainable_gec
--arch egec_bart_large
--user-dir .
--path exps/eng/expect_denoise/expect_denoise-rationalization_evidence_type_after-enc_mlp-tag_sim_l2_tw1.0_eps0.1-ew1.0_eps0.0/checkpoint_best_score.pt
--beam 10
--nbest 1
-s src
-t tgt
--bpe gpt2
--buffer-size 5000
--batch-size 128
--num-workers 4
--log-format tqdm
--remove-bpe
--fp16
--min-len 0
--left-pad-source
--explanation-format evidence-type
--explanation-setting rationalization
--eval-gec-m2-filepath ../../../../datasets/GEC/EGEC/expect/denoise/valid/expect_valid_denoise.errant
--eval-gec-raw-filepath ../../../../datasets/GEC/EGEC/expect/denoise/valid/expect_valid_denoise.json
--eval-gec-exp-filepath preprocess/eng/expect_denoise/valid/valid.bpe.exp
--use-encoder-mlp


Evaluate Explanation: {'P': 0.4244, 'R': 0.4614, 'F1': 0.4422, 'F0.5': 0.4313, 'EM': 0.3092, 'P_strict': 0.2996, 'R_strict': 0.3258, 'F1_strict': 0.3121, 'F0.5_strict': 0.3045, 'EM_strict': 0.2814, 'ACC': 0.3556, 'strict_bpe': True, 'TP': 1969, 'FP': 2670, 'FN': 2298, 'TN': 64321, 'TP_strict': 1390, 'FP_strict': 3249, 'FN_strict': 2298, 'TN_strict': 64321}
 {'num_sample': 2413, 'F': 0.3444, 'Acc': 0.218, 'P': 0.3359, 'R': 0.383, 'tp': 994, 'fp': 1965, 'fn': 1601, 'tn': 0}
 
"""
