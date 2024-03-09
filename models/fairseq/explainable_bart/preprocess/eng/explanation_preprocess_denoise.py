import argparse
import contextlib
import logging
import json
import os
import sys
import numpy as np
from itertools import chain
from multiprocessing import Pool

try:
    from gpt2_bpe_utils import get_encoder
except:
    from .gpt2_bpe_utils import get_encoder

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
LOGGER = logging.getLogger("explanation_preprocess")

ERROR_TYPES = {
    # Syntax
    "None": 0,
    "Infinitives": 1,
    "Gerund": 2,
    "Participle": 3,
    "Subject-Verb Agreement": 4,
    "Auxiliary Verb": 5,
    "Pronoun-Antecedent Agreement": 6,
    "Possessive": 7,
    # Morphology
    "Collocation": 8,
    "Preposition": 9,
    "POS Confusion": 10,
    "Number": 11,
    "Transitive Verb": 12,
    # Discourse Level
    "Verb Tense": 13,
    "Article": 14,
    "Others": 15,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json", type=str,
        default=f"{os.path.dirname(__file__)}/encoder.json",
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe", type=str,
        default=f"{os.path.dirname(__file__)}/vocab.bpe",
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--input", type=str,
        default="/vepfs/yejh/nlp/datasets/GEC/EGEC/expect/json/dev.json",
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--source-output", type=str,
        default=f"{os.path.dirname(__file__)}/eng/expect_dev/dev.bpe.src",
        help="path to save encoded source outputs",
    )
    parser.add_argument(
        "--target-output", type=str,
        default=f"{os.path.dirname(__file__)}/eng/expect_dev/dev.bpe.tgt",
        help="path to save encoded target outputs",
    )
    parser.add_argument(
        "--explanation-output", type=str,
        default=f"{os.path.dirname(__file__)}/eng/expect_dev/dev.bpe.exp",
        help="path to save encoded explanation outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with contextlib.ExitStack() as stack:
        f1 = stack.enter_context(open(args.input, "r", encoding="utf-8"))
        f2 = stack.enter_context(open(args.source_output, "w", encoding="utf-8"))
        f3 = stack.enter_context(open(args.target_output, "w", encoding="utf-8"))
        f4 = stack.enter_context(open(args.explanation_output, "w", encoding="utf-8"))

        encoder = MultiprocessingEncoder(
            encoder_json=args.encoder_json,
            vocab_bpe=args.vocab_bpe,
        )
        pool = Pool(args.workers, initializer=encoder.initializer)
        results = pool.imap(encoder.process_line, f1, 100)

        for i, (_, _, src_bpes, tgt_bpes, explanation) in enumerate(results, start=1):
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)
            print(src_bpes, file=f2)
            print(tgt_bpes, file=f3)
            print(explanation, file=f4)
        pool.close()


class MultiprocessingEncoder(object):
    def __init__(self, encoder_json, vocab_bpe):
        self.encoder_json = encoder_json
        self.vocab_bpe = vocab_bpe

    def initializer(self):
        global bpe
        bpe = get_encoder(self.encoder_json, self.vocab_bpe)

    def process_line(self, line):
        """ Preprocess EXPECT dataset
            1) Acquire source evidence index
            2) Convert words to BPEs
            3) Build evidence and type
        """
        global bpe
        raw_sample = json.loads(line)
        src_words, tgt_words = raw_sample["source"], raw_sample["target"]
        src_evidence_word_idx = raw_sample["source_evidence_index"]

        src = " ".join(src_words)
        tgt = " ".join(tgt_words)

        # 2) Convert words to BPEs
        src_word_bpes, src_word_bpe_tokens = bpe.encode(src)
        tgt_word_bpes, tgt_word_bpe_tokens = bpe.encode(tgt)
        assert len(src_word_bpes) == len(src_words), src
        assert len(tgt_word_bpes) == len(tgt_words), tgt

        # transform 2-dim into 1-dim
        src_flat_bpes = list(chain(*src_word_bpes))
        src_flat_bpe_tokens = list(chain(*src_word_bpe_tokens))
        # tgt_flat_bpes = list(chain(*tgt_word_bpes))
        tgt_flat_bpe_tokens = list(chain(*tgt_word_bpe_tokens))

        # Sanity check: evidence is not continuous
        # temp = [src_evidence_word_idx[i+1] - src_evidence_word_idx[i] for i in range(len(src_evidence_word_idx) - 1)]
        # assert all(np.array(temp)==1), src

        # 3) Build evidence and type
        src_bpe_lens = list(map(len, src_word_bpes))  # 每个 word 的 bpe 长度
        src_bpe_cum_lens = np.cumsum(src_bpe_lens).tolist()  # 每个 word 的累积 bpe 长度

        src_evidence_bpe_idx = []
        for idx in src_evidence_word_idx:
            if idx == 0:
                src_evidence_bpe_idx.extend(list(range(src_bpe_cum_lens[idx])))
            else:
                src_evidence_bpe_idx.extend(list(range(src_bpe_cum_lens[idx - 1], src_bpe_cum_lens[idx])))

        # Consistent with BPE processing
        src_evidence_words = ' '.join(
            ["".join(bpe.byte_encoder[b] for b in src_words[i].encode("utf-8")) for i in src_evidence_word_idx]
        )
        src_evidence_bpes = ''.join([src_flat_bpes[i] for i in src_evidence_bpe_idx]).replace("Ġ", " ").strip()
        # print(f"src_word_bpes: {src_word_bpes}")
        # print(f"src_word_bpe_tokens: {src_word_bpe_tokens}")
        # print(f"tgt_word_bpes: {tgt_word_bpes}")
        # print(f"tgt_word_bpe_tokens: {tgt_word_bpe_tokens}")
        # print(f"src_evidence_word_idx: {src_evidence_word_idx} {src_evidence_words}")
        # print(f"src_evidence_bpe_idx: {src_evidence_bpe_idx} {src_evidence_bpes}")
        # print()
        assert src_evidence_words == src_evidence_bpes, f"{src_evidence_words} || {src_evidence_bpes} || {src}"

        explanation_output = [ERROR_TYPES.get(raw_sample["error_type"], 0)] + src_evidence_bpe_idx
        return (
            src_word_bpes,
            tgt_word_bpes,
            " ".join(list(map(str, src_flat_bpe_tokens))),
            " ".join(list(map(str, tgt_flat_bpe_tokens))),
            " ".join(list(map(str, explanation_output))),
        )


if __name__ == "__main__":
    main()

