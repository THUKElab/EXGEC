import argparse
import os
import sys
from multiprocessing import Pool

from tqdm import tqdm

try:
    from bert import tokenization
except Exception:
    print("pip install bert-tokenization")

parser = argparse.ArgumentParser()
parser.add_argument("--lowercase", default=False, type=bool)
args = parser.parse_args()

tokenizer = tokenization.FullTokenizer(
    # For Chinese-BART V1
    # vocab_file=os.path.join(os.path.dirname(__file__), "vocab_v1.txt"),
    # For Chinese-BART V2
    vocab_file=os.path.join(os.path.dirname(__file__), "vocab_v2.txt"),
    do_lower_case=args.lowercase,  # Set to True to avoid most [UNK]
)


def split(line):
    line = line.strip()
    # origin_line = line
    line = line.replace(" ", "")
    line = tokenization.convert_to_unicode(line)
    if not line:
        return ""
    tokens = tokenizer.tokenize(line)
    return " ".join(tokens)


with Pool(64) as pool:
    for ret in pool.imap(split, tqdm(sys.stdin), chunksize=1024):
        print(ret)
