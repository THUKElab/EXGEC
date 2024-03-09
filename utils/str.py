import re
from string import punctuation
from typing import List, Union

import spacy
from opencc import OpenCC

PUNCTUATION_ZHO = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."

PUNCTUATION_ENG = punctuation

PUNCTUATION = PUNCTUATION_ZHO + PUNCTUATION_ENG

# English resources
SPACY_MODEL = spacy.load("en_core_web_sm")

# Chinese resources
SIMPLIFIER = OpenCC("t2s")

# Chinese unicode ranges
ZHO_UNICODE_RANGES = [
    ("\u3400", "\u4db5"),  # CJK Unified Ideographs Extension A, release 3.0
    ("\u4e00", "\u9fa5"),  # CJK Unified Ideographs, release 1.1
    ("\u9fa6", "\u9fbb"),  # CJK Unified Ideographs, release 4.1
    ("\uf900", "\ufa2d"),  # CJK Compatibility Ideographs, release 1.1
    ("\ufa30", "\ufa6a"),  # CJK Compatibility Ideographs, release 3.2
    ("\ufa70", "\ufad9"),  # CJK Compatibility Ideographs, release 4.1
    ("\u20000", "\u2a6d6"),  # (UTF16) CJK Unified Ideographs Extension B, release 3.1
    ("\u2f800", "\u2fa1d"),  # (UTF16) CJK Compatibility Supplement, release 3.1
    ("\uff00", "\uffef"),  # Full width ASCII, full width of English punctuation,
    # half width Katakana, half wide half width kana, Korean alphabet
    ("\u2e80", "\u2eff"),  # CJK Radicals Supplement
    ("\u3000", "\u303f"),  # CJK punctuation mark
    ("\u31c0", "\u31ef"),  # CJK stroke
    ("\u2f00", "\u2fdf"),  # Kangxi Radicals
    ("\u2ff0", "\u2fff"),  # Chinese character structure
    ("\u3100", "\u312f"),  # Phonetic symbols
    ("\u31a0", "\u31bf"),  # Phonetic symbols (Taiwanese and Hakka expansion)
    ("\ufe10", "\ufe1f"),
    ("\ufe30", "\ufe4f"),
    ("\u2600", "\u26ff"),
    ("\u2700", "\u27bf"),
    ("\u3200", "\u32ff"),
    ("\u3300", "\u33ff"),
]


def remove_space(batch: Union[str, List[str]]) -> Union[str, List[str]]:
    def _remove_space(text: str):
        text = text.strip().replace("\u3000", " ").replace("\xa0", " ")
        text = "".join(text.split())
        return text

    if isinstance(batch, str):
        return _remove_space(batch)
    else:
        return [_remove_space(x) for x in batch]


def is_punct(char: str):
    assert len(char) == 1
    return char in PUNCTUATION


def tokenize(text: str, no_space: bool = False):
    if no_space:
        text = remove_space(text)
    doc = SPACY_MODEL(
        text.strip(),
        disable=["parser", "tagger", "ner"],
    )
    tokens = [str(token) for token in doc]
    return tokens


def tokenize_batch(text_list: List[str], no_space: bool = False):
    if no_space:
        text_list = [remove_space(x) for x in text_list]
    docs = SPACY_MODEL.pipe(
        text_list,
        batch_size=1024,
        disable=["parser", "tagger", "ner"],
    )
    docs = [[x.text for x in line] for line in docs]
    return docs


def simplify_chinese(text: str) -> str:
    return SIMPLIFIER.convert(text)


def all_chinese(text: str) -> bool:
    """判断字符串是否全部由中文组成
    1) 空格、字母不是中文
    2) 日文、韩文不是中文
    """
    # return all(['\u4e00' <= ch <= '\u9fff' for ch in text])

    def is_chinese_char(uchar: str):
        for start, end in ZHO_UNICODE_RANGES:
            if start <= uchar <= end:
                return True
        return False

    return all([is_chinese_char(ch) for ch in text])


def split_sentence(
    line: str,
    lang: str = "all",
    limit: int = 510,
    enable_blingfire: bool = False,
) -> List[str]:
    """Split sentences by end dot punctuations
    Args:
        line:
        lang: "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    if enable_blingfire:
        from blingfire import text_to_sentences

        return text_to_sentences(line.strip()).split("\n")

    sent_list = []
    try:
        if lang == "zho":
            # 中文单字符断句符
            line = re.sub(
                "(?P<quotation_mark>([。？！](?![”’\"'])))",
                r"\g<quotation_mark>\n",
                line,
            )
            # 特殊引号
            line = re.sub(
                "(?P<quotation_mark>([。？！])[”’\"'])", r"\g<quotation_mark>\n", line
            )
        elif lang == "eng":
            # 英文单字符断句符
            line = re.sub(
                "(?P<quotation_mark>([.?!](?![”’\"'])))", r"\g<quotation_mark>\n", line
            )
            # 特殊引号
            line = re.sub(
                "(?P<quotation_mark>([?!.][\"']))", r"\g<quotation_mark>\n", line
            )
        else:
            # 单字符断句符
            line = re.sub(
                "(?P<quotation_mark>([。？！….?!](?![”’\"'])))",
                r"\g<quotation_mark>\n",
                line,
            )
            # 特殊引号
            line = re.sub(
                "(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’\"']))",
                r"\g<quotation_mark>\n",
                line,
            )

        sent_list_ori = line.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except RuntimeError:
        sent_list.clear()
        sent_list.append(line)
    return sent_list


# def split_word_helper(sents: List[str], word_level=False):
#     split_sents, vocab_dict = [], Counter()
#     for sent in sents:
#         if word_level:
#             split_sent = list(jieba.cut(sent))
#         else:
#             split_sent = char_pattern.findall(sent)
#         split_sents.append(split_sent)
#         vocab_dict.update(split_sent)
#     return split_sents, vocab_dict
#
#
# def split_word(sents: List[str], word_level=False, num_worker=1):
#     if num_worker == 1:
#         return split_word_helper(sents, word_level=word_level)
#     results = multiprocess_helper(split_word_helper, (sents, word_level), num_worker=num_worker)
#     total_seg_sents, total_word_vocab_dict = [], Counter()
#     for seg, vocab in results:
#         total_seg_sents.extend(seg)
#         total_word_vocab_dict += vocab
#     return total_seg_sents, total_word_vocab_dict
#
#
# def multiprocess_helper(func, args, num_worker=1):
#     returns, results = [], []
#     split_input = args[0]
#     step = math.ceil(len(split_input) / num_worker)
#     pool = Pool(processes=num_worker)
#     for i in range(0, len(split_input), step):
#         results.append(pool.apply_async(func, (split_input[i:i + step], *args[1:])))
#     pool.close()
#     pool.join()
#     for res in results:
#         returns.append(res.get())
#     return returns
#
#
# def split_sentence(document: str, lang: str = "all", strategy: str = "all", split_length: int = 60, limit: int = 510):
#     """
#     Args:
#         document: 文档
#         lang: "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
#         strategy: 分句策略
#             all: 划分每个句子
#             greedy: 划分结果可能包含多个句子，但每个部分长度不超过 split_length
#         split_length:
#         limit: 默认单句最大长度为510个字符
#     Returns: Type:list
#     """
#     assert strategy in ['all', 'greedy']
#     sent_list = []
#     try:
#         if lang == "zh":
#             # 中文单字符断句符
#             document = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', document)
#             # 特殊引号
#             document = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', document)
#         elif lang == "en":
#             # 英文单字符断句符
#             document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)
#             # 特殊引号
#             document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)
#         else:
#             document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)
#             document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n', document)
#
#         sent_list_ori = document.splitlines()
#         for sent in sent_list_ori:
#             sent = sent.strip()
#             while len(sent) > limit:
#                 stop = limit - 1
#                 while stop >= 0 and sent[stop] not in ",;，；":
#                     stop -= 1
#                 if stop < 0:
#                     stop = limit - 1
#                 sent_list.append(sent[:stop + 1])
#                 sent = sent[stop + 1:]
#             if sent:
#                 sent_list.append(sent)
#     except RuntimeError and IndexError:
#         print(f"Fail to split document: {document}")
#         sent_list.clear()
#         sent_list.append(document)
#
#     if strategy == 'all':
#         return sent_list
#     elif strategy == 'greedy':
#         cur_sent = ""
#         merge_sent_list = []
#         for sent in sent_list:
#             if len(sent) + len(cur_sent) <= split_length:
#                 cur_sent += sent
#             else:
#                 if cur_sent:
#                     merge_sent_list.append(cur_sent)
#                 cur_sent = sent
#         if cur_sent:
#             merge_sent_list.append(cur_sent)
#         return merge_sent_list
#     else:
#         return sent_list
