import re
import sys


def split_sentence(line: str, flag: str = "all", limit: int = 510):
    """ Split sentences by end dot punctuations
    Args:
        line:
        flag: "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    sent_list = []
    try:
        if flag == "zho":
            # 中文单字符断句符
            line = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', line)
            # 特殊引号
            line = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', line)
        elif flag == "eng":
            # 英文单字符断句符
            line = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n', line)
            # 特殊引号
            line = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', line)
        else:
            # 单字符断句符
            line = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n', line)
            # 特殊引号
            line = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n', line)

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


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    id_file = sys.argv[3]

    max_sent_len = 64

    with open(input_file, "r", encoding="utf-8") as f_in:
        with open(output_file, "w", encoding="utf-8") as f_out:
            with open(id_file, "w", encoding="utf-8") as f_out_id:
                for idx, line in enumerate(f_in):
                    line = line.rstrip("\n")
                    if len(line) < max_sent_len:
                        f_out.write(line + "\n")
                        f_out_id.write(str(idx) + "\n")
                        continue
                    sents = split_sentence(line, flag="zh")
                    for sent in sents:
                        f_out.write(sent + "\n")
                        f_out_id.write(str(idx) + "\n")


if __name__ == '__main__':
    main()
