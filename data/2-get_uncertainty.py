import re
import math
from collections import defaultdict
from fairseq.data import Dictionary

SRC = 'en'
TGT = 'zh'
DATASET = 'cwmt'

root = f'/root/Mono4SiM/data/cwmt-{SRC}{TGT}'
align = f'{root}/score/forward.align'
src_file = f'{root}/ready/train.clean.{SRC}'
tgt_file = f'{root}/ready/train.clean.{TGT}'
dict_file = f'{root}/data-bin/dict.{SRC}.txt'

d = defaultdict(lambda: defaultdict(int))

with open(align, 'r', encoding='utf-8') as fa, \
    open(src_file, 'r', encoding='utf-8') as fs, \
    open(tgt_file, 'r', encoding='utf-8') as ft:
    for a, s, t in zip(fa, fs, ft):
        itr = re.finditer(r"(?P<i>[0-9]+)-(?P<j>[0-9]+)", a.strip())
        left = []
        right = []
        for m in itr:
            left.append(int(m.group("i")))
            right.append(int(m.group("j")))
        s = s.strip().split()
        t = t.strip().split()
        for i, j in zip(left, right):
            d[s[i]][t[j]] += 1

    dct = Dictionary.load(dict_file)
    for key, value in d.items():
        if key in dct:
            tot = 0
            score = 0
            for _, j in value.items():
                tot += j
            for _, j in value.items():
                p = j / tot
                score += p * math.log(p)
            dct.count[dct.index(key)] = score
    dct.save(f'{root}/score/uncertainty.{SRC}.txt')
