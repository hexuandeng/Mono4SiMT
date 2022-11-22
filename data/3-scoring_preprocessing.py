
import readline
import math
import re
import os
import kenlm
import time
import matplotlib.pyplot as plt
import numpy as np
from cmath import inf
from collections import defaultdict
from fairseq.data import Dictionary
from random import randint, sample
from subprocess import Popen, PIPE, STDOUT

SRC = 'en'
TGT = 'zh'
DATASET = 'cwmt'
root = f'/root/Mono4SiM/generate/teacher_{DATASET}_mono'
base = f'/root/Mono4SiM/data/{DATASET}-{SRC}{TGT}'


def execCmd(cmd, *args):
    for arg in args:
        cmd = f'{cmd} {arg}'
    r = os.popen(cmd)
    text = r.read().rstrip()
    r.close()
    if len(text.strip()):
        print(text)
    return text


def execInteractive(cmd, *args):
    for arg in args:
        cmd = f'{cmd} {arg}'
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while p.poll() is None:
        tmp = p.stdout.readline().decode("utf-8").rstrip()
        if(len(tmp.strip())):
            print(tmp)


def cddir(dir):
    execCmd("mkdir -p", dir)
    os.chdir(dir)
    dir = execCmd("pwd").rstrip()
    if dir.endswith('/'):
        dir = dir[:-1]
    return dir


def write_now(f, str):
    f.write(str)
    f.flush()
    os.fsync(f)


def preprocess_subset(score, folder='subset', raw=False):
    os.chdir('/root/Mono4SiM/generate/')
    if raw == 'raw':
        execInteractive('bash 3s-generate_subset.sh', score, f'{folder}', SRC, TGT,
                        root, base, f'/root/Mono4SiM/data/{DATASET}-{SRC}{TGT}/ready/train.clean', '_raw')
    elif raw == 'both':
        execInteractive('bash 3s-generate_raw.sh', score, f'{folder}', SRC, TGT,
                        root, base, f'/root/Mono4SiM/generate/teacher_{DATASET}/ready/train.clean',
                        f'/root/Mono4SiM/data/{DATASET}-{SRC}{TGT}/ready/train.clean', '_both')
    else:
        execInteractive('bash 3s-generate_subset.sh', score, folder, SRC, TGT,
                        root, base, f'/root/Mono4SiM/generate/teacher_{DATASET}/ready/train.clean')


def random(root, num, len, filter=None, size=''):
    with open(f'{root}/ready/train.clean.{SRC}', 'r', encoding='utf-8') as fs,\
            open(f'{root}/ready/train.clean.{TGT}', 'r', encoding='utf-8') as ft,\
            open(f'{root}/subset/train_random{size}.{SRC}', 'w', encoding='utf-8') as ws,\
            open(f'{root}/subset/train_random{size}.{TGT}', 'w', encoding='utf-8') as wt:
        if filter is None:
            folder = 'subset'
            for src, tgt in zip(fs, ft):
                if randint(1, len) <= num:
                    ws.write(src)
                    wt.write(tgt)
                    num -= 1
                len -= 1
        else:
            folder = 'filter'
            for src, tgt, flag in zip(fs, ft, filter):
                if not flag:
                    continue
                if randint(1, len) <= num:
                    ws.write(src)
                    wt.write(tgt)
                    num -= 1
                len -= 1
    assert len == 0
    assert num == 0
    preprocess_subset(f'train_random{size}', folder)


def score_select_high(root, score, num, le, filter=None, name='', raw=False):
    with open(f'{root}/score/{score}.txt', 'r', encoding='utf-8') as f:
        lst = f.read().rstrip().split()
        lst = [float(i) for i in lst]
        assert len(lst) == le

    begin = time.time()
    print(f'Preprocess score {score} ... ...')
    if filter is None:
        idx = [i for i in range(le) if lst[i] != -1]
        folder = 'subset'
    else:
        assert len(filter) == le
        idx = [i for i in range(le) if lst[i] != -1 and filter[i]]
        folder = 'filter'

    idx = sorted(idx, key=lambda i: lst[i])
    high = sorted(idx.copy()[-num:])
    del idx
    print(f'Finished! {time.time() - begin} s')

    begin = time.time()
    print(f'Begin select {score} pairs ... ...')
    cnth = 0
    with open(f'{root}/ready/train.clean.{SRC}', 'r', encoding='utf-8') as fs,\
            open(f'{root}/ready/train.clean.{TGT}', 'r', encoding='utf-8') as ft,\
            open(f'{root}/{folder}/{score}_high{name}.{SRC}', 'w', encoding='utf-8') as wsh,\
            open(f'{root}/{folder}/{score}_high{name}.{TGT}', 'w', encoding='utf-8') as wth:
        for cnt, (src, tgt) in enumerate(zip(fs, ft)):
            if num > cnth and cnt == high[cnth]:
                wsh.write(src)
                wth.write(tgt)
                cnth += 1
    print(f'Finished! {time.time() - begin} s')

    begin = time.time()
    print(f'Begin generate {score}_high{name} ... ...')
    preprocess_subset(f'{score}_high{name}', folder, raw)
    print(f'Finished! {time.time() - begin} s')


def score_select_low(root, score, num, le, filter=None, name='', raw=False):
    with open(f'{root}/score/{score}.txt', 'r', encoding='utf-8') as f:
        lst = f.read().rstrip().split()
        lst = [float(i) for i in lst]
        assert len(lst) == le

    begin = time.time()
    print(f'Preprocess score {score} ... ...')
    if filter is None:
        idx = [i for i in range(le) if lst[i] != -1]
        folder = 'subset'
    else:
        assert len(filter) == le
        idx = [i for i in range(le) if lst[i] != -1 and filter[i]]
        folder = 'filter'
    print(len(idx), le)

    idx = sorted(idx, key=lambda i: lst[i])
    low = sorted(idx.copy()[:num])
    del idx
    print(f'Finished! {time.time() - begin} s')

    begin = time.time()
    print(f'Begin select {score} pairs ... ...')
    cnth = cntl = 0
    with open(f'{root}/ready/train.clean.{SRC}', 'r', encoding='utf-8') as fs,\
            open(f'{root}/ready/train.clean.{TGT}', 'r', encoding='utf-8') as ft,\
            open(f'{root}/{folder}/{score}_low{name}.{SRC}', 'w', encoding='utf-8') as wsl,\
            open(f'{root}/{folder}/{score}_low{name}.{TGT}', 'w', encoding='utf-8') as wtl:
        for cnt, (src, tgt) in enumerate(zip(fs, ft)):
            if num > cntl and cnt == low[cntl]:
                wsl.write(src)
                wtl.write(tgt)
                cntl += 1
    print(f'Finished! {time.time() - begin} s')

    begin = time.time()
    print(f'Begin generate {score}_low{name} ... ...')
    preprocess_subset(f'{score}_low{name}', folder, raw)
    print(f'Finished! {time.time() - begin} s')


def sentence_frequency(root, dict_file):
    dct = Dictionary.load(dict_file)
    tmp_sum = math.log(sum(dct.count))
    tmp_count = []
    for i in dct.count:
        tmp_count.append(math.log(i + 1))
    tmp_count[dct.unk_index] = 100
    with open(f'{root}/ready/train.clean.{SRC}', 'r', encoding='utf-8') as fs,\
            open(f'{root}/score/sentence_frequency.txt', 'w', encoding='utf-8') as d:
        for line in fs:
            line = line.replace('< unk >', '<unk>').strip().split()
            score = 0
            tot = 0
            for i in line:
                score = score - tmp_sum + tmp_count[dct.index(i)]
                tot += 1
            d.write(f'{float(score) / float(tot) ** 0.5}\n')


def sentence_uncertainty(root, dict_file, name=''):
    count = defaultdict(lambda: float(100.0))
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            count[line[0]] = float(line[1]) if float(line[1]) != 0 else 100.0

    with open(f'{root}/ready/train.clean.{SRC}', 'r', encoding='utf-8') as fs,\
            open(f'{root}/score/sentence_uncertainty{name}.txt', 'w', encoding='utf-8') as d:
        for line in fs:
            line = line.replace('< unk >', '<unk>').strip().split()
            score = 0.0
            tot = 0.0
            for i in line:
                score -= count[i]
                tot += 1
            if tot == 0:
                d.write('-1\n')
            else:
                d.write(f'{score / tot ** 0.5}\n')


def chunking_align(root, align, reverse=False):
    with open(align, 'r', encoding='utf-8') as corpus,\
            open(f'{root}/score/chunking_align.txt', 'w', encoding='utf-8') as d:
        for line in corpus:
            line = line.strip()
            num_chunk, tot = 0, 1e-9
            itr = re.finditer(r"(?P<i>[0-9]+)-(?P<j>[0-9]+)", line)
            left = []
            right = []
            for m in itr:
                left.append(int(m.group("j" if reverse else "i")))
                right.append(int(m.group("i" if reverse else "j")))
            while len(left) != 0:
                pair = del_pair_mono(left, right, 0)
                # chunk = [set(i) for i in pair]
                num_chunk += 1
                tot += len(set(pair[0]))
            if tot < 1:
                d.write('-1\n')
            else:
                d.write(f'{num_chunk / tot ** 0.5}\n')


def chunking_LM(root):
    en_model = kenlm.Model(f'{base}/score/{DATASET}-{SRC}{TGT}_{SRC}.bin')
    with open(f'{root}/ready/train.clean.{SRC}', 'r', encoding='utf-8') as fs,\
            open(f'{root}/score/chunking_LM.txt', 'w', encoding='utf-8') as d:
        for line in fs:
            assert len(line) != 0
            line = line.strip().split()
            pre = -inf
            begin = 0
            chunk = 1e-9
            for i in range(len(line)):
                cnt_str = ' '.join(line[begin: i + 1])
                cnt = en_model.score(cnt_str, bos=False,
                                     eos=False) / (i + 1 - begin)
                if cnt < pre:
                    chunk += 1
                    begin = i
                    cnt_str = ' '.join(line[begin:i+1])
                    pre = en_model.score(cnt_str, bos=False, eos=False)
                else:
                    pre = cnt
            d.write(f'{(chunk + 1) / len(line) ** 0.5}\n')


def del_pair_mono(left, right, i):
    l = [left[i]]
    r = [right[i]]
    del left[i], right[i]
    Flag = True
    while Flag:
        Flag = False
        for i in range(min(l), max(l) + 1):
            if i in left:
                tmp = del_pair_mono(left, right, left.index(i))
                l += tmp[0]
                r += tmp[1]
                Flag = True
        for i in range(min(r), max(r) + 1):
            if i in right:
                tmp = del_pair_mono(left, right, right.index(i))
                l += tmp[0]
                r += tmp[1]
                Flag = True

    return [l, r]


def k_anticipation_rate(root, align, k, reverse=False):
    with open(align, 'r', encoding='utf-8') as corpus,\
            open(f'{root}/score/{k}_anticipation_rate.txt', 'w', encoding='utf-8') as d:
        for line in corpus:
            line = line.strip()
            inv, tot = 0, 0
            itr = re.finditer(r"(?P<i>[0-9]+)-(?P<j>[0-9]+)", line)
            for m in itr:
                i = int(m.group("j" if reverse else "i"))
                j = int(m.group("i" if reverse else "j"))
                tot += 1
                if i - k + 1 > j:
                    inv += 1
            if tot <= 1:
                d.write('-1\n')
            else:
                d.write(f'{inv / tot ** 2}\n')


def plotScore(root, fx, fy, le):
    with open(f'{root}/score/{fx}.txt', 'r', encoding='utf-8') as f:
        x = f.read().rstrip().split()
        assert len(x) == le
    with open(f'{root}/score/{fy}.txt', 'r', encoding='utf-8') as f:
        y = f.read().rstrip().split()
        assert len(y) == le
    num = 100000
    x = [float(i) for i in x]
    y = [float(i) for i in y]
    print(np.corrcoef(x, y))
    x, y = zip(*sample(list(zip(x, y)), num))
    print("OKdy")
    plt.figure()
    plt.plot(x, y, '.', markersize=5)
    plt.savefig(f'{fx}_{fy}_ori.jpg')
    x = sorted(range(len(x)), key=lambda i: x[i])
    y = sorted(range(len(y)), key=lambda i: y[i])
    plt.figure()
    plt.plot(x, y, '.', markersize=5)
    plt.savefig(f'{fx}_{fy}_new.jpg')


def generate_filter(root, score, num, le, low=False, filter=None):
    remember = filter
    if filter is None:
        filter = [True] * le
    else:
        assert len(filter) == le
    with open(f'{root}/score/{score}.txt', 'r', encoding='utf-8') as f:
        score = f.read().rstrip().split()
        score = [float(i) for i in score]
        assert len(score) == le
    idx = [i for i in range(le) if filter[i] and score[i] != -1]
    length = len(idx)
    srt = sorted(idx, key=lambda i: score[i])
    if low:
        srt = srt[:num]
    else:
        srt = srt[-num:]

    filter = [0] * le
    for i in srt:
        filter[i] = 1
    return filter


if __name__ == '__main__':
    os.chdir('/root/Mono4SiM/generate')
    execCmd(f'mkdir -p {root}/subset')
    execCmd(f'mkdir -p {root}/filter')

    num = int(execCmd(f'wc -l < {base}/ready/train.clean.{SRC}').strip())
    assert num == int(
        execCmd(f'wc -l < {base}/ready/train.clean.{TGT}').strip())
    le = int(execCmd(f'wc -l < {root}/ready/train.clean.{SRC}').strip())
    assert le == int(
        execCmd(f'wc -l < {root}/ready/train.clean.{TGT}').strip())

    dct = f'{base}/data-bin/dict.{SRC}.txt'
    align = f'{root}/score/diag.align'
    uncertain = f'{base}/score/uncertainty.{SRC}.txt'
    uncertain_distill = f'/root/Mono4SiM/generate/teacher_cwmt/score/uncertainty.{SRC}.txt'

    random(root, num, le)

    chunking_align(root, align)
    chunking_LM(root)
    k_anticipation_rate(root, align, 3)
    score_select_high(root, 'chunking_align', num, le)
    score_select_high(root, 'chunking_LM', num, le)
    score_select_low(root, '3_anticipation_rate', num, le)

    sentence_frequency(root, dct)
    sentence_uncertainty(root, uncertain_distill)
    score_select_low(root, 'sentence_frequency', num, le)
    score_select_high(root, 'sentence_uncertainty', num, le)

    score_select_low(root, '3_anticipation_rate', num, le, filter=generate_filter(
        root, 'chunking_align', num * 1.6, le), name='_chunking_align')
    score_select_low(root, '3_anticipation_rate', num, le, filter=generate_filter(
        root, 'chunking_LM', num * 1.6, le), name='_chunking_LM')

    lst_score = ['chunking_align', 'chunking_LM', '3_anticipation_rate',
                 'sentence_frequency', 'sentence_uncertainty']
    for i in lst_score:
        with open(f'{root}/score/{i}.txt', 'r', encoding='utf-8') as f:
            x = f.read().rstrip().split()
            assert len(x) == le
        for j in lst_score:  
            with open(f'{root}/score/{j}.txt', 'r', encoding='utf-8') as f:
                y = f.read().rstrip().split()
                assert len(y) == le
            x = [float(i) for i in x]
            y = [float(i) for i in y]
            print(i, j)
            print(np.corrcoef(x, y))
