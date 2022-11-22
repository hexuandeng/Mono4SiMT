import os
import re
from collections import defaultdict
from subprocess import Popen, PIPE, STDOUT

def exec(cmd, *args):
    for arg in args:
        cmd = f'{cmd} {arg}'
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while p.poll() is None:
        tmp = p.stdout.readline().decode("utf-8").rstrip()
        if(len(tmp.strip())):
            print(tmp)


def get_dict(file):
    lst = ['S', 'T', 'D']
    record = defaultdict(dict)
    with open(file, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            if '\t' in i and i[0] in lst:
                num, sen = i.rstrip().split('\t', 1)
                record[int(num[2:])][i[0]] = sen
    return record


def chunking_align(align, src, reverse=False):
    with open(align, 'r', encoding='utf-8') as corpus,\
        open(src, 'r', encoding='utf-8') as en:
        num_chunk, tot = 0, 1e-9
        num_align, tot_align = 0, 1e-9
        for line, sen in zip(corpus, en):
            line = line.strip()
            tot_align += float(len(sen.strip().split()))
            itr = re.finditer(r"(?P<i>[0-9]+)-(?P<j>[0-9]+)", line)
            left = []
            right = []
            for m in itr:
                left.append(int(m.group("j" if reverse else "i")))
                right.append(int(m.group("i" if reverse else "j")))
            num_align += float(len(set(left)))

            while len(left) != 0:
                pair = del_pair_mono(left, right, 0)
                num_chunk += 1
                tot += len(set(pair[0]))
    return tot / num_chunk, num_align / tot_align


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


class Aligner:
    def __init__(self, fwd_params, fwd_err, rev_params, rev_err, heuristic='grow-diag-final-and'):
        self.fast_align = '/root/Mono4SiM/utility/fast_align/build/fast_align'
        self.atools = '/root/Mono4SiM/utility/fast_align/build/atools'
        self.fwd_params = fwd_params
        self.rev_params = rev_params
        self.heuristic = heuristic
        (self.fwd_T, self.fwd_m) = self.read_err(fwd_err)
        (self.rev_T, self.rev_m) = self.read_err(rev_err)

    def align(self, prefix):
        dict = get_dict(f'{prefix}/generate-test.txt')
        with open(f'{prefix}/tmp.en', 'w', encoding='utf-8') as en,\
            open(f'{prefix}/tmp.zh', 'w', encoding='utf-8') as zh:
            for j in range(3003):
                en.write(dict[j]['S'] + '\n')
                zh.write(dict[j]['D'].split('\t')[-1] + '\n')
        exec('bash /root/Mono4SiM/train/cwmt-enzh/2s-encode_test.sh', prefix, 'tmp')
        exec(self.fast_align, '-i', f'{prefix}/align_process', '-d', '-v', '-o',
             '-T', self.fwd_T, '-m', self.fwd_m, '-f', self.fwd_params, f'> {prefix}/tmp.fwd')
        with open(f'{prefix}/tmp.fwd', 'r', encoding='utf-8') as r,\
                open(f'{prefix}/fwd', 'w', encoding='utf-8') as w:
            for i in r:
                w.write(i.split('|||')[2].strip()+'\n')
        exec(self.fast_align, '-i', f'{prefix}/align_process', '-d', '-v', '-o',
             '-T', self.rev_T, '-m', self.rev_m, '-f', self.rev_params, f'-r > {prefix}/tmp.rev')
        with open(f'{prefix}/tmp.rev', 'r', encoding='utf-8') as r,\
                open(f'{prefix}/rev', 'w', encoding='utf-8') as w:
            for i in r:
                w.write(i.split('|||')[2].strip()+'\n')
        exec(self.atools, '-i', f'{prefix}/fwd', '-j',
             f'{prefix}/rev', '-c', self.heuristic, f'> {prefix}/align_process')
        exec('/root/Mono4SiM/utility/scripts/run_aligner.sh', f'{prefix}/align_process > {prefix}/align_scores.txt')
        with open(f'{prefix}/align_scores.txt', 'a', encoding='utf-8') as f:
            score1, score2 = chunking_align(f'{prefix}/align_process', f'{prefix}/tmp.en')
            f.write(f'\nChunk Avg Lengths:\n{score2}\t{score1}\n')

        exec(f'rm {prefix}/fwd')
        exec(f'rm {prefix}/rev')
        exec(f'rm {prefix}/tmp.fwd')
        exec(f'rm {prefix}/tmp.rev')
        exec(f'rm {prefix}/align_process')

    def read_err(self, err):
        (T, m) = ('', '')
        for line in open(err):
            # expected target length = source length * N
            if 'expected target length' in line:
                m = line.split()[-1]
            # final tension: N
            elif 'final tension' in line:
                T = line.split()[-1]
        return (T, m)
