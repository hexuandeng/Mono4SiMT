#!/usr/bin/env python

import argparse
import numpy as np
from multiprocessing import Pool
import tqdm

def reorder(inputs):
    """
    srcseq, tgtseq: ["tokens", ...]
    alnseq: ["0-0", "6-0", ...]
    """
    srcstr, tgtstr, alnstr = inputs
    srcseq = inputs[0].strip().split()
    tgtseq = inputs[1].strip().split()
    alnseq = inputs[2].strip().split()
    tlen = len(tgtseq)
    null = -1
    new_order = np.full(tlen, null)
    for s_t in alnseq:
        s, t = tuple(map(int, s_t.split('-')))  # (0,0)
        new_order[t] = s

    for i in range(tlen):
        if new_order[i] == null:
            new_order[i] = new_order[i - 1] if i > 0 else 0

    reordered = [tgtseq[i] for i in new_order.argsort(kind='stable')]

    return reordered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="source file")
    parser.add_argument("-t", "--target", type=str, help="target file")
    parser.add_argument("-a", "--align", type=str, help="alignment file")
    parser.add_argument("-o", "--output", type=str, help="output file")
    parser.add_argument("-j", "--jobs", type=int, help="launch j parallel jobs.")
    args = parser.parse_args()
    print(args)

    def file_len(fname):
        import subprocess
        intstr = subprocess.getoutput(f'cat {fname} | wc -l')
        return int(intstr)

    srcs = open(args.source, "r")
    tgts = open(args.target, "r")
    alns = open(args.align, "r")

    srclen = file_len(args.source)
    tgtlen = file_len(args.target)
    alnlen = file_len(args.align)
    assert srclen == tgtlen and alnlen == tgtlen

    with Pool(args.jobs) as p:
        results = list(tqdm.tqdm(p.imap(reorder, zip(srcs, tgts, alns)), total=srclen))

    srcs.close()
    tgts.close()
    alns.close()

    with open(args.output, "w") as f:
        for line in results:
            f.write(" ".join(line) + "\n")
