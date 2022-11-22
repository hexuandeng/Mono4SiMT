#!/usr/bin/env python
import sys
import argparse
import re
import numpy as np


def distance(align_lines, reverse):
    dists = []
    pattern = re.compile(r"(?P<i>[0-9]+)-(?P<j>[0-9]+)")
    for line in align_lines:
        all_i = []
        all_j = []
        for si, sj in pattern.findall(line):
            i = int(sj if reverse else si)
            j = int(si if reverse else sj)
            all_i.append(i)
            all_j.append(j)

        min_i = min(all_i)
        min_j = min(all_j)
        max_i = max(all_i)
        max_j = max(all_j)
        for i, j in zip(all_i, all_j):
            tgt = (i - min_i) / (max_i - min_i + 1e-9) * (max_j - min_j) + min_j
            dists.append(abs(tgt - j))

    return np.mean(dists), np.std(dists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--reverse", "-r", action="store_true")
    args = parser.parse_args()

    if args.input is not None:
        with open(args.input, "r") as f:
            print(distance(f.readlines(), reverse=args.reverse))
    else:
        print(distance(sys.stdin.readlines(), reverse=args.reverse))
