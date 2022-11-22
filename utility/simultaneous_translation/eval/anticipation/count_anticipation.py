#!/usr/bin/env python
import sys
import argparse
import re


def kAR(align, k, reverse=False, sent=False):
    if sent:
        corpus = align.strip().split("\n")
    else:
        corpus = [align]

    output = []

    for line in corpus:
        inv, tot = 0, 1e-9
        itr = re.finditer(r"(?P<i>[0-9]+)-(?P<j>[0-9]+)", line)
        for m in itr:
            i = int(m.group("j" if reverse else "i"))
            j = int(m.group("i" if reverse else "j"))
            tot += 1
            if i - k + 1 > j:
                inv += 1
        output.append(str(inv / tot))
    return "\n".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--reverse", "-r", action="store_true")
    parser.add_argument("--delay", "-k", type=int, required=True)
    parser.add_argument("--sentence-level", "-s", action="store_true")
    args = parser.parse_args()

    if args.input is not None:
        with open(args.input, "r") as f:
            print(kAR(f.read(), k=args.delay,
                  reverse=args.reverse, sent=args.sentence_level))
    else:
        print(kAR(sys.stdin.read(),
              k=args.delay, reverse=args.reverse, sent=args.sentence_level))
