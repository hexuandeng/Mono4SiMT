import argparse
import os


def execCmd(cmd, *args):
    for arg in args:
        cmd = f"{cmd} {arg}"
    r = os.popen(cmd)
    text = r.read()
    r.close()
    print(text)
    return text


def proc_generate(root, file):
    record = {}
    with open(f'{root}/{file}', 'r', encoding='utf-8') as f:
        for i in f.readlines():
            if '\t' in i and i[0] == 'D':
                num, sen = i.rstrip().split('\t', 1)
                record[int(num[2:])] = sen
    with open(f'{root}/detok.txt', 'w', encoding='utf-8') as f:
        i = 0
        while i in record:
            f.write(record[i].split('\t', 1)[1] + '\n')
            i += 1
    print(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default=None)
    parser.add_argument("--file", "-f", type=str, default='generate-test.txt')
    args = parser.parse_args()
    proc_generate(args.root, args.file)
