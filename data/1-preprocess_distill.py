import os
from collections import defaultdict
from subprocess import Popen, PIPE, STDOUT


src = 'en'
tgt = 'zh'
dataset = 'cwmt'
base = f'/root/Mono4SiM/data/{dataset}-{src}{tgt}'
root = f'/root/Mono4SiM/generate/teacher_{dataset}_mono'
spm_prefix = f'{base}/prep/spm_unigram32000'


def execCmd(cmd, *args):
    for arg in args:
        cmd = f"{cmd} {arg}"
    r = os.popen(cmd)
    text = r.read().rstrip()
    r.close()
    if len(text.strip()):
        print(text)
    return text


def execInteractive(cmd, *args):
    for arg in args:
        cmd = f"{cmd} {arg}"
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while p.poll() is None:
        print(p.stdout.readline().decode("utf-8"), end = "")


def cddir(dir):
    execCmd("mkdir -p", dir)
    os.chdir(dir)
    dir = execCmd("pwd").rstrip()
    if not dir.endswith('/'):
        dir += '/'
    return dir


def proc_interactive(root, n):
    print(f"Processing interactive {n}")
    lst = ['S', 'D', 'P']
    record = defaultdict(dict)

    with open(f'{root}/interactive/generate-train.{n}.txt', 'r', encoding='utf-8') as f:
        for i in f.readlines():
            if '\t' in i and i[0] in lst:
                num, sen = i.rstrip().split('\t', 1)
                record[i[0]][int(num[2:])] = sen
    with open(f'{root}/score/generate_ppl.txt', 'a', encoding='utf-8') as p,\
            open(f'{root}/interactive/detok.{tgt}', 'a', encoding='utf-8') as d,\
            open(f'{root}/interactive/detok.{src}', 'a', encoding='utf-8') as s:
        i = 0
        while i in record['S']:
            try:
                lst = [float(num) for num in record['P'][i].split()]
                p.write(f'{sum(lst)}\t{len(lst)}\n')
                d.write(record['D'][i].split('\t', 1)[1] + '\n')
                s.write(record['S'][i] + '\n')
            except:
                print(record['S'][i])
            i += 1
        i += 1
        while i in record['S']:
            try:
                lst = [float(num) for num in record['P'][i].split()]
                p.write(f'{sum(lst)}\t{len(lst)}\n')
                d.write(record['D'][i].split('\t', 1)[1] + '\n')
                s.write(record['S'][i] + '\n')
            except:
                print(record['S'][i])
            i += 1
        i += 1
        while i in record['S']:
            try:
                lst = [float(num) for num in record['P'][i].split()]
                p.write(f'{sum(lst)}\t{len(lst)}\n')
                d.write(record['D'][i].split('\t', 1)[1] + '\n')
                s.write(record['S'][i] + '\n')
            except:
                print(record['S'][i])
            i += 1
        i += 1
        while i in record['S']:
            try:
                lst = [float(num) for num in record['P'][i].split()]
                p.write(f'{sum(lst)}\t{len(lst)}\n')
                d.write(record['D'][i].split('\t', 1)[1] + '\n')
                s.write(record['S'][i] + '\n')
            except:
                print(record['S'][i])
            i += 1
    print(i)
    return i


def join_file(root, subset, lang, rm=False):
    cmd = 'cat'
    for i in range(8):
        cmd += f' {root}/{subset}.{i}.{lang}'
    execCmd(cmd, f'> {root}/{subset}.{lang}')
    if rm:
        for i in range(8):
            execCmd(f'rm {root}/{subset}.{i}.{lang}')


if __name__ == '__main__':
    os.chdir('/root/Mono4SiM/generate')
    execInteractive(f'mkdir -p {root}/score')
    execInteractive(f'mkdir -p {root}/ready')
    execInteractive(f'rm -f {root}/score/generate_ppl.txt')
    execInteractive(f'rm -f {root}/interactive/detok.{src}')
    execInteractive(f'rm -f {root}/interactive/detok.{tgt}')

    print("Running Tokenizer...")
    for i in range(8):
        proc_interactive(root, i)
        execCmd(f'wc -l < /root/Mono4SiM/data/mono-en/split/train.en.{i}')

    preprocess = 'false' if 'mono' in root else 'true'
    execCmd(f'cp {base}/ready/valid.{src} {root}/ready/valid.{src}')
    execCmd(f'cp {base}/ready/valid.{tgt} {root}/ready/valid.{tgt}')
    execCmd(f'cp {base}/ready/test.{src} {root}/ready/test.{src}')
    execCmd(f'cp {base}/ready/test.{tgt} {root}/ready/test.{tgt}')
    execInteractive(
        f'bash 1-preprocess_tokenizer.sh {root} {spm_prefix} {src} {tgt} {preprocess}')
