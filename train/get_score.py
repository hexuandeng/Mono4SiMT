import csv
from test_test import Aligner
from subprocess import Popen, PIPE, STDOUT

dataset = 'cwmt-enzh'
root = '/root/Mono4SiM/train/cwmt-enzh/checkpoints'

lst = ['raw', '', 'train_random_subset',
       'sentence_frequency_low_subset',
       'sentence_uncertainty_high_subset',
       '3_anticipation_rate_low_subset',
       'chunking_align_high_subset',
       'chunking_LM_high_subset',
       '3_anticipation_rate_low_chunking_align_filter',
       '3_anticipation_rate_low_chunking_LM_filter']

csvfile = open(f'/root/Mono4SiM/train/{dataset}.csv', 'a', encoding='utf-8')
writer = csv.writer(csvfile)
align = Aligner(f'/root/Mono4SiM/data/{dataset}/score/fwd_align',
                f'/root/Mono4SiM/data/{dataset}/score/fwd_err',
                f'/root/Mono4SiM/data/{dataset}/score/rev_align',
                f'/root/Mono4SiM/data/{dataset}/score/rev_err')


def execCmd(cmd, *args):
    for arg in args:
        cmd = f'{cmd} {arg}'
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while p.poll() is None:
        print(p.stdout.readline().decode("utf-8").rstrip())


def proc(str):
    return str[str.rfind(':') + 1:].strip()


for i in lst:
    execCmd(f'bash /root/Mono4SiM/train/{dataset}/2-test_model.sh', i)
    for k in range(1, 10, 2):
        try:
            base = f'{root}/wait_{k}_enzh_distill_{i}/log'
            if i == 'raw':
                base = f'{root}/wait_{k}_enzh_/log'
            align.align(base)

            with open(f'{base}/sacrebleu.txt', 'r', encoding='utf-8') as f:
                tmp = f.read().split(',')
                q = proc(tmp[1])
                asw = proc(tmp[3])
                e = proc(tmp[11])
                asw = asw[1:asw.rfind('(')].strip().split('/')
                asw.insert(0, q)
                asw.insert(0, e)

            with open(f'{base}/scores', 'r', encoding='utf-8') as f:
                tmp = f.read().split(',')
                asw.append(proc(tmp[1]))
                asw.append(proc(tmp[3]))
                asw.append(proc(tmp[5]))

            with open(f'{base}/align_scores.txt', 'r', encoding='utf-8') as f:
                tmp = f.read().strip().split()
                anti = tmp[3:36:4]
                chunk = tmp[-1]
                sumn = 0
                for m in anti:
                    sumn += float(m)
                sumn /= len(anti)
                asw.append(anti[k - 1])
                asw.append(chunk)

            writer.writerow([f'wait_{k}_{i}'] + asw)

        except Exception as e:
            print(f'wait_{k}_enzh_distill_{i} Error!')
            print(e)

    writer.writerow('')

csvfile.close()
