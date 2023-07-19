# Improving Simultaneous Machine Translation with Monolingual Data

## Setup

1. Install fairseq
Stick to the specified checkout version to avoid compatibility issues.

```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 8b861be
python setup.py build_ext --inplace
pip install .
```

2. (Optional) [Install apex](docs/apex_installation.md) for faster mixed precision (fp16) training.

3. Install dependencies (clone in folder [utility](utility/README.md) if possible).

```bash
pip install -r requirements.txt
```

For the installation guide, see [extra_installation](extra_installation.md).

## Data Preparation

All corresponding bashes are in folder data.

1. To download corresponding datasets, go to [Google Drive](https://drive.google.com/drive/folders/1HbzxBD0klgX-EugVGB36CFVdObJJ5Uk7?usp=sharing) for cleaned dataset, or run bashes begin with 0.

```bash
cd data
bash 0-get_data_cwmt.sh
bash 0-get_en_mono.sh
```

2. After distilling, run [1-preprocess-distill.py](data/1-preprocess-distill.py) to preprocess those data, and then run bashes beginning with 2 to calculate corresponding scores.

```bash
cd data
python 1-preprocess-distill.py
bash 2-train_align.sh
bash 2-train_kenlm.sh
bash 2-fast-align.sh
bash 2-k-anticipation.sh
python 2-get_uncertainty.py
```

3. Finally, run [3-scoring_preprocessing.py](data/3-scoring_preprocessing.py) to calculate the score of the distilled data and extract the data according to the metrics we propose.

```bash
cd data
python 3-scoring_preprocessing.py
```

**Note** that you need to change the [data path](data\data_path.sh) mannually.

## Training

We need a full-sentence model as teacher for sequence-KD.

The following command will train the teacher model.

```bash
cd train/cwmt-enzh
bash 0-teacher.sh
```

To distill the training set, run

```bash
cd train/cwmt-enzh
bash 0-distill_enzh_mono.sh
```

We provide our dataset including distill set and pseudo reference set for easier reproducibility.

We can now train vanilla wait-k model. To do this, run

```bash
bash 1b-distill_all_wait_k.sh generate/teacher_cwmt_mono/data-bin 3_anticipation_rate_low_chunking_LM_filter
```

*3_anticipation_rate_low_chunking_LM_filter* is the default name of our best strategy, change this field to run wait-k under any dataset (raw for original bilingual datasets).

Our models are released at [Google Drive](https://drive.google.com/drive/folders/19aPnAPvT75KmlLA2Y0VipNJVF3cf3CaP?usp=sharing).

## Evaluation (SimulEval)

Install [SimulEval](docs/extra_installation.md).

### full-sentence model

```bash
cd train/cwmt-enzh
bash 2-test_model_full.sh
```

### wait-k models

```bash
cd train/cwmt-enzh
bash 2-test_model.sh 3_anticipation_rate_low_chunking_LM_filter
```

Change *3_anticipation_rate_low_chunking_LM_filter* to run evaluation under any dataset (raw for original bilingual datasets).

or simply run:

```bash
cd train
python get_score.py
```

for all subsets.


## Citation
If you find this work helpful, please consider citing as follows:
```bibtex
@article{Deng_Ding_Liu_Zhang_Tao_Zhang_2023,
    title={Improving Simultaneous Machine Translation with Monolingual Data},
    volume={37},
    url={https://ojs.aaai.org/index.php/AAAI/article/view/26497},
    DOI={10.1609/aaai.v37i11.26497},
    abstractNote={Simultaneous machine translation (SiMT) is usually done via sequence-level knowledge distillation (Seq-KD) from a full-sentence neural machine translation (NMT) model. However, there is still a significant performance gap between NMT and SiMT. In this work, we propose to leverage monolingual data to improve SiMT, which trains a SiMT student on the combination of bilingual data and external monolingual data distilled by Seq-KD. Preliminary experiments on En-Zh and En-Ja news domain corpora demonstrate that monolingual data can significantly improve translation quality (e.g., +3.15 BLEU on En-Zh). Inspired by the behavior of human simultaneous interpreters, we propose a novel monolingual sampling strategy for SiMT, considering both chunk length and monotonicity. Experimental results show that our sampling strategy consistently outperforms the random sampling strategy (and other conventional typical NMT monolingual sampling strategies) by avoiding the key problem of SiMT -- hallucination, and has better scalability. We achieve +0.72 BLEU improvements on average against random sampling on En-Zh and En-Ja. Data and codes can be found at https://github.com/hexuandeng/Mono4SiMT.},
    number={11},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Deng, Hexuan and Ding, Liang and Liu, Xuebo and Zhang, Meishan and Tao, Dacheng and Zhang, Min},
    year={2023},
    month={Jun.},
    pages={12728-12736} 
}
```
