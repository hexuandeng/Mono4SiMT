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
@inproceedings{deng2023mono4simt,
    title = "Improving Simultaneous Machine Translation with Monolingual Data",
    author = "Deng, Hexuan and Ding, Liang and Liu, Xuebo and Zhang, Meishan and Tao, Dacheng and Zhang, Min",
    booktitle = "Proceedings of AAAI",
    year = "2023",
}
```
