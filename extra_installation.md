# Extra Installation

## Install PyTorch versions

First install pytorch binaries with specific version of cuda. We will use `pytorch 1.8.1 + CUDA 11.1` as our example. 

```
torch 1.8.X+cu102/cu111             ==> CUDA 10.2 / 11.1
torch 1.7.X+cu92/cu101/cu102/cu110  ==> CUDA 9.2 / 10.1 / 10.2 / 11.0
torch 1.6.X+cu92/cu101/cu102        ==> CUDA 9.2 / 10.1 / 10.2
```

## Install matching nvcc compiler

Apex will ask for same nvcc compiler version as that used to compile pytorch binary. If your system's CUDA Toolkit is of a different version than that of your pytorch binary, you need to install a matching one.

### Check NVCC Compatibility

Check your nvcc version (if any) by running

```bash
nvcc -V
```

If this version is the same as your pytorch cuda version (in our case 11.1), then you can skip to [Install apex](#install-apex).

### Download Legacy CUDA Toolkit

Go to [CUDA Archive](https://developer.nvidia.com/cuda-toolkit-archive) to download the specific version of CUDA used to compile your installed pytorch binary. For example, download the runfile installation for Ubuntu 20.04 by

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
```

### User install

We only need the `nvcc` compiler so we will use user installation which does not require root. 

1. Run the installation as non-root user

```bash
bash cuda_11.1.1_455.32.00_linux.run
```

2. De-select **Driver installation, Samples, Demo** and **Documentation**, as we don't need them.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                          <                      │
│      [ ] 455.32.00                                                           │
│ + [X] CUDA Toolkit 11.1                                                      │
│   [ ] CUDA Samples 11.1                               <                      │
│   [ ] CUDA Demo Suite 11.1                            <                      │
│   [ ] CUDA Documentation 11.1                         <                      │
│   Options                                                                    │
│   Install                                                                    │
```

2. Set **Options --> Library install** path to a non-root path

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Options                                                                      │
│   Driver Options                                                             │
│   Toolkit Options                                                            │
│   Samples Options                                                            │
│   Library install path (Blank for system default)     <                      │
│   Done                                                                       │
```

3. Set **Options --> Toolkit** options 1) set the **install path** to non-root path 2) de-select the **symbolic link, shortcuts** and **manpage documents**.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Toolkit                                                                 │
│   Change Toolkit Install Path                         <                      │
│   [ ] Create symbolic link from /usr/local/cuda       <                      │
│ - [ ] Create desktop menu shortcuts                   <                      │
│      [ ] Yes                                                                 │
│      [ ] No                                                                  │
│   [ ] Install manpage documents to /usr/share/man     <                      │
│   Done                                                                       │
```

4. Install. You should get the following

```
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in <your-non-root-path>
Samples:  Not Selected
```

## Install apex

Run the following command to install apex

## SimulEval

We updated SimulEval with two functionalities:

1. To evaluate computational aware (CA) latency metrics for text.
2. Save actual system predictions to a file, so that multi-reference BLEU can be calculated. (we found that `instances.log` forcefully add whitespaces, which is undesired for Chinese.)

```bash
git clone XXXX-1
```

Alternatively, you can use the official repository if you're skeptical of our modifications. Though you need to extract predictions manually and the result for Chinese might be inaccurate.

```bash
git clone https://github.com/facebookresearch/SimulEval.git
```

You need to add the following lines to the class `TextInstance` in `SimulEval/simuleval/scorer/instance.py` in order to obtain computational aware (CA) latency metrics:

```python
# class TextInstance(Instance):
# add following function to TextInstance
    def sentence_level_eval(self):
        super().sentence_level_eval()
        # For the computation-aware latency
        self.metrics["latency_ca"] = eval_all_latency(
            self.elapsed, self.source_length(), self.reference_length() + 1)
```

Regardless of which approach you use, proceed to install the package via pip:

```bash
cd SimulEval
pip install -e .
```

## SacreBLEU

To evaluate Translation Edit Rate (TER) or enable bootstrap resampling, we need to use SacreBLEU v2.0.0. However, version 2 currently **breaks compatibility** with the version of fairseq that we use. The solution is to use python venv to create an environment only for evaluation:

```bash
python -m venv ~/envs/sacrebleu2
```

Activate it by:

```bash
source ~/envs/sacrebleu2/bin/activate
```

Install sacrebleu version 2

```bash
git clone https://github.com/mjpost/sacrebleu.git
cd sacrebleu
pip install .
```

Then you can use sacrebleu v2, without breaking fairseq.
