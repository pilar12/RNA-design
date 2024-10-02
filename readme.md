# RNAinformer

This repository contains the source code to [*RNAinformer: Generative RNA Design with Tertiary Interactions*]



### Install virtual environment

. Support for Turing GPUs (T4, RTX 2080) is coming soon. 

```
conda env create -f environment.yml

conda activate rnadesign

```
The Flash Attention package currently requires a Ampere, Ada, or Hopper GPU (e.g., A100, RTX 3090, RTX 4090, H100). To install Falsh-attn.

```
pip install -U --no-cache-dir flash-attn==2.3.4

git clone https://github.com/HazyResearch/flash-attention \
    && printf "[safe]\n\tdirectory = /flash-attention" > ~/.gitconfig \
    && git config --global --add safe.directory /home/user/flash-attention \
    && cd flash-attention && git checkout v2.3.4 \
    && cd csrc/fused_softmax && pip install . && cd ../../ \
    && cd csrc/rotary && pip install . && cd ../../ \
    && cd csrc/xentropy && pip install . && cd ../../ \
    && cd csrc/layer_norm && pip install . && cd ../../ \
    && cd csrc/fused_dense_lib && pip install . && cd ../../ \
    && cd csrc/ft_attention && pip install . && cd ../../ \
    && cd .. && rm -rf flash-attention
```
### Data
Download and unzip data from https://www.dropbox.com/scl/fi/yaxvlsloht21i7bho2tim/data.tar.xz?rlkey=jmxqbjjcmbumt08hk2tbqxvgg&st=dy7xizw6&dl=0

### Models and predictions
Download and unzip data from

### Evaluate 
Evaluation metrics have been provided for all model and competitiors in respective metrics.csv files
To run all evaluations again

```
bash run_evaluation.sh
```

### Infernece on test sets
```
python inference.py --seed 9647359 --path path/to/model/folder/
```
Eg.
```
python inference.py --seed 9647359 --path runs/syn_pdb/version_0/
```