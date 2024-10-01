from glob import glob
import pickle
from tqdm import tqdm
import torch
import pandas as pd
import sys
import numpy as np
seq_vocab = ['A', 'C', 'G', 'U', 'N']
struct_vocab = ['.','(', ')','[',']','{','}','<','>']
seq_itos = dict(zip(range(1,len(seq_vocab)+1), seq_vocab))
struct_itos = dict(zip(range(len(struct_vocab)), struct_vocab))
seq_stoi = dict(zip(seq_vocab,range(1,len(seq_vocab)+1)))
struct_stoi = dict(zip(struct_vocab,range(len(struct_vocab))))
algo={"syn_ns2":"liblearna","syn_ns_gc2":"liblearna_gc","syn_ns_l2":"learna","syn_ns_meta_l2":"meta-learna","syn_ns_meta_a_l2":"meta-adapt-learna"}

def comb_metrics(path):
    g_path = path + f"*2/"
    save_path = path + f"metrics_dup.csv"
    files = glob(g_path + "metrics_dup.csv")
    cdfs = []
    for i in files:
        version = algo[i.split("/")[2]]
        df = pd.read_csv(i)
        df["Algo"] = version
        print(i)
        cdfs.append(df)
    cdf = pd.concat(cdfs)
    cdf.to_csv(save_path)

comb_metrics("runs/liblearna/")