import os
import tqdm
import argparse
import pathlib
import torch
import urllib.request
import logging

import torch.cuda
import numpy as np
import pandas as pd

from RNAformer.model.RNAformer import RiboFormer
from RNAformer.pl_module.datamodule_rna import IGNORE_INDEX, PAD_INDEX
from RNAformer.utils.data.rna import CollatorRNA
from RNAformer.utils.configuration import Config
from eval_predictions import evaluate, print_dict_tables
import glob
logger = logging.getLogger(__name__)

seq_vocab = ['A', 'C', 'G', 'U', 'N']
struct_vocab = ['.','(', ')']

seq_itos = dict(zip(range(1,len(seq_vocab)+1), seq_vocab))

struct_itos = dict(zip(range(len(struct_vocab)), struct_vocab))

class EvalRNAformer():

    def __init__(self, model_dir, precision=16, flash_attn=False):

        model_dir = pathlib.Path(model_dir)

        config = Config(config_file=model_dir / 'config.yml')
        state_dict = torch.load(model_dir / 'state_dict.pth')

        if precision == 32 or flash_attn == False:
            config.trainer.precision = 32
            config.RNAformer.precision = 32
            config.RNAformer.flash_attn = False
        elif precision == 16 or precision == 'fp16':
            config.trainer.precision = 16
            config.RNAformer.precision = 16
            config.RNAformer.flash_attn = True
        elif precision == 'bf16':
            config.trainer.precision = 'bf16'
            config.RNAformer.precision = 'bf16'
            config.RNAformer.flash_attn = True

        model_config = config.RNAformer
        model_config.seq_vocab_size = 5
        model_config.max_len = state_dict["seq2mat_embed.src_embed_1.embed_pair_pos.weight"].shape[1]

        model = RiboFormer(model_config)
        model.load_state_dict(state_dict, strict=True)

        model = model.cuda()
        if precision == 16 or precision == 'fp16' or precision == 'bf16':
            model = model.half()
        self.model = model.eval()

        self.ignore_index = IGNORE_INDEX
        self.pad_index = PAD_INDEX

        self.collator = CollatorRNA(self.pad_index, self.ignore_index)

    def __call__(self, sequence, mean_triual=True):
        length = len(sequence[0])
        if length > 200:
            return []
        input_sample = sequence
        input_samples = [{'src_seq': input_sample[i]-1, 'length': torch.tensor(length)} for i in range(len(input_sample))]
        batch = self.collator(input_samples)
        pred_mats=[]
        for i in range(len(batch['src_seq'])):
            with torch.no_grad():
                logits, mask, latents = self.model(batch['src_seq'][i].unsqueeze(0).cuda(), batch['length'][i].unsqueeze(0).cuda(), infer_mean=True)
            sample_logits = logits[0, :length, :length, -1].detach()
            # triangle mask
            if mean_triual:
                low_tr = torch.tril(sample_logits, diagonal=-1)
                upp_tr = torch.triu(sample_logits, diagonal=1)
                mean_logits = (low_tr.t() + upp_tr) / 2
                sample_logits = mean_logits + mean_logits.t()

            pred_mat = torch.sigmoid(sample_logits) > 0.5
            pred_mats.append(pred_mat.cpu().numpy())
        
        return pred_mats
        with torch.no_grad():
            logits, mask = self.model(batch['src_seq'].cuda(), batch['length'].cuda(), infer_mean=True)
        sample_logits = logits[:, :length, :length, -1].detach()
        # triangle mask
        if mean_triual:
            low_tr = torch.tril(sample_logits, diagonal=-1)
            upp_tr = torch.triu(sample_logits, diagonal=1)
            mean_logits = (low_tr.transpose(1,2) + upp_tr) / 2
            sample_logits = mean_logits + mean_logits.transpose(1,2)

        pred_mat = torch.sigmoid(sample_logits) > 0.5

        return pred_mat.cpu().numpy()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train RNAformer')
    parser.add_argument('-n', '--model_name', type=str, default="ts0_conform_dim256_cycling_32bit")
    parser.add_argument('-m', '--model_dir', type=str, )
    parser.add_argument('-f', '--flash_attn', type=bool, default=False )
    parser.add_argument('-p', '--precision', type=int, default=32 )
    parser.add_argument('-s', '--save_predictions', type=bool, default=True)

    args, unknown_args = parser.parse_known_args()

    if args.model_dir is None:
        model_dir = f"checkpoints/{args.model_name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print("Downloading model checkpoints")
            urllib.request.urlretrieve(
                f"https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/{args.model_name}/config.yml",
                f"checkpoints/{args.model_name}/config.yml")
            urllib.request.urlretrieve(
                f"https://ml.informatik.uni-freiburg.de/research-artifacts/RNAformer/{args.model_name}/state_dict.pth",
                f"checkpoints/{args.model_name}/state_dict.pth")
    else:
        model_dir = args.model_dir


    eval_model = EvalRNAformer(model_dir, precision=args.precision, flash_attn=args.flash_attn)

    def count_parameters(parameters):
        return sum(p.numel() for p in parameters)
    print(f"Model size: {count_parameters(eval_model.model.parameters())}")

    file = "/work/dlclarge2/patilsh-aptamer-design/RNA-design/runs/bprna_multi/version_1/"
    dataset_type = "model_infer"
    pred_files = glob.glob(file+"*.pt")
    os.makedirs(file+"val_predictions", exist_ok=True)
    for pred_file in pred_files:
        pred_file = pred_file.split("/")[-1]
        struct_file = file+"val_predictions/"+pred_file[:-3]+"_structs.pt"
        if "final" not in pred_file:
            epoch = int(pred_file.split("_")[-1][:-3])+1
            if epoch%20 != 0:
                continue
        if os.path.exists(struct_file):
            continue
        else:
            print("Processing ", pred_file)
            dataset = torch.load(file+pred_file)
            processed_samples = []
            if dataset_type == "model_infer":
                for sample in tqdm.tqdm(dataset):
                    pred_mat = eval_model(sample, mean_triual=True)
                    processed_samples.append(pred_mat)      
            else:
                for sample in tqdm.tqdm(dataset):
                    pred_mat = eval_model([sample['trg_seq'][1:]], mean_triual=True)
                    processed_samples.append(pred_mat)
            if args.save_predictions:
                torch.save(processed_samples,struct_file)