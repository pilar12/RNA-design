import torch
from glob import glob
from tqdm import tqdm
import argparse
import os

seq_vocab = ['A', 'C', 'G', 'U', 'N']
struct_vocab = ['.','(', ')','[',']','{','}','<','>']
seq_itos = dict(zip(range(1,len(seq_vocab)+1), seq_vocab))
struct_itos = dict(zip(range(len(struct_vocab)), struct_vocab))

def hotknots(seq) :
    cmd= os.environ.get("HOTKNOTS_ROOT")+"./bin/HotKnots -s  {} -m {} 2>/dev/null".format(seq, "CC")
    p = os.popen(cmd)
    rst = p.read().split('\n')
    rst = rst[2].split()
    if len(rst) > 0 :
        return (rst[-2],rst[-1])
    else :
        print("ERROR during the folding with Hotknots")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    for path in glob(f"{args.path}/*preds.pt"):
        outfile = path.replace(".pt",f"_structs.pt")
        structs = []
        preds=torch.load(path, map_location=torch.device('cpu'))
        ds = path.split('/')[-1].replace("_preds.pt","")
        for i,seqs in tqdm(enumerate(preds)):
            s=[]
            for j,seq in enumerate(seqs):
                seq = "".join(list(map(seq_itos.get, seq.tolist())))
                out = hotknots(seq)
                if out is not None:
                    s.append(out[0])
                else:
                    s.append("."*len(seq))
            structs.append(s)
        print(f"Saving to {outfile}")
        torch.save(structs, outfile)
        
            