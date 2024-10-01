import torch
import sys
from RNAinformer.utils.test_pk_mult import test_fold_pk_mult
from RNAinformer.utils.fold import test_fold
from glob import glob
import argparse
from collections import defaultdict
import pandas as pd
import pickle
from RNAinformer.utils.eval_ribo import eval_ribo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--ds', type=str)
    parser.add_argument('--gc', type=bool, default=False)
    parser.add_argument('--max_len', type=int, default=100)

    args = parser.parse_args()
    sys.stdout = open(args.path+f"test_results.txt", "w")
    metrics = defaultdict(list)
    if args.ds == 'rfam' or args.ds == 'syn_ns':
        for i in glob(f"{args.path}/*preds.pt"):
            preds=torch.load(i, map_location=torch.device('cpu'))
            ds = i.split("/")[-1].replace("_preds.pt","")
            test_data = torch.load(f'data/{args.ds}/{ds}.pt')
            if args.ds == 'syn_ns':
                same = torch.load('data/syn_ns/syn_ns_test_same2.pt')
                test_data = [test_data[i] for i in range(len(test_data)) if i not in same]
                preds = [preds[i] for i in range(len(preds)) if i not in same]
            test_data = [i for i in test_data if i['length']<= args.max_len]
            preds = [i for i in preds if len(i[0])<= args.max_len]
            print("*"*20)
            print(f"Testing on Dataset: {ds}")
            metrics["Test_set"].append(ds)
            ds_metrics = test_fold(preds, test_data, args.max_len, args.gc)
            for k in ds_metrics.keys():
                metrics[k].append(ds_metrics[k])
            print("*"*20)
    elif args.ds == 'learna' or args.ds == 'samfeo':
        ds = "syn_ns"
        test_data = torch.load(f'data/{ds}/{ds}_test_learna2.pt')
        same = torch.load(f'data/{ds}/{ds}_test_same2.pt')
        test_data = [test_data[i] for i in range(len(test_data)) if i not in same]
        test_data = [i for i in test_data if i['length']<= args.max_len]
        preds=pickle.load(open(f"{args.path}/preds.plk","rb"))
        preds = [preds[i] for i in range(len(preds)) if i not in same]
        if args.ds == 'learna':
            preds = [i['seqs'] for i in preds]
        preds = [i for i in preds if len(i[0])<= args.max_len]
        print("*"*20)
        print(f"Testing on Dataset: {ds}")
        metrics["Test_set"].append(ds)
        ds_metrics = test_fold(preds, test_data, args.max_len, args.gc)
        for k in ds_metrics.keys():
                metrics[k].append(ds_metrics[k])
        print("*"*20)
    elif args.ds == 'riboswitch':
        if args.gc:
            preds = torch.load(args.path+"/ribo_outputs_gc.pt")
            gc_targets = torch.load(args.path+"/gc_targets.pt")
            for gc in gc_targets:
                print(f"Target GC content: {gc}")
                gc_metrics = eval_ribo(preds[gc], args.path, gc)
                metrics['gc'].append(gc)
                for k in gc_metrics.keys():
                    metrics[k].append(gc_metrics[k])
                print("*****************************")
            df=pd.DataFrame(metrics)
            df.to_csv(args.path+"/metrics.csv")
        else:
            preds = torch.load(args.path+"/ribo_outputs.pt")
            metrics=eval_ribo(preds, args.path)
            df=pd.DataFrame(metrics,index=[0])
            df.to_csv(args.path+"/metrics.csv")
            print("*****************************")
    else:
        for i in glob(f"{args.path}/*structs.pt"):
            preds=torch.load(i, map_location=torch.device('cpu'))
            seqs=torch.load(i.replace("_structs",""))
            ds = i.split("/")[-1].replace("_preds_structs.pt","")
            test_data = torch.load(f'data/{args.ds}/{ds}.pt')
            test_data = [i for i in test_data if i['length']<= args.max_len]
            preds = [preds[i] for i in range(len(preds)) if len(seqs[i][0])<= args.max_len]
            seqs = [i for i in seqs if len(i[0])<= args.max_len]
            print("*"*20)
            print(f"Testing on Dataset: {ds}")
            metrics["Test_set"].append(ds)
            if ds == 'syn_hk_test' or ds=='syn_multi_test':
                same = torch.load(f'data/{args.ds}/{ds}_same2.pt')                
                test_data = [test_data[i] for i in range(len(test_data)) if i not in same]
                preds = [preds[i] for i in range(len(preds)) if i not in same]
                seqs = [seqs[i] for i in range(len(seqs)) if i not in same]
                ds_metrics = test_fold_pk_mult(preds, seqs, test_data, ds, args.gc)
            elif ds in ['pdb_ts1_test', 'pdb_ts2_test','pdb_ts3_test','pdb_ts_hard_test']:
                ds_metrics = test_fold_pk_mult(preds, seqs, test_data, ds, args.gc)
            for k in ds_metrics.keys():
                metrics[k].append(ds_metrics[k])
            print("*"*20)
    df=pd.DataFrame(metrics)
    df.to_csv(args.path+"/metrics_dup.csv")
    sys.stdout.close()