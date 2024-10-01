from RNAinformer.utils.eval_utils import db2mat,eval_hits,db2pairs, mat_from_pos, pairs2mat, solved_from_mat, f1, recall, specificity, precision, mcc, tp_from_matrices, tn_from_matrices, get_fp, get_fn, mat2pairs
import pickle
import torch
import sys
import collections
import tqdm
import numpy as np
import pandas as pd

def eval_antarna(path,gc=False,max_len=200):
    sys.stdout = open(path+f"test_results_dup.txt", "w")
    preds = pickle.load(open(path+"preds.plk","rb"))
    test_data = torch.load("data/syn_hk/syn_hk_test_antarna.pt")
    print(len(test_data))
    same = torch.load("data/syn_hk/syn_hk_test_same_antarna.pt")
    test_data = [i for i in test_data if i['length']<= max_len]
    test_data = [test_data[i] for i in range(len(test_data)) if i not in same]
    preds = [preds[i] for i in range(len(preds)) if i not in same]
    assert len(test_data)==len(preds)
    # l=list(zip([len(i[0]) for i in preds],[i['length'] for i in test_data]))
    # for i,(l1,l2) in enumerate(l):
    #     print(i,l1,l2)
    gc_tolerance = 0.01
    gc_scores = []
    gc_avg_scores = []
    pk_hits = []
    solved = np.zeros(len(test_data))
    gc_solved = np.zeros(len(test_data))
    metrics = collections.defaultdict(list)
    total = len(preds)
    for i in tqdm.tqdm(range(len(preds))):
        has_pk = test_data[i]['has_pk']
        length = test_data[i]['length']
        if gc:
            gc_content = test_data[i]['gc_content']
        try:
            assert len(preds[i][0]) == length
        except:
            print(i)
            print(len(preds[i][0]),length)
            raise
        #true_mat = mat_from_pos(test_data[i]['pos1id'], test_data[i]['pos2id'], length)
        true_mat = db2mat(preds[i][0])
        pairs = db2pairs(preds[i][0])
        pk_info = [i[2] for i in pairs]
        #assert np.equal(true_mat, true_mat2).all()
        pred_mat = db2mat(preds[i][2])
        pred_seq = preds[i][1]
        #pk_info = test_data[i]['pk']
        pred_metrics = eval_hits(torch.tensor(pred_mat), torch.tensor(true_mat), pk_info)
        metrics['num_pk_hits'].append(pred_metrics['num_pk_hits'])
        metrics['num_pk_gt'].append(pred_metrics['num_pk_gt'])
        str_acc, hits, gt = solved_from_mat(pred_mat, true_mat)
        tp = tp_from_matrices(pred_mat, true_mat)
        tn = tn_from_matrices(pred_mat, true_mat)
        fp = get_fp(pred_mat, tp)
        fn = get_fn(true_mat, tp)
        metrics['f1'].append(f1(tp, fp, tn, fn))
        metrics['recall'].append(recall(tp, fp, tn, fn))
        metrics['specificity'].append(specificity(tp, fp, tn, fn))
        metrics['precision'].append(precision(tp, fp, tn, fn))
        metrics['mcc'].append(mcc(tp, fp, tn, fn))
        metrics['num_hits'].append(hits)
        metrics['num_gt'].append(gt)
        pred_gc = (list(pred_seq).count('G') + list(pred_seq).count('C'))/length
        if gc:
            gc_score = abs(pred_gc - gc_content)
            gc_avg_scores.append(1-gc_score)
            metrics['gc_score'].append(1-gc_score)
        if str_acc == 1:
            solved[i]+=1
            if gc:
                if gc_score <= gc_tolerance:
                    gc_solved[i] += 1
                gc_scores.append(1-gc_score)
                metrics['gc_score_str'].append(1-gc_score)
    if gc:
        print("Number of structures solved:", np.sum(solved>0))
        solved = gc_solved
    print("Number of tasks solved:", np.sum(solved>0))
    print("Total no of tasks:", total)
    print("Solved score:", np.sum(solved>0)/total)
    print("Num of hits:", np.mean(metrics["num_hits"]))
    print("Num of gt:", np.mean(metrics["num_gt"]))
    print("Max F1:", np.mean(metrics["f1"]))
    print("Max MCC:", np.mean(metrics["mcc"]))
    if gc:
        print("GC-content error for structure solved:", 1-np.mean(gc_scores))
        print("GC-content error:", 1-np.mean(gc_avg_scores))
    print("pK hits:", np.mean(metrics["num_pk_hits"]))
    print("pK gt:", np.mean(metrics["num_pk_gt"]))
    metrics['p_pk_hits'] = np.mean([metrics["num_pk_hits"][i]/metrics["num_pk_gt"][i] for i in range(len(metrics["num_pk_hits"]))])
    print("pK hits %:", metrics['p_pk_hits'])
    metrics['solved'] = np.sum(solved>0)/total
    metrics['num_hits'] = np.mean(metrics["num_hits"])
    metrics['num_gt'] = np.mean(metrics["num_gt"])
    metrics['f1'] = np.mean(metrics["f1"])
    metrics['mcc'] = np.mean(metrics["mcc"])
    metrics['num_pk_hits'] = np.mean(metrics["num_pk_hits"])
    metrics['num_pk_gt'] = np.mean(metrics["num_pk_gt"])
    if gc:
        metrics['gc_score_str'] = 1-np.mean(gc_scores)
        metrics['gc_score'] = 1-np.mean(gc_avg_scores)
    df=pd.DataFrame(metrics)
    df.to_csv(path+"/metrics_dup.csv")
    sys.stdout.close()    
eval_antarna("final_runs/antarna/",False,200)
