import torch
import numpy as np

def get_struct(pred):
    pairs = list(set(tuple(sorted(pair)) for pair in np.argwhere(pred == 1)))
    pairs=np.array(pairs)
    seq=""
    for i in range(len(pred)):
        if i not in pairs:
            seq+="."
        else:
            if i in pairs[:,0]:
                seq+="("
            else:
                seq+=")"
    return seq

preds = torch.load("predictions/bprna_test_preds_v2.pt")

seq_vocab = ['A', 'C', 'G', 'U', 'N']
struct_vocab = ['.','(', ')']

seq_itos = dict(zip(range(1,len(seq_vocab)+1), seq_vocab))

struct_itos = dict(zip(range(len(struct_vocab)), struct_vocab))

count=0
pred_structs=[]
for i in preds:
    p=[get_struct(j) for j in i]
    pred_structs.append(p)

torch.save(pred_structs, "predictions/bprna_test_preds_structs_v2.pt")
        
"""
for i in range(len(preds)):
    pred=preds[i][0]
    pred_struct=get_struct(pred)
    test_struct=[struct_itos[int(j)] for j in test[i]['src_struct']]
    test_struct="".join(test_struct)
    acc.append(np.sum([1 if pred_struct[j]==test_struct[j] else 0 for j in range(len(pred_struct))])/len(pred_struct))
    if acc[-1] == 1.0:
        count+=1
print("Accuracy: ", np.mean(acc))
print("Perfect predictions: ", count)"""
