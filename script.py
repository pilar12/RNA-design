from collections import defaultdict
import pandas as pd
import torch
import numpy as np
#from RNAinformer.utils.eval_utils import db2pairs
#from RNAinformer.utils.fold import fold_pk
from tqdm import tqdm
from glob import glob
import pickle
seq_vocab = ['A', 'C', 'G', 'U', 'N']
struct_vocab = ['.','(', ')','[',']','{','}','<','>']
r_struct_vocab = ['.','(',')','N']
seq_itos = dict(zip(range(1,len(seq_vocab)+1), seq_vocab))
struct_itos = dict(zip(range(len(struct_vocab)), struct_vocab))
r_struct_itos = dict(zip(range(len(r_struct_vocab)), r_struct_vocab))
wc = ['AU', 'UA', 'GC', 'CG']
wobble = ['GU', 'UG']
canonical = wc + wobble
def mat2pairs(matrix, symmetric=True):
    """
    Convert matrix representation of structure to list of pairs.
    """
    if symmetric:
        return list(set(tuple(sorted(pair)) for pair in np.argwhere(matrix == 1)))
    else:
        return list(tuple(pair) for pair in np.argwhere(matrix == 1))

data = torch.load("final_runs/syn_pdb/multiplets.pt")
print(len(data))
p
files=glob("final_runs/syn_pdb/*/*100_2/pdb*seq.pt")
df={'version':[],'id':[],'seq_id':[],'seq':[]}
for f in files:
    print(f)
    preds=torch.load(f)
    tset=f.split("/")[-1].split("seq.pt")[0]
    df['version'].extend([f.split("/")[2]]*len(preds))
    df['id'].extend([f'{tset}{i[0]}' for i in preds])
    df['seq_id'].extend([i[1] for i in preds])
    df['seq'].extend([list(map(seq_itos.get,i[-1].tolist())) for i in preds])
df=pd.DataFrame(df,index=None)
df.to_pickle("final_runs/syn_pdb/pdb_seqs.plk")
p
files=glob("final_runs/multi_uni/*/gc/pdb_*_preds.pt")
for f in files:
    preds=torch.load(f)
    f_new = f.replace("preds.pt","test_preds.pt")
    torch.save(preds,f_new)
files=glob("final_runs/multi_uni/*/pdb_*_preds_structs.pt")
for f in files:
    preds=torch.load(f)
    f_new = f.replace("preds_structs.pt","test_preds_structs.pt")
    torch.save(preds,f_new)
files=glob("final_runs/multi_uni/*/gc/pdb_*_preds_structs.pt")
for f in files:
    preds=torch.load(f)
    f_new = f.replace("preds_structs.pt","test_preds_structs.pt")
    torch.save(preds,f_new)
p
df=pd.read_pickle("data/syn_pdb/syn_pdb.plk")
df=df[(df['set']=='train') | (df['set']=='valid')]
tset=['pdb_ts1_test','pdb_ts2_test','pdb_ts3_test','pdb_ts_hard_test']
l=0
for t in tset:
    data=torch.load(f"data/pdb/{t}.pt")
    same=defaultdict(list)
    for j in tqdm(range(len(data))):
        for i, sample in df[df['length']==data[j]['length'].item()].iterrows():
            if sample['pos1id'] == data[j]['pos1id'].tolist() and sample['pos2id'] == data[j]['pos2id'].tolist():
                same[j].append((i,sample['Id'],sample['set']))
    print(len(same))
    r=0
    if len(same)>0:
        for i in same:
            r+=len(same[i])
    print(t,r)
    l+=r
    torch.save(same,f"data/syn_pdb/syn_{t}_same.pt")
print(l)
p

vset=[0,1,2]
tset=['pdb_ts1_test','pdb_ts2_test','pdb_ts3_test','pdb_ts_hard_test']
for v in vset:
    for t in tset:
        print(v,t)
        preds =torch.load(f"runs/syn_pdb_gc/version_{v}/predictions_100/gc/{t}_preds.pt")
        df0 = [i for i in preds if len(i[0])>0 and len(i[0])<=50]
        df1 = [i for i in preds if len(i[0])>50 and len(i[0])<=60]
        df2 = [i for i in preds if len(i[0])>60 and len(i[0])<=75]
        df3 = [i for i in preds if len(i[0])>75 and len(i[0])<=100]
        new_preds = df0 + df1 + df2 + df3
        torch.save(new_preds,f"runs/syn_pdb_gc/version_{v}/predictions_100/gc/{t}_preds.pt")
p
df=torch.load("data/pdb/pdb_ts_hard_test.pt")
print(len(df))
p
df=[i for i in df if i['length']<=100]
df0 = [i for i in df if i['length']>0 and i["length"]<=50]
df1 = [i for i in df if i['length']>50 and i["length"]<=60]
df2 = [i for i in df if i['length']>60 and i["length"]<=75]
df3 = [i for i in df if i['length']>75 and i["length"]<=100]
new_df = df0 + df1 + df2 + df3
print(len(new_df))
print(len(df))
preds=torch.load("runs/syn_pdb/version_0/predictions_100/pdb_ts2_test_preds_structs.pt")
print(len(preds))
torch.save(new_df, "data/syn_pdb/pdb_ts2_test.pt")
p
# df['gc'] = df['gc'].values/df['length'].values
# df['sequence']=df['sequence'].apply(lambda x: list(x))
# df.to_pickle("data/syn_multi/syn_rnaformer.plk")
# p
nc=[]
for i, sample in tqdm(df.iterrows()):
    seq=sample['sequence']
    p1=sample['pos1id']
    p2=sample['pos2id']
    flag = False
    for j in range(len(p1)):
        if seq[p1[j]] + seq[p2[j]] not in canonical:
            flag = True
            break
    nc.append(flag)
df['has_nc']=nc
df.to_pickle("data/syn_multi/syn_rnaformer2.plk")
l
df['gc'] = df['gc'].values/df['length'].values
df['sequence']=df['sequence'].apply(lambda x: list(x))
print(df.columns)
df.to_pickle("data/syn_pdb/syn_pdb.plk")
p
# preds=pickle.load(open("final_runs/antarna/gc/preds.plk","rb"))
# test_data = torch.load("data/syn_hk/syn_hk_test_antarna.pt")
# print(len(test_data))
# same = torch.load("data/syn_hk/syn_hk_test_same_antarna.pt")
# print(same)
# print(preds[18][0])
# print("".join(list(map(struct_itos.get,test_data[20]['src_struct'].tolist()))))
# p
# test_data = pd.read_pickle("data/syn_ns/syn_ns_test.plk")
# test_data = test_data[test_data['length'] <= 200]
# test_data = test_data[test_data['pos1id'].apply(len) >= 1]
df=pd.read_pickle("data/syn_hk/syn_hk_test.plk")
df=df[df['has_pk']==1]
test_data=df.sort_values(by='length')
test_data = test_data.sort_values(by="length")
test_data_pt = torch.load("data/syn_hk/syn_hk_test.pt")
test_data_pt = [i for i in test_data_pt if i["length"] <= 200]
seqs = [list(map(seq_itos.get,i['trg_seq'][1:].tolist())) for i in test_data_pt]
gc = [i['gc_content'].item() for i in test_data_pt]
m = defaultdict(list)
for i in tqdm(range(len(test_data))):
    for j in range(len(test_data_pt)):
        if seqs[j] == test_data.iloc[i]['sequence']:
            m[i].append(j)
            #print(i,j)
print(len(m))
for k in m:
    if len(m[k])!=1:
        print(k,m[k])
# m =torch.load("data/syn_ns/syn_ns_test_learna.pt")
# test_data_pt = torch.load("data/syn_ns/syn_ns_test.pt")
# test_data_pt = [i for i in test_data_pt if i["length"] <= 200]
test_same = torch.load("data/syn_hk/syn_hk_test_same2.pt")
same=[]
mr = {v[0]:k for k,v in m.items()}
for i in test_same:
    if i in mr:
        print(i)
        print(mr[i])
        same.append(mr[i])
torch.save(m, "data/syn_hk/m.pt")
test_data = [test_data_pt[m[i][0]] for i in range(len(test_data))]
torch.save(test_data, "data/syn_hk/syn_hk_test_antarna.pt")
torch.save(same, "data/syn_hk/syn_hk_test_same_antarna.pt")
p
# df=pd.read_pickle("data/syn_hk/syn_hk_test.plk")
# df=df[df['has_pk']==1]
# df=df.sort_values(by='length')
# structs = ["".join(i) for i in df['structure'].values]
# gc_content = df['gc'].values
# ids = df['Id'].values
# d=list(zip(structs,gc_content,ids))
#d=sorted(d,key=lambda x: len(x[0]))
# pickle.dump(d,open("data/syn_hk/syn_hk_test_structs_gc.plk","wb"),protocol=2)

# s=0
# for i in range(56):
#     data = torch.load(f"runs/syn_hk/version_1/predictions_20/syn_hk_test_preds_structs_{i}.pt")
#     s+=len(data)
#     if i!=55 and len(data)!=50:
#         print(i,len(data))
#     if i==55 and len(data)!=28:
#         print(i)
# print(2778-s)


same = torch.load("data/syn_hk/syn_hk_test_same.pt")
dup = torch.load("data/syn_hk/syn_hk_test_dup.pt")
for i in dup:
    for j in dup[i]:
        if j not in same:
            same[j].append((i,'test','test'))
print(len(same))
torch.save(same, "data/syn_hk/syn_hk_test_same2.pt")
print(same)
p
# data = torch.load("data/syn_multi/syn_multi_test.pt")
# same=defaultdict(list)
# s=[]
# for i in range(len(data)):
#     if i in s:
#         continue
#     for j in range(i+1,len(data)):
#         if data[i]['length'].item() == data[j]['length'].item() and torch.equal(data[i]['pos1id'],data[j]['pos1id']) and torch.equal(data[i]['pos2id'],data[j]['pos2id']):
#             same[i].append(j)
#             s.append(j)

# print(len(same))
# torch.save(same, "data/syn_multi/syn_multi_test_dup.pt")
# t=0
# for i in same:
#     print(len(same[i]))
#     t+=len(same[i])
# print(t)
# p
# data = torch.load("data/riboswitch/ribo_design_all_len2100_designTrue_seed1_v3.pth")['train']
# struct = "".join(list(map(r_struct_itos.get,data[100]['src_struct'].tolist())))
# print(struct)
# p
# data = torch.load("data/syn_multi/syn_multi_test.pt")
# data=data[0]
# struct = torch.LongTensor(data['src_struct'].size(0), data['src_struct'].size(0)).fill_(2)
# for i in range(2,-1,-1):
#     if i == 0:
#         id = 0
#     else:
#         id = 1
#     struct[data['src_struct']==i,:] = id
#     struct[:,data['src_struct']==i] = id
# struct.fill_diagonal_(0)
# print(data['pos1id'].dtype)
# struct2 = torch.LongTensor(data['src_struct'].size(0), data['src_struct'].size(0)).fill_(0)
# struct2[data['pos1id'],data['pos2id']] = 1
# struct2[data['pos2id'],data['pos1id']] = 1
# print(struct2[0,:])
# print(struct)
# print(struct2)
# print(torch.eq(struct,struct2))
# print(torch.equal(struct,struct2))
# p
df=pd.read_pickle("data/syn_hk/syn_hk.plk")
df=df[(df['set']=='train') | (df['set']=='valid')]
data = torch.load("data/syn_hk/syn_hk_test.pt")
same=defaultdict(list)
pos1 = []
pos2 = []
for j in tqdm(range(len(data))):
    for i, sample in df[df['length']==data[j]['length'].item()].iterrows():
        if sample['pos1id'] == data[j]['pos1id'].tolist() and sample['pos2id'] == data[j]['pos2id'].tolist():
            same[j].append((i,sample['Id'],sample['set']))
print(len(same))
torch.save(same,"data/syn_hk/syn_hk_test_same.pt")
p
df = pd.read_pickle("data/syn_hk/syn_hk.plk")
df.loc[df['set']=='test_valid_test','set'] = 't_v_t'
df['gc'] = df['gc'].values/df['length'].values
p1 = []
p2 = []
pk = []
for i, sample in tqdm(df.iterrows()):
    struct = sample['structure']
    pairs = np.array(db2pairs(struct))
    if len(pairs) == 0:
        p1.append([])
        p2.append([])
        pk.append([])
        continue
    p1.append(pairs[:,0].tolist())
    p2.append(pairs[:,1].tolist())
    pk.append(pairs[:,2].tolist())        
df['pos1id']=p1
df['pos2id']=p2
df['pk']=pk
data = [0 for i in range(len(df))]
df['is_pdb'] = data
df['has_pk'] = [1 if sum(i)>0 else 0 for i in pk]
df['has_multiplet'] = data
df['has_nc'] = data
df['sequence']=df['sequence'].apply(lambda x: list(x))
df['structure']=df['structure'].apply(lambda x: list(x))
df.to_pickle("data/syn_hk/syn_hk.plk")
p
df=pd.read_pickle("data/syn_ns/syn_rnafold.plk")
df=df[(df['set']=='train') | (df['set']=='valid')]
data = torch.load("data/syn_ns/syn_ns_test.pt")
same=defaultdict(list)
train_i = []
seqs = []
structs = []
for i in range(len(data)):
    seqs.append(list(map(seq_itos.get, data[i]['trg_seq'][1:].tolist())))
    structs.append(list(map(struct_itos.get, data[i]['src_struct'].tolist())))
# print("here")
# print(df[df.isin({'sequence':seqs, 'structure':structs}).all(axis=1)]['set'].value_counts())
# p
for j in tqdm(range(len(data))):
    for i, sample in df[df['length']==data[j]['length'].item()].iterrows():
        seq = seqs[j]
        struct = structs[j]
        if sample['sequence'] == seq and sample['structure'] == struct:
            same[j].append((i,sample['Id'],sample['set']))
print(len(same))
torch.save(same, "data/syn_ns/syn_ns_test_same.pt")

df=pd.read_pickle("data/syn_multi/syn_rnaformer.plk")
df=df[(df['set']=='train') | (df['set']=='valid')]
data = torch.load("data/syn_multi/syn_multi_test.pt")
same=defaultdict(list)
pos1 = []
pos2 = []
for j in tqdm(range(len(data))):
    for i, sample in df[df['length']==data[j]['length'].item()].iterrows():
        if sample['pos1id'] == data[j]['pos1id'].tolist() and sample['pos2id'] == data[j]['pos2id'].tolist():
            same[j].append((i,sample['Id'],sample['set']))
print(len(same))
torch.save(same,"data/syn_multi/syn_multi_test_same.pt")
df=pd.read_pickle("data/syn_ipk/syn_ipk.plk")
df=df[(df['set']=='train') | (df['set']=='valid')]
data = torch.load("data/syn_ipk/syn_ipk_test.pt")
same=defaultdict(list)
pos1 = []
pos2 = []
for j in tqdm(range(len(data))):
    for i, sample in df[df['length']==data[j]['length'].item()].iterrows():
        if sample['pos1id'] == data[j]['pos1id'].tolist() and sample['pos2id'] == data[j]['pos2id'].tolist():
            same[j].append((i,sample['Id'],sample['set']))
print(len(same))
torch.save(same,"data/syn_ipk/syn_ipk_test_same.pt")
p
df=pd.read_pickle("data/syn_multi/syn_rnaformer.plk")
train = df[df['set']=='train']
test = df[df['set']=='test']
pos1=[]
pos2=[]
for i,sample in train.iterrows():
    pos1.append(sample['pos1id'])
    pos2.append(sample['pos2id'])
same = 0
for i, sample in tqdm(test.iterrows()):
    if sample['pos1id'] in pos1:
        s=0
        offset = -1
        while s==0:
            try:
                offset = pos1.index(sample['pos1id'], offset+1)
                if sample['pos2id'] == pos2[offset]:
                    print(sample['pos1id'], sample['pos2id'], pos1[offset], pos2[offset])
                    same += 1
            except:
                s=1
print(same)
p
df=pd.read_pickle("data/syn_ns/syn_rnafold.plk")
train = df[df['set']=='train']
test = df[df['set']=='test']
train_struct = ["".join(i) for i in train['structure'].values]
same = 0
for i, sample in tqdm(test.iterrows()):
    struct = "".join(sample['structure'])
    if struct in train_struct:
        s=0
        offset = -1
        while s==0:
            try:
                offset = train_struct.index(struct, offset+1)
                same += 1
            except:
                s=1
p
df.rename(columns={"gc_content":"gc"}, inplace=True)
df['gc'] = df['gc'].values/df['length'].values
print(df['gc'].head())
df.to_pickle("data/syn_ns/syn_rnafold.plk")
o
df['sequence']=df['sequence'].apply(lambda x: list(x))
df['structure']=df['structure'].apply(lambda x: list(x))
df.to_pickle("data/syn_ns/syn_rnafold.plk")
p
p1 = []
p2 = []
pk = []
for i, sample in tqdm(df.iterrows()):
    struct = sample['structure']
    pairs = np.array(db2pairs(struct))
    if len(pairs) == 0:
        p1.append([])
        p2.append([])
        pk.append([])
        continue
    p1.append(pairs[:,0].tolist())
    p2.append(pairs[:,1].tolist())
    pk.append(pairs[:,2].tolist())
        
df['pos1id']=p1
df['pos2id']=p2
df['pk']=pk
data = [0 for i in range(len(df))]
df['is_pdb'] = data
df['has_pk'] = data
df['has_multiplet'] = data
df['has_nc'] = data
df.to_pickle("data/syn_ns/syn_rfam_data_clust_300_split_ss80_rnafold.plk")
r

data = torch.load("data/pdb/pdb_data_len2100_designTrue_canonicalFalse_oversampling1_seed1_v3.pth")
pos1=[]
pos2=[]
for i in data['train']:
    pos1.append(i['pos1id'].tolist())
    pos2.append(i['pos2id'].tolist())
same=[]
pos1_same=0
pos2_same=0
test_data = torch.load("data/bprna/bprna_sort_test.pt")
n = [1 for i in test_data if i['length']<= 100]
print(len(n))
print(len(test_data))
pk = 0
for j,i in enumerate(test_data):
    if i['pos1id'].tolist() in pos1:
        ind = pos1.index(i['pos1id'].tolist())
        if i['pos2id'].tolist() == pos2[ind] and i["length"] == data['train'][ind]['length']:
            same.append(j)
            if i['has_pk']:
                pk += 1
print(len(same))
print(len(np.unique(same)))
print(pk)
torch.save(same, "data/bprna/bprna_pdb_test_same.pt")
p
collator = CollatorRNADesignMat(0,-1000)
d=collator(data[:5])
struct = d['src_struct']
batch_size = struct.shape[0]
N_seq = struct.shape[1]
N_res = struct.shape[2]
num_heads = 8
dim = 256
key_dim = dim // num_heads
l =torch.nn.Linear(dim, 3 * dim, bias=True)

pair_act = torch.randn(batch_size, N_seq, N_res, dim)

e_0 = pair_act[struct == 0][0].view(1, 1, 1, -1)
e_1 = pair_act[struct == 1][0].view(1, 1, 1, -1)

e = torch.cat([e_0, e_1], dim=2)
q, k, v = l(e).split(dim, dim=3)
_, _, V = l(pair_act).split(dim, dim=3)
V = V.view(batch_size, N_seq, N_res, num_heads, key_dim).permute(0, 1, 3, 2, 4)

q = q.view(1, 1, 2, num_heads, key_dim).permute(0, 1, 3, 2, 4)
k = k.view(1, 1, 2, num_heads, key_dim).permute(0, 1, 3, 4, 2)
v = v.view(1, 1, 2, num_heads, key_dim).permute(0, 1, 3, 2, 4)

a = q @ k / (key_dim ** 0.5)
attn = torch.zeros(batch_size, N_seq, num_heads, 2, N_res)
attn = attn.permute(3,2,0,1,4)
a = a.unsqueeze(-1)
attn[0,:, struct == 0] = a[:,:,:,0,0]
attn[0,:, struct == 1] = a[:,:,:,0,1]
attn[1,:, struct == 0] = a[:,:,:,1,0]
attn[1,:, struct == 1] = a[:,:,:,1,1]
attn = attn.permute(2,3,1,0,4).softmax(dim=-1)
print(attn.shape)
print(V.shape)
out = attn @ V
print(out.shape)
out = out.permute(3,0,1,2,4).reshape(2,batch_size,N_seq,1,dim).repeat(1,1,1,N_res,1)
pair_act[struct == 0] = out[0][struct == 0]
pair_act[struct == 1] = out[1][struct == 1]

p
q = torch.randn(5,28,28,256).view(5,28,28,8,32)
n0 = torch.randn(1,1,1,256).view(1,1,1,8,32)
n1 = torch.randn(1,1,1,256).view(1,1,1,8,32)
r = torch.randn(5,28,28,256)
q[struct == 0] = n0
q[struct == 1] = n1

n0=n0.permute(0,1,3,2,4)
n1=n1.permute(0,1,3,2,4)
n = torch.cat([n0,n1],dim=3)
nn = n@n.permute(0,1,2,4,3)/16
print(nn[0][0][0])
a = torch.randn(5,28,8,2,28)
a00 = n0@n0.permute(0,1,2,4,3)/16
a01 = n0@n1.permute(0,1,2,4,3)/16
a10 = n1@n0.permute(0,1,2,4,3)/16
a11 = n1@n1.permute(0,1,2,4,3)/16
print(a00[0][0][0], a01[0][0][0], a10[0][0][0], a11[0][0][0])
a = a.permute(3,2,0,1,4)
a[0,:, struct == 0] = a00.squeeze(-1)
a[0,:, struct == 1] = a01.squeeze(-1)
a[1,:, struct == 0] = a10.squeeze(-1)
a[1,:, struct == 1] = a11.squeeze(-1)
a = a.permute(2,3,1,0,4).softmax(dim=-1)
out = a@q.permute(0,1,3,2,4)
print(out.shape)
out = out.permute(3,0,1,2,4).reshape(2,5,28,1,256)
print(out.shape)
out0 = out[0].repeat(1,1,28,1)
out1 = out[1].repeat(1,1,28,1)
print(out0.shape)
q = q.reshape(5,28,28,256)
q[struct == 0] = out0[struct == 0]
q[struct == 1] = out1[struct == 1]

q1= torch.randn(5,28,28,256)
q1[struct == 0] = n0
q1[struct == 1] = n1
a = q1@q1.permute(0,1,3,2)/16
a = a.softmax(dim=-1)
out = a@q1

print(q[0][0][0][:10])
print(q1[0][0][0][:10])
b
print(out.shape)
print(a.shape)
print(a00)
print(a01)
print(a10)
print(a11)
print("****")
print(q.shape)
attn0 = n0@q.permute(0,1,3,2)
attn1 = n1@q.permute(0,1,3,2)

attn0 = attn0.softmax(dim=-1)
attn1 = attn1.softmax(dim=-1)
q.permute(2,0,1,3)
val0 = attn0@q
val1 = attn1@q
print(val0.shape)
print(val1.shape)
print(val0.repeat(1,1,28,1).shape)
# val0 = attn0.unsqueeze(-1)*q
# print(val0.shape)
attnr = q@r.permute(0,1,3,2)
print(attn0.shape)
print(attn1.shape)
print(attnr.shape)

s="............((((((..[[[[[[[[[[[[.....[[[[[[[[[[.(((....(((((((.((((.....)))).)...((((((((....))))))))))))))......)))[[.[[[[[....))))))..]]]]].]]....((((((((..]]]]]]]]]].......))))))))]]]]]]]]]]]]."
pairs = db2pairs(s)
seq=['.' for i in range(len(s))]
for p1,p2,pk in pairs:
    if pk == 0:
        seq[p1]='('
        seq[p2]=')'
    if pk == 1:
        seq[p1]='['
        seq[p2]=']'
    if pk == 2:
        seq[p1]='{'
        seq[p2]='}'
    if pk == 3:
        seq[p1]='<'
        seq[p2]='>'
print("".join(seq))
print("".join(seq)==s)