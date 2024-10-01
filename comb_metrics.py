import glob
import pandas as pd

def comb_metrics(path,n,gc,metrics):
    if n is not None:
        g_path = path + f"*/predictions_{n}_3/"
    else:
        g_path = path + f"*/"
    save_path = path + f"metrics_dup.csv"
    if gc:
        g_path = g_path + "gc/"
        save_path = path + f"metrics_dup_gc.csv"
    files = glob.glob(g_path + f"{metrics}.csv")
    cdfs = []
    for i in files:
        version = i.split("/")[2]
        df = pd.read_csv(i)
        df["Version"] = version
        print(i)
        cdfs.append(df)
    cdf = pd.concat(cdfs)
    cdf.fillna(0,inplace=True)
    for test_set in cdf['Test_set'].unique():
        m = cdf[cdf['Test_set'] == test_set].mean(numeric_only=True)
        s = cdf[cdf['Test_set'] == test_set].std(numeric_only=True)
        cdf.loc[f"{test_set} Mean"] = round(m,3)
        cdf.loc[f"{test_set} Std"] = round(s,3)
        cdf.loc[f"{test_set} Mean","Test_set"] = test_set
        cdf.loc[f"{test_set} Std","Test_set"] = test_set
    cdf.to_csv(save_path)
comb_metrics("final_runs/multi_uni/",None,True,"metrics_dup")
