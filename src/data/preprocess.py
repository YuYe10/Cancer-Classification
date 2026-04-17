import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_rna(rna, top_k=2000):
    rna = np.log2(rna + 1)
    rna = rna.loc[rna.mean(axis=1) > 1]

    var = rna.var(axis=1).sort_values(ascending=False)
    rna = rna.loc[var.index[:top_k]]

    return rna

def preprocess_meth(meth, top_k=2000):
    meth = meth.dropna()

    var = meth.var(axis=1).sort_values(ascending=False)
    meth = meth.loc[var.index[:top_k]]

    return meth

def scale_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df.T)