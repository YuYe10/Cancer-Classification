import pandas as pd

def load_data(config):
    rna = pd.read_csv(config['data']['rna_path'], index_col=0)
    meth = pd.read_csv(config['data']['meth_path'], index_col=0)
    clinical = pd.read_csv(config['data']['clinical_path'])
    return rna, meth, clinical