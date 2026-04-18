import pandas as pd

def load_data(config):
    rna = pd.read_csv(config['data']['rna_path'], index_col=0)
    meth = pd.read_csv(config['data']['meth_path'], index_col=0)

    clinical = pd.read_csv(config['data']['clinical_path'], sep='\t')
    clinical.columns = clinical.columns.str.strip()

    if 'PAM50' in clinical.columns and 'label' not in clinical.columns:
        clinical = clinical.rename(columns={'PAM50': 'label'})

    if 'sample' not in clinical.columns:
        raise KeyError("Clinical file must contain a 'sample' column")
    if 'label' not in clinical.columns:
        raise KeyError("Clinical file must contain a 'label' or 'PAM50' column")

    clinical['label'] = clinical['label'].replace({'Her2': 'HER2', 'her2': 'HER2'})
    clinical = clinical[clinical['label'].isin(['LumA', 'LumB', 'HER2', 'Basal'])]
    clinical = clinical.drop_duplicates(subset='sample', keep='first').set_index('sample')

    return rna, meth, clinical