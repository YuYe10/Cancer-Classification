import numpy as np

from src.data.loader import load_data
from src.data.align import align_samples
from src.data.preprocess import preprocess_rna, preprocess_meth, scale_data
from src.features.mofa import run_mofa
from src.models.train import train_svm
from src.models.evaluate import evaluate

def run_pipeline(config):
    rna, meth, clinical = load_data(config)
    rna, meth, clinical = align_samples(rna, meth, clinical)

    rna = preprocess_rna(rna, config['preprocess']['rna_top_var'])
    meth = preprocess_meth(meth, config['preprocess']['meth_top_var'])

    rna = scale_data(rna)
    meth = scale_data(meth)

    y = clinical.set_index("sample")["label"]

    if config['exp'] == "rna":
        X = rna
    elif config['exp'] == "concat":
        X = np.concatenate([rna, meth], axis=1)
    elif config['exp'] == "mofa":
        X = run_mofa(rna, meth, config['mofa']['factors'])
    else:
        raise ValueError("Unknown exp")

    model, X_test, y_test = train_svm(X, y, config)
    acc, report = evaluate(model, X_test, y_test)

    return acc, report