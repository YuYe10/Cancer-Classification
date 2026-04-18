def align_samples(rna, meth, clinical):
    common = [sample for sample in rna.columns if sample in meth.columns and sample in clinical.index]

    rna = rna.loc[:, common]
    meth = meth.loc[:, common]
    clinical = clinical.loc[common]

    return rna, meth, clinical