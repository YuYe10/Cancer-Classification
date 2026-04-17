def align_samples(rna, meth, clinical):
    common = list(set(rna.columns) & set(meth.columns) & set(clinical['sample']))

    rna = rna[common]
    meth = meth[common]
    clinical = clinical[clinical['sample'].isin(common)]

    return rna, meth, clinical