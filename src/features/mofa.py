from mofapy2.run.entry_point import entry_point

def run_mofa(rna, meth, factors=20):
    ent = entry_point()

    # MOFA expects data as a nested list with shape: views x groups.
    # Here we have two views (rna, meth) and one group (all samples).
    data = [[rna], [meth]]

    ent.set_data_options(scale_views=True)
    ent.set_data_matrix(data, views_names=["rna", "meth"], groups_names=["all_samples"])
    ent.set_model_options(factors=factors)
    ent.set_train_options()

    ent.build()
    ent.run()

    Z = ent.model.nodes["Z"].getExpectation()
    return Z