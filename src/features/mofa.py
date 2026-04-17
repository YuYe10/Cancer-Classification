from mofapy2.run.entry_point import entry_point

def run_mofa(rna, meth, factors=20):
    ent = entry_point()

    data = {
        "rna": rna,
        "meth": meth
    }

    ent.set_data_options(scale_views=True)
    ent.set_data_matrix(data)
    ent.set_model_options(factors=factors)

    ent.build()
    ent.run()

    Z = ent.model.nodes["Z"].getExpectation()
    return Z