from __future__ import annotations

import numpy as np
from mofapy2.run.entry_point import entry_point


def fit_mofa(rna, meth, factors=20, seed=None):
    ent = entry_point()

    data = [[rna], [meth]]

    ent.set_data_options(scale_views=False)
    ent.set_data_matrix(data, views_names=["rna", "meth"], groups_names=["all_samples"])
    ent.set_model_options(factors=factors)
    train_options = {"quiet": True}
    if seed is not None:
        train_options["seed"] = seed
    ent.set_train_options(**train_options)

    ent.build()
    ent.run()
    return ent


def project_mofa_latent(model, rna, meth, ridge=1e-6):
    weights = np.asarray(model.nodes["W"].getExpectation())
    views = [rna, meth]
    if len(views) != weights.shape[0]:
        raise ValueError("Number of views does not match the fitted MOFA model")

    projected_factors = []
    for view_idx, view_matrix in enumerate(views):
        view_array = np.asarray(view_matrix)
        if view_array.ndim != 2:
            raise ValueError("Each MOFA view must be a 2D matrix")

        view_weights = np.asarray(weights[view_idx])
        view_weights = np.nan_to_num(view_weights, copy=False)
        view_array = np.nan_to_num(view_array, copy=False)

        gram = view_weights.T @ view_weights
        regularized = gram + ridge * np.eye(gram.shape[0], dtype=gram.dtype)
        factor_scores = np.linalg.solve(regularized, view_weights.T @ view_array.T).T
        projected_factors.append(factor_scores)

    if not projected_factors:
        raise ValueError("No MOFA views were provided for projection")

    if len(projected_factors) == 1:
        return projected_factors[0]

    stacked = np.stack(projected_factors, axis=0)
    return np.nanmean(stacked, axis=0)


def run_mofa(rna, meth, factors=20, seed=None):
    ent = fit_mofa(rna, meth, factors=factors, seed=seed)
    return np.asarray(ent.model.nodes["Z"].getExpectation())