import numpy as np
from sklearn.preprocessing import StandardScaler


class RNAPreprocessor:
    """Preprocesses RNA data with fit/transform pattern to prevent data leakage.
    
    Note: Feature selection should be fit on the training split/fold only.
    Scaling statistics are computed only on training data.
    """
    
    def __init__(self, top_k=2000, selected_features=None):
        self.top_k = top_k
        self.selected_features = selected_features  # Can be pre-computed
        self.scaler = StandardScaler()
        self.fitted_features = None
        self.fill_values = None
    
    @staticmethod
    def select_features_from_data(rna_data, top_k=2000):
        """Select top-k high-variance features from a training slice.
        
        Returns list of feature indices to keep.
        """
        rna_log = np.log2(rna_data + 1)
        rna_filtered = rna_log.loc[rna_log.mean(axis=1) > 1]
        var = rna_filtered.var(axis=1).sort_values(ascending=False)
        return var.index[:top_k].tolist()
    
    def fit(self, rna):
        """Fit scaler on training data (using pre-selected features)."""
        if self.selected_features is None:
            raise ValueError("selected_features must be set before fit()")
        
        rna_log = np.log2(rna + 1)
        # Keep only intersection with selected features (in case training set has fewer features)
        features_to_use = [f for f in self.selected_features if f in rna_log.index]
        if not features_to_use:
            raise ValueError("No RNA features available after feature selection")

        rna_subset = rna_log.loc[features_to_use].T
        self.fitted_features = list(rna_subset.columns)
        self.fill_values = rna_subset.mean(axis=0)
        rna_subset = rna_subset.fillna(self.fill_values)
        self.scaler.fit(rna_subset[self.fitted_features])
        
        return self
    
    def transform(self, rna):
        """Transform data using fitted statistics."""
        if self.selected_features is None:
            raise ValueError("selected_features must be set before transform()")
        if self.fitted_features is None:
            raise ValueError("fit() must be called before transform()")

        rna_log = np.log2(rna + 1)
        rna_subset = rna_log.reindex(self.fitted_features).T
        rna_subset = rna_subset.fillna(self.fill_values)
        return self.scaler.transform(rna_subset[self.fitted_features])


class MethPreprocessor:
    """Preprocesses methylation data with fit/transform pattern to prevent data leakage.
    
    Note: Feature selection should be fit on the training split/fold only.
    Scaling statistics are computed only on training data.
    """
    
    def __init__(self, top_k=2000, selected_features=None):
        self.top_k = top_k
        self.selected_features = selected_features  # Can be pre-computed
        self.scaler = StandardScaler()
        self.fitted_features = None
        self.fill_values = None
    
    @staticmethod
    def select_features_from_data(meth_data, top_k=2000):
        """Select top-k high-variance features from a training slice.
        
        Returns list of feature indices to keep.
        """
        var = meth_data.var(axis=1, skipna=True).sort_values(ascending=False)
        return var.index[:top_k].tolist()
    
    def fit(self, meth):
        """Fit scaler on training data (using pre-selected features)."""
        if self.selected_features is None:
            raise ValueError("selected_features must be set before fit()")

        # Keep only intersection with selected features (in case training set has fewer features)
        features_to_use = [f for f in self.selected_features if f in meth.index]
        if not features_to_use:
            raise ValueError("No methylation features available after feature selection")

        meth_subset = meth.loc[features_to_use].T
        self.fitted_features = list(meth_subset.columns)
        self.fill_values = meth_subset.mean(axis=0)
        meth_subset = meth_subset.fillna(self.fill_values)
        self.scaler.fit(meth_subset[self.fitted_features])
        
        return self
    
    def transform(self, meth):
        """Transform data using fitted statistics."""
        if self.selected_features is None:
            raise ValueError("selected_features must be set before transform()")
        if self.fitted_features is None:
            raise ValueError("fit() must be called before transform()")

        meth_subset = meth.reindex(self.fitted_features).T
        meth_subset = meth_subset.fillna(self.fill_values)
        return self.scaler.transform(meth_subset[self.fitted_features])


# Legacy function wrappers for backward compatibility
def preprocess_rna(rna, top_k=2000):
    """Legacy wrapper: fit and transform on full data (use only for test/reference)."""
    features = RNAPreprocessor.select_features_from_data(rna, top_k)
    preprocessor = RNAPreprocessor(top_k=top_k, selected_features=features)
    return preprocessor.fit(rna).transform(rna)


def preprocess_meth(meth, top_k=2000):
    """Legacy wrapper: fit and transform on full data (use only for test/reference)."""
    features = MethPreprocessor.select_features_from_data(meth, top_k)
    preprocessor = MethPreprocessor(top_k=top_k, selected_features=features)
    return preprocessor.fit(meth).transform(meth)


def scale_data(df):
    """Legacy wrapper: fit and transform on full data (use only for test/reference)."""
    scaler = StandardScaler()
    return scaler.fit_transform(df.T)
