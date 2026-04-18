import numpy as np
from sklearn.preprocessing import StandardScaler


class RNAPreprocessor:
    """Preprocesses RNA data with fit/transform pattern to prevent data leakage.
    
    Note: Feature selection is done on full dataset to ensure consistency,
    but scaling statistics are computed only on training data.
    """
    
    def __init__(self, top_k=2000, selected_features=None):
        self.top_k = top_k
        self.selected_features = selected_features  # Can be pre-computed
        self.scaler = StandardScaler()
    
    @staticmethod
    def select_features_from_full_data(rna_full, top_k=2000):
        """Select top-k high-variance features from full data.
        
        Returns list of feature indices to keep.
        """
        rna_log = np.log2(rna_full + 1)
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
        rna_subset = rna_log.loc[features_to_use]
        self.scaler.fit(rna_subset.T)
        
        return self
    
    def transform(self, rna):
        """Transform data using fitted statistics."""
        if self.selected_features is None:
            raise ValueError("selected_features must be set before transform()")

        rna_log = np.log2(rna + 1)
        # Keep only features that are both in selected_features and in data
        features_to_use = [f for f in self.selected_features if f in rna_log.index]
        rna_subset = rna_log.loc[features_to_use]
        return self.scaler.transform(rna_subset.T)


class MethPreprocessor:
    """Preprocesses methylation data with fit/transform pattern to prevent data leakage.
    
    Note: Feature selection is done on full dataset to ensure consistency,
    but scaling statistics are computed only on training data.
    """
    
    def __init__(self, top_k=2000, selected_features=None):
        self.top_k = top_k
        self.selected_features = selected_features  # Can be pre-computed
        self.scaler = StandardScaler()
    
    @staticmethod
    def select_features_from_full_data(meth_full, top_k=2000):
        """Select top-k high-variance features from full data.
        
        Returns list of feature indices to keep.
        """
        meth_clean = meth_full.dropna()
        var = meth_clean.var(axis=1).sort_values(ascending=False)
        return var.index[:top_k].tolist()
    
    def fit(self, meth):
        """Fit scaler on training data (using pre-selected features)."""
        if self.selected_features is None:
            raise ValueError("selected_features must be set before fit()")
        
        meth_clean = meth.dropna()
        # Keep only intersection with selected features (in case training set has fewer features)
        features_to_use = [f for f in self.selected_features if f in meth_clean.index]
        meth_subset = meth_clean.loc[features_to_use]
        self.scaler.fit(meth_subset.T)
        
        return self
    
    def transform(self, meth):
        """Transform data using fitted statistics."""
        if self.selected_features is None:
            raise ValueError("selected_features must be set before transform()")

        meth_clean = meth.dropna()
        # Keep only features that are both in selected_features and in data
        features_to_use = [f for f in self.selected_features if f in meth_clean.index]
        meth_subset = meth_clean.loc[features_to_use]
        return self.scaler.transform(meth_subset.T)


# Legacy function wrappers for backward compatibility
def preprocess_rna(rna, top_k=2000):
    """Legacy wrapper: fit and transform on full data (use only for test/reference)."""
    features = RNAPreprocessor.select_features_from_full_data(rna, top_k)
    preprocessor = RNAPreprocessor(top_k=top_k, selected_features=features)
    return preprocessor.fit(rna).transform(rna)


def preprocess_meth(meth, top_k=2000):
    """Legacy wrapper: fit and transform on full data (use only for test/reference)."""
    features = MethPreprocessor.select_features_from_full_data(meth, top_k)
    preprocessor = MethPreprocessor(top_k=top_k, selected_features=features)
    return preprocessor.fit(meth).transform(meth)


def scale_data(df):
    """Legacy wrapper: fit and transform on full data (use only for test/reference)."""
    scaler = StandardScaler()
    return scaler.fit_transform(df.T)
