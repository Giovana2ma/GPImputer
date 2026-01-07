"""
KNN-based imputation.
"""

import numpy as np
from sklearn.impute import KNNImputer
from .base import BaseImputer


class KNNImputerWrapper(BaseImputer):
    """Wrapper for sklearn's KNNImputer."""
    
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", **kwargs):
        super().__init__(name="KNNImputer")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.kwargs = kwargs
        self.imputer_ = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y=None):
        """Fit KNN imputer."""
        X = self._validate_input(X)
        self.imputer_.fit(X)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute using KNN."""
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform.")
        
        X = self._validate_input(X)
        return self.imputer_.transform(X)
    
    def __repr__(self):
        return f"KNNImputer(n_neighbors={self.n_neighbors}, weights='{self.weights}')"
