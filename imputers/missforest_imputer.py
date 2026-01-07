"""
MissForest imputation using Random Forest.
"""

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from .base import BaseImputer


class MissForestImputerWrapper(BaseImputer):
    """MissForest imputer using RandomForest as estimator."""
    
    def __init__(self, max_iter: int = 10, random_state: int = None, 
                 n_estimators: int = 100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', imputation_order='ascending',
                 **kwargs):
        super().__init__(name="MissForestImputer")
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.imputation_order = imputation_order
        self.kwargs = kwargs
        
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.imputer_ = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            imputation_order=imputation_order,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y=None):
        """Fit MissForest imputer."""
        X = self._validate_input(X)
        self.imputer_.fit(X)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute using MissForest."""
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform.")
        
        X = self._validate_input(X)
        return self.imputer_.transform(X)
    
    def __repr__(self):
        return f"MissForestImputer(max_iter={self.max_iter}, n_estimators={self.n_estimators})"
