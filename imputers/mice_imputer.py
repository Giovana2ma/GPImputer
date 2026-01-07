"""
MICE (Multivariate Imputation by Chained Equations) imputation.
"""

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from .base import BaseImputer


class MICEImputerWrapper(BaseImputer):
    """Wrapper for sklearn's IterativeImputer (MICE)."""
    
    def __init__(self, max_iter: int = 10, random_state: int = None, **kwargs):
        super().__init__(name="MICEImputer")
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Use BayesianRidge as default estimator
        estimator = kwargs.pop('estimator', BayesianRidge())
        
        self.imputer_ = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y=None):
        """Fit MICE imputer."""
        X = self._validate_input(X)
        self.imputer_.fit(X)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute using MICE."""
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform.")
        
        X = self._validate_input(X)
        return self.imputer_.transform(X)
    
    def __repr__(self):
        return f"MICEImputer(max_iter={self.max_iter})"
