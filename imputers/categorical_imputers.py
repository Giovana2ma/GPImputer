"""
Categorical imputers for mode and most_frequent strategies.
"""

import numpy as np
from scipy import stats
from .base import BaseImputer


class ModeImputerWrapper(BaseImputer):
    """Mode imputation for categorical features."""
    
    def __init__(self):
        super().__init__(name="ModeImputer")
        self.modes_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """
        Fit mode imputer.
        
        Parameters
        ----------
        X : np.ndarray
            Data with potential missing values
        y : ignored
        
        Returns
        -------
        self
        """
        X = self._validate_input(X)
        
        # Calcular a moda de cada coluna (ignorando NaNs)
        self.modes_ = []
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            # Remover NaNs
            col_no_nan = col[~np.isnan(col)]
            if len(col_no_nan) > 0:
                mode_result = stats.mode(col_no_nan, keepdims=False)
                self.modes_.append(mode_result.mode)
            else:
                # Se toda a coluna é NaN, usar 0
                self.modes_.append(0)
        
        self.modes_ = np.array(self.modes_)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute using mode.
        
        Parameters
        ----------
        X : np.ndarray
            Data to impute
            
        Returns
        -------
        X_imputed : np.ndarray
            Imputed data
        """
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform.")
        
        X = self._validate_input(X)
        X_imputed = X.copy()
        
        for col_idx in range(X.shape[1]):
            mask = np.isnan(X_imputed[:, col_idx])
            X_imputed[mask, col_idx] = self.modes_[col_idx]
        
        return X_imputed
    
    def __repr__(self):
        return "ModeImputer()"
