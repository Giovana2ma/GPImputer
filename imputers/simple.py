"""
Simple statistical imputers (mean, median, mode).
"""

import numpy as np
from scipy import stats
from .base import BaseImputer


class MeanImputer(BaseImputer):
    """Impute missing values with column mean."""
    
    def __init__(self):
        super().__init__(name="MeanImputer")
        self.statistics_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """Compute mean for each column."""
        X = self._validate_input(X)
        self.statistics_ = np.nanmean(X, axis=0)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace missing values with mean."""
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform.")
        
        X = self._validate_input(X)
        X_imputed = X.copy()
        
        for col_idx in range(X.shape[1]):
            mask = np.isnan(X[:, col_idx])
            X_imputed[mask, col_idx] = self.statistics_[col_idx]
        
        return X_imputed


class MedianImputer(BaseImputer):
    """Impute missing values with column median."""
    
    def __init__(self):
        super().__init__(name="MedianImputer")
        self.statistics_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """Compute median for each column."""
        X = self._validate_input(X)
        self.statistics_ = np.nanmedian(X, axis=0)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace missing values with median."""
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform.")
        
        X = self._validate_input(X)
        X_imputed = X.copy()
        
        for col_idx in range(X.shape[1]):
            mask = np.isnan(X[:, col_idx])
            X_imputed[mask, col_idx] = self.statistics_[col_idx]
        
        return X_imputed


class ModeImputer(BaseImputer):
    """Impute missing values with column mode (most frequent value)."""
    
    def __init__(self):
        super().__init__(name="ModeImputer")
        self.statistics_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """Compute mode for each column."""
        X = self._validate_input(X)
        self.statistics_ = np.zeros(X.shape[1])
        
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            col_data = col_data[~np.isnan(col_data)]
            if len(col_data) > 0:
                mode_result = stats.mode(col_data, keepdims=True)
                self.statistics_[col_idx] = mode_result.mode[0]
            else:
                self.statistics_[col_idx] = 0  # Default for all-missing columns
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace missing values with mode."""
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform.")
        
        X = self._validate_input(X)
        X_imputed = X.copy()
        
        for col_idx in range(X.shape[1]):
            mask = np.isnan(X[:, col_idx])
            X_imputed[mask, col_idx] = self.statistics_[col_idx]
        
        return X_imputed
