"""
Base class for all imputers.
Provides a common interface for fit, transform, and fit_transform methods.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseImputer(ABC):
    """Abstract base class for imputers."""
    
    def __init__(self, name: str = "BaseImputer"):
        self.name = name
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y=None):
        """
        Fit the imputer on the data.
        
        Parameters
        ----------
        X : np.ndarray
            Data with missing values (NaN)
        y : array-like, optional
            Not used, present for API consistency
        
        Returns
        -------
        self
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values in X.
        
        Parameters
        ----------
        X : np.ndarray
            Data with missing values
        
        Returns
        -------
        X_imputed : np.ndarray
            Data with imputed values
        """
        pass
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : np.ndarray
            Data with missing values
        y : array-like, optional
            Not used, present for API consistency
        
        Returns
        -------
        X_imputed : np.ndarray
            Data with imputed values
        """
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)
    
    def __repr__(self):
        return f"{self.name}()"
