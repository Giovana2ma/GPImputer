"""
SVD/Matrix Factorization imputation.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from .base import BaseImputer


class SVDImputerWrapper(BaseImputer):
    """
    SVD-based imputation using iterative matrix factorization.
    Simple implementation using TruncatedSVD with iterative refinement.
    """
    
    def __init__(self, rank: int = 5, max_iter: int = 100, tol: float = 1e-4):
        super().__init__(name="SVDImputer")
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.mean_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """Fit SVD imputer (stores column means)."""
        X = self._validate_input(X)
        self.mean_ = np.nanmean(X, axis=0)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute using SVD with iterative refinement."""
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform.")
        
        X = self._validate_input(X)
        X_imputed = X.copy()
        
        # Initial imputation with mean
        for col_idx in range(X.shape[1]):
            mask = np.isnan(X[:, col_idx])
            X_imputed[mask, col_idx] = self.mean_[col_idx]
        
        # Iterative SVD refinement
        missing_mask = np.isnan(X)
        
        for iteration in range(self.max_iter):
            X_old = X_imputed.copy()
            
            # Apply SVD
            if X_imputed.shape[1] > self.rank:
                # Center data
                X_centered = X_imputed - self.mean_
                
                # Perform SVD
                svd = TruncatedSVD(n_components=self.rank, random_state=42)
                X_transformed = svd.fit_transform(X_centered)
                X_reconstructed = X_transformed @ svd.components_ + self.mean_
                
                # Update only missing values
                X_imputed[missing_mask] = X_reconstructed[missing_mask]
            
            # Check convergence
            change = np.sqrt(np.sum((X_imputed - X_old) ** 2))
            if change < self.tol:
                break
        
        return X_imputed
    
    def __repr__(self):
        return f"SVDImputer(rank={self.rank}, max_iter={self.max_iter})"
