"""
Fitness evaluation for GP imputation.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

def calculate_fitness(y_true: np.ndarray, y_pred: np.ndarray, metric: str = 'nrmse', 
                     parsimony_penalty: float = 0.0, tree_size: int = 0,
                     classifier: Any = None, X_complete: Optional[np.ndarray] = None, 
                     y_target: Optional[np.ndarray] = None, cv_folds: int = 5) -> Tuple[float]:
    """
    Calculate fitness for GP individual.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values (for imputation metrics).
    y_pred : np.ndarray
        Predicted values (for imputation metrics).
    metric : str
        Metric to use ('nrmse', 'rmse', 'mae', 'f1_classifier').
    parsimony_penalty : float
        Coefficient for penalizing complex trees.
    tree_size : int
        Size of the tree (number of nodes).
    classifier : sklearn classifier, optional
        Classifier to use for f1_classifier metric.
    X_complete : np.ndarray, optional
        Complete imputed dataset (for classifier training).
    y_target : np.ndarray, optional
        Target labels for classification.
    cv_folds : int
        Number of cross-validation folds for classifier evaluation.
    
    Returns
    -------
    fitness : Tuple[float]
        Fitness value (lower is better for imputation metrics, higher is better for f1).
    """
    try:
        if metric == 'f1_classifier':
            error = _calculate_classifier_fitness(classifier, X_complete, y_target, cv_folds)
        else:
            error = _calculate_imputation_fitness(y_true, y_pred, metric)
            
        # Add parsimony penalty
        penalty = parsimony_penalty * tree_size
        return (error + penalty,)
        
    except Exception as e:
        logger.error(f"Fitness calculation failed: {e}")
        return (1e10,)

def _calculate_classifier_fitness(classifier: Any, X_complete: np.ndarray, 
                                y_target: np.ndarray, cv_folds: int) -> float:
    """Calculate fitness based on classifier performance."""
    if classifier is None or X_complete is None or y_target is None:
        raise ValueError("classifier, X_complete and y_target required for f1_classifier metric")
    
    if np.any(np.isnan(X_complete)) or np.any(np.isinf(X_complete)):
        logger.warning("Invalid values in X_complete")
        return 1e10
    
    cv_scores = cross_val_score(classifier, X_complete, y_target, 
                               cv=cv_folds, scoring='f1_weighted')
    mean_f1 = cv_scores.mean()
    
    # Convert to minimization problem (1 - f1_score)
    return 1.0 - mean_f1

def _calculate_imputation_fitness(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Calculate fitness based on imputation error."""
    if y_pred is None or np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return 1e10
    
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'nrmse':
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        y_range = np.ptp(y_true)
        return rmse / y_range if y_range > 1e-6 else rmse
    else:
        return np.sqrt(mean_squared_error(y_true, y_pred))
