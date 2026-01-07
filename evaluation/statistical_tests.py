"""
Statistical tests for comparing imputation methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import warnings


def wilcoxon_test(results1: np.ndarray, results2: np.ndarray, 
                  alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test to compare two methods.
    
    Parameters
    ----------
    results1 : np.ndarray
        Results (e.g., errors) from method 1
    results2 : np.ndarray
        Results from method 2
    alternative : str
        Alternative hypothesis ('two-sided', 'less', 'greater')
    
    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value
    """
    # Remove NaN values
    mask = ~(np.isnan(results1) | np.isnan(results2))
    results1 = results1[mask]
    results2 = results2[mask]
    
    if len(results1) < 3:
        return np.nan, np.nan
    
    try:
        statistic, p_value = stats.wilcoxon(results1, results2, alternative=alternative)
        return statistic, p_value
    except Exception as e:
        warnings.warn(f"Wilcoxon test failed: {e}")
        return np.nan, np.nan


def friedman_test(results_dict: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Perform Friedman test to compare multiple methods across multiple datasets.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with method names as keys and result arrays as values
        Each array should have shape (n_datasets, n_repeats) or (n_datasets,)
    
    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value
    """
    # Convert to list of arrays
    method_names = list(results_dict.keys())
    results_list = []
    
    for method in method_names:
        res = results_dict[method]
        if res.ndim == 2:
            # Average over repeats
            res = np.nanmean(res, axis=1)
        results_list.append(res)
    
    # Stack results: shape (n_methods, n_datasets)
    results_array = np.array(results_list)
    
    # Remove datasets with any NaN
    mask = ~np.any(np.isnan(results_array), axis=0)
    results_array = results_array[:, mask]
    
    if results_array.shape[1] < 3:
        return np.nan, np.nan
    
    try:
        statistic, p_value = stats.friedmanchisquare(*results_array)
        return statistic, p_value
    except Exception as e:
        warnings.warn(f"Friedman test failed: {e}")
        return np.nan, np.nan


def nemenyi_test(results_dict: Dict[str, np.ndarray], alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform Nemenyi post-hoc test after Friedman test.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with method names as keys and result arrays as values
    alpha : float
        Significance level
    
    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame with pairwise comparisons and significance
    """
    try:
        from scikit_posthocs import posthoc_nemenyi_friedman
    except ImportError:
        warnings.warn("scikit-posthocs not installed. Install with: pip install scikit-posthocs")
        return pd.DataFrame()
    
    # Prepare data for posthoc test
    method_names = list(results_dict.keys())
    results_list = []
    
    for method in method_names:
        res = results_dict[method]
        if res.ndim == 2:
            res = np.nanmean(res, axis=1)
        results_list.append(res)
    
    # Create DataFrame: rows are datasets, columns are methods
    df = pd.DataFrame(np.array(results_list).T, columns=method_names)
    
    # Remove rows with NaN
    df = df.dropna()
    
    if len(df) < 3:
        warnings.warn("Not enough valid data for Nemenyi test")
        return pd.DataFrame()
    
    try:
        # Perform Nemenyi test
        result = posthoc_nemenyi_friedman(df)
        return result
    except Exception as e:
        warnings.warn(f"Nemenyi test failed: {e}")
        return pd.DataFrame()


def compare_methods(results_dict: Dict[str, np.ndarray], 
                   test: str = 'friedman',
                   alpha: float = 0.05) -> Dict:
    """
    Compare multiple methods using statistical tests.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with method names as keys and result arrays as values
    test : str
        Test to use ('friedman', 'wilcoxon')
    alpha : float
        Significance level
    
    Returns
    -------
    comparison_results : dict
        Dictionary with test results
    """
    if test == 'friedman':
        statistic, p_value = friedman_test(results_dict)
        
        results = {
            'test': 'Friedman',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha if not np.isnan(p_value) else False,
            'alpha': alpha
        }
        
        # If Friedman is significant, perform post-hoc Nemenyi
        if results['significant']:
            nemenyi_results = nemenyi_test(results_dict, alpha)
            results['posthoc'] = nemenyi_results
        
        return results
    
    elif test == 'wilcoxon':
        # Pairwise Wilcoxon tests
        method_names = list(results_dict.keys())
        n_methods = len(method_names)
        
        comparisons = []
        
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method1 = method_names[i]
                method2 = method_names[j]
                
                res1 = results_dict[method1]
                res2 = results_dict[method2]
                
                if res1.ndim == 2:
                    res1 = res1.flatten()
                if res2.ndim == 2:
                    res2 = res2.flatten()
                
                stat, p_val = wilcoxon_test(res1, res2)
                
                comparisons.append({
                    'method1': method1,
                    'method2': method2,
                    'statistic': stat,
                    'p_value': p_val,
                    'significant': p_val < alpha if not np.isnan(p_val) else False
                })
        
        return {
            'test': 'Wilcoxon',
            'comparisons': pd.DataFrame(comparisons),
            'alpha': alpha
        }
    
    else:
        raise ValueError(f"Unknown test: {test}")


def rank_methods(results_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Rank methods based on average performance.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with method names as keys and result arrays as values
    
    Returns
    -------
    rankings : pd.DataFrame
        DataFrame with method rankings
    """
    rankings = []
    
    for method, results in results_dict.items():
        mean_result = np.nanmean(results)
        std_result = np.nanstd(results)
        median_result = np.nanmedian(results)
        
        rankings.append({
            'method': method,
            'mean': mean_result,
            'std': std_result,
            'median': median_result,
        })
    
    df = pd.DataFrame(rankings)
    df = df.sort_values('mean')  # Lower is better for error metrics
    df['rank'] = range(1, len(df) + 1)
    
    return df
