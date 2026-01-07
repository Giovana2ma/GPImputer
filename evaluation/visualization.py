"""
Visualization utilities for results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path


def plot_results(results_dict: Dict[str, np.ndarray], 
                metric_name: str = 'Error',
                save_path: str = None,
                figsize: tuple = (12, 6)):
    """
    Plot boxplot comparison of methods.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with method names as keys and result arrays as values
    metric_name : str
        Name of the metric being plotted
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    # Prepare data for plotting
    data = []
    for method, results in results_dict.items():
        results_flat = results.flatten() if results.ndim > 1 else results
        for value in results_flat:
            if not np.isnan(value):
                data.append({'Method': method, metric_name: value})
    
    df = pd.DataFrame(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(data=df, x='Method', y=metric_name, ax=ax)
    ax.set_title(f'{metric_name} Comparison Across Methods')
    ax.set_xlabel('Method')
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_convergence(logbook, save_path: str = None, figsize: tuple = (10, 6)):
    """
    Plot GP convergence curve.
    
    Parameters
    ----------
    logbook : deap.tools.Logbook
        DEAP logbook from GP evolution
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    gen = logbook.select("gen")
    min_fits = logbook.select("min")
    avg_fits = logbook.select("avg")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(gen, min_fits, 'b-', label='Best Fitness', linewidth=2)
    ax.plot(gen, avg_fits, 'r--', label='Average Fitness', linewidth=1.5)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('GP Evolution Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_comparison(results_dict: Dict[str, Dict[str, np.ndarray]],
                   datasets: List[str],
                   metric: str = 'NRMSE',
                   save_path: str = None,
                   figsize: tuple = (14, 8)):
    """
    Plot heatmap of method performance across datasets.
    
    Parameters
    ----------
    results_dict : dict
        Nested dict: {method: {dataset: results_array}}
    datasets : list
        List of dataset names
    metric : str
        Metric name for title
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    # Prepare data matrix
    methods = list(results_dict.keys())
    data_matrix = np.zeros((len(methods), len(datasets)))
    
    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            if dataset in results_dict[method]:
                results = results_dict[method][dataset]
                data_matrix[i, j] = np.nanmean(results)
            else:
                data_matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data_matrix, 
                annot=True, 
                fmt='.4f',
                xticklabels=datasets,
                yticklabels=methods,
                cmap='RdYlGn_r',  # Red = bad, Green = good
                ax=ax,
                cbar_kws={'label': metric})
    
    ax.set_title(f'{metric} Performance Across Datasets and Methods')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Method')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_missing_rate_effect(results_by_rate: Dict[float, Dict[str, np.ndarray]],
                             metric: str = 'NRMSE',
                             save_path: str = None,
                             figsize: tuple = (10, 6)):
    """
    Plot how imputation error changes with missing data rate.
    
    Parameters
    ----------
    results_by_rate : dict
        {missing_rate: {method: results_array}}
    metric : str
        Metric name
    save_path : str, optional
        Path to save
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    missing_rates = sorted(results_by_rate.keys())
    methods = list(results_by_rate[missing_rates[0]].keys())
    
    for method in methods:
        means = []
        stds = []
        
        for rate in missing_rates:
            results = results_by_rate[rate][method]
            means.append(np.nanmean(results))
            stds.append(np.nanstd(results))
        
        means = np.array(means)
        stds = np.array(stds)
        
        ax.plot(missing_rates, means, marker='o', label=method, linewidth=2)
        ax.fill_between(missing_rates, means - stds, means + stds, alpha=0.2)
    
    ax.set_xlabel('Missing Data Rate')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Missing Data Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Missing rate plot saved to {save_path}")
    
    plt.show()
    
    return fig
