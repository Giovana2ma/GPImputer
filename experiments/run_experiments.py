"""
Main experiment runner for GP-based imputation.
"""

import sys
import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import json
import pickle
from typing import Dict, List
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import load_dataset, generate_missing_data, normalize_data
from imputers import (
    MeanImputer, MedianImputer, KNNImputerWrapper, 
    MICEImputerWrapper, MissForestImputerWrapper, SVDImputerWrapper
)
from gp import GPImputer
from evaluation import calculate_imputation_error, compare_methods, rank_methods


def setup_logging(log_dir: Path, experiment_name: str):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_classifier(config: dict):
    """
    Create classifier instance based on configuration.
    
    Parameters
    ----------
    config : dict
        Classifier configuration
    
    Returns
    -------
    classifier : sklearn classifier
    """
    clf_type = config.get('type', 'random_forest')
    params = config.get('params', {})
    
    classifiers = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'svm': SVC,
        'decision_tree': DecisionTreeClassifier,
        'knn': KNeighborsClassifier
    }
    
    if clf_type not in classifiers:
        raise ValueError(f"Unknown classifier type: {clf_type}")
    
    return classifiers[clf_type](**params)


def create_imputers(config: dict) -> Dict:
    """
    Create imputer instances based on configuration.
    
    Parameters
    ----------
    config : dict
        Imputers configuration from experiment config
    
    Returns
    -------
    imputers : dict
        Dictionary of imputer instances
    """
    imputers = {}
    
    if config.get('mean', {}).get('enabled', False):
        imputers['mean'] = MeanImputer()
    
    if config.get('median', {}).get('enabled', False):
        imputers['median'] = MedianImputer()
    
    if config.get('knn', {}).get('enabled', False):
        params = config['knn'].get('params', {})
        imputers['knn'] = KNNImputerWrapper(**params)
    
    if config.get('mice', {}).get('enabled', False):
        params = config['mice'].get('params', {})
        imputers['mice'] = MICEImputerWrapper(**params)
    
    if config.get('missforest', {}).get('enabled', False):
        params = config['missforest'].get('params', {})
        imputers['missforest'] = MissForestImputerWrapper(**params)
    
    if config.get('svd', {}).get('enabled', False):
        params = config['svd'].get('params', {})
        imputers['svd'] = SVDImputerWrapper(**params)
    
    return imputers


def run_single_experiment(X_complete: np.ndarray,
                         y_labels: np.ndarray,
                         mechanism: str,
                         missing_rate: float,
                         seed: int,
                         imputers: Dict,
                         gp_config: dict,
                         metrics: List[str],
                         logger: logging.Logger,
                         fold: int = None) -> Dict:
    """
    Run a single experiment iteration.
    
    Parameters
    ----------
    X_complete : np.ndarray
        Complete feature matrix
    y_labels : np.ndarray
        Target labels for classification
    fold : int, optional
        Fold number if using cross-validation
    
    Returns
    -------
    results : dict
        Dictionary with results for each imputer
    """
    fold_str = f", fold={fold}" if fold is not None else ""
    logger.info(f"Running: mechanism={mechanism}, rate={missing_rate}, seed={seed}{fold_str}")
    
    # Generate missing data
    X_missing, mask = generate_missing_data(X_complete, mechanism, missing_rate, seed)
    
    results = {}
    
    # Run each base imputer
    for name, imputer in imputers.items():
        try:
            logger.info(f"  Running imputer: {name}")
            
            # Fit and transform
            X_imputed = imputer.fit_transform(X_missing)
            
            # Calculate metrics on missing values only
            metrics_dict = calculate_imputation_error(X_complete, X_imputed, mask, metrics)
            
            results[name] = {
                'metrics': metrics_dict,
                'imputed_data': X_imputed
            }
            
            logger.info(f"    {name} - RMSE: {metrics_dict.get('rmse', np.nan):.4f}")
            
        except Exception as e:
            logger.error(f"  Error with {name}: {e}")
            results[name] = {
                'metrics': {m: np.nan for m in metrics},
                'imputed_data': None,
                'error': str(e)
            }
    
    # Run GP imputer if enabled
    if 'gp' in imputers or True:  # Always try GP
        try:
            logger.info("  Running GP imputer")
            
            # Prepare base imputers for GP
            base_imputers_fitted = {}
            for name, imputer in imputers.items():
                base_imputers_fitted[f'imp_{name}'] = imputer
            
            # Get ground truth for masked values
            y_true_masked = X_complete[mask]
            
            # Create classifier if fitness metric is f1_classifier
            classifier = None
            fitness_config = gp_config.get('fitness', {})
            if fitness_config.get('metric') == 'f1_classifier':
                clf_config = fitness_config.get('classifier', {})
                classifier = create_classifier(clf_config)
                logger.info(f"    Using classifier: {clf_config.get('type', 'random_forest')}")
            
            # Create and fit GP imputer
            gp_imputer = GPImputer(config=gp_config)
            gp_imputer.fit(X_missing, base_imputers_fitted, 
                          y_true=y_true_masked,
                          classifier=classifier,
                          y_target=y_labels)
            
            # Transform
            X_gp_imputed = gp_imputer.transform(X_missing)
            
            # Calculate metrics
            metrics_dict = calculate_imputation_error(X_complete, X_gp_imputed, mask, metrics)
            
            results['gp'] = {
                'metrics': metrics_dict,
                'imputed_data': X_gp_imputed,
                'best_programs': gp_imputer.get_best_program(),  # Now returns dict of programs
                'best_fitness': gp_imputer.best_fitness_
            }
            
            logger.info(f"    GP - RMSE: {metrics_dict.get('rmse', np.nan):.4f}")
            
            # Log programs per feature
            programs = gp_imputer.get_best_program()
            if isinstance(programs, dict):
                logger.info(f"    GP Programs (per feature):")
                for feat_name, prog in programs.items():
                    logger.info(f"      {feat_name}: {prog}")
            else:
                logger.info(f"    GP Program: {programs}")
            
        except Exception as e:
            logger.error(f"  Error with GP: {e}")
            results['gp'] = {
                'metrics': {m: np.nan for m in metrics},
                'imputed_data': None,
                'error': str(e)
            }
    
    return results


def run_experiments(config_path: str = 'config/experiment_config.yaml',
                   gp_config_path: str = 'config/gp_config.yaml'):
    """
    Run full experimental pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to experiment configuration file
    gp_config_path : str
        Path to GP configuration file
    """
    # Load configurations
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(gp_config_path, 'r') as f:
        gp_config = yaml.safe_load(f)
    
    # Setup logging
    experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(config['experiment']['logs_dir'])
    logger = setup_logging(log_dir, experiment_name)
    
    logger.info("=" * 80)
    logger.info("Starting GP Imputation Experiments")
    logger.info("=" * 80)
    
    # Create results directory
    results_dir = Path(config['experiment']['results_dir']) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create imputers
    imputers = create_imputers(config['imputers'])
    logger.info(f"Created {len(imputers)} imputers: {list(imputers.keys())}")
    
    # Get evaluation metrics
    metrics = config['evaluation']['metrics']['numeric']
    
    # Storage for all results
    all_results = {}
    
    # Iterate over datasets
    for dataset_config in config['datasets']['sources']:
        dataset_name = dataset_config['name']
        dataset_type = dataset_config['type']
        
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        try:
            # Load dataset
            X, y, feature_names = load_dataset(dataset_name, dataset_type)
            logger.info(f"  Shape: {X.shape}")
            
            # Normalize data
            X_normalized, norm_params = normalize_data(X, method='standard')
            
            # Initialize results storage for this dataset
            all_results[dataset_name] = {}
            
            # Setup cross-validation
            n_folds = config['experiment'].get('n_folds', 5)
            use_cv = config['experiment'].get('use_cross_validation', False)
            
            if use_cv:
                logger.info(f"  Using {n_folds}-fold cross-validation")
            
            # Iterate over missing mechanisms
            for mechanism in config['missingness']['mechanisms']:
                all_results[dataset_name][mechanism] = {}
                
                # Iterate over missing rates
                for missing_rate in config['missingness']['rates']:
                    all_results[dataset_name][mechanism][missing_rate] = {}
                    
                    # Iterate over seeds
                    for seed in config['missingness']['seeds']:
                        
                        if use_cv:
                            # Cross-validation mode
                            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                            fold_results = []
                            
                            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_normalized)):
                                # Use only test fold for imputation evaluation
                                X_fold = X_normalized[test_idx]
                                y_fold = y[test_idx] if y is not None else None
                                
                                # Run experiment on this fold
                                exp_results = run_single_experiment(
                                    X_fold,
                                    y_fold,
                                    mechanism,
                                    missing_rate,
                                    seed,
                                    imputers,
                                    gp_config,
                                    metrics,
                                    logger,
                                    fold=fold_idx + 1
                                )
                                
                                fold_results.append(exp_results)
                            
                            # Aggregate results across folds
                            all_results[dataset_name][mechanism][missing_rate][seed] = {
                                'folds': fold_results,
                                'aggregated': aggregate_fold_results(fold_results, metrics)
                            }
                        else:
                            # Standard mode without CV
                            exp_results = run_single_experiment(
                                X_normalized,
                                y,
                                mechanism,
                                missing_rate,
                                seed,
                                imputers,
                                gp_config,
                                metrics,
                                logger
                            )
                            
                            # Store results
                            all_results[dataset_name][mechanism][missing_rate][seed] = exp_results
            
            logger.info(f"Completed dataset: {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save complete results as pickle
    results_file = results_dir / 'results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"Saved complete results to {results_file}")
    
    # Create summary statistics
    use_cv = config['experiment'].get('use_cross_validation', False)
    summary = create_summary(all_results, metrics, use_cv)
    summary_file = results_dir / 'summary.csv'
    summary.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to {summary_file}")
    
    # Statistical tests
    logger.info("\nPerforming statistical tests...")
    statistical_results = perform_statistical_tests(all_results, config, logger)
    
    stats_file = results_dir / 'statistical_tests.json'
    with open(stats_file, 'w') as f:
        json.dump(statistical_results, f, indent=2)
    logger.info(f"Saved statistical tests to {stats_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Experiments completed successfully!")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("=" * 80)
    
    return all_results, summary


def aggregate_fold_results(fold_results: List[Dict], metrics: List[str]) -> Dict:
    """
    Aggregate results across folds.
    
    Parameters
    ----------
    fold_results : list
        List of results dictionaries from each fold
    metrics : list
        List of metric names
    
    Returns
    -------
    aggregated : dict
        Dictionary with mean and std for each method and metric
    """
    aggregated = {}
    
    # Get all methods from first fold
    methods = list(fold_results[0].keys())
    
    for method in methods:
        aggregated[method] = {'metrics': {}}
        
        for metric in metrics:
            # Collect metric values across folds
            values = []
            for fold in fold_results:
                if method in fold and 'metrics' in fold[method]:
                    val = fold[method]['metrics'].get(metric, np.nan)
                    if not np.isnan(val):
                        values.append(val)
            
            if values:
                aggregated[method]['metrics'][f'{metric}_mean'] = np.mean(values)
                aggregated[method]['metrics'][f'{metric}_std'] = np.std(values)
                aggregated[method]['metrics'][metric] = np.mean(values)  # For compatibility
            else:
                aggregated[method]['metrics'][f'{metric}_mean'] = np.nan
                aggregated[method]['metrics'][f'{metric}_std'] = np.nan
                aggregated[method]['metrics'][metric] = np.nan
    
    return aggregated


def create_summary(all_results: Dict, metrics: List[str], use_cv: bool = False) -> pd.DataFrame:
    """Create summary DataFrame from all results."""
    summary_data = []
    
    for dataset in all_results:
        for mechanism in all_results[dataset]:
            for rate in all_results[dataset][mechanism]:
                for seed in all_results[dataset][mechanism][rate]:
                    exp_results = all_results[dataset][mechanism][rate][seed]
                    
                    if use_cv and 'aggregated' in exp_results:
                        # Use aggregated CV results
                        agg_results = exp_results['aggregated']
                        
                        for method in agg_results:
                            row = {
                                'dataset': dataset,
                                'mechanism': mechanism,
                                'missing_rate': rate,
                                'seed': seed,
                                'method': method
                            }
                            
                            # Add metrics with mean and std
                            for metric in metrics:
                                mean_val = agg_results[method]['metrics'].get(f'{metric}_mean', np.nan)
                                std_val = agg_results[method]['metrics'].get(f'{metric}_std', np.nan)
                                row[metric] = mean_val
                                row[f'{metric}_std'] = std_val
                            
                            summary_data.append(row)
                    else:
                        # Standard results without CV
                        for method in exp_results:
                            row = {
                                'dataset': dataset,
                                'mechanism': mechanism,
                                'missing_rate': rate,
                                'seed': seed,
                                'method': method
                            }
                            
                            # Add metrics
                            for metric in metrics:
                                value = exp_results[method]['metrics'].get(metric, np.nan)
                                row[metric] = value
                            
                            summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def perform_statistical_tests(all_results: Dict, config: dict, logger: logging.Logger) -> Dict:
    """Perform statistical tests on results."""
    statistical_results = {}
    use_cv = config['experiment'].get('use_cross_validation', False)
    
    # Aggregate results by method across all conditions
    method_results = {}
    
    for dataset in all_results:
        for mechanism in all_results[dataset]:
            for rate in all_results[dataset][mechanism]:
                for seed in all_results[dataset][mechanism][rate]:
                    exp_results = all_results[dataset][mechanism][rate][seed]
                    
                    if use_cv and 'aggregated' in exp_results:
                        # Use aggregated CV results
                        agg_results = exp_results['aggregated']
                        for method in agg_results:
                            if method not in method_results:
                                method_results[method] = []
                            
                            rmse = agg_results[method]['metrics'].get('rmse', np.nan)
                            if not np.isnan(rmse):
                                method_results[method].append(rmse)
                    else:
                        # Standard results
                        for method in exp_results:
                            if method not in method_results:
                                method_results[method] = []
                            
                            rmse = exp_results[method]['metrics'].get('rmse', np.nan)
                            if not np.isnan(rmse):
                                method_results[method].append(rmse)
    
    # Convert to arrays
    for method in method_results:
        method_results[method] = np.array(method_results[method])
    
    # Perform tests
    try:
        comparison = compare_methods(method_results, test='friedman')
        statistical_results['friedman'] = {
            'statistic': float(comparison.get('statistic', np.nan)),
            'p_value': float(comparison.get('p_value', np.nan)),
            'significant': bool(comparison.get('significant', False))
        }
        
        # Rankings
        rankings = rank_methods(method_results)
        statistical_results['rankings'] = rankings.to_dict('records')
        
        logger.info("\nMethod Rankings (by mean RMSE):")
        logger.info(rankings.to_string())
        
    except Exception as e:
        logger.error(f"Error in statistical tests: {e}")
    
    return statistical_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GP imputation experiments')
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                       help='Path to experiment config')
    parser.add_argument('--gp-config', type=str, default='config/gp_config.yaml',
                       help='Path to GP config')
    
    args = parser.parse_args()
    
    results, summary = run_experiments(args.config, args.gp_config)
