"""
Utilities for experiment management.
"""

import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict


def load_results(results_dir: str) -> Dict:
    """Load experimental results from directory."""
    results_path = Path(results_dir) / 'results.pkl'
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


def load_summary(results_dir: str) -> pd.DataFrame:
    """Load summary CSV."""
    summary_path = Path(results_dir) / 'summary.csv'
    return pd.read_csv(summary_path)


def save_gp_programs(results: Dict, output_file: str):
    """Extract and save all GP programs to file."""
    programs = []
    
    for dataset in results:
        for mechanism in results[dataset]:
            for rate in results[dataset][mechanism]:
                for seed in results[dataset][mechanism][rate]:
                    exp_results = results[dataset][mechanism][rate][seed]
                    
                    if 'gp' in exp_results and 'best_program' in exp_results['gp']:
                        programs.append({
                            'dataset': dataset,
                            'mechanism': mechanism,
                            'missing_rate': rate,
                            'seed': seed,
                            'program': exp_results['gp']['best_program'],
                            'fitness': exp_results['gp'].get('best_fitness', None),
                            'nrmse': exp_results['gp']['metrics'].get('nrmse', None)
                        })
    
    df = pd.DataFrame(programs)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(programs)} GP programs to {output_file}")
    
    return df
