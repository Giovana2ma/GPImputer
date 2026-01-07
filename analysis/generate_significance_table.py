import json
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import glob
import os
import re

def load_gp_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gp_results = {} # (dataset, missing_rate) -> list of final f1s
    
    for dataset, missing_dict in data.items():
        for missing_str, seeds in missing_dict.items():
            # Normalize missing rate string to integer (e.g. "10.0%" -> 10)
            try:
                if "%" in missing_str:
                    missing_rate = int(float(missing_str.strip("%")))
                else:
                    val = float(missing_str)
                    if val < 1.0:
                        missing_rate = int(val * 100)
                    else:
                        missing_rate = int(val)
            except:
                continue
                
            f1_values = []
            for seed, runs in seeds.items():
                # Get the last generation's F1 or best_fitness
                # Assuming the list is sorted by gen, or we take the one with max gen
                if not runs:
                    continue
                
                # Sort by gen just in case
                runs.sort(key=lambda x: x.get('gen', -1))
                last_run = runs[-1]
                
                if 'f1' in last_run:
                    f1 = last_run['f1']
                elif 'best_fitness' in last_run:
                    # Assuming fitness is minimized error? Or maximized F1?
                    # In the config, metric is "f1_classifier". 
                    # Usually GP libraries minimize fitness. If fitness = 1 - F1.
                    # Let's check the values. In the json snippet: best_fitness: 0.225, f1: 0.7748.
                    # Sum is approx 1.0. So f1 is the metric.
                    f1 = last_run.get('f1', 1.0 - last_run['best_fitness'])
                else:
                    continue
                f1_values.append(f1)
            
            if f1_values:
                gp_results[(dataset, missing_rate)] = f1_values
                print(f"Loaded {len(f1_values)} seeds for {dataset} {missing_rate}%")
                
    return gp_results

def load_baselines(results_dir):
    baselines = {} # (dataset, missing_rate) -> {strategy: mean_f1}
    
    files = glob.glob(os.path.join(results_dir, 'imputer_evaluation_*.csv'))
    
    for file_path in files:
        # Extract missing rate from filename
        match = re.search(r'imputer_evaluation_(\d+)', os.path.basename(file_path))
        if not match:
            continue
        missing_rate = int(match.group(1))
        
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            dataset = row['dataset']
            strategy = row['strategy']
            mean_f1 = row['mean_f1']
            
            key = (dataset, missing_rate)
            if key not in baselines:
                baselines[key] = {}
            baselines[key][strategy] = mean_f1
            
    return baselines

def load_optuna_baselines(csv_path):
    baselines = {} # (dataset, missing_rate) -> {strategy: max_f1}
    
    if not os.path.exists(csv_path):
        print(f"Warning: Optuna results file not found: {csv_path}")
        return baselines
        
    df = pd.read_csv(csv_path)
    
    # Group by dataset, missing_ratio, imputer and get max f1_score
    grouped = df.groupby(['dataset', 'missing_ratio', 'imputer'])['f1_score'].max().reset_index()
    
    for _, row in grouped.iterrows():
        dataset = row['dataset']
        missing_ratio = row['missing_ratio']
        imputer = row['imputer']
        f1_score = row['f1_score']
        
        # Convert missing_ratio to percentage int
        try:
            if missing_ratio <= 1.0:
                missing_rate = int(round(missing_ratio * 100))
            else:
                missing_rate = int(missing_ratio)
        except:
            continue
            
        key = (dataset, missing_rate)
        if key not in baselines:
            baselines[key] = {}
        baselines[key][imputer] = f1_score
        
    return baselines

def perform_analysis():
    gp_json_path = 'logs/gp_seed_experiment_parsed.json'
    results_dir = 'results'
    optuna_csv_path = 'results/optuna_optimization/all_optimization_results_20251207_234520.csv'
    
    gp_data = load_gp_results(gp_json_path)
    baseline_data = load_baselines(results_dir)
    optuna_data = load_optuna_baselines(optuna_csv_path)
    
    # Merge optuna data into baseline data
    for key, imputers in optuna_data.items():
        if key not in baseline_data:
            baseline_data[key] = {}
        for imputer, f1 in imputers.items():
            baseline_data[key][imputer] = f1
    
    # Organize data for the new table format
    # Structure: Dataset -> Model -> Missing Rate -> F1
    
    datasets = sorted(list(set([k[0] for k in gp_data.keys()])))
    missing_rates = sorted(list(set([k[1] for k in gp_data.keys()])))
    
    # Collect all models
    models = set(['GP-Imputer'])
    for baselines in baseline_data.values():
        models.update(baselines.keys())
    models = sorted(list(models))
    
    # Move GP-Imputer to the end or beginning? Let's put it at the end or highlight it.
    if 'GP-Imputer' in models:
        models.remove('GP-Imputer')
        models.append('GP-Imputer')
        
    latex_rows = []
    
    latex_rows.append(r"\begin{table*}[t]")
    latex_rows.append(r"\centering")
    latex_rows.append(r"\caption{Resultados experimentais consolidados (F1-Score). Os melhores resultados por dataset e taxa de missing estão em \textbf{negrito}. O símbolo $^{\dagger}$ indica diferença estatisticamente significativa ($p < 0.05$) entre o GP-Imputer e o baseline (Wilcoxon Signed-Rank Test).}")
    latex_rows.append(r"\label{tab:all_results}")
    latex_rows.append(r"\resizebox{0.95\textwidth}{!}{%")
    
    # Header
    header = r"\begin{tabular}{ll" + "c" * len(missing_rates) + "}"
    latex_rows.append(header)
    latex_rows.append(r"\toprule")
    
    # Column names
    col_names = r"\textbf{Dataset} & \textbf{Modelo}"
    for mr in missing_rates:
        col_names += f" & \\textbf{{{mr}\%}}"
    col_names += r" \\"
    latex_rows.append(col_names)
    latex_rows.append(r"\midrule")
    
    for dataset in datasets:
        # Determine multirow size
        latex_rows.append(r"\multirow{" + str(len(models)) + r"}{*}{\textbf{" + dataset.replace("_", " ").title() + r"}}")
        
        # Find best values for each missing rate to bold them
        best_values = {}
        for mr in missing_rates:
            best_val = -1.0
            
            # Check GP
            if (dataset, mr) in gp_data:
                val = np.mean(gp_data[(dataset, mr)])
                if val > best_val:
                    best_val = val
            
            # Check Baselines
            if (dataset, mr) in baseline_data:
                for model, val in baseline_data[(dataset, mr)].items():
                    if val > best_val:
                        best_val = val
            
            best_values[mr] = best_val

        for i, model in enumerate(models):
            row_str = ""
            if i > 0:
                row_str += r" & "
            else:
                row_str += r" & " # First column is handled by multirow
            
            # Model Name
            if model == 'GP-Imputer':
                row_str += r"\textbf{GP-Imputer}"
            else:
                row_str += model.replace("_", " ").title()
            
            for mr in missing_rates:
                val_str = "-"
                val = -1.0
                
                if model == 'GP-Imputer':
                    if (dataset, mr) in gp_data:
                        vals = gp_data[(dataset, mr)]
                        mean = np.mean(vals)
                        std = np.std(vals)
                        val = mean
                        val_str = f"{mean:.4f} $\pm$ {std:.4f}"
                else:
                    if (dataset, mr) in baseline_data and model in baseline_data[(dataset, mr)]:
                        val = baseline_data[(dataset, mr)][model]
                        val_str = f"{val:.4f}"
                        
                        # Perform Wilcoxon Test (GP vs Baseline)
                        if (dataset, mr) in gp_data:
                            gp_vals = gp_data[(dataset, mr)]
                            try:
                                # H0: Median(GP - Baseline) = 0
                                diffs = np.array(gp_vals) - val
                                if np.all(diffs == 0):
                                    p_val = 1.0
                                else:
                                    _, p_val = wilcoxon(diffs, alternative='two-sided')
                                
                                if p_val < 0.05:
                                    val_str += r"$^{\dagger}$"
                            except Exception:
                                pass
                
                # Bold if best
                if val >= best_values[mr] and val > 0:
                    val_str = r"\textbf{" + val_str + "}"
                
                row_str += f" & {val_str}"
            
            row_str += r" \\"
            latex_rows.append(row_str)
        
        latex_rows.append(r"\midrule")
    
    # Remove last midrule and add bottomrule
    latex_rows.pop() 
    latex_rows.append(r"\bottomrule")
    latex_rows.append(r"\end{tabular}%")
    latex_rows.append(r"}")
    latex_rows.append(r"\end{table*}")
    
    return "\n".join(latex_rows)

if __name__ == "__main__":
    latex_table = perform_analysis()
    print(latex_table)
    
    # Save to CSV
    df.to_csv('analysis/significance_table.csv', index=False)
