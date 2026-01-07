#!/usr/bin/env python3
"""
Plot fitness evolution from a parsed GP results JSON (output of
`analysis/parse_results_log.py`).

Usage:
  python3 analysis/plot_parsed_results.py --input path/to/resultados_parsed.json

Generates two PNG files in the output directory:
 - fitness_curves_by_seed.png
 - fitness_curves_mean_std.png

The parsed JSON is expected to have the structure produced by
`parse_results_log.py`: { dataset: { missing: { seed: [ {gen, best_fitness, f1}, ... ] } } }
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style('whitegrid')
except Exception:
    sns = None


def get_color_cycle(n=10):
    """Return a list of n RGB tuples for plotting, preferring seaborn colorblind palette."""
    if sns is not None:
        try:
            return sns.color_palette('colorblind', n_colors=n)
        except Exception:
            pass
    cmap = plt.cm.get_cmap('tab10')
    if hasattr(cmap, 'colors'):
        cols = list(cmap.colors)
        if len(cols) >= n:
            return cols[:n]
    return [cmap(i) for i in range(n)]


def normalize_missing(missing_str):
    # Examples: '10.0%', '10%', '10.0', '0.1'
    if missing_str is None:
        return None
    s = str(missing_str).strip()
    if s.endswith('%'):
        s = s[:-1]
    try:
        val = float(s)
    except ValueError:
        return s
    if val <= 1.0:
        # decimal like 0.1 -> 10%
        return int(round(val * 100))
    return int(round(val))


def load_parsed_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_experiments(data):
    """Convert parsed JSON to experiments mapping:
    experiments[(dataset, missing_rate_int)][seed] = [f1_by_gen...]
    """
    experiments = defaultdict(dict)

    for dataset, d_missing in data.items():
        for missing, d_seeds in d_missing.items():
            missing_norm = normalize_missing(missing)
            key = (dataset, missing_norm)
            for seed, entries in d_seeds.items():
                sorted_entries = sorted(entries, key=lambda e: e.get('gen', 0))
                f1_list = []
                for e in sorted_entries:
                    if e.get('f1') is not None:
                        f1 = float(e.get('f1'))
                    else:
                        f1 = 1.0 - float(e.get('best_fitness', 0.0))
                    f1_list.append(f1)
                experiments[key][str(seed)] = f1_list

    return experiments

def format_dataset_name(name):
    name = name.replace("_", " ")
    return name.capitalize()
    
def plot_fitness_curves_by_seed(experiments, output_dir, baselines=None, font_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np
    import os

    # --- CONFIGURAÇÃO DA FONTE TOMORROW ---
    if font_path and os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        custom_font_name = prop.get_name()
        plt.rcParams.update({'font.family': custom_font_name})
    else:
        plt.rcParams.update({'font.family': 'sans-serif'})
        if font_path:
            print(f"Aviso: Arquivo de fonte não encontrado em '{font_path}'. Usando padrão.")

    # --- CONFIGURAÇÃO ESTÉTICA ---
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 15,
        'ytick.labelsize': 11,
        'legend.fontsize': 20,
        'axes.linewidth': 0.8,
        'grid.color': '#E6E6E6',
        'lines.linewidth': 1.8,
        'figure.autolayout': False,
    })

    datasets = sorted({k[0] for k in experiments.keys() if k[0].lower() != 'wdbc'})
    missing_rates = sorted({k[1] for k in experiments.keys()},
                           key=lambda x: (0 if isinstance(x, int) else 1, x))

    if not datasets:
        print("No experiments found in parsed JSON.")
        return

    def format_dataset_name(name):
        return name.replace("_", " ").title()

    datasets_fmt = {d: format_dataset_name(d) for d in datasets}

    n_rows = len(datasets)
    n_cols = len(missing_rates)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        dpi=300,
        sharex='col', sharey='row'
    )

    plt.subplots_adjust(hspace=0.25, wspace=0.1)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # ------- Limites de Y --------
    row_limits = {}
    for dataset in datasets:
        vals = []
        for missing in missing_rates:
            key = (dataset, missing)
            if key in experiments:
                for v in experiments[key].values():
                    vals.extend(v)
            
            # Include baselines in limits
            if baselines is not None:
                key_bas = (dataset, int(missing)) if isinstance(missing, (int, float)) else (dataset, missing)
                if key_bas in baselines:
                    vals.extend(baselines[key_bas].values())

        if vals:
            ymin, ymax = min(vals), max(vals)
            pad = 0.1 * (ymax - ymin if ymax > ymin else 1)
            row_limits[dataset] = (max(0.0, ymin - pad), min(1.0, ymax + pad))
        else:
            row_limits[dataset] = (0.0, 1.0)

    # Dicionário para coletar itens da legenda de TODOS os plots para evitar duplicatas
    unique_legend_items = {}

    COLOR_GP = "#006064"
    COLOR_MEAN = "#C2185B"    # Rosa escuro
    COLOR_MEDIAN = "#E65100"  # Laranja
    COLOR_BASELINES_OTHER = ["#7B1FA2", "#689F38", "#1976D2", "#5D4037"]

    # --------- Plotagem ---------
    for i, dataset in enumerate(datasets):
        for j, missing in enumerate(missing_rates):

            ax = axes[i, j]
            key = (dataset, missing)

            ax.set_axisbelow(True)
            ax.grid(True, linestyle='-', linewidth=0.5, color='#f0f0f0')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)

            if key not in experiments:
                ax.set_visible(False)
                continue

            seeds = experiments[key]

            # Média GP
            curves = [np.array(v) for v in seeds.values() if len(v) > 0]
            if curves:
                max_len = max(len(c) for c in curves)
                padded = np.vstack([np.pad(c, (0, max_len - len(c)), "edge") for c in curves])
                mean = padded.mean(axis=0)
                gens = np.arange(len(mean))

                line_gp, = ax.plot(
                    gens, mean,
                    color=COLOR_GP,
                    linewidth=2.0,
                    label="Proposed GP"
                )
                if "Proposed GP" not in unique_legend_items:
                    unique_legend_items["Proposed GP"] = line_gp

            # Baselines
            if baselines is not None:
                key_bas = (dataset, int(missing)) if isinstance(missing, (int, float)) else (dataset, missing)
                
                if key_bas in baselines:
                    sorted_methods = sorted(baselines[key_bas].items())
                    
                    other_idx = 0
                    for method, f1_score in sorted_methods:
                        method_key = str(method).lower()
                        
                        display_label = str(method)
                        color_use = COLOR_BASELINES_OTHER[other_idx % len(COLOR_BASELINES_OTHER)]
                        linestyle_use = '--'
                        print(f"Method key: {method_key}")
                        
                        if method_key == 'mean':
                            display_label = 'média'
                            color_use = COLOR_MEAN
                            linestyle_use = '--'
                        elif method_key == 'median':
                            display_label = 'mediana'
                            color_use = COLOR_MEDIAN
                            linestyle_use = '-.'
                        else:
                            other_idx += 1

                        line_base = ax.axhline(
                            y=f1_score,
                            linestyle=linestyle_use,
                            color=color_use,
                            linewidth=1.5,
                            alpha=0.8,
                            label=display_label
                        )
                        
                        if display_label not in unique_legend_items:
                            unique_legend_items[display_label] = line_base

            # Título e Eixos
            ax.set_title(
                f"{datasets_fmt[dataset]} ({missing}%)",
                fontsize=20, pad=8, loc='left', weight='bold'
            )
            ymin, ymax = row_limits.get(dataset, (0.0, 1.0))
            ax.set_ylim(ymin, ymax)

            if j == 0:
                ax.set_ylabel("F1-Score", fontsize=20, labelpad=8)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelleft=False)

            if i == n_rows - 1:
                ax.set_xlabel("Generations", fontsize=20, labelpad=8)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # --- Construção da Legenda Unificada (CORRIGIDO) ---
    if unique_legend_items:
        # Função auxiliar para chave de ordenação
        def legend_sort_key(item):
            lbl = item[0]
            # Retorna TUPLAS para todos os casos para permitir a comparação
            if lbl == "Proposed GP": return (0, lbl)
            if lbl == "Média": return (1, lbl)
            if lbl == "Mediana": return (2, lbl)
            return (3, lbl) # Outros

        sorted_items = sorted(unique_legend_items.items(), key=legend_sort_key)
        
        ordered_handles = []
        ordered_labels = []
        
        for lbl, hdl in sorted_items:
            ordered_labels.append(lbl)
            ordered_handles.append(hdl)

        fig.legend(
            ordered_handles, ordered_labels, 
            loc='lower center', ncol=len(ordered_labels),
            bbox_to_anchor=(0.5, -0.02),
            frameon=False, fontsize=15
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = os.path.join(output_dir, 'fitness_curves_tomorrow.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot fitness evolution from parsed JSON')
    parser.add_argument('--input', '-i', default='logs/gp_seed_experiment_parsed.json', help='Path to parsed JSON')
    parser.add_argument('--output-dir', '-o', default='results/gp_seed_experiment', help='Directory to save plots')
    parser.add_argument('--baselines-csv', '-b', default=None, help='Optional CSV of baselines')
    args = parser.parse_args()

    data = load_parsed_json(args.input)
    experiments = build_experiments(data)

    os.makedirs(args.output_dir, exist_ok=True)

    
    baselines = None
    # if user provided a baselines CSV use it; otherwise try to auto-detect common baseline files
    csv_to_try = []
    if args.baselines_csv:
        csv_to_try.append(args.baselines_csv)
    else:
        csv_to_try.append('results/optuna_optimization/all_optimization_results_20251207_234520.csv')
        # try to pick any CSV under results/optuna_optimization if present
        # try:
        #     import glob
        #     more = glob.glob('results/optuna_optimization/*.csv')
        #     csv_to_try.extend([p for p in more if p not in csv_to_try])
        # except Exception:
        #     pass

    csv_found = None
    for p in csv_to_try:
        if p and os.path.exists(p):
            csv_found = p
            break

    if csv_found:
        print(f'Loading baselines from: {csv_found}')
        args.baselines_csv = csv_found

    if args.baselines_csv:
        try:
            import pandas as pd
            df = pd.read_csv(args.baselines_csv)
            baselines = {}
            # Expecting columns: dataset, missing_ratio, imputer, f1_score (or f1)
            # Group by dataset, missing_ratio, imputer
            for (dataset, missing_ratio, imputer), group in df.groupby(['dataset', 'missing_ratio', 'imputer']):
                if 'f1_score' in group.columns:
                    best_f1 = group['f1_score'].max()
                elif 'f1' in group.columns:
                    best_f1 = group['f1'].max()
                else:
                    # try to find any column with 'f1' in name
                    f1_cols = [c for c in group.columns if 'f1' in c]
                    if f1_cols:
                        best_f1 = group[f1_cols[0]].max()
                    else:
                        continue

                # normalize missing_ratio
                try:
                    m = float(missing_ratio)
                    if m <= 1.0:
                        m_int = int(round(m * 100))
                    else:
                        m_int = int(round(m))
                except Exception:
                    m_int = missing_ratio

                key = (dataset, m_int)
                baselines.setdefault(key, {})[imputer] = float(best_f1)
        except Exception as e:
            print(f"Warning: failed to load baselines CSV: {e}")

    # 2. Carregar CSVs de Imputer Evaluation (onde geralmente estão Mean/Median)
    try:
        import glob
        imputer_files = glob.glob('results/imputer_evaluation*.csv')
        for p in imputer_files:
            try:
                # Tenta extrair missingness do nome do arquivo
                m = re.search(r'imputer_evaluation[_-]?(\d+)', os.path.basename(p))
                file_missing_int = int(m.group(1)) if m else None

                df_imp = pd.read_csv(p)
                for _, row in df_imp.iterrows():
                    ds = row.get('dataset')
                    strategy = row.get('strategy') or row.get('imputer')
                    
                    f1 = None
                    if 'mean_f1' in row.index: f1 = row['mean_f1']
                    elif 'f1_score' in row.index: f1 = row['f1_score']
                    elif 'f1' in row.index: f1 = row['f1']
                    
                    if ds is not None and strategy is not None and pd.notna(f1):
                        # Usa o missing do arquivo se disponível, senão assume que é igual para todos
                        key = (ds, file_missing_int) 
                        baselines.setdefault(key, {})[str(strategy)] = float(f1)
                print(f"Merged baselines from: {p}")
            except Exception:
                pass
    except Exception:
        pass

    plot_fitness_curves_by_seed(experiments, args.output_dir, baselines=baselines, font_path='analysis/Tomorrow-Regular.ttf')


if __name__ == '__main__':
    main()