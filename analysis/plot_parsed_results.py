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
    # some colormaps expose .colors, fallback to sampling
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
                # entries is a list of dicts with keys gen,best_fitness,f1
                sorted_entries = sorted(entries, key=lambda e: e.get('gen', 0))
                f1_list = []
                for e in sorted_entries:
                    if e.get('f1') is not None:
                        f1 = float(e.get('f1'))
                    else:
                        # assume best_fitness is error and compute f1 = 1 - error
                        f1 = 1.0 - float(e.get('best_fitness', 0.0))
                    f1_list.append(f1)
                experiments[key][str(seed)] = f1_list

    return experiments

def format_dataset_name(name):
    name = name.replace("_", " ")
    return name.capitalize()
    
def plot_fitness_curves_by_seed(experiments, output_dir, baselines=None, font_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm # Importante para carregar a fonte
    import numpy as np
    import os

    # --- CONFIGURAÇÃO DA FONTE TOMORROW ---
    if font_path and os.path.exists(font_path):
        # Carrega a fonte diretamente do arquivo
        fm.fontManager.addfont(font_path)
        # O nome da família geralmente é o nome do arquivo sem extensão ou metadados
        # Para garantir, pegamos o nome real da propriedade da fonte
        prop = fm.FontProperties(fname=font_path)
        custom_font_name = prop.get_name()
        
        plt.rcParams.update({
            'font.family': custom_font_name
        })
    else:
        # Fallback caso não passe o caminho, tenta usar se estiver instalada no sistema
        # Se não tiver, o Matplotlib usará a padrão (DejaVu Sans) e avisará
        plt.rcParams.update({'font.family': 'sans-serif'})
        if font_path:
            print(f"Aviso: Arquivo de fonte não encontrado em '{font_path}'. Usando padrão.")

    # --- CONFIGURAÇÃO ESTÉTICA (Mantendo estilo limpo) ---
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'axes.linewidth': 0.8,
        'grid.color': '#E6E6E6',
        'lines.linewidth': 1.8,
        'figure.autolayout': False,
    })

    # Listas ordenadas
    datasets = sorted({k[0] for k in experiments.keys()})
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
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        dpi=300,
        sharex='col',
        sharey='row'
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
        
        if vals:
            ymin, ymax = min(vals), max(vals)
            pad = 0.1 * (ymax - ymin if ymax > ymin else 1)
            row_limits[dataset] = (max(0.0, ymin - pad), min(1.0, ymax + pad))
        else:
            row_limits[dataset] = (0.0, 1.0)

    legend_handles = []
    legend_labels = []

    # Cores (Mantendo o tom profissional, mas a fonte Tomorrow pede algo mais "Tech/Clean")
    COLOR_GP = "#006064" # Cyan escuro/Teal (combina com a fonte Tomorrow)
    COLOR_BASELINES = ["#C2185B", "#7B1FA2", "#E65100", "#689F38"]

    # --------- Plotagem ---------
    for i, dataset in enumerate(datasets):
        for j, missing in enumerate(missing_rates):

            ax = axes[i, j]
            key = (dataset, missing)

            ax.set_axisbelow(True)
            ax.grid(True, linestyle='-', linewidth=0.5, color='#f0f0f0') # Grid muito leve

            # Estilo mais "Tech" remove top/right
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

                if i == 0 and j == 0:
                    legend_handles.append(line_gp)
                    legend_labels.append("Proposed GP")

            # Baselines
            if baselines is not None:
                key_bas = (dataset, int(missing)) if isinstance(missing, (int, float)) else (dataset, missing)
                if key_bas in baselines:
                    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1))]
                    for idx, (method, f1_score) in enumerate(sorted(baselines[key_bas].items())):
                        line_base = ax.axhline(
                            y=f1_score,
                            linestyle=linestyles[idx % len(linestyles)],
                            color=COLOR_BASELINES[idx % len(COLOR_BASELINES)],
                            linewidth=1.5,
                            alpha=0.8,
                            label=method
                        )
                        if i == 0 and j == 0:
                            legend_handles.append(line_base)
                            legend_labels.append(method)

            # Título
            ax.set_title(
                f"{datasets_fmt[dataset]} ({missing}%)",
                fontsize=11,
                pad=8,
                loc='left',
                weight='bold' # Tomorrow fica bem em negrito
            )

            # Eixos
            ymin, ymax = row_limits.get(dataset, (0.0, 1.0))
            ax.set_ylim(ymin, ymax)

            if j == 0:
                ax.set_ylabel("F1-Score", fontsize=10, labelpad=8)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelleft=False)

            # Lógica do Eixo X (Somente última linha)
            if i == n_rows - 1:
                ax.set_xlabel("Generations", fontsize=10, labelpad=8)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            if ax.get_legend():
                ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    out_path = os.path.join(output_dir, 'fitness_curves_tomorrow.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_path}")

    # Legenda
    if legend_handles:
        fig_leg = plt.figure(figsize=(8, 0.8))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis('off')
        fig_leg.legend(
            legend_handles, legend_labels, loc='center', ncol=len(legend_labels),
            frameon=False, fontsize=10
        )
        leg_path = os.path.join(output_dir, 'fitness_legend_tomorrow.png')
        fig_leg.savefig(leg_path, dpi=300, bbox_inches='tight')
        plt.close(fig_leg)
        print(f"Saved legend: {leg_path}")




def plot_fitness_curves_mean_std(experiments, output_dir, baselines=None):
    datasets = sorted({k[0] for k in experiments.keys()})
    missing_rates = sorted({k[1] for k in experiments.keys()}, key=lambda x: (0 if isinstance(x, int) else 1, x))

    n_rows = len(datasets)
    n_cols = len(missing_rates)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, dataset in enumerate(datasets):
        for j, missing in enumerate(missing_rates):
            ax = axes[i, j]
            key = (dataset, missing)
            if key not in experiments:
                ax.set_visible(False)
                continue

            seeds = experiments[key]
            curves = [np.array(v) for v in seeds.values() if len(v) > 0]
            if not curves:
                ax.set_visible(False)
                continue

            max_len = max(len(c) for c in curves)
            # pad curves to same length
            padded = np.vstack([np.pad(c, (0, max_len - len(c)), 'edge') for c in curves])

            mean = padded.mean(axis=0)
            std = padded.std(axis=0)
            gens = np.arange(len(mean))

            ax.plot(gens, mean, color='blue', linewidth=2, label='Média GP')

            # Add baseline lines if available
            if baselines is not None:
                key_bas = (dataset, int(missing)) if isinstance(missing, (int, float)) else (dataset, missing)
                if key_bas in baselines:
                    baseline_methods = baselines[key_bas]
                    linestyles = ['--', '-.', ':']
                    baseline_colors = get_color_cycle(6)
                    for idx, (method, f1_score) in enumerate(sorted(baseline_methods.items())):
                        ax.axhline(y=f1_score,
                                  linestyle=linestyles[idx % len(linestyles)],
                                  color=baseline_colors[idx % len(baseline_colors)],
                                  label=f'{method}',
                                  alpha=0.6,
                                  linewidth=2)

            ax.set_title(f"{dataset} — Missing: {missing}%")
            ax.set_xlabel('Geração')
            ax.set_ylabel('F1')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fitness_curves_mean_std.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_poster_large(experiments, output_dir, baselines=None):
    """High-resolution single large plot per dataset/missing suitable for posters."""
    # large figure per combination
    rc = {
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    }
    plt.rcParams.update(rc)

    for key, seeds in experiments.items():
        dataset, missing = key
        fig, ax = plt.subplots(figsize=(9,6), dpi=300)
        cmap = plt.cm.get_cmap('tab10')
        # plot only GP mean across seeds (do not plot individual seed curves)
        curves = [np.array(v) for v in seeds.values() if len(v) > 0]
        if curves:
            max_len = max(len(c) for c in curves)
            padded = np.vstack([np.pad(c, (0, max_len-len(c)), 'edge') for c in curves])
            mean = padded.mean(axis=0)
            gens = np.arange(len(mean))
            ax.plot(gens, mean, color='black', linewidth=2.5, label='GP mean')

        # baselines
        if baselines is not None:
            key_bas = (dataset, int(missing)) if isinstance(missing, (int,float)) else (dataset, missing)
            if key_bas in baselines:
                linestyles = ['--','-.',':']
                baseline_colors = get_color_cycle(6)
                for idx,(method,f1_score) in enumerate(sorted(baselines[key_bas].items())):
                    ax.axhline(y=f1_score, linestyle=linestyles[idx%len(linestyles)], color=baseline_colors[idx%len(baseline_colors)], linewidth=2.0, label=method)

        ax.set_xlabel('Geração')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{dataset} — Missing: {missing}%')
        ax.legend(loc='lower right', frameon=True)
        ax.grid(True, alpha=0.3)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'poster_{dataset}_{missing}_large.png')
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved poster-style plot: {out_path}")


def plot_poster_grid(experiments, output_dir, baselines=None):
    """Multi-panel grid optimized for poster panels: larger fonts, cleaner legend, consistent colors."""
    plt.rcParams.update({'font.size':14, 'figure.dpi':300})

    datasets = sorted({k[0] for k in experiments.keys()})
    missing_rates = sorted({k[1] for k in experiments.keys()})
    n_rows = len(datasets)
    n_cols = len(missing_rates)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols,4*n_rows), dpi=300)
    if n_rows==1 and n_cols==1:
        axes=np.array([[axes]])
    elif n_rows==1:
        axes=axes.reshape(1,-1)
    elif n_cols==1:
        axes=axes.reshape(-1,1)

    palette = get_color_cycle(10)
    for i,dataset in enumerate(datasets):
        for j,missing in enumerate(missing_rates):
            ax=axes[i,j]
            key=(dataset,missing)
            if key not in experiments:
                ax.set_visible(False); continue
            seeds = experiments[key]
            # plot only mean across seeds
            curves = [np.array(v) for v in seeds.values() if len(v) > 0]
            if curves:
                max_len = max(len(c) for c in curves)
                padded = np.vstack([np.pad(c, (0, max_len-len(c)), 'edge') for c in curves])
                mean = padded.mean(axis=0)
                ax.plot(np.arange(len(mean)), mean, color='black', linewidth=2.0, label='GP mean')

            # baselines
            if baselines is not None:
                key_bas = (dataset, int(missing)) if isinstance(missing,(int,float)) else (dataset, missing)
                if key_bas in baselines:
                    linestyles=['--','-.',':']
                    baseline_colors = get_color_cycle(6)
                    for idx,(method,f1_score) in enumerate(sorted(baselines[key_bas].items())):
                        ax.axhline(y=f1_score, linestyle=linestyles[idx%len(linestyles)], color=baseline_colors[idx%len(baseline_colors)], linewidth=1.8, alpha=0.9, label=method)

            ax.set_title(f'{dataset} — {missing}%')
            ax.set_xlabel('Geração')
            ax.set_ylabel('F1')
            ax.grid(alpha=0.2)

    plt.tight_layout()
    out_path=os.path.join(output_dir,'poster_grid.png')
    fig.savefig(out_path,dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(f'Saved poster grid: {out_path}')


def plot_summary_violin(experiments, output_dir, baselines=None):
    """Violin/box plot of final F1 across seeds per dataset/missing — good for posters summarizing variability."""
    plt.rcParams.update({'font.size':14, 'figure.dpi':300})
    rows = []
    for (dataset, missing), seeds in experiments.items():
        for seed, f1_list in seeds.items():
            if not f1_list: continue
            final = f1_list[-1]
            rows.append((dataset, missing, seed, final))

    if not rows:
        print('No data for violin plot'); return

    import pandas as pd
    df = pd.DataFrame(rows, columns=['dataset','missing','seed','final_f1'])

    datasets = sorted(df['dataset'].unique())
    for dataset in datasets:
        df_d = df[df['dataset']==dataset]
        plt.figure(figsize=(8,6), dpi=300)
        if sns is not None:
            sns.violinplot(x='missing', y='final_f1', data=df_d, inner='quartile', palette='muted')
            sns.swarmplot(x='missing', y='final_f1', data=df_d, color='k', alpha=0.6, size=3)
        else:
            # fallback: boxplot
            df_pivot = [group['final_f1'].values for _,group in df_d.groupby('missing')]
            plt.boxplot(df_pivot, labels=sorted(df_d['missing'].unique()))

        # baselines
        if baselines is not None:
            for missing in sorted(df_d['missing'].unique()):
                key_bas = (dataset, int(missing)) if isinstance(missing,(int,float)) else (dataset, missing)
                if key_bas in baselines:
                    # draw median of baselines as horizontal lines grouped by x value
                    baseline_colors = get_color_cycle(6)
                    for idx,(method,f1_score) in enumerate(sorted(baselines[key_bas].items())):
                        plt.axhline(y=f1_score, linestyle='--' if idx%2==0 else ':', color=baseline_colors[idx%len(baseline_colors)], linewidth=1.6, alpha=0.9)

        plt.xlabel('Missing ratio (%)')
        plt.ylabel('Final F1')
        plt.title(f'{dataset} — Final F1 distribution across seeds')
        out_path = os.path.join(output_dir, f'violin_{dataset}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f'Saved violin summary: {out_path}')


def plot_mean_by_missing_subplots(experiments, output_dir, baselines=None, max_cols=3):
    """Plot mean (and std) of final F1 across seeds for each missing ratio.

    - X axis: missing percentages
    - Y axis: mean final F1 (errorbars = std across seeds)
    - One subplot per dataset (arranged in grid)
    - Add baseline lines/curves per imputer if `baselines` provided (dict keyed by (dataset,missing) -> {imputer: f1})
    """
    import pandas as pd

    rows = []
    for (dataset, missing), seeds in experiments.items():
        for seed, f1_list in seeds.items():
            if not f1_list:
                continue
            rows.append((dataset, int(missing) if isinstance(missing, (int, float)) else missing, seed, float(f1_list[-1])))

    if not rows:
        print('No data available to plot mean by missing.');
        return

    df = pd.DataFrame(rows, columns=['dataset', 'missing', 'seed', 'final_f1'])

    datasets = sorted(df['dataset'].unique())
    n = len(datasets)
    ncols = min(max_cols, n) if n>0 else 1
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False, dpi=200)

    for idx, dataset in enumerate(datasets):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        df_d = df[df['dataset'] == dataset]
        missing_sorted = sorted(df_d['missing'].unique())
        means = []
        stds = []
        for m in missing_sorted:
            vals = df_d[df_d['missing'] == m]['final_f1'].values
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std(ddof=0) if len(vals) else 0.0)

        xs = missing_sorted
        ax.errorbar(xs, means, yerr=stds, marker='o', linestyle='-', color='tab:blue', capsize=4, linewidth=2, label='GP mean ± std')

        # plot baselines: for each imputer, connect baseline values across missing rates
        if baselines is not None:
            # find all imputers present for this dataset across missing rates
            imputers = set()
            for m in missing_sorted:
                keyb = (dataset, int(m) if isinstance(m, (int,float)) else m)
                if keyb in baselines:
                    imputers.update(baselines[keyb].keys())

            linestyles = ['--', '-.', ':']
            baseline_colors = get_color_cycle(6)
            for j, imputer in enumerate(sorted(imputers)):
                yvals = []
                for m in missing_sorted:
                    keyb = (dataset, int(m) if isinstance(m, (int,float)) else m)
                    if keyb in baselines and imputer in baselines[keyb]:
                        yvals.append(baselines[keyb][imputer])
                    else:
                        yvals.append(np.nan)
                # convert to np.array and plot, skipping NaNs
                yarr = np.array(yvals, dtype=float)
                if np.all(np.isnan(yarr)):
                    continue
                ax.plot(xs, yarr, linestyle=linestyles[j % len(linestyles)], marker='s', color=baseline_colors[j % len(baseline_colors)], linewidth=1.8, label=f'{imputer} (baseline)')

        ax.set_title(dataset)
        ax.set_xlabel('Missing ratio (%)')
        ax.set_ylabel('Final F1 (mean across seeds)')
        ax.set_xticks(xs)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

    # hide empty axes
    total_plots = nrows * ncols
    for extra in range(n, total_plots):
        r = extra // ncols
        c = extra % ncols
        axes[r][c].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'mean_by_missing_subplots.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Plot fitness evolution from parsed JSON')
    parser.add_argument('--input', '-i', default='resultado_final_copy_parsed.json', help='Path to parsed JSON (e.g. resultados_parsed.json)')
    parser.add_argument('--output-dir', '-o', default='results/gp_seed_experiment', help='Directory to save plots')
    parser.add_argument('--baselines-csv', '-b', default=None, help='Optional CSV of baselines (optuna results) to draw horizontal lines')
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
        csv_to_try.append('results/optuna_optimization/_teste.csv')
        # try to pick any CSV under results/optuna_optimization if present
        try:
            import glob
            more = glob.glob('results/optuna_optimization/*.csv')
            csv_to_try.extend([p for p in more if p not in csv_to_try])
        except Exception:
            pass

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

    # By default generate the main plots so the script works when invoked with:
    #   python3 analysis/plot_parsed_results.py
    plot_fitness_curves_by_seed(experiments, args.output_dir, baselines=baselines, font_path='analysis/Tomorrow-Regular.ttf')
    plot_fitness_curves_mean_std(experiments, args.output_dir, baselines=baselines)
    plot_mean_by_missing_subplots(experiments, args.output_dir, baselines=baselines)


if __name__ == '__main__':
    main()
