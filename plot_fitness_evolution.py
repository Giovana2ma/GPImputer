"""
Script para plotar curvas de fitness por geração do GP.
Gera dois tipos de gráficos:
1. Fitness por geração com uma linha por seed
2. Média e desvio padrão por geração
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from collections import defaultdict

def load_all_fitnesses():
    """Carrega dados do arquivo all_fitnesses.pkl."""
    results_dir = Path("results/gp_categorical_experiment")
    
    # Procurar arquivo all_fitnesses mais recente
    fitness_files = sorted(results_dir.glob("gp_categorical_full_20251124_203956.pkl"))
    
    if not fitness_files:
        raise FileNotFoundError("Nenhum arquivo all_fitnesses_*.pkl encontrado")
    
    # Usar o mais recente
    filepath = fitness_files[-1]
    print(f"   Usando arquivo: {filepath.name}")
    
    with open(filepath, 'rb') as f:
        all_fitnesses = pickle.load(f)
    
    return all_fitnesses

def load_experiment_metadata():
    """Carrega metadados dos experimentos do CSV de resumo."""
    summary_file = Path("results/gp_seed_experiment/gp_seeds_summary_20251122_193743.csv")
    
    if not summary_file.exists():
        raise FileNotFoundError(f"Arquivo {summary_file} não encontrado")
    
    df = pd.read_csv(summary_file)
    return df

def load_baseline_results():
    """Carrega os melhores resultados dos baselines."""
    results_file = Path("results/optuna_optimization/all_optimization_results_20251119_185221.csv")
    
    if not results_file.exists():
        print("Arquivo de baselines não encontrado")
        return {}
    
    df = pd.read_csv(results_file)
    
    # Agrupar por dataset, missing_ratio e imputer para pegar o melhor F1
    baselines = {}
    
    for (dataset, missing_ratio, imputer), group in df.groupby(['dataset', 'missing_ratio', 'imputer']):
        best_f1 = group['f1_score'].max()
        
        key = (dataset, int(missing_ratio * 100))
        if key not in baselines:
            baselines[key] = {}
        
        baselines[key][imputer] = best_f1
    
    return baselines

def organize_experiments_by_dataset_and_missing(all_fitnesses):
    """
    Organiza experimentos por dataset e missing rate usando all_fitnesses.
    
    all_fitnesses é um dict com chaves no formato:
    'dataset_{nome}_missing_{taxa}_seed_{seed}' -> {gen: [fitness1, fitness2, ...]}
    """
    
    experiments = defaultdict(lambda: defaultdict(list))
    
    # Processar cada experimento
    for key, fitness_by_gen in all_fitnesses.items():
        # Extrair informações da chave
        parts = key.split('_')
        
        # Encontrar índices das palavras-chave
        dataset_idx = parts.index('dataset') + 1
        missing_idx = parts.index('missing') + 1
        seed_idx = parts.index('seed') + 1
        
        # Extrair dataset (pode ter múltiplas partes)
        dataset_parts = []
        for i in range(dataset_idx, missing_idx - 1):
            dataset_parts.append(parts[i])
        dataset = '_'.join(dataset_parts)
        
        # Extrair missing ratio (é um float, converter para int %)
        missing_ratio = float(parts[missing_idx])
        if missing_ratio < 1:  # Se for decimal (0.1)
            missing_ratio = int(missing_ratio * 100)
        else:  # Se já for percentual (10)
            missing_ratio = int(missing_ratio)
        
        # Extrair seed
        seed = int(parts[seed_idx])
        
        # Calcular fitness máximo por geração
        # Fitness armazenado é erro (1 - f1), converter para F1 score
        max_fitness_by_gen = []
        for gen in sorted(fitness_by_gen.keys()):
            fitnesses = fitness_by_gen[gen]
            if fitnesses:  # Se houver fitness nesta geração
                # Converter erro para F1: f1 = 1 - erro
                min_error = min(fitnesses)  # Melhor fitness (menor erro)
                max_f1 = 1.0 - min_error
                max_fitness_by_gen.append(max_f1)
        
        if max_fitness_by_gen:
            exp_key = (dataset, missing_ratio)
            experiments[exp_key][str(seed)] = max_fitness_by_gen
    
    return experiments

def plot_fitness_curves_by_seed(experiments, baselines, output_dir="results/gp_seed_experiment"):
    """
    Gráfico 1: Curva de melhor fitness por geração, uma linha por seed.
    Um subplot para cada combinação de dataset e taxa de missing.
    """
    
    # Organizar subplots
    datasets = sorted(set([k[0] for k in experiments.keys()]))
    missing_rates = sorted(set([k[1] for k in experiments.keys()]))
    
    n_datasets = len(datasets)
    n_missing = len(missing_rates)
    
    if n_datasets == 0 or n_missing == 0:
        print("Nenhum experimento encontrado para plotar")
        return
    
    fig, axes = plt.subplots(n_datasets, n_missing, figsize=(6*n_missing, 5*n_datasets))
    
    # Garantir que axes seja sempre 2D
    if n_datasets == 1 and n_missing == 1:
        axes = np.array([[axes]])
    elif n_datasets == 1:
        axes = axes.reshape(1, -1)
    elif n_missing == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, dataset in enumerate(datasets):
        for j, missing_rate in enumerate(missing_rates):
            ax = axes[i, j]
            
            key = (dataset, missing_rate)
            if key not in experiments:
                ax.set_visible(False)
                continue
            
            seeds_data = experiments[key]
            
            # Plotar uma linha por seed
            for seed_idx, (seed, max_fitness_list) in enumerate(sorted(seeds_data.items())):
                generations = list(range(len(max_fitness_list)))
                
                ax.plot(generations, max_fitness_list, 
                       label=f'Seed {seed}',
                       color=colors[seed_idx % 10],
                       alpha=0.7,
                       linewidth=1.5)
            
            # Adicionar linhas tracejadas para baselines
            if key in baselines:
                baseline_methods = baselines[key]
                linestyles = ['--', '-.', ':']
                baseline_colors = ['red', 'green', 'orange', 'purple', 'brown']
                
                for idx, (method, f1_score) in enumerate(sorted(baseline_methods.items())):
                    ax.axhline(y=f1_score,
                              linestyle=linestyles[idx % len(linestyles)],
                              color=baseline_colors[idx % len(baseline_colors)],
                              label=f'{method}',
                              alpha=0.6,
                              linewidth=2)
            
            ax.set_xlabel('Geração', fontsize=10)
            ax.set_ylabel('F1 Score', fontsize=10)
            ax.set_title(f'{dataset}\nMissing: {missing_rate}%', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/fitness_curves_by_seed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico 1 salvo em: {output_path}")
    plt.close()

def plot_fitness_curves_mean_std(experiments, baselines, output_dir="results/gp_seed_experiment"):
    """
    Gráfico 2: Média de fitness por geração com desvio padrão.
    Um subplot para cada combinação de dataset e taxa de missing.
    """
    
    # Organizar subplots
    datasets = sorted(set([k[0] for k in experiments.keys()]))
    missing_rates = sorted(set([k[1] for k in experiments.keys()]))
    
    n_datasets = len(datasets)
    n_missing = len(missing_rates)
    
    if n_datasets == 0 or n_missing == 0:
        print("Nenhum experimento encontrado para plotar")
        return
    
    fig, axes = plt.subplots(n_datasets, n_missing, figsize=(6*n_missing, 5*n_datasets))
    
    # Garantir que axes seja sempre 2D
    if n_datasets == 1 and n_missing == 1:
        axes = np.array([[axes]])
    elif n_datasets == 1:
        axes = axes.reshape(1, -1)
    elif n_missing == 1:
        axes = axes.reshape(-1, 1)
    
    for i, dataset in enumerate(datasets):
        for j, missing_rate in enumerate(missing_rates):
            ax = axes[i, j]
            
            key = (dataset, missing_rate)
            if key not in experiments:
                ax.set_visible(False)
                continue
            
            seeds_data = experiments[key]
            
            # Coletar todas as curvas de fitness
            all_curves = []
            max_gen = 0
            
            for seed, max_fitness_list in seeds_data.items():
                all_curves.append(max_fitness_list)
                max_gen = max(max_gen, len(max_fitness_list))
            
            # Garantir que todas as curvas tenham o mesmo tamanho
            for idx in range(len(all_curves)):
                if len(all_curves[idx]) < max_gen:
                    last_val = all_curves[idx][-1]
                    all_curves[idx] = all_curves[idx] + [last_val] * (max_gen - len(all_curves[idx]))
            
            all_curves = np.array(all_curves)
            
            # Calcular média e desvio padrão
            mean_fitness = np.mean(all_curves, axis=0)
            std_fitness = np.std(all_curves, axis=0)
            generations = np.arange(len(mean_fitness))
            
            # Plotar média
            ax.plot(generations, mean_fitness, 
                   label='Média GP',
                   color='blue',
                   linewidth=2.5)
            
            # Plotar desvio padrão como área sombreada
            ax.fill_between(generations,
                           mean_fitness - std_fitness,
                           mean_fitness + std_fitness,
                           alpha=0.3,
                           color='blue',
                           label='± Desvio Padrão')
            
            # Adicionar linhas tracejadas para baselines
            if key in baselines:
                baseline_methods = baselines[key]
                linestyles = ['--', '-.', ':']
                baseline_colors = ['red', 'green', 'orange', 'purple', 'brown']
                
                for idx, (method, f1_score) in enumerate(sorted(baseline_methods.items())):
                    ax.axhline(y=f1_score,
                              linestyle=linestyles[idx % len(linestyles)],
                              color=baseline_colors[idx % len(baseline_colors)],
                              label=f'{method}',
                              alpha=0.6,
                              linewidth=2)
            
            ax.set_xlabel('Geração', fontsize=10)
            ax.set_ylabel('F1 Score', fontsize=10)
            ax.set_title(f'{dataset}\nMissing: {missing_rate}%', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/fitness_curves_mean_std.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico 2 salvo em: {output_path}")
    plt.close()

def main():
    """Função principal."""
    print("="*60)
    print("Gerando gráficos de evolução do GP")
    print("="*60)
    
    print("\n1. Carregando all_fitnesses...")
    all_fitnesses = load_all_fitnesses()
    print(f"   Carregados {len(all_fitnesses)} experimentos")
    
    print("\n2. Carregando resultados dos baselines...")
    baselines = load_baseline_results()
    print(f"   Encontrados baselines para {len(baselines)} combinações dataset/missing")
    
    print("\n3. Organizando experimentos por dataset e missing rate...")
    experiments = organize_experiments_by_dataset_and_missing(all_fitnesses)
    print(f"   Organizados {len(experiments)} experimentos")
    
    for key, seeds in experiments.items():
        print(f"   - {key}: {len(seeds)} seeds")
    
    print("\n4. Gerando Gráfico 1: Fitness por seed...")
    plot_fitness_curves_by_seed(experiments, baselines)
    
    print("\n5. Gerando Gráfico 2: Média e desvio padrão...")
    plot_fitness_curves_mean_std(experiments, baselines)
    
    print("\n" + "="*60)
    print("Concluído!")
    print("="*60)

if __name__ == "__main__":
    main()
