"""
Script para gerar gráficos customizados de convergência do GP.

Gera dois tipos de gráficos:
1. Curva de melhor fitness por geração, uma linha por seed, com linhas tracejadas dos baselines
2. Média por geração com desvio padrão, com linhas tracejadas dos baselines
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

plt.style.use('seaborn-v0_8-whitegrid')


class ConvergencePlotter:
    """Classe para criar gráficos de convergência personalizados."""
    
    def __init__(self, results_dir: str):
        """
        Inicializa o plotter.
        
        Args:
            results_dir: Diretório com os resultados do experimento
        """
        self.results_dir = Path(results_dir)
        
        # Carregar logbooks (contém fitness por geração)
        logbooks_files = list(self.results_dir.glob("logbooks_*.pkl"))
        if not logbooks_files:
            raise FileNotFoundError("Arquivo logbooks não encontrado!")
        
        with open(logbooks_files[-1], 'rb') as f:
            self.logbooks = pickle.load(f)
        
        print(f"Logbooks carregados: {len(self.logbooks)} seeds")
        
        # Carregar summary para obter informações de dataset e missing rate
        summary_files = list(self.results_dir.glob("gp_seeds_summary_*.csv"))
        if not summary_files:
            raise FileNotFoundError("Arquivo summary não encontrado!")
        
        self.summary_df = pd.read_csv(summary_files[-1])
        print(f"Summary carregado: {len(self.summary_df)} execuções")
        
        # Carregar baselines dos imputadores
        self.imputer_baselines = self._load_imputer_baselines()
    
    def _load_imputer_baselines(self):
        """
        Carrega baselines dos imputadores do Optuna.
        
        Returns:
            Dict com {dataset: {missing_rate: {imputer: f1_score}}}
        """
        optuna_dir = Path('results/optuna_optimization')
        if not optuna_dir.exists():
            print("⚠️  Diretório optuna_optimization não encontrado")
            return {}
        
        baselines = {}
        
        # Procurar arquivos de resultados do Optuna
        result_files = list(optuna_dir.glob('classifier_evaluation_*.csv'))
        
        if not result_files:
            print("⚠️  Nenhum resultado de Optuna encontrado")
            return {}
        
        # Usar o arquivo mais recente
        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        print(f"Carregando baselines de: {latest_file}")
        
        try:
            df_optuna = pd.read_csv(latest_file)
            
            # Organizar por dataset e missing rate
            for dataset in df_optuna['dataset'].unique():
                baselines[dataset] = {}
                df_dataset = df_optuna[df_optuna['dataset'] == dataset]
                
                for missing_rate in df_dataset['missing_rate'].unique():
                    df_config = df_dataset[df_dataset['missing_rate'] == missing_rate]
                    
                    # Pegar melhor F1 de cada imputador simples (mean, median, mode, knn, mice)
                    simple_imputers = ['mean', 'median', 'mode', 'knn', 'mice']
                    imputer_scores = {}
                    
                    for imputer in df_config['imputer'].unique():
                        if imputer in simple_imputers:
                            best_f1 = df_config[df_config['imputer'] == imputer]['f1_weighted'].max()
                            imputer_scores[imputer] = best_f1
                    
                    baselines[dataset][missing_rate] = imputer_scores
            
            print(f"✓ Baselines carregados para {len(baselines)} datasets")
            
        except Exception as e:
            print(f"❌ Erro ao carregar baselines: {e}")
            return {}
        
        return baselines
    
    def _extract_fitness_by_generation(self):
        """
        Extrai fitness por geração de todos os logbooks.
        
        Returns:
            DataFrame com colunas: seed, dataset, missing_ratio, generation, fitness
        """
        data = []
        
        for idx, row in self.summary_df.iterrows():
            seed = row['seed']
            dataset = row['dataset']
            missing_ratio = row['missing_ratio']
            
            # Encontrar o logbook correspondente
            # Os logbooks são indexados por uma chave que combina seed, dataset e missing_ratio
            logbook_key = None
            for key in self.logbooks.keys():
                if str(seed) in str(key) and dataset in str(key) and str(missing_ratio) in str(key):
                    logbook_key = key
                    break
            
            if logbook_key is None:
                # Tentar apenas com seed (se logbooks usam só seed como chave)
                logbook_key = str(seed)
            
            if logbook_key not in self.logbooks:
                print(f"⚠️  Logbook não encontrado para seed={seed}, dataset={dataset}, missing={missing_ratio}")
                continue
            
            logbook = self.logbooks[logbook_key]
            
            # Extrair fitness por geração (max = melhor fitness na geração)
            for gen_idx, record in enumerate(logbook):
                # Converter fitness para F1 (fitness = 1 - F1, então F1 = 1 - fitness)
                fitness_error = record['min']  # min porque queremos minimizar o erro
                f1_score = 1.0 - fitness_error
                
                data.append({
                    'seed': seed,
                    'dataset': dataset,
                    'missing_ratio': missing_ratio,
                    'generation': gen_idx,
                    'fitness': fitness_error,
                    'f1': f1_score
                })
        
        return pd.DataFrame(data)
    
    def plot_best_fitness_per_seed(self, save_dir=None):
        """
        Gráfico 1: Curva de melhor fitness por geração, uma linha por seed.
        Com linhas tracejadas para cada método simples de imputação (baseado em F1).
        Um subplot para cada combinação de dataset e missing rate.
        """
        print("\n" + "="*70)
        print("GRÁFICO 1: Melhor Fitness por Seed")
        print("="*70)
        
        # Extrair dados
        df = self._extract_fitness_by_generation()
        
        if df.empty:
            print("❌ Nenhum dado disponível!")
            return None
        
        # Obter combinações únicas de dataset e missing_ratio
        combinations = df[['dataset', 'missing_ratio']].drop_duplicates().sort_values(['dataset', 'missing_ratio'])
        n_plots = len(combinations)
        
        # Calcular grid de subplots
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, (_, row_comb) in enumerate(combinations.iterrows()):
            dataset = row_comb['dataset']
            missing_ratio = row_comb['missing_ratio']
            
            ax = axes[idx]
            
            # Filtrar dados para esta combinação
            df_config = df[(df['dataset'] == dataset) & (df['missing_ratio'] == missing_ratio)]
            
            # Plotar uma linha por seed
            seeds = df_config['seed'].unique()
            colors_seeds = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
            
            for seed_idx, seed in enumerate(seeds):
                df_seed = df_config[df_config['seed'] == seed].sort_values('generation')
                
                # Calcular melhor fitness acumulado (best fitness até aquela geração)
                best_f1_cumulative = df_seed['f1'].cummax()
                
                ax.plot(df_seed['generation'], best_f1_cumulative, 
                       linewidth=2, alpha=0.7, color=colors_seeds[seed_idx],
                       label=f'Seed {seed}')
            
            # Adicionar linhas tracejadas dos baselines
            if (self.imputer_baselines and 
                dataset in self.imputer_baselines and 
                missing_ratio in self.imputer_baselines[dataset]):
                
                baselines = self.imputer_baselines[dataset][missing_ratio]
                colors_imp = plt.cm.Set2(np.linspace(0, 1, len(baselines)))
                
                for imp_idx, (imputer, f1_score) in enumerate(baselines.items()):
                    ax.axhline(f1_score, linestyle='--', linewidth=2,
                              color=colors_imp[imp_idx], alpha=0.8,
                              label=f'{imputer.upper()}', zorder=5)
            
            # Configurações do subplot
            ax.set_xlabel('Geração', fontsize=11)
            ax.set_ylabel('F1 Score', fontsize=11)
            ax.set_title(f'{dataset}\nMissing: {missing_ratio*100:.0f}%', 
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=8, ncol=1)
        
        # Remover subplots não utilizados
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        # Salvar
        if save_dir is None:
            save_dir = self.results_dir
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f'convergence_best_per_seed_{timestamp}.png'
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {filepath}")
        plt.close()
        
        return filepath
    
    def plot_mean_with_std(self, save_dir=None):
        """
        Gráfico 2: Média de cada geração com desvio padrão (área sombreada).
        Com linhas tracejadas para os métodos de baseline.
        Um subplot para cada combinação de dataset e missing rate.
        """
        print("\n" + "="*70)
        print("GRÁFICO 2: Média com Desvio Padrão")
        print("="*70)
        
        # Extrair dados
        df = self._extract_fitness_by_generation()
        
        if df.empty:
            print("❌ Nenhum dado disponível!")
            return None
        
        # Obter combinações únicas de dataset e missing_ratio
        combinations = df[['dataset', 'missing_ratio']].drop_duplicates().sort_values(['dataset', 'missing_ratio'])
        n_plots = len(combinations)
        
        # Calcular grid de subplots
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, (_, row_comb) in enumerate(combinations.iterrows()):
            dataset = row_comb['dataset']
            missing_ratio = row_comb['missing_ratio']
            
            ax = axes[idx]
            
            # Filtrar dados para esta combinação
            df_config = df[(df['dataset'] == dataset) & (df['missing_ratio'] == missing_ratio)]
            
            # Para cada seed, calcular melhor F1 acumulado
            df_cumulative = []
            for seed in df_config['seed'].unique():
                df_seed = df_config[df_config['seed'] == seed].sort_values('generation')
                df_seed['f1_cummax'] = df_seed['f1'].cummax()
                df_cumulative.append(df_seed)
            
            df_cum = pd.concat(df_cumulative)
            
            # Calcular média e std por geração
            stats = df_cum.groupby('generation')['f1_cummax'].agg(['mean', 'std']).reset_index()
            
            # Plotar média
            ax.plot(stats['generation'], stats['mean'], 
                   linewidth=2.5, color='darkblue', label='Média GP', zorder=10)
            
            # Plotar área de desvio padrão
            ax.fill_between(stats['generation'], 
                           stats['mean'] - stats['std'],
                           stats['mean'] + stats['std'],
                           alpha=0.3, color='darkblue', label='±1 Desvio Padrão')
            
            # Adicionar linhas tracejadas dos baselines
            if (self.imputer_baselines and 
                dataset in self.imputer_baselines and 
                missing_ratio in self.imputer_baselines[dataset]):
                
                baselines = self.imputer_baselines[dataset][missing_ratio]
                colors_imp = plt.cm.Set2(np.linspace(0, 1, len(baselines)))
                
                for imp_idx, (imputer, f1_score) in enumerate(baselines.items()):
                    ax.axhline(f1_score, linestyle='--', linewidth=2,
                              color=colors_imp[imp_idx], alpha=0.8,
                              label=f'{imputer.upper()}', zorder=5)
            
            # Configurações do subplot
            ax.set_xlabel('Geração', fontsize=11)
            ax.set_ylabel('F1 Score', fontsize=11)
            ax.set_title(f'{dataset}\nMissing: {missing_ratio*100:.0f}%', 
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=8, ncol=1)
        
        # Remover subplots não utilizados
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        # Salvar
        if save_dir is None:
            save_dir = self.results_dir
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f'convergence_mean_std_{timestamp}.png'
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {filepath}")
        plt.close()
        
        return filepath
    
    def generate_all_plots(self, save_dir=None):
        """Gera todos os gráficos solicitados."""
        print("\n" + "="*70)
        print("GERANDO GRÁFICOS DE CONVERGÊNCIA")
        print("="*70)
        
        plot1 = self.plot_best_fitness_per_seed(save_dir)
        plot2 = self.plot_mean_with_std(save_dir)
        
        print("\n" + "="*70)
        print("CONCLUÍDO!")
        print("="*70)
        
        return {'best_per_seed': plot1, 'mean_std': plot2}


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gera gráficos customizados de convergência do GP'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/gp_seed_experiment',
        help='Diretório com os resultados do experimento'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Diretório para salvar os gráficos (padrão: mesmo do results-dir)'
    )
    
    args = parser.parse_args()
    
    # Executar
    plotter = ConvergencePlotter(args.results_dir)
    plotter.generate_all_plots(args.save_dir)


if __name__ == '__main__':
    main()
