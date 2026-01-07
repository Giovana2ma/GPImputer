"""
Script para analisar resultados da otimização com Optuna.

Gera visualizações e relatórios dos resultados da otimização de hiperparâmetros.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class OptunaResultsAnalyzer:
    """Classe para análise de resultados do Optuna."""
    
    def __init__(self, results_path: str):
        """
        Inicializa o analisador.
        
        Args:
            results_path: Caminho para o arquivo CSV com todos os resultados
        """
        self.df = pd.read_csv(results_path)
        self.results_dir = Path(results_path).parent
        
        # Determine metric and direction
        self.metric_col = 'f1_score'
        self.metric_name = 'F1-Score'
        self.direction = 'maximize'
        
        if 'optimization_score' in self.df.columns:
            self.metric_col = 'optimization_score'
            if 'rmse' in self.df.columns:
                self.metric_name = 'RMSE'
                self.direction = 'minimize'
            else:
                self.metric_name = 'Score'
                
        print(f"Carregados {len(self.df)} trials de otimização")
        print(f"Métrica: {self.metric_name} ({self.direction})")
        print(f"Imputadores: {self.df['imputer'].unique()}")
        print(f"Datasets: {self.df['dataset'].unique()}")
        print(f"Missing ratios: {self.df['missing_ratio'].unique()}")
    
    def plot_best_scores_comparison(self):
        """Compara os melhores scores de cada imputador por dataset."""
        # Agrupar por imputador, dataset e missing_ratio
        agg_func = 'min' if self.direction == 'minimize' else 'max'
        best_scores = self.df.groupby(['imputer', 'dataset', 'missing_ratio'])[self.metric_col].agg(agg_func).reset_index()
        
        # Plot
        fig, axes = plt.subplots(1, len(best_scores['missing_ratio'].unique()), 
                                figsize=(15, 5))
        
        if len(best_scores['missing_ratio'].unique()) == 1:
            axes = [axes]
        
        for idx, missing_ratio in enumerate(sorted(best_scores['missing_ratio'].unique())):
            ax = axes[idx]
            data = best_scores[best_scores['missing_ratio'] == missing_ratio]
            
            # Pivot para heatmap
            pivot_data = data.pivot(index='imputer', columns='dataset', values=self.metric_col)
            
            # Inverter colormap se minimizar
            cmap = 'YlGnBu_r' if self.direction == 'minimize' else 'YlGnBu'
            
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, 
                       ax=ax, cbar_kws={'label': self.metric_name})
            ax.set_title(f'Missing Ratio: {missing_ratio*100:.0f}%')
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Imputador')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'best_scores_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Salvo: {self.results_dir / 'best_scores_heatmap.png'}")
        plt.close()
    
    def plot_convergence_by_imputer(self):
        """Plota a convergência da otimização por imputador."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, imputer in enumerate(sorted(self.df['imputer'].unique())):
            if idx >= 4:
                break
            
            ax = axes[idx]
            data = self.df[self.df['imputer'] == imputer]
            
            for dataset in data['dataset'].unique():
                for missing_ratio in data['missing_ratio'].unique():
                    subset = data[(data['dataset'] == dataset) & 
                                 (data['missing_ratio'] == missing_ratio)]
                    
                    if len(subset) > 0:
                        # Calcular best score até o momento
                        sorted_subset = subset.sort_values('trial_number')
                        if self.direction == 'minimize':
                            cumbest = sorted_subset[self.metric_col].cummin()
                        else:
                            cumbest = sorted_subset[self.metric_col].cummax()
                            
                        ax.plot(subset['trial_number'], cumbest, 
                               label=f'{dataset} ({missing_ratio*100:.0f}%)', 
                               alpha=0.7)
            
            ax.set_xlabel('Trial')
            ax.set_ylabel(f'Best {self.metric_name}')
            ax.set_title(f'{imputer.upper()}')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'convergence_plots.png', dpi=300, bbox_inches='tight')
        print(f"Salvo: {self.results_dir / 'convergence_plots.png'}")
        plt.close()
    
    def plot_parameter_distributions(self, imputer: str):
        """Plota distribuições dos parâmetros otimizados para um imputador."""
        data = self.df[self.df['imputer'] == imputer].copy()
        
        # Identificar colunas de parâmetros (excluir metadados)
        exclude_cols = {'trial_number', 'f1_score', 'optimization_score', 'rmse', 'imputer', 'dataset', 'missing_ratio'}
        param_cols = [col for col in data.columns if col not in exclude_cols]
        
        if not param_cols:
            print(f"Nenhum parâmetro encontrado para {imputer}")
            return
        
        n_params = len(param_cols)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]
        
        for idx, param in enumerate(param_cols):
            ax = axes[idx]
            
            # Verificar se é numérico ou categórico
            if pd.api.types.is_numeric_dtype(data[param]):
                # Scatter plot: parâmetro vs score
                cmap = 'viridis_r' if self.direction == 'minimize' else 'viridis'
                scatter = ax.scatter(data[param], data[self.metric_col], 
                                   c=data[self.metric_col], cmap=cmap, 
                                   alpha=0.6, s=50)
                ax.set_xlabel(param)
                ax.set_ylabel(self.metric_name)
                plt.colorbar(scatter, ax=ax, label=self.metric_name)
            else:
                # Box plot para categóricos
                data_clean = data[data[param].notna()]
                if len(data_clean) > 0:
                    data_clean.boxplot(column=self.metric_col, by=param, ax=ax)
                    ax.set_xlabel(param)
                    ax.set_ylabel(self.metric_name)
                    ax.set_title('')
            
            ax.set_title(f'{param}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Remover subplots extras
        for idx in range(n_params, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'Distribuição de Parâmetros - {imputer.upper()}', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.results_dir / f'param_distributions_{imputer}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Salvo: {self.results_dir / f'param_distributions_{imputer}.png'}")
        plt.close()
    
    def generate_summary_report(self):
        """Gera relatório resumido em texto."""
        report_path = self.results_dir / 'optimization_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO DE OTIMIZAÇÃO - OPTUNA\n")
            f.write("="*70 + "\n\n")
            
            # Melhores resultados por imputador
            f.write("MELHORES RESULTADOS POR IMPUTADOR\n")
            f.write("-"*70 + "\n")
            
            for imputer in sorted(self.df['imputer'].unique()):
                data = self.df[self.df['imputer'] == imputer]
                
                if self.direction == 'minimize':
                    best_idx = data[self.metric_col].idxmin()
                else:
                    best_idx = data[self.metric_col].idxmax()
                    
                best_row = data.loc[best_idx]
                
                f.write(f"\n{imputer.upper()}:\n")
                f.write(f"  Best {self.metric_name}: {best_row[self.metric_col]:.4f}\n")
                f.write(f"  Dataset: {best_row['dataset']}\n")
                f.write(f"  Missing Ratio: {best_row['missing_ratio']*100:.0f}%\n")
                f.write(f"  Parâmetros:\n")
                
                exclude_cols = {'trial_number', 'f1_score', 'optimization_score', 'rmse', 'imputer', 'dataset', 'missing_ratio'}
                param_cols = [col for col in data.columns if col not in exclude_cols]
                
                for param in param_cols:
                    if pd.notna(best_row[param]):
                        f.write(f"    {param}: {best_row[param]}\n")
            
            # Estatísticas por dataset
            f.write("\n" + "="*70 + "\n")
            f.write("ESTATÍSTICAS POR DATASET\n")
            f.write("-"*70 + "\n\n")
            
            agg_funcs = ['mean', 'std', 'min', 'max', 'count']
            stats = self.df.groupby(['dataset', 'missing_ratio', 'imputer'])[self.metric_col].agg(
                agg_funcs
            ).round(4)
            f.write(stats.to_string())
            
            # Ranking geral
            f.write("\n\n" + "="*70 + "\n")
            f.write(f"RANKING GERAL (por {self.metric_name} médio)\n")
            f.write("-"*70 + "\n\n")
            
            ranking = self.df.groupby('imputer')[self.metric_col].agg(['mean', 'min', 'max']).round(4)
            ascending = True if self.direction == 'minimize' else False
            ranking = ranking.sort_values('mean', ascending=ascending)
            f.write(ranking.to_string())
            
        print(f"Salvo: {report_path}")
    
    def analyze_all(self):
        """Executa todas as análises."""
        print("\nGerando análises...")
        
        # 1. Comparação de scores
        print("1. Comparação de melhores scores...")
        self.plot_best_scores_comparison()
        
        # 2. Convergência
        print("2. Plots de convergência...")
        self.plot_convergence_by_imputer()
        
        # 3. Distribuições de parâmetros
        print("3. Distribuições de parâmetros...")
        for imputer in self.df['imputer'].unique():
            self.plot_parameter_distributions(imputer)
        
        # 4. Relatório
        print("4. Gerando relatório...")
        self.generate_summary_report()
        
        print("\nAnálise completa!")


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Análise de resultados do Optuna')
    parser.add_argument(
        'results_path',
        type=str,
        help='Caminho para o CSV com todos os resultados'
    )
    
    args = parser.parse_args()
    
    # Executar análise
    analyzer = OptunaResultsAnalyzer(args.results_path)
    analyzer.analyze_all()


if __name__ == '__main__':
    main()
