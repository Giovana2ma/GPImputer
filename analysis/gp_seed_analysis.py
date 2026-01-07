"""
Análise de resultados do experimento GP com múltiplas seeds.

Carrega resultados salvos e gera visualizações e estatísticas.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

sns.set_style("whitegrid")


class GPSeedResultsAnalyzer:
    """Classe para análise de resultados do experimento GP."""
    
    def __init__(self, results_csv: str):
        """
        Inicializa o analisador.
        
        Args:
            results_csv: Caminho para o arquivo CSV com resumo dos resultados
        """
        self.results_path = Path(results_csv)
        self.results_dir = self.results_path.parent
        self.df = pd.read_csv(results_csv)
        
        print(f"Carregados resultados de {len(self.df)} execuções")
        print(f"Datasets: {sorted(self.df['dataset'].unique().tolist())}")
        print(f"Missing rates: {sorted(self.df['missing_ratio'].unique().tolist())}")
        print(f"Seeds: {sorted(self.df['seed'].unique().tolist())}")
        
        # Carregar logbooks se disponíveis
        self.logbooks = None
        logbooks_files = list(self.results_dir.glob("logbooks_*.pkl"))
        if logbooks_files:
            with open(logbooks_files[-1], 'rb') as f:
                self.logbooks = pickle.load(f)
        
        # Carregar fitness por geração
        self.fitness_per_gen = None
        fitness_files = list(self.results_dir.glob("fitness_per_generation_*.csv"))
        if fitness_files:
            self.fitness_per_gen = pd.read_csv(fitness_files[-1])
            print(f"Carregadas {len(self.fitness_per_gen)} fitness de gerações")
        
        # Carregar baselines dos imputadores (Optuna)
        self.imputer_baselines = self._load_imputer_baselines()
    
    def _load_imputer_baselines(self):
        """
        Carrega baselines dos imputadores do Optuna.
        
        Returns:
            Dict com {dataset: {missing_rate: {imputer: f1_score}}}
        """
        optuna_dir = Path('results/optuna_optimization')
        if not optuna_dir.exists():
            print("Diretório optuna_optimization não encontrado")
            return {}
        
        baselines = {}
        
        # Procurar arquivos de resultados do Optuna
        result_files = list(optuna_dir.glob('**/classifier_evaluation_*.csv'))
        
        if not result_files:
            print("Nenhum resultado de Optuna encontrado")
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
                    
                    # Pegar melhor F1 de cada imputador
                    imputer_scores = {}
                    for imputer in df_config['imputer'].unique():
                        best_f1 = df_config[df_config['imputer'] == imputer]['f1_weighted'].max()
                        imputer_scores[imputer] = best_f1
                    
                    baselines[dataset][missing_rate] = imputer_scores
            
            print(f"Baselines carregados: {len(baselines)} datasets")
            
        except Exception as e:
            print(f"Erro ao carregar baselines: {e}")
            return {}
        
        return baselines
    
    def print_statistics(self):
        """Imprime estatísticas descritivas."""
        print("\n" + "="*70)
        print("ANÁLISE ESTATÍSTICA DOS RESULTADOS")
        print("="*70)
        
        # Estatísticas descritivas
        print("\n1. ESTATÍSTICAS DE FITNESS:")
        print(f"   Média: {self.df['best_fitness'].mean():.6f}")
        print(f"   Desvio padrão: {self.df['best_fitness'].std():.6f}")
        print(f"   Mínimo: {self.df['best_fitness'].min():.6f}")
        print(f"   Máximo: {self.df['best_fitness'].max():.6f}")
        print(f"   Mediana: {self.df['best_fitness'].median():.6f}")
        print(f"   CV (Coef. Variação): {(self.df['best_fitness'].std()/self.df['best_fitness'].mean())*100:.2f}%")
        
        # Tempo de execução
        print("\n2. TEMPO DE EXECUÇÃO:")
        print(f"   Média: {self.df['execution_time'].mean():.2f}s")
        print(f"   Desvio padrão: {self.df['execution_time'].std():.2f}s")
        print(f"   Mínimo: {self.df['execution_time'].min():.2f}s")
        print(f"   Máximo: {self.df['execution_time'].max():.2f}s")
        
        # Programas
        print("\n3. DIVERSIDADE DE PROGRAMAS:")
        if 'total_program_length' in self.df.columns:
            print(f"   Tamanho médio: {self.df['total_program_length'].mean():.0f} caracteres")
            print(f"   Tamanho std: {self.df['total_program_length'].std():.0f} caracteres")
        elif 'program_length' in self.df.columns:
            print(f"   Tamanho médio: {self.df['program_length'].mean():.0f} caracteres")
            print(f"   Tamanho std: {self.df['program_length'].std():.0f} caracteres")
        else:
            print(f"   (Dados de tamanho de programa não disponíveis)")
        
        # Melhor e pior seed
        best_seed = self.df.loc[self.df['best_fitness'].idxmax(), 'seed']
        worst_seed = self.df.loc[self.df['best_fitness'].idxmin(), 'seed']
        
        print("\n4. MELHORES E PIORES SEEDS:")
        print(f"   🥇 Melhor seed: {best_seed}")
        print(f"      Fitness: {self.df.loc[self.df['seed']==best_seed, 'best_fitness'].values[0]:.6f}")
        
        print(f"\n   🥉 Pior seed: {worst_seed}")
        print(f"      Fitness: {self.df.loc[self.df['seed']==worst_seed, 'best_fitness'].values[0]:.6f}")
        
        # Análise de convergência
        if self.logbooks:
            print("\n5. ANÁLISE DE CONVERGÊNCIA:")
            improved_gens = []
            for seed, logbook in self.logbooks.items():
                best_fitness_per_gen = [rec['max'] for rec in logbook]
                improved = sum(1 for i in range(1, len(best_fitness_per_gen)) 
                             if best_fitness_per_gen[i] > best_fitness_per_gen[i-1])
                improved_gens.append(improved)
            
            avg_improved_gens = np.mean(improved_gens)
            print(f"   Gerações com melhoria (média): {avg_improved_gens:.1f}")
    
    def plot_fitness_distribution(self):
        """Plota distribuição de fitness."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.df['best_fitness'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(self.df['best_fitness'].mean(), color='red', linestyle='--', 
                  label=f'Média: {self.df["best_fitness"].mean():.4f}')
        ax.axvline(self.df['best_fitness'].median(), color='green', linestyle='--',
                  label=f'Mediana: {self.df["best_fitness"].median():.4f}')
        ax.set_xlabel('Best Fitness', fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        ax.set_title('Distribuição de Fitness Across Seeds', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f'fitness_distribution_{timestamp}.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_fitness_by_seed(self):
        """Plota fitness por seed."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df_sorted = self.df.sort_values('best_fitness', ascending=False)
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
        
        ax.bar(range(len(df_sorted)), df_sorted['best_fitness'], color=colors, alpha=0.8)
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['seed'], rotation=45)
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel('Best Fitness', fontsize=12)
        ax.set_title('Fitness por Seed (Ordenado)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        ax.axhline(self.df['best_fitness'].mean(), color='red', linestyle='--', 
                  alpha=0.7, label='Média')
        ax.legend()
        
        plt.tight_layout()
        
        filename = f'fitness_by_seed_{timestamp}.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_convergence_curves(self):
        """Plota curvas de convergência."""
        if not self.logbooks:
            print("Logbooks não disponíveis, pulando plot de convergência")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        avg_fitness_per_gen = []
        max_gens = 0
        
        for seed, logbook in self.logbooks.items():
            generations = [rec['gen'] for rec in logbook]
            fitness_progression = [rec['max'] for rec in logbook]
            max_gens = max(max_gens, len(generations))
            
            ax.plot(generations, fitness_progression, alpha=0.6, label=f"Seed {seed}")
        
        # Calcular média
        for gen in range(max_gens):
            fitness_at_gen = []
            for logbook in self.logbooks.values():
                if gen < len(logbook):
                    fitness_at_gen.append(logbook[gen]['max'])
            avg_fitness_per_gen.append(np.mean(fitness_at_gen))
        
        ax.plot(range(max_gens), avg_fitness_per_gen, 
               color='red', linewidth=3, linestyle='--', 
               label='Média', zorder=100)
        
        ax.set_xlabel('Geração', fontsize=12)
        ax.set_ylabel('Best Fitness', fontsize=12)
        ax.set_title('Convergência do GP Across Seeds', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f'convergence_curves_{timestamp}.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_fitness_boxplot(self):
        """Plota boxplot de fitness."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot([self.df['best_fitness']], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        # Adicionar pontos individuais
        ax.scatter([1]*len(self.df), self.df['best_fitness'], alpha=0.5, color='red', s=50)
        
        ax.set_xticklabels(['GP Fitness'])
        ax.set_ylabel('Best Fitness', fontsize=12)
        ax.set_title('Variabilidade de Fitness Across Seeds', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar estatísticas
        stats_text = (f"n={len(self.df)}\n"
                     f"μ={self.df['best_fitness'].mean():.4f}\n"
                     f"σ={self.df['best_fitness'].std():.4f}")
        ax.text(1.15, self.df['best_fitness'].median(), stats_text, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        filename = f'fitness_boxplot_{timestamp}.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def plot_convergence_per_dataset_and_rate(self):
        """
        Plota gráficos de convergência separados para cada dataset e taxa de missing.
        Mostra melhor indivíduo e média±std, com linhas de baseline dos imputadores.
        """
        if self.fitness_per_gen is None:
            print("Dados de fitness por geração não disponíveis")
            return []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_created = []
        
        # Criar diretório para os plots
        plots_dir = self.results_dir / 'convergence_plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Iterar sobre cada combinação dataset/missing_rate
        for dataset in self.fitness_per_gen['dataset'].unique():
            for missing_rate in self.fitness_per_gen[self.fitness_per_gen['dataset'] == dataset]['missing_ratio'].unique():
                
                # Filtrar dados para esta configuração
                df_config = self.fitness_per_gen[
                    (self.fitness_per_gen['dataset'] == dataset) &
                    (self.fitness_per_gen['missing_ratio'] == missing_rate)
                ]
                
                if len(df_config) == 0:
                    continue
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Calcular estatísticas por geração
                stats_per_gen = df_config.groupby('generation')['fitness'].agg(['min', 'mean', 'std'])
                generations = stats_per_gen.index
                
                # PLOT 1: Melhor indivíduo por geração
                ax1.plot(generations, stats_per_gen['min'], 
                        linewidth=2.5, color='steelblue', label='Melhor Indivíduo', zorder=10)
                ax1.set_xlabel('Geração', fontsize=12)
                ax1.set_ylabel('Fitness (error = 1 - F1)', fontsize=12)
                ax1.set_title(f'Melhor Indivíduo - {dataset} (Missing {missing_rate*100:.0f}%)', 
                             fontsize=14, fontweight='bold')
                ax1.grid(alpha=0.3)
                
                # PLOT 2: Média e desvio padrão
                ax2.plot(generations, stats_per_gen['mean'], 
                        linewidth=2.5, color='darkgreen', label='Média', zorder=10)
                ax2.fill_between(generations, 
                                stats_per_gen['mean'] - stats_per_gen['std'],
                                stats_per_gen['mean'] + stats_per_gen['std'],
                                alpha=0.3, color='darkgreen', label='±1 Desvio Padrão')
                ax2.set_xlabel('Geração', fontsize=12)
                ax2.set_ylabel('Fitness (error = 1 - F1)', fontsize=12)
                ax2.set_title(f'Média ± Std - {dataset} (Missing {missing_rate*100:.0f}%)', 
                             fontsize=14, fontweight='bold')
                ax2.grid(alpha=0.3)
                
                # Adicionar baselines dos imputadores
                if (self.imputer_baselines and 
                    dataset in self.imputer_baselines and 
                    missing_rate in self.imputer_baselines[dataset]):
                    
                    baselines = self.imputer_baselines[dataset][missing_rate]
                    colors = plt.cm.tab10(np.linspace(0, 1, len(baselines)))
                    
                    for idx, (imputer, f1_score) in enumerate(baselines.items()):
                        # Converter F1 para fitness (error = 1 - F1)
                        baseline_fitness = 1.0 - f1_score
                        
                        # Adicionar linha tracejada nos dois plots
                        for ax in [ax1, ax2]:
                            ax.axhline(baseline_fitness, linestyle='--', linewidth=2,
                                      color=colors[idx], alpha=0.8,
                                      label=f'{imputer} (F1={f1_score:.3f})', zorder=5)
                
                # Adicionar legendas
                ax1.legend(loc='best', fontsize=9)
                ax2.legend(loc='best', fontsize=9)
                
                plt.tight_layout()
                
                # Salvar figura
                filename = f'convergence_{dataset}_missing{int(missing_rate*100)}_{timestamp}.png'
                filepath = plots_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                plots_created.append(filename)
                print(f"  ✓ {filename}")
        
        return plots_created
    
    def plot_fitness_vs_time(self):
        """Plota fitness vs tempo de execução."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(self.df['execution_time'], self.df['best_fitness'],
                           s=100, alpha=0.6, c=self.df['seed'], cmap='viridis')
        
        # Adicionar labels das seeds
        for _, row in self.df.iterrows():
            ax.annotate(f"{int(row['seed'])}", 
                       (row['execution_time'], row['best_fitness']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Tempo de Execução (s)', fontsize=12)
        ax.set_ylabel('Best Fitness', fontsize=12)
        ax.set_title('Fitness vs Tempo de Execução', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Seed', ax=ax)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = f'fitness_vs_time_{timestamp}.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def generate_all_plots(self):
        """Gera todas as visualizações."""
        print("\nGerando visualizações...")
        
        plots = {}
        
        # Plots de convergência por dataset/missing rate (NOVO)
        print("\nGráficos de convergência por dataset e missing rate:")
        plots['convergence_detailed'] = self.plot_convergence_per_dataset_and_rate()
        
        plots['distribution'] = self.plot_fitness_distribution()
        print(f"  ✓ {plots['distribution']}")
        
        plots['by_seed'] = self.plot_fitness_by_seed()
        print(f"  ✓ {plots['by_seed']}")
        
        plots['convergence'] = self.plot_convergence_curves()
        if plots['convergence']:
            print(f"  ✓ {plots['convergence']}")
        
        plots['boxplot'] = self.plot_fitness_boxplot()
        print(f"  ✓ {plots['boxplot']}")
        
        plots['vs_time'] = self.plot_fitness_vs_time()
        print(f"  ✓ {plots['vs_time']}")
        
        return plots
    
    def generate_report(self):
        """Gera relatório detalhado."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"gp_seed_analysis_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE VARIABILIDADE DO GP\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.df['dataset'].iloc[0]}\n")
            f.write(f"Missing ratio: {self.df['missing_ratio'].iloc[0]*100}%\n")
            f.write(f"Número de seeds testadas: {len(self.df)}\n")
            f.write(f"Seeds: {sorted(self.df['seed'].tolist())}\n\n")
            
            # Estatísticas
            f.write("ESTATÍSTICAS DE FITNESS\n")
            f.write("-"*70 + "\n")
            f.write(self.df['best_fitness'].describe().to_string())
            f.write(f"\nCoeficiente de Variação: {(self.df['best_fitness'].std()/self.df['best_fitness'].mean())*100:.2f}%\n\n")
            
            # Ranking
            f.write("RANKING DE SEEDS POR FITNESS\n")
            f.write("-"*70 + "\n")
            ranking = self.df.sort_values('best_fitness', ascending=False)[['seed', 'best_fitness', 'execution_time']]
            f.write(ranking.to_string(index=False))
            f.write("\n\n")
            
            # Recomendações
            f.write("="*70 + "\n")
            f.write("RECOMENDAÇÕES\n")
            f.write("="*70 + "\n\n")
            
            best_seed = self.df.loc[self.df['best_fitness'].idxmax(), 'seed']
            f.write(f"🥇 Melhor seed encontrada: {best_seed}\n")
            f.write(f"   Fitness: {self.df.loc[self.df['seed']==best_seed, 'best_fitness'].values[0]:.6f}\n\n")
            
            cv = (self.df['best_fitness'].std()/self.df['best_fitness'].mean())*100
            if cv < 5:
                f.write("✅ Baixa variabilidade (CV < 5%): Resultados consistentes\n")
                f.write("   Recomendação: Qualquer seed razoável funcionará bem\n")
            elif cv < 15:
                f.write("⚠️  Variabilidade moderada (5% < CV < 15%)\n")
                f.write("   Recomendação: Testar algumas seeds e escolher a melhor\n")
            else:
                f.write("❌ Alta variabilidade (CV > 15%): Resultados muito dependentes da seed\n")
                f.write("   Recomendação: Executar múltiplas vezes e usar ensemble ou melhor seed\n")
        
        print(f"\nSalvo: {report_path}")
        return report_path
    
    def analyze_all(self):
        """Executa análise completa."""
        self.print_statistics()
        self.generate_all_plots()
        self.generate_report()


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Análise de resultados do experimento GP')
    parser.add_argument(
        'results_csv',
        type=str,
        help='Caminho para o CSV com resumo dos resultados'
    )
    
    args = parser.parse_args()
    
    # Executar análise
    analyzer = GPSeedResultsAnalyzer(args.results_csv)
    analyzer.analyze_all()
    
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA!")
    print("="*70)


if __name__ == '__main__':
    main()
