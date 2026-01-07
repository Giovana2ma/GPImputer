"""
Script para executar GP com múltiplas seeds.

Executa o GP múltiplas vezes variando as seeds e salva os resultados
para análise posterior.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, List
import yaml
from joblib import Parallel, delayed

from sklearn.datasets import load_breast_cancer, load_iris, load_wine

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from gp.gp_imputer import GPImputer
from imputers.simple import MeanImputer, MedianImputer
from imputers.knn_imputer import KNNImputerWrapper
from imputers.mice_imputer import MICEImputerWrapper
from imputers.svd_imputer import  SVDImputerWrapper
from imputers.categorical_imputers import ModeImputerWrapper
from imputers.missforest_imputer import MissForestImputerWrapper
from data.missingness import generate_missing_data
from data.dataset_loader import load_dataset_configs, load_and_preprocess_dataset


class GPSeedExperiment:
    """Classe para experimentos de GP com diferentes seeds."""
    
    def __init__(
        self,
        config_path: str = "config/gp_config.yaml",
        datasets_config_path: str = "config/datasets_config.yaml",
        results_dir: str = "results/gp_seed_experiment"
    ):
        """
        Inicializa o experimento.
        
        Args:
            config_path: Caminho para configuração do GP
            datasets_config_path: Caminho para configuração dos datasets
            results_dir: Diretório para salvar resultados
        """
        self.config_path = config_path
        self.datasets_config_path = datasets_config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar configurações de datasets
        self.dataset_configs = load_dataset_configs(datasets_config_path)
        
        self.all_runs = []
    
    def _extract_all_fitnesses(self, evolution_history) -> Dict:
        """
        Extrai todas as fitness de todos os indivíduos em cada geração.
        
        Args:
            evolution_history: Histórico de evolução do GP (lista de gerações)
            
        Returns:
            Dict com fitness por geração: {gen: [fitness1, fitness2, ...]}
        """
        all_fitnesses = {}
        
        # evolution_history é uma lista onde cada elemento é uma geração
        if isinstance(evolution_history, list):
            for gen_idx, gen_data in enumerate(evolution_history):
                if isinstance(gen_data, dict) and 'individuals' in gen_data:
                    # Extrair fitness de cada indivíduo (já está no formato dict)
                    fitnesses = [ind['fitness'] for ind in gen_data['individuals'] 
                               if ind['fitness'] is not None]
                    all_fitnesses[gen_idx] = fitnesses
        
        return all_fitnesses
        
    def run_gp_with_seed(
        self,
        seed: int,
        X: np.ndarray,
        base_imputers: Dict,
        imputer_outputs: Dict,
        config: Dict,
        y_true: np.ndarray = None,
        y_target: np.ndarray = None
    ) -> Dict:
        """
        Executa GP com uma seed específica.
        
        Args:
            seed: Seed para reproducibilidade
            X: Dados com missing values
            base_imputers: Imputadores base
            imputer_outputs: Saídas pré-calculadas dos imputadores base
            config: Configuração do GP
            y_true: Valores verdadeiros (para fitness de imputação)
            y_target: Labels para classificação (para fitness f1_classifier)
            
        Returns:
            Dicionário com resultados da execução
        """
        print(f"\n{'='*70}")
        print(f"Executando GP com SEED={seed}")
        print(f"{'='*70}")
        
        # Atualizar seed na config
        config_copy = config.copy()
        config_copy['seed'] = seed
        
        # Criar e executar GP
        gp = GPImputer(config=config_copy)
        
        import time
        start_time = time.time()
        gp.fit(X, base_imputers, imputer_outputs=imputer_outputs, y_true=y_true, y_target=y_target)
        execution_time = time.time() - start_time
        
        # Coletar resultados
        result = {
            'seed': seed,
            'best_program': [str(tree) for tree in gp.best_individual_],  # Converter cada árvore individualmente
            'best_fitness': gp.best_fitness_,
            'execution_time': execution_time,
            'n_generations': len(gp.logbook_),
            'evolution_history': gp.evolution_history_,
            'logbook': gp.logbook_,
            'all_fitnesses': self._extract_all_fitnesses(gp.evolution_history_)  # NOVO: todas as fitness por geração
        }
        
        print(f"\nResultados:")
        print(f"  Melhor fitness: {result['best_fitness']:.6f}")
        print(f"  Tempo de execução: {execution_time:.2f}s")
        print(f"  Número de árvores: {len(result['best_program'])}")
        print(f"  Primeira árvore: {result['best_program'][0][:100] if result['best_program'] else 'N/A'}...")
        
        return result
    
    def run_multiple_datasets_and_ratios(
        self,
        seeds: List[int],
        missing_ratios: List[float] = [0.1, 0.2, 0.3]
    ):
        """
        Executa GP para todos os datasets e múltiplas taxas de missing.
        
        Args:
            seeds: Lista de seeds a testar
            missing_ratios: Lista de percentuais de missing values
        """
        print("="*70)
        print(f"EXPERIMENTO GP COMPLETO")
        print("="*70)
        print(f"\nDatasets: {list(self.dataset_configs.keys())}")
        print(f"Missing ratios: {[f'{r*100}%' for r in missing_ratios]}")
        print(f"Seeds: {seeds}")
        print(f"Total de execuções: {len(self.dataset_configs) * len(missing_ratios) * len(seeds)}")
        
        total_runs = 0
        
        # Configuração do GP
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Iterar sobre todos os datasets
        for dataset_name in self.dataset_configs.keys():
            print(f"\n{'='*70}")
            print(f"DATASET: {dataset_name}")
            print(f"{'='*70}")
            
            try:
                # Carregar dataset
                dataset_config = self.dataset_configs[dataset_name]
                X, y = load_and_preprocess_dataset(dataset_name, dataset_config, verbose=True)
                print(f"Dataset carregado: X={X.shape}, y={y.shape}")
                
                # Iterar sobre taxas de missing
                for missing_ratio in missing_ratios:
                    print(f"\n  Missing ratio: {missing_ratio*100}%")
                    
                    # Introduzir missing values
                    X_missing, _ = generate_missing_data(
                        X.copy(),
                        mechanism='MCAR',
                        missing_rate=missing_ratio,
                        seed=42
                    )
                    
                    # Criar imputadores base
                    base_imputers = {
                        'mean': MeanImputer(),
                        'median': MedianImputer(),
                        'knn_5': KNNImputerWrapper(n_neighbors=5),
                        'knn_10': KNNImputerWrapper(n_neighbors=10),
                        'mice': MICEImputerWrapper(max_iter=10, random_state=42),
                        'svd': SVDImputerWrapper(rank=10),
                        # 'mode': ModeImputerWrapper(),
                    }
                    
                    # Treinar imputadores em paralelo
                    print("Treinando imputadores base em paralelo...")
                    def run_imputer(name, imputer, X):
                        try:
                            return name, imputer.fit_transform(X)
                        except Exception as e:
                            print(f"Imputer {name} failed: {e}")
                            return name, None

                    results = Parallel(n_jobs=-1)(
                        delayed(run_imputer)(name, imputer, X_missing) 
                        for name, imputer in base_imputers.items()
                    )
                    
                    imputer_outputs = {name: output for name, output in results if output is not None}
                    
                    # Executar para cada seed
                    for seed in seeds:
                        print(f"\n    Seed: {seed}")
                        result = self.run_gp_with_seed(
                            seed=seed,
                            X=X_missing,
                            base_imputers=base_imputers,
                            imputer_outputs=imputer_outputs,
                            config=config,
                            y_true=X,
                            y_target=y
                        )
                        
                        result['dataset'] = dataset_name
                        result['missing_ratio'] = missing_ratio
                        self.all_runs.append(result)
                        total_runs += 1
                        
                        print(f"    Progresso: {total_runs}/{len(self.dataset_configs) * len(missing_ratios) * len(seeds)}")
                        
            except Exception as e:
                print(f"ERRO ao processar {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "="*70)
        print(f"CONCLUÍDO! {total_runs} execuções realizadas.")
        print("="*70)
    
    def run_multiple_seeds(
        self,
        seeds: List[int],
        dataset_name: str = 'statlog_heart',
        missing_ratio: float = 0.2
    ):
        """
        Executa GP com múltiplas seeds.
        
        Args:
            seeds: Lista de seeds a testar
            dataset_name: Nome do dataset (de datasets_config.yaml)
            missing_ratio: Percentual de missing values
        """
        print("="*70)
        print(f"EXPERIMENTO GP - {len(seeds)} SEEDS")
        print("="*70)
        print(f"\nDataset: {dataset_name}")
        print(f"Missing ratio: {missing_ratio*100}%")
        print(f"Seeds: {seeds}")
        
        # Carregar dataset usando dataset_loader
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset {dataset_name} não encontrado em datasets_config.yaml")
        
        config = self.dataset_configs[dataset_name]
        X, y = load_and_preprocess_dataset(dataset_name, config, verbose=True)
        
        print(f"\nDataset carregado: X={X.shape}, y={y.shape}")
        
        # Introduzir missing values
        X_missing, _ = generate_missing_data(
            X.copy(),
            mechanism='MCAR',
            missing_rate=missing_ratio,
            seed=42
        )
        
        # Criar imputadores base
        print("\nCriando imputadores base...")
        base_imputers = {
                        'mean': MeanImputer(),
                        'median': MedianImputer(),
                        'knn_5': KNNImputerWrapper(n_neighbors=5),
                        'knn_10': KNNImputerWrapper(n_neighbors=10),
                        'mice': MICEImputerWrapper(max_iter=10, random_state=42),
                        'svd': SVDImputerWrapper(rank=10),
                        # 'mode': ModeImputerWrapper(),
                    }
        
        # Treinar imputadores em paralelo
        print("Treinando imputadores base em paralelo...")
        def run_imputer(name, imputer, X):
            try:
                return name, imputer.fit_transform(X)
            except Exception as e:
                print(f"Imputer {name} failed: {e}")
                return name, None

        results = Parallel(n_jobs=-1)(
            delayed(run_imputer)(name, imputer, X_missing) 
            for name, imputer in base_imputers.items()
        )
        
        imputer_outputs = {name: output for name, output in results if output is not None}
        
        # Configuração do GP
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Executar para cada seed
        for seed in seeds:
            result = self.run_gp_with_seed(
                seed=seed,
                X=X_missing,
                base_imputers=base_imputers,
                imputer_outputs=imputer_outputs,
                config=config,
                y_true=X,  # Usar dados originais para fitness de imputação
                y_target=y  # Labels para classificação
            )
            
            result['dataset'] = dataset_name
            result['missing_ratio'] = missing_ratio
            self.all_runs.append(result)
        
        print("\n" + "="*70)
        print(f"Concluído! {len(seeds)} execuções realizadas.")
        print("="*70)
    
    def save_results(self):
        """Salva todos os resultados."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Salvar resumo em CSV
        summary_data = []
        for run in self.all_runs:
            summary_data.append({
                'seed': run['seed'],
                'dataset': run['dataset'],
                'missing_ratio': run['missing_ratio'],
                'best_fitness': run['best_fitness'],
                'execution_time': run['execution_time'],
                'n_generations': run['n_generations'],
                'n_trees': len(run['best_program']),
                'total_program_length': sum(len(tree) for tree in run['best_program'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        csv_path = self.results_dir / f"gp_seeds_summary_{timestamp}.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"\nSalvo resumo: {csv_path}")
        
        # 2. Salvar programas completos
        programs_path = self.results_dir / f"gp_programs_{timestamp}.json"
        programs_data = {
            str(run['seed']): {
                'program': run['best_program'],
                'fitness': float(run['best_fitness'])
            }
            for run in self.all_runs
        }
        
        with open(programs_path, 'w') as f:
            json.dump(programs_data, f, indent=2)
        print(f"Salvo programas: {programs_path}")
        
        # 3. Salvar histórico de evolução
        for run in self.all_runs:
            history_path = self.results_dir / f"evolution_seed{run['seed']}_{timestamp}.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(run['evolution_history'], f)
        print(f"Salvo históricos de evolução")
        
        # 4. Salvar logbooks
        logbooks_path = self.results_dir / f"logbooks_{timestamp}.pkl"
        logbooks = {run['seed']: run['logbook'] for run in self.all_runs}
        with open(logbooks_path, 'wb') as f:
            pickle.dump(logbooks, f)
        print(f"Salvo logbooks: {logbooks_path}")
        
        # 5. Salvar todas as fitness por geração
        all_fitnesses_path = self.results_dir / f"all_fitnesses_{timestamp}.pkl"
        all_fitnesses_data = {
            f"dataset_{run['dataset']}_missing_{run['missing_ratio']}_seed_{run['seed']}": run['all_fitnesses']
            for run in self.all_runs
        }
        with open(all_fitnesses_path, 'wb') as f:
            pickle.dump(all_fitnesses_data, f)
        print(f"Salvo todas as fitness: {all_fitnesses_path}")
        
        # 6. Salvar fitness por geração em CSV (formato tidy para análise)
        fitness_records = []
        for run in self.all_runs:
            for gen, fitnesses in run['all_fitnesses'].items():
                for idx, fitness in enumerate(fitnesses):
                    fitness_records.append({
                        'dataset': run['dataset'],
                        'missing_ratio': run['missing_ratio'],
                        'seed': run['seed'],
                        'generation': gen,
                        'individual_idx': idx,
                        'fitness': fitness
                    })
        
        if fitness_records:
            df_fitness = pd.DataFrame(fitness_records)
            fitness_csv_path = self.results_dir / f"fitness_per_generation_{timestamp}.csv"
            df_fitness.to_csv(fitness_csv_path, index=False)
            print(f"Salvo fitness por geração: {fitness_csv_path}")
        
        return csv_path


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Experimento GP com múltiplas seeds')
    parser.add_argument(
        '--config',
        type=str,
        default='config/gp_config_classifier.yaml',
        help='Caminho para configuração do GP'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[48, 101, 202, 404,505,606,707,808,909,1001],
        help='Lista de seeds a testar (default: 10 seeds)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset específico (se None, roda todos)'
    )
    parser.add_argument(
        '--missing-ratio',
        type=float,
        default=None,
        help='Taxa de missing específica (se None, usa 0.1, 0.2, 0.3)'
    )
    parser.add_argument(
        '--missing-ratios',
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3],
        help='Lista de taxas de missing (default: 10%%, 20%%, 30%%)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/gp_seed_experiment',
        help='Diretório para salvar resultados'
    )
    parser.add_argument(
        '--datasets-config',
        type=str,
        default='config/datasets_config.yaml',
        help='Caminho para arquivo de configuração dos datasets'
    )
    
    args = parser.parse_args()
    
    # Executar experimento
    experiment = GPSeedExperiment(
        config_path=args.config,
        datasets_config_path=args.datasets_config,
        results_dir=args.output_dir
    )
    
    # Decidir se roda um dataset específico ou todos
    if args.dataset is not None:
        # Modo legado: um dataset, uma taxa de missing
        print("\nExecutando GP com múltiplas seeds para dataset específico...")
        missing_ratio = args.missing_ratio if args.missing_ratio is not None else 0.2
        experiment.run_multiple_seeds(
            seeds=args.seeds,
            dataset_name=args.dataset,
            missing_ratio=missing_ratio
        )
    else:
        # Modo novo: todos os datasets, todas as taxas de missing
        print("\nExecutando GP para TODOS os datasets e múltiplas taxas de missing...")
        missing_ratios = args.missing_ratios
        experiment.run_multiple_datasets_and_ratios(
            seeds=args.seeds,
            missing_ratios=missing_ratios
        )
    
    print("\nSalvando resultados...")
    csv_path = experiment.save_results()
    
    print("\n" + "="*70)
    print("EXPERIMENTO CONCLUÍDO!")
    print("="*70)
    print(f"\nResultados disponíveis em: {experiment.results_dir}")
    print(f"\nPara analisar os resultados, execute:")
    print(f"python analysis/gp_seed_analysis.py {csv_path}")


if __name__ == '__main__':
    main()
