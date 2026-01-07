"""
Experimento para testar GP com handling explícito de features categóricas.

Pipeline: Categórico → One-Hot → Imputador → Categórico → GP → One-Hot → Logística
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import yaml
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from gp.gp_imputer_categorical import GPImputerCategorical
from gp.gp_imputer_hybrid import GPImputerHybrid
from gp.gp_imputer_mixed import GPImputerMixed
from imputers.simple import MeanImputer, MedianImputer, ModeImputer
from imputers.knn_imputer import KNNImputerWrapper
from imputers.mice_imputer import MICEImputerWrapper
from data.missingness import generate_missing_data
from data.dataset_loader import load_dataset_configs, load_and_preprocess_dataset


def identify_categorical_features(dataset_name: str, dataset_config: dict):
    """
    Identifica features categóricas e cria mapeamentos para one-hot encoding.
    
    Returns
    -------
    categorical_features : List[int]
        Índices das features categóricas no espaço categórico
    categorical_mappings : Dict[int, List[int]]
        Mapeamento de feature categórica → colunas one-hot
    n_numeric_features : int
        Número de features puramente numéricas
    """
    from ucimlrepo import fetch_ucirepo
    
    # Fetch dataset
    dataset = fetch_ucirepo(id=dataset_config['id'])
    
    # Separar numéricas e categóricas
    X_numeric = dataset.data.features.select_dtypes(include=[np.number])
    X_categorical = dataset.data.features.select_dtypes(exclude=[np.number])
    
    n_numeric_features = X_numeric.shape[1]
    n_categorical_features = X_categorical.shape[1]
    
    print(f"\n{dataset_name}:")
    print(f"  Numeric features: {n_numeric_features}")
    print(f"  Categorical features: {n_categorical_features}")
    
    if n_categorical_features == 0:
        return [], {}, n_numeric_features
    
    # Simular one-hot encoding para descobrir quantas colunas cada feature gera
    from sklearn.preprocessing import OneHotEncoder
    
    # Remover NaNs para o encoder
    y = dataset.data.targets.values.ravel()
    y_binary = dataset_config['binarize_lambda'](y)
    X_combined = pd.concat([X_numeric, X_categorical], axis=1)
    mask = ~(X_combined.isna().any(axis=1) | np.isnan(y_binary))
    X_clean = X_combined[mask]
    
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_clean[X_categorical.columns])
    
    n_onehot_cols = X_cat_encoded.shape[1]
    
    print(f"  One-hot columns: {n_onehot_cols}")
    print(f"  Categories per feature:")
    
    # Criar mapeamentos
    categorical_features = []
    categorical_mappings = {}
    
    onehot_col_idx = n_numeric_features  # One-hot columns start after numeric
    
    for cat_feature_idx, (cat_name, categories) in enumerate(
        zip(X_categorical.columns, encoder.categories_)
    ):
        n_categories = len(categories) - 1  # -1 porque drop='first'
        
        print(f"    {cat_name}: {len(categories)} categories → {n_categories} one-hot cols")
        
        # Índice no espaço categórico (após todas as numéricas)
        cat_idx_in_categorical_space = n_numeric_features + cat_feature_idx
        categorical_features.append(cat_idx_in_categorical_space)
        
        # Colunas one-hot correspondentes
        oh_cols = list(range(onehot_col_idx, onehot_col_idx + n_categories))
        categorical_mappings[cat_idx_in_categorical_space] = oh_cols
        
        onehot_col_idx += n_categories
    
    return categorical_features, categorical_mappings, n_numeric_features


class GPCategoricalExperiment:
    """Experimento para GP com handling categórico."""
    
    def __init__(
        self,
        config_path: str = "config/gp_config_classifier.yaml",
        datasets_config_path: str = "config/datasets_config.yaml",
        results_dir: str = "results/gp_categorical_experiment"
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
        
        # Carregar configurações
        self.dataset_configs = load_dataset_configs(datasets_config_path)
        
        self.all_runs = []
    
    def run_single_experiment(
        self,
        dataset_name: str,
        missing_ratio: float,
        seed: int
    ):
        """
        Executa um experimento único.
        
        Args:
            dataset_name: Nome do dataset
            missing_ratio: Taxa de missing values
            seed: Seed para reproducibilidade
        """
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name} | Missing: {missing_ratio*100}% | Seed: {seed}")
        print(f"{'='*70}")
        
        # Carregar dataset
        dataset_config = self.dataset_configs[dataset_name]
        X, y = load_and_preprocess_dataset(dataset_name, dataset_config, verbose=True)
        
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        # Identificar features categóricas ANTES de introduzir missing
        categorical_features, categorical_mappings, n_numeric = identify_categorical_features(
            dataset_name, dataset_config
        )
        
        # Introduzir missing values
        X_missing, _ = generate_missing_data(
            X.copy(),
            mechanism='MCAR',
            missing_rate=missing_ratio,
            seed=42
        )
        
        print(f"\nMissing values introduced: {np.sum(np.isnan(X_missing))}")
        
        # Criar imputadores base
        base_imputers = {
            'mean': MeanImputer(),
            'median': MedianImputer(),
            'mode': ModeImputer(),
            'knn': KNNImputerWrapper(n_neighbors=5),
            'mice': MICEImputerWrapper(max_iter=10, random_state=42)
        }
        
        # Adicionar imputadores adicionais se disponíveis
        try:
            from imputers.missforest_imputer import MissForestImputerWrapper
            base_imputers['missforest'] = MissForestImputerWrapper(random_state=42)
        except ImportError:
            print("  ⚠ MissForest não disponível")
        
        try:
            from imputers.svd_imputer import SVDImputerWrapper
            base_imputers['svd'] = SVDImputerWrapper(rank=5)  # Usar 'rank' em vez de 'n_components'
        except ImportError:
            print("  ⚠ SVD Imputer não disponível")
        
        # Treinar imputadores
        print("\nTreinando imputadores base...")
        for name, imputer in base_imputers.items():
            imputer.fit(X_missing)
            print(f"  ✓ {name}")
        
        # Carregar config do GP
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['seed'] = seed
        
        # Criar GP misto (trata categóricas e numéricas simultaneamente)
        print("\nCriando GP Misto...")
        gp = GPImputerMixed(config=config)
        
        # Configurar features categóricas
        if categorical_features:
            print(f"  Configurando {len(categorical_features)} features categóricas")
            gp.set_categorical_features(categorical_features, categorical_mappings)
        else:
            print("  ⚠ Nenhuma feature categórica - todas serão tratadas como numéricas")
        
        # Executar GP
        print("\nExecutando GP Misto...")
        import time
        start_time = time.time()
        
        gp.fit(X_missing, base_imputers, y_true=X, y_target=y)
        
        execution_time = time.time() - start_time
        
        # Extrair all_fitnesses do evolution_history
        all_fitnesses = []
        if hasattr(gp, 'evolution_history_') and gp.evolution_history_:
            for generation_data in gp.evolution_history_:
                if 'population' in generation_data:
                    # Extrair fitness de cada indivíduo
                    gen_fitnesses = [ind.fitness.values[0] for ind in generation_data['population']]
                    all_fitnesses.append(gen_fitnesses)
        
        # Resultados
        result = {
            'dataset': dataset_name,
            'missing_ratio': missing_ratio,
            'seed': seed,
            'best_fitness': gp.best_fitness_,
            'best_f1': 1.0 - gp.best_fitness_,
            'execution_time': execution_time,
            'n_generations': len(gp.logbook_) if gp.logbook_ else 0,
            'n_features': gp.n_features_,
            'n_categorical_features': len(categorical_features),
            'categorical_features': categorical_features,
            'categorical_mappings': categorical_mappings,
            'all_fitnesses': all_fitnesses
        }
        
        print(f"\n{'='*70}")
        print(f"RESULTADOS:")
        print(f"  Best Fitness: {result['best_fitness']:.6f}")
        print(f"  Best F1 Score: {result['best_f1']:.4f}")
        print(f"  Execution Time: {execution_time:.2f}s")
        print(f"  Features: {result['n_features']} ({result['n_categorical_features']} categóricas)")
        print(f"{'='*70}")
        
        self.all_runs.append(result)
        
        return result
    
    def run_multiple_experiments(
        self,
        seeds: List[int] = [42, 123, 456],
        missing_ratios: List[float] = [0.1, 0.2, 0.3]
    ):
        """
        Executa experimentos para todos os datasets, seeds e missing ratios.
        
        Args:
            seeds: Lista de seeds a testar
            missing_ratios: Lista de taxas de missing values
        """
        print("="*70)
        print("EXPERIMENTO GP MISTO")
        print("="*70)
        print(f"\nDatasets: {list(self.dataset_configs.keys())}")
        print(f"Missing ratios: {[f'{r*100}%' for r in missing_ratios]}")
        print(f"Seeds: {seeds}")
        
        total_experiments = len(self.dataset_configs) * len(missing_ratios) * len(seeds)
        print(f"\nTotal de experimentos: {total_experiments}")
        
        experiment_count = 0
        
        for dataset_name in self.dataset_configs.keys():
            for missing_ratio in missing_ratios:
                for seed in seeds:
                    experiment_count += 1
                    print(f"\n[{experiment_count}/{total_experiments}]")
                    
                    try:
                        self.run_single_experiment(dataset_name, missing_ratio, seed)
                    except Exception as e:
                        print(f"\n❌ ERRO no experimento:")
                        print(f"   Dataset: {dataset_name}")
                        print(f"   Missing: {missing_ratio}")
                        print(f"   Seed: {seed}")
                        print(f"   Erro: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        print("\n" + "="*70)
        print(f"CONCLUÍDO! {len(self.all_runs)}/{total_experiments} experimentos bem-sucedidos")
        print("="*70)
    
    def save_results(self):
        """Salva resultados em CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar DataFrame
        summary_data = []
        for run in self.all_runs:
            summary_data.append({
                'dataset': run['dataset'],
                'missing_ratio': run['missing_ratio'],
                'seed': run['seed'],
                'best_fitness': run['best_fitness'],
                'best_f1': run['best_f1'],
                'execution_time': run['execution_time'],
                'n_generations': run['n_generations'],
                'n_features': run['n_features'],
                'n_categorical_features': run['n_categorical_features']
            })
        
        df = pd.DataFrame(summary_data)
        
        # Salvar CSV
        csv_path = self.results_dir / f"gp_mixed_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Resultados salvos: {csv_path}")
        
        # Salvar detalhes completos em pickle
        pickle_path = self.results_dir / f"gp_mixed_full_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.all_runs, f)
        print(f"✓ Detalhes completos salvos: {pickle_path}")
        
        # Estatísticas resumidas
        print(f"\n{'='*70}")
        print("ESTATÍSTICAS RESUMIDAS:")
        print(f"{'='*70}")
        
        print(f"\nF1 Score por dataset:")
        for dataset in df['dataset'].unique():
            df_dataset = df[df['dataset'] == dataset]
            mean_f1 = df_dataset['best_f1'].mean()
            std_f1 = df_dataset['best_f1'].std()
            print(f"  {dataset}: {mean_f1:.4f} ± {std_f1:.4f}")
        
        print(f"\nF1 Score por missing ratio:")
        for ratio in sorted(df['missing_ratio'].unique()):
            df_ratio = df[df['missing_ratio'] == ratio]
            mean_f1 = df_ratio['best_f1'].mean()
            std_f1 = df_ratio['best_f1'].std()
            print(f"  {ratio*100:.0f}%: {mean_f1:.4f} ± {std_f1:.4f}")
        
        return csv_path


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Experimento GP Categórico')
    parser.add_argument(
        '--config',
        type=str,
        default='config/gp_config_classifier.yaml',
        help='Caminho para configuração do GP'
    )
    parser.add_argument(
        '--datasets-config',
        type=str,
        default='config/datasets_config.yaml',
        help='Caminho para configuração dos datasets'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 123, 456],
        help='Lista de seeds a testar'
    )
    parser.add_argument(
        '--missing-ratios',
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3],
        help='Lista de taxas de missing'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/gp_categorical_experiment',
        help='Diretório para salvar resultados'
    )
    
    args = parser.parse_args()
    
    # Criar e executar experimento
    experiment = GPCategoricalExperiment(
        config_path=args.config,
        datasets_config_path=args.datasets_config,
        results_dir=args.output_dir
    )
    
    experiment.run_multiple_experiments(
        seeds=args.seeds,
        missing_ratios=args.missing_ratios
    )
    
    # Salvar resultados
    experiment.save_results()
    
    print("\n✓ Experimento concluído!")


if __name__ == '__main__':
    main()
